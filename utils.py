"""
DocuScan Utility Functions
Contains image loading, resizing, contour detection, corner ordering,
perspective transformation, and scan enhancement routines.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def load_image(image_path: str | Path) -> np.ndarray:
    """
    Loads an image from the specified file path and validates its existence and readability.
    
    Args:
        image_path: Path to the input image file.
        
    Returns:
        BGR image numpy array.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid or readable image.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
        
    image = cv2.imread(str(path))
    if image is None or image.size == 0:
        raise ValueError(f"Unable to decode image from path: {path}")
        
    return image


def resize_image(image: np.ndarray, max_dim: int = 800) -> tuple[np.ndarray, float]:
    """
    Resizes an image maintaining aspect ratio so its largest dimension equals max_dim.
    
    Args:
        image: Source image numpy array.
        max_dim: Target maximum height or width.
        
    Returns:
        Tuple of (resized_image, scale_factor) where scale_factor = orig_dim / resized_dim.
    """
    h, w = image.shape[:2]
    max_current = max(h, w)
    
    if max_current <= max_dim:
        return image.copy(), 1.0
        
    scale_factor = max_current / float(max_dim)
    new_w = int(round(w / scale_factor))
    new_h = int(round(h / scale_factor))
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale_factor


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 points in consistent top-left, top-right, bottom-right, bottom-left sequence.
    
    Args:
        pts: Array of shape (4, 2) containing corner coordinates.
        
    Returns:
        Array of shape (4, 2) with ordered coordinates [TL, TR, BR, BL].
    """
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")

    # Sum of coordinates: Top-Left has smallest sum, Bottom-Right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left
    rect[2] = pts[np.argmax(s)]  # Bottom-Right

    # Difference (x - y): Top-Right has maximum difference, Bottom-Left has minimum difference
    diff = pts[:, 0] - pts[:, 1]
    rect[1] = pts[np.argmax(diff)]  # Top-Right
    rect[3] = pts[np.argmin(diff)]  # Bottom-Left

    return rect


def is_valid_quadrilateral(pts: np.ndarray, image_shape: tuple[int, int], min_area_ratio: float = 0.10) -> bool:
    """
    Validates that 4 points form a non-degenerate, convex quadrilateral of sufficient area.
    
    Args:
        pts: Ordered (4, 2) corner points.
        image_shape: (height, width) of the image.
        min_area_ratio: Minimum fraction of total image area required.
        
    Returns:
        True if valid quadrilateral, False otherwise.
    """
    if pts.shape != (4, 2):
        return False

    # Check for duplicate points
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(pts[i] - pts[j]) < 5.0:
                return False

    # Check convexity
    contour = pts.reshape((-1, 1, 2)).astype(np.int32)
    if not cv2.isContourConvex(contour):
        return False

    # Check area
    area = cv2.contourArea(contour)
    total_area = image_shape[0] * image_shape[1]
    if area < (min_area_ratio * total_area):
        return False

    return True


def detect_document_contour(
    image_bgr: np.ndarray,
    min_area_ratio: float = 0.10,
    debug: bool = False
) -> tuple[np.ndarray | None, dict]:
    """
    Detects the main rectangular document contour in a BGR image.
    
    Args:
        image_bgr: BGR input image.
        min_area_ratio: Minimum area threshold as fraction of image size.
        debug: If True, populates dictionary with intermediate debug images.
        
    Returns:
        Tuple of (ordered_corners or None, debug_dict).
    """
    debug_dict = {}
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive Canny edge detection thresholds using median
    median_val = np.median(blur)
    lower_thresh = int(max(0, (1.0 - 0.33) * median_val))
    upper_thresh = int(min(255, (1.0 + 0.33) * median_val))
    edges = cv2.Canny(blur, max(30, lower_thresh), min(200, upper_thresh))

    # Morphological closing to bridge small edge gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    if debug:
        debug_dict['gray'] = gray
        debug_dict['blur'] = blur
        debug_dict['edges'] = edges
        debug_dict['closed'] = closed

    # Find contours
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, debug_dict

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            ordered_pts = order_points(pts)
            
            if is_valid_quadrilateral(ordered_pts, image_bgr.shape[:2], min_area_ratio):
                return ordered_pts, debug_dict

    return None, debug_dict


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Applies a 4-point perspective transformation to un-warp document image.
    
    Args:
        image: Source image numpy array.
        pts: (4, 2) corner array.
        
    Returns:
        Warped top-down perspective rectified image array.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate width of new image
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(round(width_a)), int(round(width_b)))

    # Calculate height of new image
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(round(height_a)), int(round(height_b)))

    # Guarantee non-zero output dimensions
    max_width = max(max_width, 10)
    max_height = max(max_height, 10)

    # Destination points for top-down view
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Compute perspective transform matrix and warp
    matrix = cv2.getPerspectiveTransform(rect.astype("float32"), dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped


def enhance_scans(warped_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates color, grayscale, and adaptive-threshold B&W scanned image variants.
    
    Args:
        warped_bgr: Perspective-corrected BGR image.
        
    Returns:
        Tuple of (scanned_color, scanned_gray, scanned_bw).
    """
    scanned_color = warped_bgr.copy()
    scanned_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    
    # Gentle blur before adaptive thresholding reduces noise artifacts
    blurred_gray = cv2.GaussianBlur(scanned_gray, (3, 3), 0)
    scanned_bw = cv2.adaptiveThreshold(
        blurred_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=8
    )
    
    return scanned_color, scanned_gray, scanned_bw


def draw_boundary_visualization(image_bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Draws the detected document contour and labeled corners on a copy of the original image.
    
    Args:
        image_bgr: Source image numpy array.
        corners: (4, 2) corner points.
        
    Returns:
        Annotated BGR image array.
    """
    vis = image_bgr.copy()
    pts = order_points(corners).astype(np.int32)
    
    # Scale line thickness and font size based on image dimensions
    h, w = vis.shape[:2]
    thickness = max(2, int(min(h, w) / 300))
    radius = max(5, int(min(h, w) / 100))
    font_scale = max(0.5, min(h, w) / 1000.0)

    # Draw bounding polygon
    cv2.polylines(vis, [pts.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=thickness)

    # Draw corner circles and labels
    labels = ["1: TL", "2: TR", "3: BR", "4: BL"]
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    
    for i, pt in enumerate(pts):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(vis, (x, y), radius, colors[i], -1)
        cv2.putText(
            vis,
            labels[i],
            (x + 10, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA
        )
        cv2.putText(
            vis,
            labels[i],
            (x + 10, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return vis
