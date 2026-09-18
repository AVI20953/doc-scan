"""
Synthetic Sample Image Generator for DocuScan
Generates a realistic synthetic document image placed on a contrasting background
with known perspective distortion and fixed random seed (42).
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def generate_synthetic_sample(output_path: str | Path = "samples/sample.jpg", seed: int = 42) -> Path:
    """
    Generates a synthetic document image with text, diagrams, and perspective distortion.
    
    Args:
        output_path: Path where the sample image will be saved.
        seed: Random seed for reproducibility.
        
    Returns:
        Path object of saved sample file.
    """
    np.random.seed(seed)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Create a dark wood-grained / textured background (1000 x 1200 pixels)
    bg_h, bg_w = 1200, 1000
    background = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)
    # Dark brown / slate background color gradient
    for y in range(bg_h):
        for x in range(bg_w):
            r = int(35 + 15 * (y / bg_h))
            g = int(45 + 10 * (x / bg_w))
            b = int(55 + 20 * (y / bg_h))
            background[y, x] = [b, g, r]
            
    # Add subtle background texture noise
    noise = np.random.randint(-10, 10, (bg_h, bg_w, 3), dtype=np.int16)
    background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 2. Create flat document sheet (600 x 800 pixels, white/off-white paper)
    doc_w, doc_h = 600, 800
    doc = np.ones((doc_h, doc_w, 3), dtype=np.uint8) * 250

    # Draw dark border inset on paper
    cv2.rectangle(doc, (20, 20), (doc_w - 20, doc_h - 20), (220, 220, 220), 1)

    # Header section
    cv2.rectangle(doc, (40, 40), (doc_w - 40, 110), (230, 240, 250), -1)
    cv2.rectangle(doc, (40, 40), (doc_w - 40, 110), (40, 80, 160), 2)
    cv2.putText(doc, "DOCUSCAN COMPUTER VISION DEMO", (60, 75), cv2.FONT_HERSHEY_DUPLEX, 0.7, (40, 80, 160), 2, cv2.LINE_AA)
    cv2.putText(doc, "Automatic Boundary & Perspective Correction", (60, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

    # Horizontal divider line
    cv2.line(doc, (40, 130), (doc_w - 40, 130), (100, 100, 100), 2)

    # Simulated text lines
    lines = [
        "1. INTRODUCTION & OVERVIEW",
        "DocuScan processes high-resolution photographic captures of document pages.",
        "It performs planar boundary localization, corner point ordering, homography estimation,",
        "and quadrilateral perspective rectification to yield flat readable scans.",
        "",
        "2. ALGORITHMIC STAGES",
        "  [A] Grayscale Conversion & Gaussian Pre-filtering",
        "  [B] Dynamic Canny Edge Detection & Morphological Closing",
        "  [C] Polygon Contour Approximation & Area Geometry Filtering",
        "  [D] Homography Mapping via 4-Point Perspective Transform",
        "  [E] Adaptive Binarization Enhancement",
        "",
        "3. TEST TARGET METRICS"
    ]

    y_offset = 160
    for line in lines:
        if line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
            cv2.putText(doc, line, (40, y_offset), cv2.FONT_HERSHEY_HELSINKI if hasattr(cv2, 'FONT_HERSHEY_HELSINKI') else cv2.FONT_HERSHEY_DUPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
        elif line.startswith("  ["):
            cv2.putText(doc, line, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 100, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(doc, line, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)
        y_offset += 25

    # Table grid test target
    cv2.rectangle(doc, (40, 520), (doc_w - 40, 680), (200, 200, 200), 1)
    cv2.rectangle(doc, (40, 520), (doc_w - 40, 550), (220, 220, 230), -1)
    cv2.putText(doc, "Stage", (50, 542), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(doc, "Status", (250, 542), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(doc, "Pass/Fail", (450, 542), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(doc, (40, 550), (doc_w - 40, 550), (0, 0, 0), 1)

    rows = [
        ("Canny Edge Detection", "Completed", "PASS"),
        ("Quad Detection", "4 Vertices", "PASS"),
        ("Warp Perspective", "Resampled", "PASS"),
        ("Adaptive Threshold", "Clean B&W", "PASS"),
    ]
    ry = 575
    for r_stage, r_stat, r_pf in rows:
        cv2.putText(doc, r_stage, (50, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(doc, r_stat, (250, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(doc, r_pf, (450, ry), cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 120, 0), 1, cv2.LINE_AA)
        cv2.line(doc, (40, ry + 8), (doc_w - 40, ry + 8), (230, 230, 230), 1)
        ry += 25

    # Seal / Stamp in bottom right
    cv2.circle(doc, (doc_w - 100, doc_h - 70), 40, (0, 0, 180), 2)
    cv2.putText(doc, "VERIFIED", (doc_w - 130, doc_h - 72), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 180), 1, cv2.LINE_AA)
    cv2.putText(doc, "SYNTHETIC", (doc_w - 132, doc_h - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 180), 1, cv2.LINE_AA)

    # 3. Apply Perspective Transformation to position paper on canvas
    src_pts = np.array([
        [0, 0],
        [doc_w - 1, 0],
        [doc_w - 1, doc_h - 1],
        [0, doc_h - 1]
    ], dtype=np.float32)

    # Target quadrilaterals on canvas with realistic camera angle skew
    dst_pts = np.array([
        [180, 160],   # Top-Left
        [830, 220],   # Top-Right
        [770, 1040],  # Bottom-Right
        [130, 960]    # Bottom-Left
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Warp paper image onto background using perspective matrix
    warped_doc = cv2.warpPerspective(doc, M, (bg_w, bg_h))
    
    # Create mask of warped paper
    paper_mask = cv2.warpPerspective(np.ones((doc_h, doc_w), dtype=np.uint8) * 255, M, (bg_w, bg_h))

    # Blend paper into background
    paper_mask_3d = cv2.cvtColor(paper_mask, cv2.COLOR_GRAY2BGR) / 255.0
    final_img = (warped_doc * paper_mask_3d + background * (1.0 - paper_mask_3d)).astype(np.uint8)

    # Save output
    cv2.imwrite(str(path), final_img)
    print(f"[SUCCESS] Generated synthetic test image: {path} (Seed: {seed}, Size: {bg_w}x{bg_h})")
    return path


if __name__ == "__main__":
    generate_synthetic_sample()
