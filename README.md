<<<<<<< HEAD
# doc-scan
A Python and OpenCV document scanner that detects page boundaries, corrects perspective distortion, and generates color, grayscale, and black-and-white scans. Runs entirely from the command line, with optional debug images and JSON processing summaries.
=======
# DocuScan: Automatic Document Detection and Perspective Correction Using OpenCV

DocuScan is a lightweight, fully automated Computer Vision application built with Python and OpenCV. It detects rectangular document pages in photographic images, corrects perspective distortions, and generates high-contrast scanned document images (color, grayscale, and adaptive black-and-white).

---

## 📌 Project Overview & Features

- **Automated Quadrilateral Detection**: Employs adaptive Canny edge detection, morphological closing, and polygon approximation to locate paper boundaries.
- **Robust Corner Ordering**: Algorithmically sorts detected corners into Top-Left, Top-Right, Bottom-Right, and Bottom-Left coordinates to eliminate twisting or orientation flips.
- **4-Point Perspective Transformation**: Calculates the homography matrix and warps tilted or angled document photos into flat, top-down rectangular views.
- **Multi-Format Output Scanning**: Exports color, grayscale, and clean binarized black-and-white scans suitable for archiving or OCR.
- **Deterministic & Lightweight**: Requires no GPU, neural networks, or external APIs. Fully reproducible using CPU OpenCV operations.
- **CLI & Structured Metadata**: Provides a terminal command-line interface and outputs execution logs and corner metrics to `summary.json`.

---

## 💡 Computer Vision Concepts Demonstrated

1. **Color Space Conversion (`cv2.cvtColor`)**: RGB/BGR to Grayscale reduction.
2. **Noise Reduction & Smoothing (`cv2.GaussianBlur`)**: Low-pass filtering to remove high-frequency camera sensor noise.
3. **Edge Detection (`cv2.Canny`)**: Intensity gradient computation to extract prominent structural boundaries.
4. **Morphological Operations (`cv2.morphologyEx`)**: Dilation and erosion closing kernels to bridge minor edge gaps.
5. **Contour Extraction & Polygon Approximation (`cv2.findContours`, `cv2.approxPolyDP`)**: Vector representation of boundaries and Ramer-Douglas-Peucker polygon simplification.
6. **Convexity & Area Geometry Filtering (`cv2.isContourConvex`, `cv2.contourArea`)**: Quad verification and background noise rejection.
7. **Homography & Perspective Transformation (`cv2.getPerspectiveTransform`, `cv2.warpPerspective`)**: Projective geometry transformation.
8. **Adaptive Thresholding (`cv2.adaptiveThreshold`)**: Local Gaussian window binarization for crisp text extraction under uneven lighting.

---

## 📁 Repository Structure

```text
cv/
├── main.py              # CLI entry point & main processing pipeline
├── utils.py             # Detection, perspective warp, scan enhancement, & visualizers
├── generate_sample.py   # Reproducible synthetic test image generator (Seed: 42)
├── requirements.txt     # Python package dependencies
├── README.md            # User manual and project documentation
├── report.md            # Comprehensive academic/project technical report
├── .gitignore           # Version control exclusion rules
├── samples/             # Synthetic test input images
│   └── sample.jpg
└── results/             # Output scan artifacts & metadata summary
    ├── detected_boundary.jpg
    ├── scanned_color.jpg
    ├── scanned_gray.jpg
    ├── scanned_bw.png
    └── summary.json
```

---

## ⚙️ Environment Setup & Installation

### Prerequisites
- Python 3.9+ (Python 3.10+ recommended)

### 1. Create Virtual Environment

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows (Command Prompt / PowerShell):**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage & Commands

### 1. Generate Synthetic Demonstration Image
Creates a synthetic page with text, grid targets, and perspective distortion on a dark background:
```bash
python generate_sample.py
```

### 2. Run Main Processing Pipeline
Process the synthetic sample image and save results to `results/`:
```bash
python main.py --input samples/sample.jpg --output results
```

### 3. Run Pipeline with Debug Output
Saves intermediate edge maps and grayscale debug images:
```bash
python main.py --input samples/sample.jpg --output results --debug
```

### 4. Process Custom Image
```bash
python main.py --input /path/to/your_photo.jpg --output results
```

### 5. Display Help & Options
```bash
python main.py --help
```

---

## 🖼️ Processing Pipeline Stages

```
Input Image ➔ Grayscale & Gaussian Blur ➔ Canny Edge & Morph Closing ➔ Contour Extraction ➔ 4-Corner Ordering ➔ Perspective Homography Warp ➔ Multi-Format Scans
```

1. **Input Validation & Resizing**: The image is validated and scaled to a working copy (`max_dim=800`) to ensure fast and consistent contour detection regardless of original resolution.
2. **Preprocessing**: Image is converted to grayscale and blurred with a 5x5 Gaussian kernel to reduce high-frequency noise.
3. **Edge & Morphological Processing**: Canny edge detection extracts gradients. Morphological closing (`MORPH_CLOSE`) bridges broken line segments.
4. **Contour Selection**: External contours are extracted, sorted by area, and approximated with `approxPolyDP`. The largest convex 4-point polygon above the minimum area threshold (10%) is selected.
5. **Coordinate Mapping & Corner Ordering**: Corner coordinates are scaled back to the original high-resolution image and ordered consistently (Top-Left, Top-Right, Bottom-Right, Bottom-Left).
6. **Perspective Correction**: Destination width and height are calculated using Euclidean distance between corners. `cv2.getPerspectiveTransform` computes the transformation matrix, and `cv2.warpPerspective` un-warps the document.
7. **Scan Generation**: Generates color, grayscale, and adaptive binarized black-and-white output files.

---

## 📊 Output Artifacts

- **`detected_boundary.jpg`**: Original photograph overlaid with green document boundary lines and labeled corners (1: TL, 2: TR, 3: BR, 4: BL).
- **`scanned_color.jpg`**: Perspective-rectified full-color output document.
- **`scanned_gray.jpg`**: Perspective-rectified grayscale scan.
- **`scanned_bw.png`**: High-contrast, adaptive Gaussian-thresholded black-and-white scan.
- **`summary.json`**: Execution metadata including input path, detection status, corner coordinates, output file paths, and execution time.

---

## ⚠️ Limitations & Troubleshooting

- **Flat Document Requirement**: Assumes the document page is mostly flat. Wrinkled, curved, or folded documents may exhibit minor residual warping.
- **Contrasting Background**: The document page must contrast with the underlying surface (e.g., white paper on a dark desk).
- **Four Corners Visible**: All four corners of the page must be within the image frame and unobscured.
- **Error Exit**: If no valid 4-corner document contour is found, DocuScan prints a descriptive error message to `sys.stderr`, writes an error `summary.json`, and exits with a non-zero status code (`1`).

---

## 📚 References & OpenCV Documentation

- [OpenCV `cv2.Canny`](https://docs.opencv.org/4.x/dd/1a1/group__imgproc__feature.html#ga044d70b090447229b76f2d44dfd85d00)
- [OpenCV `cv2.findContours`](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga17829f2441477558499ca04549231647)
- [OpenCV `cv2.approxPolyDP`](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga0012b5dab6dd43f443b7470c32b508f7)
- [OpenCV `cv2.getPerspectiveTransform`](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga8c80b544c86558e7d9cd759a3555330a)
- [OpenCV `cv2.warpPerspective`](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87)
- [OpenCV `cv2.adaptiveThreshold`](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3)
>>>>>>> d004581 (Initial commit: DocuScan document detection and perspective correction pipeline)
