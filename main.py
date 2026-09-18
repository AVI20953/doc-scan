"""
DocuScan: Automatic Document Detection and Perspective Correction
Command-line interface and processing pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import cv2

import utils


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DocuScan: Automatic document boundary detection and perspective correction using OpenCV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to the input document image file."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results",
        type=str,
        help="Directory path where scanned outputs and summary JSON will be saved."
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Save intermediate processing stages (grayscale, edges, morphological operations)."
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.10,
        help="Minimum required document area as a fraction of total image size (0.0 to 1.0)."
    )
    return parser.parse_args()


def write_summary_json(output_dir: Path, summary_data: dict) -> Path:
    """Writes processing metadata summary to JSON file."""
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    return summary_path


def main() -> None:
    start_time = time.time()
    args = parse_arguments()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read and validate input image
    try:
        orig_img = utils.load_image(input_path)
    except (FileNotFoundError, ValueError) as e:
        elapsed = time.time() - start_time
        summary = {
            "input_path": str(input_path),
            "status": "error",
            "detected_corners": None,
            "output_paths": {},
            "processing_time_seconds": round(elapsed, 4),
            "error": str(e)
        }
        write_summary_json(output_dir, summary)
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Resize working copy preserving scale factor
    working_img, scale_factor = utils.resize_image(orig_img, max_dim=800)

    # 3-6. Detect document quadrilateral contour
    corners_working, debug_dict = utils.detect_document_contour(
        working_img,
        min_area_ratio=args.min_area_ratio,
        debug=args.debug
    )

    # Save debug artifacts if requested
    if args.debug and debug_dict:
        if 'gray' in debug_dict:
            cv2.imwrite(str(output_dir / "debug_gray.jpg"), debug_dict['gray'])
        if 'edges' in debug_dict:
            cv2.imwrite(str(output_dir / "debug_edges.jpg"), debug_dict['edges'])
        if 'closed' in debug_dict:
            cv2.imwrite(str(output_dir / "debug_closed.jpg"), debug_dict['closed'])

    # Validate detection
    if corners_working is None:
        elapsed = time.time() - start_time
        err_msg = f"No rectangular document detected in image '{input_path}'. Ensure document is flat and contrasting against background."
        summary = {
            "input_path": str(input_path),
            "status": "error",
            "detected_corners": None,
            "output_paths": {},
            "processing_time_seconds": round(elapsed, 4),
            "error": err_msg
        }
        write_summary_json(output_dir, summary)
        print(f"[ERROR] {err_msg}", file=sys.stderr)
        sys.exit(1)

    # 7-8. Map corners back to original image scale and order points
    corners_orig = utils.order_points(corners_working * scale_factor)

    # 9. Apply 4-point perspective transformation at original resolution
    warped_color = utils.four_point_transform(orig_img, corners_orig)

    # 10. Generate color, grayscale, and adaptive threshold scan variants
    scanned_color, scanned_gray, scanned_bw = utils.enhance_scans(warped_color)
    boundary_vis = utils.draw_boundary_visualization(orig_img, corners_orig)

    # Save output artifacts
    out_boundary = output_dir / "detected_boundary.jpg"
    out_color = output_dir / "scanned_color.jpg"
    out_gray = output_dir / "scanned_gray.jpg"
    out_bw = output_dir / "scanned_bw.png"

    cv2.imwrite(str(out_boundary), boundary_vis)
    cv2.imwrite(str(out_color), scanned_color)
    cv2.imwrite(str(out_gray), scanned_gray)
    cv2.imwrite(str(out_bw), scanned_bw)

    elapsed = time.time() - start_time

    output_paths_dict = {
        "detected_boundary": str(out_boundary),
        "scanned_color": str(out_color),
        "scanned_gray": str(out_gray),
        "scanned_bw": str(out_bw)
    }

    if args.debug:
        output_paths_dict["debug_gray"] = str(output_dir / "debug_gray.jpg")
        output_paths_dict["debug_edges"] = str(output_dir / "debug_edges.jpg")
        output_paths_dict["debug_closed"] = str(output_dir / "debug_closed.jpg")

    summary = {
        "input_path": str(input_path),
        "status": "success",
        "detected_corners": [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in corners_orig],
        "output_paths": output_paths_dict,
        "processing_time_seconds": round(elapsed, 4),
        "error": None
    }

    summary_path = write_summary_json(output_dir, summary)

    print(f"[SUCCESS] Document scanned successfully!")
    print(f"  Input: {input_path}")
    print(f"  Corners: {summary['detected_corners']}")
    print(f"  Execution Time: {elapsed:.4f}s")
    print(f"  Output Directory: {output_dir}")
    print(f"  Summary Report: {summary_path}")


if __name__ == "__main__":
    main()
