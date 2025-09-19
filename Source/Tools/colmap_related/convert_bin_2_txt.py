#!/usr/bin/env python3
"""
COLMAP Binary to Text Format Converter
======================================

Converts COLMAP binary files to text format with comprehensive error handling
and validation. Supports all COLMAP camera models and file formats.

Features:
- Automatic format detection
- Support for all COLMAP camera models
- Comprehensive error handling
- Detailed conversion statistics
- Batch processing capability

This script converts COLMAP binary files (cameras.bin, images.bin, points3D.bin)
to their corresponding text versions (cameras.txt, images.txt, points3D.txt).

Usage:
    python Convert_bin_2_txt.py --input-dir INPUT_DIR [--output-dir OUTPUT_DIR]

Example:
    python Convert_bin_2_txt.py -i /path/to/colmap/sparse/0 -o /path/to/output
"""

import os
import sys
import struct
import argparse
import numpy as np
from pathlib import Path
from colmap_utils import (
    read_cameras_binary, read_images_binary, read_points3d_binary,
    write_cameras_txt, write_images_txt, write_points3d_txt, validate_colmap_directory
)

def convert_bin_to_txt(input_dir: str, output_dir: str = None, verbose: bool = False) -> bool:
    """
    Convert COLMAP binary files to text format.

    Args:
        input_dir: Path to folder containing binary files
        output_dir: Output directory (defaults to input_dir)
        verbose: Enable verbose output

    Returns:
        True if conversion successful, False otherwise
    """
    if output_dir is None:
        output_dir = input_dir

    # Validate input directory
    data_dir = validate_colmap_directory(input_dir)
    dataset_path = Path(data_dir)
    output_path = Path(output_dir)

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # File paths
    cameras_bin = dataset_path / "cameras.bin"
    images_bin = dataset_path / "images.bin"
    points3D_bin = dataset_path / "points3D.bin"

    cameras_txt = output_path / "cameras.txt"
    images_txt = output_path / "images.txt"
    points3D_txt = output_path / "points3D.txt"

    success = True
    converted_files = []

    # Convert cameras.bin
    if cameras_bin.exists():
        try:
            if verbose:
                print(f"Converting {cameras_bin} to {cameras_txt}...")
            cameras = read_cameras_binary(str(cameras_bin))
            write_cameras_txt(str(cameras_txt), cameras)
            converted_files.append(f"cameras: {len(cameras)} items")
        except Exception as e:
            print(f"Error converting cameras.bin: {e}")
            success = False
    elif verbose:
        print(f"Warning: {cameras_bin} not found")

    # Convert images.bin
    if images_bin.exists():
        try:
            if verbose:
                print(f"Converting {images_bin} to {images_txt}...")
            images = read_images_binary(str(images_bin))
            write_images_txt(str(images_txt), images)
            converted_files.append(f"images: {len(images)} items")
        except Exception as e:
            print(f"Error converting images.bin: {e}")
            success = False
    elif verbose:
        print(f"Warning: {images_bin} not found")

    # Convert points3D.bin
    if points3D_bin.exists():
        try:
            if verbose:
                print(f"Converting {points3D_bin} to {points3D_txt}...")
            points3D = read_points3d_binary(str(points3D_bin))
            write_points3d_txt(str(points3D_txt), points3D)
            converted_files.append(f"3D points: {len(points3D)} items")
        except Exception as e:
            print(f"Error converting points3D.bin: {e}")
            success = False
    elif verbose:
        print(f"Warning: {points3D_bin} not found")

    if success and converted_files:
        print(f"Conversion successful: {', '.join(converted_files)}")
    elif not converted_files:
        print("No binary files found to convert")
        success = False

    return success

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="COLMAP Binary to Text Format Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert binary files in place
  python Convert_bin_2_txt.py -i /path/to/sparse/0

  # Convert to different output directory
  python Convert_bin_2_txt.py -i /path/to/sparse/0 -o /path/to/output
        """
    )

    # Required arguments
    parser.add_argument('--input-dir', '-i', type=str, default=r"D:\Data\BaoAn\colmap\aligned_colmap_bin",
                       help='Input COLMAP model directory')
    parser.add_argument('--output-dir', '-o', type=str, default=r"D:\Data\BaoAn\colmap\aligned_colmap",
                       help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                       help='Enable verbose output')
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()

    print("=== COLMAP Binary to Text Format Converter ===")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir or args.input_dir}")

    try:
        if convert_bin_to_txt(args.input_dir, args.output_dir, args.verbose):
            print("\n=== Conversion completed successfully! ===")
            return 0
        else:
            print("\n=== Conversion completed with errors! ===")
            return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())