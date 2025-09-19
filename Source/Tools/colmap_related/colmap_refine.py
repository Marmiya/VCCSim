#!/usr/bin/env python3
"""
COLMAP Dataset Refinement Tool
===============================

Advanced COLMAP dataset processing tool with multiple refinement capabilities:
- Convert multiple camera models to unified PINHOLE model
- Resize images to target resolution
- Limit POINTS2D per image to reduce file size
- Filter cameras by distance from reference point
- Apply outlier filtering and height-based filtering
- Support for both binary and text COLMAP formats

Usage:
    python colmap_refine.py --input-dir INPUT_DIR --output-dir OUTPUT_DIR [options]

Example:
    python colmap_refine.py -i ./sparse/0 -o ./sparse/refined --max-points2d 100 --filter-cameras
"""

import os
import shutil
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from colmap_utils import (
    ColmapCamera, ColmapImage,
    read_cameras_txt, read_images_txt, write_cameras_txt, 
    write_images_txt, calculate_camera_center, validate_colmap_directory
)


def convert_camera_to_pinhole(camera: ColmapCamera, target_width: int = None, target_height: int = None) -> ColmapCamera:
    """Convert any camera model to PINHOLE model"""
    width = target_width if target_width else camera.width
    height = target_height if target_height else camera.height
    
    # Extract or estimate focal length
    if camera.model == "PINHOLE":
        fx, fy = camera.params[0], camera.params[1]
    elif camera.model == "SIMPLE_PINHOLE":
        fx = fy = camera.params[0]
    elif camera.model == "SIMPLE_RADIAL":
        fx = fy = camera.params[0]
    elif camera.model == "RADIAL":
        fx = fy = camera.params[0]
    elif camera.model == "OPENCV":
        fx, fy = camera.params[0], camera.params[1]
    elif camera.model == "OPENCV_FISHEYE":
        fx, fy = camera.params[0], camera.params[1]
    elif camera.model == "FULL_OPENCV":
        fx, fy = camera.params[0], camera.params[1]
    else:
        # Estimate focal length from image dimensions (typical field of view ~60 degrees)
        fx = fy = max(width, height) * 1.2
        print(f"Warning: Unknown camera model '{camera.model}', estimating focal length: {fx}")
    
    # Scale focal length if resolution changed
    if target_width and target_height:
        fx = fx * (target_width / camera.width)
        fy = fy * (target_height / camera.height)
    
    # Principal point (image center)
    cx = width / 2.0
    cy = height / 2.0
    
    # PINHOLE model parameters: [fx, fy, cx, cy]
    return ColmapCamera(
        camera_id=camera.camera_id,
        model="PINHOLE",
        width=width,
        height=height,
        params=[fx, fy, cx, cy]
    )


def calculate_average_focal_length(cameras: Dict[int, ColmapCamera]) -> Tuple[float, float]:
    """Calculate average focal length from all cameras"""
    fx_values = []
    fy_values = []
    
    for camera in cameras.values():
        if camera.model == "PINHOLE":
            fx_values.append(camera.params[0])
            fy_values.append(camera.params[1])
        elif camera.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
            fx_values.append(camera.params[0])
            fy_values.append(camera.params[0])
        elif camera.model in ["OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"]:
            fx_values.append(camera.params[0])
            fy_values.append(camera.params[1])
        else:
            # Estimate from image dimensions
            fx = fy = max(camera.width, camera.height) * 1.2
            fx_values.append(fx)
            fy_values.append(fy)
    
    avg_fx = np.mean(fx_values) if fx_values else 1000.0
    avg_fy = np.mean(fy_values) if fy_values else 1000.0
    
    return avg_fx, avg_fy


def convert_dataset(input_dir: str, output_dir: str, target_resolution: Tuple[int, int] = None, max_points2d: int = None, filter_far_cameras: bool = False, camera_filter_ratio: float = 0.2, reference_point: List[float] = None):
    """Convert COLMAP dataset to single pinhole camera model"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Use common utility to validate and detect format
    data_dir = validate_colmap_directory(input_dir)
    data_path = Path(data_dir)

    cameras_txt = data_path / "cameras.txt"
    images_txt = data_path / "images.txt"
    points3d_txt = data_path / "points3D.txt"

    if not cameras_txt.exists() or not images_txt.exists():
        raise FileNotFoundError("Required COLMAP files (cameras.txt, images.txt) not found")

    # Try to find images directory - common locations relative to COLMAP model
    images_dir = None
    potential_dirs = [
        input_path / "images",  # Same level as sparse model
        input_path.parent / "images",  # Parent directory
        input_path.parent.parent / "images",  # Two levels up (for sparse/0 structure)
    ]

    for potential_dir in potential_dirs:
        if potential_dir.exists() and potential_dir.is_dir():
            images_dir = str(potential_dir)
            break

    if images_dir:
        print(f"Images directory: {images_dir}")
    else:
        print("Warning: Images directory not found - filename correction will be skipped")

    print(f"Reading COLMAP dataset from: {input_dir}")

    # Read original data
    cameras = read_cameras_txt(str(cameras_txt))
    images = read_images_txt(str(images_txt))

    print(f"Found {len(cameras)} cameras and {len(images)} images")
    print(f"Camera models: {set(cam.model for cam in cameras.values())}")

    # Find actual image files and create name mapping
    image_names = [img.name for img in images.values()]
    name_mapping = find_actual_image_files(images_dir, image_names) if images_dir else {}
    
    if name_mapping:
        changed_count = sum(1 for orig, actual in name_mapping.items() if orig != actual)
        print(f"Found {len(name_mapping)} images, {changed_count} names will be corrected")
    
    # Optional: Filter cameras by distance from reference point
    if filter_far_cameras and camera_filter_ratio > 0:
        if reference_point is None:
            reference_point = DEFAULT_REFERENCE_POINT
        print(f"Filtering cameras: removing {camera_filter_ratio*100:.1f}% farthest cameras from reference point {reference_point}")
        images, n_removed_cameras = filter_far_cameras(images, camera_filter_ratio, reference_point)
        print(f"Camera filtering: removed {n_removed_cameras} cameras, kept {len(images)} cameras")
    else:
        print("Camera filtering disabled")
    
    # Calculate unified resolution and focal length
    if target_resolution:
        target_width, target_height = target_resolution
    else:
        # Use the most common resolution
        resolutions = [(cam.width, cam.height) for cam in cameras.values()]
        target_width, target_height = max(set(resolutions), key=resolutions.count)
    
    print(f"Target resolution: {target_width}x{target_height}")
    
    # Calculate average focal length for unified camera
    avg_fx, avg_fy = calculate_average_focal_length(cameras)
    
    # Scale focal length to target resolution
    if target_resolution:
        # Use average original resolution for scaling
        avg_width = np.mean([cam.width for cam in cameras.values()])
        avg_height = np.mean([cam.height for cam in cameras.values()])
        scale_x = target_width / avg_width
        scale_y = target_height / avg_height
        avg_fx *= scale_x
        avg_fy *= scale_y
    
    print(f"Unified focal length: fx={avg_fx:.2f}, fy={avg_fy:.2f}")
    
    # Create unified pinhole camera
    unified_camera = ColmapCamera(
        camera_id=1,
        model="PINHOLE",
        width=target_width,
        height=target_height,
        params=[avg_fx, avg_fy, target_width/2.0, target_height/2.0]
    )
    
    # Update all images to use the unified camera
    updated_images = {}
    for img in images.values():
        # Scale 2D points if resolution changed
        original_camera = cameras[img.camera_id]
        scale_x = target_width / original_camera.width
        scale_y = target_height / original_camera.height
        
        scaled_points2d = []
        for x, y, point3d_id in img.points2d:
            scaled_x = x * scale_x
            scaled_y = y * scale_y
            scaled_points2d.append((scaled_x, scaled_y, point3d_id))
        
        # Use corrected image name if available
        corrected_name = name_mapping.get(img.name, img.name)
        
        updated_images[img.image_id] = ColmapImage(
            image_id=img.image_id,
            qvec=img.qvec,
            tvec=img.tvec,
            camera_id=1,  # Use unified camera ID
            name=corrected_name,  # Use corrected filename
            points2d=scaled_points2d
        )
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write converted files
    print(f"Writing converted dataset to: {output_dir}")
    
    # Show POINTS2D statistics if limiting
    if max_points2d is not None:
        total_original_points = sum(len(img.points2d) for img in updated_images.values())
        total_limited_points = sum(min(len(img.points2d), max_points2d) for img in updated_images.values())
        reduction_percentage = ((total_original_points - total_limited_points) / total_original_points * 100) if total_original_points > 0 else 0
        print(f"POINTS2D reduction: {total_original_points} -> {total_limited_points} ({reduction_percentage:.1f}% reduction)")
    
    write_cameras_txt(str(output_path / "cameras.txt"), {1: unified_camera})
    write_images_txt(str(output_path / "images.txt"), updated_images, max_points2d)
    
    # Copy points3D.txt if it exists
    if points3d_txt.exists():
        shutil.copy2(str(points3d_txt), str(output_path / "points3D.txt"))
        print("Copied points3D.txt")
    
    print("Conversion completed successfully!")
    print(f"Unified camera model: PINHOLE")
    print(f"Parameters: fx={unified_camera.params[0]:.2f}, fy={unified_camera.params[1]:.2f}, "
          f"cx={unified_camera.params[2]:.2f}, cy={unified_camera.params[3]:.2f}")


def find_actual_image_files(images_dir: str, image_names: List[str]) -> Dict[str, str]:
    """Find actual image files and return mapping from original name to actual name"""
    if not os.path.exists(images_dir):
        return {}
    
    # Get all files in images directory
    actual_files = set()
    for f in os.listdir(images_dir):
        if os.path.isfile(os.path.join(images_dir, f)):
            actual_files.add(f.lower())
    
    name_mapping = {}
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    for original_name in image_names:
        # Try exact match first
        if original_name.lower() in actual_files:
            name_mapping[original_name] = original_name
            continue
        
        # Get base name without extension
        base_name = os.path.splitext(original_name)[0]
        
        # Try different extensions
        found = False
        for ext in supported_extensions:
            candidate = base_name + ext
            if candidate.lower() in actual_files:
                # Find the actual case-sensitive filename
                for actual_file in os.listdir(images_dir):
                    if actual_file.lower() == candidate.lower():
                        name_mapping[original_name] = actual_file
                        found = True
                        break
                if found:
                    break
        
        if not found:
            print(f"Warning: Could not find actual file for {original_name}")
            name_mapping[original_name] = original_name  # Keep original name
    
    return name_mapping


# qvec2rotmat function now imported from colmap_utils


def calculate_camera_world_positions(images: Dict[int, ColmapImage]) -> Dict[int, np.ndarray]:
    """Calculate camera world positions from COLMAP images data"""
    camera_positions = {}
    
    for image_id, image in images.items():
        # Calculate camera center using utility function
        camera_center = calculate_camera_center(image.qvec, image.tvec)
        
        camera_positions[image_id] = camera_center
    
    return camera_positions


def filter_far_cameras(images: Dict[int, ColmapImage], filter_ratio: float = 0.2,
                      reference_point: List[float] = [0.0, 0.0, 0.0]) -> Tuple[Dict[int, ColmapImage], int]:
    """Remove cameras that are farthest from the reference point
    
    Args:
        images: Dictionary of images with camera poses
        filter_ratio: Ratio of farthest cameras to remove (0.2 = 20%)
        reference_point: Reference point for distance calculation [x, y, z]
    
    Returns:
        Filtered images dictionary and number of removed cameras
    """
    if not images or filter_ratio <= 0:
        return images, 0
    
    # Calculate camera world positions
    camera_positions = calculate_camera_world_positions(images)
    
    # Calculate distances from reference point
    ref_point = np.array(reference_point)
    distances = {}
    
    for image_id, camera_pos in camera_positions.items():
        distance = np.linalg.norm(camera_pos - ref_point)
        distances[image_id] = distance
    
    # Sort by distance and determine which cameras to keep
    sorted_cameras = sorted(distances.items(), key=lambda x: x[1])
    n_total = len(sorted_cameras)
    n_to_remove = max(1, int(n_total * filter_ratio))
    n_to_keep = n_total - n_to_remove
    
    # Keep the closest cameras
    cameras_to_keep = set(image_id for image_id, _ in sorted_cameras[:n_to_keep])
    
    # Filter images
    filtered_images = {image_id: image for image_id, image in images.items() 
                      if image_id in cameras_to_keep}
    
    n_removed = n_total - len(filtered_images)
    
    return filtered_images, n_removed

# ============================ Main ============================

# Default configuration parameters
DEFAULT_MAX_POINTS2D = 100
DEFAULT_FILTER_RATIO = 0.2
DEFAULT_REFERENCE_POINT = [0.0, 0.0, 0.0]

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="COLMAP Dataset Refinement Tool - Convert and refine COLMAP datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion to unified pinhole model
  python colmap_refine.py -i ./sparse/0 -o ./refined

  # Convert with custom resolution and point limiting
  python colmap_refine.py -i ./sparse/0 -o ./refined --target-width 1920 --target-height 1080 --max-points2d 100

  # Apply camera filtering to remove outliers
  python colmap_refine.py -i ./sparse/0 -o ./refined --filter-cameras --filter-ratio 0.2
        """
    )

    # Add common arguments
    parser.add_argument('--input-dir', '-i', type=str, default=r"D:\Data\BaoAn\colmap\rc_colmap",
                       help='Input COLMAP model directory')
    parser.add_argument('--output-dir', '-o', type=str, default=r"D:\Data\BaoAn\colmap\rc_colmap_refine",
                       help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                       help='Enable verbose output')

    # Target resolution
    parser.add_argument('--target-width', type=int,
                       help='Target image width (automatic if not specified)')
    parser.add_argument('--target-height', type=int,
                       help='Target image height (automatic if not specified)')

    # Points2D limiting
    parser.add_argument('--max-points2d', type=int, default=DEFAULT_MAX_POINTS2D,
                       help=f'Maximum POINTS2D per image (default: {DEFAULT_MAX_POINTS2D}, use 0 for unlimited)')

    # Camera filtering
    parser.add_argument('--filter-cameras', action='store_true',
                       help='Enable camera position filtering to remove outliers')
    parser.add_argument('--filter-ratio', type=float, default=DEFAULT_FILTER_RATIO,
                       help=f'Ratio of farthest cameras to remove (default: {DEFAULT_FILTER_RATIO})')
    parser.add_argument('--reference-point', type=float, nargs=3,
                       default=DEFAULT_REFERENCE_POINT, metavar=('X', 'Y', 'Z'),
                       help=f'Reference point for distance calculation (default: {DEFAULT_REFERENCE_POINT})')

    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()

    print("=== COLMAP Dataset Refinement Tool ===")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")

    # Set target resolution if specified
    target_resolution = None
    if args.target_width and args.target_height:
        target_resolution = (args.target_width, args.target_height)
        print(f"Target resolution: {args.target_width}x{args.target_height}")
    else:
        print("Using automatic resolution detection")

    # Set POINTS2D limit
    max_points2d = args.max_points2d if args.max_points2d > 0 else None
    if max_points2d is not None:
        print(f"Limiting POINTS2D to {max_points2d} per image")
    else:
        print("Keeping all POINTS2D entries")

    # Show camera filtering settings
    if args.filter_cameras and args.filter_ratio > 0:
        print(f"Camera filtering enabled: removing {args.filter_ratio*100:.1f}% farthest cameras from reference point {args.reference_point}")
    else:
        print("Camera filtering disabled")

    if args.verbose:
        print("Verbose output enabled")

    try:
        convert_dataset(
            args.input_dir, args.output_dir,
            target_resolution, max_points2d,
            args.filter_cameras, args.filter_ratio, args.reference_point
        )
        print("\n=== Refinement completed successfully! ===")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())