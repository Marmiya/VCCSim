#!/usr/bin/env python3
"""
COLMAP Pinhole Converter
Converts Reality Capture COLMAP datasets with multiple non-pinhole cameras 
to a single pinhole camera model for 3D Gaussian Splatting compatibility.

Features:
- Convert multiple camera models to unified PINHOLE model
- Resize images to target resolution
- Limit POINTS2D per image to reduce file size
- Filter cameras by distance from reference point (NEW)

Usage:
    Modify the parameters in the CONFIG section below and run:
    python colmap_refine.py
"""

import os
import sys
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, NamedTuple
import struct

# ================================
# CONFIG - Modify these parameters
# ================================
INPUT_DIR = r"E:\BaoAn\BaoAnColmap\sparse\0_txt"
OUTPUT_DIR = r"E:\BaoAn\BaoAnColmap\sparse\0_txt\1"

# Optional: Set target resolution (None for automatic)
TARGET_WIDTH = None   # e.g., 1920 or None
TARGET_HEIGHT = None  # e.g., 1080 or None

# Optional: Limit number of POINTS2D per image to reduce file size
# Set to None to keep all points, or a number (e.g., 100) to limit
MAX_POINTS2D_PER_IMAGE = 10  # e.g., 100 or None

# Optional: Filter cameras too far from reference point
FILTER_FAR_CAMERAS = True      # Enable/disable camera position filtering
CAMERA_FILTER_RATIO = 0.7      # Ratio of farthest cameras to remove (0.2 = 20%)
REFERENCE_POINT = [0.0, 0.0, 0.0]  # Reference point for distance calculation (default: origin)

# ================================


class Camera(NamedTuple):
    """Camera parameters structure"""
    id: int
    model: str
    width: int
    height: int
    params: List[float]


class Image(NamedTuple):
    """Image parameters structure"""
    id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str
    points2d: List[Tuple[float, float, int]]  # x, y, point3d_id


def read_cameras_txt(filepath: str) -> Dict[int, Camera]:
    """Read cameras.txt file"""
    cameras = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            
            cameras[camera_id] = Camera(camera_id, model, width, height, params)
    
    return cameras


def read_images_txt(filepath: str) -> Dict[int, Image]:
    """Read images.txt file"""
    images = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue
        
        # Image line
        parts = line.split()
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]
        
        # Points2D line (next line)
        points2d = []
        if i + 1 < len(lines):
            points_line = lines[i + 1].strip()
            if points_line and not points_line.startswith('#'):
                points_parts = points_line.split()
                for j in range(0, len(points_parts), 3):
                    if j + 2 < len(points_parts):
                        x = float(points_parts[j])
                        y = float(points_parts[j + 1])
                        point3d_id = int(points_parts[j + 2])
                        points2d.append((x, y, point3d_id))
        
        images[image_id] = Image(image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name, points2d)
        i += 2  # Skip the points2D line
    
    return images


def write_cameras_txt(filepath: str, cameras: Dict[int, Camera]):
    """Write cameras.txt file"""
    with open(filepath, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: {}\n".format(len(cameras)))
        
        for camera in cameras.values():
            params_str = ' '.join(map(str, camera.params))
            f.write(f"{camera.id} {camera.model} {camera.width} {camera.height} {params_str}\n")


def write_images_txt(filepath: str, images: Dict[int, Image], max_points2d: int = None):
    """Write images.txt file with optional limit on POINTS2D per image"""
    with open(filepath, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        # Calculate statistics
        total_points = sum(len(img.points2d) for img in images.values())
        if max_points2d is not None:
            limited_points = sum(min(len(img.points2d), max_points2d) for img in images.values())
            mean_obs = limited_points / len(images) if images else 0
            f.write(f"# Number of images: {len(images)}, mean observations per image: {mean_obs:.2f} (limited to {max_points2d} per image)\n")
        else:
            mean_obs = total_points / len(images) if images else 0
            f.write(f"# Number of images: {len(images)}, mean observations per image: {mean_obs:.2f}\n")
        
        for image in images.values():
            f.write(f"{image.id} {image.qw} {image.qx} {image.qy} {image.qz} "
                   f"{image.tx} {image.ty} {image.tz} {image.camera_id} {image.name}\n")
            
            # Limit points2d if specified
            points2d_to_write = image.points2d
            if max_points2d is not None and len(points2d_to_write) > max_points2d:
                points2d_to_write = points2d_to_write[:max_points2d]
            
            if points2d_to_write:
                points_str = ' '.join([f"{x} {y} {pid}" for x, y, pid in points2d_to_write])
                f.write(f"{points_str}\n")
            else:
                f.write("\n")


def convert_camera_to_pinhole(camera: Camera, target_width: int = None, target_height: int = None) -> Camera:
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
    return Camera(
        id=camera.id,
        model="PINHOLE",
        width=width,
        height=height,
        params=[fx, fy, cx, cy]
    )


def calculate_average_focal_length(cameras: Dict[int, Camera]) -> Tuple[float, float]:
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


def convert_dataset(input_dir: str, output_dir: str, target_resolution: Tuple[int, int] = None, max_points2d: int = None):
    """Convert COLMAP dataset to single pinhole camera model"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Validate input directory structure
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Check if input_dir contains sparse/0 subdirectory (new structure)
    sparse_dir = input_path / "sparse" / "0"
    images_dir = input_path / "images"
    
    if sparse_dir.exists():
        # New structure: input_dir/sparse/0/ contains COLMAP files
        cameras_txt = sparse_dir / "cameras.txt"
        images_txt = sparse_dir / "images.txt"
        points3d_txt = sparse_dir / "points3D.txt"
        print(f"Using COLMAP data from: {sparse_dir}")
    else:
        # Old structure: input_dir directly contains COLMAP files
        cameras_txt = input_path / "cameras.txt"
        images_txt = input_path / "images.txt" 
        points3d_txt = input_path / "points3D.txt"
        images_dir = input_path.parent / "images"  # Assume images are in parent/images
        print(f"Using COLMAP data from: {input_path}")
    
    if not cameras_txt.exists() or not images_txt.exists():
        raise FileNotFoundError("Required COLMAP files (cameras.txt, images.txt) not found")
    
    print(f"Images directory: {images_dir}")
    
    print(f"Reading COLMAP dataset from: {input_dir}")
    
    # Read original data
    cameras = read_cameras_txt(str(cameras_txt))
    images = read_images_txt(str(images_txt))
    
    print(f"Found {len(cameras)} cameras and {len(images)} images")
    print(f"Camera models: {set(cam.model for cam in cameras.values())}")
    
    # Find actual image files and create name mapping
    image_names = [img.name for img in images.values()]
    name_mapping = find_actual_image_files(str(images_dir), image_names)
    
    if name_mapping:
        changed_count = sum(1 for orig, actual in name_mapping.items() if orig != actual)
        print(f"Found {len(name_mapping)} images, {changed_count} names will be corrected")
    
    # Optional: Filter cameras by distance from reference point
    if FILTER_FAR_CAMERAS and CAMERA_FILTER_RATIO > 0:
        print(f"Filtering cameras: removing {CAMERA_FILTER_RATIO*100:.1f}% farthest cameras from reference point {REFERENCE_POINT}")
        images, n_removed_cameras = filter_far_cameras(images, CAMERA_FILTER_RATIO, REFERENCE_POINT)
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
    unified_camera = Camera(
        id=1,
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
        
        updated_images[img.id] = Image(
            img.id, img.qw, img.qx, img.qy, img.qz,
            img.tx, img.ty, img.tz,
            1,  # Use unified camera ID
            corrected_name,  # Use corrected filename
            scaled_points2d
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


def qvec2rotmat(qvec: List[float]) -> np.ndarray:
    """Convert quaternion vector to rotation matrix (from colmap_2_ply.py)"""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ], dtype=float)


def calculate_camera_world_positions(images: Dict[int, Image]) -> Dict[int, np.ndarray]:
    """Calculate camera world positions from COLMAP images data"""
    camera_positions = {}
    
    for image_id, image in images.items():
        # Convert quaternion to rotation matrix (world -> camera)
        qvec = [image.qw, image.qx, image.qy, image.qz]
        R = qvec2rotmat(qvec)
        
        # Calculate camera center in world coordinates
        # R is world-to-camera, so camera-to-world is R.T
        t = np.array([image.tx, image.ty, image.tz])
        camera_center = -R.T @ t
        
        camera_positions[image_id] = camera_center
    
    return camera_positions


def filter_far_cameras(images: Dict[int, Image], filter_ratio: float = 0.2, 
                      reference_point: List[float] = [0.0, 0.0, 0.0]) -> Tuple[Dict[int, Image], int]:
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


def main():
    """Main function"""
    # Use parameters from CONFIG section
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    
    print(f"Using configured paths:")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    
    # Set target resolution if specified
    target_resolution = None
    if TARGET_WIDTH and TARGET_HEIGHT:
        target_resolution = (TARGET_WIDTH, TARGET_HEIGHT)
        print(f"Target resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    else:
        print("Using automatic resolution detection")
    
    # Set POINTS2D limit if specified
    if MAX_POINTS2D_PER_IMAGE is not None:
        print(f"Limiting POINTS2D to {MAX_POINTS2D_PER_IMAGE} per image")
    else:
        print("Keeping all POINTS2D entries")
    
    # Show camera filtering settings
    if FILTER_FAR_CAMERAS and CAMERA_FILTER_RATIO > 0:
        print(f"Camera filtering enabled: removing {CAMERA_FILTER_RATIO*100:.1f}% farthest cameras from reference point {REFERENCE_POINT}")
    else:
        print("Camera filtering disabled")
    
    try:
        convert_dataset(input_dir, output_dir, target_resolution, MAX_POINTS2D_PER_IMAGE)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())