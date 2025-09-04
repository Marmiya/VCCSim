#!/usr/bin/env python3
"""
COLMAP Pinhole Converter
Converts Reality Capture COLMAP datasets with multiple non-pinhole cameras 
to a single pinhole camera model for 3D Gaussian Splatting compatibility.

Usage:
    python Colmap_pinhole.py input_path output_path
    
Example:
    python Colmap_pinhole.py "D:\Data\360_v2\garden\mesh\Colmap\sparse\0" "D:\Data\360_v2\garden\mesh\Colmap_Refine"
"""

import os
import sys
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, NamedTuple
import struct


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


def write_images_txt(filepath: str, images: Dict[int, Image]):
    """Write images.txt file"""
    with open(filepath, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write("# Number of images: {}, mean observations per image: {:.2f}\n".format(
            len(images), np.mean([len(img.points2d) for img in images.values()]) if images else 0))
        
        for image in images.values():
            f.write(f"{image.id} {image.qw} {image.qx} {image.qy} {image.qz} "
                   f"{image.tx} {image.ty} {image.tz} {image.camera_id} {image.name}\n")
            
            if image.points2d:
                points_str = ' '.join([f"{x} {y} {pid}" for x, y, pid in image.points2d])
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


def convert_dataset(input_dir: str, output_dir: str, target_resolution: Tuple[int, int] = None):
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
    
    write_cameras_txt(str(output_path / "cameras.txt"), {1: unified_camera})
    write_images_txt(str(output_path / "images.txt"), updated_images)
    
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


def main():
    """Main function"""
    # Default paths - updated to use base directory
    default_input = r"D:\Data\360_v2\garden\mesh\Colmap"
    default_output = r"D:\Data\360_v2\garden\mesh\Colmap_Refine/sparse/0/"
    
    if len(sys.argv) >= 3:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
    elif len(sys.argv) == 2:
        input_dir = sys.argv[1]
        output_dir = default_output
    else:
        input_dir = default_input
        output_dir = default_output
        print(f"Using default paths:")
        print(f"Input:  {input_dir}")
        print(f"Output: {output_dir}")
        print("Usage: python Colmap_pinhole.py [input_dir] [output_dir] [target_width] [target_height]")
        print("Expected structure: input_dir/sparse/0/ (COLMAP files) and input_dir/images/ (image files)")
    
    target_resolution = None
    if len(sys.argv) >= 5:
        try:
            target_width = int(sys.argv[3])
            target_height = int(sys.argv[4])
            target_resolution = (target_width, target_height)
        except ValueError:
            print("Warning: Invalid resolution parameters, using automatic resolution")
    
    try:
        convert_dataset(input_dir, output_dir, target_resolution)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())