#!/usr/bin/env python3
"""
COLMAP vs VCCSim CameraInfo Quaternion Comparison Script

This script compares quaternions between COLMAP images.txt/images.bin files 
and VCCSim CameraInfo data files to verify rotation consistency.

Usage:
    # Use default paths
    python Colmap_CameraInfo_comparison.py
    
    # Specify custom paths
    python Colmap_CameraInfo_comparison.py --colmap <colmap_path> --vccsim <vccsim_path>
    
    # Paths can be files or directories:
    # - COLMAP: sparse directory or images.txt/images.bin file
    # - VCCSim: transform directory or CameraInfo data file

Default paths:
    COLMAP: D:\Data\360_v2\garden\mesh\Colmap\sparse\0
    VCCSim: C:\UEProjects\VCCSimDev\Saved\TriangleSplatting\Test Transform

Author: VCCSim Development Team
"""

import argparse
import os
import struct
import numpy as np
from typing import Dict, List, Tuple, NamedTuple
from pathlib import Path


class CameraData(NamedTuple):
    """Structure to hold camera data for comparison"""
    image_name: str
    position: np.ndarray  # [x, y, z]
    quaternion: np.ndarray  # [qw, qx, qy, qz]


def read_colmap_images_txt(file_path: str) -> Dict[str, CameraData]:
    """
    Read COLMAP images.txt file and extract camera poses.
    
    Format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    """
    cameras = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            if len(parts) < 10:
                continue
            
            try:
                # Parse COLMAP format
                image_id = int(parts[0])
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = int(parts[8])
                image_name = parts[9]
                
                # Store camera data
                cameras[image_name] = CameraData(
                    image_name=image_name,
                    position=np.array([tx, ty, tz]),
                    quaternion=np.array([qw, qx, qy, qz])
                )
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse line: {line}")
                continue
    
    print(f"Read {len(cameras)} cameras from COLMAP images.txt")
    return cameras


def read_colmap_images_bin(file_path: str) -> Dict[str, CameraData]:
    """
    Read COLMAP images.bin file and extract camera poses.
    
    Binary format structure for each image:
    - IMAGE_ID (uint64)
    - QW, QX, QY, QZ (4 * double)  
    - TX, TY, TZ (3 * double)
    - CAMERA_ID (uint64)
    - NAME (null-terminated string)
    - POINTS2D (variable length)
    """
    cameras = {}
    
    with open(file_path, 'rb') as f:
        # Read number of images
        num_images = struct.unpack('<Q', f.read(8))[0]  # uint64
        
        for _ in range(num_images):
            # Read image data
            image_id = struct.unpack('<Q', f.read(8))[0]  # uint64
            
            # Read quaternion (qw, qx, qy, qz)
            qw, qx, qy, qz = struct.unpack('<4d', f.read(32))  # 4 * double
            
            # Read translation (tx, ty, tz)  
            tx, ty, tz = struct.unpack('<3d', f.read(24))  # 3 * double
            
            # Read camera ID
            camera_id = struct.unpack('<Q', f.read(8))[0]  # uint64
            
            # Read image name (null-terminated string)
            name_chars = []
            while True:
                char = f.read(1)
                if not char or char == b'\x00':
                    break
                name_chars.append(char)
            image_name = b''.join(name_chars).decode('utf-8')
            
            # Skip 2D points data
            num_points2d = struct.unpack('<Q', f.read(8))[0]
            f.read(num_points2d * 24)  # Each 2D point: 2 * double + uint64
            
            # Store camera data
            cameras[image_name] = CameraData(
                image_name=image_name,
                position=np.array([tx, ty, tz]),
                quaternion=np.array([qw, qx, qy, qz])
            )
    
    print(f"Read {len(cameras)} cameras from COLMAP images.bin")
    return cameras


def read_vccsim_camerainfo(file_path: str) -> Dict[str, CameraData]:
    """
    Read VCCSim CameraInfo data file.
    
    Format: ImageName X Y Z QW QX QY QZ
    """
    cameras = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            if len(parts) < 8:
                continue
            
            try:
                image_name = parts[0]
                x, y, z = map(float, parts[1:4])
                qw, qx, qy, qz = map(float, parts[4:8])
                
                cameras[image_name] = CameraData(
                    image_name=image_name,
                    position=np.array([x, y, z]),
                    quaternion=np.array([qw, qx, qy, qz])
                )
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse line: {line}")
                continue
    
    print(f"Read {len(cameras)} cameras from VCCSim CameraInfo file")
    return cameras


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length"""
    return q / np.linalg.norm(q)


def quaternion_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Calculate angular distance between two quaternions.
    Returns angle in degrees.
    """
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    
    # Handle quaternion double cover (q and -q represent same rotation)
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, 0.0, 1.0)
    
    # Convert to angle in degrees
    angle_rad = 2 * np.arccos(dot)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def compare_quaternions(colmap_cameras: Dict[str, CameraData], 
                       vccsim_cameras: Dict[str, CameraData]) -> None:
    """Compare quaternions between COLMAP and VCCSim data"""
    
    # Find common images
    common_images = set(colmap_cameras.keys()) & set(vccsim_cameras.keys())
    
    if not common_images:
        print("ERROR: No common images found between datasets!")
        print(f"COLMAP images: {sorted(list(colmap_cameras.keys())[:5])}...")
        print(f"VCCSim images: {sorted(list(vccsim_cameras.keys())[:5])}...")
        return
    
    print(f"\nFound {len(common_images)} common images")
    print(f"COLMAP dataset: {len(colmap_cameras)} total images")
    print(f"VCCSim dataset: {len(vccsim_cameras)} total images")
    
    # Compare quaternions
    angular_errors = []
    position_errors = []
    
    print(f"\n{'Image Name':<30} {'Angular Error (�)':<15} {'Position Error (m)':<20} {'Status':<10}")
    print("-" * 80)
    
    sorted_images = sorted(common_images)
    for image_name in sorted_images:
        colmap_cam = colmap_cameras[image_name]
        vccsim_cam = vccsim_cameras[image_name]
        
        # Calculate angular error
        angular_error = quaternion_distance(colmap_cam.quaternion, vccsim_cam.quaternion)
        angular_errors.append(angular_error)
        
        # Calculate position error (L2 distance)
        position_error = np.linalg.norm(colmap_cam.position - vccsim_cam.position)
        position_errors.append(position_error)
        
        # Status indicator
        status = "OK" if angular_error < 5.0 else "WARN"
        if angular_error > 15.0:
            status = "ERROR"
        
        print(f"{image_name:<30} {angular_error:<15.2f} {position_error:<20.3f} {status:<10}")
    
    # Summary statistics
    angular_errors = np.array(angular_errors)
    position_errors = np.array(position_errors)
    
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Angular Errors (degrees):")
    print(f"  Mean:    {np.mean(angular_errors):.3f}�")
    print(f"  Median:  {np.median(angular_errors):.3f}�")
    print(f"  Std Dev: {np.std(angular_errors):.3f}�")
    print(f"  Min:     {np.min(angular_errors):.3f}�")
    print(f"  Max:     {np.max(angular_errors):.3f}�")
    print(f"  95th %:  {np.percentile(angular_errors, 95):.3f}�")
    
    print(f"\nPosition Errors (meters):")
    print(f"  Mean:    {np.mean(position_errors):.6f} m")
    print(f"  Median:  {np.median(position_errors):.6f} m")
    print(f"  Std Dev: {np.std(position_errors):.6f} m")
    print(f"  Min:     {np.min(position_errors):.6f} m")
    print(f"  Max:     {np.max(position_errors):.6f} m")
    print(f"  95th %:  {np.percentile(position_errors, 95):.6f} m")
    
    # Quality assessment
    good_rotations = np.sum(angular_errors < 5.0)
    acceptable_rotations = np.sum(angular_errors < 15.0)
    
    print(f"\nQUALITY ASSESSMENT:")
    print(f"  Excellent rotations (< 5�):   {good_rotations:>3}/{len(angular_errors)} ({100*good_rotations/len(angular_errors):.1f}%)")
    print(f"  Acceptable rotations (< 15�): {acceptable_rotations:>3}/{len(angular_errors)} ({100*acceptable_rotations/len(angular_errors):.1f}%)")
    
    if np.mean(angular_errors) < 5.0:
        print(f"   OVERALL STATUS: EXCELLENT - Rotations are very consistent")
    elif np.mean(angular_errors) < 15.0:
        print(f"  �  OVERALL STATUS: ACCEPTABLE - Some rotation differences detected")
    else:
        print(f"  L OVERALL STATUS: POOR - Significant rotation inconsistencies")


# Default paths
DEFAULT_COLMAP_PATH = r"D:\Data\360_v2\garden\mesh\Colmap\sparse\0"
DEFAULT_VCCSIM_PATH = r"C:\UEProjects\VCCSimDev\Saved\TriangleSplatting\Test Transform"


def find_colmap_images_file(sparse_dir: str) -> str:
    """
    Find COLMAP images file (prefer .bin, fallback to .txt) in sparse directory.
    """
    images_bin = os.path.join(sparse_dir, "images.bin")
    images_txt = os.path.join(sparse_dir, "images.txt")
    
    if os.path.exists(images_bin):
        return images_bin
    elif os.path.exists(images_txt):
        return images_txt
    else:
        raise FileNotFoundError(f"No images.bin or images.txt found in {sparse_dir}")


def find_vccsim_camerainfo_file(transform_dir: str) -> str:
    """
    Find VCCSim CameraInfo file in transform directory.
    Look for common naming patterns.
    """
    possible_names = [
        "camerainfo.txt",
        "CameraInfo.txt", 
        "camera_info.txt",
        "transforms.txt",
        "poses.txt"
    ]
    
    for name in possible_names:
        filepath = os.path.join(transform_dir, name)
        if os.path.exists(filepath):
            return filepath
    
    # If no standard name found, list available files
    if os.path.exists(transform_dir):
        files = [f for f in os.listdir(transform_dir) if f.endswith('.txt')]
        if files:
            print(f"Available .txt files in {transform_dir}:")
            for f in files:
                print(f"  - {f}")
            if len(files) == 1:
                return os.path.join(transform_dir, files[0])
    
    raise FileNotFoundError(f"No CameraInfo file found in {transform_dir}")


def main():
    parser = argparse.ArgumentParser(description='Compare COLMAP and VCCSim CameraInfo quaternions')
    parser.add_argument('--colmap', 
                       help=f'Path to COLMAP images file (images.txt or images.bin) or sparse directory. Default: {DEFAULT_COLMAP_PATH}')
    parser.add_argument('--vccsim',
                       help=f'Path to VCCSim CameraInfo data file or transform directory. Default: {DEFAULT_VCCSIM_PATH}')
    
    args = parser.parse_args()
    
    # Use default paths if not provided
    colmap_path = args.colmap if args.colmap else DEFAULT_COLMAP_PATH
    vccsim_path = args.vccsim if args.vccsim else DEFAULT_VCCSIM_PATH
    
    print("COLMAP vs VCCSim CameraInfo Quaternion Comparison")
    print("=" * 50)
    
    # Handle COLMAP path (could be file or directory)
    try:
        if os.path.isdir(colmap_path):
            # It's a sparse directory, find the images file
            colmap_file = find_colmap_images_file(colmap_path)
            print(f"COLMAP sparse dir: {colmap_path}")
            print(f"COLMAP images file: {colmap_file}")
        else:
            # It's already a file path
            colmap_file = colmap_path
            print(f"COLMAP file: {colmap_file}")
            
        if not os.path.exists(colmap_file):
            print(f"ERROR: COLMAP file not found: {colmap_file}")
            return 1
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    
    # Handle VCCSim path (could be file or directory)  
    try:
        if os.path.isdir(vccsim_path):
            # It's a directory, find the CameraInfo file
            vccsim_file = find_vccsim_camerainfo_file(vccsim_path)
            print(f"VCCSim transform dir: {vccsim_path}")
            print(f"VCCSim CameraInfo file: {vccsim_file}")
        else:
            # It's already a file path
            vccsim_file = vccsim_path
            print(f"VCCSim file: {vccsim_file}")
            
        if not os.path.exists(vccsim_file):
            print(f"ERROR: VCCSim file not found: {vccsim_file}")
            return 1
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    
    # Read COLMAP data
    try:
        if colmap_file.endswith('.bin'):
            colmap_cameras = read_colmap_images_bin(colmap_file)
        else:
            colmap_cameras = read_colmap_images_txt(colmap_file)
    except Exception as e:
        print(f"ERROR: Failed to read COLMAP file: {e}")
        return 1
    
    # Read VCCSim data
    try:
        vccsim_cameras = read_vccsim_camerainfo(vccsim_file)
    except Exception as e:
        print(f"ERROR: Failed to read VCCSim file: {e}")
        return 1
    
    # Compare quaternions
    compare_quaternions(colmap_cameras, vccsim_cameras)
    
    return 0


if __name__ == "__main__":
    exit(main())