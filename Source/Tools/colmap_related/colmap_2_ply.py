#!/usr/bin/env python3
"""
COLMAP to UE Coordinate System Converter
=========================================

Converts COLMAP sparse reconstruction (cameras and 3D points) to Unreal Engine coordinate system.
Supports both binary and text COLMAP formats with comprehensive coordinate system transformation.

Features:
- Automatic format detection and conversion
- COLMAP to UE coordinate system transformation
- Point cloud filtering (outlier removal, height filtering)
- Multiple output formats (PLY, pose files)
- Comprehensive error handling and validation

Outputs:
- cameras_colmap.ply: Original COLMAP camera positions
- points_colmap.ply: Original COLMAP 3D points
- cameras_ue.ply: UE camera positions (left-handed, cm)
- points_ue.ply: UE 3D points (left-handed, cm)
- pose_ue.txt: UE camera poses with quaternions

Usage:
    python Colmap_2_ply.py --input-dir INPUT_DIR --output-dir OUTPUT_DIR [options]

Example:
    python Colmap_2_ply.py -i ./sparse/0 -o ./output --filter-outliers --filter-ratio 0.05
"""

import os
import sys
import math
import argparse
from typing import Tuple
import numpy as np
from colmap_utils import (
    read_images_txt, read_points3d_txt, read_images_binary, read_points3d_binary,
    detect_colmap_format, write_ply, qvec2rotmat, colmap_to_ue_transform, 
    filter_outlier_points, filter_low_height_points 
)

def rotation_matrix_to_ue_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to UE quaternion format.
    UE quaternion order: [x, y, z, w] (scalar last)
    COLMAP quaternion order: [w, x, y, z] (scalar first)
    
    Args:
        R: 3x3 rotation matrix (camera to world in UE coordinates)
    
    Returns:
        UE quaternion [x, y, z, w]
    """
    # Extract quaternion from rotation matrix using Shepperd's method
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    # Return UE format: [x, y, z, w]
    return np.array([qx, qy, qz, qw], dtype=float)


def parse_images_txt_to_camera_data(path: str):
    """Parse images.txt and extract camera centers and directions"""
    images = read_images_txt(path)
    cams = []

    for img in images.values():
        if not (np.all(np.isfinite(img.qvec)) and np.all(np.isfinite(img.tvec))):
            continue

        R = qvec2rotmat(img.qvec)  # world -> cam
        Rt = R.T                   # cam -> world
        C = -Rt @ img.tvec
        fwd = Rt @ np.array([0.0, 0.0, 1.0])
        nrm = np.linalg.norm(fwd)
        if not np.isfinite(nrm) or nrm < 1e-12:
            continue
        fwd /= nrm

        cams.append({
            "id": img.image_id,
            "center": C,
            "dir": fwd,
            "name": img.name,
            "cam_id": img.camera_id,
            "R_w2c": R
        })

    return cams

def parse_points3D_txt_to_arrays(path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Parse points3D.txt and return xyz, rgb arrays"""
    points3d = read_points3d_txt(path)

    xyz, rgb = [], []
    skipped = 0

    for point in points3d.values():
        if np.isfinite(point.xyz).all():
            xyz.append(point.xyz)
            rgb.append(point.rgb)
        else:
            skipped += 1

    xyz = np.array(xyz, dtype=float) if xyz else np.zeros((0,3), dtype=float)
    rgb = np.array(rgb, dtype=np.uint8) if rgb else np.zeros((0,3), dtype=np.uint8)
    return xyz, rgb, skipped


def parse_images_bin_to_camera_data(path: str):
    """Parse images.bin and extract camera centers and directions"""
    images = read_images_binary(path)
    cams = []

    for img in images.values():
        if not (np.all(np.isfinite(img.qvec)) and np.all(np.isfinite(img.tvec))):
            continue

        R = qvec2rotmat(img.qvec)  # world -> cam
        Rt = R.T                   # cam -> world
        C = -Rt @ img.tvec
        fwd = Rt @ np.array([0.0, 0.0, 1.0])
        nrm = np.linalg.norm(fwd)
        if not np.isfinite(nrm) or nrm < 1e-12:
            continue
        fwd /= nrm

        cams.append({
            "id": img.image_id,
            "center": C,
            "dir": fwd,
            "name": img.name,
            "cam_id": img.camera_id,
            "R_w2c": R,
        })

    return cams


def parse_points3D_bin_to_arrays(path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Parse points3D.bin and return xyz, rgb arrays"""
    points3d = read_points3d_binary(path)

    xyz, rgb = [], []
    skipped = 0

    for point in points3d.values():
        if np.isfinite(point.xyz).all():
            xyz.append(point.xyz)
            rgb.append(point.rgb)
        else:
            skipped += 1

    xyz = np.array(xyz, dtype=float) if xyz else np.zeros((0,3), dtype=float)
    rgb = np.array(rgb, dtype=np.uint8) if rgb else np.zeros((0,3), dtype=np.uint8)
    return xyz, rgb, skipped


# ============================ Main ============================

# Default configuration parameters
DEFAULT_CAMERA_COLOR = [255, 0, 0]        # COLMAP camera point color (red)
DEFAULT_CAMERA_UE_COLOR = [0, 255, 0]     # UE camera point color (green)
DEFAULT_FILTER_RATIO = 0.05               # Ratio of farthest points to remove
DEFAULT_MIN_HEIGHT = -25.0                # Minimum height threshold in meters

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="COLMAP to UE Coordinate System Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python Colmap_2_ply.py -i ./sparse/0 -o ./output

  # With outlier filtering
  python Colmap_2_ply.py -i ./sparse/0 -o ./output --filter-outliers --filter-ratio 0.05

  # With height filtering
  python Colmap_2_ply.py -i ./sparse/0 -o ./output --filter-height --min-height -10.0
        """
    )

    parser.add_argument('--input-dir', '-i', type=str, default=r"D:\Data\BaoAnS\colmap\refined",
                       help='Input COLMAP model directory')
    parser.add_argument('--output-dir', '-o', type=str, default=r"D:\Data\BaoAnS\colmap\ply",
                       help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                       help='Enable verbose output')

    # Camera colors
    parser.add_argument('--camera-color', type=int, nargs=3, default=DEFAULT_CAMERA_COLOR,
                       metavar=('R', 'G', 'B'), help='COLMAP camera point color (RGB)')
    parser.add_argument('--camera-ue-color', type=int, nargs=3, default=DEFAULT_CAMERA_UE_COLOR,
                       metavar=('R', 'G', 'B'), help='UE camera point color (RGB)')

    # Filtering options
    parser.add_argument('--sanitize', action='store_true',
                       help='Remove NaN/Inf vertices')
    parser.add_argument('--filter-outliers', action='store_true', default=True,
                       help='Filter outlier points (remove farthest points as noise)')
    parser.add_argument('--filter-ratio', type=float, default=DEFAULT_FILTER_RATIO,
                       help=f'Ratio of farthest points to remove (default: {DEFAULT_FILTER_RATIO})')
    parser.add_argument('--filter-height', action='store_true', default=True,
                       help='Remove points below minimum height threshold')
    parser.add_argument('--min-height', type=float, default=DEFAULT_MIN_HEIGHT,
                       help=f'Minimum height (Z coordinate) in meters (default: {DEFAULT_MIN_HEIGHT})')

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()

    print("=== COLMAP to UE Coordinate System Converter ===")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Detect format and validate
    has_txt, has_bin, data_dir = detect_colmap_format(args.input_dir)

    if not has_txt and not has_bin:
        raise RuntimeError(f"No COLMAP model found at {args.input_dir}")

    if args.verbose:
        print(f"[info] Using COLMAP data from: {data_dir}")

    images_txt = os.path.join(data_dir, 'images.txt')
    points3D_txt = os.path.join(data_dir, 'points3D.txt')
    images_bin = os.path.join(data_dir, 'images.bin')
    points3D_bin = os.path.join(data_dir, 'points3D.bin')

    cameras = []
    xyz_pts = np.zeros((0,3), dtype=float)
    rgb_pts = np.zeros((0,3), dtype=np.uint8)

    try:
        if has_txt:
            if args.verbose:
                print(f"[info] Parsing TEXT model at {data_dir}")
            cameras = parse_images_txt_to_camera_data(images_txt)
            xyz_pts, rgb_pts, n_bad = parse_points3D_txt_to_arrays(points3D_txt)
            if args.verbose:
                print(f"[info] Text model: cameras={len(cameras)}, points={xyz_pts.shape[0]} (skipped={n_bad})")
        elif has_bin:
            if args.verbose:
                print(f"[info] Parsing BINARY model at {data_dir}")
            cameras = parse_images_bin_to_camera_data(images_bin)
            xyz_pts, rgb_pts, n_bad = parse_points3D_bin_to_arrays(points3D_bin)
            if args.verbose:
                print(f"[info] Binary model: cameras={len(cameras)}, points={xyz_pts.shape[0]} (skipped={n_bad})")
    except Exception as e:
        print(f"[error] Parse failed: {e}")
        raise RuntimeError("Failed to parse COLMAP model (TXT/BIN).")

    # Prepare arrays
    if len(cameras) > 0:
        cam_xyz = np.stack([c["center"] for c in cameras], axis=0)
        cam_dirs = np.stack([c["dir"] for c in cameras], axis=0)
        cam_rgb = np.tile(np.array(args.camera_color, dtype=np.uint8), (len(cameras), 1))

        # Convert cameras directly to UE coordinates (left-handed, centimeters)
        cam_ue_xyz, cam_ue_dirs = colmap_to_ue_transform(cam_xyz, cam_dirs)
        cam_ue_rgb = np.tile(np.array(args.camera_ue_color, dtype=np.uint8), (len(cameras), 1))
    else:
        print("[warn] No cameras found in the model.")

    # Transform point cloud directly to UE coordinates (left-handed, centimeters)
    pts_ue_xyz, _ = colmap_to_ue_transform(xyz_pts, None)
    pts_ue_rgb = rgb_pts

    # Optional height filtering - remove points below minimum height threshold
    if args.filter_height and len(xyz_pts) > 0:
        if args.verbose:
            print(f"[info] Height filtering: removing points below {args.min_height}m (Z coordinate in COLMAP)")
        xyz_pts, rgb_pts, n_removed = filter_low_height_points(xyz_pts, rgb_pts, args.min_height)
        # Recompute UE coordinates after height filtering
        pts_ue_xyz, _ = colmap_to_ue_transform(xyz_pts, None)
        pts_ue_rgb = rgb_pts
        if args.verbose:
            print(f"[info] Height filtering: removed {n_removed} points, kept {len(xyz_pts)} points")
    elif args.verbose and len(xyz_pts) > 0:
        print(f"[info] Height filtering disabled, keeping all {len(xyz_pts)} points")

    # Optional outlier filtering - remove farthest points as noise
    if args.filter_outliers and len(xyz_pts) > 0:
        if args.verbose:
            print(f"[info] Filtering outliers: removing {args.filter_ratio*100:.1f}% farthest points from point cloud center")
        xyz_pts, rgb_pts, n_removed = filter_outlier_points(xyz_pts, rgb_pts, args.filter_ratio)
        # Recompute UE coordinates after filtering
        pts_ue_xyz, _ = colmap_to_ue_transform(xyz_pts, None)
        pts_ue_rgb = rgb_pts
        if args.verbose:
            print(f"[info] Outlier filtering: removed {n_removed} points, kept {len(xyz_pts)} points")
    elif args.verbose and len(xyz_pts) > 0:
        print(f"[info] Outlier filtering disabled, keeping all {len(xyz_pts)} points")

    # Optional sanitization to avoid NaN warnings in Meshlab
    if args.sanitize:
        cam_mask = (np.isfinite(cam_xyz).all(axis=1) &
                    np.isfinite(cam_dirs).all(axis=1) &
                    np.isfinite(cam_ue_xyz).all(axis=1) &
                    np.isfinite(cam_ue_dirs).all(axis=1))
        pts_mask = (np.isfinite(xyz_pts).all(axis=1) &
                    np.isfinite(pts_ue_xyz).all(axis=1))
        kept_cam = int(cam_mask.sum())
        kept_pts = int(pts_mask.sum())
        if args.verbose:
            print(f"[info] sanitize: cameras kept={kept_cam}/{len(cam_xyz)}, points kept={kept_pts}/{len(xyz_pts)}")
        cam_xyz, cam_dirs, cam_rgb = cam_xyz[cam_mask], cam_dirs[cam_mask], cam_rgb[cam_mask]
        cam_ue_xyz, cam_ue_dirs, cam_ue_rgb = cam_ue_xyz[cam_mask], cam_ue_dirs[cam_mask], cam_ue_rgb[cam_mask]
        xyz_pts, rgb_pts = xyz_pts[pts_mask], rgb_pts[pts_mask]
        pts_ue_xyz, pts_ue_rgb = pts_ue_xyz[pts_mask], pts_ue_rgb[pts_mask]
        cameras = [cam for i, cam in enumerate(cameras) if cam_mask[i]]

    # Write PLYs
    cams_out = os.path.join(args.output_dir, 'cameras_colmap.ply')
    pts_out = os.path.join(args.output_dir, 'points_colmap.ply')
    cams_ue_out = os.path.join(args.output_dir, 'cameras_ue.ply')
    pts_ue_out = os.path.join(args.output_dir, 'points_ue.ply')
    pose_ue_out = os.path.join(args.output_dir, 'pose_ue.txt')

    # Create zero normals for point clouds
    if len(xyz_pts) > 0:
        pts_normals = np.zeros_like(xyz_pts)  # (N, 3) array of zeros for COLMAP points
        pts_ue_normals = np.zeros_like(pts_ue_xyz)  # (N, 3) array of zeros for UE points
    else:
        pts_normals = None
        pts_ue_normals = None

    if args.verbose:
        print("[info] Writing output PLY files...")
    write_ply(cams_out, cam_xyz, rgb=cam_rgb, normals=cam_dirs)
    write_ply(pts_out, xyz_pts, rgb=rgb_pts, normals=pts_normals)
    write_ply(cams_ue_out, cam_ue_xyz, rgb=cam_ue_rgb, normals=cam_ue_dirs)
    write_ply(pts_ue_out, pts_ue_xyz, rgb=pts_ue_rgb, normals=pts_ue_normals)

    # Write UE pose file with quaternions
    if args.verbose:
        print(f"[info] Writing UE pose file: {pose_ue_out}")
    qaq = 0
    with open(pose_ue_out, 'w', encoding='utf-8') as f:
        f.write("# UE coordinate system poses (left-handed, cm)\n")
        f.write("# UE coordinate axes: +X forward, +Y right, +Z up\n")
        f.write("# Format: Timestamp X Y Z Qx Qy Qz Qw\n")
        f.write("# Quaternion order: [x, y, z, w] (UE format, scalar last)\n")
        for i, cam in enumerate(cameras):
            # Get UE position
            ue_pos = cam_ue_xyz[i]
            
            R_w2c_colmap = cam["R_w2c"]
            R_c2w_colmap = R_w2c_colmap.T

            T_world = np.array([
                [0,  1,  0],  # X_ue_world = Y_colmap_world
                [1,  0,  0],  # Y_ue_world = X_colmap_world
                [0,  0,  1]   # Z_ue_world = Z_colmap_world (both Z-up)
            ])

            T_cam = np.array([
                [0,  0,  1],  # X_actor (forward) = Z_cam (forward)
                [1,  0,  0],  # Y_actor (right) = X_cam (right)
                [0, -1,  0]   # Z_actor (up) = -Y_cam (down)
            ])

            R_c2w_ue = T_world @ R_c2w_colmap @ T_cam.T

            ue_quat = rotation_matrix_to_ue_quaternion(R_c2w_ue)
            
            # Add pseudo timestamp (1 decimal place like in UE code)
            pseudo_timestamp = float(i)
            f.write(f"{pseudo_timestamp:.1f} {ue_pos[0]:.6f} {ue_pos[1]:.6f} {ue_pos[2]:.6f} {ue_quat[0]:.6f} {ue_quat[1]:.6f} {ue_quat[2]:.6f} {ue_quat[3]:.6f}\n")

    if args.verbose:
        print("[done] Outputs:")
        print(f"  COLMAP cameras: {cams_out}\n  COLMAP points:  {pts_out}\n  UE cameras:     {cams_ue_out}\n  UE points:      {pts_ue_out}\n  UE poses:       {pose_ue_out}")
    else:
        print("[done] Conversion complete.")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as ex:
        print(f"[error] {ex}", file=sys.stderr)
        sys.exit(1)
