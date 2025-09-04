#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COLMAP to UE Converter

Converts COLMAP sparse reconstruction (cameras and 3D points) to UE coordinate system.

Key assumptions:
- COLMAP world coordinate system: Right-handed, arbitrary orientation (determined by reconstruction)
- We assume COLMAP world frame follows OpenCV convention: +X right, +Y down, +Z forward (right-handed)
- UE coordinate system: Left-handed, +X forward, +Y right, +Z up, centimeters

Conversion:
- COLMAP (+X right, +Y down, +Z forward) -> UE (+X forward, +Y right, +Z up)
- Position: (X, Y, Z) -> (Y*100, X*100, Z*100)  [Y->X, X->Y, Z->Z, scale to cm]
- Direction: (X, Y, Z) -> (Y, X, Z)  [same transformation, no scaling]

Outputs:
- cameras_colmap.ply: Original COLMAP camera positions
- points_colmap.ply: Original COLMAP 3D points  
- cameras_ue.ply: UE camera positions (left-handed, cm)
- points_ue.ply: UE 3D points (left-handed, cm)
- pose_ue.txt: UE camera poses with quaternions
"""

import os
import sys
import argparse
import math
import struct
from typing import Tuple
import numpy as np


# ============================ Math helpers ============================

def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    if qvec.shape != (4,):
        qvec = np.asarray(qvec).reshape(4,)
    qw, qx, qy, qz = qvec.astype(float)
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if not np.isfinite(n) or n == 0.0:
        raise ValueError("Invalid quaternion norm")
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ], dtype=float)
    return R


def filter_outlier_points(xyz: np.ndarray, rgb: np.ndarray, filter_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, int]:
    """Remove the farthest points from the point cloud center as noise.
    
    Args:
        xyz: Point coordinates (N, 3)
        rgb: Point colors (N, 3)
        filter_ratio: Fraction of farthest points to remove (default 0.1 = 10%)
    
    Returns:
        Filtered xyz, rgb arrays and number of removed points
    """
    if xyz.size == 0 or filter_ratio <= 0:
        return xyz, rgb, 0
    
    # Calculate point cloud center (centroid)
    cloud_center = np.mean(xyz, axis=0)
    
    # Calculate distances from cloud center
    distances = np.linalg.norm(xyz - cloud_center, axis=1)
    
    # Find the threshold for the farthest points to remove
    n_total = len(xyz)
    n_to_remove = max(1, int(n_total * filter_ratio))
    distance_threshold = np.partition(distances, -n_to_remove)[-n_to_remove]
    
    # Keep points that are closer than the threshold
    keep_mask = distances < distance_threshold
    
    # If we have exactly the threshold distance points, keep some of them
    if keep_mask.sum() < n_total - n_to_remove:
        threshold_mask = distances == distance_threshold
        threshold_indices = np.where(threshold_mask)[0]
        n_threshold_to_keep = n_total - n_to_remove - keep_mask.sum()
        if n_threshold_to_keep > 0:
            keep_threshold_indices = threshold_indices[:n_threshold_to_keep]
            keep_mask[keep_threshold_indices] = True
    
    xyz_filtered = xyz[keep_mask]
    rgb_filtered = rgb[keep_mask]
    n_removed = n_total - len(xyz_filtered)
    
    return xyz_filtered, rgb_filtered, n_removed


def colmap_to_ue_transform(xyz: np.ndarray, dirs: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Convert COLMAP coordinates directly to UE coordinates.

    COLMAP uses an arbitrary right-handed world coordinate system determined by reconstruction.
    We assume COLMAP world follows OpenCV convention: +X right, +Y down, +Z forward (right-handed)
    UE uses: +X forward, +Y right, +Z up (left-handed)
    
    Conversion: COLMAP (X=right, Y=down, Z=forward) -> UE (X=forward, Y=right, Z=up)
    Mapping: (Xc, Yc, Zc) -> (Yc, Xc, Zc) and scale from meters to centimeters
    """
    if xyz.size == 0:
        ue_xyz = xyz.reshape(0, 3)
    else:
        # Convert COLMAP -> UE coordinate transformation
        # X_ue = Y_colmap (forward, was down in COLMAP)
        # Y_ue = X_colmap (right, same as COLMAP)
        # Z_ue = Z_colmap (up, was forward in COLMAP)
        ue_xyz = np.column_stack([xyz[:, 1], xyz[:, 0], xyz[:, 2]])
        ue_xyz *= 100.0  # Scale: meters -> centimeters

    ue_dirs = None
    if dirs is not None:
        if dirs.size == 0:
            ue_dirs = dirs.reshape(0, 3)
        else:
            # Same coordinate transformation for directions (no scaling)
            ue_dirs = np.column_stack([dirs[:, 1], dirs[:, 0], dirs[:, 2]])

    return ue_xyz, ue_dirs


def rotation_matrix_to_ue_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to UE quaternion format.
    UE quaternion order: [x, y, z, w] (scalar last)
    COLMAP quaternion order: [w, x, y, z] (scalar first)
    
    Args:
        R: 3x3 rotation matrix (world to camera in UE coordinates)
    
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



# ============================ IO helpers ============================

def read_exact(f, n: int) -> bytes:
    """Read exactly n bytes or raise EOFError."""
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"Unexpected EOF: need {n}, got {len(b)}")
    return b

def file_size(path: str) -> int:
    return os.path.getsize(path)

def remaining_bytes(f) -> int:
    cur = f.tell()
    f.seek(0, os.SEEK_END)
    end = f.tell()
    f.seek(cur, os.SEEK_SET)
    return end - cur

def _read_c_string(f) -> str:
    """Read a null-terminated C string (UTF-8)."""
    bs = bytearray()
    while True:
        c = f.read(1)
        if not c or c == b'\x00':
            break
        bs += c
    return bs.decode('utf-8', errors='ignore')


# ============================ TEXT parsers ============================

def parse_images_txt(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"images.txt not found at: {path}")

    cams = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = [ln.strip() for ln in f.readlines()]
    i = 0
    while i < len(raw):
        line = raw[i]
        if (not line) or line.startswith('#'):
            i += 1
            continue
        parts = line.split()
        if len(parts) < 9:
            i += 1
            continue
        try:
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            cam_id = int(parts[8])
            name = " ".join(parts[9:]) if len(parts) > 9 else ""
        except Exception:
            i += 1
            continue

        q = np.array([qw, qx, qy, qz], dtype=float)
        t = np.array([tx, ty, tz], dtype=float)
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(t))):
            i += 2
            continue

        R = qvec2rotmat(q)      # world -> cam
        Rt = R.T                # cam -> world
        C = -Rt @ t
        fwd = Rt @ np.array([0.0, 0.0, 1.0])
        nrm = np.linalg.norm(fwd)
        if not np.isfinite(nrm) or nrm < 1e-12:
            i += 2
            continue
        fwd /= nrm

        cams.append({"id": image_id, "center": C, "dir": fwd, "name": name, "cam_id": cam_id, "R_w2c": R})

        # skip correspondences line
        i += 2
    return cams

def parse_points3D_txt(path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"points3D.txt not found at: {path}")

    xyz, rgb = [], []
    skipped = 0
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                skipped += 1
                continue
            try:
                x, y, z = map(float, parts[1:4])
                r, g, b = map(int, parts[4:7])
            except Exception:
                skipped += 1
                continue
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                xyz.append([x, y, z])
                rgb.append([r, g, b])
            else:
                skipped += 1

    xyz = np.array(xyz, dtype=float) if xyz else np.zeros((0,3), dtype=float)
    rgb = np.array(rgb, dtype=np.uint8) if rgb else np.zeros((0,3), dtype=np.uint8)
    return xyz, rgb, skipped


# ============================ BIN parsers (robust) ============================

def parse_images_bin(path: str):
    cams = []
    with open(path, 'rb') as f:
        # 先读数量头
        hdr = f.read(8)
        if len(hdr) != 8:
            return cams
        num_imgs = struct.unpack('<Q', hdr)[0]

        for _ in range(int(num_imgs)):
            image_id = struct.unpack('<I', f.read(4))[0]
            q = struct.unpack('<4d', f.read(32))
            t = struct.unpack('<3d', f.read(24))
            cam_id = struct.unpack('<I', f.read(4))[0]

            # 读以 \0 结尾的名字
            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if not c or c == b'\x00':
                    break
                name_bytes += c
            name = name_bytes.decode('utf-8', errors='ignore')

            n_pts = struct.unpack('<Q', f.read(8))[0]
            # 跳过 2D 点 (24 bytes each)
            if n_pts:
                f.seek(int(24 * n_pts), os.SEEK_CUR)

            q = np.array(q, dtype=float)
            t = np.array(t, dtype=float)
            if not (np.all(np.isfinite(q)) and np.all(np.isfinite(t))):
                continue

            R = qvec2rotmat(q)    # world -> cam
            Rt = R.T
            C = -Rt @ t
            fwd = Rt @ np.array([0.0, 0.0, 1.0])
            nrm = np.linalg.norm(fwd)
            if not np.isfinite(nrm) or nrm < 1e-12:
                continue
            fwd /= nrm

            cams.append({
                "id": int(image_id),
                "center": C,
                "dir": fwd,
                "name": name,
                "cam_id": int(cam_id),
                "R_w2c": R,
            })
    return cams


def parse_points3D_bin(path: str):
    xyz, rgb = [], []
    skipped = 0
    with open(path, 'rb') as f:
        # 先读数量头
        hdr = f.read(8)
        if len(hdr) != 8:
            return np.zeros((0,3)), np.zeros((0,3), np.uint8), 0
        num_pts = struct.unpack('<Q', hdr)[0]

        for _ in range(int(num_pts)):
            _ptid = struct.unpack('<Q', f.read(8))[0]
            try:
                X, Y, Z = struct.unpack('<3d', f.read(24))
                r, g, b = struct.unpack('<3B', f.read(3))
                _err = struct.unpack('<d', f.read(8))[0]
                track_len = struct.unpack('<Q', f.read(8))[0]
                if track_len:
                    f.seek(int(8 * track_len), os.SEEK_CUR)  # (uint32,uint32)×N

                if np.isfinite(X) and np.isfinite(Y) and np.isfinite(Z):
                    xyz.append([X, Y, Z])
                    rgb.append([r, g, b])
                else:
                    skipped += 1
            except Exception:
                skipped += 1
                break

    xyz = np.asarray(xyz, dtype=float) if xyz else np.zeros((0,3), dtype=float)
    rgb = np.asarray(rgb, dtype=np.uint8) if rgb else np.zeros((0,3), dtype=np.uint8)
    return xyz, rgb, skipped


# ============================ PLY writer ============================

def write_ply(path: str,
              xyz: np.ndarray,
              rgb: np.ndarray = None,
              normals: np.ndarray = None):
    n = int(xyz.shape[0])
    has_rgb = rgb is not None and rgb.shape[0] == n
    has_n = normals is not None and normals.shape[0] == n
    with open(path, 'w', encoding='utf-8') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_n:
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
        if has_rgb:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            row = [f"{xyz[i,0]:.6f}", f"{xyz[i,1]:.6f}", f"{xyz[i,2]:.6f}"]
            if has_n:
                row += [f"{normals[i,0]:.6f}", f"{normals[i,1]:.6f}", f"{normals[i,2]:.6f}"]
            if has_rgb:
                row += [str(int(rgb[i,0])), str(int(rgb[i,1])), str(int(rgb[i,2]))]
            f.write(" ".join(row) + "\n")


# ============================ Utilities ============================

def validate_cameras_txt(path: str):
    if not os.path.isfile(path):
        print(f"[warn] cameras.txt not found at: {path} (this is fine)")

def bin_exists(colmap_dir: str) -> bool:
    return (os.path.isfile(os.path.join(colmap_dir, "images.bin")) and
            os.path.isfile(os.path.join(colmap_dir, "points3D.bin")))

def txt_exists(colmap_dir: str) -> bool:
    return (os.path.isfile(os.path.join(colmap_dir, "images.txt")) and
            os.path.isfile(os.path.join(colmap_dir, "points3D.txt")))


# ============================ Main ============================

def main():
    ap = argparse.ArgumentParser(description="Convert COLMAP (TXT or BIN) model to PLY point clouds and UE poses.")
    ap.add_argument('--colmap_dir', type=str,
                    default=r'D:\Data\360_v2\garden\mesh\Colmap',
                    help='Directory containing COLMAP sparse reconstruction files')
    ap.add_argument('--out_dir', type=str,
                    default=r'C:\UEProjects\VCCSimDev\Saved',
                    help='Output directory for PLY files and UE poses')
    ap.add_argument('--camera_color', type=int, nargs=3, default=[255, 0, 0],
                    help='RGB color for COLMAP camera points (default: red)')
    ap.add_argument('--camera_ue_color', type=int, nargs=3, default=[0, 255, 0],
                    help='RGB color for UE camera points (default: green)')
    ap.add_argument('--sanitize', action='store_true',
                    help='Remove NaN/Inf vertices before writing PLY files')
    ap.add_argument('--no_filter_outliers', action='store_true',
                    help='Disable outlier filtering (by default, farthest 5%% of points are removed as noise)')
    ap.add_argument('--filter_ratio', type=float, default=0.05,
                    help='Fraction of farthest points to remove (default: 0.05 = 5%%)')
    ap.add_argument('--verbose', action='store_true',
                    help='Enable verbose output with conversion details')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    src_dir = args.colmap_dir
    use_txt = txt_exists(src_dir)
    use_bin = bin_exists(src_dir)

    images_txt = os.path.join(src_dir, 'images.txt')
    points3D_txt = os.path.join(src_dir, 'points3D.txt')
    cameras_txt = os.path.join(src_dir, 'cameras.txt')

    images_bin = os.path.join(src_dir, 'images.bin')
    points3D_bin = os.path.join(src_dir, 'points3D.bin')

    cameras = []
    xyz_pts = np.zeros((0,3), dtype=float)
    rgb_pts = np.zeros((0,3), dtype=np.uint8)

    parsed_ok = False

    try:
        if use_txt:
            if args.verbose:
                print(f"[info] Parsing TEXT model at {src_dir}")
            validate_cameras_txt(cameras_txt)
            cameras = parse_images_txt(images_txt)
            xyz_pts, rgb_pts, n_bad = parse_points3D_txt(points3D_txt)
            if args.verbose:
                print(f"[info] Text model: cameras={len(cameras)}, points={xyz_pts.shape[0]} (skipped={n_bad})")
            parsed_ok = True
        elif use_bin:
            if args.verbose:
                print(f"[info] Parsing BINARY model at {src_dir}")
            cameras = parse_images_bin(images_bin)
            xyz_pts, rgb_pts, n_bad = parse_points3D_bin(points3D_bin)
            if args.verbose:
                print(f"[info] Binary model: cameras={len(cameras)}, points={xyz_pts.shape[0]} (skipped={n_bad})")
            parsed_ok = True
        else:
            print(f"[warn] No COLMAP model files found at: {src_dir}")
    except Exception as e:
        print(f"[warn] Parse failed: {e}")

    if not parsed_ok:
        if not use_txt and not use_bin:
            raise RuntimeError(f"No COLMAP model found at {src_dir}. Need either TXT files (images.txt, points3D.txt) or BIN files (images.bin, points3D.bin).")
        else:
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

    # Optional outlier filtering - remove farthest points as noise (enabled by default)
    if not args.no_filter_outliers and len(xyz_pts) > 0:
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
    cams_out = os.path.join(args.out_dir, 'cameras_colmap.ply')
    pts_out = os.path.join(args.out_dir, 'points_colmap.ply')
    cams_ue_out = os.path.join(args.out_dir, 'cameras_ue.ply')
    pts_ue_out = os.path.join(args.out_dir, 'points_ue.ply')
    pose_ue_out = os.path.join(args.out_dir, 'pose_ue.txt')

    if args.verbose:
        print("[info] Writing output PLY files...")
    write_ply(cams_out, cam_xyz, rgb=cam_rgb, normals=cam_dirs)
    write_ply(pts_out, xyz_pts, rgb=rgb_pts, normals=None)
    write_ply(cams_ue_out, cam_ue_xyz, rgb=cam_ue_rgb, normals=cam_ue_dirs)
    write_ply(pts_ue_out, pts_ue_xyz, rgb=pts_ue_rgb, normals=None)

    # Write UE pose file with quaternions
    if args.verbose:
        print(f"[info] Writing UE pose file: {pose_ue_out}")
    with open(pose_ue_out, 'w', encoding='utf-8') as f:
        f.write("# UE coordinate system poses (left-handed, cm)\n")
        f.write("# UE coordinate axes: +X forward, +Y right, +Z up\n")
        f.write("# Format: X Y Z Qx Qy Qz Qw\n")
        f.write("# Quaternion order: [x, y, z, w] (UE format, scalar last)\n")
        for i, cam in enumerate(cameras):
            # Get UE position
            ue_pos = cam_ue_xyz[i]
            
            R_w2c_colmap = cam["R_w2c"]
            
            R_c2w_colmap = R_w2c_colmap.T
            
            T_colmap_to_ue = np.array([
                [0,  1,  0],  # X_ue = Y_colmap  
                [1,  0,  0],  # Y_ue = X_colmap
                [0,  0,  1]   # Z_ue = Z_colmap
            ])
            
            # Apply transformation: R_ue = T * R_colmap * T^T
            R_c2w_ue = T_colmap_to_ue @ R_c2w_colmap @ T_colmap_to_ue.T
            
            # Extract UE quaternion [x, y, z, w] from camera-to-world rotation
            ue_quat = rotation_matrix_to_ue_quaternion(R_c2w_ue)
            
            f.write(f"{ue_pos[0]:.6f} {ue_pos[1]:.6f} {ue_pos[2]:.6f} {ue_quat[0]:.6f} {ue_quat[1]:.6f} {ue_quat[2]:.6f} {ue_quat[3]:.6f}\n")

    if args.verbose:
        print("[done] Outputs:")
        print(f"  COLMAP cameras: {cams_out}\n  COLMAP points:  {pts_out}\n  UE cameras:     {cams_ue_out}\n  UE points:      {pts_ue_out}\n  UE poses:       {pose_ue_out}")
        print("[note] COLMAP->UE: Assumes COLMAP world follows OpenCV convention (+X right, +Y down, +Z forward).")
        print("      UE conversion: (X,Y,Z) -> (Y,X,Z) to get left-handed (+X forward, +Y right, +Z up).")
    else:
        print("[done] Conversion complete. Use --verbose for details.")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(f"[error] {ex}", file=sys.stderr)
        sys.exit(1)
