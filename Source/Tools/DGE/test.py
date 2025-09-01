#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a COLMAP text model (images.txt, points3D.txt, cameras.txt)
into two PLY point clouds:
  1) cameras.ply  -- one point per registered image (camera center), with normals
                     encoding the viewing direction.
  2) points.ply   -- the sparse scene point cloud with RGB colors.

Usage:
    python colmap_to_ply.py --colmap_dir /path/to/model --out_dir ./out \
        --cams_ply cameras.ply --points_ply points.ply

Notes:
- Only the text model is supported (images.txt/points3D.txt/cameras.txt).
- cameras.txt is not strictly needed for this conversion but validated if present.
- Viewing direction convention: we assume COLMAP's R,t maps world -> camera, so
  the camera "forward" (optical axis) in world coordinates is R^T * [0,0,1].
  The camera center in world coordinates is C = -R^T * t.
"""

import os
import sys
import argparse
import math
from typing import List, Tuple
import numpy as np


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix.
    Returns a matrix R that maps world -> camera when using COLMAP's convention.
    """
    if qvec.shape != (4,):
        qvec = np.asarray(qvec).reshape(4,)
    qw, qx, qy, qz = qvec.astype(float)
    # Normalize to be safe
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n == 0.0:
        raise ValueError("Zero-length quaternion encountered.")
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n

    # Quaternion to rotation matrix (right-handed)
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ], dtype=float)
    return R


def parse_images_txt(path: str):
    """
    Parse images.txt from COLMAP text model.
    Returns a list of dicts with keys: id, center (3,), dir (3,), name, cam_id
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"images.txt not found at: {path}")

    cameras = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = [ln.strip() for ln in f.readlines()]
    i = 0
    while i < len(raw):
        line = raw[i]
        if len(line) == 0 or line.startswith('#'):
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
        R = qvec2rotmat(q)        # world -> camera
        Rt = R.T                  # camera -> world
        C = -Rt @ t               # camera center in world
        forward = Rt @ np.array([0.0, 0.0, 1.0])  # viewing direction (+Z in cam)
        forward_norm = forward / (np.linalg.norm(forward) + 1e-12)

        cameras.append({
            "id": image_id,
            "center": C,
            "dir": forward_norm,
            "name": name,
            "cam_id": cam_id
        })

        # Skip the next line (2D-3D correspondences)
        i += 1
        if i < len(raw):
            i += 1

    return cameras


def parse_points3D_txt(path: str):
    """
    Parse points3D.txt from COLMAP text model.
    Returns (xyz: Nx3 float array, rgb: Nx3 uint8 array, count_bad:int)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"points3D.txt not found at: {path}")

    xyz = []
    rgb = []
    count_bad = 0
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                count_bad += 1
                continue
            try:
                # Format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, [TRACK...]
                x, y, z = map(float, parts[1:4])
                r, g, b = map(int, parts[4:7])
                xyz.append([x, y, z])
                rgb.append([r, g, b])
            except Exception:
                count_bad += 1
                continue

    xyz = np.array(xyz, dtype=float) if xyz else np.zeros((0, 3), dtype=float)
    rgb = np.array(rgb, dtype=np.uint8) if rgb else np.zeros((0, 3), dtype=np.uint8)
    return xyz, rgb, count_bad


def write_ply(path: str,
              xyz: np.ndarray,
              rgb: np.ndarray = None,
              normals: np.ndarray = None):
    """
    Write an ASCII PLY file with vertex positions (and optional normals/RGB).
    """
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


def validate_cameras_txt(path: str):
    """
    Optionally warn if cameras.txt is missing; not needed for conversion.
    """
    if not os.path.isfile(path):
        print(f"[warn] cameras.txt not found at: {path} (this is fine for this script)")


def main():
    ap = argparse.ArgumentParser(description="Convert COLMAP text model to PLY point clouds.")
    ap.add_argument('--colmap_dir', type=str, default='C:\\UEProjects\\VCCSimDev\\Saved\\TriangleSplatting\\colmap_output\\colmap_dataset_20250829_174920\\sparse_txt', help='Directory containing images.txt, points3D.txt, cameras.txt')
    ap.add_argument('--out_dir', type=str, default='C:\\UEProjects\\VCCSimDev\\Saved\\TriangleSplatting\\colmap_output\\colmap_dataset_20250829_174920\\sparse_txt', help='Output directory for PLY files')
    ap.add_argument('--cams_ply', type=str, default='cameras.ply', help='Output PLY for camera poses')
    ap.add_argument('--points_ply', type=str, default='points.ply', help='Output PLY for scene points')
    ap.add_argument('--camera_color', type=int, nargs=3, default=[255, 0, 0],
                    help='RGB color for camera points (default: 255 0 0)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    images_txt = os.path.join(args.colmap_dir, 'images.txt')
    points3D_txt = os.path.join(args.colmap_dir, 'points3D.txt')
    cameras_txt = os.path.join(args.colmap_dir, 'cameras.txt')

    validate_cameras_txt(cameras_txt)

    print(f"[info] Reading images: {images_txt}")
    cameras = parse_images_txt(images_txt)
    print(f"[info] Found {len(cameras)} registered cameras")

    print(f"[info] Reading points3D: {points3D_txt}")
    xyz_pts, rgb_pts, n_bad = parse_points3D_txt(points3D_txt)
    print(f"[info] Loaded {xyz_pts.shape[0]} 3D points ({n_bad} lines skipped)")

    # Prepare cameras data
    if len(cameras) > 0:
        cam_xyz = np.stack([c["center"] for c in cameras], axis=0)
        cam_dirs = np.stack([c["dir"] for c in cameras], axis=0)
        cam_rgb = np.tile(np.array(args.camera_color, dtype=np.uint8), (len(cameras), 1))
    else:
        cam_xyz = np.zeros((0, 3), dtype=float)
        cam_dirs = np.zeros((0, 3), dtype=float)
        cam_rgb = np.zeros((0, 3), dtype=np.uint8)

    # Write PLYs
    cams_out = os.path.join(args.out_dir, args.cams_ply)
    pts_out = os.path.join(args.out_dir, args.points_ply)

    print(f"[info] Writing cameras PLY: {cams_out}")
    write_ply(cams_out, cam_xyz, rgb=cam_rgb, normals=cam_dirs)

    print(f"[info] Writing points PLY: {pts_out}")
    write_ply(pts_out, xyz_pts, rgb=rgb_pts, normals=None)

    print("[done] All files written successfully.")
    print(f"  - Cameras PLY: {cams_out}")
    print(f"  - Points  PLY: {pts_out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(f"[error] {ex}", file=sys.stderr)
        sys.exit(1)
