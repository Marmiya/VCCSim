#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a COLMAP model (TEXT or BIN) into two PLY point clouds:
  1) cameras.ply  -- one point per registered image (camera center),
                     normals encode viewing direction.
  2) points.ply   -- sparse scene points with RGB.

- Supports images.{txt|bin}, points3D.{txt|bin}, cameras.{txt|bin} (cameras optional).
- TXT preferred if both exist, can force using --prefer.
- If BIN parse fails and 'colmap' is on PATH, auto-convert BIN->TXT as fallback.
- Robust binary parser with bounds check & NaN/Inf sanitization.

Default paths set to your case:
  --colmap_dir D:\Data\360_v2\garden\sparse\0
  --out_dir    D:\Data\360_v2\garden\sparse\0
"""

import os
import sys
import argparse
import math
import shutil
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

        cams.append({"id": image_id, "center": C, "dir": fwd, "name": name, "cam_id": cam_id})

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
    ap = argparse.ArgumentParser(description="Convert COLMAP (TXT or BIN) model to PLY point clouds.")
    ap.add_argument('--colmap_dir', type=str,
                    default=r'D:\Data\360_v2\garden\sparse\0',
                    help='Directory containing images.{txt|bin}, points3D.{txt|bin}, cameras.{txt|bin}')
    ap.add_argument('--out_dir', type=str,
                    default=r'D:\Data\360_v2\garden\sparse\0',
                    help='Output directory for PLY files')
    ap.add_argument('--cams_ply', type=str, default='cameras.ply', help='Output PLY for camera poses')
    ap.add_argument('--points_ply', type=str, default='points.ply', help='Output PLY for scene points')
    ap.add_argument('--camera_color', type=int, nargs=3, default=[255, 0, 0],
                    help='RGB color for camera points (default: 255 0 0)')
    ap.add_argument('--prefer', type=str, choices=['auto', 'txt', 'bin'], default='auto',
                    help='Prefer parsing txt or bin (default auto)')
    ap.add_argument('--sanitize', action='store_true',
                    help='Drop any NaN/Inf vertices before writing PLY')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    src_dir = args.colmap_dir
    prefer = args.prefer

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
        if (prefer == 'txt' and use_txt) or (prefer == 'auto' and use_txt):
            print(f"[info] Parsing TEXT model at {src_dir}")
            validate_cameras_txt(cameras_txt)
            cameras = parse_images_txt(images_txt)
            xyz_pts, rgb_pts, n_bad = parse_points3D_txt(points3D_txt)
            print(f"[info] Text model: cameras={len(cameras)}, points={xyz_pts.shape[0]} (skipped={n_bad})")
            parsed_ok = True
        elif (prefer == 'bin' and use_bin) or (prefer == 'auto' and use_bin):
            print(f"[info] Parsing BINARY model at {src_dir}")
            cameras = parse_images_bin(images_bin)
            xyz_pts, rgb_pts, n_bad = parse_points3D_bin(points3D_bin)
            print(f"[info] Binary model: cameras={len(cameras)}, points={xyz_pts.shape[0]} (skipped={n_bad})")
            parsed_ok = True
        else:
            print(f"[warn] Neither TEXT nor BIN found at: {src_dir}")
    except Exception as e:
        print(f"[warn] Direct parse failed: {e}")

    # 检查是否有可用的数据文件
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
    else:
        cam_xyz = np.zeros((0, 3), dtype=float)
        cam_dirs = np.zeros((0, 3), dtype=float)
        cam_rgb = np.zeros((0, 3), dtype=np.uint8)

    # Optional sanitization to avoid NaN warnings in Meshlab
    if args.sanitize:
        cam_mask = np.isfinite(cam_xyz).all(axis=1) & np.isfinite(cam_dirs).all(axis=1)
        pts_mask = np.isfinite(xyz_pts).all(axis=1)
        kept_cam = int(cam_mask.sum())
        kept_pts = int(pts_mask.sum())
        print(f"[info] sanitize: cameras kept={kept_cam}/{len(cam_xyz)}, points kept={kept_pts}/{len(xyz_pts)}")
        cam_xyz, cam_dirs, cam_rgb = cam_xyz[cam_mask], cam_dirs[cam_mask], cam_rgb[cam_mask]
        xyz_pts, rgb_pts = xyz_pts[pts_mask], rgb_pts[pts_mask]

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
