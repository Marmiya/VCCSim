#!/usr/bin/env python3
"""
COLMAP Utilities Library
========================

Common utilities for COLMAP data processing including:
- File I/O operations for cameras, images, and points3D
- Coordinate system transformations
- Geometric operations (quaternions, rotations)
- Mathematical utilities (statistics, error computation)
- File format conversion utilities

Author: VCCSim Project
"""

import os
import sys
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple, Union


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ColmapCamera(NamedTuple):
    """COLMAP camera representation"""
    camera_id: int
    model: str
    width: int
    height: int
    params: List[float]


class ColmapImage(NamedTuple):
    """COLMAP image representation with pose"""
    image_id: int
    qvec: np.ndarray  # [qw, qx, qy, qz]
    tvec: np.ndarray  # [tx, ty, tz]
    camera_id: int
    name: str
    points2d: List[Tuple[float, float, int]]  # (x, y, point3d_id)


class ColmapPoint3D(NamedTuple):
    """COLMAP 3D point representation"""
    point3d_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    track: List[Tuple[int, int]]  # [(image_id, point2d_idx), ...]


# COLMAP camera model definitions
CAMERA_MODELS = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE"
}

class CameraModel:
    """Camera model definition with parameter count"""
    def __init__(self, model_id: int, model_name: str, num_params: int):
        self.model_id = model_id
        self.model_name = model_name
        self.num_params = num_params

CAMERA_MODEL_IDS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
    5: CameraModel(5, "OPENCV_FISHEYE", 8),
    6: CameraModel(6, "FULL_OPENCV", 12),
    7: CameraModel(7, "FOV", 5),
    8: CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    9: CameraModel(9, "RADIAL_FISHEYE", 5),
    10: CameraModel(10, "THIN_PRISM_FISHEYE", 12)
}


# =============================================================================
# GEOMETRIC OPERATIONS
# =============================================================================

def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix.

    Args:
        qvec: Quaternion as numpy array [qw, qx, qy, qz]

    Returns:
        3x3 rotation matrix
    """
    if qvec.shape != (4,):
        qvec = np.asarray(qvec).reshape(4,)

    qw, qx, qy, qz = qvec.astype(float)
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)

    if not np.isfinite(n) or n == 0.0:
        raise ValueError("Invalid quaternion norm")

    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n

    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ], dtype=float)

    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion [qw, qx, qy, qz].

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion as numpy array [qw, qx, qy, qz]
    """
    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return np.array([qw, qx, qy, qz])


def calculate_camera_center(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Calculate camera center in world coordinates from COLMAP pose.
    COLMAP convention: x_cam = R * x_world + t
    Camera center: C = -R^T * t

    Args:
        qvec: Quaternion [qw, qx, qy, qz]
        tvec: Translation vector [tx, ty, tz]

    Returns:
        Camera center in world coordinates
    """
    R = qvec2rotmat(qvec)
    return -R.T @ tvec


def colmap_to_ue_transform(xyz: np.ndarray, dirs: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Transform COLMAP coordinates to UE coordinates.
    COLMAP: Right-handed, Y-down, Z-forward (meters)
    UE: Left-handed, Y-right, Z-up (centimeters)

    Args:
        xyz: Points in COLMAP coordinates (N, 3)
        dirs: Optional directions in COLMAP coordinates (N, 3)

    Returns:
        Tuple of (transformed_xyz, transformed_dirs)
    """
    if xyz.size == 0:
        ue_xyz = xyz.reshape(0, 3)
    else:
        # Coordinate transformation: COLMAP(X,Y,Z) -> UE(Y,X,Z)
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


# =============================================================================
# FILE I/O UTILITIES
# =============================================================================

def read_exact(f, n: int) -> bytes:
    """Read exactly n bytes or raise EOFError."""
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"Unexpected EOF: need {n}, got {len(b)}")
    return b


def read_next_bytes(fid, num_bytes: int, format_char_sequence: str, endian_character: str = "<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


# =============================================================================
# TEXT FILE PARSERS
# =============================================================================

def read_cameras_txt(file_path: str) -> Dict[int, ColmapCamera]:
    """
    Read cameras.txt file.

    Args:
        file_path: Path to cameras.txt file

    Returns:
        Dictionary mapping camera_id to ColmapCamera
    """
    cameras = {}

    if not os.path.exists(file_path):
        return cameras

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]

            cameras[camera_id] = ColmapCamera(camera_id, model, width, height, params)

    return cameras


def read_images_txt(file_path: str) -> Dict[int, ColmapImage]:
    """
    Read images.txt file.

    Args:
        file_path: Path to images.txt file

    Returns:
        Dictionary mapping image_id to ColmapImage
    """
    images = {}

    if not os.path.exists(file_path):
        return images

    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue

        # Parse image line
        parts = line.split()
        if len(parts) < 9:
            i += 1
            continue

        try:
            image_id = int(parts[0])
            qvec = np.array([float(parts[j]) for j in range(1, 5)])  # [qw, qx, qy, qz]
            tvec = np.array([float(parts[j]) for j in range(5, 8)])  # [tx, ty, tz]
            camera_id = int(parts[8])
            name = " ".join(parts[9:]) if len(parts) > 9 else ""
        except (ValueError, IndexError):
            i += 2
            continue

        # Validate pose
        if not (np.all(np.isfinite(qvec)) and np.all(np.isfinite(tvec))):
            i += 2
            continue

        # Parse points2D line (next line)
        points2d = []
        if i + 1 < len(lines):
            points_line = lines[i + 1].strip()
            if points_line and not points_line.startswith('#'):
                points_parts = points_line.split()
                for j in range(0, len(points_parts), 3):
                    if j + 2 < len(points_parts):
                        try:
                            x = float(points_parts[j])
                            y = float(points_parts[j + 1])
                            point3d_id = int(points_parts[j + 2])
                            points2d.append((x, y, point3d_id))
                        except (ValueError, IndexError):
                            continue

        images[image_id] = ColmapImage(image_id, qvec, tvec, camera_id, name, points2d)
        i += 2  # Skip to next image

    return images


def read_points3d_txt(file_path: str) -> Dict[int, ColmapPoint3D]:
    """
    Read points3D.txt file.

    Args:
        file_path: Path to points3D.txt file

    Returns:
        Dictionary mapping point3d_id to ColmapPoint3D
    """
    points3d = {}

    if not os.path.exists(file_path):
        return points3d

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            try:
                point3d_id = int(parts[0])
                xyz = np.array([float(parts[j]) for j in range(1, 4)])
                rgb = np.array([int(parts[j]) for j in range(4, 7)])
                error = float(parts[7])

                # Parse track (image_id, point2d_idx pairs)
                track = []
                for j in range(8, len(parts), 2):
                    if j + 1 < len(parts):
                        image_id = int(parts[j])
                        point2d_idx = int(parts[j + 1])
                        track.append((image_id, point2d_idx))

                points3d[point3d_id] = ColmapPoint3D(point3d_id, xyz, rgb, error, track)

            except (ValueError, IndexError):
                continue

    return points3d


# =============================================================================
# BINARY FILE PARSERS
# =============================================================================

def read_cameras_binary(file_path: str) -> Dict[int, ColmapCamera]:
    """
    Read cameras.bin file.

    Args:
        file_path: Path to cameras.bin file

    Returns:
        Dictionary mapping camera_id to ColmapCamera
    """
    cameras = {}

    if not os.path.exists(file_path):
        return cameras

    with open(file_path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = ColmapCamera(
                camera_id, CAMERA_MODELS[model_id], width, height, list(params)
            )

    return cameras


def read_images_binary(file_path: str) -> Dict[int, ColmapImage]:
    """
    Read images.bin file.

    Args:
        file_path: Path to images.bin file

    Returns:
        Dictionary mapping image_id to ColmapImage
    """
    images = {}

    if not os.path.exists(file_path):
        return images

    with open(file_path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]

            # Read image name
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]

            # Read 2D points
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            points2d = []
            for _ in range(num_points2D):
                point_data = read_next_bytes(fid, 24, "ddq")
                x, y, point3d_id = point_data
                points2d.append((x, y, point3d_id))

            images[image_id] = ColmapImage(image_id, qvec, tvec, camera_id, image_name, points2d)

    return images


def read_points3d_binary(file_path: str) -> Dict[int, ColmapPoint3D]:
    """
    Read points3D.bin file.

    Args:
        file_path: Path to points3D.bin file

    Returns:
        Dictionary mapping point3d_id to ColmapPoint3D
    """
    points3d = {}

    if not os.path.exists(file_path):
        return points3d

    with open(file_path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            try:
                binary_point_line_properties = read_next_bytes(fid, 43, "QdddBBBd")
                point3d_id = binary_point_line_properties[0]
                xyz = np.array(binary_point_line_properties[1:4])
                rgb = np.array(binary_point_line_properties[4:7])
                error = binary_point_line_properties[7]

                # Read track
                track_length = read_next_bytes(fid, 8, "Q")[0]
                track = []
                for _ in range(track_length):
                    track_elem = read_next_bytes(fid, 8, "ii")
                    image_id, point2d_idx = track_elem
                    track.append((image_id, point2d_idx))

                points3d[point3d_id] = ColmapPoint3D(point3d_id, xyz, rgb, error, track)

            except (struct.error, EOFError):
                break

    return points3d


# =============================================================================
# TEXT FILE WRITERS
# =============================================================================

def write_cameras_txt(file_path: str, cameras: Dict[int, ColmapCamera]):
    """
    Write cameras.txt file.

    Args:
        file_path: Output file path
        cameras: Dictionary of cameras to write
    """
    with open(file_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")

        for camera in cameras.values():
            params_str = ' '.join(map(str, camera.params))
            f.write(f"{camera.camera_id} {camera.model} {camera.width} {camera.height} {params_str}\n")


def write_images_txt(file_path: str, images: Dict[int, ColmapImage], max_points2d: Optional[int] = None):
    """
    Write images.txt file.

    Args:
        file_path: Output file path
        images: Dictionary of images to write
        max_points2d: Optional limit on points2D per image
    """
    with open(file_path, 'w') as f:
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
            f.write(f"{image.image_id} {image.qvec[0]} {image.qvec[1]} {image.qvec[2]} {image.qvec[3]} ")
            f.write(f"{image.tvec[0]} {image.tvec[1]} {image.tvec[2]} {image.camera_id} {image.name}\n")

            # Limit points2d if specified
            points2d_to_write = image.points2d
            if max_points2d is not None and len(points2d_to_write) > max_points2d:
                points2d_to_write = points2d_to_write[:max_points2d]

            if points2d_to_write:
                points_str = ' '.join([f"{x} {y} {pid}" for x, y, pid in points2d_to_write])
                f.write(f"{points_str}\n")
            else:
                f.write("\n")


def write_points3d_txt(file_path: str, points3d: Dict[int, ColmapPoint3D]):
    """
    Write points3D.txt file.

    Args:
        file_path: Output file path
        points3d: Dictionary of 3D points to write
    """
    with open(file_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points3d)}, mean track length: {np.mean([len(pt.track) for pt in points3d.values()]) if points3d else 0:.2f}\n")

        for point in points3d.values():
            xyz_str = ' '.join(map(str, point.xyz))
            rgb_str = ' '.join(map(str, point.rgb.astype(int)))

            track_str = ""
            for image_id, point2d_idx in point.track:
                track_str += f"{image_id} {point2d_idx} "

            f.write(f"{point.point3d_id} {xyz_str} {rgb_str} {point.error} {track_str.rstrip()}\n")


# =============================================================================
# PLY FILE UTILITIES
# =============================================================================

def write_ply(file_path: str, xyz: np.ndarray, rgb: np.ndarray = None, normals: np.ndarray = None):
    """
    Write PLY file with points, optional colors and normals.

    Args:
        file_path: Output PLY file path
        xyz: Point coordinates (N, 3)
        rgb: Optional point colors (N, 3)
        normals: Optional point normals (N, 3)
    """
    n = int(xyz.shape[0])
    has_rgb = rgb is not None and rgb.shape[0] == n
    has_normals = normals is not None and normals.shape[0] == n

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_normals:
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
        if has_rgb:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        for i in range(n):
            row = [f"{xyz[i,0]:.6f}", f"{xyz[i,1]:.6f}", f"{xyz[i,2]:.6f}"]
            if has_normals:
                row += [f"{normals[i,0]:.6f}", f"{normals[i,1]:.6f}", f"{normals[i,2]:.6f}"]
            if has_rgb:
                row += [str(int(rgb[i,0])), str(int(rgb[i,1])), str(int(rgb[i,2]))]
            f.write(" ".join(row) + "\n")


# =============================================================================
# FILE FORMAT UTILITIES
# =============================================================================

def detect_colmap_format(model_dir: str) -> Tuple[bool, bool, str]:
    """
    Detect COLMAP model format and find data directory.

    Args:
        model_dir: COLMAP model directory

    Returns:
        Tuple of (has_txt, has_bin, data_dir)
    """
    model_path = Path(model_dir)

    # Check for sparse/0 subdirectory structure
    sparse_dir = model_path / "sparse" / "0"
    if sparse_dir.exists():
        data_dir = str(sparse_dir)
    else:
        data_dir = str(model_path)

    # Check for text format
    has_txt = (
        os.path.exists(os.path.join(data_dir, "cameras.txt")) and
        os.path.exists(os.path.join(data_dir, "images.txt"))
    )

    # Check for binary format
    has_bin = (
        os.path.exists(os.path.join(data_dir, "cameras.bin")) and
        os.path.exists(os.path.join(data_dir, "images.bin"))
    )

    return has_txt, has_bin, data_dir


def ensure_txt_format(model_dir: str) -> str:
    """
    Ensure COLMAP model is in TXT format, convert if necessary.

    Args:
        model_dir: COLMAP model directory

    Returns:
        Path to directory containing TXT files
    """
    has_txt, has_bin, data_dir = detect_colmap_format(model_dir)

    if has_txt:
        return data_dir

    if has_bin:
        print(f"Converting {model_dir} from binary to TXT format...")
        txt_dir = model_dir + "_txt"
        os.makedirs(txt_dir, exist_ok=True)

        cmd = f'colmap model_converter --input_path "{data_dir}" --output_path "{txt_dir}" --output_type TXT'
        result = os.system(cmd)

        if result == 0:
            return txt_dir
        else:
            raise RuntimeError(f"Failed to convert COLMAP model to TXT format: {model_dir}")

    raise FileNotFoundError(f"No valid COLMAP model found in {model_dir}")


# =============================================================================
# MATHEMATICAL UTILITIES
# =============================================================================

def filter_outlier_points(xyz: np.ndarray, rgb: np.ndarray, filter_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Remove the farthest points from point cloud center as noise.

    Args:
        xyz: Point coordinates (N, 3)
        rgb: Point colors (N, 3)
        filter_ratio: Fraction of farthest points to remove

    Returns:
        Tuple of (filtered_xyz, filtered_rgb, num_removed)
    """
    if xyz.size == 0 or filter_ratio <= 0:
        return xyz, rgb, 0

    # Calculate point cloud center
    cloud_center = np.mean(xyz, axis=0)

    # Calculate distances from center
    distances = np.linalg.norm(xyz - cloud_center, axis=1)

    # Find threshold for farthest points to remove
    n_total = len(xyz)
    n_to_remove = max(1, int(n_total * filter_ratio))
    distance_threshold = np.partition(distances, -n_to_remove)[-n_to_remove]

    # Keep points closer than threshold
    keep_mask = distances < distance_threshold

    # Handle points exactly at threshold
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


def filter_low_height_points(xyz: np.ndarray, rgb: np.ndarray, min_height: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Remove points below a certain height (Z coordinate) threshold.

    Args:
        xyz: Point coordinates (N, 3) in COLMAP coordinate system
        rgb: Point colors (N, 3)
        min_height: Minimum height (Z coordinate) threshold

    Returns:
        Tuple of (filtered_xyz, filtered_rgb, num_removed)
    """
    if xyz.size == 0:
        return xyz, rgb, 0

    # Filter points based on Z coordinate
    height_mask = xyz[:, 2] >= min_height

    xyz_filtered = xyz[height_mask]
    rgb_filtered = rgb[height_mask]
    n_removed = len(xyz) - len(xyz_filtered)

    return xyz_filtered, rgb_filtered, n_removed


def extract_camera_centers(images: Dict[int, ColmapImage], image_ids: List[int]) -> np.ndarray:
    """
    Extract camera centers for given image IDs.

    Args:
        images: Dictionary of COLMAP images
        image_ids: List of image IDs to extract centers for

    Returns:
        Array of camera centers (N, 3)
    """
    centers = []
    for img_id in image_ids:
        if img_id in images:
            center = calculate_camera_center(images[img_id].qvec, images[img_id].tvec)
            centers.append(center)
    return np.array(centers)


def find_common_images(images1: Dict[int, ColmapImage], images2: Dict[int, ColmapImage]) -> Tuple[List[str], List[int], List[int]]:
    """
    Find common images between two COLMAP models by filename.

    Args:
        images1: First COLMAP image dictionary
        images2: Second COLMAP image dictionary

    Returns:
        Tuple of (common_names, ids1, ids2)
    """
    names1 = {img.name: img_id for img_id, img in images1.items()}
    names2 = {img.name: img_id for img_id, img in images2.items()}

    common_names = sorted(list(set(names1.keys()) & set(names2.keys())))
    ids1 = [names1[name] for name in common_names]
    ids2 = [names2[name] for name in common_names]

    return common_names, ids1, ids2


# =============================================================================
# ARGUMENT PARSING UTILITIES
# =============================================================================

def validate_colmap_directory(dir_path: str):
    """Validate that directory contains COLMAP model files."""
    has_txt, has_bin, data_dir = detect_colmap_format(dir_path)
    if not has_txt and not has_bin:
        raise ValueError(f"No valid COLMAP model found in {dir_path}")
    return data_dir