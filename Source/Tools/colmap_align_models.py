"""
COLMAP Model Alignment Tool
===========================

Aligns two COLMAP reconstructions of the same scene using camera poses.
Uses robust Procrustes analysis with RANSAC for 7-DOF similarity transformation:
- 3D rotation (3 DOF)
- 3D translation (3 DOF)
- Uniform scaling (1 DOF)

Author: VCCSim Project
Based on: Classical Procrustes analysis and RANSAC algorithms
"""

import numpy as np
import os
import random
import shutil
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import itertools
from pathlib import Path
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================

# Source COLMAP model (to be transformed)
SOURCE_MODEL_PATH = r"E:\BaoAn\BaoAnColmap\sparse\0"

# Target COLMAP model (reference coordinate system)
TARGET_MODEL_PATH = r"E:\BaoAn\rc_colmap_refine"

# Output directory for aligned model
OUTPUT_MODEL_PATH = r"E:\BaoAn\aligned_colmap"

# Algorithm parameters
RANSAC_ITERATIONS = 1000  # Number of RANSAC iterations
MIN_INLIERS = 10  # Minimum number of inliers for valid transformation
INLIER_THRESHOLD = 0.1  # Distance threshold for inliers (in target coordinate units)
MIN_CAMERAS_FOR_ALIGNMENT = 6  # Minimum cameras needed for robust alignment

# =============================================================================
# COLMAP DATA STRUCTURES AND I/O
# =============================================================================

class ColmapCamera:
    """COLMAP camera representation"""
    def __init__(self, camera_id: int, model: str, width: int, height: int, params: List[float]):
        self.camera_id = camera_id
        self.model = model
        self.width = width
        self.height = height
        self.params = params

class ColmapImage:
    """COLMAP image representation with pose"""
    def __init__(self, image_id: int, qvec: np.ndarray, tvec: np.ndarray,
                 camera_id: int, name: str, points2d: List):
        self.image_id = image_id
        self.qvec = qvec  # Quaternion [qw, qx, qy, qz]
        self.tvec = tvec  # Translation vector [tx, ty, tz]
        self.camera_id = camera_id
        self.name = name
        self.points2d = points2d

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        return quaternion_to_rotation_matrix(self.qvec)

    @property
    def camera_center(self) -> np.ndarray:
        """Calculate camera center in world coordinates: C = -R^T * t"""
        R = self.rotation_matrix
        return -R.T @ self.tvec

class ColmapPoint3D:
    """COLMAP 3D point representation"""
    def __init__(self, point3d_id: int, xyz: np.ndarray, rgb: np.ndarray,
                 error: float, track: List):
        self.point3d_id = point3d_id
        self.xyz = xyz
        self.rgb = rgb
        self.error = error
        self.track = track

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix"""
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
    ], dtype=np.float64)

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [qw, qx, qy, qz]"""
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

def read_cameras_txt(file_path: str) -> Dict[int, ColmapCamera]:
    """Read cameras.txt file"""
    cameras = {}

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
    """Read images.txt file"""
    images = {}

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
        image_id = int(parts[0])
        qvec = np.array([float(parts[j]) for j in range(1, 5)])  # [qw, qx, qy, qz]
        tvec = np.array([float(parts[j]) for j in range(5, 8)])  # [tx, ty, tz]
        camera_id = int(parts[8])
        name = parts[9]

        # Parse points2D line (skip for now)
        i += 1
        points2d = []
        if i < len(lines):
            points_line = lines[i].strip()
            if points_line and not points_line.startswith('#'):
                # Parse 2D points if needed
                pass

        images[image_id] = ColmapImage(image_id, qvec, tvec, camera_id, name, points2d)
        i += 1

    return images

def read_points3d_txt(file_path: str) -> Dict[int, ColmapPoint3D]:
    """Read points3D.txt file"""
    points3d = {}

    if not os.path.exists(file_path):
        return points3d

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
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

    return points3d

def ensure_txt_format(model_dir: str) -> str:
    """Ensure COLMAP model is in TXT format, convert if necessary"""
    images_txt = os.path.join(model_dir, 'images.txt')

    if os.path.exists(images_txt):
        return model_dir

    # Check for binary format
    images_bin = os.path.join(model_dir, 'images.bin')
    if os.path.exists(images_bin):
        print(f"Converting {model_dir} from binary to TXT format...")
        txt_dir = model_dir + "_txt"
        os.makedirs(txt_dir, exist_ok=True)

        cmd = f'colmap model_converter --input_path "{model_dir}" --output_path "{txt_dir}" --output_type TXT'
        result = os.system(cmd)

        if result == 0:
            return txt_dir
        else:
            raise RuntimeError(f"Failed to convert COLMAP model to TXT format: {model_dir}")

    raise FileNotFoundError(f"No valid COLMAP model found in {model_dir}")

# =============================================================================
# SIMILARITY TRANSFORMATION ALGORITHMS
# =============================================================================

class SimilarityTransformation:
    """7-DOF similarity transformation: R, t, s"""
    def __init__(self, rotation: np.ndarray, translation: np.ndarray, scale: float):
        self.rotation = rotation  # 3x3 rotation matrix
        self.translation = translation  # 3x1 translation vector
        self.scale = scale  # scalar scale factor

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Apply transformation: s * R * points + t"""
        return self.scale * (self.rotation @ points.T).T + self.translation

    def transform_pose(self, qvec: np.ndarray, tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform COLMAP pose (quaternion + translation) correctly

        COLMAP uses camera-to-world transformation: P_world = R_cam2world * P_cam + t_world
        Where R_cam2world = R^T and t_world = -R^T * t_colmap

        For similarity transformation, we need to transform the camera pose in world coordinates:
        1. Extract camera center and orientation in world coordinates
        2. Apply similarity transformation to both
        3. Convert back to COLMAP convention
        """
        # Convert COLMAP pose to camera-to-world transformation
        R_cam2world = quaternion_to_rotation_matrix(qvec).T  # Transpose for cam2world
        t_world = -quaternion_to_rotation_matrix(qvec).T @ tvec  # Camera center in world

        # Apply similarity transformation to camera center
        t_world_new = self.scale * (self.rotation @ t_world) + self.translation

        # Apply rotation transformation to camera orientation
        R_cam2world_new = self.rotation @ R_cam2world

        # Convert back to COLMAP convention (world-to-camera)
        R_world2cam_new = R_cam2world_new.T
        tvec_new = -R_world2cam_new @ t_world_new

        # Convert rotation matrix back to quaternion
        qvec_new = rotation_matrix_to_quaternion(R_world2cam_new)

        return qvec_new, tvec_new

def procrustes_analysis(source_points: np.ndarray, target_points: np.ndarray) -> SimilarityTransformation:
    """
    Compute optimal similarity transformation using Procrustes analysis

    Args:
        source_points: Nx3 array of source points
        target_points: Nx3 array of target points (same order as source)

    Returns:
        SimilarityTransformation object
    """
    assert source_points.shape == target_points.shape
    assert source_points.shape[1] == 3

    # Center the point sets
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    # Compute optimal scale
    source_scale = np.sqrt(np.sum(source_centered ** 2))
    target_scale = np.sqrt(np.sum(target_centered ** 2))

    if source_scale < 1e-12:
        raise ValueError("Source points are degenerate (all at same location)")

    scale = target_scale / source_scale

    # Normalize for rotation computation
    source_normalized = source_centered / source_scale
    target_normalized = target_centered / target_scale

    # Compute optimal rotation using SVD
    H = source_normalized.T @ target_normalized
    U, _, Vt = np.linalg.svd(H)

    rotation = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T

    # Compute translation
    translation = target_centroid - scale * (rotation @ source_centroid)

    return SimilarityTransformation(rotation, translation, scale)

def compute_alignment_error(source_points: np.ndarray, target_points: np.ndarray,
                          transformation: SimilarityTransformation) -> Tuple[np.ndarray, float]:
    """Compute alignment errors after transformation"""
    transformed_points = transformation.transform_points(source_points)
    errors = np.linalg.norm(transformed_points - target_points, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))
    return errors, rmse

def ransac_procrustes(source_points: np.ndarray, target_points: np.ndarray,
                     max_iterations: int = 1000, inlier_threshold: float = 0.1,
                     min_inliers: int = 10) -> Tuple[SimilarityTransformation, np.ndarray, Dict]:
    """
    Robust Procrustes analysis using RANSAC

    Args:
        source_points: Nx3 source points
        target_points: Nx3 target points
        max_iterations: Maximum RANSAC iterations
        inlier_threshold: Distance threshold for inliers
        min_inliers: Minimum number of inliers required

    Returns:
        Best transformation, inlier mask, statistics dictionary
    """
    assert source_points.shape == target_points.shape
    n_points = len(source_points)

    if n_points < 3:
        raise ValueError("Need at least 3 point correspondences")

    best_transformation = None
    best_inliers = np.array([])
    best_score = 0

    stats = {
        'iterations': 0,
        'best_inlier_count': 0,
        'best_rmse': float('inf'),
        'inlier_ratio': 0.0
    }

    for iteration in range(max_iterations):
        # Randomly sample minimum points needed (3 for 3D)
        sample_indices = np.random.choice(n_points, size=3, replace=False)
        sample_source = source_points[sample_indices]
        sample_target = target_points[sample_indices]

        try:
            # Compute transformation from sample
            transformation = procrustes_analysis(sample_source, sample_target)

            # Evaluate on all points
            errors, _ = compute_alignment_error(source_points, target_points, transformation)
            inliers = errors < inlier_threshold
            n_inliers = np.sum(inliers)

            # Update best if this is better
            if n_inliers > best_score and n_inliers >= min_inliers:
                best_score = n_inliers
                best_transformation = transformation
                best_inliers = inliers

                # Refine using all inliers
                if n_inliers > 3:
                    try:
                        refined_transformation = procrustes_analysis(
                            source_points[inliers], target_points[inliers]
                        )

                        # Check if refinement is better
                        refined_errors, refined_rmse = compute_alignment_error(
                            source_points, target_points, refined_transformation
                        )
                        refined_inliers = refined_errors < inlier_threshold

                        if np.sum(refined_inliers) >= n_inliers:
                            best_transformation = refined_transformation
                            best_inliers = refined_inliers
                            best_score = np.sum(refined_inliers)
                    except:
                        pass  # Keep original if refinement fails

        except Exception as e:
            continue  # Skip this iteration if transformation fails

    if best_transformation is None:
        raise RuntimeError(f"RANSAC failed to find valid transformation with >= {min_inliers} inliers")

    # Compute final statistics
    final_errors, final_rmse = compute_alignment_error(
        source_points, target_points, best_transformation
    )

    stats.update({
        'iterations': max_iterations,
        'best_inlier_count': best_score,
        'best_rmse': final_rmse,
        'inlier_ratio': best_score / n_points,
        'median_error': np.median(final_errors[best_inliers]) if best_score > 0 else float('inf')
    })

    return best_transformation, best_inliers, stats

# =============================================================================
# MODEL ALIGNMENT PIPELINE
# =============================================================================

def find_common_images(source_images: Dict[int, ColmapImage],
                      target_images: Dict[int, ColmapImage]) -> Tuple[List[str], List[int], List[int]]:
    """Find common images between two COLMAP models"""
    source_names = {img.name: img_id for img_id, img in source_images.items()}
    target_names = {img.name: img_id for img_id, img in target_images.items()}

    common_names = set(source_names.keys()) & set(target_names.keys())

    if len(common_names) < MIN_CAMERAS_FOR_ALIGNMENT:
        raise ValueError(f"Insufficient common images: {len(common_names)} < {MIN_CAMERAS_FOR_ALIGNMENT}")

    common_names = sorted(list(common_names))
    source_ids = [source_names[name] for name in common_names]
    target_ids = [target_names[name] for name in common_names]

    return common_names, source_ids, target_ids

def extract_camera_centers(images: Dict[int, ColmapImage], image_ids: List[int]) -> np.ndarray:
    """Extract camera centers for given image IDs"""
    centers = []
    for img_id in image_ids:
        centers.append(images[img_id].camera_center)
    return np.array(centers)

def apply_transformation_to_model(source_model_dir: str, output_model_dir: str,
                                transformation: SimilarityTransformation):
    """Apply similarity transformation to entire COLMAP model"""
    os.makedirs(output_model_dir, exist_ok=True)

    # Copy cameras.txt unchanged
    shutil.copy(os.path.join(source_model_dir, 'cameras.txt'),
                os.path.join(output_model_dir, 'cameras.txt'))

    # Transform images.txt
    source_images = read_images_txt(os.path.join(source_model_dir, 'images.txt'))

    with open(os.path.join(output_model_dir, 'images.txt'), 'w') as f:
        f.write("# Image list with one line of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for img_id, img in source_images.items():
            # Transform pose
            qvec_new, tvec_new = transformation.transform_pose(img.qvec, img.tvec)

            # Write image line
            f.write(f"{img_id} {qvec_new[0]:.8f} {qvec_new[1]:.8f} {qvec_new[2]:.8f} {qvec_new[3]:.8f} ")
            f.write(f"{tvec_new[0]:.8f} {tvec_new[1]:.8f} {tvec_new[2]:.8f} {img.camera_id} {img.name}\n")

            # Write empty points2D line (or copy original if needed)
            f.write("\n")

    # Transform points3D.txt if it exists
    points3d_file = os.path.join(source_model_dir, 'points3D.txt')
    if os.path.exists(points3d_file):
        source_points3d = read_points3d_txt(points3d_file)

        with open(os.path.join(output_model_dir, 'points3D.txt'), 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

            for point_id, point in source_points3d.items():
                # Transform 3D point
                xyz_new = transformation.transform_points(point.xyz.reshape(1, -1))[0]

                # Write point line
                f.write(f"{point_id} {xyz_new[0]:.8f} {xyz_new[1]:.8f} {xyz_new[2]:.8f} ")
                f.write(f"{point.rgb[0]} {point.rgb[1]} {point.rgb[2]} {point.error:.8f}")

                # Write track
                for image_id, point2d_idx in point.track:
                    f.write(f" {image_id} {point2d_idx}")
                f.write("\n")

def visualize_alignment_results(source_centers: np.ndarray, target_centers: np.ndarray,
                              transformation: SimilarityTransformation, inliers: np.ndarray,
                              output_path: str):
    """Visualize alignment results in 3D"""
    fig = plt.figure(figsize=(15, 5))

    # Transform source centers
    transformed_centers = transformation.transform_points(source_centers)

    # Plot 1: Before alignment
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(source_centers[:, 0], source_centers[:, 1], source_centers[:, 2],
               c='red', alpha=0.6, label='Source', s=20)
    ax1.scatter(target_centers[:, 0], target_centers[:, 1], target_centers[:, 2],
               c='blue', alpha=0.6, label='Target', s=20)
    ax1.set_title('Before Alignment')
    ax1.legend()

    # Plot 2: After alignment
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(transformed_centers[:, 0], transformed_centers[:, 1], transformed_centers[:, 2],
               c='red', alpha=0.6, label='Source (aligned)', s=20)
    ax2.scatter(target_centers[:, 0], target_centers[:, 1], target_centers[:, 2],
               c='blue', alpha=0.6, label='Target', s=20)
    ax2.set_title('After Alignment')
    ax2.legend()

    # Plot 3: Error distribution
    ax3 = fig.add_subplot(133)
    errors, _ = compute_alignment_error(source_centers, target_centers, transformation)

    ax3.hist(errors[inliers], bins=30, alpha=0.7, label=f'Inliers ({np.sum(inliers)})', color='green')
    ax3.hist(errors[~inliers], bins=30, alpha=0.7, label=f'Outliers ({np.sum(~inliers)})', color='red')
    ax3.axvline(INLIER_THRESHOLD, color='black', linestyle='--', label=f'Threshold ({INLIER_THRESHOLD})')
    ax3.set_xlabel('Alignment Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Alignment visualization saved to: {output_path}")

def verify_transformation(source_images: Dict[int, ColmapImage], target_images: Dict[int, ColmapImage],
                         transformation: SimilarityTransformation, common_names: List[str],
                         source_ids: List[int], target_ids: List[int]) -> Dict:
    """Verify that the transformation correctly aligns camera poses"""
    verification_stats = {}

    # Check camera center alignment
    source_centers = extract_camera_centers(source_images, source_ids)
    target_centers = extract_camera_centers(target_images, target_ids)

    # Transform source centers
    transformed_centers = transformation.transform_points(source_centers)

    # Compute alignment errors
    center_errors = np.linalg.norm(transformed_centers - target_centers, axis=1)
    center_rmse = np.sqrt(np.mean(center_errors ** 2))

    # Check pose transformation consistency (sample a few poses)
    pose_errors = []
    sample_indices = np.random.choice(len(common_names), min(10, len(common_names)), replace=False)

    for idx in sample_indices:
        source_id = source_ids[idx]
        target_id = target_ids[idx]

        source_img = source_images[source_id]
        target_img = target_images[target_id]

        # Transform source pose
        qvec_new, tvec_new = transformation.transform_pose(source_img.qvec, source_img.tvec)

        # Compute transformed camera center using consistent method
        R_new = quaternion_to_rotation_matrix(qvec_new)
        C_transformed = -R_new.T @ tvec_new

        # Compare with target camera center
        C_target = target_img.camera_center
        pose_error = np.linalg.norm(C_transformed - C_target)
        pose_errors.append(pose_error)

    verification_stats = {
        'center_rmse': center_rmse,
        'center_errors_mean': np.mean(center_errors),
        'center_errors_max': np.max(center_errors),
        'pose_consistency_rmse': np.sqrt(np.mean(np.array(pose_errors) ** 2)),
        'pose_errors_mean': np.mean(pose_errors),
        'pose_errors_max': np.max(pose_errors),
        'n_samples_checked': len(sample_indices)
    }

    return verification_stats

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=== COLMAP Model Alignment Tool ===")
    print(f"Source model: {SOURCE_MODEL_PATH}")
    print(f"Target model: {TARGET_MODEL_PATH}")
    print(f"Output model: {OUTPUT_MODEL_PATH}")
    print()

    # Ensure TXT format
    print("Ensuring TXT format...")
    source_dir = ensure_txt_format(SOURCE_MODEL_PATH)
    target_dir = ensure_txt_format(TARGET_MODEL_PATH)

    # Read COLMAP models
    print("Reading COLMAP models...")
    source_images = read_images_txt(os.path.join(source_dir, 'images.txt'))
    target_images = read_images_txt(os.path.join(target_dir, 'images.txt'))

    print(f"Source model: {len(source_images)} images")
    print(f"Target model: {len(target_images)} images")

    # Find common images
    print("Finding common images...")
    common_names, source_ids, target_ids = find_common_images(source_images, target_images)
    print(f"Found {len(common_names)} common images")

    # Extract camera centers
    print("Extracting camera centers...")
    source_centers = extract_camera_centers(source_images, source_ids)
    target_centers = extract_camera_centers(target_images, target_ids)

    # Compute transformation using RANSAC
    print("Computing similarity transformation using RANSAC...")
    print(f"RANSAC parameters: max_iter={RANSAC_ITERATIONS}, threshold={INLIER_THRESHOLD}, min_inliers={MIN_INLIERS}")

    transformation, inliers, stats = ransac_procrustes(
        source_centers, target_centers,
        max_iterations=RANSAC_ITERATIONS,
        inlier_threshold=INLIER_THRESHOLD,
        min_inliers=MIN_INLIERS
    )

    # Print results
    print("\n=== Alignment Results ===")
    print(f"Inliers: {stats['best_inlier_count']}/{len(common_names)} ({stats['inlier_ratio']:.1%})")
    print(f"RMSE: {stats['best_rmse']:.6f}")
    print(f"Median error (inliers): {stats['median_error']:.6f}")
    print(f"Scale factor: {transformation.scale:.6f}")
    print(f"Translation: [{transformation.translation[0]:.3f}, {transformation.translation[1]:.3f}, {transformation.translation[2]:.3f}]")

    # Check if alignment is good enough
    if stats['inlier_ratio'] < 0.5:
        print(f"Warning: Low inlier ratio ({stats['inlier_ratio']:.1%}). Alignment may be unreliable.")

    if stats['best_rmse'] > 1.0:
        print(f"Warning: High RMSE ({stats['best_rmse']:.3f}). Check if models represent the same scene.")

    # Verify transformation consistency
    print("\nVerifying transformation consistency...")
    verification_stats = verify_transformation(
        source_images, target_images, transformation,
        common_names, source_ids, target_ids
    )

    print("=== Verification Results ===")
    print(f"Center alignment RMSE: {verification_stats['center_rmse']:.6f}")
    print(f"Pose consistency RMSE: {verification_stats['pose_consistency_rmse']:.6f}")
    print(f"Max center error: {verification_stats['center_errors_max']:.6f}")
    print(f"Max pose error: {verification_stats['pose_errors_max']:.6f}")

    # Apply transformation to model
    print("\nApplying transformation to source model...")
    apply_transformation_to_model(source_dir, OUTPUT_MODEL_PATH, transformation)

    # Convert to binary format
    print("Converting to binary format...")
    binary_output = OUTPUT_MODEL_PATH + "_bin"
    os.makedirs(binary_output, exist_ok=True)  # Ensure binary output directory exists
    cmd = f'colmap model_converter --input_path "{OUTPUT_MODEL_PATH}" --output_path "{binary_output}" --output_type BIN'
    result = os.system(cmd)
    if result != 0:
        print(f"Warning: Binary conversion failed (exit code {result}), but TXT format is available")

    # Generate visualization
    print("Generating alignment visualization...")
    viz_output = os.path.join(os.path.dirname(OUTPUT_MODEL_PATH), "alignment_visualization.png")
    visualize_alignment_results(source_centers, target_centers, transformation, inliers, viz_output)

    print("\n=== Alignment Complete ===")
    print(f"Aligned model (TXT): {OUTPUT_MODEL_PATH}")
    print(f"Aligned model (BIN): {binary_output}")
    print(f"Visualization: {viz_output}")

    # Save transformation parameters
    transform_file = os.path.join(OUTPUT_MODEL_PATH, "transformation.txt")
    with open(transform_file, 'w') as f:
        f.write("# Similarity transformation from source to target coordinate system\n")
        f.write(f"# Scale: {transformation.scale:.8f}\n")
        f.write("# Rotation matrix (3x3):\n")
        for row in transformation.rotation:
            f.write(f"# {row[0]:.8f} {row[1]:.8f} {row[2]:.8f}\n")
        f.write("# Translation vector:\n")
        f.write(f"# {transformation.translation[0]:.8f} {transformation.translation[1]:.8f} {transformation.translation[2]:.8f}\n")
        f.write(f"# RMSE: {stats['best_rmse']:.6f}\n")
        f.write(f"# Inlier ratio: {stats['inlier_ratio']:.3f}\n")

    print(f"Transformation parameters saved to: {transform_file}")

if __name__ == "__main__":
    # Parse command line arguments (optional)
    parser = argparse.ArgumentParser(description="Align two COLMAP reconstructions")
    parser.add_argument("--source", type=str, help="Source COLMAP model path")
    parser.add_argument("--target", type=str, help="Target COLMAP model path")
    parser.add_argument("--output", type=str, help="Output aligned model path")
    parser.add_argument("--threshold", type=float, default=INLIER_THRESHOLD, help="RANSAC inlier threshold")
    parser.add_argument("--iterations", type=int, default=RANSAC_ITERATIONS, help="RANSAC iterations")

    args = parser.parse_args()

    # Override defaults if provided
    if args.source:
        SOURCE_MODEL_PATH = args.source
    if args.target:
        TARGET_MODEL_PATH = args.target
    if args.output:
        OUTPUT_MODEL_PATH = args.output
    if args.threshold:
        INLIER_THRESHOLD = args.threshold
    if args.iterations:
        RANSAC_ITERATIONS = args.iterations

    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()