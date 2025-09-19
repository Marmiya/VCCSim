"""
COLMAP Model Alignment Tool
===========================

Advanced COLMAP reconstruction alignment tool using robust geometric algorithms.
Aligns two COLMAP reconstructions of the same scene using camera poses with:
- Robust Procrustes analysis with RANSAC for 7-DOF similarity transformation
- 3D rotation (3 DOF) + 3D translation (3 DOF) + Uniform scaling (1 DOF)
- Comprehensive outlier detection and statistical validation
- Automatic format detection and conversion
- Detailed alignment verification and error analysis

Usage:
    python colmap_align_models.py --source SOURCE_DIR --target TARGET_DIR --output OUTPUT_DIR [options]

Example:
    python colmap_align_models.py --source ./model1 --target ./model2 --output ./aligned

Author: VCCSim Project
Based on: Classical Procrustes analysis and RANSAC algorithms
"""

import numpy as np
import os
import shutil
import argparse
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from colmap_utils import (
    ColmapImage, read_images_txt, read_points3d_txt,
    qvec2rotmat, rotation_matrix_to_quaternion, calculate_camera_center,
    ensure_txt_format, find_common_images,
    validate_colmap_directory, extract_camera_centers
)

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
        R_cam2world = qvec2rotmat(qvec).T  # Transpose for cam2world
        t_world = -qvec2rotmat(qvec).T @ tvec  # Camera center in world

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
    # Get threshold from visualization function attribute or use default
    threshold = getattr(visualize_alignment_results, '_threshold', DEFAULT_INLIER_THRESHOLD)
    ax3.axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold})')
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
        R_new = qvec2rotmat(qvec_new)
        C_transformed = -R_new.T @ tvec_new

        # Compare with target camera center
        C_target = calculate_camera_center(target_img.qvec, target_img.tvec)
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

def main(args):
    print("=== COLMAP Model Alignment Tool ===")
    print(f"Source model: {args.source}")
    print(f"Target model: {args.target}")
    print(f"Output model: {args.output}")
    print()

    # Ensure TXT format
    print("Ensuring TXT format...")
    source_dir = ensure_txt_format(args.source)
    target_dir = ensure_txt_format(args.target)

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

    if len(common_names) < args.min_cameras:
        raise ValueError(f"Insufficient common images: {len(common_names)} < {args.min_cameras}")

    # Extract camera centers
    print("Extracting camera centers...")
    source_centers = extract_camera_centers(source_images, source_ids)
    target_centers = extract_camera_centers(target_images, target_ids)

    # Compute transformation using RANSAC
    print("Computing similarity transformation using RANSAC...")
    print(f"RANSAC parameters: max_iter={args.iterations}, threshold={args.threshold}, min_inliers={args.min_inliers}")

    transformation, inliers, stats = ransac_procrustes(
        source_centers, target_centers,
        max_iterations=args.iterations,
        inlier_threshold=args.threshold,
        min_inliers=args.min_inliers
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
    apply_transformation_to_model(source_dir, args.output, transformation)

    # Convert to binary format
    print("Converting to binary format...")
    binary_output = args.output + "_bin"
    os.makedirs(binary_output, exist_ok=True)  # Ensure binary output directory exists
    cmd = f'colmap model_converter --input_path "{args.output}" --output_path "{binary_output}" --output_type BIN'
    result = os.system(cmd)
    if result != 0:
        print(f"Warning: Binary conversion failed (exit code {result}), but TXT format is available")

    # Generate visualization
    if not args.no_visualization:
        print("Generating alignment visualization...")
        viz_output = os.path.join(os.path.dirname(args.output), "alignment_visualization.png")
        setup_visualization_threshold(args.threshold)
        visualize_alignment_results(source_centers, target_centers, transformation, inliers, viz_output)
        print(f"Visualization: {viz_output}")

    print("\n=== Alignment Complete ===")
    print(f"Aligned model (TXT): {args.output}")
    print(f"Aligned model (BIN): {binary_output}")

    # Save transformation parameters
    transform_file = os.path.join(args.output, "transformation.txt")
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

def setup_visualization_threshold(threshold):
    """Helper to pass threshold to visualization function"""
    visualize_alignment_results._threshold = threshold

# ============================ Main ============================

# Default algorithm parameters
DEFAULT_RANSAC_ITERATIONS = 1000
DEFAULT_MIN_INLIERS = 10
DEFAULT_INLIER_THRESHOLD = 0.1
DEFAULT_MIN_CAMERAS = 6

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align two COLMAP reconstructions using robust Procrustes analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("--source", "-s", type=str, default=r"D:\Data\BaoAn\colmap\colmap",
                       help="Source COLMAP model directory path")
    parser.add_argument("--target", "-t", type=str, default=r"D:\Data\BaoAn\colmap\rc_colmap_refine",
                       help="Target COLMAP model directory path")
    parser.add_argument("--output", "-o", type=str, default=r"D:\Data\BaoAn\colmap\aligned_colmap",
                       help="Output aligned model directory path")

    # Algorithm parameters
    parser.add_argument("--threshold", type=float, default=DEFAULT_INLIER_THRESHOLD,
                       help="RANSAC inlier threshold for alignment")
    parser.add_argument("--iterations", type=int, default=DEFAULT_RANSAC_ITERATIONS,
                       help="Maximum RANSAC iterations")
    parser.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS,
                       help="Minimum number of inliers required")
    parser.add_argument("--min-cameras", type=int, default=DEFAULT_MIN_CAMERAS,
                       help="Minimum number of common cameras required")

    # Output options
    parser.add_argument("--no-visualization", action="store_true",
                       help="Skip generating alignment visualization")
    parser.add_argument("--verbose", "-v", action="store_true", default=True,
                       help="Enable verbose output")

    args = parser.parse_args()

    # Validate input directories
    try:
        validate_colmap_directory(args.source)
        validate_colmap_directory(args.target)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    try:
        main(args)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        exit(1)