import numpy as np
import os
import random
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import itertools

# Configuration paths
rc_images_path = r"D:\Data\360_v2\garden\mesh\rc_colmap\images.txt"
base_images_path = r"D:\Data\360_v2\garden\sparse\0\images.txt"
output_dir = r"D:\Data\360_v2\garden\mesh\rc_scale_aligned"

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix"""
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
    ], dtype=float)

def read_camera_centers_from_images_txt(filepath: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    Read camera center coordinates from images.txt
    COLMAP definition: x_cam = R * X_world + t
    Camera center: C = -R^T * t
    """
    centers = []
    image_names = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        
        parts = line.split()
        if len(parts) >= 10:
            # Read pose parameters
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            image_name = parts[9]  # Image name
            
            # Calculate camera center
            q = np.array([qw, qx, qy, qz])
            t = np.array([tx, ty, tz])
            R = quaternion_to_rotation_matrix(q)
            
            # C = -R^T * t
            center = -R.T @ t
            
            centers.append(center)
            image_names.append(image_name)
        
        i += 2  # Skip next line (keypoint line)
    
    return centers, image_names

def find_common_cameras(centers1: List[np.ndarray], names1: List[str], 
                       centers2: List[np.ndarray], names2: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Find common cameras and return corresponding center coordinates"""
    name_to_center1 = {name: center for name, center in zip(names1, centers1)}
    name_to_center2 = {name: center for name, center in zip(names2, centers2)}
    
    common_names = set(names1) & set(names2)
    print(f"Found {len(common_names)} common images")
    
    if len(common_names) < 3:
        raise ValueError(f"Insufficient common images: {len(common_names)} < 3")
    
    # Extract center coordinates of common cameras
    common_centers1 = []
    common_centers2 = []
    
    for name in common_names:
        common_centers1.append(name_to_center1[name])
        common_centers2.append(name_to_center2[name])
    
    return np.array(common_centers1), np.array(common_centers2)

def estimate_scale_robust_median(centers_rc: np.ndarray, centers_base: np.ndarray, 
                                max_pairs: int = 5000) -> Tuple[float, Dict]:
    """
    Method 1: Robust median method for scale estimation
    Calculate distance ratios for all camera pairs and take the median
    """
    n_cameras = len(centers_rc)
    
    # Generate all possible camera pairs
    all_pairs = list(itertools.combinations(range(n_cameras), 2))
    
    # If too many camera pairs, randomly sample
    if len(all_pairs) > max_pairs:
        pairs_to_use = random.sample(all_pairs, max_pairs)
        print(f"Randomly sampling {max_pairs} camera pairs (total {len(all_pairs)} pairs)")
    else:
        pairs_to_use = all_pairs
        print(f"Using all {len(all_pairs)} camera pairs")
    
    ratios = []
    valid_pairs = []
    
    for i, j in pairs_to_use:
        # Calculate distance in RC reconstruction
        dist_rc = np.linalg.norm(centers_rc[i] - centers_rc[j])
        # Calculate distance in BASE reconstruction
        dist_base = np.linalg.norm(centers_base[i] - centers_base[j])
        
        # Avoid division by zero
        if dist_rc > 1e-6 and dist_base > 1e-6:
            ratio = dist_base / dist_rc
            ratios.append(ratio)
            valid_pairs.append((i, j))
    
    ratios = np.array(ratios)
    
    # Calculate median as scale factor
    scale_median = np.median(ratios)
    
    # Calculate MAD (Median Absolute Deviation) for outlier detection
    mad = np.median(np.abs(ratios - scale_median))
    
    # Filter outliers (those exceeding 3*MAD are considered outliers)
    mad_threshold = 3.0
    valid_mask = np.abs(ratios - scale_median) <= mad_threshold * mad
    ratios_filtered = ratios[valid_mask]
    
    scale_robust = np.median(ratios_filtered)
    
    stats = {
        'n_pairs_total': len(all_pairs),
        'n_pairs_used': len(valid_pairs),
        'n_pairs_after_filter': len(ratios_filtered),
        'scale_median': scale_median,
        'scale_robust': scale_robust,
        'mad': mad,
        'ratios_mean': np.mean(ratios),
        'ratios_std': np.std(ratios),
        'ratios_min': np.min(ratios),
        'ratios_max': np.max(ratios),
        'outlier_ratio': 1 - len(ratios_filtered) / len(ratios)
    }
    
    return scale_robust, stats

def estimate_scale_procrustes(centers_rc: np.ndarray, centers_base: np.ndarray) -> Tuple[float, Dict]:
    """
    Method 2: Procrustes analysis for scale estimation
    Find optimal scale allowing for rotation and translation
    """
    # Centering
    centroid_rc = np.mean(centers_rc, axis=0)
    centroid_base = np.mean(centers_base, axis=0)
    
    centered_rc = centers_rc - centroid_rc
    centered_base = centers_base - centroid_base
    
    # Calculate covariance matrix
    H = centered_rc.T @ centered_base
    
    # SVD decomposition
    U, sigma, Vt = np.linalg.svd(H)
    
    # Optimal scale: s = trace(Sigma) / ||X_RC_centered||_F^2
    trace_sigma = np.sum(sigma)
    frobenius_norm_squared = np.sum(centered_rc ** 2)
    
    scale_procrustes = trace_sigma / frobenius_norm_squared
    
    # Calculate optimal rotation matrix (for statistics)
    R_optimal = Vt.T @ U.T
    
    # Ensure determinant of rotation matrix is positive
    if np.linalg.det(R_optimal) < 0:
        Vt[-1, :] *= -1
        R_optimal = Vt.T @ U.T
    
    # Calculate residuals after alignment
    aligned_rc = scale_procrustes * (R_optimal @ centered_rc.T).T + centroid_base
    residuals = np.linalg.norm(aligned_rc - centers_base, axis=1)
    
    stats = {
        'scale_procrustes': scale_procrustes,
        'trace_sigma': trace_sigma,
        'frobenius_norm_squared': frobenius_norm_squared,
        'mean_residual': np.mean(residuals),
        'max_residual': np.max(residuals),
        'rms_residual': np.sqrt(np.mean(residuals**2))
    }
    
    return scale_procrustes, stats

def visualize_distance_ratios(ratios: np.ndarray, scale_estimate: float, title: str, output_path: str):
    """Visualize distance ratio distribution"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(ratios, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(scale_estimate, color='red', linestyle='--', linewidth=2, label=f'Estimated scale: {scale_estimate:.4f}')
    plt.xlabel('Distance ratio')
    plt.ylabel('Frequency')
    plt.title('Distance ratio distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.boxplot(ratios)
    plt.ylabel('Distance ratio')
    plt.title('Distance ratio boxplot')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(sorted(ratios), marker='o', markersize=2, alpha=0.6)
    plt.axhline(scale_estimate, color='red', linestyle='--', linewidth=2, label=f'Estimated scale: {scale_estimate:.4f}')
    plt.xlabel('Sorted index')
    plt.ylabel('Distance ratio')
    plt.title('Distance ratio sorted plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Distance ratio visualization saved to: {output_path}")
    plt.close()

def apply_scale_to_colmap_model(input_dir: str, output_dir: str, scale: float):
    """Apply scale transformation to COLMAP model"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy cameras.txt
    import shutil
    shutil.copy(os.path.join(input_dir, 'cameras.txt'), 
                os.path.join(output_dir, 'cameras.txt'))
    
    # Process images.txt - only scale translation vectors
    with open(os.path.join(input_dir, 'images.txt'), 'r') as f:
        lines = f.readlines()
    
    with open(os.path.join(output_dir, 'images.txt'), 'w') as f:
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith('#'):
                f.write(line)
                i += 1
                continue
            
            parts = line.strip().split()
            if len(parts) >= 10:
                # Scale translation vectors
                tx, ty, tz = map(float, parts[5:8])
                tx_scaled = tx * scale
                ty_scaled = ty * scale  
                tz_scaled = tz * scale
                
                # Rewrite this line
                new_line = f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]} "
                new_line += f"{tx_scaled:.8f} {ty_scaled:.8f} {tz_scaled:.8f} {parts[8]} {parts[9]}\n"
                f.write(new_line)
                
                # Copy next line (keypoint line)
                i += 1
                if i < len(lines):
                    f.write(lines[i])
            i += 1
    
    # Process points3D.txt - scale all 3D points
    with open(os.path.join(input_dir, 'points3D.txt'), 'r') as f:
        lines = f.readlines()
    
    with open(os.path.join(output_dir, 'points3D.txt'), 'w') as f:
        for line in lines:
            if line.startswith('#'):
                f.write(line)
                continue
            
            parts = line.strip().split()
            if len(parts) >= 8:
                # Scale 3D point coordinates
                x, y, z = map(float, parts[1:4])
                x_scaled = x * scale
                y_scaled = y * scale
                z_scaled = z * scale
                
                # Rewrite this line
                new_line = f"{parts[0]} {x_scaled:.8f} {y_scaled:.8f} {z_scaled:.8f}"
                for part in parts[4:]:
                    new_line += f" {part}"
                new_line += "\n"
                f.write(new_line)

def main():
    print("=== Scale estimation based on camera center distances ===")
    
    # Check and convert file format
    def ensure_txt_format(model_dir, images_path):
        if not os.path.exists(images_path):
            print(f"Converting {model_dir} to TXT format...")
            txt_dir = model_dir + "_txt"
            os.system(f'colmap model_converter --input_path "{model_dir}" --output_path "{txt_dir}" --output_type TXT')
            return os.path.join(txt_dir, 'images.txt')
        return images_path
    
    rc_images_path_local = ensure_txt_format(r"D:\Data\360_v2\garden\mesh\rc_colmap", rc_images_path)
    base_images_path_local = ensure_txt_format(r"D:\Data\360_v2\garden\sparse\0", base_images_path)
    
    # Read camera centers
    print("Reading camera centers from RC reconstruction...")
    centers_rc, names_rc = read_camera_centers_from_images_txt(rc_images_path_local)
    
    print("Reading camera centers from BASE reconstruction...")
    centers_base, names_base = read_camera_centers_from_images_txt(base_images_path_local)
    
    print(f"RC reconstruction: {len(centers_rc)} cameras")
    print(f"BASE reconstruction: {len(centers_base)} cameras")
    
    # Find common cameras
    common_centers_rc, common_centers_base = find_common_cameras(
        centers_rc, names_rc, centers_base, names_base
    )
    
    print(f"\n=== Method 1: Robust median method ===")
    scale_median, stats_median = estimate_scale_robust_median(common_centers_rc, common_centers_base)
    
    print(f"Scale estimation results:")
    print(f"  Robust scale factor: {scale_median:.6f}")
    print(f"  Camera pairs used: {stats_median['n_pairs_used']}")
    print(f"  Camera pairs after filtering: {stats_median['n_pairs_after_filter']}")
    print(f"  Outlier ratio: {stats_median['outlier_ratio']:.2%}")
    print(f"  Ratio statistics: mean={stats_median['ratios_mean']:.6f}, std={stats_median['ratios_std']:.6f}")
    print(f"  Ratio range: [{stats_median['ratios_min']:.6f}, {stats_median['ratios_max']:.6f}]")
    
    print(f"\n=== Method 2: Procrustes analysis ===")
    scale_procrustes, stats_procrustes = estimate_scale_procrustes(common_centers_rc, common_centers_base)
    
    print(f"Procrustes scale estimation:")
    print(f"  Scale factor: {scale_procrustes:.6f}")
    print(f"  Mean residual: {stats_procrustes['mean_residual']:.6f}")
    print(f"  Max residual: {stats_procrustes['max_residual']:.6f}")
    print(f"  RMS residual: {stats_procrustes['rms_residual']:.6f}")
    
    print(f"\n=== Method comparison ===")
    print(f"Median method scale: {scale_median:.6f}")
    print(f"Procrustes scale: {scale_procrustes:.6f}")
    print(f"Relative difference: {abs(scale_median - scale_procrustes) / scale_median * 100:.2f}%")
    
    # Select final scale (recommended to use median method, more robust)
    final_scale = scale_median
    print(f"\nRecommended to use the result of robust median method: {final_scale:.6f}")
    
    # Visualize distance ratio distribution
    # Recalculate all ratios for visualization
    all_pairs = list(itertools.combinations(range(len(common_centers_rc)), 2))
    all_ratios = []
    for i, j in all_pairs:
        dist_rc = np.linalg.norm(common_centers_rc[i] - common_centers_rc[j])
        dist_base = np.linalg.norm(common_centers_base[i] - common_centers_base[j])
        if dist_rc > 1e-6 and dist_base > 1e-6:
            all_ratios.append(dist_base / dist_rc)
    
    visualize_distance_ratios(
        np.array(all_ratios), 
        final_scale,
        "Distance ratio distribution", 
        os.path.join(os.path.dirname(output_dir), "scale_estimation_analysis.png")
    )
    
    # Apply scale transformation
    print(f"\nApplying scale transformation to RC model...")
    rc_model_dir = os.path.dirname(rc_images_path)
    apply_scale_to_colmap_model(rc_model_dir, output_dir, final_scale)
    
    # Convert to binary format
    print("Converting to binary format...")
    os.system(f'colmap model_converter --input_path "{output_dir}" --output_path "{output_dir}_bin" --output_type BIN')
    
    print(f"\n=== Done! ===")
    print(f"Scaled model saved at: {output_dir}")
    print(f"Binary format: {output_dir}_bin")
    print(f"Final scale factor: {final_scale:.6f}")
    
    # Verify results
    print(f"\nVerification: RC model coordinates will be multiplied by {final_scale:.6f} to match BASE model scale")

if __name__ == "__main__":
    main()