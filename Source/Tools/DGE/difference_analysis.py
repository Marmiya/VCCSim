import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.feature import local_binary_pattern
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm
from datetime import datetime
warnings.filterwarnings('ignore')

def convert_numpy_types(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    else:
        return obj

class ImageDifferenceAnalyzer:
    """Comprehensive image difference analysis system for comparing real photos with UE-rendered images."""
    
    def __init__(self, data_root: str = "./Source/Tools/DGE/data", n_workers: int = None):
        self.data_root = Path(data_root)
        self.real_images_dir = self.data_root / "real_images_downscaled"
        self.rendered_images_dir = self.data_root / "rendered_images"  
        self.depth_images_dir = self.data_root / "depth_images"
        self.poses_file = self.data_root / "poses" / "0528_pose.txt"
        
        # Image specifications
        self.img_width = 1216
        self.img_height = 912
        self.fov = 67.38
        
        # Calculate FOV parameters (using the correct approach from generator script)
        self.aspect_ratio = float(self.img_width) / self.img_height
        half_fov_rad = np.radians(self.fov * 0.5)
        self.tan_half_horizontal_fov = np.tan(half_fov_rad)
        self.tan_half_vertical_fov = self.tan_half_horizontal_fov / self.aspect_ratio
        
        # Parallel processing
        self.n_workers = n_workers or min(mp.cpu_count(), 16)
        
        self.poses_df = None
        self.image_names = []
        self.results = {}
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f'./Logs/analysis_results/{timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'metrics').mkdir(exist_ok=True)
        (self.output_dir / 'data').mkdir(exist_ok=True)
        
        print(f"Output directory created: {self.output_dir}")
        print(f"Camera parameters: {self.img_width}x{self.img_height}, FOV: {self.fov}°, Aspect: {self.aspect_ratio:.3f}")
        
    def load_poses(self) -> pd.DataFrame:
        """Load pose information from txt file and get corresponding image names."""
        try:
            # Read poses from txt file (no headers, format: x y z pitch yaw roll in UE coordinates)
            poses_data = np.loadtxt(self.poses_file)
            
            # Get sorted list of rendered images to match with poses
            rendered_images = sorted([f.name for f in self.rendered_images_dir.glob("*.png")])
            self.image_names = rendered_images
            
            if len(poses_data) != len(rendered_images):
                print(f"Warning: Number of poses ({len(poses_data)}) doesn't match number of images ({len(rendered_images)})")
                min_count = min(len(poses_data), len(rendered_images))
                poses_data = poses_data[:min_count]
                self.image_names = rendered_images[:min_count]
            
            # Create DataFrame with UE coordinates (in centimeters and degrees)
            ue_poses = []
            for i, pose in enumerate(poses_data):
                ue_x, ue_y, ue_z, ue_pitch, ue_yaw, ue_roll = pose
                ue_poses.append([
                    self.image_names[i], ue_x, ue_y, ue_z, ue_yaw, ue_pitch, ue_roll
                ])
            
            # Create DataFrame with UE coordinate naming
            self.poses_df = pd.DataFrame(ue_poses, 
                                       columns=['name', 'x', 'y', 'z', 'yaw', 'pitch', 'roll'])
            
            print(f"Loaded {len(self.poses_df)} pose entries")
            print(f"Using UE coordinate system (left-handed, centimeters)")
            return self.poses_df
            
        except Exception as e:
            print(f"Error loading poses: {e}")
            return None
    
    def load_image_pair(self, image_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Load corresponding real, rendered, and depth images."""
        base_name = Path(image_name).stem
        
        real_path = self.real_images_dir / f"{base_name}.jpg"
        rendered_path = self.rendered_images_dir / f"{base_name}.png"
        depth_path = self.depth_images_dir / f"{base_name}.png"
        
        real_img = cv2.imread(str(real_path))
        rendered_img = cv2.imread(str(rendered_path))
        
        # Load depth image as 16-bit (values directly in centimeters)
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED) if depth_path.exists() else None
        
        if real_img is not None:
            real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
        if rendered_img is not None:
            rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
            
        return real_img, rendered_img, depth_img
    
    def align_images(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align two images using ORB feature matching."""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        orb = cv2.ORB_create(nfeatures=5000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return img2, img1, np.eye(3)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 10:
            return img2, img1, np.eye(3)
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
        
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            return img2, img1, np.eye(3)
        
        aligned_img2 = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))
        return aligned_img2, img1, M
    
    def create_valid_mask(self, aligned_img: np.ndarray) -> np.ndarray:
        """Create mask to exclude black edges from aligned images."""
        if len(aligned_img.shape) == 3:
            # Color image: all channels must be non-zero
            valid_mask = np.all(aligned_img > 5, axis=2)  # Use threshold > 5 to handle near-black artifacts
        else:
            # Grayscale image
            valid_mask = aligned_img > 5
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        valid_mask = cv2.morphologyEx(valid_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_OPEN, kernel)
        
        return valid_mask.astype(bool)

    def compute_basic_metrics(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """Compute basic image comparison metrics, excluding black edges."""
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        
        if img1.shape != img2.shape:
            h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            img1, img2 = img1[:h, :w], img2[:h, :w]
        
        # Create valid mask to exclude black edges
        valid_mask = self.create_valid_mask((img2 * 255).astype(np.uint8))
        
        if np.sum(valid_mask) < 0.1 * valid_mask.size:  # Less than 10% valid pixels
            print("Warning: Very few valid pixels after alignment, using full image")
            valid_mask = np.ones(img1.shape[:2], dtype=bool)
        
        # Apply mask to flatten only valid pixels
        img1_valid = img1[valid_mask]
        img2_valid = img2[valid_mask]
        
        mse = float(mean_squared_error(img1_valid, img2_valid))
        
        # For SSIM and PSNR, we need the full images, so create masked versions
        img1_masked = img1.copy()
        img2_masked = img2.copy()
        img1_masked[~valid_mask] = 0
        img2_masked[~valid_mask] = 0
        
        return {
            'mse': mse,
            'mae': float(mean_absolute_error(img1_valid, img2_valid)),
            'rmse': float(np.sqrt(mse)),
            'psnr': float(psnr(img1_masked, img2_masked, data_range=1.0)),
            'ssim': float(np.mean([ssim(img1_masked[:,:,i], img2_masked[:,:,i], data_range=1.0) for i in range(3)])),
            'valid_pixel_ratio': float(np.sum(valid_mask) / valid_mask.size)
        }
    
    def compute_perceptual_metrics(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """Compute perceptual difference metrics, excluding black edges."""
        if img1.shape != img2.shape:
            min_h, min_w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            img1, img2 = img1[:min_h, :min_w], img2[:min_h, :min_w]
        
        # Create valid mask to exclude black edges
        valid_mask = self.create_valid_mask(img2)
        
        metrics = {}
        
        # LAB color space analysis - only on valid pixels
        img1_lab = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
        img2_lab = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)
        
        img1_lab_valid = img1_lab[valid_mask]
        img2_lab_valid = img2_lab[valid_mask]
        metrics['lab_color_diff'] = float(np.mean(np.abs(img1_lab_valid.astype(np.float32) - img2_lab_valid.astype(np.float32))))
        
        # Histogram comparison - only on valid pixels
        for i, channel in enumerate(['r', 'g', 'b']):
            img1_channel_valid = img1[valid_mask, i]
            img2_channel_valid = img2[valid_mask, i]
            
            hist1, _ = np.histogram(img1_channel_valid, bins=256, range=(0, 256))
            hist2, _ = np.histogram(img2_channel_valid, bins=256, range=(0, 256))
            
            hist1 = hist1 / np.sum(hist1) if np.sum(hist1) > 0 else hist1
            hist2 = hist2 / np.sum(hist2) if np.sum(hist2) > 0 else hist2
            
            correlation = float(cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL))
            metrics[f'hist_corr_{channel}'] = correlation
        
        # Texture analysis using LBP - on full images but mask applied later
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        lbp1 = local_binary_pattern(gray1, 24, 8, method='uniform')
        lbp2 = local_binary_pattern(gray2, 24, 8, method='uniform')
        
        # Apply mask to LBP results
        lbp1_valid = lbp1[valid_mask]
        lbp2_valid = lbp2[valid_mask]
        
        hist_lbp1, _ = np.histogram(lbp1_valid.ravel(), bins=26, range=(0, 26))
        hist_lbp2, _ = np.histogram(lbp2_valid.ravel(), bins=26, range=(0, 26))
        
        hist_lbp1 = hist_lbp1 / np.sum(hist_lbp1) if np.sum(hist_lbp1) > 0 else hist_lbp1
        hist_lbp2 = hist_lbp2 / np.sum(hist_lbp2) if np.sum(hist_lbp2) > 0 else hist_lbp2
        
        metrics['texture_lbp_diff'] = float(np.sum(np.abs(hist_lbp1 - hist_lbp2)))
        
        return metrics
        
    def compute_difference_maps(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute difference maps, excluding black edges."""
        img1_float = img1.astype(np.float32) / 255.0
        img2_float = img2.astype(np.float32) / 255.0
        
        # Create valid mask to exclude black edges
        valid_mask = self.create_valid_mask(img2)
        
        # Compute difference maps
        mse_map = np.mean((img1_float - img2_float) ** 2, axis=2)
        mae_map = np.mean(np.abs(img1_float - img2_float), axis=2)
        
        # Mask out invalid regions (set to 0 so they won't be detected as problems)
        mse_map[~valid_mask] = 0
        mae_map[~valid_mask] = 0
        
        difference_maps = {
            'mse': mse_map,
            'mae': mae_map
        }
        
        # LAB color difference
        img1_lab = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB).astype(np.float32)
        img2_lab = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_diff_map = np.mean(np.abs(img1_lab - img2_lab), axis=2)
        lab_diff_map[~valid_mask] = 0
        difference_maps['lab'] = lab_diff_map
        
        # Gradient difference
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        grad1 = np.sqrt(cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)**2 + 
                       cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)**2)
        grad2 = np.sqrt(cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)**2 + 
                       cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)**2)
        
        gradient_diff_map = np.abs(grad1 - grad2)
        gradient_diff_map[~valid_mask] = 0
        difference_maps['gradient'] = gradient_diff_map
        
        return difference_maps
    
    def identify_problematic_regions(self, diff_map: np.ndarray, threshold_percentile: float = 90) -> np.ndarray:
        """Identify regions with high rendering errors."""
        threshold = np.percentile(diff_map, threshold_percentile)
        problematic_mask = diff_map > threshold
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        problematic_mask = cv2.morphologyEx(problematic_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        problematic_mask = cv2.morphologyEx(problematic_mask, cv2.MORPH_OPEN, kernel)
        
        return problematic_mask.astype(bool)
    
    def create_rotation_matrix(self, yaw: float, pitch: float, roll: float) -> np.ndarray:
        """
        Create rotation matrix compatible with Unreal Engine's coordinate system.
        Based on the correct implementation from the generator script.
        """
        # Adjust yaw to match UE convention
        yaw = -yaw
        
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
        cos_roll, sin_roll = np.cos(roll), np.sin(roll)
        
        # Individual rotation matrices
        R_roll = np.array([
            [1, 0, 0],
            [0, cos_roll, -sin_roll],
            [0, sin_roll, cos_roll]
        ])
        
        R_pitch = np.array([
            [cos_pitch, 0, sin_pitch],
            [0, 1, 0],
            [-sin_pitch, 0, cos_pitch]
        ])
        
        R_yaw = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: Roll -> Pitch -> Yaw
        return R_roll @ R_pitch @ R_yaw
    
    def _transform_to_original_space(self, x_aligned: float, y_aligned: float, 
                                   transform_matrix: np.ndarray) -> Tuple[float, float]:
        """
        Transform pixel coordinates from aligned space back to original space.
        This ensures we use the correct depth values from the original depth image.
        """
        if transform_matrix is None or np.allclose(transform_matrix, np.eye(3)):
            return x_aligned, y_aligned
        
        try:
            # Convert to homogeneous coordinates
            aligned_coords = np.array([[x_aligned, y_aligned, 1.0]]).T
            
            # Apply inverse transformation to get original coordinates
            inv_transform = np.linalg.inv(transform_matrix)
            original_coords = inv_transform @ aligned_coords
            
            # Normalize homogeneous coordinates
            if abs(original_coords[2, 0]) > 1e-10:  # Avoid division by zero
                x_original = original_coords[0, 0] / original_coords[2, 0]
                y_original = original_coords[1, 0] / original_coords[2, 0]
            else:
                x_original, y_original = x_aligned, y_aligned
            
            # Clamp coordinates to image bounds
            x_original = np.clip(x_original, 0, self.img_width - 1)
            y_original = np.clip(y_original, 0, self.img_height - 1)
            
            return float(x_original), float(y_original)
            
        except (np.linalg.LinAlgError, ValueError) as e:
            # Handle singular matrix or other errors
            print(f"Warning: Transform error ({e}), using aligned coordinates")
            return x_aligned, y_aligned

    def extract_3d_positions(self, mask: np.ndarray, depth_img: np.ndarray, 
                           pose_info: Dict, transform_matrix: np.ndarray = None) -> List[Dict]:
        """
        Extract 3D positions of problematic regions in UE coordinate system.
        CORRECTED VERSION with alignment transform consideration.
        
        Args:
            mask: Binary mask of problematic regions (in aligned image space)
            depth_img: Original depth image (NOT transformed - this is the ground truth)
            pose_info: Camera pose information
            transform_matrix: Homography matrix from image alignment (optional)
        """
        if depth_img is None:
            return []
        
        # Depth values are directly in centimeters (16-bit PNG)
        if len(depth_img.shape) == 3:
            depth_cm = depth_img[:, :, 0].astype(np.float32)  # Take first channel if multi-channel
        else:
            depth_cm = depth_img.astype(np.float32)
        
        # Camera pose
        cam_pos = np.array([
            float(pose_info.get('x', 0)),
            float(pose_info.get('y', 0)),
            float(pose_info.get('z', 0))
        ])
        
        # Create rotation matrix using the correct approach
        R = self.create_rotation_matrix(
            np.radians(float(pose_info.get('yaw', 0))), 
            np.radians(float(pose_info.get('pitch', 0))), 
            np.radians(float(pose_info.get('roll', 0)))
        )
        
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        positions_3d = []
        
        for label_id in range(1, num_labels):
            region_mask = (labels == label_id)
            if np.sum(region_mask) < 50:
                continue
            
            y_coords, x_coords = np.where(region_mask)
            
            # Get center coordinates in aligned image space
            center_x_aligned = float(np.mean(x_coords))
            center_y_aligned = float(np.mean(y_coords))
            
            # ✅ CRITICAL FIX: Transform coordinates back to original space
            # This ensures we get the correct depth value from the original depth image
            center_x_original, center_y_original = self._transform_to_original_space(
                center_x_aligned, center_y_aligned, transform_matrix
            )
            
            # Sample depth values around the original center point for robustness
            x_int, y_int = int(round(center_x_original)), int(round(center_y_original))
            
            # Sample in a small neighborhood to get robust depth estimate
            sample_size = 3
            x_min = max(0, x_int - sample_size)
            x_max = min(depth_cm.shape[1], x_int + sample_size + 1)
            y_min = max(0, y_int - sample_size)
            y_max = min(depth_cm.shape[0], y_int + sample_size + 1)
            
            depth_sample = depth_cm[y_min:y_max, x_min:x_max]
            valid_depths = depth_sample[depth_sample > 0]
            
            if len(valid_depths) == 0:
                continue
                
            mean_depth = float(np.mean(valid_depths))
            
            # Convert to NDC coordinates using ORIGINAL coordinates
            ndc_x = (2.0 * center_x_original / (self.img_width - 1)) - 1.0
            ndc_y = 1.0 - (2.0 * center_y_original / (self.img_height - 1))
            
            # Calculate view space coordinates
            view_x = ndc_x * self.tan_half_horizontal_fov * mean_depth
            view_y = ndc_y * self.tan_half_vertical_fov * mean_depth
            
            # Camera space position (UE: X=Forward, Y=Right, Z=Up)
            camera_space = np.array([
                mean_depth,  # Forward (X)
                view_x,      # Right (Y)  
                view_y       # Up (Z)
            ])
            
            # Transform to world space using the corrected approach
            world_pos = (R.T @ camera_space) + cam_pos
            x_world, y_world, z_world = world_pos
            
            positions_3d.append({
                'region_id': int(label_id),
                'pixel_center_aligned': (center_x_aligned, center_y_aligned),      # Where problem was detected
                'pixel_center_original': (center_x_original, center_y_original),    # Where we sample depth
                'ndc_coords': (float(ndc_x), float(ndc_y)),
                'view_coords': (float(view_x), float(view_y)),
                'world_position': (float(x_world), float(y_world), float(z_world)),
                'camera_relative': (float(camera_space[0]), float(camera_space[1]), float(camera_space[2])),
                'region_size': int(np.sum(region_mask)),
                'mean_depth': float(mean_depth),
                'coordinate_transform_applied': transform_matrix is not None and not np.allclose(transform_matrix, np.eye(3))
            })
        
        return positions_3d

    def validate_3d_positions(self, positions_3d: List[Dict], image_name: str):
        """Validate extracted 3D positions with debug output including coordinate transform info."""
        if not positions_3d:
            return
        
        print(f"\nDEBUG - 3D Position Validation for {image_name}:")
        print(f"Extracted {len(positions_3d)} regions")
        
        transform_applied = any(pos.get('coordinate_transform_applied', False) for pos in positions_3d)
        print(f"Coordinate transform applied: {transform_applied}")
        
        for i, pos in enumerate(positions_3d[:3]):  # Show first 3 regions
            print(f"  Region {i+1}:")
            if 'pixel_center_aligned' in pos:
                print(f"    Pixel (aligned): ({pos['pixel_center_aligned'][0]:.1f}, {pos['pixel_center_aligned'][1]:.1f})")
                print(f"    Pixel (original): ({pos['pixel_center_original'][0]:.1f}, {pos['pixel_center_original'][1]:.1f})")
            else:
                # Backward compatibility
                pixel_center = pos.get('pixel_center', (0, 0))
                print(f"    Pixel: ({pixel_center[0]:.1f}, {pixel_center[1]:.1f})")
            
            print(f"    NDC: ({pos['ndc_coords'][0]:.3f}, {pos['ndc_coords'][1]:.3f})")
            print(f"    View: ({pos['view_coords'][0]:.1f}, {pos['view_coords'][1]:.1f})")
            print(f"    Depth: {pos['mean_depth']:.1f} cm")
            print(f"    World: ({pos['world_position'][0]:.1f}, {pos['world_position'][1]:.1f}, {pos['world_position'][2]:.1f})")
            
            if pos.get('coordinate_transform_applied', False):
                print(f"    ✅ Coordinates corrected for alignment")

    def save_visualization(self, image_name: str, real_img: np.ndarray, rendered_img: np.ndarray, 
                          depth_img: np.ndarray, basic_metrics: Dict, perceptual_metrics: Dict,
                          difference_maps: Dict, problematic_regions: Dict):
        """Save comprehensive visualization with problem regions on rendered images."""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'Analysis: {image_name}', fontsize=16, fontweight='bold')
        
        # Original images
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(real_img)
        ax1.set_title('Real Image (Reference)', fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(rendered_img)
        ax2.set_title('Rendered Image (Aligned)', fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        if depth_img is not None:
            # Depth values are directly in centimeters (16-bit PNG)
            if len(depth_img.shape) == 3:
                depth_display = depth_img[:, :, 0]
            else:
                depth_display = depth_img
            im3 = ax3.imshow(depth_display, cmap='viridis')
            plt.colorbar(im3, ax=ax3, fraction=0.046, label='Depth (cm)')
            ax3.set_title('Depth Image (16-bit)', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No Depth', ha='center', va='center')
            ax3.set_title('Depth Image', fontweight='bold')
        ax3.axis('off')
        
        # Metrics summary with mask info
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis('off')
        valid_ratio = basic_metrics.get('valid_pixel_ratio', 1.0)
        metrics_text = f"""Metrics:
PSNR: {basic_metrics.get('psnr', 0):.2f} dB
SSIM: {basic_metrics.get('ssim', 0):.3f}
MSE: {basic_metrics.get('mse', 0):.4f}
MAE: {basic_metrics.get('mae', 0):.4f}
LAB Diff: {perceptual_metrics.get('lab_color_diff', 0):.2f}
Texture: {perceptual_metrics.get('texture_lbp_diff', 0):.3f}

Valid Pixels: {valid_ratio*100:.1f}%
Image: {self.img_width}x{self.img_height}
FOV: {self.fov}°
Aspect: {self.aspect_ratio:.3f}"""
        
        ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Difference maps
        diff_titles = ['MSE Difference', 'MAE Difference', 'LAB Difference', 'Gradient Difference']
        diff_keys = ['mse', 'mae', 'lab', 'gradient']
        
        for i, (key, title) in enumerate(zip(diff_keys, diff_titles)):
            if key in difference_maps:
                ax = fig.add_subplot(gs[1, i])
                im = ax.imshow(difference_maps[key], cmap='hot')
                plt.colorbar(im, ax=ax, fraction=0.046)
                ax.set_title(title, fontsize=10)
                ax.axis('off')
        
        # Problem regions: Show on RENDERED image (where the problems are)
        region_titles = ['MSE Problems', 'LAB Problems', 'Gradient Problems', 'Combined Problems']
        region_keys = ['mse', 'lab', 'gradient']
        
        for i, (key, title) in enumerate(zip(region_keys, region_titles[:3])):
            if key in problematic_regions:
                ax = fig.add_subplot(gs[2, i])
                # ✅ FIXED: Show problems on the rendered image where they actually occur
                ax.imshow(rendered_img)
                ax.imshow(problematic_regions[key], alpha=0.6, cmap='Reds')
                ax.set_title(title, fontsize=10)
                ax.axis('off')
        
        # Combined regions on rendered image
        if problematic_regions:
            ax = fig.add_subplot(gs[2, 3])
            ax.imshow(rendered_img)
            combined_mask = np.zeros_like(list(problematic_regions.values())[0])
            for mask in problematic_regions.values():
                combined_mask = combined_mask | mask
            ax.imshow(combined_mask, alpha=0.6, cmap='Oranges')
            ax.set_title('Combined Problems', fontsize=10)
            ax.axis('off')
        
        # Histograms comparison
        for i, (channel, color) in enumerate(zip(['R', 'G', 'B'], ['red', 'green', 'blue'])):
            ax = fig.add_subplot(gs[3, i])
            
            # Create valid mask for histogram calculation
            valid_mask = self.create_valid_mask(rendered_img)
            
            # Calculate histograms only on valid pixels
            real_channel_valid = real_img[valid_mask, i]
            rendered_channel_valid = rendered_img[valid_mask, i]
            
            hist_real, bins = np.histogram(real_channel_valid, bins=256, range=(0, 256))
            hist_rendered, _ = np.histogram(rendered_channel_valid, bins=256, range=(0, 256))
            
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.plot(bin_centers, hist_real, color=color, alpha=0.7, label='Real')
            ax.plot(bin_centers, hist_rendered, color=color, alpha=0.7, label='Rendered', linestyle='--')
            ax.set_title(f'{channel} Histogram (Valid Pixels)', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Side-by-side comparison
        ax_compare = fig.add_subplot(gs[3, 3])
        # Create side-by-side comparison
        comparison_height = min(real_img.shape[0], rendered_img.shape[0])
        real_resized = real_img[:comparison_height, :]
        rendered_resized = rendered_img[:comparison_height, :]
        
        # Add a separator line
        separator = np.ones((comparison_height, 3, 3)) * 255  # White separator
        comparison = np.hstack([real_resized, separator, rendered_resized])
        
        ax_compare.imshow(comparison.astype(np.uint8))
        ax_compare.set_title('Real vs Rendered', fontsize=10)
        ax_compare.axis('off')
        
        plt.tight_layout()
        vis_path = self.output_dir / 'visualizations' / f'{Path(image_name).stem}_analysis.png'
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save metric maps
        for key, diff_map in difference_maps.items():
            np.save(self.output_dir / 'metrics' / f'{Path(image_name).stem}_{key}.npy', diff_map)

    def analyze_single_image(self, image_name: str, save_vis: bool = True) -> Dict:
        """Analyze a single image pair."""
        try:
            real_img, rendered_img, depth_img = self.load_image_pair(image_name)
            
            if real_img is None or rendered_img is None:
                return {}
            
            pose_info = {}
            if self.poses_df is not None:
                pose_row = self.poses_df[self.poses_df['name'] == image_name]
                if not pose_row.empty:
                    pose_info = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                               for k, v in pose_row.iloc[0].to_dict().items()}
            
            aligned_rendered, aligned_real, transform_matrix = self.align_images(real_img, rendered_img)
            
            basic_metrics = self.compute_basic_metrics(aligned_real, aligned_rendered)
            perceptual_metrics = self.compute_perceptual_metrics(aligned_real, aligned_rendered)
            difference_maps = self.compute_difference_maps(aligned_real, aligned_rendered)
            
            problematic_regions = {}
            positions_3d = {}
            
            for map_name, diff_map in difference_maps.items():
                problematic_regions[map_name] = self.identify_problematic_regions(diff_map, 85)
                positions_3d[map_name] = self.extract_3d_positions(
                    problematic_regions[map_name], depth_img, pose_info, transform_matrix)
            
            # Validate 3D positions (debugging)
            if positions_3d.get('mse'):
                self.validate_3d_positions(positions_3d['mse'], image_name)
            
            results = {
                'image_name': image_name,
                'pose_info': pose_info,
                'basic_metrics': basic_metrics,
                'perceptual_metrics': perceptual_metrics,
                'problematic_regions_counts': {name: len(pos_list) for name, pos_list in positions_3d.items()},
                'positions_3d': positions_3d,
                'alignment_transform': transform_matrix.tolist() if transform_matrix is not None else None,
                'image_specs': {
                    'width': self.img_width,
                    'height': self.img_height,
                    'fov': self.fov,
                    'aspect_ratio': self.aspect_ratio
                }
            }
            
            results = convert_numpy_types(results)
            
            if save_vis:
                self.save_visualization(image_name, aligned_real, aligned_rendered, depth_img,
                                      basic_metrics, perceptual_metrics, difference_maps, problematic_regions)
            
            # Save individual data
            data_path = self.output_dir / 'data' / f'{Path(image_name).stem}_data.json'
            with open(data_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            return results
            
        except Exception as e:
            print(f"Error analyzing {image_name}: {e}")
            import traceback
            traceback.print_exc()
            return {}

def analyze_image_worker(args):
    """Worker function for parallel processing with alignment correction."""
    image_name, data_root, img_width, img_height, fov, output_dir = args
    
    try:
        analyzer = ImageDifferenceAnalyzer.__new__(ImageDifferenceAnalyzer)
        analyzer.data_root = Path(data_root)
        analyzer.real_images_dir = analyzer.data_root / "real_images_downscaled"
        analyzer.rendered_images_dir = analyzer.data_root / "rendered_images"  
        analyzer.depth_images_dir = analyzer.data_root / "depth_images"
        analyzer.poses_file = analyzer.data_root / "poses" / "0528_pose.txt"
        analyzer.img_width = img_width
        analyzer.img_height = img_height
        analyzer.fov = fov
        analyzer.aspect_ratio = float(img_width) / img_height
        
        # Calculate FOV parameters correctly
        half_fov_rad = np.radians(fov * 0.5)
        analyzer.tan_half_horizontal_fov = np.tan(half_fov_rad)
        analyzer.tan_half_vertical_fov = analyzer.tan_half_horizontal_fov / analyzer.aspect_ratio
        
        analyzer.output_dir = Path(output_dir)
        
        try:
            # Load poses
            poses_data = np.loadtxt(analyzer.poses_file)
            rendered_images = sorted([f.name for f in analyzer.rendered_images_dir.glob("*.png")])
            
            # Find index of current image
            img_index = rendered_images.index(image_name)
            if img_index < len(poses_data):
                pose = poses_data[img_index]
                ue_x, ue_y, ue_z, ue_pitch, ue_yaw, ue_roll = pose
                
                analyzer.poses_df = pd.DataFrame([[image_name, ue_x, ue_y, ue_z, ue_yaw, ue_pitch, ue_roll]], 
                                               columns=['name', 'x', 'y', 'z', 'yaw', 'pitch', 'roll'])
            else:
                analyzer.poses_df = None
        except:
            analyzer.poses_df = None
        
        # Add the required methods to the worker analyzer instance
        analyzer.create_valid_mask = ImageDifferenceAnalyzer.create_valid_mask.__get__(analyzer)
        analyzer._transform_to_original_space = ImageDifferenceAnalyzer._transform_to_original_space.__get__(analyzer)
        analyzer.create_rotation_matrix = ImageDifferenceAnalyzer.create_rotation_matrix.__get__(analyzer)
        
        result = analyzer.analyze_single_image(image_name, save_vis=True)
        return convert_numpy_types(result)
        
    except Exception as e:
        print(f"Worker error for {image_name}: {e}")
        return {}

class OptimizedImageDifferenceAnalyzer(ImageDifferenceAnalyzer):
    """Extended analyzer with parallel processing."""
    
    def analyze_all_images_parallel(self, max_images: int = None) -> Dict:
        """Analyze all image pairs using parallel processing."""
        if self.poses_df is None:
            self.load_poses()
            
        if self.poses_df is None:
            print("No pose data available")
            return {}
        
        image_names = self.poses_df['name'].head(max_images).tolist() if max_images else self.poses_df['name'].tolist()
        
        print(f"Starting parallel analysis of {len(image_names)} images using {self.n_workers} workers...")
        print(f"Image specifications: {self.img_width}x{self.img_height}, FOV: {self.fov}°, Aspect: {self.aspect_ratio:.3f}")
        
        worker_args = [(name, str(self.data_root), self.img_width, self.img_height, self.fov, str(self.output_dir)) 
                      for name in image_names]
        
        all_results = {}
        summary_metrics = {
            'mse': [], 'mae': [], 'psnr': [], 'ssim': [], 'rmse': [],
            'lab_color_diff': [], 'texture_lbp_diff': [],
            'hist_corr_r': [], 'hist_corr_g': [], 'hist_corr_b': []
        }
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_image = {executor.submit(analyze_image_worker, args): args[0] 
                             for args in worker_args}
            
            for future in tqdm(as_completed(future_to_image), total=len(image_names), desc="Processing"):
                image_name = future_to_image[future]
                try:
                    result = future.result()
                    
                    if result:
                        all_results[image_name] = result
                        
                        # Collect metrics
                        if 'basic_metrics' in result:
                            for metric in ['mse', 'mae', 'psnr', 'ssim', 'rmse']:
                                if metric in result['basic_metrics']:
                                    summary_metrics[metric].append(result['basic_metrics'][metric])
                        
                        if 'perceptual_metrics' in result:
                            for metric in ['lab_color_diff', 'texture_lbp_diff', 'hist_corr_r', 'hist_corr_g', 'hist_corr_b']:
                                if metric in result['perceptual_metrics']:
                                    summary_metrics[metric].append(result['perceptual_metrics'][metric])
                        
                except Exception as e:
                    print(f"Error processing {image_name}: {e}")
        
        processing_time = time.time() - start_time
        
        print(f"Analysis completed in {processing_time:.2f} seconds")
        print(f"Successfully analyzed {len(all_results)} out of {len(image_names)} images")
        
        # Generate summary statistics
        summary_stats = {}
        for metric, values in summary_metrics.items():
            if values:
                summary_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'count': len(values)
                }
        
        self.results = {
            'individual_results': all_results,
            'summary_statistics': summary_stats,
            'total_images_analyzed': len(all_results),
            'processing_time_seconds': processing_time,
            'images_per_second': len(all_results) / processing_time if processing_time > 0 else 0,
            'workers_used': self.n_workers,
            'output_directory': str(self.output_dir),
            'analysis_settings': {
                'image_width': self.img_width,
                'image_height': self.img_height,
                'fov': self.fov,
                'aspect_ratio': self.aspect_ratio
            }
        }
        
        self.generate_summary_report()
        return self.results
    
    def generate_summary_report(self):
        """Generate summary report with visualizations."""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dataset Analysis Summary', fontsize=16, fontweight='bold')
        
        summary_stats = self.results['summary_statistics']
        
        # Plot key metrics
        metrics = ['psnr', 'ssim', 'mse', 'mae', 'lab_color_diff', 'texture_lbp_diff']
        titles = ['PSNR (dB)', 'SSIM', 'MSE', 'MAE', 'LAB Color Diff', 'Texture Diff']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            if metric in summary_stats:
                row, col = i // 3, i % 3
                ax = axes[row, col]
                
                values = [result['basic_metrics'].get(metric) or result['perceptual_metrics'].get(metric, 0)
                         for result in self.results['individual_results'].values()
                         if result.get('basic_metrics') or result.get('perceptual_metrics')]
                
                ax.hist(values, bins=30, alpha=0.7, edgecolor='black')
                ax.axvline(summary_stats[metric]['mean'], color='red', linestyle='--', 
                          label=f"Mean: {summary_stats[metric]['mean']:.3f}")
                ax.set_title(title, fontweight='bold')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary_report.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print summary
        print(f"\n{'='*80}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"Output directory: {self.output_dir}")
        print(f"Total images: {self.results['total_images_analyzed']}")
        print(f"Processing time: {self.results['processing_time_seconds']:.2f}s")
        print(f"Speed: {self.results['images_per_second']:.2f} images/sec")
        print(f"Image specs: {self.img_width}x{self.img_height}, FOV: {self.fov}°, Aspect: {self.aspect_ratio:.3f}")
        
        for metric, stats in summary_stats.items():
            print(f"\n{metric.upper().replace('_', ' ')}:")
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"{'='*80}")

# Usage
if __name__ == "__main__":
    analyzer = OptimizedImageDifferenceAnalyzer("./Source/Tools/DGE/data", n_workers=8)
    analyzer.load_poses()
    
    # Analyze all images (remove max_images for full dataset)
    # Now with improved coordinate alignment correction for precise 3D positioning
    results = analyzer.analyze_all_images_parallel(max_images=5)
    
    print("Analysis completed with alignment-corrected 3D positioning!")