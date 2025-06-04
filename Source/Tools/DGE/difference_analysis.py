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
import pickle
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
    elif hasattr(obj, 'item'):  # Handle numpy scalar types
        return obj.item()
    elif hasattr(obj, 'tolist'):  # Handle any numpy array-like objects
        return obj.tolist()
    else:
        return obj

class ImageDifferenceAnalyzer:
    """
    Comprehensive image difference analysis system for comparing real photos 
    with UE-rendered images, specifically designed for 3DGS enhancement research.
    Now with parallel processing and comprehensive visualizations.
    """
    
    def __init__(self, data_root: str = "./Source/Tools/DGE/data", n_workers: int = None):
        self.data_root = Path(data_root)
        self.real_images_dir = self.data_root / "real_images_downscaled"
        self.rendered_images_dir = self.data_root / "rendered_images"  
        self.depth_images_dir = self.data_root  / "depth_images"
        self.poses_file = self.data_root / "poses" / "0528_filtered.csv"
        
        # Image specifications
        self.img_width = 1216
        self.img_height = 912
        self.fov = 67.38
        
        # Parallel processing
        self.n_workers = n_workers or min(mp.cpu_count(), 8)  # Limit to 8 to avoid memory issues
        
        self.poses_df = None
        self.results = {}
        
        # Create output directories
        self.output_dir = Path('./Logs/analysis_results')
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'individual_visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'metric_maps').mkdir(exist_ok=True)
        (self.output_dir / 'data').mkdir(exist_ok=True)
        
    def load_poses(self) -> pd.DataFrame:
        """Load pose information from CSV file."""
        try:
            self.poses_df = pd.read_csv(self.poses_file, skiprows=1)
            self.poses_df.columns = ['name', 'x', 'y', 'z', 'heading', 'pitch', 'roll']
            print(f"Loaded {len(self.poses_df)} pose entries")
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
        
        real_img = None
        rendered_img = None
        depth_img = None
        
        if real_path.exists():
            real_img = cv2.imread(str(real_path))
            real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
            
        if rendered_path.exists():
            rendered_img = cv2.imread(str(rendered_path))
            rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
            
        if depth_path.exists():
            depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            
        return real_img, rendered_img, depth_img
    
    def align_images(self, img1: np.ndarray, img2: np.ndarray, method='orb') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align two images to handle small pose errors."""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        if method == 'orb':
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
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is None:
                return img2, img1, np.eye(3)
            
            aligned_img2 = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))
            return aligned_img2, img1, M
        
        return img2, img1, np.eye(3)
    
    def compute_basic_metrics(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """Compute basic image comparison metrics."""
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        
        if img1.shape != img2.shape:
            h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            img1 = img1[:h, :w]
            img2 = img2[:h, :w]
        
        metrics = {}
        
        metrics['mse'] = float(mean_squared_error(img1.flatten(), img2.flatten()))
        metrics['mae'] = float(mean_absolute_error(img1.flatten(), img2.flatten()))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        metrics['psnr'] = float(psnr(img1, img2, data_range=1.0))
        
        ssim_scores = []
        for i in range(img1.shape[2]):
            ssim_score = float(ssim(img1[:,:,i], img2[:,:,i], data_range=1.0))
            ssim_scores.append(ssim_score)
        metrics['ssim'] = float(np.mean(ssim_scores))
        metrics['ssim_per_channel'] = ssim_scores
        
        return metrics
    
    def compute_perceptual_metrics(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """Compute perceptual difference metrics."""
        metrics = {}
        
        if img1.shape != img2.shape:
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
        
        # LAB color space analysis
        img1_lab = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
        img2_lab = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)
        lab_diff = float(np.mean(np.abs(img1_lab.astype(np.float32) - img2_lab.astype(np.float32))))
        metrics['lab_color_diff'] = lab_diff
        
        # Histogram comparison
        for i, channel in enumerate(['R', 'G', 'B']):
            hist1 = cv2.calcHist([img1], [i], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [i], None, [256], [0, 256])
            
            hist1 = hist1.astype(np.float32)
            hist2 = hist2.astype(np.float32)
            
            total_pixels = np.float32(img1.shape[0] * img1.shape[1])
            hist1_norm = hist1 / total_pixels
            hist2_norm = hist2 / total_pixels
            
            sum1 = np.sum(hist1_norm)
            sum2 = np.sum(hist2_norm)
            
            if sum1 > 0 and sum2 > 0:
                hist1_final = (hist1_norm / sum1).astype(np.float32)
                hist2_final = (hist2_norm / sum2).astype(np.float32)
                
                correlation = float(cv2.compareHist(hist1_final, hist2_final, cv2.HISTCMP_CORREL))
                metrics[f'hist_corr_{channel.lower()}'] = correlation
            else:
                metrics[f'hist_corr_{channel.lower()}'] = 0.0
        
        # Texture analysis
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        lbp1 = local_binary_pattern(gray1, 24, 8, method='uniform')
        lbp2 = local_binary_pattern(gray2, 24, 8, method='uniform')
        
        hist_lbp1, _ = np.histogram(lbp1.ravel(), bins=26, range=(0, 26))
        hist_lbp2, _ = np.histogram(lbp2.ravel(), bins=26, range=(0, 26))
        
        hist_lbp1 = hist_lbp1.astype(np.float32)
        hist_lbp2 = hist_lbp2.astype(np.float32)
        
        sum1 = np.sum(hist_lbp1)
        sum2 = np.sum(hist_lbp2)
        
        if sum1 > 0:
            hist_lbp1 = (hist_lbp1 / sum1).astype(np.float32)
        if sum2 > 0:
            hist_lbp2 = (hist_lbp2 / sum2).astype(np.float32)
            
        lbp_diff = float(np.sum(np.abs(hist_lbp1 - hist_lbp2)))
        metrics['texture_lbp_diff'] = lbp_diff
        
        return metrics
        
    def compute_all_difference_maps(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute multiple types of difference maps."""
        img1_float = img1.astype(np.float32) / 255.0
        img2_float = img2.astype(np.float32) / 255.0
        
        difference_maps = {}
        
        # MSE difference map
        difference_maps['mse'] = np.mean((img1_float - img2_float) ** 2, axis=2)
        
        # MAE difference map
        difference_maps['mae'] = np.mean(np.abs(img1_float - img2_float), axis=2)
        
        # SSIM difference map (simplified version for speed)
        difference_maps['ssim'] = np.zeros((img1.shape[0], img1.shape[1]))
        window_size = 11
        step = 5  # Reduced sampling for speed
        
        for i in range(window_size//2, img1.shape[0] - window_size//2, step):
            for j in range(window_size//2, img1.shape[1] - window_size//2, step):
                patch1 = img1_float[i-window_size//2:i+window_size//2+1, 
                            j-window_size//2:j+window_size//2+1]
                patch2 = img2_float[i-window_size//2:i+window_size//2+1, 
                            j-window_size//2:j+window_size//2+1]
                
                ssim_val = np.mean([ssim(patch1[:,:,k], patch2[:,:,k], data_range=1.0) for k in range(3)])
                difference_maps['ssim'][i-step//2:i+step//2+1, j-step//2:j+step//2+1] = 1 - ssim_val
        
        # LAB color difference map
        img1_lab = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB).astype(np.float32)
        img2_lab = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB).astype(np.float32)
        difference_maps['lab'] = np.mean(np.abs(img1_lab - img2_lab), axis=2)
        
        # Gradient difference map
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        grad1_x = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
        grad1_y = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
        grad2_x = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
        grad2_y = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
        
        grad1_mag = np.sqrt(grad1_x**2 + grad1_y**2)
        grad2_mag = np.sqrt(grad2_x**2 + grad2_y**2)
        
        difference_maps['gradient'] = np.abs(grad1_mag - grad2_mag)
        
        return difference_maps
    
    def identify_problematic_regions(self, diff_map: np.ndarray, threshold_percentile: float = 90) -> np.ndarray:
        """Identify regions with high rendering errors."""
        threshold = np.percentile(diff_map, threshold_percentile)
        problematic_mask = diff_map > threshold
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        problematic_mask = cv2.morphologyEx(problematic_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        problematic_mask = cv2.morphologyEx(problematic_mask, cv2.MORPH_OPEN, kernel)
        
        return problematic_mask.astype(bool)
    
    def extract_3d_positions_from_regions(self, mask: np.ndarray, depth_img: np.ndarray, pose_info: Dict) -> List[Dict]:
        """Extract 3D positions of problematic regions using depth information."""
        if depth_img is None:
            return []
        
        if depth_img.dtype == np.uint16:
            depth_normalized = depth_img.astype(np.float32) / 65535.0
        else:
            depth_normalized = depth_img.astype(np.float32) / 255.0
        
        fov_rad = np.radians(self.fov)
        focal_length = self.img_width / (2 * np.tan(fov_rad / 2))
        cx, cy = self.img_width / 2, self.img_height / 2
        
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        positions_3d = []
        
        for label_id in range(1, num_labels):
            region_mask = (labels == label_id)
            if np.sum(region_mask) < 50:
                continue
            
            y_coords, x_coords = np.where(region_mask)
            depth_values = depth_normalized[region_mask]
            mean_depth = np.mean(depth_values[depth_values > 0])
            
            if mean_depth == 0:
                continue
            
            center_x = float(np.mean(x_coords))
            center_y = float(np.mean(y_coords))
            
            x_cam = (center_x - cx) * mean_depth / focal_length
            y_cam = (center_y - cy) * mean_depth / focal_length
            z_cam = mean_depth
            
            x_world = float(pose_info.get('x', 0)) + float(x_cam)
            y_world = float(pose_info.get('y', 0)) + float(y_cam)
            z_world = float(pose_info.get('z', 0)) + float(z_cam)
            
            positions_3d.append({
                'region_id': int(label_id),
                'pixel_center': (center_x, center_y),
                'world_position': (x_world, y_world, z_world),
                'region_size': int(np.sum(region_mask)),
                'mean_depth': float(mean_depth),
                'error_magnitude': float(np.mean(depth_values))
            })
        
        return positions_3d
    
    def save_comprehensive_visualization(self, image_name: str, real_img: np.ndarray, 
                                       rendered_img: np.ndarray, depth_img: np.ndarray,
                                       basic_metrics: Dict, perceptual_metrics: Dict,
                                       difference_maps: Dict, problematic_regions: Dict):
        """Save comprehensive visualization with all metrics."""
        
        # Create main comparison figure
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(6, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'Comprehensive Analysis: {image_name}', fontsize=20, fontweight='bold')
        
        # Row 1: Original images and depth
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(real_img)
        ax1.set_title('Real Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(rendered_img)
        ax2.set_title('Rendered Image', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        if depth_img is not None:
            im3 = ax3.imshow(depth_img, cmap='viridis')
            plt.colorbar(im3, ax=ax3, fraction=0.046)
            ax3.set_title('Depth Image', fontsize=14, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No Depth Image', ha='center', va='center', fontsize=12)
            ax3.set_title('Depth Image', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # Metrics summary
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis('off')
        metrics_text = f"""Basic Metrics:
PSNR: {basic_metrics.get('psnr', 0):.2f} dB
SSIM: {basic_metrics.get('ssim', 0):.3f}
MSE: {basic_metrics.get('mse', 0):.4f}
MAE: {basic_metrics.get('mae', 0):.4f}
RMSE: {basic_metrics.get('rmse', 0):.4f}

Perceptual Metrics:
LAB Diff: {perceptual_metrics.get('lab_color_diff', 0):.2f}
Texture Diff: {perceptual_metrics.get('texture_lbp_diff', 0):.3f}
Hist Corr R: {perceptual_metrics.get('hist_corr_r', 0):.3f}
Hist Corr G: {perceptual_metrics.get('hist_corr_g', 0):.3f}
Hist Corr B: {perceptual_metrics.get('hist_corr_b', 0):.3f}"""
        
        ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax4.set_title('Metrics Summary', fontsize=14, fontweight='bold')
        
        # Row 2-4: Difference maps
        difference_map_titles = {
            'mse': 'MSE Difference',
            'mae': 'MAE Difference', 
            'ssim': 'SSIM Difference',
            'lab': 'LAB Color Difference',
            'gradient': 'Gradient Difference'
        }
        
        positions = [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0)]
        
        for i, (key, title) in enumerate(difference_map_titles.items()):
            if i < len(positions) and key in difference_maps:
                row, col = positions[i]
                ax = fig.add_subplot(gs[row, col])
                
                im = ax.imshow(difference_maps[key], cmap='hot')
                plt.colorbar(im, ax=ax, fraction=0.046)
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.axis('off')
                
                # Save individual difference map
                diff_map_path = self.output_dir / 'metric_maps' / f'{Path(image_name).stem}_{key}_diff.npy'
                np.save(diff_map_path, difference_maps[key])
        
        # Row 3: Problematic regions overlays
        overlay_titles = ['MSE Regions', 'SSIM Regions', 'LAB Regions', 'Combined Regions']
        region_keys = ['mse', 'ssim', 'lab', 'combined']
        colors = ['Reds', 'Blues', 'Greens', 'Oranges']
        
        for i, (key, title, cmap) in enumerate(zip(region_keys, overlay_titles, colors)):
            if i < 4:
                ax = fig.add_subplot(gs[2, i])
                ax.imshow(real_img)
                
                if key in problematic_regions:
                    ax.imshow(problematic_regions[key], alpha=0.4, cmap=cmap)
                elif key == 'combined':
                    # Create combined mask
                    combined_mask = np.zeros_like(list(problematic_regions.values())[0])
                    for mask in problematic_regions.values():
                        combined_mask = combined_mask | mask
                    ax.imshow(combined_mask, alpha=0.4, cmap=cmap)
                
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.axis('off')
        
        # Row 4: Histograms comparison
        channels = ['R', 'G', 'B']
        colors_hist = ['red', 'green', 'blue']
        
        for i, (channel, color) in enumerate(zip(channels, colors_hist)):
            ax = fig.add_subplot(gs[3, i])
            
            # Calculate histograms
            hist_real = cv2.calcHist([real_img], [i], None, [256], [0, 256])
            hist_rendered = cv2.calcHist([rendered_img], [i], None, [256], [0, 256])
            
            hist_real = hist_real.flatten()
            hist_rendered = hist_rendered.flatten()
            
            ax.plot(hist_real, color=color, alpha=0.7, label='Real', linewidth=2)
            ax.plot(hist_rendered, color=color, alpha=0.7, label='Rendered', linestyle='--', linewidth=2)
            ax.set_title(f'{channel} Channel Histogram', fontsize=12, fontweight='bold')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Row 4, Col 3: Color distribution comparison
        ax = fig.add_subplot(gs[3, 3])
        
        # Create color distribution plot
        real_colors = real_img.reshape(-1, 3).astype(np.float32) / 255.0
        rendered_colors = rendered_img.reshape(-1, 3).astype(np.float32) / 255.0
        
        # Sample for plotting (to avoid memory issues)
        sample_size = min(5000, len(real_colors))
        idx = np.random.choice(len(real_colors), sample_size, replace=False)
        
        ax.scatter(real_colors[idx, 0], real_colors[idx, 1], 
                  c=real_colors[idx], alpha=0.5, s=1, label='Real')
        ax.scatter(rendered_colors[idx, 0], rendered_colors[idx, 1], 
                  c=rendered_colors[idx], alpha=0.5, s=1, marker='x', label='Rendered')
        ax.set_xlabel('Red Channel')
        ax.set_ylabel('Green Channel')
        ax.set_title('Color Distribution (R-G)', fontsize=12, fontweight='bold')
        ax.legend()
        
        # Row 5: Error analysis plots
        if difference_maps:
            # Error magnitude distribution
            ax = fig.add_subplot(gs[4, 0])
            mse_flat = difference_maps['mse'].flatten()
            ax.hist(mse_flat, bins=50, alpha=0.7, color='red', edgecolor='black')
            ax.set_xlabel('MSE Error')
            ax.set_ylabel('Frequency')
            ax.set_title('Error Distribution', fontsize=12, fontweight='bold')
            ax.set_yscale('log')
            
            # Spatial error analysis
            ax = fig.add_subplot(gs[4, 1])
            error_by_row = np.mean(difference_maps['mse'], axis=1)
            error_by_col = np.mean(difference_maps['mse'], axis=0)
            
            ax2 = ax.twinx()
            ax.plot(error_by_row, 'r-', label='Row-wise error')
            ax2.plot(error_by_col, 'b-', label='Column-wise error')
            ax.set_xlabel('Pixel Position')
            ax.set_ylabel('Mean MSE (Row)', color='r')
            ax2.set_ylabel('Mean MSE (Col)', color='b')
            ax.set_title('Spatial Error Pattern', fontsize=12, fontweight='bold')
            
            # Correlation plot
            ax = fig.add_subplot(gs[4, 2])
            if 'ssim' in difference_maps:
                mse_sampled = difference_maps['mse'][::10, ::10].flatten()
                ssim_sampled = difference_maps['ssim'][::10, ::10].flatten()
                
                ax.scatter(mse_sampled, ssim_sampled, alpha=0.5, s=1)
                ax.set_xlabel('MSE Error')
                ax.set_ylabel('SSIM Error')
                ax.set_title('MSE vs SSIM Correlation', fontsize=12, fontweight='bold')
                
                # Add correlation coefficient
                corr = np.corrcoef(mse_sampled, ssim_sampled)[0, 1]
                ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white'))
        
        plt.tight_layout()
        
        # Save the comprehensive visualization
        vis_path = self.output_dir / 'individual_visualizations' / f'{Path(image_name).stem}_comprehensive.png'
        plt.savefig(vis_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Save raw data
        data_to_save = {
            'basic_metrics': basic_metrics,
            'perceptual_metrics': perceptual_metrics,
            'difference_maps_stats': {
                key: {
                    'mean': float(np.mean(diff_map)),
                    'std': float(np.std(diff_map)),
                    'min': float(np.min(diff_map)),
                    'max': float(np.max(diff_map)),
                    'percentiles': {
                        '25': float(np.percentile(diff_map, 25)),
                        '50': float(np.percentile(diff_map, 50)),
                        '75': float(np.percentile(diff_map, 75)),
                        '90': float(np.percentile(diff_map, 90)),
                        '95': float(np.percentile(diff_map, 95))
                    }
                } for key, diff_map in difference_maps.items()
            },
            'problematic_regions_stats': {
                key: {
                    'total_pixels': int(np.sum(mask)),
                    'percentage': float(np.sum(mask) / mask.size * 100),
                    'num_components': int(cv2.connectedComponents(mask.astype(np.uint8))[0] - 1)
                } for key, mask in problematic_regions.items()
            }
        }
        
        # Convert numpy types to JSON-serializable types
        data_to_save = convert_numpy_types(data_to_save)
        
        data_path = self.output_dir / 'data' / f'{Path(image_name).stem}_metrics.json'
        with open(data_path, 'w') as f:
            json.dump(data_to_save, f, indent=2)

    def analyze_single_image_pair(self, image_name: str, save_visualizations: bool = True) -> Dict:
        """Analyze a single image pair comprehensively."""
        try:
            # Load images
            real_img, rendered_img, depth_img = self.load_image_pair(image_name)
            
            if real_img is None or rendered_img is None:
                return {}
            
            # Get pose information and convert to JSON-serializable types
            pose_info = {}
            if self.poses_df is not None:
                pose_row = self.poses_df[self.poses_df['name'] == image_name]
                if not pose_row.empty:
                    pose_info = pose_row.iloc[0].to_dict()
                    # Convert pandas/numpy types to Python types
                    pose_info = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                               for k, v in pose_info.items()}
            
            # Align images
            aligned_rendered, aligned_real, transform_matrix = self.align_images(real_img, rendered_img)
            
            # Compute all metrics
            basic_metrics = self.compute_basic_metrics(aligned_real, aligned_rendered)
            perceptual_metrics = self.compute_perceptual_metrics(aligned_real, aligned_rendered)
            
            # Compute all difference maps
            difference_maps = self.compute_all_difference_maps(aligned_real, aligned_rendered)
            
            # Identify problematic regions for each difference map
            problematic_regions = {}
            positions_3d = {}
            
            for map_name, diff_map in difference_maps.items():
                problematic_regions[map_name] = self.identify_problematic_regions(diff_map, 85)
                positions_3d[map_name] = self.extract_3d_positions_from_regions(
                    problematic_regions[map_name], depth_img, pose_info)
            
            # Combine results
            results = {
                'image_name': image_name,
                'pose_info': pose_info,
                'basic_metrics': basic_metrics,
                'perceptual_metrics': perceptual_metrics,
                'problematic_regions_counts': {
                    name: len(pos_list) for name, pos_list in positions_3d.items()
                },
                'positions_3d': positions_3d,
                'alignment_transform': transform_matrix.tolist() if transform_matrix is not None else None,
                'difference_maps_summary': {
                    name: {
                        'mean': float(np.mean(diff_map)),
                        'max': float(np.max(diff_map)),
                        'std': float(np.std(diff_map))
                    } for name, diff_map in difference_maps.items()
                }
            }
            
            # Convert all numpy types to JSON-serializable types
            results = convert_numpy_types(results)
            
            # Save visualizations
            if save_visualizations:
                self.save_comprehensive_visualization(
                    image_name, aligned_real, aligned_rendered, depth_img,
                    basic_metrics, perceptual_metrics, difference_maps, problematic_regions
                )
            
            return results
            
        except Exception as e:
            print(f"Error analyzing {image_name}: {e}")
            return {}

# Worker function for parallel processing
def analyze_image_worker(args):
    """Worker function for parallel image analysis."""
    import numpy as np
    import pandas as pd
    from pathlib import Path
    
    image_name, data_root, img_width, img_height, fov = args
    
    try:
        # Create a temporary analyzer instance for this worker
        analyzer = ImageDifferenceAnalyzer.__new__(ImageDifferenceAnalyzer)
        analyzer.data_root = Path(data_root)
        analyzer.real_images_dir = analyzer.data_root / "real_images_downscaled"
        analyzer.rendered_images_dir = analyzer.data_root / "rendered_images"  
        analyzer.depth_images_dir = analyzer.data_root / "depth_images"
        analyzer.poses_file = analyzer.data_root / "poses" / "0528_filtered.csv"
        analyzer.img_width = img_width
        analyzer.img_height = img_height
        analyzer.fov = fov
        analyzer.output_dir = Path('./analysis_results')
        
        # Load poses for this worker
        try:
            analyzer.poses_df = pd.read_csv(analyzer.poses_file, skiprows=1)
            analyzer.poses_df.columns = ['name', 'x', 'y', 'z', 'heading', 'pitch', 'roll']
        except:
            analyzer.poses_df = None
        
        # Analyze the image
        result = analyzer.analyze_single_image_pair(image_name, save_visualizations=True)
        
        # Convert all numpy types to JSON-serializable types before returning
        result = convert_numpy_types(result)
        
        return result
        
    except Exception as e:
        print(f"Error in worker for {image_name}: {e}")
        return {}

class OptimizedImageDifferenceAnalyzer(ImageDifferenceAnalyzer):
    """Extended analyzer with parallel processing capabilities."""
    
    def analyze_all_images_parallel(self, max_images: int = None) -> Dict:
        """Analyze all image pairs using parallel processing."""
        if self.poses_df is None:
            self.load_poses()
            
        if self.poses_df is None:
            print("No pose data available")
            return {}
        
        # Limit number of images for testing if specified
        if max_images:
            image_names = self.poses_df['name'].head(max_images).tolist()
        else:
            image_names = self.poses_df['name'].tolist()
        
        print(f"Starting parallel analysis of {len(image_names)} images using {self.n_workers} workers...")
        
        # Prepare arguments for workers
        worker_args = [(name, str(self.data_root), self.img_width, self.img_height, self.fov) 
                      for name in image_names]
        
        all_results = {}
        summary_metrics = {
            'mse': [], 'mae': [], 'psnr': [], 'ssim': [], 'rmse': [],
            'lab_color_diff': [], 'texture_lbp_diff': [],
            'hist_corr_r': [], 'hist_corr_g': [], 'hist_corr_b': [],
            'problematic_regions_count': []
        }
        
        start_time = time.time()
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            future_to_image = {executor.submit(analyze_image_worker, args): args[0] 
                             for args in worker_args}
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_image), total=len(image_names), 
                             desc="Analyzing images"):
                image_name = future_to_image[future]
                try:
                    result = future.result()
                    
                    if result:
                        all_results[image_name] = result
                        
                        # Collect summary metrics
                        if 'basic_metrics' in result:
                            for metric in ['mse', 'mae', 'psnr', 'ssim', 'rmse']:
                                if metric in result['basic_metrics']:
                                    summary_metrics[metric].append(result['basic_metrics'][metric])
                        
                        if 'perceptual_metrics' in result:
                            for metric in ['lab_color_diff', 'texture_lbp_diff', 
                                         'hist_corr_r', 'hist_corr_g', 'hist_corr_b']:
                                if metric in result['perceptual_metrics']:
                                    summary_metrics[metric].append(result['perceptual_metrics'][metric])
                        
                        total_regions = sum(result.get('problematic_regions_counts', {}).values())
                        summary_metrics['problematic_regions_count'].append(total_regions)
                        
                except Exception as e:
                    print(f"Error processing {image_name}: {e}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Parallel processing completed in {processing_time:.2f} seconds")
        print(f"Successfully analyzed {len(all_results)} out of {len(image_names)} images")
        print(f"Average time per image: {processing_time/len(all_results):.2f} seconds")
        
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
        
        # Save results
        self.results = {
            'individual_results': all_results,
            'summary_statistics': summary_stats,
            'total_images_analyzed': len(all_results),
            'processing_time_seconds': processing_time,
            'images_per_second': len(all_results) / processing_time,
            'workers_used': self.n_workers
        }
        
        # Save comprehensive results
        results_to_save = convert_numpy_types(self.results)
        with open(self.output_dir / 'comprehensive_analysis_parallel.json', 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        # Generate summary report
        self.generate_enhanced_summary_report()
        
        return self.results
    
    def generate_enhanced_summary_report(self):
        """Generate an enhanced summary report with all metrics."""
        if not self.results:
            return
        
        # Create enhanced summary plots
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 6, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Comprehensive Dataset Analysis Summary', fontsize=20, fontweight='bold')
        
        summary_stats = self.results['summary_statistics']
        
        # Plot distributions for all basic metrics
        basic_metrics = ['psnr', 'ssim', 'mse', 'mae', 'rmse']
        basic_titles = ['PSNR (dB)', 'SSIM', 'MSE', 'MAE', 'RMSE']
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for i, (metric, title, color) in enumerate(zip(basic_metrics, basic_titles, colors)):
            if metric in summary_stats:
                ax = fig.add_subplot(gs[0, i])
                values = [result['basic_metrics'][metric] 
                         for result in self.results['individual_results'].values()
                         if 'basic_metrics' in result and metric in result['basic_metrics']]
                
                ax.hist(values, bins=30, alpha=0.7, color=color, edgecolor='black')
                ax.axvline(summary_stats[metric]['mean'], color='red', linestyle='--', 
                          label=f"Mean: {summary_stats[metric]['mean']:.3f}")
                ax.set_title(title, fontweight='bold')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Perceptual metrics
        perceptual_metrics = ['lab_color_diff', 'texture_lbp_diff', 'hist_corr_r', 'hist_corr_g', 'hist_corr_b']
        perceptual_titles = ['LAB Color Diff', 'Texture Diff', 'Hist Corr R', 'Hist Corr G', 'Hist Corr B']
        
        for i, (metric, title) in enumerate(zip(perceptual_metrics, perceptual_titles)):
            if metric in summary_stats and i < 5:
                ax = fig.add_subplot(gs[1, i])
                values = [result['perceptual_metrics'][metric] 
                         for result in self.results['individual_results'].values()
                         if 'perceptual_metrics' in result and metric in result['perceptual_metrics']]
                
                ax.hist(values, bins=30, alpha=0.7, edgecolor='black')
                ax.axvline(summary_stats[metric]['mean'], color='red', linestyle='--')
                ax.set_title(title, fontweight='bold')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
        
        # Correlation matrix
        ax = fig.add_subplot(gs[2, :3])
        metrics_for_corr = ['psnr', 'ssim', 'mse', 'mae', 'lab_color_diff']
        
        correlation_data = []
        for metric in metrics_for_corr:
            if metric in ['psnr', 'ssim', 'mse', 'mae']:
                values = [result['basic_metrics'].get(metric, 0) 
                         for result in self.results['individual_results'].values()]
            else:
                values = [result['perceptual_metrics'].get(metric, 0) 
                         for result in self.results['individual_results'].values()]
            correlation_data.append(values)
        
        correlation_matrix = np.corrcoef(correlation_data)
        im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(metrics_for_corr)))
        ax.set_yticks(range(len(metrics_for_corr)))
        ax.set_xticklabels(metrics_for_corr, rotation=45)
        ax.set_yticklabels(metrics_for_corr)
        ax.set_title('Metrics Correlation Matrix', fontweight='bold')
        
        # Add correlation values as text
        for i in range(len(metrics_for_corr)):
            for j in range(len(metrics_for_corr)):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black" if abs(correlation_matrix[i, j]) < 0.5 else "white")
        
        plt.colorbar(im, ax=ax)
        
        # Performance summary
        ax = fig.add_subplot(gs[2, 3:])
        ax.axis('off')
        
        perf_text = f"""Processing Performance:
Total Images: {self.results['total_images_analyzed']}
Processing Time: {self.results['processing_time_seconds']:.1f} seconds
Images/Second: {self.results['images_per_second']:.2f}
Workers Used: {self.results['workers_used']}

Quality Statistics:
Mean PSNR: {summary_stats.get('psnr', {}).get('mean', 0):.2f} dB
Mean SSIM: {summary_stats.get('ssim', {}).get('mean', 0):.3f}
Mean LAB Diff: {summary_stats.get('lab_color_diff', {}).get('mean', 0):.2f}

Error Analysis:
Mean MSE: {summary_stats.get('mse', {}).get('mean', 0):.4f}
Mean MAE: {summary_stats.get('mae', {}).get('mean', 0):.4f}
Avg Problematic Regions: {summary_stats.get('problematic_regions_count', {}).get('mean', 0):.1f}"""
        
        ax.text(0.05, 0.95, perf_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Scatter plots for key relationships
        scatter_configs = [
            ('mse', 'ssim', 'MSE vs SSIM'),
            ('psnr', 'ssim', 'PSNR vs SSIM'),
            ('lab_color_diff', 'texture_lbp_diff', 'Color vs Texture Diff')
        ]
        
        for i, (x_metric, y_metric, title) in enumerate(scatter_configs):
            ax = fig.add_subplot(gs[3, i*2:(i+1)*2])
            
            x_values = []
            y_values = []
            
            for result in self.results['individual_results'].values():
                if x_metric in ['psnr', 'ssim', 'mse', 'mae']:
                    x_val = result.get('basic_metrics', {}).get(x_metric)
                else:
                    x_val = result.get('perceptual_metrics', {}).get(x_metric)
                
                if y_metric in ['psnr', 'ssim', 'mse', 'mae']:
                    y_val = result.get('basic_metrics', {}).get(y_metric)
                else:
                    y_val = result.get('perceptual_metrics', {}).get(y_metric)
                
                if x_val is not None and y_val is not None:
                    x_values.append(x_val)
                    y_values.append(y_val)
            
            if x_values and y_values:
                ax.scatter(x_values, y_values, alpha=0.6, s=10)
                ax.set_xlabel(x_metric.upper())
                ax.set_ylabel(y_metric.upper())
                ax.set_title(title, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                if len(x_values) > 1:
                    z = np.polyfit(x_values, y_values, 1)
                    p = np.poly1d(z)
                    ax.plot(sorted(x_values), p(sorted(x_values)), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'enhanced_summary_report.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print detailed text summary
        print("\n" + "="*100)
        print("COMPREHENSIVE PARALLEL ANALYSIS SUMMARY")
        print("="*100)
        print(f"Total images analyzed: {self.results['total_images_analyzed']}")
        print(f"Processing time: {self.results['processing_time_seconds']:.2f} seconds")
        print(f"Processing speed: {self.results['images_per_second']:.2f} images/second")
        print(f"Workers used: {self.results['workers_used']}")
        
        for metric, stats in summary_stats.items():
            print(f"\n{metric.upper().replace('_', ' ')}:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean:   {stats['mean']:.4f}")
            print(f"  Std:    {stats['std']:.4f}")
            print(f"  Range:  [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  Median: {stats['median']:.4f}")
        
        print("\n" + "="*100)
        print("Analysis completed! Check the following outputs:")
        print(f"  - Individual visualizations: {self.output_dir / 'individual_visualizations'}")
        print(f"  - Metric maps: {self.output_dir / 'metric_maps'}")
        print(f"  - Raw data: {self.output_dir / 'data'}")
        print(f"  - Summary report: {self.output_dir / 'enhanced_summary_report.png'}")
        print("="*100)

# Usage
if __name__ == "__main__":
    # Create optimized analyzer
    analyzer = OptimizedImageDifferenceAnalyzer("./Source/Tools/DGE/data", n_workers=16)  # Adjust n_workers based on your CPU
    
    # Load poses
    analyzer.load_poses()
    
    # For testing, analyze a subset first
    print("Starting parallel analysis...")
    results = analyzer.analyze_all_images_parallel()  # Remove max_images for full dataset
    
    print("Analysis completed!")