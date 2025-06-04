import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
import cv2
from typing import Dict, List, Tuple, Optional
import open3d as o3d
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import warnings
import time
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

class ErrorPointCloudGenerator:
    
    def __init__(self, analysis_results_dir: str = "./analysis_results", 
                 max_points: int = 100000, n_workers: int = None):
        self.analysis_dir = Path(analysis_results_dir)
        self.data_dir = self.analysis_dir / "data"
        self.base_output_dir = Path("./Logs/pointcloud_results")
        self.base_output_dir.mkdir(exist_ok=True)
        
        self.run_timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.base_output_dir / f"run_{self.run_timestamp}"
        self.output_dir.mkdir(exist_ok=True)
        
        self.max_points = max_points
        self.n_workers = n_workers or min(mp.cpu_count(), 4)
        
        self.error_types = ['mse', 'ssim', 'lab', 'gradient']
        self.color_maps = {
            'mse': np.array([1.0, 0.0, 0.0]),
            'ssim': np.array([0.0, 0.0, 1.0]),
            'lab': np.array([0.0, 1.0, 0.0]),
            'gradient': np.array([1.0, 0.5, 0.0])
        }
        
    def load_comprehensive_results(self) -> Dict:
        comprehensive_file = self.analysis_dir / "comprehensive_analysis_parallel.json"
        
        if not comprehensive_file.exists():
            print("Comprehensive analysis file not found!")
            return {}
        
        try:
            with open(comprehensive_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading comprehensive results: {e}")
            return {}
    
    def smart_filter_error_points(self, comprehensive_results: Dict, 
                                target_points_per_type: int = 12500,
                                max_images: int = None) -> Dict[str, List[Dict]]:
        
        print("Applying smart filtering to reduce point count...")
        
        if max_images is not None:
            print(f"Limiting to first {max_images} images for testing")
        
        if 'individual_results' not in comprehensive_results:
            return {}
        
        individual_results = comprehensive_results['individual_results']
        
        if max_images is not None:
            image_names = list(individual_results.keys())[:max_images]
            individual_results = {name: individual_results[name] for name in image_names}
            print(f"Processing {len(individual_results)} images (limited from {len(comprehensive_results['individual_results'])} total)")
        else:
            print(f"Processing all {len(individual_results)} images")
        
        all_error_data = {error_type: [] for error_type in self.error_types}
        
        for image_name, result in tqdm(individual_results.items(), desc="Collecting error data"):
            if 'positions_3d' not in result:
                continue
                
            for error_type in self.error_types:
                if error_type not in result['positions_3d']:
                    continue
                
                for region in result['positions_3d'][error_type]:
                    point_info = {
                        'error_magnitude': region.get('error_magnitude', 0),
                        'region_size': region.get('region_size', 0),
                        'position': region['world_position'],
                        'image_name': image_name,
                        'region_id': region['region_id'],
                        'pixel_center': region['pixel_center'],
                        'mean_depth': region.get('mean_depth', 0),
                        'error_type': error_type,
                        'composite_score': region.get('error_magnitude', 0) * np.log(1 + region.get('region_size', 0))
                    }
                    all_error_data[error_type].append(point_info)
        
        filtered_points = {error_type: [] for error_type in self.error_types}
        
        for error_type in self.error_types:
            error_list = all_error_data[error_type]
            if not error_list:
                continue
            
            print(f"  {error_type.upper()}: {len(error_list)} raw points")
            
            error_list.sort(key=lambda x: x['composite_score'], reverse=True)
            
            min_region_size = max(50, np.percentile([p['region_size'] for p in error_list], 25))
            
            selected_points = []
            for point in error_list:
                if len(selected_points) >= target_points_per_type:
                    break
                if point['region_size'] >= min_region_size:
                    result = individual_results.get(point['image_name'], {})
                    if 'pose_info' in result:
                        point['camera_position'] = [
                            result['pose_info'].get('x', 0),
                            result['pose_info'].get('y', 0), 
                            result['pose_info'].get('z', 0)
                        ]
                    if 'basic_metrics' in result:
                        point['image_psnr'] = result['basic_metrics'].get('psnr', 0)
                        point['image_ssim'] = result['basic_metrics'].get('ssim', 0)
                    
                    selected_points.append(point)
            
            filtered_points[error_type] = selected_points
            print(f"    → Selected {len(selected_points)} top points")
        
        return filtered_points
    
    def light_clustering_by_type(self, error_points: Dict[str, List[Dict]], 
                                cluster_distance: float = 1.0,
                                max_reduction: float = 0.3) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        
        print("Applying light clustering by error type...")
        
        all_final_positions = []
        all_final_colors = []
        all_final_attributes = []
        
        for error_type, points_list in error_points.items():
            if not points_list:
                continue
            
            print(f"  Processing {error_type}: {len(points_list)} points")
            
            positions = np.array([p['position'] for p in points_list])
            error_mags = np.array([p['error_magnitude'] for p in points_list])
            
            if len(error_mags) > 1:
                min_err, max_err = np.min(error_mags), np.max(error_mags)
                if max_err > min_err:
                    intensities = (error_mags - min_err) / (max_err - min_err)
                else:
                    intensities = np.ones_like(error_mags)
            else:
                intensities = np.ones_like(error_mags)
            
            intensities = 0.4 + 0.6 * intensities
            base_color = self.color_maps[error_type]
            colors = base_color[np.newaxis, :] * intensities[:, np.newaxis]
            
            if len(positions) > 1000:
                target_points = max(int(len(positions) * (1 - max_reduction)), len(positions) // 2)
                
                print(f"    Clustering {len(positions)} → ~{target_points} points")
                
                clustering = DBSCAN(eps=cluster_distance, min_samples=2).fit(positions)
                labels = clustering.labels_
                
                clustered_positions = []
                clustered_colors = []
                clustered_attributes = []
                
                noise_mask = labels == -1
                if np.any(noise_mask):
                    for i in np.where(noise_mask)[0]:
                        clustered_positions.append(positions[i])
                        clustered_colors.append(colors[i])
                        clustered_attributes.append(points_list[i])
                
                unique_labels = set(labels) - {-1}
                for label in unique_labels:
                    cluster_mask = labels == label
                    cluster_indices = np.where(cluster_mask)[0]
                    
                    if len(cluster_indices) == 1:
                        idx = cluster_indices[0]
                        clustered_positions.append(positions[idx])
                        clustered_colors.append(colors[idx])
                        clustered_attributes.append(points_list[idx])
                    else:
                        cluster_positions = positions[cluster_mask]
                        cluster_colors = colors[cluster_mask]
                        cluster_attrs = [points_list[i] for i in cluster_indices]
                        
                        centroid = np.mean(cluster_positions, axis=0)
                        avg_color = np.mean(cluster_colors, axis=0)
                        
                        combined_attr = {
                            'position': centroid.tolist(),
                            'error_magnitude': np.mean([attr['error_magnitude'] for attr in cluster_attrs]),
                            'region_size': sum([attr['region_size'] for attr in cluster_attrs]),
                            'cluster_size': len(cluster_attrs),
                            'error_type': error_type,
                            'images_involved': list(set([attr['image_name'] for attr in cluster_attrs]))
                        }
                        
                        clustered_positions.append(centroid)
                        clustered_colors.append(avg_color)
                        clustered_attributes.append(combined_attr)
                
                print(f"    → {len(clustered_positions)} points after clustering")
                
                all_final_positions.extend(clustered_positions)
                all_final_colors.extend(clustered_colors)
                all_final_attributes.extend(clustered_attributes)
            
            else:
                print(f"    → Keeping all {len(positions)} points (no clustering)")
                all_final_positions.extend(positions)
                all_final_colors.extend(colors)
                all_final_attributes.extend(points_list)
        
        final_positions = np.array(all_final_positions)
        final_colors = np.array(all_final_colors)
        
        error_type_counts = {}
        for attr in all_final_attributes:
            error_type = attr.get('error_type', 'unknown')
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        print(f"Final distribution after clustering:")
        for error_type, count in error_type_counts.items():
            print(f"  {error_type}: {count:,} points")
        
        return final_positions, final_colors, all_final_attributes
    
    def no_clustering_processing(self, error_points: Dict[str, List[Dict]]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        
        print("Processing points without clustering...")
        
        all_positions = []
        all_colors = []
        all_attributes = []
        
        for error_type, points_list in error_points.items():
            if not points_list:
                continue
            
            print(f"  {error_type}: {len(points_list)} points")
            
            positions = np.array([p['position'] for p in points_list])
            error_mags = np.array([p['error_magnitude'] for p in points_list])
            
            if len(error_mags) > 1:
                min_err, max_err = np.min(error_mags), np.max(error_mags)
                if max_err > min_err:
                    intensities = (error_mags - min_err) / (max_err - min_err)
                else:
                    intensities = np.ones_like(error_mags)
            else:
                intensities = np.ones_like(error_mags)
            
            intensities = 0.4 + 0.6 * intensities
            base_color = self.color_maps[error_type]
            colors = base_color[np.newaxis, :] * intensities[:, np.newaxis]
            
            all_positions.extend(positions)
            all_colors.extend(colors)
            all_attributes.extend(points_list)
        
        return np.array(all_positions), np.array(all_colors), all_attributes
    
    def save_with_ue_conversion(self, points: np.ndarray, colors: np.ndarray, 
                              attributes: List[Dict], config_info: str = "", max_images: int = None):
        
        print(f"Saving to: {self.output_dir}")
        
        (self.output_dir / "ply_files").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        
        base_filename = f"error_pointcloud{config_info}"
        if max_images is not None:
            base_filename += f"_test{max_images}imgs"
        
        print("Saving standard PLY file...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        ply_path = self.output_dir / "ply_files" / f"{base_filename}.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        
        print("Converting to UE coordinate system...")
        ue_points = points.copy() * 100.0
        
        ue_points_converted = np.zeros_like(ue_points)
        ue_points_converted[:, 0] = ue_points[:, 0] 
        ue_points_converted[:, 1] = -ue_points[:, 1]
        ue_points_converted[:, 2] = ue_points[:, 2]         
        
        pcd_ue = o3d.geometry.PointCloud()
        pcd_ue.points = o3d.utility.Vector3dVector(ue_points_converted)
        pcd_ue.colors = o3d.utility.Vector3dVector(colors)
        
        ue_ply_path = self.output_dir / "ply_files" / f"{base_filename}_UE.ply"
        o3d.io.write_point_cloud(str(ue_ply_path), pcd_ue)
        
        print("Saving CSV data...")
        data_rows = []
        for i, (point, ue_point, color, attr) in enumerate(zip(points, ue_points_converted, colors, attributes)):
            row = {
                'x_m': float(point[0]), 'y_m': float(point[1]), 'z_m': float(point[2]),
                'x_ue_cm': float(ue_point[0]), 'y_ue_cm': float(ue_point[1]), 'z_ue_cm': float(ue_point[2]),
                'color_r': float(color[0]), 'color_g': float(color[1]), 'color_b': float(color[2]),
                'error_magnitude': float(attr.get('error_magnitude', 0)),
                'error_type': attr.get('error_type', 'unknown'),
                'region_size': int(attr.get('region_size', 0)),
                'cluster_size': int(attr.get('cluster_size', 1)),
                'image_name': attr.get('image_name', 'unknown')
            }
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        csv_path = self.output_dir / "data" / f"{base_filename}.csv"
        df.to_csv(csv_path, index=False)
        
        print("Saving summary...")
        summary = {
            'generation_info': {
                'timestamp': self.run_timestamp,
                'total_points': len(points),
                'config': config_info.replace("_", " ").strip(),
                'max_images_processed': max_images,
                'images_were_limited': max_images is not None,
                'coordinate_systems': {
                    'standard': 'Right-handed, meters (X=East, Y=North, Z=Up)',
                    'unreal_engine': 'Left-handed, centimeters (X=Forward, Y=Right, Z=Up)'
                }
            },
            'files': {
                'standard_ply': f"ply_files/{base_filename}.ply",
                'unreal_ply': f"ply_files/{base_filename}_UE.ply", 
                'csv_data': f"data/{base_filename}.csv"
            },
            'error_distribution': {},
            'spatial_bounds_meters': {
                'x_range': [float(points[:, 0].min()), float(points[:, 0].max())],
                'y_range': [float(points[:, 1].min()), float(points[:, 1].max())],
                'z_range': [float(points[:, 2].min()), float(points[:, 2].max())]
            },
            'spatial_bounds_ue_cm': {
                'x_range': [float(ue_points_converted[:, 0].min()), float(ue_points_converted[:, 0].max())],
                'y_range': [float(ue_points_converted[:, 1].min()), float(ue_points_converted[:, 1].max())],
                'z_range': [float(ue_points_converted[:, 2].min()), float(ue_points_converted[:, 2].max())]
            },
            'error_stats': {
                'mean': float(np.mean([a.get('error_magnitude', 0) for a in attributes])),
                'std': float(np.std([a.get('error_magnitude', 0) for a in attributes])),
                'max': float(np.max([a.get('error_magnitude', 0) for a in attributes]))
            }
        }
        
        error_type_counts = {}
        for attr in attributes:
            error_type = attr.get('error_type', 'unknown')
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        for error_type, count in error_type_counts.items():
            percentage = count / len(attributes) * 100
            summary['error_distribution'][error_type] = {
                'count': count,
                'percentage': round(percentage, 1)
            }
        
        summary_path = self.output_dir / f"run_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        image_limit_text = f" (LIMITED TO {max_images} IMAGES FOR TESTING)" if max_images is not None else ""
        config_clean = config_info.replace('_', ' ').strip()
        images_text = f"Images processed: {max_images} (test mode)" if max_images is not None else "Images processed: All available"
        
        readme_content = f"""# Error Point Cloud Analysis Run{image_limit_text}
Generated: {self.run_timestamp}
Configuration: {config_clean}
{images_text}

## Files in this directory:

### PLY Files (ply_files/)
- {base_filename}.ply - Standard point cloud (right-handed, meters)
- {base_filename}_UE.ply - Unreal Engine compatible (left-handed, centimeters)

### Data (data/)
- {base_filename}.csv - Point data with both coordinate systems

### Summary
- run_summary.json - Complete analysis summary
- README.md - This file

## Point Cloud Statistics:
- Total Points: {len(points):,}
- Error Types Distribution:
"""
        
        for error_type, count in error_type_counts.items():
            percentage = count / len(attributes) * 100
            readme_content += f"  - {error_type}: {count:,} points ({percentage:.1f}%)\n"
        
        spatial_info = f"""
## Spatial Extent:
### Standard Coordinates (meters):
- X: [{points[:, 0].min():.1f}, {points[:, 0].max():.1f}]
- Y: [{points[:, 1].min():.1f}, {points[:, 1].max():.1f}]
- Z: [{points[:, 2].min():.1f}, {points[:, 2].max():.1f}]

### UE Coordinates (centimeters):
- X (Forward): [{ue_points_converted[:, 0].min():.1f}, {ue_points_converted[:, 0].max():.1f}]
- Y (Right): [{ue_points_converted[:, 1].min():.1f}, {ue_points_converted[:, 1].max():.1f}]
- Z (Up): [{ue_points_converted[:, 2].min():.1f}, {ue_points_converted[:, 2].max():.1f}]

## Usage:
1. For standard 3D software: Use {base_filename}.ply
2. For Unreal Engine: Use {base_filename}_UE.ply
3. For data analysis: Use {base_filename}.csv

## Color Coding:
- Red = MSE errors (pixel intensity differences)
- Blue = SSIM errors (structural differences)  
- Green = LAB errors (color differences)
- Orange = Gradient errors (edge/texture differences)
"""
        
        readme_content += spatial_info
        
        if max_images is not None:
            readme_content += f"\n## Testing Note:\nThis run was generated using only the first {max_images} images for testing purposes."
        
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        return {
            'ply': ply_path,
            'ue_ply': ue_ply_path,
            'csv': csv_path,
            'summary': summary_path,
            'readme': readme_path,
            'dataframe': df,
            'run_directory': self.output_dir
        }
        
    def create_run_visualizations(self, points: np.ndarray, colors: np.ndarray, 
                                attributes: List[Dict], df: pd.DataFrame, max_images: int = None):
        
        print("Creating visualizations...")
        
        vis_dir = self.output_dir / "visualizations"
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        title_suffix = f" (Test: {max_images} images)" if max_images is not None else ""
        fig.suptitle(f'Error Point Cloud Analysis - Run {self.run_timestamp}{title_suffix}', fontsize=16, fontweight='bold')
        
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        if len(points) > 5000:
            indices = np.random.choice(len(points), 5000, replace=False)
            sample_points = points[indices]
            sample_colors = colors[indices]
        else:
            sample_points = points
            sample_colors = colors
            
        ax1.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], 
                   c=sample_colors, s=2, alpha=0.7)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Error Point Cloud')
        
        error_mags = [attr.get('error_magnitude', 0) for attr in attributes]
        scatter = axes[0,1].scatter(points[:, 0], points[:, 1], c=error_mags, 
                                   cmap='hot', s=2, alpha=0.7)
        axes[0,1].set_xlabel('X (m)')
        axes[0,1].set_ylabel('Y (m)')
        axes[0,1].set_title('Top-down View (Error Magnitude)')
        axes[0,1].set_aspect('equal')
        plt.colorbar(scatter, ax=axes[0,1], label='Error Magnitude')
        
        axes[0,2].hist(error_mags, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
        axes[0,2].set_xlabel('Error Magnitude')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].set_title('Error Magnitude Distribution')
        axes[0,2].set_yscale('log')
        axes[0,2].grid(True, alpha=0.3)
        
        error_types = [attr.get('error_type', 'unknown') for attr in attributes]
        type_counts = pd.Series(error_types).value_counts()
        bars = type_counts.plot(kind='bar', ax=axes[1,0], color=['red', 'blue', 'green', 'orange'])
        axes[1,0].set_title('Points by Error Type')
        axes[1,0].set_ylabel('Count')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        for i, bar in enumerate(bars.patches):
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 50,
                          f'{int(height):,}', ha='center', va='bottom')
        
        axes[1,1].hist2d(points[:, 0], points[:, 1], bins=50, cmap='hot')
        axes[1,1].set_xlabel('X (m)')
        axes[1,1].set_ylabel('Y (m)')
        axes[1,1].set_title('Spatial Error Density')
        
        axes[1,2].axis('off')
        images_info = f"Images: {max_images} (TEST MODE)" if max_images is not None else "Images: All available"
        summary_text = f"""Run Summary:
Timestamp: {self.run_timestamp}
Total Points: {len(points):,}
{images_info}

Error Statistics:
Mean: {np.mean(error_mags):.4f}
Std:  {np.std(error_mags):.4f}
Max:  {np.max(error_mags):.4f}

Spatial Extent (meters):
X: [{points[:, 0].min():.1f}, {points[:, 0].max():.1f}]
Y: [{points[:, 1].min():.1f}, {points[:, 1].max():.1f}]  
Z: [{points[:, 2].min():.1f}, {points[:, 2].max():.1f}]

Error Type Distribution:
{type_counts.to_string()}"""
        
        axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes,
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(vis_dir / 'error_analysis_overview.png', 
                   dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors_pie = ['red', 'blue', 'green', 'orange'][:len(type_counts)]
        wedges, texts, autotexts = ax.pie(type_counts.values, labels=type_counts.index, 
                                         colors=colors_pie, autopct='%1.1f%%', startangle=90)
        title_suffix = f' - Test ({max_images} images)' if max_images is not None else ''
        ax.set_title(f'Error Type Distribution - {len(points):,} Points{title_suffix}', fontsize=14, fontweight='bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.savefig(vis_dir / 'error_type_distribution.png', 
                   dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Visualizations saved to: {vis_dir}")
        
        return {
            'overview': vis_dir / 'error_analysis_overview.png',
            'distribution': vis_dir / 'error_type_distribution.png'
        }
    
    def generate_better_point_cloud(self, max_points_per_type: int = None,
                                   clustering_mode: str = "light",
                                   cluster_distance: float = 1.0,
                                   max_images: int = None):
        
        start_time = time.time()
        
        print("="*80)
        print("ERROR POINT CLOUD GENERATION")
        print("="*80)
        print(f"Run directory: {self.output_dir}")
        
        if max_points_per_type is None:
            max_points_per_type = self.max_points // len(self.error_types)
        
        print(f"Target: {max_points_per_type} points per error type (max {self.max_points} total)")
        print(f"Clustering mode: {clustering_mode}")
        print(f"Cluster distance: {cluster_distance}m")
        
        if max_images is not None:
            print(f"TEST MODE: Processing only first {max_images} images")
        else:
            print("FULL MODE: Processing all available images")
        
        print("Loading analysis results...")
        comprehensive_results = self.load_comprehensive_results()
        
        if not comprehensive_results:
            print("No comprehensive results found!")
            return None
        
        filter_start = time.time()
        error_points = self.smart_filter_error_points(
            comprehensive_results, max_points_per_type, max_images=max_images)
        filter_time = time.time() - filter_start
        
        if not any(error_points.values()):
            print("No error points found after filtering!")
            return None
        
        total_filtered = sum(len(points) for points in error_points.values())
        print(f"Filtered to {total_filtered} high-quality points in {filter_time:.1f}s")
        
        process_start = time.time()
        
        if clustering_mode == "none":
            points, colors, attributes = self.no_clustering_processing(error_points)
        elif clustering_mode == "light":
            points, colors, attributes = self.light_clustering_by_type(
                error_points, cluster_distance, max_reduction=0.2)
        elif clustering_mode == "medium":
            points, colors, attributes = self.light_clustering_by_type(
                error_points, cluster_distance, max_reduction=0.4)
        else:
            print(f"Unknown clustering mode: {clustering_mode}")
            return None
        
        process_time = time.time() - process_start
        
        if len(points) == 0:
            print("No points in final point cloud!")
            return None
        
        save_start = time.time()
        config_info = f"_{clustering_mode}_clustering" if clustering_mode != "none" else "_no_clustering"
        output_files = self.save_with_ue_conversion(points, colors, attributes, config_info, max_images)
        
        vis_files = self.create_run_visualizations(points, colors, attributes, output_files['dataframe'], max_images)
        output_files['visualizations'] = vis_files
        
        save_time = time.time() - save_start
        
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("GENERATION COMPLETE")
        print("="*80)
        print(f"Total processing time: {total_time:.1f}s")
        print(f"  - Filtering: {filter_time:.1f}s")
        print(f"  - Processing: {process_time:.1f}s") 
        print(f"  - Saving: {save_time:.1f}s")
        
        if max_images is not None:
            print(f"TEST RUN: Used {max_images} images only")
        
        print(f"Final point cloud: {len(points):,} points")
        
        error_type_counts = {}
        for attr in attributes:
            error_type = attr.get('error_type', 'unknown')
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        for error_type, count in error_type_counts.items():
            percentage = count / len(attributes) * 100
            print(f"  {error_type}: {count:,} points ({percentage:.1f}%)")
        
        print(f"Spatial extent (standard):")
        print(f"  X: [{points[:, 0].min():.1f}, {points[:, 0].max():.1f}] m")
        print(f"  Y: [{points[:, 1].min():.1f}, {points[:, 1].max():.1f}] m")
        print(f"  Z: [{points[:, 2].min():.1f}, {points[:, 2].max():.1f}] m")
        
        print(f"All files saved to: {output_files['run_directory']}")
        print(f"PLY files: {output_files['run_directory']}/ply_files/")
        print(f"Data: {output_files['run_directory']}/data/")
        print(f"Visualizations: {output_files['run_directory']}/visualizations/")
        print(f"Run summary: {output_files['summary'].name}")
        print(f"README: {output_files['readme'].name}")
        print("="*80)
        
        return {
            'points': points,
            'colors': colors,
            'attributes': attributes,
            'output_files': output_files,
            'processing_time': total_time,
            'run_directory': output_files['run_directory'],
            'max_images_used': max_images
        }

# Convenience functions with max_images parameter
def generate_no_clustering(max_points: int = 100000, max_images: int = None):
    generator = ErrorPointCloudGenerator(max_points=max_points)
    return generator.generate_better_point_cloud(clustering_mode="none", max_images=max_images)

def generate_light_clustering(max_points: int = 100000, cluster_distance: float = 1.0, max_images: int = None):
    generator = ErrorPointCloudGenerator(max_points=max_points)
    return generator.generate_better_point_cloud(
        clustering_mode="light",
        cluster_distance=cluster_distance,
        max_images=max_images
    )

def generate_medium_clustering(max_points: int = 100000, cluster_distance: float = 1.5, max_images: int = None):
    generator = ErrorPointCloudGenerator(max_points=max_points)
    return generator.generate_better_point_cloud(
        clustering_mode="medium", 
        cluster_distance=cluster_distance,
        max_images=max_images
    )

if __name__ == "__main__":
    print("ERROR POINT CLOUD GENERATOR")
    print("===========================")
    print("Now supports limiting the number of images for testing!")
    print()
    
    results = generate_light_clustering(max_points=200000, max_images=1)
    
    if results:
        print(f"Success! Generated {len(results['points']):,} points in {results['processing_time']:.1f}s")
        print(f"Run directory: {results['run_directory']}")
        print(f"UE PLY: {results['output_files']['ue_ply']}")
        print(f"Standard PLY: {results['output_files']['ply']}")
    else:
        print("Generation failed!")
