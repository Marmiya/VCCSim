#!/usr/bin/env python3
"""
UE Error Point Cloud Generator - Enhanced Version

This script generates point clouds from image difference analysis results,
creating PLY files for both Unreal Engine (left-handed) and standard right-handed coordinate systems.

FEATURES:
- Automatic detection of newest analysis results directory
- Both UE (left-handed, cm) and standard (right-handed, m) coordinate systems
- Image range selection (specific images or ranges)
- Comprehensive filtering options

USAGE:
1. Modify filter settings and image selection as needed
2. Run: python ue_error_pointcloud_generator.py

OUTPUT FILES:
- error_points_ue_TIMESTAMP.ply - Error points in Unreal Engine coordinates (cm)
- error_points_rh_TIMESTAMP.ply - Error points in right-handed coordinates (m)
- camera_poses_ue_TIMESTAMP.ply - Camera positions in UE system (optional)
- camera_poses_rh_TIMESTAMP.ply - Camera positions in right-handed system (optional)
- pointcloud_summary.txt - Statistics and analysis report

COORDINATE SYSTEMS:
- UE (left-handed): X=Forward, Y=Right, Z=Up (centimeters)
- Standard (right-handed): X=Forward, Y=Left, Z=Up (meters)

DEPTH FORMAT:
- Depth images are 16-bit PNG files with direct centimeter values (no processing required)
- Values range from 0 to 65535 cm (0 to 655.35 meters)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import glob
import re

# ============================================================================
# CONFIGURATION SETTINGS - Modify these values as needed
# ============================================================================

# Analysis results directory settings
ANALYSIS_RESULTS_BASE_DIR = "./Logs/analysis_results"  # Base directory for analysis results
AUTO_SELECT_NEWEST = True  # Automatically select newest analysis directory
MANUAL_ANALYSIS_DIR = None  # Set this if you want to specify a particular directory
                           # Example: "./Logs/analysis_results/20250611_173333"

# Image selection settings
IMAGE_SELECTION_MODE = "single"  # Options:
                              # "all" - process all images
                              # "range" - process a range of images (use IMAGE_START and IMAGE_END)
                              # "specific" - process specific images (use SPECIFIC_IMAGES)
                              # "single" - process a single image (use SINGLE_IMAGE)

IMAGE_START = 0           # Start index for range mode (0-based)
IMAGE_END = 10            # End index for range mode (exclusive, so 0-10 means images 0-9)
SPECIFIC_IMAGES = [2, 5, 8, 12]  # List of specific image indices for specific mode
SINGLE_IMAGE = 1          # Single image index for single mode

# Filter settings
FILTER_ERROR_TYPES = None  # Options:
                          # None - include all error types
                          # ['mse'] - only MSE errors
                          # ['mse', 'lab'] - MSE and LAB color errors
                          # ['mse', 'mae', 'lab', 'gradient'] - multiple types

MIN_REGION_SIZE = 1      # Minimum region size in pixels (larger = fewer, bigger errors)
MAX_POINTS = None         # Options:
                          # None - include all points
                          # 1000 - limit to top 1000 highest intensity errors
                          # 5000 - limit to top 5000 points

# Output settings
GENERATE_CAMERA_POSES = True  # Generate separate PLY files with camera positions
GENERATE_BOTH_COORDINATES = True  # Generate both UE and right-handed coordinate systems

def find_newest_analysis_directory(base_dir: str) -> Optional[str]:
    """Find the newest analysis results directory based on timestamp in directory name."""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Base analysis directory not found: {base_dir}")
        return None
    
    # Look for directories with timestamp pattern (YYYYMMDD_HHMMSS)
    timestamp_pattern = re.compile(r'(\d{8}_\d{6})')
    
    analysis_dirs = []
    for item in base_path.iterdir():
        if item.is_dir():
            match = timestamp_pattern.search(item.name)
            if match:
                timestamp_str = match.group(1)
                try:
                    # Parse timestamp to ensure it's valid
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    analysis_dirs.append((timestamp, item))
                except ValueError:
                    continue
    
    if not analysis_dirs:
        print(f"No analysis directories with timestamp pattern found in {base_dir}")
        return None
    
    # Sort by timestamp and get the newest
    analysis_dirs.sort(key=lambda x: x[0], reverse=True)
    newest_dir = analysis_dirs[0][1]
    
    print(f"Found {len(analysis_dirs)} analysis directories")
    print(f"Selected newest directory: {newest_dir}")
    return str(newest_dir)

class EnhancedErrorPointCloudGenerator:
    """Enhanced point cloud generator with multiple coordinate systems and image selection."""
    
    def __init__(self, analysis_results_dir: str):
        self.analysis_dir = Path(analysis_results_dir)
        self.data_dir = self.analysis_dir / 'data'
        self.output_dir = self.analysis_dir / 'pointclouds'
        self.output_dir.mkdir(exist_ok=True)
        
        # Color schemes for different error types
        self.error_colors = {
            'mse': [255, 0, 0],      # Red
            'mae': [255, 128, 0],    # Orange  
            'lab': [255, 255, 0],    # Yellow
            'gradient': [0, 255, 0], # Green
            'texture': [0, 255, 255], # Cyan
            'combined': [255, 0, 255] # Magenta
        }
        
        print(f"Initialized Enhanced UE Error Point Cloud Generator")
        print(f"Analysis directory: {self.analysis_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Depth format: 16-bit PNG with direct centimeter values")
    
    def load_analysis_results(self, image_selection_mode: str = "all", 
                            image_start: int = 0, image_end: int = None,
                            specific_images: List[int] = None, 
                            single_image: int = None) -> List[Dict]:
        """Load analysis result files with image selection options."""
        results = []
        
        if not self.data_dir.exists():
            print(f"Error: Data directory not found: {self.data_dir}")
            return results
        
        json_files = sorted(list(self.data_dir.glob('*_data.json')))
        print(f"Found {len(json_files)} total analysis result files")
        
        # Apply image selection filter
        selected_files = []
        
        if image_selection_mode == "all":
            selected_files = json_files
            print("Selected: All images")
            
        elif image_selection_mode == "range":
            end_idx = image_end if image_end is not None else len(json_files)
            end_idx = min(end_idx, len(json_files))  # Clamp to available files
            start_idx = max(0, min(image_start, len(json_files) - 1))  # Clamp start
            
            selected_files = json_files[start_idx:end_idx]
            print(f"Selected: Range [{start_idx}:{end_idx}] = {len(selected_files)} images")
            
        elif image_selection_mode == "specific":
            if specific_images:
                for idx in specific_images:
                    if 0 <= idx < len(json_files):
                        selected_files.append(json_files[idx])
                print(f"Selected: Specific images {specific_images} = {len(selected_files)} images")
            else:
                print("Warning: No specific images provided, using all images")
                selected_files = json_files
                
        elif image_selection_mode == "single":
            if single_image is not None and 0 <= single_image < len(json_files):
                selected_files = [json_files[single_image]]
                print(f"Selected: Single image {single_image}")
            else:
                print(f"Warning: Invalid single image index {single_image}, using first image")
                selected_files = [json_files[0]] if json_files else []
                
        else:
            print(f"Warning: Unknown selection mode '{image_selection_mode}', using all images")
            selected_files = json_files
        
        # Load selected files
        for json_file in selected_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if data:  # Only add non-empty results
                        results.append(data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        print(f"Successfully loaded {len(results)} analysis results")
        return results
    
    def extract_error_points(self, results: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Extract error points for both UE and right-handed coordinate systems."""
        ue_points = []
        rh_points = []
        
        for result in results:
            image_name = result.get('image_name', 'unknown')
            pose_info = result.get('pose_info', {})
            positions_3d = result.get('positions_3d', {})
            basic_metrics = result.get('basic_metrics', {})
            perceptual_metrics = result.get('perceptual_metrics', {})
            
            # Camera pose information (in UE coordinates)
            cam_x = pose_info.get('x', 0)
            cam_y = pose_info.get('y', 0) 
            cam_z = pose_info.get('z', 0)
            cam_yaw = pose_info.get('yaw', 0)
            cam_pitch = pose_info.get('pitch', 0)
            cam_roll = pose_info.get('roll', 0)
            
            # Process each error type
            for error_type, regions in positions_3d.items():
                if not regions:
                    continue
                    
                for region in regions:
                    world_pos = region.get('world_position', [0, 0, 0])
                    region_size = region.get('region_size', 1)
                    mean_depth = region.get('mean_depth', 1.0)
                    region_id = region.get('region_id', 0)
                    
                    # Get error intensity from metrics
                    error_intensity = self._get_error_intensity(error_type, basic_metrics, perceptual_metrics)
                    
                    # UE point (coordinates already in UE system - centimeters)
                    ue_point = {
                        'x': float(world_pos[0]),
                        'y': float(world_pos[1]),
                        'z': float(world_pos[2]), 
                        'error_type': error_type,
                        'error_intensity': error_intensity,
                        'region_size': region_size,
                        'depth': mean_depth,
                        'image_name': image_name,
                        'region_id': region_id,
                        'camera_pose': [cam_x, cam_y, cam_z, cam_yaw, cam_pitch, cam_roll]
                    }
                    ue_points.append(ue_point)
                    
                    # Right-handed point (convert from UE coordinates)
                    # UE: X=Forward, Y=Right, Z=Up (left-handed, cm)
                    # RH: X=Forward, Y=Left, Z=Up (right-handed, m)
                    rh_point = {
                        'x': float(world_pos[0]) / 100.0,      # Forward (cm to m)
                        'y': -float(world_pos[1]) / 100.0,     # Right -> Left (flip Y, cm to m)
                        'z': float(world_pos[2]) / 100.0,      # Up (cm to m)
                        'error_type': error_type,
                        'error_intensity': error_intensity,
                        'region_size': region_size,
                        'depth': mean_depth / 100.0,           # cm to m
                        'image_name': image_name,
                        'region_id': region_id,
                        'camera_pose': [cam_x / 100.0, -cam_y / 100.0, cam_z / 100.0, 
                                      cam_yaw, cam_pitch, cam_roll]  # Convert pose to RH (m)
                    }
                    rh_points.append(rh_point)
        
        print(f"Extracted {len(ue_points)} UE points and {len(rh_points)} RH points")
        return ue_points, rh_points
    
    def _get_error_intensity(self, error_type: str, basic_metrics: Dict, perceptual_metrics: Dict) -> float:
        """Get normalized error intensity for a given error type."""
        if error_type == 'mse':
            return min(basic_metrics.get('mse', 0) * 1000, 1.0)  # Scale MSE
        elif error_type == 'mae':
            return min(basic_metrics.get('mae', 0) * 10, 1.0)   # Scale MAE
        elif error_type == 'lab':
            return min(perceptual_metrics.get('lab_color_diff', 0) / 100, 1.0)  # Scale LAB
        elif error_type == 'gradient':
            return min(basic_metrics.get('rmse', 0) * 5, 1.0)   # Use RMSE for gradient
        else:
            return 0.5  # Default intensity
    
    def _get_point_color(self, error_type: str, error_intensity: float) -> List[int]:
        """Get RGB color based on error type and intensity."""
        base_color = self.error_colors.get(error_type, [128, 128, 128])
        
        # Modulate intensity (0.3 to 1.0 range for visibility)
        intensity = 0.3 + 0.7 * min(error_intensity, 1.0)
        
        return [int(c * intensity) for c in base_color]
    
    def write_ply_file(self, points: List[Dict], filename: str, coordinate_system: str, units: str):
        """Write points to PLY file with colors and properties."""
        if not points:
            print(f"No points to write for {filename}")
            return
        
        ply_path = self.output_dir / filename
        
        # Group points by error type for statistics
        error_stats = {}
        for point in points:
            error_type = point['error_type']
            if error_type not in error_stats:
                error_stats[error_type] = []
            error_stats[error_type].append(point)
        
        # Write PLY header
        with open(ply_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"comment Error Point Cloud - {coordinate_system} coordinate system\n")
            
            if coordinate_system.lower() == "unreal engine":
                f.write(f"comment UE Coordinates: X=Forward, Y=Right, Z=Up (left-handed)\n")
            else:
                f.write(f"comment RH Coordinates: X=Forward, Y=Left, Z=Up (right-handed)\n")
                
            f.write(f"comment Units: {units}\n")
            f.write(f"comment Depth format: 16-bit PNG with direct centimeter values\n")
            f.write(f"comment Generated: {datetime.now().isoformat()}\n")
            f.write(f"comment Total points: {len(points)}\n")
            
            # Write error type statistics
            for error_type, type_points in error_stats.items():
                f.write(f"comment {error_type}: {len(type_points)} points\n")
            
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n") 
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("property float error_intensity\n")
            f.write("property int region_size\n")
            f.write("property float depth\n")
            f.write("end_header\n")
            
            # Write vertex data
            for point in points:
                color = self._get_point_color(point['error_type'], point['error_intensity'])
                
                f.write(f"{point['x']:.6f} {point['y']:.6f} {point['z']:.6f} ")
                f.write(f"{color[0]} {color[1]} {color[2]} ")
                f.write(f"{point['error_intensity']:.6f} ")
                f.write(f"{point['region_size']} ")
                f.write(f"{point['depth']:.6f}\n")
        
        print(f"Saved {len(points)} points to {ply_path}")
    
    def write_camera_poses_ply(self, points: List[Dict], filename: str, coordinate_system: str, units: str):
        """Write camera poses as separate PLY file."""
        if not points:
            return
        
        # Extract unique camera poses
        unique_poses = {}
        for point in points:
            image_name = point['image_name']
            if image_name not in unique_poses:
                pose = point['camera_pose']
                unique_poses[image_name] = {
                    'x': pose[0], 'y': pose[1], 'z': pose[2],
                    'yaw': pose[3], 'pitch': pose[4], 'roll': pose[5]
                }
        
        ply_path = self.output_dir / filename
        
        with open(ply_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"comment Camera Poses - {coordinate_system} coordinate system\n")
            
            if coordinate_system.lower() == "unreal engine":
                f.write(f"comment UE Coordinates: X=Forward, Y=Right, Z=Up (left-handed)\n")
            else:
                f.write(f"comment RH Coordinates: X=Forward, Y=Left, Z=Up (right-handed)\n")
                
            f.write(f"comment Units: {units}, degrees\n")
            f.write(f"comment Depth format: 16-bit PNG with direct centimeter values\n")
            f.write(f"comment Generated: {datetime.now().isoformat()}\n")
            f.write(f"element vertex {len(unique_poses)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("property float yaw\n")
            f.write("property float pitch\n")
            f.write("property float roll\n")
            f.write("end_header\n")
            
            # Write camera positions (blue color)
            for image_name, pose in unique_poses.items():
                f.write(f"{pose['x']:.6f} {pose['y']:.6f} {pose['z']:.6f} ")
                f.write(f"0 0 255 ")  # Blue for cameras
                f.write(f"{pose['yaw']:.3f} {pose['pitch']:.3f} {pose['roll']:.3f}\n")
        
        print(f"Saved {len(unique_poses)} camera poses to {ply_path}")
    
    def write_summary_report(self, ue_points: List[Dict], rh_points: List[Dict]):
        """Write a summary report of the point cloud generation."""
        report_path = self.output_dir / 'pointcloud_summary.txt'
        
        # Analyze point distributions
        ue_stats = self._analyze_points(ue_points, "Unreal Engine (Left-handed, cm)")
        rh_stats = self._analyze_points(rh_points, "Standard Right-handed (m)")
        
        with open(report_path, 'w') as f:
            f.write("ENHANCED UE ERROR POINT CLOUD GENERATION SUMMARY\n")
            f.write("=" * 55 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Analysis directory: {self.analysis_dir}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            f.write("COORDINATE SYSTEMS:\n")
            f.write("-" * 20 + "\n")
            f.write("1. Unreal Engine (UE): Left-handed coordinate system\n") 
            f.write("   - X: Forward (North)\n")
            f.write("   - Y: Right (East)\n")
            f.write("   - Z: Up\n")
            f.write("   - Units: centimeters\n")
            f.write("   - Rotations: Yaw (Z), Pitch (Y), Roll (X) in degrees\n\n")
            
            f.write("2. Standard Right-handed: Right-handed coordinate system\n")
            f.write("   - X: Forward (North)\n")
            f.write("   - Y: Left (West)\n")
            f.write("   - Z: Up\n")
            f.write("   - Units: meters\n")
            f.write("   - Rotations: Yaw (Z), Pitch (Y), Roll (X) in degrees\n\n")
            
            f.write("COORDINATE TRANSFORMATION:\n")
            f.write("-" * 26 + "\n")
            f.write("UE to Right-handed conversion:\n")
            f.write("  X_rh = X_ue / 100     (Forward, cm to m)\n")
            f.write("  Y_rh = -Y_ue / 100    (Right to Left, cm to m)\n")
            f.write("  Z_rh = Z_ue / 100     (Up, cm to m)\n\n")
            
            f.write("DEPTH FORMAT:\n")
            f.write("-" * 13 + "\n")
            f.write("16-bit PNG depth images with direct centimeter values\n")
            f.write("  - No processing required\n")
            f.write("  - Range: 0-65535 cm (0-655.35 meters)\n")
            f.write("  - Single channel or first channel if multi-channel\n\n")
            
            f.write("STATISTICS:\n")
            f.write("-" * 12 + "\n")
            f.write(ue_stats)
            f.write("\n")
            f.write(rh_stats)
            
            f.write("\nERROR TYPE COLOR CODING:\n")
            f.write("-" * 25 + "\n")
            for error_type, color in self.error_colors.items():
                f.write(f"{error_type.upper()}: RGB{color}\n")
            
            f.write("\nVIEWING RECOMMENDATIONS:\n")
            f.write("-" * 25 + "\n")
            f.write("- CloudCompare: Open PLY files directly\n")
            f.write("- MeshLab: Import and adjust point size for visibility\n")
            f.write("- Blender: Import as mesh, enable point cloud display\n")
            f.write("- UE Editor: Import UE coordinate PLY as static mesh or point cloud asset\n")
            f.write("- Python/Matplotlib: Use right-handed PLY for standard visualization\n")
        
        print(f"Summary report saved to {report_path}")
    
    def _analyze_points(self, points: List[Dict], system_name: str) -> str:
        """Analyze point distribution and return statistics string."""
        if not points:
            return f"{system_name}: No points\n"
        
        # Extract coordinates
        coords = np.array([[p['x'], p['y'], p['z']] for p in points])
        
        # Calculate bounds
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        mean_coords = np.mean(coords, axis=0)
        extent = max_coords - min_coords
        
        # Error type distribution
        error_types = {}
        for point in points:
            error_type = point['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Error intensity statistics
        intensities = [p['error_intensity'] for p in points]
        
        # Depth statistics
        depths = [p['depth'] for p in points]
        
        # Determine units
        units = "cm" if "cm" in system_name.lower() else "m"
        
        stats = f"{system_name}:\n"
        stats += f"  Total points: {len(points)}\n"
        stats += f"  Bounds X: [{min_coords[0]:.3f}, {max_coords[0]:.3f}] {units} (extent: {extent[0]:.3f} {units})\n"
        stats += f"  Bounds Y: [{min_coords[1]:.3f}, {max_coords[1]:.3f}] {units} (extent: {extent[1]:.3f} {units})\n"
        stats += f"  Bounds Z: [{min_coords[2]:.3f}, {max_coords[2]:.3f}] {units} (extent: {extent[2]:.3f} {units})\n"
        stats += f"  Center: ({mean_coords[0]:.3f}, {mean_coords[1]:.3f}, {mean_coords[2]:.3f}) {units}\n"
        stats += f"  Error intensity: min={min(intensities):.3f}, max={max(intensities):.3f}, avg={np.mean(intensities):.3f}\n"
        stats += f"  Depth range: {min(depths):.3f} - {max(depths):.3f} {units} (avg: {np.mean(depths):.3f} {units})\n"
        stats += f"  Error type distribution:\n"
        for error_type, count in error_types.items():
            percentage = (count / len(points)) * 100
            stats += f"    {error_type}: {count} ({percentage:.1f}%)\n"
        
        return stats
    
    def generate_point_clouds(self, image_selection_mode: str = "all", 
                            image_start: int = 0, image_end: int = None,
                            specific_images: List[int] = None, single_image: int = None,
                            filter_error_types: Optional[List[str]] = None,
                            min_region_size: int = 10, max_points: Optional[int] = None,
                            generate_camera_poses: bool = True, 
                            generate_both_coordinates: bool = True):
        """Main function to generate point clouds from analysis results."""
        print("Starting Enhanced UE point cloud generation...")
        print("Depth format: 16-bit PNG with direct centimeter values")
        
        # Load analysis results with image selection
        results = self.load_analysis_results(
            image_selection_mode, image_start, image_end, 
            specific_images, single_image
        )
        
        if not results:
            print("No analysis results found!")
            return
        
        # Extract error points for both coordinate systems
        ue_points, rh_points = self.extract_error_points(results)
        
        # Apply filters to both coordinate systems
        if filter_error_types:
            ue_points = [p for p in ue_points if p['error_type'] in filter_error_types]
            rh_points = [p for p in rh_points if p['error_type'] in filter_error_types]
            print(f"Filtered to error types: {filter_error_types}")
        
        if min_region_size > 0:
            ue_points = [p for p in ue_points if p['region_size'] >= min_region_size]
            rh_points = [p for p in rh_points if p['region_size'] >= min_region_size]
            print(f"Filtered to minimum region size: {min_region_size}")
        
        if max_points and len(ue_points) > max_points:
            # Sort by error intensity and take top points
            ue_points.sort(key=lambda x: x['error_intensity'], reverse=True)
            rh_points.sort(key=lambda x: x['error_intensity'], reverse=True)
            ue_points = ue_points[:max_points]
            rh_points = rh_points[:max_points]
            print(f"Limited to top {max_points} points by error intensity")
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        generated_files = []
        
        # Write UE coordinate PLY files
        self.write_ply_file(ue_points, f"error_points_ue_{timestamp}.ply", 
                           "Unreal Engine", "centimeters")
        generated_files.append(f"error_points_ue_{timestamp}.ply")
        
        # Write right-handed coordinate PLY files if requested
        if generate_both_coordinates:
            self.write_ply_file(rh_points, f"error_points_rh_{timestamp}.ply", 
                               "Standard Right-handed", "meters")
            generated_files.append(f"error_points_rh_{timestamp}.ply")
        
        # Write camera poses if requested
        if generate_camera_poses:
            self.write_camera_poses_ply(ue_points, f"camera_poses_ue_{timestamp}.ply", 
                                       "Unreal Engine", "centimeters")
            generated_files.append(f"camera_poses_ue_{timestamp}.ply")
            
            if generate_both_coordinates:
                self.write_camera_poses_ply(rh_points, f"camera_poses_rh_{timestamp}.ply", 
                                           "Standard Right-handed", "meters")
                generated_files.append(f"camera_poses_rh_{timestamp}.ply")
        
        # Write summary report
        self.write_summary_report(ue_points, rh_points)
        generated_files.append("pointcloud_summary.txt")
        
        print(f"\nEnhanced point cloud generation completed!")
        print(f"Generated files in: {self.output_dir}")
        print(f"UE points: {len(ue_points)}")
        if generate_both_coordinates:
            print(f"RH points: {len(rh_points)}")
        print(f"Files generated:")
        for file in generated_files:
            print(f"  - {file}")

def main():
    """Main function using configuration settings."""
    print("=" * 70)
    print("ENHANCED UE ERROR POINT CLOUD GENERATOR")
    print("=" * 70)
    
    # Determine analysis directory
    if MANUAL_ANALYSIS_DIR and not AUTO_SELECT_NEWEST:
        analysis_dir = MANUAL_ANALYSIS_DIR
        print(f"Using manual directory: {analysis_dir}")
    else:
        analysis_dir = find_newest_analysis_directory(ANALYSIS_RESULTS_BASE_DIR)
        if not analysis_dir:
            print("❌ Could not find or access analysis directory")
            sys.exit(1)
    
    print("Configuration:")
    print(f"  Analysis directory: {analysis_dir}")
    print(f"  Image selection mode: {IMAGE_SELECTION_MODE}")
    
    if IMAGE_SELECTION_MODE == "range":
        print(f"  Image range: {IMAGE_START} to {IMAGE_END}")
    elif IMAGE_SELECTION_MODE == "specific":
        print(f"  Specific images: {SPECIFIC_IMAGES}")
    elif IMAGE_SELECTION_MODE == "single":
        print(f"  Single image: {SINGLE_IMAGE}")
    
    print(f"  Filter error types: {FILTER_ERROR_TYPES if FILTER_ERROR_TYPES else 'All types'}")
    print(f"  Min region size: {MIN_REGION_SIZE} pixels")
    print(f"  Max points: {MAX_POINTS if MAX_POINTS else 'Unlimited'}")
    print(f"  Generate camera poses: {GENERATE_CAMERA_POSES}")
    print(f"  Generate both coordinates: {GENERATE_BOTH_COORDINATES}")
    print(f"  Depth format: 16-bit PNG with direct centimeter values")
    print("=" * 70)
    
    # Check if analysis directory exists
    if not Path(analysis_dir).exists():
        print(f"❌ Error: Analysis directory does not exist: {analysis_dir}")
        sys.exit(1)
    
    print(f"✓ Analysis directory found: {analysis_dir}")
    
    # Create generator and run
    generator = EnhancedErrorPointCloudGenerator(analysis_dir)
    generator.generate_point_clouds(
        image_selection_mode=IMAGE_SELECTION_MODE,
        image_start=IMAGE_START,
        image_end=IMAGE_END,
        specific_images=SPECIFIC_IMAGES,
        single_image=SINGLE_IMAGE,
        filter_error_types=FILTER_ERROR_TYPES,
        min_region_size=MIN_REGION_SIZE,
        max_points=MAX_POINTS,
        generate_camera_poses=GENERATE_CAMERA_POSES,
        generate_both_coordinates=GENERATE_BOTH_COORDINATES
    )

if __name__ == "__main__":
    main()