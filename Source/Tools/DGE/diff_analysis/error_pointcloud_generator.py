import os
import math
import glob
import cv2
import numpy as np
import open3d as o3d
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from datetime import datetime
import re
import sys
from typing import List, Dict, Tuple, Optional

try:
    import OpenEXR
    import Imath
    HAS_EXR = True
except ImportError:
    HAS_EXR = False

class Config:
    """Configuration settings for point cloud initialization"""
    
    ANALYSIS_RESULTS_BASE_DIR = "./Logs/analysis_results"
    AUTO_SELECT_NEWEST = True
    MANUAL_ANALYSIS_DIR = None

    UE_POSES_FILE = "./Source/Tools/DGE/data/poses/0528_pose.txt"
    UE_REAL_IMAGES_DIR = "./Source/Tools/DGE/data/real_images_downscaled"
    UE_NORMAL_IMAGES_DIR = "./Source/Tools/DGE/data/normal_images"

    OUTPUT_BASE_DIR = "./Logs/pointcloud_datasets"
    AUTO_CREATE_TIMESTAMP_DIR = True
    MANUAL_OUTPUT_DIR = None

    ERROR_TYPE_FILTER = None
    MIN_REGION_SIZE = 10
    MAX_POINTS = None
    MIN_POINT_DISTANCE_CM = 50.0

    FINAL_MAX_POINTS = 100000
    APPLY_FINAL_LIMIT = True

    IMAGE_WIDTH = 1216
    IMAGE_HEIGHT = 912
    HFOV_DEGREES = 67.38
    COORDINATE_SCALE = 100.0

class DirectoryManager:
    """Handles directory selection and validation"""
    
    @staticmethod
    def find_newest_analysis_directory(base_dir: str) -> Optional[str]:
        """Find the newest analysis results directory based on timestamp."""
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"ERROR: Analysis base directory not found: {base_dir}")
            return None
        
        timestamp_pattern = re.compile(r'(\d{8}_\d{6})')
        analysis_dirs = []
        
        for item in base_path.iterdir():
            if item.is_dir():
                match = timestamp_pattern.search(item.name)
                if match:
                    timestamp_str = match.group(1)
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        analysis_dirs.append((timestamp, item))
                    except ValueError:
                        continue
        
        if not analysis_dirs:
            print(f"ERROR: No analysis directories found in {base_dir}")
            return None
        
        analysis_dirs.sort(key=lambda x: x[0], reverse=True)
        newest_dir = analysis_dirs[0][1]
        print(f"Selected analysis directory: {newest_dir}")
        
        return str(newest_dir)

    @staticmethod
    def determine_analysis_directory() -> str:
        """Determine which analysis directory to use."""
        if Config.MANUAL_ANALYSIS_DIR and not Config.AUTO_SELECT_NEWEST:
            analysis_dir = Config.MANUAL_ANALYSIS_DIR
        else:
            analysis_dir = DirectoryManager.find_newest_analysis_directory(Config.ANALYSIS_RESULTS_BASE_DIR)
            if not analysis_dir:
                sys.exit(1)
        
        if not Path(analysis_dir).exists():
            print(f"ERROR: Analysis directory does not exist: {analysis_dir}")
            sys.exit(1)
            
        return analysis_dir

    @staticmethod
    def determine_output_directory() -> str:
        """Determine output directory."""
        if Config.MANUAL_OUTPUT_DIR and not Config.AUTO_CREATE_TIMESTAMP_DIR:
            return Config.MANUAL_OUTPUT_DIR
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return str(Path(Config.OUTPUT_BASE_DIR) / f"pointcloud_dataset_{timestamp}")

    @staticmethod
    def validate_input_paths():
        """Validate required input paths exist."""
        paths_to_check = [
            ("Real Images Directory", Config.UE_REAL_IMAGES_DIR),
            ("Normal Images Directory", Config.UE_NORMAL_IMAGES_DIR),
            ("UE Poses File", Config.UE_POSES_FILE),
        ]
        
        missing_paths = []
        for name, path in paths_to_check:
            if not Path(path).exists():
                missing_paths.append(f"{name}: {path}")
        
        if missing_paths:
            print("ERROR: Missing required files:")
            for missing in missing_paths:
                print(f"  {missing}")
            sys.exit(1)

class NormalImageProcessor:
    """Handles loading and processing of UE normal images in EXR format"""
    
    @staticmethod
    def load_normal_image_exr(normal_path: str) -> Optional[np.ndarray]:
        """Load normal image from EXR file format."""
        try:
            if not Path(normal_path).exists():
                return None
            
            if HAS_EXR:
                exr_file = OpenEXR.InputFile(normal_path)
                header = exr_file.header()
                
                dw = header['dataWindow']
                width = dw.max.x - dw.min.x + 1
                height = dw.max.y - dw.min.y + 1
                
                FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                r_str = exr_file.channel('R', FLOAT)
                g_str = exr_file.channel('G', FLOAT)
                b_str = exr_file.channel('B', FLOAT)
                
                r = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
                g = np.frombuffer(g_str, dtype=np.float32).reshape(height, width)
                b = np.frombuffer(b_str, dtype=np.float32).reshape(height, width)
                
                normal_image = np.stack([r, g, b], axis=2)
                exr_file.close()
            else:
                normal_image = cv2.imread(normal_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if normal_image is not None:
                    normal_image = cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB)
                else:
                    return None
    
            return normal_image
            
        except Exception as e:
            print(f"Error loading EXR normal image {normal_path}: {e}")
            return None
    
    @staticmethod
    def find_normal_image_path(image_name: str, normal_images_dir: str) -> Optional[str]:
        """Find the corresponding normal image file for a given rendered image."""
        base_name = Path(image_name).stem
        normal_path = Path(normal_images_dir)
        
        exact_match = normal_path / f"{base_name}.exr"
        if exact_match.exists():
            return str(exact_match)
        
        try:
            image_num = int(base_name)
            patterns = [
                f"Normal_Cam00_Pose{image_num:03d}.exr",
                f"Normal_Cam0_Pose{image_num:03d}.exr",
                f"normal_{base_name}.exr"
            ]
            
            for pattern in patterns:
                normal_file = normal_path / pattern
                if normal_file.exists():
                    return str(normal_file)
        except ValueError:
            pass
        
        exr_files = list(normal_path.glob("*.exr"))
        for exr_file in exr_files:
            if base_name in exr_file.stem:
                return str(exr_file)
        
        return None
    
    @staticmethod
    def decode_ue_normal(encoded_normal: np.ndarray) -> np.ndarray:
        """Decode UE SCS_Normal format: (encoded * 2.0) - 1.0"""
        decoded = encoded_normal * 2.0 - 1.0
        length = np.linalg.norm(decoded)
        if length > 1e-6:
            return decoded / length
        else:
            return np.array([0.0, 0.0, 1.0])
    
    @staticmethod
    def extract_normal_at_pixel(normal_image: np.ndarray, x: int, y: int) -> Optional[np.ndarray]:
        """Extract and decode normal vector at specific pixel coordinates."""
        if normal_image is None:
            return None
        
        height, width = normal_image.shape[:2]
        
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        
        encoded_normal = normal_image[y, x, :3]
        
        decoded_normal = NormalImageProcessor.decode_ue_normal(encoded_normal)
        
        return decoded_normal

class CoordinateConverter:
    """Handles coordinate system conversions between UE and COLMAP"""
    
    @staticmethod
    def convert_pose_to_matrix(x: float, y: float, z: float, 
                             pitch: float, yaw: float, roll: float, 
                             scale: float = 10.0) -> np.ndarray:
        """Convert UE pose to COLMAP coordinate system transformation matrix"""
        pitch, yaw, roll = map(np.deg2rad, [pitch, yaw, roll])
        R = o3d.geometry.get_rotation_matrix_from_yzx([-yaw, pitch, roll])
        R_colmap = np.column_stack((R[:, 2], -R[:, 1], R[:, 0]))

        T = np.eye(4)
        T[:3, :3] = R_colmap
        T[:3, 3] = [x / scale, z / scale, y / scale]
        return T

    @staticmethod
    def ue_world_to_colmap_world(ue_world_pos: List[float], scale: float = 10.0) -> np.ndarray:
        """Convert UE world coordinates to COLMAP world coordinates"""
        x_ue, y_ue, z_ue = ue_world_pos
        return np.array([x_ue / scale, z_ue / scale, y_ue / scale])
    
    @staticmethod
    def convert_ue_normal_to_colmap(ue_normal: np.ndarray) -> np.ndarray:
        """Convert UE normal vector to COLMAP coordinate system."""
        x_ue, y_ue, z_ue = ue_normal
        colmap_normal = np.array([x_ue, z_ue, y_ue])
        
        length = np.linalg.norm(colmap_normal)
        if length > 1e-6:
            colmap_normal = colmap_normal / length
        
        return colmap_normal

class CameraUtils:
    """Camera-related utility functions"""
    
    @staticmethod
    def compute_intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
        """Compute camera intrinsics from UE parameters"""
        hfov_rad = math.radians(hfov_deg)
        fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
        vfov_rad = 2.0 * math.atan((height / width) * math.tan(hfov_rad / 2.0))
        fy = (height / 2.0) / math.tan(vfov_rad / 2.0)
        cx, cy = width / 2.0, height / 2.0

        K = np.eye(4, dtype=np.float32)
        K[0, 0], K[1, 1] = fx, fy
        K[0, 2], K[1, 2] = cx, cy
        return K

    @staticmethod
    def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert rotation matrix to quaternion (qw, qx, qy, qz)"""
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * s
            qx = (R[2,1] - R[1,2]) / s
            qy = (R[0,2] - R[2,0]) / s
            qz = (R[1,0] - R[0,1]) / s
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
                qw = (R[2,1] - R[1,2]) / s
                qx = 0.25 * s
                qy = (R[0,1] + R[1,0]) / s
                qz = (R[0,2] + R[2,0]) / s
            elif R[1,1] > R[2,2]:
                s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
                qw = (R[0,2] - R[2,0]) / s
                qx = (R[0,1] + R[1,0]) / s
                qy = 0.25 * s
                qz = (R[1,2] + R[2,1]) / s
            else:
                s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
                qw = (R[1,0] - R[0,1]) / s
                qx = (R[0,2] + R[2,0]) / s
                qy = (R[1,2] + R[2,1]) / s
                qz = 0.25 * s
        
        return qw, qx, qy, qz

class AlignmentUtils:
    """Utilities for handling image alignment transformations"""
    
    @staticmethod
    def transform_to_original_space(x_aligned: float, y_aligned: float, 
                                  transform_matrix: np.ndarray, 
                                  img_width: int, img_height: int) -> Tuple[float, float]:
        """Transform pixel coordinates from aligned space back to original space."""
        if transform_matrix is None or np.allclose(transform_matrix, np.eye(3)):
            return x_aligned, y_aligned
        
        try:
            aligned_coords = np.array([[x_aligned, y_aligned, 1.0]]).T
            inv_transform = np.linalg.inv(transform_matrix)
            original_coords = inv_transform @ aligned_coords
            
            if abs(original_coords[2, 0]) > 1e-10:
                x_original = original_coords[0, 0] / original_coords[2, 0]
                y_original = original_coords[1, 0] / original_coords[2, 0]
            else:
                x_original, y_original = x_aligned, y_aligned
            
            x_original = np.clip(x_original, 0, img_width - 1)
            y_original = np.clip(y_original, 0, img_height - 1)
            
            return float(x_original), float(y_original)
            
        except (np.linalg.LinAlgError, ValueError):
            return x_aligned, y_aligned

class AnalysisDataProcessor:
    """Handles loading and filtering of analysis results"""
    
    @staticmethod
    def load_analysis_results(analysis_results_dir: str) -> List[Dict]:
        """Load error analysis results from JSON files"""
        data_dir = Path(analysis_results_dir) / 'data'
        if not data_dir.exists():
            print(f"ERROR: Data directory not found: {data_dir}")
            return []
        
        results = []
        json_files = sorted(list(data_dir.glob('*_data.json')))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if data:
                        results.append(data)
            except Exception as e:
                print(f"Warning: Error loading {json_file}: {e}")
        
        print(f"Loaded {len(results)} analysis results from {len(json_files)} files")
        return results

    @staticmethod
    def filter_analysis_results(analysis_results: List[Dict]) -> List[Dict]:
        """Apply filtering based on configuration settings"""
        if not analysis_results:
            return []
        
        filtered_results = []
        total_points = 0
        filtered_points = 0
        
        for result in analysis_results:
            filtered_result = result.copy()
            positions_3d = result.get('positions_3d', {})
            filtered_positions = {}
            
            for error_type, regions in positions_3d.items():
                if Config.ERROR_TYPE_FILTER and error_type not in Config.ERROR_TYPE_FILTER:
                    continue
                
                filtered_regions = []
                for region in regions:
                    total_points += 1
                    region_size = region.get('region_size', 0)
                    if region_size >= Config.MIN_REGION_SIZE:
                        filtered_regions.append(region)
                        filtered_points += 1
                
                if filtered_regions:
                    filtered_positions[error_type] = filtered_regions
            
            filtered_result['positions_3d'] = filtered_positions
            filtered_results.append(filtered_result)
        
        print(f"Applied filters: {filtered_points}/{total_points} points kept")
        return filtered_results

    @staticmethod
    def extract_error_intensity_improved(region_data: Dict) -> float:
        """Extract error intensity with fallback methods"""
        for field in ['error_intensity', 'mean_error', 'avg_error', 'intensity']:
            if field in region_data and isinstance(region_data[field], (int, float)):
                value = float(region_data[field])
                if value > 0:
                    return value
        
        region_size = region_data.get('region_size', 100)
        return min(1.0, max(0.1, region_size / 5000.0))

class PointProcessor:
    """Handles point cloud processing operations"""
    
    @staticmethod
    def reduce_nearby_points(ue_points: List[Dict], point_info: List[Dict], 
                           min_distance_cm: float) -> Tuple[List[Dict], List[Dict]]:
        """Memory-efficient removal of nearby points using spatial hashing"""
        if len(ue_points) <= 1:
            return ue_points, point_info
        
        print(f"Starting point reduction with {len(ue_points)} points...")
        
        grid_size = min_distance_cm
        spatial_grid = {}
        
        def get_grid_coords(pos):
            return (
                int(pos[0] // grid_size),
                int(pos[1] // grid_size), 
                int(pos[2] // grid_size)
            )
        
        def get_neighbor_cells(grid_coord):
            x, y, z = grid_coord
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        neighbors.append((x + dx, y + dy, z + dz))
            return neighbors
        
        indexed_points = [(i, ue_points[i], point_info[i]) for i in range(len(ue_points))]
        indexed_points.sort(key=lambda x: x[1]['error_intensity'], reverse=True)
        
        kept_points = []
        kept_info = []
        points_removed = 0
        
        for i, (original_idx, ue_point, info) in enumerate(indexed_points):
            if i % 50000 == 0:
                print(f"  Processed {i}/{len(indexed_points)} points, removed {points_removed}")
            
            pos = ue_point['ue_pos']
            grid_coord = get_grid_coords(pos)
            
            should_keep = True
            
            for neighbor_coord in get_neighbor_cells(grid_coord):
                if neighbor_coord in spatial_grid:
                    for existing_pos in spatial_grid[neighbor_coord]:
                        distance = np.sqrt(sum((pos[i] - existing_pos[i])**2 for i in range(3)))
                        if distance < min_distance_cm:
                            should_keep = False
                            break
                if not should_keep:
                    break
            
            if should_keep:
                if grid_coord not in spatial_grid:
                    spatial_grid[grid_coord] = []
                spatial_grid[grid_coord].append(pos)
                
                kept_points.append(ue_point)
                kept_info.append(info)
            else:
                points_removed += 1
        
        print(f"Point reduction complete: {points_removed} points removed, {len(kept_points)} points kept")
        return kept_points, kept_info

    @staticmethod
    def extract_error_points_from_regions(analysis_results: List[Dict], 
                                        normal_images_dir: str,
                                        scale: float = 10.0) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
        """Extract one point per region center with real image colors."""
        print("Extracting error points from region centers with real colors...")
        
        ue_points = []
        
        for result in analysis_results:
            image_name = result.get('image_name', 'unknown')
            positions_3d = result.get('positions_3d', {})
            alignment_transform = result.get('alignment_transform', None)
            
            transform_matrix = None
            if alignment_transform is not None:
                transform_matrix = np.array(alignment_transform)
            
            normal_image_path = NormalImageProcessor.find_normal_image_path(
                image_name, normal_images_dir)
            normal_image = None
            if normal_image_path:
                normal_image = NormalImageProcessor.load_normal_image_exr(normal_image_path)
            
            for error_type, regions in positions_3d.items():
                if not regions:
                    continue
                
                for region_data in regions:
                    region_size = region_data.get('region_size', 1)
                    world_position = region_data.get('world_position', [0, 0, 0])
                    error_intensity = AnalysisDataProcessor.extract_error_intensity_improved(region_data)
                    
                    pixel_center_aligned = region_data.get('pixel_center_aligned', [Config.IMAGE_WIDTH//2, Config.IMAGE_HEIGHT//2])
                    pixel_center_original = region_data.get('pixel_center_original', pixel_center_aligned)
                    
                    if transform_matrix is not None:
                        x_orig, y_orig = AlignmentUtils.transform_to_original_space(
                            pixel_center_aligned[0], pixel_center_aligned[1], transform_matrix, 
                            Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT)
                    else:
                        x_orig, y_orig = pixel_center_aligned[0], pixel_center_aligned[1]
                    
                    x_orig = max(0, min(x_orig, Config.IMAGE_WIDTH - 1))
                    y_orig = max(0, min(y_orig, Config.IMAGE_HEIGHT - 1))
                    
                    normal_vector = None
                    if normal_image is not None:
                        normal_vector = NormalImageProcessor.extract_normal_at_pixel(
                            normal_image, int(round(x_orig)), int(round(y_orig)))
                    
                    # Get real color from analysis results
                    real_color = region_data.get('real_color', [128, 128, 128])
                    
                    ue_points.append({
                        'ue_pos': world_position,
                        'error_type': error_type,
                        'error_intensity': error_intensity,
                        'region_size': region_size,
                        'image_name': image_name,
                        'pixel_aligned': pixel_center_aligned,
                        'pixel_original': [x_orig, y_orig],
                        'normal_ue': normal_vector.tolist() if normal_vector is not None else None,
                        'has_normal': normal_vector is not None,
                        'real_color': real_color
                    })
        
        print(f"Created {len(ue_points)} points from region centers")
        
        point_info = []
        for ue_point in ue_points:
            point_info.append({
                'error_type': ue_point['error_type'],
                'error_intensity': ue_point['error_intensity'],
                'region_size': ue_point['region_size'],
                'image_name': ue_point['image_name'],
                'original_ue_pos': ue_point['ue_pos'],
                'pixel_aligned': ue_point['pixel_aligned'],
                'pixel_original': ue_point['pixel_original'],
                'normal_ue': ue_point['normal_ue'],
                'normal_colmap': None,
                'has_normal': ue_point['has_normal'],
                'colmap_pos': None,
                'real_color': ue_point['real_color']
            })
        
        if Config.MIN_POINT_DISTANCE_CM > 0:
            ue_points, point_info = PointProcessor.reduce_nearby_points(
                ue_points, point_info, Config.MIN_POINT_DISTANCE_CM)
        
        if Config.MAX_POINTS and len(ue_points) > Config.MAX_POINTS:
            print(f"Limiting to top {Config.MAX_POINTS} points by error intensity")
            combined = list(zip(ue_points, point_info))
            combined.sort(key=lambda x: x[0]['error_intensity'], reverse=True)
            ue_points = [x[0] for x in combined[:Config.MAX_POINTS]]
            point_info = [x[1] for x in combined[:Config.MAX_POINTS]]
        
        colmap_points = []
        for i, ue_point in enumerate(ue_points):
            colmap_pos = CoordinateConverter.ue_world_to_colmap_world(ue_point['ue_pos'], scale)
            colmap_points.append(colmap_pos)
            point_info[i]['colmap_pos'] = colmap_pos.tolist()
            
            if ue_point['normal_ue'] is not None:
                normal_ue = np.array(ue_point['normal_ue'])
                normal_colmap = CoordinateConverter.convert_ue_normal_to_colmap(normal_ue)
                point_info[i]['normal_colmap'] = normal_colmap.tolist()
        
        print(f"Point count after extraction: {len(colmap_points)}")
        return np.array(colmap_points), point_info, ue_points

    @staticmethod
    def apply_final_point_limit(colmap_points: np.ndarray, point_info: List[Dict], 
                              ue_points: List[Dict], max_points: int) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
        """Apply final point cloud limitation"""
        if not Config.APPLY_FINAL_LIMIT or len(colmap_points) <= max_points:
            return colmap_points, point_info, ue_points
        
        print(f"Applying final point limit: keeping top {max_points} points from {len(colmap_points)}")
        
        combined_data = list(zip(colmap_points, point_info, ue_points))
        combined_data.sort(key=lambda x: x[1]['error_intensity'], reverse=True)
        
        limited_data = combined_data[:max_points]
        
        limited_colmap_points = np.array([item[0] for item in limited_data])
        limited_point_info = [item[1] for item in limited_data]
        limited_ue_points = [item[2] for item in limited_data]
        
        points_removed = len(colmap_points) - len(limited_colmap_points)
        print(f"Final point limitation complete: {points_removed} points removed, {len(limited_colmap_points)} points kept")
        
        return limited_colmap_points, limited_point_info, limited_ue_points

class VisualizationUtils:
    """Utilities for creating visualizations and color schemes"""
    
    @staticmethod
    def error_intensity_to_color(error_intensity: float, error_type: str) -> List[int]:
        """Convert error intensity to RGB color based on error type"""
        color_schemes = {
            'mse': ([255, 0, 0], [255, 100, 100]),
            'mae': ([255, 128, 0], [255, 200, 128]),
            'lab': ([255, 255, 0], [255, 255, 128]),
            'gradient': ([0, 255, 0], [128, 255, 128])
        }
        
        base_color, light_color = color_schemes.get(error_type, ([128, 128, 128], [200, 200, 200]))
        intensity = max(0.3, min(1.0, error_intensity))
        
        color = [
            int(light_color[0] + (base_color[0] - light_color[0]) * intensity),
            int(light_color[1] + (base_color[1] - light_color[1]) * intensity), 
            int(light_color[2] + (base_color[2] - light_color[2]) * intensity)
        ]
        
        return color

class FileCreator:
    """Handles creation of various output files"""
    
    @staticmethod
    def create_colmap_points3d_ply_with_real_colors(colmap_points: np.ndarray, point_info: List[Dict], output_path: str):
        """Create points3D.ply file for COLMAP initialization with real image colors"""
        print(f"Creating COLMAP coordinate point cloud with real colors: {output_path}")
        
        vertices = []
        points_with_normals = 0
        points_without_normals = 0
        
        for i, (point_pos, info) in enumerate(zip(colmap_points, point_info)):
            # Use real color from image
            real_color = info.get('real_color', [128, 128, 128])
            
            if info['has_normal'] and info['normal_colmap'] is not None:
                normal = info['normal_colmap']
                points_with_normals += 1
            else:
                normal = [0.0, 0.0, 1.0]
                points_without_normals += 1
            
            vertices.append((
                float(point_pos[0]), float(point_pos[1]), float(point_pos[2]),
                float(normal[0]), float(normal[1]), float(normal[2]),
                int(real_color[0]), int(real_color[1]), int(real_color[2])
            ))
        
        vertices = np.array(vertices, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        
        el = PlyElement.describe(vertices, 'vertex')
        PlyData([el]).write(output_path)
        
        print(f"Saved {len(vertices)} points in COLMAP coordinates with real colors")
        print(f"  Points with extracted normals: {points_with_normals}")
        print(f"  Points with default normals: {points_without_normals}")

    @staticmethod
    def create_ue_points3d_ply_with_real_colors(ue_points: List[Dict], point_info: List[Dict], output_path: str):
        """Create points3D.ply file with UE coordinate system and real image colors"""
        print(f"Creating UE coordinate point cloud with real colors: {output_path}")
        
        vertices = []
        
        for i, (ue_point, info) in enumerate(zip(ue_points, point_info)):
            ue_pos = ue_point['ue_pos']
            real_color = info.get('real_color', [128, 128, 128])
            
            if info['has_normal'] and info['normal_ue'] is not None:
                normal = info['normal_ue']
            else:
                normal = [0.0, 0.0, 1.0]
            
            vertices.append((
                float(ue_pos[0]), float(ue_pos[1]), float(ue_pos[2]),
                float(normal[0]), float(normal[1]), float(normal[2]),
                int(real_color[0]), int(real_color[1]), int(real_color[2])
            ))
        
        vertices = np.array(vertices, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        
        el = PlyElement.describe(vertices, 'vertex')
        PlyData([el]).write(output_path)
        
        print(f"Saved {len(vertices)} points in UE coordinates with real colors")

    @staticmethod
    def create_ue_points3d_ply_with_error_colors(ue_points: List[Dict], point_info: List[Dict], output_path: str):
        """Create points3D.ply file with UE coordinate system and error classification colors"""
        print(f"Creating UE coordinate point cloud with error colors: {output_path}")
        
        vertices = []
        
        for i, (ue_point, info) in enumerate(zip(ue_points, point_info)):
            ue_pos = ue_point['ue_pos']
            error_color = VisualizationUtils.error_intensity_to_color(info['error_intensity'], info['error_type'])
            
            if info['has_normal'] and info['normal_ue'] is not None:
                normal = info['normal_ue']
            else:
                normal = [0.0, 0.0, 1.0]
            
            vertices.append((
                float(ue_pos[0]), float(ue_pos[1]), float(ue_pos[2]),
                float(normal[0]), float(normal[1]), float(normal[2]),
                int(error_color[0]), int(error_color[1]), int(error_color[2])
            ))
        
        vertices = np.array(vertices, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        
        el = PlyElement.describe(vertices, 'vertex')
        PlyData([el]).write(output_path)
        
        print(f"Saved {len(vertices)} points in UE coordinates with error colors")

    @staticmethod
    def create_camera_pose_pointcloud(ue_poses_file: str, output_path: str, scale: float = 10.0):
        """Create a point cloud representing camera poses in COLMAP coordinate system"""
        print(f"Creating camera pose point cloud: {output_path}")
        
        poses = []
        with open(ue_poses_file, 'r') as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if len(vals) == 6:
                    poses.append(vals)
        
        if not poses:
            print("Warning: No poses found in poses file")
            return
        
        vertices = []
        
        for i, pose in enumerate(poses):
            x, y, z, pitch, yaw, roll = pose
            
            T = CoordinateConverter.convert_pose_to_matrix(x, y, z, pitch, yaw, roll, scale)
            camera_pos = T[:3, 3]
            R = T[:3, :3]
            
            vertices.append((
                float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2]),
                0.0, 0.0, 1.0,
                0, 255, 255
            ))
            
            arrow_length = 0.5
            
            forward_end = camera_pos + R[:, 2] * arrow_length
            vertices.append((
                float(forward_end[0]), float(forward_end[1]), float(forward_end[2]),
                float(R[0, 2]), float(R[1, 2]), float(R[2, 2]),
                255, 0, 0
            ))
            
            right_end = camera_pos + R[:, 0] * arrow_length * 0.7
            vertices.append((
                float(right_end[0]), float(right_end[1]), float(right_end[2]),
                float(R[0, 0]), float(R[1, 0]), float(R[2, 0]),
                0, 255, 0
            ))
            
            up_end = camera_pos - R[:, 1] * arrow_length * 0.7
            vertices.append((
                float(up_end[0]), float(up_end[1]), float(up_end[2]),
                float(-R[0, 1]), float(-R[1, 1]), float(-R[2, 1]),
                0, 0, 255
            ))
        
        vertices = np.array(vertices, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        
        el = PlyElement.describe(vertices, 'vertex')
        PlyData([el]).write(output_path)
        print(f"Saved {len(poses)} camera poses ({len(vertices)} points total) in COLMAP coordinates")

    @staticmethod
    def create_colmap_cameras_txt(K: np.ndarray, width: int, height: int, output_path: str):
        """Create cameras.txt file for COLMAP format"""
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        with open(output_path, 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

    @staticmethod
    def create_colmap_images_txt(ue_poses_file: str, image_names: List[str], 
                               K: np.ndarray, scale: float = 10.0, output_path: str = None):
        """Create images.txt file for COLMAP format from UE poses"""
        poses = []
        with open(ue_poses_file, 'r') as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if len(vals) == 6:
                    poses.append(vals)
        
        with open(output_path, 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            
            for idx, (pose, img_name) in enumerate(zip(poses[:len(image_names)], image_names)):
                x, y, z, pitch, yaw, roll = pose
                T = CoordinateConverter.convert_pose_to_matrix(x, y, z, pitch, yaw, roll, scale)
                R, t = T[:3, :3], T[:3, 3]
                qw, qx, qy, qz = CameraUtils.rotation_matrix_to_quaternion(R)
                
                f.write(f"{idx+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {img_name}\n")
                f.write("\n")

class ImageProcessor:
    """Handles image processing operations"""
    
    @staticmethod
    def process_real_images(src_dir: str, dst_dir: str, width: int, height: int) -> List[str]:
        """Process real images for training"""
        img_out = os.path.join(dst_dir, "images")
        os.makedirs(img_out, exist_ok=True)

        rgb_list = sorted(glob.glob(os.path.join(src_dir, "*.png")))
        if not rgb_list:
            rgb_list = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
        
        if not rgb_list:
            raise FileNotFoundError(f"No image files found in: {src_dir}")

        image_names = []
        for new_id, rgb_path in enumerate(rgb_list):
            rgb = cv2.imread(rgb_path)
            if rgb is None:
                continue
                
            rgb_resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            img_name = f"{new_id:04d}.png"
            cv2.imwrite(os.path.join(img_out, img_name), rgb_resized)
            image_names.append(img_name)
        
        print(f"Processed {len(image_names)} real images for training")
        return image_names

class PointCloudDatasetCreator:
    """Main class for creating point cloud datasets"""
    
    @staticmethod
    def create_pointcloud_dataset_from_error_analysis(
        analysis_results_dir: str, real_images_dir: str, normal_images_dir: str, 
        ue_poses_file: str, output_dir: str,
        width: int = None, height: int = None, 
        hfov_deg: float = None, scale: float = None
    ):
        """Create COLMAP-format dataset for 3D reconstruction from error analysis with real colors"""
        
        width = width or Config.IMAGE_WIDTH
        height = height or Config.IMAGE_HEIGHT
        hfov_deg = hfov_deg or Config.HFOV_DEGREES
        scale = scale or Config.COORDINATE_SCALE
        
        print("Creating point cloud dataset from error analysis with real colors...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        sparse_dir = Path(output_dir) / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        
        analysis_results = AnalysisDataProcessor.load_analysis_results(analysis_results_dir)
        if not analysis_results:
            raise ValueError("No analysis results found!")
        
        filtered_results = AnalysisDataProcessor.filter_analysis_results(analysis_results)
        
        colmap_points, point_info, ue_points = PointProcessor.extract_error_points_from_regions(
            filtered_results, normal_images_dir, scale)
        
        if len(colmap_points) == 0:
            print("ERROR: No points remaining after filtering!")
            sys.exit(1)
        
        colmap_points, point_info, ue_points = PointProcessor.apply_final_point_limit(
            colmap_points, point_info, ue_points, Config.FINAL_MAX_POINTS
        )
        
        points_with_normals = sum(1 for info in point_info if info['has_normal'])
        
        print(f"Final point count: {len(colmap_points)} error points")
        print(f"Points with surface normals: {points_with_normals}/{len(colmap_points)} ({points_with_normals/len(colmap_points)*100:.1f}%)")
        
        # Create COLMAP point cloud with real colors
        FileCreator.create_colmap_points3d_ply_with_real_colors(colmap_points, point_info, sparse_dir / "points3D.ply")
        
        # Create UE coordinate point clouds - real colors and error colors
        ue_real_colors_path = Path(output_dir) / "error_points_ue_real_colors.ply"
        FileCreator.create_ue_points3d_ply_with_real_colors(ue_points, point_info, ue_real_colors_path)
        
        ue_error_colors_path = Path(output_dir) / "error_points_ue_error_colors.ply"
        FileCreator.create_ue_points3d_ply_with_error_colors(ue_points, point_info, ue_error_colors_path)
        
        camera_poses_path = Path(output_dir) / "camera_poses_colmap.ply"
        FileCreator.create_camera_pose_pointcloud(ue_poses_file, camera_poses_path, scale)
        
        K = CameraUtils.compute_intrinsics(width, height, hfov_deg)
        image_names = ImageProcessor.process_real_images(real_images_dir, output_dir, width, height)
        
        FileCreator.create_colmap_cameras_txt(K, width, height, sparse_dir / "cameras.txt")
        FileCreator.create_colmap_images_txt(ue_poses_file, image_names, K, scale, sparse_dir / "images.txt")
        
        with open(sparse_dir / "points3D.txt", 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        
        metadata = {
            'creation_time': datetime.now().isoformat(),
            'source_analysis_dir': str(analysis_results_dir),
            'training_images_source': str(real_images_dir),
            'normal_images_source': str(normal_images_dir),
            'num_images': len(image_names),
            'num_cameras': len(image_names),
            'num_error_points': len(colmap_points),
            'num_points_with_normals': points_with_normals,
            'normal_coverage_percentage': points_with_normals/len(colmap_points)*100 if colmap_points.size > 0 else 0,
            'coordinate_scale': scale,
            'filtering_applied': {
                'error_types': Config.ERROR_TYPE_FILTER,
                'min_region_size': Config.MIN_REGION_SIZE,
                'max_points': Config.MAX_POINTS,
                'min_point_distance_cm': Config.MIN_POINT_DISTANCE_CM,
                'final_max_points': Config.FINAL_MAX_POINTS,
                'final_limit_applied': Config.APPLY_FINAL_LIMIT,
            },
            'error_point_distribution': {},
            'color_information': {
                'colmap_colors': 'Real image colors from analysis',
                'ue_real_colors': 'Real image colors for visualization',
                'ue_error_colors': 'Error classification colors'
            },
            'output_files': {
                'colmap_pointcloud': str(sparse_dir / "points3D.ply"),
                'ue_real_colors': str(ue_real_colors_path),
                'ue_error_colors': str(ue_error_colors_path),
                'camera_poses': str(camera_poses_path)
            }
        }
        
        for info in point_info:
            error_type = info['error_type']
            metadata['error_point_distribution'][error_type] = metadata['error_point_distribution'].get(error_type, 0) + 1
        
        with open(Path(output_dir) / "dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        PointCloudDatasetCreator._print_summary(output_dir, image_names, colmap_points, 
                                              metadata, sparse_dir, ue_real_colors_path, ue_error_colors_path, camera_poses_path)

    @staticmethod
    def _print_summary(output_dir: str, image_names: List[str], colmap_points: np.ndarray,
                      metadata: Dict, sparse_dir: Path, ue_real_colors_path: Path, ue_error_colors_path: Path, camera_poses_path: Path):
        """Print creation summary"""
        print("=" * 60)
        print("POINT CLOUD DATASET CREATION COMPLETED!")
        print("=" * 60)
        print(f"Output: {output_dir}")
        print(f"Images: {len(image_names)}")
        print(f"Error points: {len(colmap_points)}")
        print(f"Points with normals: {metadata['num_points_with_normals']}/{len(colmap_points)} ({metadata['normal_coverage_percentage']:.1f}%)")
        print(f"Error distribution:")
        for error_type, count in metadata['error_point_distribution'].items():
            print(f"  {error_type}: {count} points")
        print(f"\nOutput files:")
        print(f"  COLMAP point cloud (real colors): {sparse_dir / 'points3D.ply'}")
        print(f"  UE point cloud (real colors): {ue_real_colors_path}")
        print(f"  UE point cloud (error colors): {ue_error_colors_path}")
        print(f"  Camera poses: {camera_poses_path}")
        print("=" * 60)

def main():
    """Main function"""
    print("POINT CLOUD INITIALIZATION WITH REAL COLORS")
    print("=" * 60)
    
    analysis_dir = DirectoryManager.determine_analysis_directory()
    DirectoryManager.validate_input_paths()
    output_dir = DirectoryManager.determine_output_directory()
    
    print(f"Configuration:")
    print(f"  Analysis: {analysis_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Training images: {Config.UE_REAL_IMAGES_DIR}")
    print(f"  Normal images: {Config.UE_NORMAL_IMAGES_DIR}")
    if Config.ERROR_TYPE_FILTER:
        print(f"  Error types: {Config.ERROR_TYPE_FILTER}")
    if Config.APPLY_FINAL_LIMIT:
        print(f"  Final max points: {Config.FINAL_MAX_POINTS}")
    
    PointCloudDatasetCreator.create_pointcloud_dataset_from_error_analysis(
        analysis_results_dir=analysis_dir,
        real_images_dir=Config.UE_REAL_IMAGES_DIR,
        normal_images_dir=Config.UE_NORMAL_IMAGES_DIR,
        ue_poses_file=Config.UE_POSES_FILE,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()