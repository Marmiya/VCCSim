"""Data processing module with COLMAP coordinate conventions"""

import os
import glob
import json
import math
import cv2
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import re
from collections import defaultdict

from convert_ue_2_colmap import Config

try:
    import OpenEXR
    import Imath
    HAS_EXR = True
except ImportError:
    HAS_EXR = False


class CoordinateConverter:
    """
    Coordinate conversion from Unreal Engine to COLMAP 
    
    UE Coordinate System:
    - X: Forward, Y: Right, Z: Up (Centimeters)
    
    COLMAP Coordinate System:
    - X: Forward, Y: Right, Z: Down (Meters)
    """
    
    _UE_TO_COLMAP = np.array([
        [1,  0,  0],   # X: forward -> forward
        [0,  1,  0],   # Y: right -> right
        [0,  0, -1]    # Z: up -> down (flipped)
    ], dtype=np.float64)
    
    @staticmethod
    def ue_to_colmap_position(ue_pos: np.ndarray, scale: float = 100.0) -> np.ndarray:
        """Convert UE position (cm) to COLMAP position (m)"""
        ue_pos = np.asarray(ue_pos, dtype=np.float64)
        colmap_pos = CoordinateConverter._UE_TO_COLMAP @ ue_pos
        return colmap_pos / scale
    
    @staticmethod
    def ue_to_colmap_rotation(ue_rotation_matrix: np.ndarray) -> np.ndarray:
        """Convert UE rotation matrix to COLMAP rotation matrix"""
        C = CoordinateConverter._UE_TO_COLMAP
        return C @ ue_rotation_matrix @ C.T
    
    @staticmethod
    def ue_to_colmap_normal(ue_normal: np.ndarray) -> np.ndarray:
        """Convert UE normal vector to COLMAP normal vector"""
        ue_normal = np.asarray(ue_normal, dtype=np.float64)
        colmap_normal = CoordinateConverter._UE_TO_COLMAP @ ue_normal
        norm = np.linalg.norm(colmap_normal)
        return colmap_normal / norm if norm > 1e-8 else np.array([0, 0, 1], dtype=np.float64)
    
    @staticmethod
    def ue_pose_to_colmap_camera_pose(ue_x: float, ue_y: float, ue_z: float, 
                                     ue_pitch: float, ue_yaw: float, ue_roll: float,
                                     scale: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert UE camera pose to COLMAP camera pose
        
        Returns:
            R_w2c: World-to-camera rotation matrix (3x3)
            t_w2c: World-to-camera translation vector (3,)
        """
        # Build UE rotation matrix
        pitch_rad = np.radians(ue_pitch)
        yaw_rad = np.radians(ue_yaw)
        roll_rad = np.radians(ue_roll)
        
        cos_r, sin_r = np.cos(roll_rad), np.sin(roll_rad)
        cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        
        R_c2w_ue = np.array([
            [cos_y*cos_p, cos_y*sin_p*sin_r - sin_y*cos_r, cos_y*sin_p*cos_r + sin_y*sin_r],
            [sin_y*cos_p, sin_y*sin_p*sin_r + cos_y*cos_r, sin_y*sin_p*cos_r - cos_y*sin_r],
            [-sin_p,      cos_p*sin_r,                      cos_p*cos_r                     ]
        ], dtype=np.float64)
        
        # Apply coordinate transformation
        ue_position = np.array([ue_x, ue_y, ue_z], dtype=np.float64)
        colmap_position = CoordinateConverter.ue_to_colmap_position(ue_position, scale)
        R_c2w_colmap = CoordinateConverter.ue_to_colmap_rotation(R_c2w_ue)
        
        # Convert to world-to-camera
        R_w2c = R_c2w_colmap.T
        t_w2c = -R_w2c @ colmap_position
        
        return R_w2c, t_w2c


class CameraUtils:
    """Camera-related utility functions"""
    
    @staticmethod
    def compute_intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
        """Compute camera intrinsics matrix"""
        hfov_rad = math.radians(hfov_deg)
        fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
        vfov_rad = 2.0 * math.atan((height / width) * math.tan(hfov_rad / 2.0))
        fy = (height / 2.0) / math.tan(vfov_rad / 2.0)
        cx, cy = width / 2.0, height / 2.0

        K = np.eye(4, dtype=np.float64)
        K[0, 0], K[1, 1] = fx, fy
        K[0, 2], K[1, 2] = cx, cy
        return K

    @staticmethod
    def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert rotation matrix to quaternion (qw, qx, qy, qz)"""
        R = np.asarray(R, dtype=np.float64)
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * s
            qx = (R[2,1] - R[1,2]) / s
            qy = (R[0,2] - R[2,0]) / s
            qz = (R[1,0] - R[0,1]) / s
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
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
        
        if qw < 0:
            qw, qx, qy, qz = -qw, -qx, -qy, -qz
            
        return float(qw), float(qx), float(qy), float(qz)
    
    @staticmethod
    def validate_camera_pose(R_w2c: np.ndarray, t_w2c: np.ndarray, 
                           world_point: np.ndarray) -> np.ndarray:
        """Transform a world point to camera coordinates"""
        camera_point = R_w2c @ world_point + t_w2c
        return camera_point


class PointProcessor:
    """Point processing with COLMAP coordinate conversion"""
    
    @staticmethod
    def extract_error_points(analysis_results: List[Dict], 
                           normal_images_dir: str,
                           scale: float = 100.0) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
        """Extract error points with COLMAP coordinate conversion"""
        print(f"Extracting error points...(data processing)")
        
        chunk_size = max(1, len(analysis_results) // (mp.cpu_count() * 2))
        chunks = [analysis_results[i:i + chunk_size] for i in range(0, len(analysis_results), chunk_size)]
        
        all_ue_points = []
        all_point_info = []
        
        with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), len(chunks))) as executor:
            future_to_chunk = {
                executor.submit(PointProcessor._process_analysis_chunk, 
                              chunk, normal_images_dir, scale): i 
                for i, chunk in enumerate(chunks)
            }
            
            for future in as_completed(future_to_chunk):
                try:
                    ue_points_chunk, point_info_chunk = future.result()
                    all_ue_points.extend(ue_points_chunk)
                    all_point_info.extend(point_info_chunk)
                except Exception as e:
                    print(f"Chunk processing failed: {e}")
        
        print(f"Extracted {len(all_ue_points)} raw error points")
        
        # Apply distance filtering
        if Config.MIN_POINT_DISTANCE_CM > 0:
            all_ue_points, all_point_info = PointProcessor.reduce_nearby_points_optimized(
                all_ue_points, all_point_info, Config.MIN_POINT_DISTANCE_CM)
        
        # Apply max points limit
        if Config.MAX_POINTS and len(all_ue_points) > Config.MAX_POINTS:
            combined = list(zip(all_ue_points, all_point_info))
            combined.sort(key=lambda x: x[0]['error_intensity'], reverse=True)
            all_ue_points = [x[0] for x in combined[:Config.MAX_POINTS]]
            all_point_info = [x[1] for x in combined[:Config.MAX_POINTS]]
        
        # Convert to COLMAP coordinates
        colmap_points = []
        for i, ue_point in enumerate(all_ue_points):
            ue_pos = np.array(ue_point['ue_pos'], dtype=np.float64)
            colmap_pos = CoordinateConverter.ue_to_colmap_position(ue_pos, scale)
            colmap_points.append(colmap_pos)
            all_point_info[i]['colmap_pos'] = colmap_pos.tolist()
            
            if ue_point['normal_ue'] is not None:
                ue_normal = np.array(ue_point['normal_ue'], dtype=np.float64)
                colmap_normal = CoordinateConverter.ue_to_colmap_normal(ue_normal)
                all_point_info[i]['normal_colmap'] = colmap_normal.tolist()
        
        colmap_points_array = np.array(colmap_points)
        if len(colmap_points_array) > 0:
            print(f"Point cloud bounds:")
            print(f"  X: [{np.min(colmap_points_array[:, 0]):.3f}, {np.max(colmap_points_array[:, 0]):.3f}] m")
            print(f"  Y: [{np.min(colmap_points_array[:, 1]):.3f}, {np.max(colmap_points_array[:, 1]):.3f}] m") 
            print(f"  Z: [{np.min(colmap_points_array[:, 2]):.3f}, {np.max(colmap_points_array[:, 2]):.3f}] m")
        
        return colmap_points_array, all_point_info, all_ue_points
    
    @staticmethod  
    def _process_analysis_chunk(analysis_chunk: List[Dict], normal_images_dir: str, 
                              scale: float) -> Tuple[List[Dict], List[Dict]]:
        """Process analysis chunk with coordinate conversion"""
        from data_processing import NormalImageProcessor, AlignmentUtils, AnalysisDataProcessor
        
        normal_cache = {}
        
        def get_cached_normal_data(image_name: str):
            if image_name not in normal_cache:
                normal_path = NormalImageProcessor.find_normal_image_path(image_name, normal_images_dir)
                normal_image = None
                if normal_path:
                    normal_image = NormalImageProcessor.load_normal_image_exr(normal_path)
                normal_cache[image_name] = normal_image
            return normal_cache[image_name]
        
        chunk_ue_points = []
        chunk_point_info = []
        
        for result in analysis_chunk:
            image_name = result.get('image_name', 'unknown')
            positions_3d = result.get('positions_3d', {})
            
            if not positions_3d:
                continue
            
            alignment_transform = result.get('alignment_transform')
            transform_matrix = None
            if alignment_transform is not None:
                transform_matrix = np.array(alignment_transform)
            
            normal_image = get_cached_normal_data(image_name)
            
            for error_type, regions in positions_3d.items():
                for region_data in regions:
                    ue_point, point_info = PointProcessor._process_single_region(
                        region_data, error_type, image_name, transform_matrix, 
                        normal_image, scale
                    )
                    if ue_point:
                        chunk_ue_points.append(ue_point)
                        chunk_point_info.append(point_info)
        
        return chunk_ue_points, chunk_point_info
    
    @staticmethod
    def _process_single_region(region_data: Dict, error_type: str, image_name: str,
                             transform_matrix: Optional[np.ndarray], 
                             normal_image: Optional[np.ndarray],
                             scale: float) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Process single region"""
        from convert_ue_2_colmap import Config
        from data_processing import AnalysisDataProcessor, AlignmentUtils, NormalImageProcessor
        
        region_size = region_data.get('region_size', 1)
        world_position = region_data.get('world_position', [0, 0, 0])
        error_intensity = AnalysisDataProcessor.extract_error_intensity_improved(region_data)
        
        pixel_center_aligned = region_data.get('pixel_center_aligned', 
                                              [Config.IMAGE_WIDTH//2, Config.IMAGE_HEIGHT//2])
        
        if transform_matrix is not None:
            x_orig, y_orig = AlignmentUtils.transform_to_original_space(
                pixel_center_aligned[0], pixel_center_aligned[1], transform_matrix, 
                Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT)
        else:
            x_orig, y_orig = pixel_center_aligned[0], pixel_center_aligned[1]
        
        x_orig = np.clip(x_orig, 0, Config.IMAGE_WIDTH - 1)
        y_orig = np.clip(y_orig, 0, Config.IMAGE_HEIGHT - 1)
        
        normal_vector_ue = None
        if normal_image is not None:
            normal_vector_ue = NormalImageProcessor.extract_normal_at_pixel(
                normal_image, int(round(x_orig)), int(round(y_orig)))
        
        real_color = region_data.get('real_color', [128, 128, 128])
        
        ue_point = {
            'ue_pos': world_position,
            'error_type': error_type,
            'error_intensity': error_intensity,
            'region_size': region_size,
            'image_name': image_name,
            'pixel_aligned': pixel_center_aligned,
            'pixel_original': [x_orig, y_orig],
            'normal_ue': normal_vector_ue.tolist() if normal_vector_ue is not None else None,
            'has_normal': normal_vector_ue is not None,
            'real_color': real_color
        }
        
        point_info = {
            'error_type': error_type,
            'error_intensity': error_intensity,
            'region_size': region_size,
            'image_name': image_name,
            'original_ue_pos': world_position,
            'pixel_aligned': pixel_center_aligned,
            'pixel_original': [x_orig, y_orig],
            'normal_ue': normal_vector_ue.tolist() if normal_vector_ue is not None else None,
            'normal_colmap': None,
            'has_normal': normal_vector_ue is not None,
            'colmap_pos': None,
            'real_color': real_color
        }
        
        return ue_point, point_info
    
    @staticmethod
    def reduce_nearby_points_optimized(ue_points: List[Dict], point_info: List[Dict], 
                                     min_distance_cm: float) -> Tuple[List[Dict], List[Dict]]:
        """Spatial filtering to reduce nearby points"""
        if len(ue_points) <= 1:
            return ue_points, point_info
        
        print(f"Optimizing {len(ue_points)} points with min distance {min_distance_cm}cm...")
        
        positions = np.array([point['ue_pos'] for point in ue_points])
        intensities = np.array([point['error_intensity'] for point in ue_points])
        
        sorted_indices = np.argsort(-intensities)
        keep_mask = np.ones(len(positions), dtype=bool)
        grid_size = min_distance_cm
        
        spatial_hash = defaultdict(list)
        
        def get_hash_key(pos):
            return tuple((pos / grid_size).astype(int))
        
        kept_count = 0
        for idx in sorted_indices:
            if not keep_mask[idx]:
                continue
                
            pos = positions[idx]
            hash_key = get_hash_key(pos)
            
            should_keep = True
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        neighbor_key = (hash_key[0] + dx, hash_key[1] + dy, hash_key[2] + dz)
                        if neighbor_key in spatial_hash:
                            for neighbor_idx in spatial_hash[neighbor_key]:
                                if np.linalg.norm(pos - positions[neighbor_idx]) < min_distance_cm:
                                    should_keep = False
                                    break
                        if not should_keep:
                            break
                    if not should_keep:
                        break
                if not should_keep:
                    break
            
            if should_keep:
                spatial_hash[hash_key].append(idx)
                kept_count += 1
            else:
                keep_mask[idx] = False
        
        kept_indices = sorted_indices[keep_mask[sorted_indices]]
        kept_ue_points = [ue_points[i] for i in kept_indices]
        kept_point_info = [point_info[i] for i in kept_indices]
        
        print(f"Reduced to {len(kept_ue_points)} points ({kept_count/len(ue_points)*100:.1f}% kept)")
        return kept_ue_points, kept_point_info


class NormalImageProcessor:
    """Handles loading and processing of UE normal images in EXR format"""
    
    _image_cache = {}
    _cache_size_limit = 50
    
    @classmethod
    def _add_to_cache(cls, path: str, image: np.ndarray):
        if len(cls._image_cache) >= cls._cache_size_limit:
            oldest_key = next(iter(cls._image_cache))
            del cls._image_cache[oldest_key]
        cls._image_cache[path] = image
    
    @classmethod
    def load_normal_image_exr(cls, normal_path: str) -> Optional[np.ndarray]:
        if normal_path in cls._image_cache:
            return cls._image_cache[normal_path]
            
        if not Path(normal_path).exists():
            return None
        
        try:
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
            
            cls._add_to_cache(normal_path, normal_image)
            return normal_image
            
        except Exception as e:
            print(f"Error loading EXR normal image {normal_path}: {e}")
            return None
    
    @staticmethod
    def find_normal_image_path(image_name: str, normal_images_dir: str) -> Optional[str]:
        base_name = Path(image_name).stem
        normal_path = Path(normal_images_dir)
        
        exact_match = normal_path / f"{base_name}.exr"
        if exact_match.exists():
            return str(exact_match)
        
        numbers = re.findall(r'\d+', base_name)
        if numbers:
            image_num = int(numbers[-1])
            patterns = [
                f"Normal_Cam00_Pose{image_num:03d}.exr",
                f"Normal_Cam0_Pose{image_num:03d}.exr",
                f"normal_{base_name}.exr"
            ]
            
            for pattern in patterns:
                normal_file = normal_path / pattern
                if normal_file.exists():
                    return str(normal_file)
        
        if not hasattr(NormalImageProcessor, '_exr_files_cache'):
            NormalImageProcessor._exr_files_cache = {
                f.stem: str(f) for f in normal_path.glob("*.exr")
            }
        
        for stem, path in NormalImageProcessor._exr_files_cache.items():
            if base_name in stem:
                return path
        
        return None
    
    @staticmethod
    def decode_ue_normal(encoded_normal: np.ndarray) -> np.ndarray:
        decoded = encoded_normal * 2.0 - 1.0
        length = np.linalg.norm(decoded)
        return decoded / length if length > 1e-6 else np.array([0.0, 0.0, 1.0])
    
    @staticmethod
    def extract_normal_at_pixel(normal_image: np.ndarray, x: int, y: int) -> Optional[np.ndarray]:
        if normal_image is None:
            return None
        
        height, width = normal_image.shape[:2]
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)
        
        encoded_normal = normal_image[y, x, :3]
        return NormalImageProcessor.decode_ue_normal(encoded_normal)


class ImageProcessor:
    """Image processing operations"""
    
    @staticmethod
    def process_real_images(src_dir: str, dst_dir: str, width: int, height: int) -> List[str]:
        img_out = os.path.join(dst_dir, "images")
        os.makedirs(img_out, exist_ok=True)

        extensions = ['*.png', '*.jpg', '*.jpeg']
        rgb_list = []
        for ext in extensions:
            files = sorted(glob.glob(os.path.join(src_dir, ext)))
            if files:
                rgb_list = files
                break
        
        if not rgb_list:
            raise FileNotFoundError(f"No image files found in: {src_dir}")

        print(f"Processing {len(rgb_list)} images...")
        
        image_names = []
        with ThreadPoolExecutor(max_workers=min(8, len(rgb_list))) as executor:
            future_to_id = {}
            
            for new_id, rgb_path in enumerate(rgb_list):
                future = executor.submit(ImageProcessor._process_single_image, 
                                       rgb_path, new_id, img_out, width, height)
                future_to_id[future] = new_id
            
            results = [None] * len(rgb_list)
            for future in as_completed(future_to_id):
                original_id = future_to_id[future]
                result = future.result()
                if result:
                    results[original_id] = result
            
            image_names = [name for name in results if name is not None]
        
        return image_names
    
    @staticmethod
    def _process_single_image(rgb_path: str, new_id: int, img_out: str, width: int, height: int) -> Optional[str]:
        try:
            rgb = cv2.imread(rgb_path)
            if rgb is None:
                return None
                
            rgb_resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            img_name = f"{new_id:04d}.png"
            output_path = os.path.join(img_out, img_name)
            cv2.imwrite(output_path, rgb_resized)
            return img_name
        except Exception as e:
            print(f"Error processing {rgb_path}: {e}")
            return None


class AnalysisDataProcessor:
    """Analysis data processing"""
    
    @staticmethod
    def load_analysis_results(analysis_results_dir: str) -> List[Dict]:
        data_dir = Path(analysis_results_dir) / 'data'
        if not data_dir.exists():
            print(f"ERROR: Data directory not found: {data_dir}")
            return []
        
        json_files = sorted(list(data_dir.glob('*_data.json')))
        print(f"Loading {len(json_files)} analysis files...")
        
        results = []
        with ThreadPoolExecutor(max_workers=min(16, len(json_files))) as executor:
            future_to_file = {
                executor.submit(AnalysisDataProcessor._load_single_json_file, json_file): json_file 
                for json_file in json_files
            }
            
            for future in as_completed(future_to_file):
                data = future.result()
                if data:
                    results.append(data)
        
        return results
    
    @staticmethod
    def _load_single_json_file(json_file: Path) -> Optional[Dict]:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                return data if data else None
        except Exception as e:
            print(f"Warning: Error loading {json_file}: {e}")
            return None
    
    @staticmethod
    def filter_analysis_results(analysis_results: List[Dict]) -> List[Dict]:
        if not analysis_results:
            return []
        
        filtered_results = []
        error_types = Config.ERROR_TYPE_FILTER
        min_size = Config.MIN_REGION_SIZE
        
        for result in analysis_results:
            positions_3d = result.get('positions_3d', {})
            if not positions_3d:
                continue
                
            filtered_positions = {}
            
            for error_type, regions in positions_3d.items():
                if error_types and error_type not in error_types:
                    continue
                
                filtered_regions = [
                    region for region in regions 
                    if region.get('region_size', 0) >= min_size
                ]
                
                if filtered_regions:
                    filtered_positions[error_type] = filtered_regions
            
            if filtered_positions:
                result_copy = result.copy()
                result_copy['positions_3d'] = filtered_positions
                filtered_results.append(result_copy)
        
        return filtered_results

    @staticmethod
    def extract_error_intensity_improved(region_data: Dict) -> float:
        for field in ['error_intensity', 'mean_error', 'avg_error', 'intensity']:
            value = region_data.get(field)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
        
        region_size = region_data.get('region_size', 100)
        return min(1.0, max(0.1, region_size / 5000.0))


class AlignmentUtils:
    """Utilities for handling image alignment transformations"""
    
    @staticmethod
    def transform_to_original_space(x_aligned: float, y_aligned: float, 
                                  transform_matrix: np.ndarray, 
                                  img_width: int, img_height: int) -> Tuple[float, float]:
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