"""Output generation module with COLMAP conventions"""

import json
import struct
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Optional
from plyfile import PlyData, PlyElement
import re
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

from convert_ue_2_colmap import Config
from data_processing import (
    AnalysisDataProcessor, PointProcessor, ImageProcessor, 
    CameraUtils, CoordinateConverter
)

import io
import time


class FileCreator:
    """File creation with COLMAP conventions"""
    
    @staticmethod
    def create_points3d_bin(colmap_points: np.ndarray, point_info: List[Dict], 
                          image_to_points: Dict, output_path: str):
        """Create points3D.bin with proper COLMAP format"""
        print(f"Writing {len(colmap_points):,} points to {output_path}")
        
        if len(colmap_points) != len(point_info):
            raise ValueError(f"Point count mismatch: {len(colmap_points)} vs {len(point_info)}")
        
        # Build point-to-images mapping for tracks
        point_to_images = {}
        for img_id, point_indices in image_to_points.items():
            for point_idx in point_indices:
                if point_idx < len(point_info):
                    point_to_images.setdefault(point_idx, []).append(img_id)
        
        with open(output_path, 'wb') as f:
            f.write(struct.pack('<Q', len(colmap_points)))
            
            for point_idx, (point_pos, info) in enumerate(zip(colmap_points, point_info)):
                point_id = point_idx + 1
                
                x, y, z = float(point_pos[0]), float(point_pos[1]), float(point_pos[2])
                
                real_color = info.get('real_color', [128, 128, 128])
                r = max(1, min(255, int(real_color[0])))
                g = max(1, min(255, int(real_color[1])))
                b = max(1, min(255, int(real_color[2])))
                
                error = float(info.get('error_intensity', 1.0))
                
                # Write point data
                f.write(struct.pack('<Q', point_id))
                f.write(struct.pack('<ddd', x, y, z))
                f.write(struct.pack('<BBB', r, g, b))
                f.write(struct.pack('<d', error))
                
                # Write track (observations)
                observing_images = point_to_images.get(point_idx, [])
                if observing_images:
                    f.write(struct.pack('<Q', len(observing_images)))
                    for img_id in observing_images:
                        f.write(struct.pack('<I', img_id))
                        f.write(struct.pack('<I', point_idx))
                else:
                    f.write(struct.pack('<Q', 1))
                    f.write(struct.pack('<I', 1))
                    f.write(struct.pack('<I', point_idx))
    
    @staticmethod
    def create_images_bin(ue_poses_file: str, image_names: List[str], K: np.ndarray,
                        point_info: List[Dict], image_to_points: Dict,
                        scale: float, output_path: str):
        """Create images.bin with COLMAP camera pose convention"""
        print(f"Creating images.bin...")
        
        # Load UE poses
        poses = []
        with open(ue_poses_file, 'r') as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) == 6:
                    poses.append([float(v) for v in vals])
        
        num_images = min(len(poses), len(image_names))
        
        with open(output_path, 'wb') as f:
            f.write(struct.pack('<Q', num_images))
            
            total_observations = 0
            images_with_points = 0
            
            for idx, (pose, img_name) in enumerate(zip(poses[:num_images], image_names[:num_images])):
                image_id = idx + 1
                ue_x, ue_y, ue_z, ue_pitch, ue_yaw, ue_roll = pose
                
                # Convert UE pose to COLMAP world-to-camera pose
                R_w2c, t_w2c = CoordinateConverter.ue_pose_to_colmap_camera_pose(
                    ue_x, ue_y, ue_z, ue_pitch, ue_yaw, ue_roll, scale)
                
                qw, qx, qy, qz = CameraUtils.rotation_matrix_to_quaternion(R_w2c)
                
                # Write image header
                f.write(struct.pack('<I', image_id))
                f.write(struct.pack('<dddd', qw, qx, qy, qz))
                f.write(struct.pack('<ddd', t_w2c[0], t_w2c[1], t_w2c[2]))
                f.write(struct.pack('<I', 1))
                
                # Write image name
                img_name_bytes = img_name.encode('utf-8') + b'\0'
                f.write(img_name_bytes)
                
                # Write 2D points
                point_indices = image_to_points.get(image_id, [])
                if point_indices:
                    images_with_points += 1
                    valid_points = [pi for pi in point_indices if pi < len(point_info)]
                    
                    f.write(struct.pack('<Q', len(valid_points)))
                    
                    for point_idx in valid_points:
                        info = point_info[point_idx]
                        pixel_coords = info.get('pixel_original', [Config.IMAGE_WIDTH//2, Config.IMAGE_HEIGHT//2])
                        
                        x_2d = float(np.clip(pixel_coords[0], 0, Config.IMAGE_WIDTH - 1))
                        y_2d = float(np.clip(pixel_coords[1], 0, Config.IMAGE_HEIGHT - 1))
                        point_3d_id = point_idx + 1
                        
                        f.write(struct.pack('<dd', x_2d, y_2d))
                        f.write(struct.pack('<Q', point_3d_id))
                        total_observations += 1
                else:
                    f.write(struct.pack('<Q', 0))
        
        print(f"  Written {total_observations:,} 2D observations across {images_with_points} images")
    
    @staticmethod
    def create_cameras_bin(K: np.ndarray, width: int, height: int, output_path: str):
        """Create cameras.bin file (PINHOLE model)"""
        print(f"Creating cameras.bin for {width}x{height} images")
        
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        
        with open(output_path, 'wb') as f:
            f.write(struct.pack('<Q', 1))
            f.write(struct.pack('<I', 1))
            f.write(struct.pack('<I', 1))
            f.write(struct.pack('<Q', width))
            f.write(struct.pack('<Q', height))
            f.write(struct.pack('<dddd', fx, fy, cx, cy))
    
    @staticmethod
    def create_image_to_point_mapping(point_info: List[Dict], image_names: List[str]) -> Dict[int, List[int]]:
        """Create mapping from image IDs to point indices"""
        print(f"Creating image-to-point mapping...")
        
        # Build efficient name-to-ID lookup
        name_to_id = {}
        for img_id, img_name in enumerate(image_names):
            image_id = img_id + 1
            base_name = Path(img_name).stem
            
            name_to_id[base_name] = image_id
            name_to_id[img_name] = image_id
            name_to_id[Path(img_name).name] = image_id
            
            # Extract numbers for flexible matching
            numbers = re.findall(r'\d+', base_name)
            for num_str in numbers:
                try:
                    num = int(num_str)
                    for key in [num_str, f"{num:03d}", f"{num:04d}", f"{num:05d}"]:
                        if key not in name_to_id:
                            name_to_id[key] = image_id
                except ValueError:
                    continue
        
        # Map points to their source images
        image_to_points = defaultdict(list)
        matched_count = 0
        
        for point_idx, info in enumerate(point_info):
            image_name = info.get('image_name', '')
            
            if not image_name:
                continue
            
            matching_image_id = FileCreator._find_matching_image_id(image_name, name_to_id)
            
            if matching_image_id:
                image_to_points[matching_image_id].append(point_idx)
                matched_count += 1
        
        print(f"  Matched points: {matched_count:,}/{len(point_info):,}")
        
        return dict(image_to_points)
    
    @staticmethod
    def _find_matching_image_id(image_name: str, name_to_id: Dict[str, int]) -> Optional[int]:
        """Find matching image ID with fallback strategies"""
        clean_name = Path(image_name).stem.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        
        # Try direct matches first
        if clean_name in name_to_id:
            return name_to_id[clean_name]
        if image_name in name_to_id:
            return name_to_id[image_name]
        
        # Try number extraction
        numbers = re.findall(r'\d+', clean_name)
        for num_str in numbers:
            if num_str in name_to_id:
                return name_to_id[num_str]
            try:
                num = int(num_str)
                for formatted in [f"{num:03d}", f"{num:04d}", f"{num:05d}"]:
                    if formatted in name_to_id:
                        return name_to_id[formatted]
            except ValueError:
                continue
        
        return None
    
    @staticmethod
    def create_ply_file(points: np.ndarray, point_info: List[Dict], 
                      output_path: str, coordinate_system: str = 'colmap',
                      color_mode: str = 'real'):
        """Create PLY file"""
        print(f"Creating {coordinate_system} PLY file: {output_path}")
        
        vertices_data = []
        
        for i, info in enumerate(point_info):
            # Get position based on coordinate system
            if coordinate_system == 'colmap':
                pos = points[i] if len(points) > i else [0, 0, 0]
            else:  # 'ue' coordinate system
                ue_pos = info.get('original_ue_pos', [0, 0, 0])
                pos = ue_pos
            
            # Get color
            if color_mode == 'real':
                color = info.get('real_color', [128, 128, 128])
            else:  # error color
                error_type = info.get('error_type', 'unknown')
                error_intensity = info.get('error_intensity', 1.0)
                color = VisualizationUtils.error_intensity_to_color(error_intensity, error_type)
            
            # Get normal based on coordinate system
            if coordinate_system == 'colmap':
                normal = info.get('normal_colmap', [0.0, 0.0, 1.0])
            else:  # 'ue'
                normal = info.get('normal_ue', [0.0, 0.0, 1.0])
            
            if normal is None:
                normal = [0.0, 0.0, 1.0]
            
            r = max(1, min(255, int(color[0])))
            g = max(1, min(255, int(color[1])))
            b = max(1, min(255, int(color[2])))
            
            vertices_data.append((
                float(pos[0]), float(pos[1]), float(pos[2]),
                float(normal[0]), float(normal[1]), float(normal[2]),
                r, g, b
            ))
        
        vertices = np.array(vertices_data, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        
        el = PlyElement.describe(vertices, 'vertex')
        PlyData([el]).write(output_path)


class VisualizationUtils:
    """Visualization utilities"""
    
    @staticmethod
    def error_intensity_to_color(error_intensity: float, error_type: str) -> List[int]:
        """Convert error intensity to RGB color based on error type"""
        color_schemes = {
            'mse': ([255, 0, 0], [255, 100, 100]),
            'mae': ([255, 128, 0], [255, 200, 128]),
            'lab': ([255, 255, 0], [255, 255, 128]),
            'gradient': ([0, 255, 0], [128, 255, 128])
        }
        
        base_color, light_color = color_schemes.get(error_type.lower(), ([128, 128, 128], [200, 200, 200]))
        intensity = max(0.3, min(1.0, float(error_intensity)))
        
        return [
            int(light_color[0] + (base_color[0] - light_color[0]) * intensity),
            int(light_color[1] + (base_color[1] - light_color[1]) * intensity), 
            int(light_color[2] + (base_color[2] - light_color[2]) * intensity)
        ]


def create_text_files(sparse_dir, colmap_points, point_info, image_to_points, 
                     ue_poses_file, image_names, K, scale):
    """Create COLMAP text files"""
    if len(colmap_points) > 100000:
        print(f"  Skipping text files for large dataset ({len(colmap_points):,} points)")
        return
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.submit(_create_cameras_txt, sparse_dir, K)
        executor.submit(_create_images_txt, sparse_dir, ue_poses_file, image_names, 
                       point_info, image_to_points, scale)
        executor.submit(_create_points3d_txt, sparse_dir, colmap_points, 
                       point_info, image_to_points)

def _create_cameras_txt(sparse_dir, K):
    """Create cameras.txt"""
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    content = (f"# Camera list with one line of data per camera:\n"
               f"# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
               f"1 PINHOLE {Config.IMAGE_WIDTH} {Config.IMAGE_HEIGHT} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
    with open(sparse_dir / "cameras.txt", 'w') as f:
        f.write(content)

def _create_images_txt(sparse_dir, ue_poses_file, image_names, point_info, image_to_points, scale):
    """Create images.txt"""
    poses = []
    with open(ue_poses_file, 'r') as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) == 6:
                poses.append([float(v) for v in vals])
    
    lines = ["# Image list with two lines of data per image:\n",
             "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
             "# POINTS2D[] as (X, Y, POINT3D_ID)\n"]
    
    for idx, (pose, img_name) in enumerate(zip(poses[:len(image_names)], image_names)):
        image_id = idx + 1
        ue_x, ue_y, ue_z, ue_pitch, ue_yaw, ue_roll = pose
        
        R_w2c, t_w2c = CoordinateConverter.ue_pose_to_colmap_camera_pose(
            ue_x, ue_y, ue_z, ue_pitch, ue_yaw, ue_roll, scale)
        
        qw, qx, qy, qz = CameraUtils.rotation_matrix_to_quaternion(R_w2c)
        
        lines.append(f"{image_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{t_w2c[0]:.9f} {t_w2c[1]:.9f} {t_w2c[2]:.9f} 1 {img_name}\n")
        
        # 2D points
        point_indices = image_to_points.get(image_id, [])
        if point_indices:
            valid_points = [pi for pi in point_indices if pi < len(point_info)]
            points_2d = []
            for pi in valid_points:
                pixel_coords = point_info[pi].get('pixel_original', [Config.IMAGE_WIDTH//2, Config.IMAGE_HEIGHT//2])
                x_2d = np.clip(pixel_coords[0], 0, Config.IMAGE_WIDTH - 1)
                y_2d = np.clip(pixel_coords[1], 0, Config.IMAGE_HEIGHT - 1)
                points_2d.append(f"{x_2d:.6f} {y_2d:.6f} {pi + 1}")
            lines.append(" ".join(points_2d) + "\n")
        else:
            lines.append("\n")
    
    with open(sparse_dir / "images.txt", 'w') as f:
        f.writelines(lines)

def _create_points3d_txt(sparse_dir, colmap_points, point_info, image_to_points):
    """Create points3D.txt"""
    point_to_images = {}
    for img_id, point_indices in image_to_points.items():
        for pi in point_indices:
            point_to_images.setdefault(pi, []).append(img_id)
    
    output = io.StringIO()
    output.write("# 3D point list with one line of data per point:\n")
    output.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    
    for i, (point_pos, info) in enumerate(zip(colmap_points, point_info)):
        point_id = i + 1
        x, y, z = point_pos
        
        real_color = info.get('real_color', [128, 128, 128])
        r = max(1, min(255, int(real_color[0])))
        g = max(1, min(255, int(real_color[1])))
        b = max(1, min(255, int(real_color[2])))
        error = info.get('error_intensity', 1.0)
        
        observing_images = point_to_images.get(i, [])
        if observing_images:
            track_str = " ".join(f"{img_id} {i}" for img_id in observing_images)
        else:
            track_str = f"1 {i}"
        
        output.write(f"{point_id} {x:.9f} {y:.9f} {z:.9f} {r} {g} {b} {error:.6f} {track_str}\n")
    
    with open(sparse_dir / "points3D.txt", 'w') as f:
        f.write(output.getvalue())
    output.close()


class PointCloudDatasetCreator:
    """Dataset creator with COLMAP conventions"""
    
    @staticmethod
    def create_pointcloud_dataset_from_error_analysis(
        analysis_results_dir: str, real_images_dir: str, normal_images_dir: str, 
        ue_poses_file: str, output_dir: str,
        width: int = None, height: int = None, 
        hfov_deg: float = None, scale: float = None
    ):
        """Create dataset with COLMAP coordinate conventions"""
        
        width = width or Config.IMAGE_WIDTH
        height = height or Config.IMAGE_HEIGHT
        hfov_deg = hfov_deg or Config.HFOV_DEGREES
        scale = scale or Config.COORDINATE_SCALE
        
        print("Creating point cloud dataset with COLMAP conventions")
        
        # Create output directories
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        sparse_dir = Path(output_dir) / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and filter analysis results
        print("Loading analysis results...")
        analysis_results = AnalysisDataProcessor.load_analysis_results(analysis_results_dir)
        if not analysis_results:
            raise ValueError("No analysis results found!")
        
        filtered_results = AnalysisDataProcessor.filter_analysis_results(analysis_results)
        print(f"Kept {len(filtered_results)}/{len(analysis_results)} analysis results")
        
        # Extract points with coordinate conversion
        print("Extracting error points...")
        colmap_points, point_info, ue_points = PointProcessor.extract_error_points(
            filtered_results, normal_images_dir, scale)
        
        if len(colmap_points) == 0:
            print("ERROR: No points remaining after filtering!")
            return
        
        # Apply final point limit
        if Config.APPLY_FINAL_LIMIT and len(colmap_points) > Config.FINAL_MAX_POINTS:
            print(f"Applying final point limit ({Config.FINAL_MAX_POINTS:,})...")
            combined = list(zip(colmap_points, point_info, ue_points))
            combined.sort(key=lambda x: x[1]['error_intensity'], reverse=True)
            
            colmap_points = np.array([x[0] for x in combined[:Config.FINAL_MAX_POINTS]])
            point_info = [x[1] for x in combined[:Config.FINAL_MAX_POINTS]]
            ue_points = [x[2] for x in combined[:Config.FINAL_MAX_POINTS]]
        
        points_with_normals = sum(1 for info in point_info if info['has_normal'])
        print(f"Final point count: {len(colmap_points):,} error points")
        print(f"Points with surface normals: {points_with_normals:,}/{len(colmap_points):,}")
        
        # Process images
        print("Processing images...")
        K = CameraUtils.compute_intrinsics(width, height, hfov_deg)
        image_names = ImageProcessor.process_real_images(real_images_dir, output_dir, width, height)
        
        # Create image-to-point mapping
        print("Creating image-to-point mapping...")
        image_to_points = FileCreator.create_image_to_point_mapping(point_info, image_names)
        
        # Create output files
        print("Creating output files...")
        
        # COLMAP binary files
        FileCreator.create_points3d_bin(colmap_points, point_info, image_to_points, 
                                      sparse_dir / "points3D.bin")
        FileCreator.create_cameras_bin(K, width, height, sparse_dir / "cameras.bin")
        FileCreator.create_images_bin(ue_poses_file, image_names, K, point_info, 
                                    image_to_points, scale, sparse_dir / "images.bin")
        
        # PLY files for visualization
        FileCreator.create_ply_file(
            colmap_points, point_info, sparse_dir / "points3D.ply", 'colmap', 'real')
        
        FileCreator.create_ply_file(
            colmap_points, point_info, Path(output_dir) / "error_points_ue_real_colors.ply", 'ue', 'real')
        
        FileCreator.create_ply_file(
            colmap_points, point_info, Path(output_dir) / "error_points_colmap_error_colors.ply", 'colmap', 'error')
        
        # Text files
        print("Creating COLMAP text files...")
        create_text_files(sparse_dir, colmap_points, point_info, image_to_points, 
                         ue_poses_file, image_names, K, scale)
        
        # Create metadata
        metadata = PointCloudDatasetCreator._create_metadata(
            analysis_results_dir, real_images_dir, normal_images_dir,
            len(image_names), len(colmap_points), points_with_normals,
            scale, image_to_points, point_info)
        
        with open(Path(output_dir) / "dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Print summary
        PointCloudDatasetCreator._print_summary(output_dir, metadata)
        
        print("Dataset creation completed successfully!")
    
    @staticmethod
    def _create_metadata(analysis_results_dir: str, real_images_dir: str, normal_images_dir: str,
                        num_images: int, num_points: int, points_with_normals: int,
                        scale: float, image_to_points: Dict, point_info: List[Dict]) -> Dict:
        """Create comprehensive metadata"""
        
        error_distribution = defaultdict(int)
        for info in point_info:
            error_distribution[info['error_type']] += 1
        
        return {
            'creation_time': datetime.now().isoformat(),
            'coordinate_system': 'COLMAP',
            'coordinate_conversion': f'UE_to_COLMAP_flip_Z_scale_{scale}',
            'coordinate_details': {
                'transformation': 'X→X, Y→Y, Z→-Z',
                'ue_system': 'X=forward, Y=right, Z=up (cm)',
                'colmap_system': 'X=forward, Y=right, Z=down (m)', 
                'camera_poses': 'world-to-camera (R_w2c, t_w2c)',
                'point_coordinates': 'world_frame_consistent_with_cameras'
            },
            'source_analysis_dir': str(analysis_results_dir),
            'training_images_source': str(real_images_dir),
            'normal_images_source': str(normal_images_dir),
            'num_images': num_images,
            'num_cameras': num_images,
            'num_error_points': num_points,
            'num_points_with_normals': points_with_normals,
            'normal_coverage_percentage': points_with_normals/num_points*100 if num_points > 0 else 0,
            'coordinate_scale': scale,
            'track_information': {
                'images_with_points': len(image_to_points),
                'total_track_observations': sum(len(points) for points in image_to_points.values())
            },
            'filtering_applied': {
                'error_types': Config.ERROR_TYPE_FILTER,
                'min_region_size': Config.MIN_REGION_SIZE,
                'max_points': Config.MAX_POINTS,
                'min_point_distance_cm': Config.MIN_POINT_DISTANCE_CM,
                'final_max_points': Config.FINAL_MAX_POINTS,
                'final_limit_applied': Config.APPLY_FINAL_LIMIT,
            },
            'error_point_distribution': dict(error_distribution)
        }
    
    @staticmethod
    def _print_summary(output_dir: str, metadata: Dict):
        """Print creation summary"""
        print(f"\nOutput: {output_dir}")
        print(f"Images: {metadata['num_images']:,}")
        print(f"Error points: {metadata['num_error_points']:,}")
        print(f"Points with normals: {metadata['num_points_with_normals']:,}/{metadata['num_error_points']:,} "
              f"({metadata['normal_coverage_percentage']:.1f}%)")
        print(f"Images with observations: {metadata['track_information']['images_with_points']:,}")
        print(f"Total track observations: {metadata['track_information']['total_track_observations']:,}")
        
        print(f"\nError distribution:")
        for error_type, count in metadata['error_point_distribution'].items():
            print(f"  {error_type}: {count:,} points")