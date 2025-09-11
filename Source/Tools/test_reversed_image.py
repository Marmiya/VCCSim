"""
Optimized full dataset writer - tests 10 coordinate transformations with maximum speed
Creates complete COLMAP datasets using parallel processing and optimized I/O
"""

import numpy as np
import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict
import io

class OptimizedTransformationTester:
    """Optimized tester for coordinate transformations with parallel processing"""
    
    def __init__(self):
        # 10 most likely coordinate transformations to test
        self.transformations = {
            1: {
                "name": "Identity (no change)",
                "coord_matrix": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                "negate_translation": False,
                "transpose_rotation": False,
                "invert_rotation": False
            },
            2: {
                "name": "UE to COLMAP standard",
                "coord_matrix": np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]),
                "negate_translation": True,
                "transpose_rotation": True,
                "invert_rotation": False
            },
            3: {
                "name": "Flip Z axis only",
                "coord_matrix": np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                "negate_translation": True,
                "transpose_rotation": True,
                "invert_rotation": False
            },
            4: {
                "name": "Swap Y and Z axes", 
                "coord_matrix": np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
                "negate_translation": True,
                "transpose_rotation": True,
                "invert_rotation": False
            },
            5: {
                "name": "Flip Y axis only",
                "coord_matrix": np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                "negate_translation": True,
                "transpose_rotation": True,
                "invert_rotation": False
            },
            6: {
                "name": "Flip X axis only",
                "coord_matrix": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                "negate_translation": True,
                "transpose_rotation": True,
                "invert_rotation": False
            },
            7: {
                "name": "Rotate 90° around Z",
                "coord_matrix": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                "negate_translation": True,
                "transpose_rotation": True,
                "invert_rotation": False
            },
            8: {
                "name": "Rotate 180° around Y", 
                "coord_matrix": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                "negate_translation": True,
                "transpose_rotation": True,
                "invert_rotation": False
            },
            9: {
                "name": "UE to ROS (Z up, right-handed)",
                "coord_matrix": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                "negate_translation": False,
                "transpose_rotation": False,
                "invert_rotation": True
            },
            10: {
                "name": "Complete axis reversal",
                "coord_matrix": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                "negate_translation": False,
                "transpose_rotation": False,
                "invert_rotation": True
            }
        }
    
    def create_all_optimized_datasets(self, analysis_dir: str, output_base_dir: str):
        """Create all test datasets using optimized parallel processing"""
        print("=" * 90)
        print("OPTIMIZED COORDINATE TRANSFORMATION TESTER - 10 POSSIBILITIES")
        print("=" * 90)
        print(f"Analysis source: {analysis_dir}")
        print(f"Output base: {output_base_dir}")
        print(f"Testing {len(self.transformations)} coordinate transformations")
        print(f"Using {min(mp.cpu_count(), 10)} parallel workers")
        print("=" * 90)
        
        start_time = time.time()
        
        # Load ALL data once with optimized processing
        print("🔄 Loading and preprocessing ALL data...")
        all_data = self._load_all_data_optimized(analysis_dir)
        
        if not all_data:
            print("❌ Failed to load data!")
            return {}
        
        load_time = time.time() - start_time
        print(f"✅ Data loaded in {load_time:.1f}s")
        print(f"   Points: {len(all_data['points_ue']):,}")
        print(f"   Cameras: {len(all_data['poses']):,}")
        
        # Create base output directory
        base_path = Path(output_base_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Process all transformations in parallel
        print(f"\n🚀 Creating {len(self.transformations)} datasets in parallel...")
        
        results = {}
        with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), len(self.transformations))) as executor:
            # Submit all transformation jobs
            future_to_transform = {}
            for transform_id, transform_info in self.transformations.items():
                dataset_dir = base_path / f"test_{transform_id:02d}_{transform_id}"
                
                future = executor.submit(
                    self._create_single_dataset_optimized,
                    all_data, transform_info, transform_id, str(dataset_dir)
                )
                future_to_transform[future] = transform_id
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_transform):
                transform_id = future_to_transform[future]
                completed += 1
                
                try:
                    result = future.result()
                    results[transform_id] = result
                    status = "✅" if result['success'] else "❌"
                    print(f"{status} Test {transform_id:2d}/10: {self.transformations[transform_id]['name']} "
                          f"({completed}/{len(self.transformations)})")
                    if not result['success']:
                        print(f"     Error: {result.get('error', 'Unknown')}")
                except Exception as e:
                    results[transform_id] = {
                        'success': False,
                        'error': str(e),
                        'transform_id': transform_id
                    }
                    print(f"❌ Test {transform_id:2d}/10: FAILED - {e}")
        
        total_time = time.time() - start_time
        
        # Create optimized summary
        self._create_optimized_summary(base_path, results, all_data, total_time)
        
        successful = sum(1 for r in results.values() if r['success'])
        print(f"\n" + "=" * 90)
        print(f"🎯 COMPLETED {successful}/10 TRANSFORMATIONS IN {total_time:.1f}s")
        print(f"Each dataset: {len(all_data['points_ue']):,} points, {len(all_data['poses']):,} cameras")
        print(f"Output location: {output_base_dir}")
        print(f"Average time per dataset: {total_time/len(self.transformations):.1f}s")
        print(f"\n🔍 TEST INSTRUCTIONS:")
        print(f"1. Open COLMAP GUI")
        print(f"2. Test each test_XX_* folder (sparse/0)")
        print(f"3. Look for: cameras facing points, proper scene orientation")
        print(f"4. The working transformation number is your solution!")
        print("=" * 90)
        
        return results
    
    def _load_all_data_optimized(self, analysis_dir: str) -> Dict:
        """Load all data with optimized parallel processing"""
        try:
            # Load analysis results with parallel processing
            print("   Loading analysis results...")
            from data_processing import AnalysisDataProcessor
            analysis_results = AnalysisDataProcessor.load_analysis_results(analysis_dir)
            filtered_results = AnalysisDataProcessor.filter_analysis_results(analysis_results)
            
            if not filtered_results:
                print("   ❌ No analysis results found!")
                return None
            
            # Extract points with optimized processing
            print("   Extracting points...")
            points_ue, point_info = self._extract_points_optimized(filtered_results)
            
            # Load poses
            print("   Loading camera poses...")
            poses = self._load_poses_optimized()
            
            # Pre-compute camera intrinsics
            K = self._compute_camera_intrinsics()
            
            return {
                'points_ue': points_ue,
                'point_info': point_info,
                'poses': poses,
                'intrinsics': K,
                'analysis_count': len(filtered_results)
            }
            
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return None
    
    def _extract_points_optimized(self, analysis_results: List[Dict]) -> Tuple[List, List]:
        """Optimized point extraction with parallel processing"""
        
        # Use parallel processing for large datasets
        if len(analysis_results) > 100:
            chunk_size = max(10, len(analysis_results) // mp.cpu_count())
            chunks = [analysis_results[i:i + chunk_size] for i in range(0, len(analysis_results), chunk_size)]
            
            all_points_ue = []
            all_point_info = []
            
            with ThreadPoolExecutor(max_workers=min(mp.cpu_count(), len(chunks))) as executor:
                futures = [executor.submit(self._extract_chunk, chunk, len(all_points_ue)) for chunk in chunks]
                
                for future in as_completed(futures):
                    chunk_points, chunk_info = future.result()
                    all_points_ue.extend(chunk_points)
                    all_point_info.extend(chunk_info)
            
            return all_points_ue, all_point_info
        else:
            # Direct processing for small datasets
            return self._extract_chunk(analysis_results, 0)
    
    def _extract_chunk(self, analysis_chunk: List[Dict], point_id_offset: int) -> Tuple[List, List]:
        """Extract points from a chunk of analysis results"""
        chunk_points = []
        chunk_info = []
        
        point_id = point_id_offset
        for result in analysis_chunk:
            image_name = result.get('image_name', 'unknown')
            positions_3d = result.get('positions_3d', {})
            
            for error_type, regions in positions_3d.items():
                for region_data in regions:
                    world_position = region_data.get('world_position', [0, 0, 0])
                    error_intensity = region_data.get('error_intensity', 1.0)
                    region_size = region_data.get('region_size', 1)
                    real_color = region_data.get('real_color', [128, 128, 128])
                    pixel_coords = region_data.get('pixel_center_aligned', [608, 456])
                    
                    chunk_points.append(world_position)
                    chunk_info.append({
                        'point_id': point_id + 1,
                        'error_type': error_type,
                        'error_intensity': error_intensity,
                        'region_size': region_size,
                        'image_name': image_name,
                        'real_color': real_color,
                        'pixel_coords': pixel_coords,
                        'original_ue_pos': world_position
                    })
                    point_id += 1
        
        return chunk_points, chunk_info
    
    def _load_poses_optimized(self) -> List[Dict]:
        """Optimized pose loading"""
        from convert_ue_2_colmap import Config
        
        poses = []
        try:
            with open(Config.UE_POSES_FILE, 'r') as f:
                lines = f.readlines()
                
            # Parse all lines in one go
            for i, line in enumerate(lines):
                vals = line.strip().split()
                if len(vals) == 6:
                    ue_x, ue_y, ue_z, ue_pitch, ue_yaw, ue_roll = [float(v) for v in vals]
                    poses.append({
                        'image_id': i + 1,
                        'ue_x': ue_x, 'ue_y': ue_y, 'ue_z': ue_z,
                        'ue_pitch': ue_pitch, 'ue_yaw': ue_yaw, 'ue_roll': ue_roll,
                        'image_name': f'image_{i:04d}.png'
                    })
        except Exception as e:
            print(f"Warning: Error loading poses: {e}")
            
        return poses
    
    def _compute_camera_intrinsics(self) -> Dict:
        """Compute camera intrinsics"""
        from convert_ue_2_colmap import Config
        
        # Simple pinhole model
        fx = fy = 1000.0  # Reasonable focal length
        cx = Config.IMAGE_WIDTH / 2.0
        cy = Config.IMAGE_HEIGHT / 2.0
        
        return {
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'width': Config.IMAGE_WIDTH,
            'height': Config.IMAGE_HEIGHT
        }
    
    def _create_single_dataset_optimized(self, all_data: Dict, transform_info: Dict, 
                                       transform_id: int, dataset_dir: str) -> Dict:
        """Create a single dataset with optimized processing"""
        try:
            start_time = time.time()
            
            # Create directories
            dataset_path = Path(dataset_dir)
            dataset_path.mkdir(parents=True, exist_ok=True)
            sparse_dir = dataset_path / "sparse" / "0"
            sparse_dir.mkdir(parents=True, exist_ok=True)
            
            # Transform points with vectorized operations
            colmap_points = self._transform_points_vectorized(
                all_data['points_ue'], transform_info)
            
            # Transform camera poses with optimized processing
            colmap_cameras = self._transform_cameras_optimized(
                all_data['poses'], transform_info)
            
            # Create optimized point-camera associations
            image_to_points = self._create_associations_optimized(
                colmap_points, colmap_cameras, all_data['point_info'])
            
            # Write COLMAP files with optimized I/O
            self._write_colmap_files_optimized(
                sparse_dir, colmap_points, all_data['point_info'], 
                colmap_cameras, image_to_points, all_data['intrinsics'])
            
            # Save transformation metadata
            metadata = self._create_transform_metadata(
                transform_info, transform_id, colmap_points, colmap_cameras, all_data)
            
            with open(dataset_path / "transform_info.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'transform_id': transform_id,
                'output_dir': str(dataset_path),
                'num_points': len(colmap_points),
                'num_cameras': len(colmap_cameras),
                'processing_time': processing_time,
                'transform_info': transform_info['name']
            }
            
        except Exception as e:
            return {
                'success': False,
                'transform_id': transform_id,
                'error': str(e)
            }
    
    def _transform_points_vectorized(self, points_ue: List, transform_info: Dict) -> np.ndarray:
        """Vectorized point transformation for maximum speed"""
        # Convert to numpy array for vectorized operations
        points_array = np.array(points_ue, dtype=np.float64)
        coord_matrix = transform_info['coord_matrix'].astype(np.float64)
        scale = 100.0  # cm to m
        
        # Apply transformation to all points at once
        colmap_points = (coord_matrix @ points_array.T).T / scale
        
        return colmap_points
    
    def _transform_cameras_optimized(self, poses: List[Dict], transform_info: Dict) -> List[Dict]:
        """Optimized camera transformation with batch processing"""
        coord_matrix = transform_info['coord_matrix'].astype(np.float64)
        scale = 100.0
        
        colmap_cameras = []
        
        # Pre-compute rotation matrices for all poses
        rotation_matrices = []
        positions = []
        
        for pose in poses:
            # Build UE rotation matrix
            R_ue = self._build_ue_rotation_fast(
                pose['ue_pitch'], pose['ue_yaw'], pose['ue_roll'])
            rotation_matrices.append(R_ue)
            
            # Collect positions for vectorized transformation
            positions.append([pose['ue_x'], pose['ue_y'], pose['ue_z']])
        
        # Vectorized position transformation
        positions_array = np.array(positions, dtype=np.float64)
        colmap_positions = (coord_matrix @ positions_array.T).T / scale
        
        # Process rotations and build camera data
        for i, (pose, R_ue, colmap_pos) in enumerate(zip(poses, rotation_matrices, colmap_positions)):
            # Transform rotation
            R_colmap = coord_matrix @ R_ue @ coord_matrix.T
            
            # Apply transformation options
            if transform_info.get('transpose_rotation', True):
                R_w2c = R_colmap.T
            else:
                R_w2c = R_colmap
                
            if transform_info.get('invert_rotation', False):
                R_w2c = R_w2c.T
            
            # Translation
            if transform_info.get('negate_translation', True):
                t_w2c = -R_w2c @ colmap_pos
            else:
                t_w2c = R_w2c @ colmap_pos
            
            # Convert to quaternion
            qw, qx, qy, qz = self._rotation_to_quaternion_fast(R_w2c)
            
            colmap_cameras.append({
                'image_id': pose['image_id'],
                'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                'tx': t_w2c[0], 'ty': t_w2c[1], 'tz': t_w2c[2],
                'camera_id': 1,
                'name': pose['image_name']
            })
        
        return colmap_cameras
    
    def _build_ue_rotation_fast(self, pitch: float, yaw: float, roll: float) -> np.ndarray:
        """Fast UE rotation matrix construction"""
        p, y, r = np.radians([pitch, yaw, roll])
        
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y) 
        cr, sr = np.cos(r), np.sin(r)
        
        # Pre-computed combined rotation matrix (more efficient than matrix multiplication)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr           ]
        ], dtype=np.float64)
    
    def _rotation_to_quaternion_fast(self, R: np.ndarray) -> Tuple[float, float, float, float]:
        """Fast rotation to quaternion conversion"""
        trace = R[0,0] + R[1,1] + R[2,2]
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            return (0.25 * s, (R[2,1] - R[1,2]) / s, 
                   (R[0,2] - R[2,0]) / s, (R[1,0] - R[0,1]) / s)
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            return ((R[2,1] - R[1,2]) / s, 0.25 * s,
                   (R[0,1] + R[1,0]) / s, (R[0,2] + R[2,0]) / s)
        elif R[1,1] > R[2,2]:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            return ((R[0,2] - R[2,0]) / s, (R[0,1] + R[1,0]) / s,
                   0.25 * s, (R[1,2] + R[2,1]) / s)
        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            return ((R[1,0] - R[0,1]) / s, (R[0,2] + R[2,0]) / s,
                   (R[1,2] + R[2,1]) / s, 0.25 * s)
    
    def _create_associations_optimized(self, points: np.ndarray, cameras: List[Dict], 
                                     point_info: List[Dict]) -> Dict[int, List[int]]:
        """Optimized point-camera associations"""
        image_to_points = defaultdict(list)
        
        # Distribute points across cameras more intelligently
        max_points_per_camera = max(20, len(points) // len(cameras))
        
        for i, info in enumerate(point_info):
            # Try to match by image name pattern
            image_name = info.get('image_name', '')
            assigned_camera = (i % len(cameras)) + 1
            
            # Simple pattern matching for better associations
            if image_name:
                try:
                    # Extract number from image name
                    import re
                    numbers = re.findall(r'\d+', image_name)
                    if numbers:
                        img_num = int(numbers[-1]) % len(cameras)
                        assigned_camera = img_num + 1
                except:
                    pass
            
            # Limit points per camera
            if len(image_to_points[assigned_camera]) < max_points_per_camera:
                image_to_points[assigned_camera].append(i)
        
        return dict(image_to_points)
    
    def _write_colmap_files_optimized(self, sparse_dir: Path, points: np.ndarray, 
                                     point_info: List[Dict], cameras: List[Dict],
                                     image_to_points: Dict[int, List[int]], 
                                     intrinsics: Dict):
        """Optimized COLMAP file writing with batch I/O"""
        
        # Write all files in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all writing tasks
            futures = []
            
            futures.append(executor.submit(
                self._write_cameras_bin_fast, sparse_dir, intrinsics))
            
            futures.append(executor.submit(
                self._write_images_bin_fast, sparse_dir, cameras, image_to_points, point_info))
            
            futures.append(executor.submit(
                self._write_points_bin_fast, sparse_dir, points, point_info, image_to_points))
            
            # Wait for all to complete
            for future in as_completed(futures):
                future.result()  # Raises exception if any failed
    
    def _write_cameras_bin_fast(self, sparse_dir: Path, intrinsics: Dict):
        """Fast cameras.bin writing"""
        with open(sparse_dir / "cameras.bin", 'wb') as f:
            data = struct.pack('<Q', 1)  # 1 camera
            data += struct.pack('<I', 1)  # camera_id
            data += struct.pack('<I', 1)  # PINHOLE model
            data += struct.pack('<Q', intrinsics['width'])
            data += struct.pack('<Q', intrinsics['height'])
            data += struct.pack('<dddd', intrinsics['fx'], intrinsics['fy'], 
                              intrinsics['cx'], intrinsics['cy'])
            f.write(data)
    
    def _write_images_bin_fast(self, sparse_dir: Path, cameras: List[Dict], 
                              image_to_points: Dict, point_info: List[Dict]):
        """Fast images.bin writing with batched I/O"""
        with open(sparse_dir / "images.bin", 'wb') as f:
            # Pre-build all data in memory for maximum I/O efficiency
            buffer = io.BytesIO()
            buffer.write(struct.pack('<Q', len(cameras)))
            
            for cam in cameras:
                buffer.write(struct.pack('<I', cam['image_id']))
                buffer.write(struct.pack('<dddd', cam['qw'], cam['qx'], cam['qy'], cam['qz']))
                buffer.write(struct.pack('<ddd', cam['tx'], cam['ty'], cam['tz']))
                buffer.write(struct.pack('<I', cam['camera_id']))
                buffer.write(cam['name'].encode('utf-8') + b'\0')
                
                # 2D point observations
                point_indices = image_to_points.get(cam['image_id'], [])
                buffer.write(struct.pack('<Q', len(point_indices)))
                
                for point_idx in point_indices:
                    if point_idx < len(point_info):
                        pixel_coords = point_info[point_idx].get('pixel_coords', [608, 456])
                        x_2d = float(np.clip(pixel_coords[0], 0, 1215))
                        y_2d = float(np.clip(pixel_coords[1], 0, 911))
                        buffer.write(struct.pack('<dd', x_2d, y_2d))
                        buffer.write(struct.pack('<Q', point_idx + 1))
            
            # Single write operation
            f.write(buffer.getvalue())
            buffer.close()
    
    def _write_points_bin_fast(self, sparse_dir: Path, points: np.ndarray, 
                              point_info: List[Dict], image_to_points: Dict):
        """Fast points3D.bin writing with batched I/O"""
        with open(sparse_dir / "points3D.bin", 'wb') as f:
            # Pre-build point-to-images lookup
            point_to_images = defaultdict(list)
            for img_id, point_list in image_to_points.items():
                for point_idx in point_list:
                    point_to_images[point_idx].append(img_id)
            
            # Build all data in memory first
            buffer = io.BytesIO()
            buffer.write(struct.pack('<Q', len(points)))
            
            for i, (point, info) in enumerate(zip(points, point_info)):
                buffer.write(struct.pack('<Q', i + 1))  # point3D_id
                buffer.write(struct.pack('<ddd', point[0], point[1], point[2]))
                
                color = info.get('real_color', [128, 128, 128])
                r = max(1, min(255, int(color[0])))
                g = max(1, min(255, int(color[1])))
                b = max(1, min(255, int(color[2])))
                buffer.write(struct.pack('<BBB', r, g, b))
                
                error = info.get('error_intensity', 1.0)
                buffer.write(struct.pack('<d', error))
                
                # Track information
                observing_images = point_to_images.get(i, [])
                buffer.write(struct.pack('<Q', len(observing_images)))
                
                for img_id in observing_images:
                    try:
                        point_2d_idx = image_to_points[img_id].index(i)
                        buffer.write(struct.pack('<I', img_id))
                        buffer.write(struct.pack('<I', point_2d_idx))
                    except (ValueError, KeyError):
                        buffer.write(struct.pack('<I', img_id))
                        buffer.write(struct.pack('<I', 0))
            
            # Single write operation
            f.write(buffer.getvalue())
            buffer.close()
    
    def _create_transform_metadata(self, transform_info: Dict, transform_id: int,
                                  colmap_points: np.ndarray, colmap_cameras: List[Dict],
                                  all_data: Dict) -> Dict:
        """Create transformation metadata"""
        return {
            'transform_id': transform_id,
            'transform_name': transform_info['name'],
            'transformation_matrix': transform_info['coord_matrix'].tolist(),
            'negate_translation': transform_info['negate_translation'],
            'transpose_rotation': transform_info['transpose_rotation'],
            'invert_rotation': transform_info.get('invert_rotation', False),
            'num_points': len(colmap_points),
            'num_cameras': len(colmap_cameras),
            'point_cloud_bounds': {
                'min': colmap_points.min(axis=0).tolist(),
                'max': colmap_points.max(axis=0).tolist(),
                'center': colmap_points.mean(axis=0).tolist(),
                'scale': float(np.linalg.norm(colmap_points.max(axis=0) - colmap_points.min(axis=0)))
            },
            'camera_stats': {
                'positions': {
                    'min': [min(cam['tx'] for cam in colmap_cameras),
                           min(cam['ty'] for cam in colmap_cameras),
                           min(cam['tz'] for cam in colmap_cameras)],
                    'max': [max(cam['tx'] for cam in colmap_cameras),
                           max(cam['ty'] for cam in colmap_cameras),
                           max(cam['tz'] for cam in colmap_cameras)]
                }
            },
            'source_analysis_results': all_data['analysis_count']
        }
    
    def _create_optimized_summary(self, base_path: Path, results: Dict, 
                                 all_data: Dict, total_time: float):
        """Create comprehensive summary with performance metrics"""
        
        summary_data = {
            'creation_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_processing_time': total_time,
            'transformations_tested': len(self.transformations),
            'successful_datasets': sum(1 for r in results.values() if r['success']),
            'total_points': len(all_data['points_ue']),
            'total_cameras': len(all_data['poses']),
            'source_analysis_results': all_data['analysis_count'],
            'performance': {
                'average_time_per_dataset': total_time / len(self.transformations),
                'points_per_second': len(all_data['points_ue']) * len(self.transformations) / total_time,
                'parallel_workers_used': min(mp.cpu_count(), len(self.transformations))
            },
            'results': results
        }
        
        # Write JSON summary
        with open(base_path / "optimization_summary.json", 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Write human-readable summary
        with open(base_path / "test_results_summary.txt", 'w') as f:
            f.write("OPTIMIZED COORDINATE TRANSFORMATION TEST RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Processing completed in {total_time:.1f} seconds\n")
            f.write(f"Tested {len(self.transformations)} coordinate transformations\n")
            f.write(f"Each dataset: {len(all_data['points_ue']):,} points, {len(all_data['poses']):,} cameras\n")
            f.write(f"Performance: {summary_data['performance']['points_per_second']:.0f} points/sec\n\n")
            
            f.write("TRANSFORMATION RESULTS:\n")
            f.write("-" * 40 + "\n")
            for transform_id in sorted(results.keys()):
                result = results[transform_id]
                transform_name = self.transformations[transform_id]['name']
                
                if result['success']:
                    f.write(f"✅ Test {transform_id:2d}: {transform_name}\n")
                    f.write(f"    Output: test_{transform_id:02d}_{transform_id}/sparse/0\n")
                    if 'processing_time' in result:
                        f.write(f"    Time: {result['processing_time']:.1f}s\n")
                else:
                    f.write(f"❌ Test {transform_id:2d}: {transform_name}\n")
                    f.write(f"    Error: {result.get('error', 'Unknown')}\n")
                f.write("\n")
            
            f.write("COLMAP TESTING INSTRUCTIONS:\n")
            f.write("-" * 40 + "\n")
            f.write("1. Open COLMAP GUI\n")
            f.write("2. For each successful test_XX_* folder:\n")
            f.write("   - File → Import Model\n")
            f.write("   - Navigate to the sparse/0 subfolder\n")
            f.write("   - Import the reconstruction\n")
            f.write("   - Check if:\n")
            f.write("     • Cameras are visible and properly oriented\n")
            f.write("     • Cameras point towards the point cloud\n")
            f.write("     • Scene has correct scale and orientation\n")
            f.write("     • No extreme outliers or flipped coordinates\n")
            f.write("3. The working transformation number is your solution!\n")
            f.write("4. Use that transformation in your main dataset creation script\n")


def main():
    """Main function with optimized argument handling"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python optimized_test_script.py <analysis_results_dir> <output_dir>")
        print("Example: python optimized_test_script.py ./Logs/analysis_results/20231201_120000 ./coordinate_tests")
        print("\nThis will create 10 test datasets with different coordinate transformations.")
        print("Test each one in COLMAP to find the working transformation.")
        sys.exit(1)
    
    analysis_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Validate input
    if not Path(analysis_dir).exists():
        print(f"❌ ERROR: Analysis directory does not exist: {analysis_dir}")
        sys.exit(1)
    
    data_dir = Path(analysis_dir) / 'data'
    if not data_dir.exists() or not list(data_dir.glob('*_data.json')):
        print(f"❌ ERROR: No analysis data found in: {data_dir}")
        sys.exit(1)
    
    # Create tester and run
    tester = OptimizedTransformationTester()
    results = tester.create_all_optimized_datasets(analysis_dir, output_dir)
    
    # Final summary
    successful = sum(1 for r in results.values() if r['success'])
    print(f"\n🏁 FINAL RESULT: {successful}/10 datasets created successfully")
    
    if successful > 0:
        print(f"✅ Ready for COLMAP testing!")
        print(f"📁 Location: {output_dir}")
        print(f"🔍 Test each test_XX_* folder in COLMAP GUI")
    else:
        print(f"❌ No datasets were created successfully")
        print(f"Check error messages above and verify input data")


if __name__ == "__main__":
    # Set up multiprocessing
    mp.freeze_support()
    if mp.get_start_method(allow_none=True) != 'spawn':
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    main()