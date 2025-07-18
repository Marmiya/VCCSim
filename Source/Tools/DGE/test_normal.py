#!/usr/bin/env python3
"""
Normal Extraction Script for UE SCS_Normal format
Extracts and converts normals from UE world space to COLMAP coordinates
"""

import numpy as np
import json
import cv2
from pathlib import Path
from plyfile import PlyData, PlyElement
import sys

# Try to import OpenEXR
try:
    import OpenEXR
    import Imath
    HAS_EXR = True
except ImportError:
    print("Warning: OpenEXR not available. Install with: pip install OpenEXR")
    HAS_EXR = False

# Configuration
ANALYSIS_RESULTS_DIR = "./Logs/analysis_results"
NORMAL_IMAGES_DIR = "./Source/Tools/DGE/data/normal_images"
IMAGE_WIDTH = 1216
IMAGE_HEIGHT = 912
SCALE = 100.0
MAX_POINTS_PER_REGION = 20
SAMPLING_STRIDE = 5

def find_newest_analysis_directory(base_dir: str):
    """Find the newest analysis results directory"""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"ERROR: Analysis base directory not found: {base_dir}")
        return None
    
    analysis_dirs = [item for item in base_path.iterdir() 
                    if item.is_dir() and any(char.isdigit() for char in item.name)]
    
    if not analysis_dirs:
        print(f"ERROR: No analysis directories found in {base_dir}")
        return None
    
    return str(max(analysis_dirs, key=lambda x: x.stat().st_mtime))

def load_exr_normal_image(exr_path: str):
    """Load EXR normal image"""
    if not Path(exr_path).exists():
        print(f"ERROR: File does not exist: {exr_path}")
        return None
    
    try:
        if HAS_EXR:
            exr_file = OpenEXR.InputFile(exr_path)
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
            normal_image = cv2.imread(exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if normal_image is not None:
                normal_image = cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB)
            else:
                print("ERROR: OpenCV failed to load EXR")
                return None
        
        print(f"Loaded EXR: {normal_image.shape}")
        return normal_image
        
    except Exception as e:
        print(f"ERROR loading EXR: {e}")
        return None

def decode_ue_normal(encoded_normal):
    """Decode UE SCS_Normal: (encoded * 2.0) - 1.0"""
    decoded = encoded_normal * 2.0 - 1.0
    length = np.linalg.norm(decoded)
    if length > 1e-6:
        return decoded / length
    else:
        return np.array([0.0, 0.0, 1.0])

def convert_ue_normal_to_colmap(ue_normal):
    """Convert UE normal to COLMAP coordinates (consistent with position conversion)"""
    x_ue, y_ue, z_ue = ue_normal
    colmap_normal = np.array([x_ue, z_ue, y_ue])  # [X, Z, Y]
    
    length = np.linalg.norm(colmap_normal)
    if length > 1e-6:
        return colmap_normal / length
    else:
        return np.array([0.0, 0.0, 1.0])

def convert_ue_pos_to_colmap(ue_pos, scale=100.0):
    """Convert UE position to COLMAP coordinates"""
    x_ue, y_ue, z_ue = ue_pos
    return np.array([x_ue / scale, z_ue / scale, y_ue / scale])

def find_normal_image_for_rendered_image(image_name: str, normal_dir: str):
    """Find corresponding normal image file"""
    base_name = Path(image_name).stem
    normal_path = Path(normal_dir)
    
    # Try exact match first
    exact_match = normal_path / f"{base_name}.exr"
    if exact_match.exists():
        return str(exact_match)
    
    # Try numbered patterns
    patterns = [
        f"Normal_Cam00_Pose{int(base_name):03d}.exr",
        f"Normal_Cam0_Pose{int(base_name):03d}.exr",
        f"normal_{base_name}.exr"
    ]
    
    for pattern in patterns:
        normal_file = normal_path / pattern
        if normal_file.exists():
            return str(normal_file)
    
    # Fuzzy matching
    exr_files = list(normal_path.glob("*.exr"))
    for exr_file in exr_files:
        if base_name in exr_file.stem:
            return str(exr_file)
    
    return None

def extract_error_intensity_improved(region_data):
    """Extract error intensity with fallback"""
    for field in ['error_intensity', 'mean_error', 'avg_error', 'intensity']:
        if field in region_data and isinstance(region_data[field], (int, float)):
            value = float(region_data[field])
            if value > 0:
                return value
    
    region_size = region_data.get('region_size', 100)
    return min(1.0, max(0.1, region_size / 5000.0))

def transform_to_original_space(x_aligned, y_aligned, transform_matrix):
    """Transform pixel coordinates back to original space"""
    if transform_matrix is None or np.allclose(transform_matrix, np.eye(3)):
        return x_aligned, y_aligned
    
    try:
        aligned_coords = np.array([[x_aligned, y_aligned, 1.0]]).T
        inv_transform = np.linalg.inv(transform_matrix)
        original_coords = inv_transform @ aligned_coords
        
        if abs(original_coords[2, 0]) > 1e-10:
            x_orig = original_coords[0, 0] / original_coords[2, 0]
            y_orig = original_coords[1, 0] / original_coords[2, 0]
        else:
            x_orig, y_orig = x_aligned, y_aligned
        
        x_orig = max(0, min(x_orig, IMAGE_WIDTH - 1))
        y_orig = max(0, min(y_orig, IMAGE_HEIGHT - 1))
        
        return float(x_orig), float(y_orig)
        
    except (np.linalg.LinAlgError, ValueError):
        return x_aligned, y_aligned

def extract_all_problem_points(analysis_data, normal_image, transform_matrix):
    """Extract all problem points with correct normal decoding"""
    positions_3d = analysis_data.get('positions_3d', {})
    total_regions = sum(len(regions) for regions in positions_3d.values())
    print(f"Processing {total_regions} regions...")
    
    all_points = []
    
    for error_type, regions in positions_3d.items():
        for region_idx, region in enumerate(regions):
            world_pos = region.get('world_position', [0, 0, 0])
            pixel_center = region.get('pixel_center_aligned', [IMAGE_WIDTH//2, IMAGE_HEIGHT//2])
            error_intensity = extract_error_intensity_improved(region)
            region_size = region.get('region_size', 100)
            
            sampling_radius = max(3, min(15, int(np.sqrt(region_size / np.pi))))
            max_points_this_region = min(MAX_POINTS_PER_REGION, max(1, region_size // 100))
            points_this_region = 0
            
            for dy in range(-sampling_radius, sampling_radius + 1, SAMPLING_STRIDE):
                for dx in range(-sampling_radius, sampling_radius + 1, SAMPLING_STRIDE):
                    if points_this_region >= max_points_this_region:
                        break
                    
                    if dx*dx + dy*dy > sampling_radius*sampling_radius:
                        continue
                    
                    x_aligned = pixel_center[0] + dx
                    y_aligned = pixel_center[1] + dy
                    x_orig, y_orig = transform_to_original_space(x_aligned, y_aligned, transform_matrix)
                    
                    if normal_image is not None:
                        x_int, y_int = int(round(x_orig)), int(round(y_orig))
                        if 0 <= x_int < normal_image.shape[1] and 0 <= y_int < normal_image.shape[0]:
                            encoded_normal = normal_image[y_int, x_int, :3]
                            normal_ue = decode_ue_normal(encoded_normal)
                            normal_colmap = convert_ue_normal_to_colmap(normal_ue)
                            
                            offset_scale = 10.0
                            ue_pos = [
                                world_pos[0] + dx * offset_scale,
                                world_pos[1] + dy * offset_scale,
                                world_pos[2]
                            ]
                            
                            colmap_pos = convert_ue_pos_to_colmap(ue_pos, SCALE)
                            
                            all_points.append({
                                'ue_pos': ue_pos,
                                'colmap_pos': colmap_pos,
                                'normal_ue': normal_ue,
                                'normal_colmap': normal_colmap,
                                'pixel_coords': [x_orig, y_orig],
                                'error_type': error_type,
                                'error_intensity': error_intensity,
                                'region_size': region_size,
                                'region_id': region_idx,
                                'encoded_normal': encoded_normal.tolist()
                            })
                            
                            points_this_region += 1
                
                if points_this_region >= max_points_this_region:
                    break
    
    print(f"Extracted {len(all_points)} points")
    return all_points

def create_point_cloud_with_normals(points_data, output_path, coordinate_system="UE"):
    """Create point cloud with normals"""
    print(f"Creating {coordinate_system} point cloud: {len(points_data)} points")
    
    vertices = []
    
    for point in points_data:
        if coordinate_system == "UE":
            pos = point['ue_pos']
            normal = point['normal_ue']
        else:  # COLMAP
            pos = point['colmap_pos']
            normal = point['normal_colmap']
        
        error_intensity = point['error_intensity']
        
        # Color based on error intensity
        red = int(255 * min(1.0, error_intensity))
        green = int(255 * (1.0 - min(1.0, error_intensity)))
        blue = 50
        
        vertices.append((
            float(pos[0]), float(pos[1]), float(pos[2]),
            float(normal[0]), float(normal[1]), float(normal[2]),
            red, green, blue
        ))
    
    vertices = np.array(vertices, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el]).write(output_path)
    print(f"Saved to {output_path}")

def analyze_normal_statistics(points_data):
    """Analyze extracted normals"""
    ue_normals = np.array([point['normal_ue'] for point in points_data])
    colmap_normals = np.array([point['normal_colmap'] for point in points_data])
    
    ue_lengths = np.linalg.norm(ue_normals, axis=1)
    colmap_lengths = np.linalg.norm(colmap_normals, axis=1)
    
    print(f"\nNormal statistics:")
    print(f"  UE normals - mean length: {ue_lengths.mean():.3f}")
    print(f"  COLMAP normals - mean length: {colmap_lengths.mean():.3f}")
    
    error_types = {}
    for point in points_data:
        error_type = point['error_type']
        error_types[error_type] = error_types.get(error_type, 0) + 1
    
    print(f"  Error type distribution:")
    for error_type, count in error_types.items():
        print(f"    {error_type}: {count} points")

def main():
    print("Normal Extraction Script")
    
    # Find analysis directory
    analysis_dir = find_newest_analysis_directory(ANALYSIS_RESULTS_DIR)
    if not analysis_dir:
        sys.exit(1)
    
    print(f"Using analysis directory: {Path(analysis_dir).name}")
    
    # Load analysis data
    data_dir = Path(analysis_dir) / 'data'
    json_files = sorted(list(data_dir.glob('*_data.json')))
    
    if not json_files:
        print("ERROR: No analysis data files found!")
        sys.exit(1)
    
    first_json = json_files[0]
    with open(first_json, 'r') as f:
        analysis_data = json.load(f)
    
    image_name = analysis_data.get('image_name', 'unknown')
    print(f"Processing image: {image_name}")
    
    # Find and load normal image
    normal_image_path = find_normal_image_for_rendered_image(image_name, NORMAL_IMAGES_DIR)
    if not normal_image_path:
        print("ERROR: Could not find normal image!")
        sys.exit(1)
    
    normal_image = load_exr_normal_image(normal_image_path)
    if normal_image is None:
        print("ERROR: Could not load normal image!")
        sys.exit(1)
    
    # Get alignment transform
    alignment_transform = analysis_data.get('alignment_transform', None)
    transform_matrix = np.array(alignment_transform) if alignment_transform else None
    
    # Extract points
    all_points = extract_all_problem_points(analysis_data, normal_image, transform_matrix)
    
    if len(all_points) == 0:
        print("No points extracted!")
        return
    
    # Create output
    output_dir = Path("./correct_normals_output")
    output_dir.mkdir(exist_ok=True)
    
    create_point_cloud_with_normals(all_points, output_dir / "correct_normals_ue.ply", "UE")
    create_point_cloud_with_normals(all_points, output_dir / "correct_normals_colmap.ply", "COLMAP")
    
    analyze_normal_statistics(all_points)
    
    # Save debug data
    debug_data = {
        'analysis_file': str(first_json),
        'normal_image_path': normal_image_path,
        'image_name': image_name,
        'total_points': len(all_points),
        'decoding_method': 'UE SCS_Normal: (encoded * 2.0) - 1.0',
        'coordinate_conversion': 'UE [X,Y,Z] -> COLMAP [X,Z,Y]',
        'sample_points': all_points[:5]
    }
    
    with open(output_dir / "correct_normals_debug.json", 'w') as f:
        json.dump(debug_data, f, indent=2, default=str)
    
    print(f"\nComplete! Output in: {output_dir}")
    print("Files created:")
    print("  - correct_normals_ue.ply")
    print("  - correct_normals_colmap.ply") 
    print("  - correct_normals_debug.json")

if __name__ == "__main__":
    main()