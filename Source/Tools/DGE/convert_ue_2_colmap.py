"""Core management module with COLMAP coordinate conventions"""

import re
import sys
import time
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Optional


class Config:
    """Configuration settings with COLMAP conventions"""
    
    # Analysis results
    ANALYSIS_RESULTS_BASE_DIR = "./Logs/analysis_results"
    AUTO_SELECT_NEWEST = True
    MANUAL_ANALYSIS_DIR = None

    # Input paths
    UE_POSES_FILE = "./Source/Tools/DGE/data/poses/0528_pose.txt"
    UE_REAL_IMAGES_DIR = "./Source/Tools/DGE/data/real_images_downscaled"
    UE_NORMAL_IMAGES_DIR = "./Source/Tools/DGE/data/normal_images"

    # Output settings
    OUTPUT_BASE_DIR = "./Logs/pointcloud_datasets"
    AUTO_CREATE_TIMESTAMP_DIR = True
    MANUAL_OUTPUT_DIR = None

    # Filtering settings
    ERROR_TYPE_FILTER = None  # None = all types, or list like ['mse', 'mae']
    MIN_REGION_SIZE = 20
    MAX_POINTS = None
    MIN_POINT_DISTANCE_CM = 20.0

    # Final limits
    FINAL_MAX_POINTS = 500000
    APPLY_FINAL_LIMIT = True

    # Camera parameters
    IMAGE_WIDTH = 1216
    IMAGE_HEIGHT = 912
    HFOV_DEGREES = 67.38
    
    # Coordinate conversion: UE (cm) -> COLMAP (m)
    COORDINATE_SCALE = 100.0

    # Performance settings
    MAX_WORKER_THREADS = min(12, mp.cpu_count())
    BATCH_SIZE_MULTIPLIER = 2


class DirectoryManager:
    """Directory management with validation"""
    
    @staticmethod
    def find_newest_analysis_directory(base_dir: str) -> Optional[str]:
        """Find the newest analysis results directory."""
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"ERROR: Analysis base directory not found: {base_dir}")
            return None
        
        timestamp_patterns = [
            re.compile(r'(\d{8}_\d{6})'),
            re.compile(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})'),
        ]
        
        analysis_dirs = []
        
        for item in base_path.iterdir():
            if not item.is_dir():
                continue
                
            for pattern in timestamp_patterns:
                match = pattern.search(item.name)
                if match:
                    timestamp_str = match.group(1)
                    try:
                        if '_' in timestamp_str and '-' not in timestamp_str:
                            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        else:
                            timestamp_str_clean = timestamp_str.replace('-', '')
                            timestamp = datetime.strptime(timestamp_str_clean, "%Y%m%d_%H%M%S")
                        
                        analysis_dirs.append((timestamp, item))
                        break
                    except ValueError:
                        continue
        
        if not analysis_dirs:
            print(f"ERROR: No analysis directories with valid timestamps found in {base_dir}")
            available_dirs = [item.name for item in base_path.iterdir() if item.is_dir()][:5]
            if available_dirs:
                print(f"Available directories: {available_dirs}")
            return None
        
        analysis_dirs.sort(key=lambda x: x[0], reverse=True)
        newest_dir = analysis_dirs[0][1]
        
        print(f"Selected newest analysis directory: {newest_dir.name}")
        return str(newest_dir)

    @staticmethod
    def determine_analysis_directory() -> str:
        """Determine which analysis directory to use with validation."""
        if Config.MANUAL_ANALYSIS_DIR and not Config.AUTO_SELECT_NEWEST:
            analysis_dir = Config.MANUAL_ANALYSIS_DIR
            print(f"Using manual analysis directory: {analysis_dir}")
        else:
            analysis_dir = DirectoryManager.find_newest_analysis_directory(
                Config.ANALYSIS_RESULTS_BASE_DIR)
            if not analysis_dir:
                sys.exit(1)
        
        analysis_path = Path(analysis_dir)
        if not analysis_path.exists():
            print(f"ERROR: Analysis directory does not exist: {analysis_dir}")
            sys.exit(1)
        
        # Validate analysis directory structure
        data_dir = analysis_path / 'data'
        if not data_dir.exists():
            print(f"ERROR: Analysis data directory not found: {data_dir}")
            sys.exit(1)
        
        json_files = list(data_dir.glob('*_data.json'))
        if not json_files:
            print(f"ERROR: No analysis data files found in: {data_dir}")
            sys.exit(1)
        
        print(f"Found {len(json_files)} analysis data files")
        return analysis_dir

    @staticmethod
    def determine_output_directory() -> str:
        """Determine output directory with conflict resolution."""
        if Config.MANUAL_OUTPUT_DIR and not Config.AUTO_CREATE_TIMESTAMP_DIR:
            output_dir = Config.MANUAL_OUTPUT_DIR
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"colmap_dataset_{timestamp}"
            output_dir = str(Path(Config.OUTPUT_BASE_DIR) / base_name)
            
            # Handle conflicts
            counter = 1
            while Path(output_dir).exists():
                output_dir = str(Path(Config.OUTPUT_BASE_DIR) / f"{base_name}_{counter}")
                counter += 1
        
        return output_dir

    @staticmethod
    def validate_input_paths():
        """Enhanced input path validation with detailed error messages."""
        paths_to_check = [
            ("Real Images Directory", Config.UE_REAL_IMAGES_DIR, "directory"),
            ("Normal Images Directory", Config.UE_NORMAL_IMAGES_DIR, "directory"),
            ("UE Poses File", Config.UE_POSES_FILE, "file"),
        ]
        
        missing_paths = []
        invalid_paths = []
        
        for name, path, path_type in paths_to_check:
            path_obj = Path(path)
            if not path_obj.exists():
                missing_paths.append(f"{name}: {path}")
            elif path_type == "directory" and not path_obj.is_dir():
                invalid_paths.append(f"{name}: {path} (not a directory)")
            elif path_type == "file" and not path_obj.is_file():
                invalid_paths.append(f"{name}: {path} (not a file)")
        
        if missing_paths or invalid_paths:
            print("ERROR: Invalid input paths detected:")
            for missing in missing_paths:
                print(f"  MISSING: {missing}")
            for invalid in invalid_paths:
                print(f"  INVALID: {invalid}")
            sys.exit(1)
        
        # Additional validation for image directories
        real_images = list(Path(Config.UE_REAL_IMAGES_DIR).glob("*.png")) + \
                     list(Path(Config.UE_REAL_IMAGES_DIR).glob("*.jpg"))
        if not real_images:
            print(f"ERROR: No image files found in {Config.UE_REAL_IMAGES_DIR}")
            sys.exit(1)
        
        # Validate poses file format
        try:
            with open(Config.UE_POSES_FILE, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    vals = first_line.split()
                    if len(vals) != 6:
                        print(f"ERROR: Poses file format invalid. Expected 6 values per line, got {len(vals)}")
                        sys.exit(1)
                    [float(v) for v in vals]
        except Exception as e:
            print(f"ERROR: Cannot read poses file {Config.UE_POSES_FILE}: {e}")
            sys.exit(1)
        
        print(f"✓ Validation passed - found {len(real_images)} training images")


def print_info():
    """Print information about the coordinate system"""
    print("POINT CLOUD DATASET CREATION WITH COLMAP CONVENTIONS")
    print("=" * 60)
    
    num_cores = mp.cpu_count()
    max_workers = Config.MAX_WORKER_THREADS
    
    print(f"System Performance:")
    print(f"  CPU cores detected: {num_cores}")
    print(f"  Max worker threads: {max_workers}")
    print(f"")
    print(f"Coordinate System:")
    print(f"  UE System: X=forward, Y=right, Z=up (cm)")
    print(f"  COLMAP System: X=forward, Y=right, Z=down (m)")
    print(f"  Transformation: Flip Z axis (Z → -Z)")
    print(f"  Scale factor: ÷{Config.COORDINATE_SCALE} (cm → m)")
    print(f"  Camera poses: World-to-camera convention")
    print("=" * 60)


def print_configuration(analysis_dir: str, output_dir: str):
    """Print configuration summary"""
    print(f"Configuration:")
    print(f"  Analysis source: {analysis_dir}")
    print(f"  Output destination: {output_dir}")
    print(f"  Training images: {Config.UE_REAL_IMAGES_DIR}")
    print(f"  Normal images: {Config.UE_NORMAL_IMAGES_DIR}")
    print(f"  Poses file: {Config.UE_POSES_FILE}")
    
    print(f"")
    print(f"Coordinate System:")
    print(f"  Transformation: Flip Z axis (X→X, Y→Y, Z→-Z)")
    print(f"  Scale conversion: cm ÷ {Config.COORDINATE_SCALE} → m")
    print(f"  Camera pose format: World-to-camera (COLMAP standard)")
    print(f"  Point frame: Consistent with camera world frame")
    
    # Filtering settings
    filters = []
    if Config.ERROR_TYPE_FILTER:
        filters.append(f"Error types: {Config.ERROR_TYPE_FILTER}")
    if Config.MIN_REGION_SIZE > 0:
        filters.append(f"Min region size: {Config.MIN_REGION_SIZE}")
    if Config.MAX_POINTS:
        filters.append(f"Initial max points: {Config.MAX_POINTS:,}")
    if Config.APPLY_FINAL_LIMIT:
        filters.append(f"Final max points: {Config.FINAL_MAX_POINTS:,}")
    if Config.MIN_POINT_DISTANCE_CM > 0:
        filters.append(f"Min point distance: {Config.MIN_POINT_DISTANCE_CM}cm")
    
    if filters:
        print(f"")
        print(f"Filtering: {' | '.join(filters)}")
    
    print("=" * 60)


def main():
    """Main function with coordinate conversion"""
    start_time = time.time()
    
    print_info()
    
    try:
        # Determine directories
        print("Step 1: Determining input and output paths...")
        analysis_dir = DirectoryManager.determine_analysis_directory()
        DirectoryManager.validate_input_paths()
        output_dir = DirectoryManager.determine_output_directory()
        
        print_configuration(analysis_dir, output_dir)
        
        # Import modules and create dataset
        print("Step 2: Initializing processing modules...")
        from output_generation import PointCloudDatasetCreator
        
        print("Step 3: Creating point cloud dataset with COLMAP conventions...")
        dataset_start_time = time.time()
        
        PointCloudDatasetCreator.create_pointcloud_dataset_from_error_analysis(
            analysis_results_dir=analysis_dir,
            real_images_dir=Config.UE_REAL_IMAGES_DIR,
            normal_images_dir=Config.UE_NORMAL_IMAGES_DIR,
            ue_poses_file=Config.UE_POSES_FILE,
            output_dir=output_dir,
            scale=Config.COORDINATE_SCALE
        )
        
        dataset_time = time.time() - dataset_start_time
        total_time = time.time() - start_time
        
        print("=" * 60)
        print("PROCESSING COMPLETED SUCCESSFULLY!")
        print(f"Dataset creation time: {dataset_time:.1f}s")
        print(f"Total execution time: {total_time:.1f}s")
        print(f"Output saved to: {output_dir}")
        print("")
        print("Dataset Features:")
        print("  ✅ COLMAP coordinate system")
        print("  ✅ World-to-camera pose convention")
        print("  ✅ Consistent world frame for points and cameras")
        print("  ✅ Proper scale (meters)")
        print("  ✅ Valid 2D pixel coordinates")
        print("")
        print("Compatible with:")
        print("  • COLMAP (reconstruction, bundle adjustment)")
        print("  • 3D Gaussian Splatting")
        print("  • NeRF implementations")
        print("  • Other computer vision tools expecting COLMAP format")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("PROCESSING INTERRUPTED BY USER")
        print("=" * 60)
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"ERROR: Processing failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        print("")
        print("Common issues and solutions:")
        print("  • Check that all input paths exist and are accessible")
        print("  • Verify poses file format (6 values per line)")
        print("  • Ensure sufficient disk space for output")
        print("  • Check that analysis results contain valid data")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    # Ensure proper multiprocessing support
    mp.freeze_support()
    
    # Set start method for better memory management
    if mp.get_start_method(allow_none=True) != 'spawn':
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    main()