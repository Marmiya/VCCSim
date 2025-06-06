import numpy as np
import pandas as pd
import cv2
import open3d as o3d
from pathlib import Path
import time
from tqdm import tqdm
import math

class DepthPointCloudGenerator:
    
    def __init__(self, depth_dir: str, poses_file: str, output_dir: str = "./Logs/test_pointclouds"):
        self.depth_dir = Path(depth_dir)
        self.poses_file = Path(poses_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Camera intrinsics for 1216x912 depth camera with 67.38° FOV
        self.image_width = 1216
        self.image_height = 912
        self.cx = self.image_width / 2.0   # 608
        self.cy = self.image_height / 2.0  # 456
        # Calculate focal length from FOV: fx = (width/2) / tan(fov/2)
        fov_rad = math.radians(67.38)
        self.fx = self.cx / math.tan(fov_rad / 2.0)  # ~916
        self.fy = self.fx  # assume square pixels
        
        # Depth scaling (adjust based on your depth format)
        self.depth_scale = 1000.0  # if depth is in mm, use 1000 to convert to meters
        
        print(f"Initialized with camera FOV: {67.38}°, resolution: {self.image_width}x{self.image_height}")
        print(f"Focal length: fx={self.fx:.1f}, fy={self.fy:.1f}")
        print(f"Expected pose format: X Y Z Pitch Yaw Roll (coordinates in cm, angles in degrees)")
        
    def load_poses(self):
        """Load camera poses from space-separated file."""
        print(f"Loading poses from: {self.poses_file}")
        
        try:
            # Read all lines
            with open(self.poses_file, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            print("First few lines of pose file:")
            for i, line in enumerate(lines[:3]):
                print(f"  {i}: {line}")
            
            # Parse data - expected format: X Y Z Pitch Yaw Roll (space-separated)
            data_rows = []
            for i, line in enumerate(lines):
                values = line.split()
                if len(values) == 6:
                    try:
                        x, y, z, pitch, yaw, roll = map(float, values)
                        # Generate image name based on line number (1-indexed)
                        image_name = f"DJI_{i+1:04d}.png"
                        data_rows.append({
                            'name': image_name,
                            'x': x,
                            'y': y, 
                            'z': z,
                            'pitch': pitch,
                            'yaw': yaw,
                            'roll': roll
                        })
                    except ValueError:
                        print(f"Skipping malformed line {i}: {line}")
                else:
                    print(f"Skipping line {i} (expected 6 values, got {len(values)}): {line}")
            
            # Create DataFrame
            df = pd.DataFrame(data_rows)
            
            print(f"Successfully loaded {len(df)} poses")
            print(f"Columns: {list(df.columns)}")
            print(f"Format: X Y Z Pitch Yaw Roll (coordinates in cm)")
            
            # Show first row as example
            if len(df) > 0:
                print(f"First pose: {df.iloc[0]['name']} at ({df.iloc[0]['x']:.1f}, {df.iloc[0]['y']:.1f}, {df.iloc[0]['z']:.1f})")
            
            return df
            
        except Exception as e:
            print(f"Error loading pose file: {e}")
            print("Expected format: space-separated values")
            print("X Y Z Pitch Yaw Roll")
            print("Example: -10108.379200 -5219.965500 10650.680600 -42.653287 5.391365 0.785717")
            raise
    
    def euler_to_rotation_matrix(self, pitch, yaw, roll):
        """Convert Euler angles (degrees) to rotation matrix.
        Args:
            pitch: Rotation around X-axis (degrees)
            yaw: Rotation around Y-axis (degrees) 
            roll: Rotation around Z-axis (degrees)
        """
        # Convert to radians
        p = math.radians(pitch)  # X-axis rotation
        y = math.radians(yaw)    # Y-axis rotation
        r = math.radians(roll)   # Z-axis rotation
        
        # Rotation matrices for each axis
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(p), -math.sin(p)],
                       [0, math.sin(p), math.cos(p)]])
        
        Ry = np.array([[math.cos(y), 0, math.sin(y)],
                       [0, 1, 0],
                       [-math.sin(y), 0, math.cos(y)]])
        
        Rz = np.array([[math.cos(r), -math.sin(r), 0],
                       [math.sin(r), math.cos(r), 0],
                       [0, 0, 1]])
        
        # Combined rotation: R = Rz * Ry * Rx (Roll * Yaw * Pitch)
        R = Rz @ Ry @ Rx
        return R
    
    def depth_to_pointcloud(self, depth_image, pose_row):
        """Convert depth image to 3D points using camera pose."""
        # Handle multi-channel images by taking first channel
        if len(depth_image.shape) > 2:
            print(f"    Multi-channel depth image detected: {depth_image.shape}, using first channel")
            depth_image = depth_image[:, :, 0]
        
        height, width = depth_image.shape
        
        # Create meshgrid of pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u = u.flatten()
        v = v.flatten()
        depth = depth_image.flatten()
        
        # Filter out invalid depths (0 or too large)
        valid = (depth > 0) & (depth < 65000)  # adjust max depth as needed
        u = u[valid]
        v = v[valid] 
        depth = depth[valid] / self.depth_scale  # convert to meters
        
        if len(depth) == 0:
            return np.empty((0, 3))
        
        # Convert to 3D camera coordinates
        x_cam = (u - self.cx) * depth / self.fx
        y_cam = (v - self.cy) * depth / self.fy
        z_cam = depth
        
        # Stack into homogeneous coordinates
        points_cam = np.vstack([x_cam, y_cam, z_cam, np.ones(len(x_cam))])
        
        # Get camera pose - coordinates are already in cm, convert to meters
        camera_pos = np.array([pose_row['x'], pose_row['y'], pose_row['z']]) / 100.0  # cm to meters
        R = self.euler_to_rotation_matrix(pose_row['pitch'], pose_row['yaw'], pose_row['roll'])
        
        # Create transformation matrix (camera to world)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = camera_pos
        
        # Transform to world coordinates
        points_world = T @ points_cam
        
        return points_world[:3].T  # return as Nx3 array
    
    def load_depth_image(self, image_name):
        """Load depth image. Tries multiple formats and naming patterns."""
        base_name = Path(image_name).stem
        
        # Try different depth file extensions and patterns
        possible_names = [
            f"{base_name}.png",
            f"{base_name}.exr", 
            f"{base_name}_depth.png",
            f"{base_name}_depth.exr",
            f"{base_name}.JPG",  # in case original was JPG
            f"{image_name}",  # exact name
        ]
        
        # Also try with different case variations
        for name in possible_names.copy():
            possible_names.append(name.lower())
            possible_names.append(name.upper())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for name in possible_names:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)
        
        for name in unique_names:
            depth_path = self.depth_dir / name
            if depth_path.exists():
                if depth_path.suffix.lower() == '.exr':
                    # EXR format (usually float)
                    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                else:
                    # PNG/JPG format (usually uint16 or uint8)
                    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                
                if depth is not None:
                    print(f"  Loaded: {name} (shape: {depth.shape}, dtype: {depth.dtype})")
                    return depth
        
        # If no exact match found, try to find any file with similar name
        if self.depth_dir.exists():
            all_files = list(self.depth_dir.glob("*"))
            print(f"  Warning: No depth image found for {image_name}")
            print(f"  Available files: {[f.name for f in all_files[:5]]}{'...' if len(all_files) > 5 else ''}")
        
        return None
    
    def generate_pointcloud(self, max_images: int = None, subsample: int = 4):
        """
        Generate point cloud from depth images and poses.
        
        Args:
            max_images: Maximum number of images to process (None for all)
            subsample: Subsample factor to reduce points (1=no subsample, 4=every 4th pixel)
        """
        start_time = time.time()
        
        print("="*60)
        print("DEPTH IMAGE POINT CLOUD GENERATOR")
        print("="*60)
        
        # Load poses
        poses_df = self.load_poses()
        
        if max_images is not None:
            poses_df = poses_df.head(max_images)
            print(f"Limited to first {max_images} images")
        
        print(f"Processing {len(poses_df)} images")
        print(f"Subsample factor: {subsample} (every {subsample}th pixel)")
        print(f"Depth directory: {self.depth_dir}")
        
        all_points = []
        successful_images = 0
        
        for idx, row in tqdm(poses_df.iterrows(), total=len(poses_df), desc="Processing images"):
            image_name = row['name']
            
            # Load depth image
            depth_image = self.load_depth_image(image_name)
            if depth_image is None:
                continue
            
            # Subsample if requested
            if subsample > 1:
                # Handle multi-channel images before subsampling
                if len(depth_image.shape) > 2:
                    depth_image = depth_image[::subsample, ::subsample, :]
                else:
                    depth_image = depth_image[::subsample, ::subsample]
            
            # Convert to point cloud
            points = self.depth_to_pointcloud(depth_image, row)
            
            if len(points) > 0:
                all_points.append(points)
                successful_images += 1
                print(f"  {image_name}: {len(points):,} points")
            else:
                print(f"  {image_name}: No valid points")
        
        if not all_points:
            print("No points generated! Check depth images and poses.")
            return None
        
        # Combine all points
        print("Combining point clouds...")
        combined_points = np.vstack(all_points)
        print(f"Total points: {len(combined_points):,}")
        
        # Convert to UE coordinates (right-handed to left-handed, meters to cm)
        print("Converting to UE coordinate system...")
        ue_points = combined_points.copy() * 100.0  # meters to centimeters
        
        # Coordinate system conversion: (X,Y,Z) -> (X,-Y,Z)  
        ue_points_converted = np.zeros_like(ue_points)
        ue_points_converted[:, 0] = ue_points[:, 0]   # X stays the same
        ue_points_converted[:, 1] = -ue_points[:, 1]  # Y flipped
        ue_points_converted[:, 2] = ue_points[:, 2]   # Z stays the same
        
        # Save UE PLY
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        images_suffix = f"_test{max_images}imgs" if max_images is not None else "_all"
        subsample_suffix = f"_sub{subsample}" if subsample > 1 else ""
        filename = f"depth_pointcloud{images_suffix}{subsample_suffix}_UE_{timestamp}.ply"
        
        output_path = self.output_dir / filename
        
        print(f"Saving UE point cloud: {output_path}")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ue_points_converted)
        
        # Add simple gray color
        colors = np.ones_like(ue_points_converted) * 0.7  # gray color
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(str(output_path), pcd)
        
        total_time = time.time() - start_time
        
        print("="*60)
        print("GENERATION COMPLETE")
        print("="*60)
        print(f"Processing time: {total_time:.1f}s")
        print(f"Images processed: {successful_images}/{len(poses_df)}")
        print(f"Total points: {len(combined_points):,}")
        print(f"UE PLY saved: {output_path}")
        
        # Print spatial extent
        print(f"Spatial extent (UE coordinates, cm):")
        print(f"  X: [{ue_points_converted[:, 0].min():.1f}, {ue_points_converted[:, 0].max():.1f}]")
        print(f"  Y: [{ue_points_converted[:, 1].min():.1f}, {ue_points_converted[:, 1].max():.1f}]")
        print(f"  Z: [{ue_points_converted[:, 2].min():.1f}, {ue_points_converted[:, 2].max():.1f}]")
        print("="*60)
        
        return {
            'points_world': combined_points,
            'points_ue': ue_points_converted,
            'output_file': output_path,
            'num_images': successful_images,
            'total_points': len(combined_points)
        }

# Convenience functions
def test_depth_pointcloud(depth_dir: str, poses_file: str, max_images: int = 5, subsample: int = 4):
    """Quick test with limited images."""
    generator = DepthPointCloudGenerator(depth_dir, poses_file)
    return generator.generate_pointcloud(max_images=max_images, subsample=subsample)

def generate_depth_pointcloud(depth_dir: str, poses_file: str, max_images: int = None, subsample: int = 1):
    """Generate point cloud from all depth images."""
    generator = DepthPointCloudGenerator(depth_dir, poses_file)
    return generator.generate_pointcloud(max_images=max_images, subsample=subsample)

if __name__ == "__main__":
    print("DEPTH POINT CLOUD TEST GENERATOR")
    print("=================================")
    
    # Your actual paths
    depth_directory = "./Source/Tools/DGE/data/depth_images"
    poses_file = "./Source/Tools/DGE/data/poses/pose.txt"  # Updated extension
    
    print(f"Depth dir: {depth_directory}")
    print(f"Poses file: {poses_file}")
    print(f"Camera: 1216x912, FOV: 67.38°")
    print(f"Expected pose format: X Y Z Pitch Yaw Roll (space-separated, coordinates in cm)")
    print()
    
    # Test with 1 image first
    result = test_depth_pointcloud(depth_directory, poses_file, max_images=1)
    
    if result:
        print(f"\n✅ Success! Generated {result['total_points']:,} points")
        print(f"📁 Output: {result['output_file']}")
    else:
        print("❌ Failed to generate point cloud")