#!/usr/bin/env python3

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime

class DepthPointCloudGenerator:
    """
    Generate 3D point clouds from depth images and camera poses.
    Compatible with Unreal Engine coordinate system and transformations.
    """
    
    def __init__(self, data_root: str, fov: float = 67.38, width: int = 1216, height: int = 912):
        self.data_root = Path(data_root)
        self.depth_images_dir = self.data_root / "depth_images"
        self.poses_file = self.data_root / "poses" / "0528_pose.txt"
        
        # Camera parameters
        self.width = width
        self.height = height
        self.fov = fov
        
        # Calculate FOV parameters
        self.aspect_ratio = float(width) / height
        half_fov_rad = np.radians(fov * 0.5)
        self.tan_half_horizontal_fov = np.tan(half_fov_rad)
        self.tan_half_vertical_fov = self.tan_half_horizontal_fov / self.aspect_ratio
        
        print(f"Initialized: {width}x{height}, FOV: {fov}°")
    
    def load_poses(self) -> Optional[pd.DataFrame]:
        """Load camera poses from file."""
        try:
            poses_data = np.loadtxt(self.poses_file)
            depth_images = sorted([f.name for f in self.depth_images_dir.glob("*.png")])
            
            min_count = min(len(poses_data), len(depth_images))
            poses_data = poses_data[:min_count]
            depth_images = depth_images[:min_count]
            
            poses_df = pd.DataFrame(poses_data, columns=['x', 'y', 'z', 'pitch', 'yaw', 'roll'])
            poses_df['name'] = depth_images
            
            print(f"Loaded {len(poses_df)} poses")
            return poses_df
            
        except Exception as e:
            print(f"Error loading poses: {e}")
            return None
    
    def load_depth_image(self, image_name: str) -> Optional[np.ndarray]:
        """Load depth image (16-bit PNG with centimeter values)."""
        depth_path = self.depth_images_dir / image_name
        
        if not depth_path.exists():
            print(f"Depth image not found: {depth_path}")
            return None
        
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            return None
        
        # Take first channel if multi-channel
        if len(depth_img.shape) == 3:
            depth_img = depth_img[:, :, 0]
        
        return depth_img.astype(np.float32)
    
    def create_rotation_matrix(self, yaw: float, pitch: float, roll: float) -> np.ndarray:
        """
        Create rotation matrix compatible with Unreal Engine's coordinate system.
        Note: UE uses a different yaw sign convention than standard math.
        """
        # Adjust yaw to match UE convention
        yaw = -yaw
        
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
        cos_roll, sin_roll = np.cos(roll), np.sin(roll)
        
        # Individual rotation matrices
        R_roll = np.array([
            [1, 0, 0],
            [0, cos_roll, -sin_roll],
            [0, sin_roll, cos_roll]
        ])
        
        R_pitch = np.array([
            [cos_pitch, 0, sin_pitch],
            [0, 1, 0],
            [-sin_pitch, 0, cos_pitch]
        ])
        
        R_yaw = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        return R_roll @ R_pitch @ R_yaw
    
    def depth_to_points(self, depth_img: np.ndarray, pose: pd.Series, 
                       min_depth: float = 50, max_depth: float = 10000,
                       step: int = 4) -> np.ndarray:
        """Convert depth image to 3D points using camera pose."""
        
        # Camera pose
        cam_pos = np.array([pose['x'], pose['y'], pose['z']])
        
        # Create rotation matrix
        R = self.create_rotation_matrix(
            np.radians(pose['yaw']), 
            np.radians(pose['pitch']), 
            np.radians(pose['roll'])
        )
        
        # Sample pixels
        y_coords, x_coords = np.meshgrid(
            np.arange(0, self.height, step),
            np.arange(0, self.width, step),
            indexing='ij'
        )
        
        y_coords = y_coords.flatten()
        x_coords = x_coords.flatten()
        depth_values = depth_img[y_coords, x_coords]
        
        # Filter valid depths
        valid_mask = (depth_values >= min_depth) & (depth_values <= max_depth) & (depth_values > 0)
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        depth_values = depth_values[valid_mask]
        
        if len(depth_values) == 0:
            return np.array([]).reshape(0, 3)
        
        # Convert to NDC coordinates
        ndc_x = (2.0 * x_coords / (self.width - 1)) - 1.0
        ndc_y = 1.0 - (2.0 * y_coords / (self.height - 1))
        
        # Calculate view space coordinates
        view_x = ndc_x * self.tan_half_horizontal_fov * depth_values
        view_y = ndc_y * self.tan_half_vertical_fov * depth_values
        
        # Camera space positions (UE: X=Forward, Y=Right, Z=Up)
        camera_space = np.column_stack([
            depth_values,  # Forward (X)
            view_x,        # Right (Y)  
            view_y         # Up (Z)
        ])
        
        # Transform to world space
        world_points = (R.T @ camera_space.T).T + cam_pos
        
        return world_points
    
    def generate_point_cloud(self, max_images: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Generate point cloud from depth images."""
        poses_df = self.load_poses()
        if poses_df is None:
            return np.array([]).reshape(0, 3), {}
        
        if max_images:
            poses_df = poses_df.head(max_images)
        
        all_points = []
        stats = {'images_processed': 0, 'total_points': 0}
        
        for idx, pose in poses_df.iterrows():
            depth_img = self.load_depth_image(pose['name'])
            if depth_img is None:
                continue
            
            points = self.depth_to_points(depth_img, pose)
            if len(points) > 0:
                all_points.append(points)
                stats['total_points'] += len(points)
                print(f"Processed {pose['name']}: {len(points)} points")
            
            stats['images_processed'] += 1
        
        if not all_points:
            return np.array([]).reshape(0, 3), stats
        
        point_cloud = np.vstack(all_points)
        print(f"Generated {len(point_cloud)} total points from {stats['images_processed']} images")
        
        return point_cloud, stats
    
    def save_ply(self, points: np.ndarray, filename: str = None):
        """Save point cloud to PLY file."""
        if len(points) == 0:
            print("No points to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"depth_pointcloud_{timestamp}.ply"
        
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"comment Generated from depth images - UE coordinate system\n")
            f.write(f"comment X=Forward, Y=Right, Z=Up (left-handed, centimeters)\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            for point in points:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        
        print(f"Saved {len(points)} points to {output_path}")

def main():
    """Generate point cloud from depth images and camera poses."""
    # Configuration
    DATA_ROOT = "./Source/Tools/DGE/data"  # Update this path
    MAX_IMAGES = 4  # Process all images (or set to number for testing)
    
    # Check data directory
    if not Path(DATA_ROOT).exists():
        print(f"Error: Data directory not found: {DATA_ROOT}")
        return
    
    # Generate point cloud
    generator = DepthPointCloudGenerator(DATA_ROOT)
    points, stats = generator.generate_point_cloud(max_images=MAX_IMAGES)
    
    if len(points) > 0:
        # Save to PLY file
        generator.save_ply(points)
        
        # Print statistics
        print(f"\nPoint cloud statistics:")
        print(f"  Total points: {len(points)}")
        print(f"  Images processed: {stats['images_processed']}")
        print(f"  Bounds X: [{points[:, 0].min():.1f}, {points[:, 0].max():.1f}] cm")
        print(f"  Bounds Y: [{points[:, 1].min():.1f}, {points[:, 1].max():.1f}] cm") 
        print(f"  Bounds Z: [{points[:, 2].min():.1f}, {points[:, 2].max():.1f}] cm")
    else:
        print("No points generated!")

if __name__ == "__main__":
    main()