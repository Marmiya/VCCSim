import sys
import os
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Add the parent directory of PyClient to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VCCSim import VCCSimClient
from VCCSim import VCCSim_pb2
from VCCSim.VCCSimClient import RGBImageUtils

ip = "172.31.178.19"
# Test script for RGB camera functionality
def test_rgb_camera():
    # Create VCCSim client with default connection settings
    client = VCCSimClient(host=ip, port=50996)

    try:
        # Configuration settings
        robot_name = "Mavic"  # Change to your robot name
        camera_index = 0       # Camera index to use
        
        print(f"Connecting to robot: {robot_name}")
        print(f"Requesting images from camera index: {camera_index}")
        
        # Test and save different image formats
        format_files = {
            VCCSim_pb2.Format.PNG: "rgb_camera_test_png.png",
            VCCSim_pb2.Format.JPEG: "rgb_camera_test_jpeg.jpg",
            VCCSim_pb2.Format.RAW: "rgb_camera_test_raw.png"  # We'll convert RAW to PNG for viewing
        }
        
        saved_images = []
        
        print("\nTesting and saving different image formats:")
        for format_type, output_file in format_files.items():
            try:
                start_time = time.time()
                image_data = client.get_rgb_indexed_camera_image_data(
                    robot_name=robot_name,
                    index=camera_index,
                    format=format_type
                )
                elapsed = time.time() - start_time
                format_name = VCCSim_pb2.Format.Name(format_type)
                
                print(f"  {format_name}: received {len(image_data.data)} bytes in {elapsed:.3f} seconds")
                print(f"  Image dimensions: {image_data.width}x{image_data.height}")
                
                # Save the image
                if format_type == VCCSim_pb2.Format.RAW:
                    # For RAW format, we need to convert it to a viewable format
                    img_array = RGBImageUtils.process_rgb_image_data(image_data)
                    img = Image.fromarray(img_array)
                    img.save(output_file)
                else:
                    # For PNG and JPEG, we can use the utility method
                    RGBImageUtils.save_rgb_image(image_data, output_file)
                
                saved_images.append(output_file)
                print(f"  Saved to {output_file}")
                
                # Calculate and print image stats
                img_array = RGBImageUtils.process_rgb_image_data(image_data)
                print(f"  Mean pixel value: {np.mean(img_array):.2f}")
                print(f"  Min/Max pixel values: {np.min(img_array)}/{np.max(img_array)}")
                
            except Exception as e:
                print(f"  Error with {VCCSim_pb2.Format.Name(format_type)} format: {e}")
                
        # Print summary of saved images
        if saved_images:
            print("\nSaved the following image files for visualization:")
            for filename in saved_images:
                print(f"  - {filename}")
                
        # Optional: Get camera odometry
        try:
            odom = client.get_rgb_camera_odom(robot_name)
            print("\nRGB Camera Odometry:")
            print(f"  Position: ({odom.pose.x:.2f}, {odom.pose.y:.2f}, {odom.pose.z:.2f})")
            print(f"  Orientation: ({odom.pose.roll:.2f}, {odom.pose.pitch:.2f}, {odom.pose.yaw:.2f})")
        except Exception as e:
            print(f"Error getting camera odometry: {e}")
            
    except Exception as e:
        print(f"Error in RGB camera test: {e}")
    finally:
        # Close the client
        client.close()
        print("Client connection closed")

def test_depth_camera():
    # Create VCCSim client with default connection settings
    client = VCCSimClient(host=ip, port=50996)

    try:
        # Configuration settings
        robot_name = "Mavic"  # Change to your robot name
        camera_index = 0       # Camera index to use
        
        print(f"Connecting to robot: {robot_name}")
        print(f"Requesting depth camera data from camera index: {camera_index}")
        
        # Get and process depth image data
        try:
            start_time = time.time()
            depth_data = client.get_depth_camera_image_data(robot_name=robot_name, index=camera_index)
            elapsed = time.time() - start_time
            
            print(f"Received {len(depth_data)} depth values in {elapsed:.3f} seconds")
            
            # Basic statistics
            depth_array = np.array(depth_data)
            print(f"Mean depth: {np.mean(depth_array):.2f} m")
            print(f"Min/Max depths: {np.min(depth_array):.2f}/{np.max(depth_array):.2f} m")

            width, height = client.get_depth_camera_image_size(robot_name=robot_name, index=camera_index)
            # Reshape depth data into a 2D image
            depth_image = depth_array.reshape((height, width))
            print(f"Depth image shape: {depth_image.shape}")
            
            point_data = client.get_depth_camera_point_data(robot_name=robot_name, index=camera_index)
            
            # Save depth visualization
            plt.figure(figsize=(10, 8))
            plt.imshow(depth_image, cmap='viridis')
            plt.colorbar(label='Depth (m)')
            plt.title('Depth Camera Visualization')
            plt.savefig('depth_camera_test.png')
            plt.close()
            
            print("Saved depth visualization to depth_camera_test.png")
            
            # Optional: Save raw depth data
            np.save('depth_data_raw.npy', depth_array)
            print("Saved raw depth data to depth_data_raw.npy")
            
            # Get 3D point cloud data
            start_time = time.time()
            point_data = client.get_depth_camera_point_data(robot_name=robot_name, index=camera_index)
            elapsed = time.time() - start_time
            
            print(f"Received {len(point_data)} 3D points in {elapsed:.3f} seconds")
            
            # Convert point data to numpy array for easier processing
            point_array = np.array(point_data)
            
            # Save point cloud data
            if len(point_data) > 0:
                # Save as CSV for easy viewing in other tools
                with open('depth_point_cloud.csv', 'w') as f:
                    f.write('x,y,z\n')
                    for point in point_data:
                        f.write(f"{point[0]},{point[1]},{point[2]}\n")
                print("Saved 3D point cloud to depth_point_cloud.csv")
                
                # Optional: Basic visualization of point cloud
                if len(point_data) < 10000:  # Only visualize if not too large
                    plt.figure(figsize=(10, 8))
                    plt.scatter(point_array[:, 0], point_array[:, 1], s=0.5, c=point_array[:, 2], cmap='viridis')
                    plt.colorbar(label='Z (height)')
                    plt.axis('equal')
                    plt.title('Depth Camera Point Cloud (Top-Down View)')
                    plt.savefig('depth_point_cloud_viz.png')
                    plt.close()
                    print("Saved point cloud visualization to depth_point_cloud_viz.png")
            
        except Exception as e:
            print(f"Error retrieving depth data: {e}")
        
        # Get camera odometry
        try:
            odom = client.get_depth_camera_odom(robot_name)
            print("\nDepth Camera Odometry:")
            print(f"  Position: ({odom.pose.x:.2f}, {odom.pose.y:.2f}, {odom.pose.z:.2f})")
            print(f"  Orientation: ({odom.pose.roll:.2f}, {odom.pose.pitch:.2f}, {odom.pose.yaw:.2f})")
        except Exception as e:
            print(f"Error getting depth camera odometry: {e}")
            
    except Exception as e:
        print(f"Error in depth camera test: {e}")
    finally:
        # Close the client
        client.close()
        print("Client connection closed")

if __name__ == "__main__":
    
    # test_rgb_camera()
    test_depth_camera()