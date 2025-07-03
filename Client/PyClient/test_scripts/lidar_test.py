import sys
import os
import time
import numpy as np
import csv
from datetime import datetime
import signal
import threading

# Add the parent directory of PyClient to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VCCSim import VCCSimClient
from VCCSim import VCCSim_pb2

class LiDARDataCollector:
    def __init__(self, host="172.31.178.18", port=50996, robot_name="Husky"):
        """Initialize the LiDAR data collector.
        
        Args:
            host: Server hostname
            port: Server port
            robot_name: Name of the robot to collect data from
        """
        self.host = host
        self.port = port
        self.robot_name = robot_name
        self.client = None
        self.running = False
        self.data_points = []
        self.frequency_stats = []
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        self.stop_collection()
    
    def connect(self):
        """Connect to the VCCSim server."""
        try:
            print(f"Connecting to VCCSim server at {self.host}:{self.port}")
            self.client = VCCSimClient(host=self.host, port=self.port)
            print(f"Successfully connected! Using robot: {self.robot_name}")
            return True
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the VCCSim server."""
        if self.client:
            self.client.close()
            print("Disconnected from server")
    
    def collect_lidar_data_5hz(self, duration_seconds=None, save_data=True):
        """Collect LiDAR data at 5 Hz frequency.
        
        Args:
            duration_seconds: How long to collect data (None for infinite)
            save_data: Whether to save collected data to files
        """
        if not self.client:
            print("Error: Not connected to server")
            return
        
        target_frequency = 5.0  # Hz
        target_period = 1.0 / target_frequency  # 0.2 seconds
        
        self.running = True
        start_time = time.time()
        iteration_count = 0
        
        print(f"Starting LiDAR data collection at {target_frequency} Hz")
        print("Press Ctrl+C to stop collection")
        
        try:
            while self.running:
                loop_start_time = time.time()
                
                # Request LiDAR data
                try:
                    # Get both LiDAR data and odometry
                    points, odom = self.client.get_lidar_data_and_odom(self.robot_name)
                    
                    # Record data with timestamp
                    timestamp = time.time()
                    data_entry = {
                        'timestamp': timestamp,
                        'iteration': iteration_count,
                        'points': points,
                        'pose': odom.pose,
                        'twist': odom.twist,
                        'point_count': len(points)
                    }
                    
                    if save_data:
                        self.data_points.append(data_entry)
                    
                    # Calculate actual frequency
                    if iteration_count > 0:
                        actual_period = timestamp - self.frequency_stats[-1]['timestamp'] if self.frequency_stats else target_period
                        actual_frequency = 1.0 / actual_period if actual_period > 0 else 0
                        
                        self.frequency_stats.append({
                            'timestamp': timestamp,
                            'actual_frequency': actual_frequency,
                            'actual_period': actual_period
                        })
                    
                    # Print progress every 25 iterations (every 5 seconds at 5Hz)
                    if iteration_count % 25 == 0:
                        elapsed_time = timestamp - start_time
                        avg_frequency = iteration_count / elapsed_time if elapsed_time > 0 else 0
                        print(f"Iteration {iteration_count}: {len(points)} points, "
                              f"Avg freq: {avg_frequency:.2f} Hz, "
                              f"Elapsed: {elapsed_time:.1f}s")
                
                except Exception as e:
                    print(f"Error getting LiDAR data at iteration {iteration_count}: {e}")
                
                # Check if we should stop based on duration
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    print(f"Reached target duration of {duration_seconds} seconds")
                    break
                
                # Calculate sleep time to maintain target frequency
                loop_elapsed = time.time() - loop_start_time
                sleep_time = max(0, target_period - loop_elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif loop_elapsed > target_period * 1.5:  # Warning if we're running too slow
                    print(f"Warning: Loop took {loop_elapsed:.3f}s (target: {target_period:.3f}s)")
                
                iteration_count += 1
        
        except KeyboardInterrupt:
            print("\nCollection stopped by user")
        except Exception as e:
            print(f"Unexpected error during collection: {e}")
        finally:
            self.running = False
            
        # Print summary statistics
        total_time = time.time() - start_time
        print(f"\nCollection Summary:")
        print(f"Total iterations: {iteration_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average frequency: {iteration_count / total_time:.2f} Hz")
        
        if self.frequency_stats:
            frequencies = [stat['actual_frequency'] for stat in self.frequency_stats]
            print(f"Frequency stats - Mean: {np.mean(frequencies):.2f} Hz, "
                  f"Std: {np.std(frequencies):.2f} Hz, "
                  f"Min: {np.min(frequencies):.2f} Hz, "
                  f"Max: {np.max(frequencies):.2f} Hz")
        
        # Save data if requested
        if save_data and self.data_points:
            self.save_collected_data()
    
    def save_pointcloud_as_ply(self, points, filename):
        """Save point cloud data as PLY file.
        
        Args:
            points: List of 3D points [(x, y, z), ...]
            filename: Output filename
        """
        try:
            with open(filename, 'w') as f:
                # Write PLY header
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(points)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("end_header\n")
                
                # Write point data
                for point in points:
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
                
                # Ensure data is written to disk
                f.flush()
                os.fsync(f.fileno())
            
        except Exception as e:
            print(f"Error saving PLY file {filename}: {e}")
            raise  # Re-raise to ensure we know about failures
    
    def save_collected_data(self):
        """Save the collected LiDAR data to files."""
        if not self.data_points:
            print("No data to save")
            return
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        output_dir = f"lidar_data_{timestamp_str}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save pose data in txt format: x y z pitch yaw roll
        pose_filename = os.path.join(output_dir, f"poses_{timestamp_str}.txt")
        try:
            with open(pose_filename, 'w') as f:
                f.write("# x y z pitch yaw roll\n")  # Header comment
                for entry in self.data_points:
                    pose = entry['pose']
                    # Format: x y z pitch yaw roll
                    f.write(f"{pose.x:.6f} {pose.y:.6f} {pose.z:.6f} {pose.pitch:.6f} {pose.yaw:.6f} {pose.roll:.6f}\n")
            
            print(f"Saved pose data to {pose_filename}")
        except Exception as e:
            print(f"Error saving pose file: {e}")
        
        # Save summary data (CSV)
        summary_filename = os.path.join(output_dir, f"lidar_summary_{timestamp_str}.csv")
        try:
            with open(summary_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'iteration', 'point_count', 'pose_x', 'pose_y', 'pose_z', 
                               'pose_pitch', 'pose_yaw', 'pose_roll'])
                
                for entry in self.data_points:
                    pose = entry['pose']
                    writer.writerow([
                        entry['timestamp'], entry['iteration'], entry['point_count'],
                        pose.x, pose.y, pose.z, pose.pitch, pose.yaw, pose.roll
                    ])
            
            print(f"Saved summary data to {summary_filename}")
        except Exception as e:
            print(f"Error saving summary file: {e}")
        
        # Save frequency statistics
        if self.frequency_stats:
            freq_filename = os.path.join(output_dir, f"lidar_frequency_stats_{timestamp_str}.csv")
            try:
                with open(freq_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['timestamp', 'actual_frequency', 'actual_period'])
                    
                    for stat in self.frequency_stats:
                        writer.writerow([stat['timestamp'], stat['actual_frequency'], stat['actual_period']])
                
                print(f"Saved frequency statistics to {freq_filename}")
            except Exception as e:
                print(f"Error saving frequency stats: {e}")
        
        # Save point clouds as PLY files (save all frames)
        print(f"Saving {len(self.data_points)} point cloud PLY files...")
        ply_dir = os.path.join(output_dir, "pointclouds")
        os.makedirs(ply_dir, exist_ok=True)
        
        for i, entry in enumerate(self.data_points):
            ply_filename = os.path.join(ply_dir, f"pointcloud_{i:06d}_{timestamp_str}.ply")
            self.save_pointcloud_as_ply(entry['points'], ply_filename)
            
            # Progress indicator for large datasets
            if (i + 1) % 10 == 0 or i == len(self.data_points) - 1:
                print(f"  Saved {i + 1}/{len(self.data_points)} PLY files...")
        
        # Ensure all I/O operations complete
        import sys
        sys.stdout.flush()
        
        print(f"All data saved to directory: {output_dir}")
        print(f"Total files: {len(self.data_points)} poses, {len(self.data_points)} PLY files")
    
    def stop_collection(self):
        """Stop the data collection."""
        self.running = False

def main():
    # Configuration
    HOST = "172.31.178.18"  # Change to your server IP
    PORT = 50996
    ROBOT_NAME = "Husky"  # Change to your robot name
    DURATION = None  # Set to a number of seconds to limit collection time, or None for infinite
    
    # Create collector instance
    collector = LiDARDataCollector(host=HOST, port=PORT, robot_name=ROBOT_NAME)
    
    # Connect to server
    if not collector.connect():
        return
    
    try:
        # Start data collection at 5 Hz
        collector.collect_lidar_data_5hz(duration_seconds=DURATION, save_data=True)
    
    except Exception as e:
        print(f"Error during data collection: {e}")
    
    finally:
        # Ensure all operations complete before cleanup
        print("Finalizing data saving...")
        time.sleep(0.5)  # Brief pause to ensure all I/O completes
        
        # Clean up
        collector.disconnect()
        
        # Final flush of all outputs
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        print("Program completed successfully.")

if __name__ == "__main__":
    main()