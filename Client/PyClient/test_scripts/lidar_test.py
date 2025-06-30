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
    
    def collect_lidar_data_10hz(self, duration_seconds=None, save_data=True):
        """Collect LiDAR data at 10 Hz frequency.
        
        Args:
            duration_seconds: How long to collect data (None for infinite)
            save_data: Whether to save collected data to files
        """
        if not self.client:
            print("Error: Not connected to server")
            return
        
        target_frequency = 10.0  # Hz
        target_period = 1.0 / target_frequency  # 0.1 seconds
        
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
                    
                    # Print progress every 50 iterations (every 5 seconds at 10Hz)
                    if iteration_count % 50 == 0:
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
    
    def save_collected_data(self):
        """Save the collected LiDAR data to files."""
        if not self.data_points:
            print("No data to save")
            return
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary data
        summary_filename = f"lidar_summary_{timestamp_str}.csv"
        with open(summary_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'iteration', 'point_count', 'pose_x', 'pose_y', 'pose_z', 
                           'pose_roll', 'pose_pitch', 'pose_yaw'])
            
            for entry in self.data_points:
                pose = entry['pose']
                writer.writerow([
                    entry['timestamp'], entry['iteration'], entry['point_count'],
                    pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw
                ])
        
        print(f"Saved summary data to {summary_filename}")
        
        # Save frequency statistics
        if self.frequency_stats:
            freq_filename = f"lidar_frequency_stats_{timestamp_str}.csv"
            with open(freq_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'actual_frequency', 'actual_period'])
                
                for stat in self.frequency_stats:
                    writer.writerow([stat['timestamp'], stat['actual_frequency'], stat['actual_period']])
            
            print(f"Saved frequency statistics to {freq_filename}")
        
        # Optionally save point cloud data for a few samples (to avoid huge files)
        sample_indices = [0, len(self.data_points)//4, len(self.data_points)//2, 
                         3*len(self.data_points)//4, -1]
        
        # for i, idx in enumerate(sample_indices):
        #     if 0 <= idx < len(self.data_points):
        #         points_filename = f"lidar_points_sample_{i}_{timestamp_str}.csv"
        #         with open(points_filename, 'w', newline='') as csvfile:
        #             writer = csv.writer(csvfile)
        #             writer.writerow(['x', 'y', 'z'])
                    
        #             for point in self.data_points[idx]['points']:
        #                 writer.writerow([point[0], point[1], point[2]])
                
        #         print(f"Saved sample point cloud {i} to {points_filename}")
    
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
        # Start data collection at 10 Hz
        collector.collect_lidar_data_10hz(duration_seconds=DURATION, save_data=True)
    
    finally:
        # Clean up
        collector.disconnect()

if __name__ == "__main__":
    main()