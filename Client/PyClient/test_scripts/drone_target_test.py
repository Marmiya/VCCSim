# !/usr/bin/env python3
import sys
import os
import time
from PIL import Image
import numpy as np
import io
import logging
import traceback

# Add the parent directory to the path to import VCCSimClient
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VCCSim import VCCSimClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_send_drone_pose():
    """Test the send_drone_pose method of VCCSimClient."""

    try:
        # Initialize the client
        logger.info("Initializing VCCSimClient...")
        client = VCCSimClient(host="172.31.178.18", port=50996)

        # Drone name
        drone_name = "Mavic"

        # Get the current drone pose (for reference)
        try:
            current_pose = client.get_drone_pose(drone_name)
            logger.info(f"Current drone pose: x={current_pose.x}, y={current_pose.y}, z={current_pose.z}, "
                        f"roll={current_pose.roll}, pitch={current_pose.pitch}, yaw={current_pose.yaw}")
        except Exception as e:
            logger.warning(f"Could not get current drone pose: {e}")
            logger.info("Using default reference pose")
            current_x, current_y, current_z = 0.0, 0.0, 5.0
            current_roll, current_pitch, current_yaw = 0.0, 0.0, 0.0
        else:
            current_x, current_y, current_z = current_pose.x, current_pose.y, current_pose.z
            current_roll, current_pitch, current_yaw = current_pose.roll, current_pose.pitch, current_pose.yaw

        # Define target poses to test
        test_poses = [
            {
                'name': 'Move Up 1000 units',
                'pose': (current_x, current_y, current_z + 1000, current_roll, current_pitch, current_yaw)
            },
            {
                'name': 'Move Forward 4550 units',
                'pose': (current_x + 4550, current_y, current_z + 1000, current_roll, current_pitch, current_yaw)
            },
            {
                'name': 'Move Right 1310 units with 45° yaw',
                'pose': (current_x + 4550, current_y + 1310, current_z + 1000, current_roll, current_pitch, 45.0)
            },
            {
                'name': 'Return to modified origin',
                'pose': (current_x + 10, current_y + 5, current_z + 5, 0.0, 0.0, 0.0)
            }
        ]

        # Test each pose
        for i, test_case in enumerate(test_poses, 1):
            logger.info(f"\n--- Test {i}: {test_case['name']} ---")
            
            x, y, z, roll, pitch, yaw = test_case['pose']
            logger.info(f"Sending drone to: x={x}, y={y}, z={z}, roll={roll}, pitch={pitch}, yaw={yaw}")
            
            # Send the pose to the drone
            success = client.send_drone_pose(drone_name, x, y, z, roll, pitch, yaw)
            
            if success:
                logger.info("Successfully sent drone pose")
            else:
                logger.error("Failed to send drone pose")
                continue
            
            # Wait for the drone to reach the position
            logger.info("Waiting for drone to reach target position...")
            time.sleep(10)  # Adjust based on your environment
            
            # Verify the new pose
            try:
                new_pose = client.get_drone_pose(drone_name)
                logger.info(f"New drone pose: x={new_pose.x:.2f}, y={new_pose.y:.2f}, z={new_pose.z:.2f}, "
                           f"roll={new_pose.roll:.2f}, pitch={new_pose.pitch:.2f}, yaw={new_pose.yaw:.2f}")
                
                # Check if the drone reached approximately the target position
                tolerance = 1.0  # Position tolerance in units
                angle_tolerance = 5.0  # Angle tolerance in degrees
                
                pos_diff = ((new_pose.x - x)**2 + (new_pose.y - y)**2 + (new_pose.z - z)**2)**0.5
                yaw_diff = abs(new_pose.yaw - yaw)
                
                if pos_diff <= tolerance and yaw_diff <= angle_tolerance:
                    logger.info(f"✓ Drone reached target position (diff: {pos_diff:.2f})")
                else:
                    logger.warning(f"⚠ Drone may not have reached exact target (pos_diff: {pos_diff:.2f}, yaw_diff: {yaw_diff:.2f})")
                    
            except Exception as e:
                logger.error(f"Could not verify new drone pose: {e}")
            
            # Small delay between tests
            time.sleep(1)

        logger.info("\n--- All pose tests completed ---")

    except Exception as e:
        logger.error(f"Error during test execution: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Close the client connection
        try:
            client.close()
            logger.info("Client connection closed")
        except Exception as e:
            logger.error(f"Error closing client connection: {e}")


def test_single_drone_pose():
    """Simple test to send a single drone pose."""
    
    try:
        logger.info("Starting simple drone pose test...")
        client = VCCSimClient(host="localhost", port=50996)
        
        drone_name = "Mavic"
        
        # Simple target pose: move to (100, 50, 20) with no rotation
        target_x, target_y, target_z = 100.0, 50.0, 20.0
        target_roll, target_pitch, target_yaw = 0.0, 0.0, 0.0
        
        logger.info(f"Sending drone to simple target: x={target_x}, y={target_y}, z={target_z}")
        
        success = client.send_drone_pose(drone_name, target_x, target_y, target_z,
                                       target_roll, target_pitch, target_yaw)
        
        if success:
            logger.info("✓ Successfully sent simple drone pose")
            
            # Wait and check final position
            time.sleep(5)
            final_pose = client.get_drone_pose(drone_name)
            logger.info(f"Final position: x={final_pose.x:.2f}, y={final_pose.y:.2f}, z={final_pose.z:.2f}")
            
        else:
            logger.error("✗ Failed to send simple drone pose")
            
        client.close()
        
    except Exception as e:
        logger.error(f"Error in simple test: {e}")


if __name__ == "__main__":
    # You can run either the comprehensive test or the simple test
    print("Choose test type:")
    print("1. Comprehensive pose test (multiple positions)")
    print("2. Simple single pose test")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_send_drone_pose()
    elif choice == "2":
        test_single_drone_pose()
    else:
        logger.info("Running comprehensive test by default...")
        test_send_drone_pose()