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


def test_send_drone_path():
    """Test the send_drone_path method of VCCSimClient."""

    try:
        # Initialize the client
        logger.info("Initializing VCCSimClient...")
        client = VCCSimClient(host="172.31.178.18", port=50996)

        # Drone name
        drone_name = "Mavic"

        # Get the current drone pose (to determine the starting point)
        try:
            current_pose = client.get_drone_pose(drone_name)
            logger.info(f"Current drone pose: x={current_pose.x}, y={current_pose.y}, z={current_pose.z}, "
                        f"roll={current_pose.roll}, pitch={current_pose.pitch}, yaw={current_pose.yaw}")
        except Exception as e:
            logger.warning(f"Could not get current drone pose: {e}")
            logger.info("Using default starting pose")
            # Default starting pose if the current one cannot be retrieved
            current_x, current_y, current_z = 0.0, 0.0, 5.0
            current_roll, current_pitch, current_yaw = 0.0, 0.0, 0.0
        else:
            current_x, current_y, current_z = current_pose.x, current_pose.y, current_pose.z
            current_roll, current_pitch, current_yaw = current_pose.roll, current_pose.pitch, current_pose.yaw

        # Define a path for the drone to follow (S-shaped trajectory example)
        # Each point is defined as an offset relative to the current starting position:
        # (x, y, z, roll, pitch, yaw)
        path = [

            (current_x + 0, current_y + 0, current_z + 10, current_roll, current_pitch, current_yaw),

            (current_x + 200, current_y + 143, current_z + 11, current_roll, current_pitch, current_yaw),

            (current_x + 400, current_y + 88, current_z + 10, current_roll, current_pitch, current_yaw),

            (current_x + 600, current_y - 88, current_z + 9, current_roll, current_pitch, current_yaw),

            (current_x + 800, current_y - 143, current_z + 10, current_roll, current_pitch, current_yaw),

            (current_x + 1000, current_y + 0, current_z + 10, current_roll, current_pitch, current_yaw),
        ]

        # Send the path to the drone
        logger.info("Sending drone path...")
        success = client.send_drone_path(drone_name, path)

        if success:
            logger.info("Successfully sent drone path")
        else:
            logger.error("Failed to send drone path")

        # Wait for the drone to complete the path
        logger.info("Waiting for drone to complete path...")
        time.sleep(10)  # Adjust this based on your environment and the length of the path

        # Get the final pose to verify
        try:
            final_pose = client.get_drone_pose(drone_name)
            logger.info(f"Final drone pose: x={final_pose.x}, y={final_pose.y}, z={final_pose.z}, "
                        f"roll={final_pose.roll}, pitch={final_pose.pitch}, yaw={final_pose.yaw}")
        except Exception as e:
            logger.error(f"Could not get final drone pose: {e}")

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


if __name__ == "__main__":
    test_send_drone_path()
