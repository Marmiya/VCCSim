#!/usr/bin/env python3
"""Minimal script to exercise get_drone_pose and send_drone_pose."""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VCCSim import VCCSimClient

HOST = "127.0.0.1"
PORT = 50996
DRONE_NAME = "Mavic"
TARGET_POSE = (1200.0, 0.0, 600.0, 0.0, 0.0, 30.0)
WAIT_SECONDS = 2.0

if __name__ == "__main__":
    client = VCCSimClient(host=HOST, port=PORT)
    try:
        initial_pose = client.get_drone_pose(DRONE_NAME)
        print("Initial pose:", initial_pose)

        result = client.send_drone_pose(DRONE_NAME, *TARGET_POSE)
        print("send_drone_pose returned:", result)

        time.sleep(WAIT_SECONDS)

        updated_pose = client.get_drone_pose(DRONE_NAME)
        print("Updated pose:", updated_pose)
    finally:
        client.close()
