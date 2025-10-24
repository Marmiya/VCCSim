#!/usr/bin/env python3
"""Minimal script to exercise send_drone_path."""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VCCSim import VCCSimClient

HOST = "127.0.0.1"
PORT = 50996
DRONE_NAME = "Mavic"
PATH_POSES = [
    (0.0, 0.0, 500.0, 0.0, 0.0, 0.0),
    (500.0, 0.0, 550.0, 0.0, 0.0, 10.0),
    (1000.0, 200.0, 600.0, 0.0, 0.0, 20.0),
]
WAIT_SECONDS = 3.0

if __name__ == "__main__":
    client = VCCSimClient(host=HOST, port=PORT)
    try:
        initial_pose = client.get_drone_pose(DRONE_NAME)
        print("Initial pose:", initial_pose)

        result = client.send_drone_path(DRONE_NAME, PATH_POSES)
        print("send_drone_path returned:", result)

        time.sleep(WAIT_SECONDS)

        final_pose = client.get_drone_pose(DRONE_NAME)
        print("Final pose:", final_pose)
    finally:
        client.close()
