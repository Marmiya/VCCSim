#!/usr/bin/env python3
"""Minimal script to exercise Flash gRPC APIs."""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VCCSim import VCCSimClient

HOST = "127.0.0.1"
PORT = 50996
FLASH_NAME = "Flash"
POSE_TARGET = (0.0, 0.0, 600.0, 0.0, 0.0, 0.0)
PATH_POSES = [
    (0.0, 0.0, 600.0, 0.0, 0.0, 0.0),
    (200.0, 100.0, 620.0, 0.0, 0.0, 15.0),
    (400.0, 200.0, 640.0, 0.0, 0.0, 30.0),
]
WAIT_AFTER_POSE = 1.5
WAIT_AFTER_PATH = 2.5

if __name__ == "__main__":
    client = VCCSimClient(host=HOST, port=PORT)
    try:
        initial_pose = client.get_flash_pose(FLASH_NAME)
        print("Initial flash pose:", initial_pose)

        pose_result = client.send_flash_pose(FLASH_NAME, *POSE_TARGET)
        print("send_flash_pose returned:", pose_result)
        time.sleep(WAIT_AFTER_POSE)

        ready_status = client.check_flash_ready(FLASH_NAME)
        print("check_flash_ready returned:", ready_status)

        path_result = client.send_flash_path(FLASH_NAME, PATH_POSES)
        print("send_flash_path returned:", path_result)
        time.sleep(WAIT_AFTER_PATH)

        next_result = client.move_flash_to_next(FLASH_NAME)
        print("move_flash_to_next returned:", next_result)

        final_pose = client.get_flash_pose(FLASH_NAME)
        print("Final flash pose:", final_pose)
    finally:
        client.close()
