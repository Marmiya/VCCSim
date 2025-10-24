#!/usr/bin/env python3
"""Minimal script to exercise car gRPC APIs."""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VCCSim import VCCSimClient

HOST = "127.0.0.1"
PORT = 50996
CAR_NAME = "Husky"
POSE_TARGET = (-50.0, 0.0, 0.0, 0.0)
PATH_POSES = [
    (0.0, 0.0, 0.0, 0.0),
    (100.0, 20.0, 0.0, 10.0),
    (200.0, 40.0, 0.0, 25.0),
]
WAIT_AFTER_POSE = 1.5
WAIT_AFTER_PATH = 3.0


def test_car_pose(client: VCCSimClient) -> None:
    initial_odom = client.get_car_odom(CAR_NAME)
    print("Initial odom for pose test:", initial_odom)

    # pose_result = client.send_car_pose(CAR_NAME, *POSE_TARGET)
    # print("send_car_pose returned:", pose_result)
    # time.sleep(WAIT_AFTER_POSE)

    odom_after_pose = client.get_car_odom(CAR_NAME)
    print("Odom after pose:", odom_after_pose)


def test_car_path(client: VCCSimClient) -> None:
    initial_odom = client.get_car_odom(CAR_NAME)
    print("Initial odom for path test:", initial_odom)

    path_result = client.send_car_path(CAR_NAME, PATH_POSES)
    print("send_car_path returned:", path_result)
    time.sleep(WAIT_AFTER_PATH)

    final_odom = client.get_car_odom(CAR_NAME)
    print("Final odom:", final_odom)


def prompt_mode() -> str:
    print("Select test mode:")
    print("  1) Pose only")
    print("  2) Path only")
    print("  3) Both pose and path")
    choice = input("Enter choice (1/2/3): ").strip()
    return choice


if __name__ == "__main__":
    mode = prompt_mode()
    client = VCCSimClient(host=HOST, port=PORT)
    try:
        if mode == "1":
            test_car_pose(client)
        elif mode == "2":
            test_car_path(client)
        else:
            print("Running both pose and path tests.")
            test_car_pose(client)
            test_car_path(client)
    finally:
        client.close()
