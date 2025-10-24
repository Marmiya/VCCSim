#!/usr/bin/env python3
"""Exercise camera service RPCs and capture outputs."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VCCSim import VCCSimClient
from VCCSim.VCCSimClient import RGBImageUtils

HOST = "127.0.0.1"
PORT = 50996
ROBOT_NAME = "Mavic"
CAMERA_INDEX = 0

OUTPUT_BASE_DIR = os.path.abspath("camera_service_outputs")
IMAGES_DIR = os.path.join(OUTPUT_BASE_DIR, "images")
POINT_CLOUD_DIR = os.path.join(OUTPUT_BASE_DIR, "point_clouds")

RGB_IMAGE_PATH = os.path.join(IMAGES_DIR, "rgb_camera.png")
DEPTH_IMAGE_PATH = os.path.join(IMAGES_DIR, "depth_camera.png")
SEGMENTATION_IMAGE_PATH = os.path.join(IMAGES_DIR, "segmentation_camera.png")

def main() -> None:
    client = VCCSimClient(host=HOST, port=PORT)
    try:
        os.makedirs(IMAGES_DIR, exist_ok=True)
        os.makedirs(POINT_CLOUD_DIR, exist_ok=True)

        rgb_frame = client.get_rgb_data(ROBOT_NAME, CAMERA_INDEX)
        print("RGB frame width:", rgb_frame.width)
        print("RGB frame height:", rgb_frame.height)
        print("RGB frame timestamp:", getattr(rgb_frame, "timestamp", 0))
        print("RGB frame data length:", len(rgb_frame.data))
        if RGBImageUtils.save_rgb_image(rgb_frame, RGB_IMAGE_PATH):
            print("RGB snapshot saved to:", RGB_IMAGE_PATH)
        else:
            print("Failed to save RGB snapshot")

        depth_frame = client.get_depth_data(ROBOT_NAME, CAMERA_INDEX)
        print("Depth frame width:", depth_frame.width)
        print("Depth frame height:", depth_frame.height)
        print("Depth frame timestamp:", getattr(depth_frame, "timestamp", 0))
        print("Depth sample count:", len(depth_frame.data))
        if len(depth_frame.data) > 0:
            print("First depth sample:", depth_frame.data[0])
        if RGBImageUtils.save_depth_image(depth_frame, DEPTH_IMAGE_PATH):
            print("Depth image saved to:", DEPTH_IMAGE_PATH)
        else:
            print("Failed to save depth image")

        seg_frame = client.get_seg_data(ROBOT_NAME, CAMERA_INDEX)
        print("Segmentation frame width:", seg_frame.width)
        print("Segmentation frame height:", seg_frame.height)
        print("Segmentation frame timestamp:", getattr(seg_frame, "timestamp", 0))
        print("Segmentation data length:", len(seg_frame.data))
        if RGBImageUtils.save_segmentation_image(seg_frame, SEGMENTATION_IMAGE_PATH):
            print("Segmentation image saved to:", SEGMENTATION_IMAGE_PATH)
        else:
            print("Failed to save segmentation image")

        normal_frame = client.get_normal_data(ROBOT_NAME, CAMERA_INDEX)
        print("Normal frame width:", normal_frame.width)
        print("Normal frame height:", normal_frame.height)
        print("Normal frame timestamp:", getattr(normal_frame, "timestamp", 0))
        print("Normal vector count:", len(normal_frame.data))
        if normal_frame.data:
            first_normal = normal_frame.data[0]
            print("First normal vector:", (first_normal.x, first_normal.y, first_normal.z))
    finally:
        client.close()


if __name__ == "__main__":
    main()
