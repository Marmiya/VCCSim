#!/usr/bin/env python3
"""Minimal script to exercise LiDAR-related RPCs."""

import os
import sys
from typing import Iterable, Sequence

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VCCSim import VCCSimClient

HOST = "127.0.0.1"
PORT = 50996
ROBOT_NAME = "Mavic"

OUTPUT_DIR = os.path.abspath("lidar_service_outputs")
LIDAR_DATA_PLY = os.path.join(OUTPUT_DIR, "lidar_data_points.ply")
COMBINED_DATA_PLY = os.path.join(OUTPUT_DIR, "lidar_data_and_odom_points.ply")


def _extract_point_tuples(container) -> list[tuple[float, float, float]]:
    if container is None:
        return []

    def _normalize_point(point):
        if isinstance(point, (list, tuple)):
            return tuple(point)
        if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
            return (point.x, point.y, point.z)
        return None

    if isinstance(container, list):
        result = []
        for point in container:
            normalized_point = _normalize_point(point)
            if normalized_point is not None:
                result.append(normalized_point)
        return result

    if isinstance(container, Sequence) and len(container) > 0:
        first = container[0]
        if _normalize_point(first) is not None:
            result = []
            for point in container:
                normalized_point = _normalize_point(point)
                if normalized_point is not None:
                    result.append(normalized_point)
            return result

    data_field = getattr(container, "data", None)
    if data_field is not None:
        return _extract_point_tuples(data_field)

    if isinstance(container, Iterable):
        result = []
        for point in container:
            normalized_point = _normalize_point(point)
            if normalized_point is not None:
                result.append(normalized_point)
        if result:
            return result

    return []


def save_points_to_ply(points, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("end_header\n")
        for x, y, z in points:
            handle.write(f"{x} {y} {z}\n")
    print(f"Saved {len(points)} points to: {output_path}")


def print_point_preview(points, label: str) -> None:
    print(f"{label} point count:", len(points))
    if points:
        print(f"{label} first point:", points[0])
        if len(points) > 1:
            print(f"{label} last point:", points[-1])


def print_pose(prefix: str, pose) -> None:
    if pose is None:
        print(f"{prefix} pose: <none>")
        return
    position = pose.position
    rotation = pose.rotation
    print(
        f"{prefix} pose position:",
        (position.x, position.y, position.z),
    )
    print(
        f"{prefix} pose rotation:",
        (rotation.x, rotation.y, rotation.z, rotation.w),
    )


if __name__ == "__main__":
    client = VCCSimClient(host=HOST, port=PORT)
    try:
        lidar_points_raw = client.get_lidar_data(ROBOT_NAME)
        lidar_points = _extract_point_tuples(lidar_points_raw)
        print_point_preview(lidar_points, "get_lidar_data")
        save_points_to_ply(lidar_points, LIDAR_DATA_PLY)

        lidar_odom = client.get_lidar_odom(ROBOT_NAME)
        print_pose("get_lidar_odom", getattr(lidar_odom, "pose", None))

        combined_response = client.get_lidar_data_and_odom(ROBOT_NAME)
        if isinstance(combined_response, tuple) and len(combined_response) == 2:
            combined_points_raw, combined_odom = combined_response
        else:
            combined_points_raw = getattr(combined_response, "data", None)
            combined_odom = getattr(combined_response, "odom", None)

        combined_points = _extract_point_tuples(combined_points_raw)
        print_point_preview(combined_points, "get_lidar_data_and_odom")
        save_points_to_ply(combined_points, COMBINED_DATA_PLY)

        odom_pose = getattr(combined_odom, "pose", None)
        print_pose("get_lidar_data_and_odom", odom_pose)
    finally:
        client.close()
