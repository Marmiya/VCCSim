#!/usr/bin/env python3

"""
Convert COLMAP binary files to text format.

This script converts COLMAP binary files (cameras.bin, images.bin, point3D.bin)
to their corresponding text versions (cameras.txt, images.txt, points3D.txt).

Usage:
    python Convert_bin_2_txt.py <colmap_dataset_folder>
    
Example:
    python Convert_bin_2_txt.py /path/to/colmap/sparse/0
"""

import os
import sys
import struct
import numpy as np
from pathlib import Path
import argparse

# COLMAP camera model IDs
CAMERA_MODELS = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE", 
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE"
}

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path_to_model_file):
    """
    Read cameras from COLMAP cameras.bin file.
    
    Returns:
        dict: Dictionary mapping camera_id to camera data
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = {
                'model': CAMERA_MODELS[model_id],
                'width': width,
                'height': height,
                'params': params
            }
    return cameras

def read_images_binary(path_to_model_file):
    """
    Read images from COLMAP images.bin file.
    
    Returns:
        dict: Dictionary mapping image_id to image data
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            
            # Read image name
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            # Read 2D points
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            xys = np.column_stack([np.array(x_y_id_s[0::3]), np.array(x_y_id_s[1::3])])
            point3D_ids = np.array(x_y_id_s[2::3])
            
            images[image_id] = {
                'qvec': qvec,
                'tvec': tvec,
                'camera_id': camera_id,
                'name': image_name,
                'xys': xys,
                'point3D_ids': point3D_ids
            }
    return images

def read_points3D_binary(path_to_model_file):
    """
    Read 3D points from COLMAP points3D.bin file.
    
    Returns:
        dict: Dictionary mapping point3D_id to point data
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, 43, "QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            
            # Read track
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track_elems = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            
            points3D[point3D_id] = {
                'xyz': xyz,
                'rgb': rgb,
                'error': error,
                'image_ids': image_ids,
                'point2D_idxs': point2D_idxs
            }
    return points3D

def write_cameras_text(cameras, path):
    """Write cameras to text file."""
    with open(path, "w") as fid:
        fid.write("# Camera list with one line of data per camera:\n")
        fid.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        fid.write("# Number of cameras: {}\n".format(len(cameras)))
        
        for camera_id, camera in cameras.items():
            params_str = " ".join(map(str, camera['params']))
            fid.write("{} {} {} {} {}\n".format(
                camera_id, camera['model'], camera['width'], 
                camera['height'], params_str))

def write_images_text(images, path):
    """Write images to text file."""
    with open(path, "w") as fid:
        fid.write("# Image list with two lines of data per image:\n")
        fid.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        fid.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        fid.write("# Number of images: {}, mean observations per image: {}\n".format(
            len(images), np.mean([len(img['xys']) for img in images.values()])))
        
        for image_id, image in images.items():
            qvec_str = " ".join(map(str, image['qvec']))
            tvec_str = " ".join(map(str, image['tvec']))
            fid.write("{} {} {} {} {}\n".format(
                image_id, qvec_str, tvec_str, image['camera_id'], image['name']))
            
            points_str = ""
            for xy, point3D_id in zip(image['xys'], image['point3D_ids']):
                points_str += "{} {} {} ".format(xy[0], xy[1], point3D_id)
            fid.write(points_str.rstrip() + "\n")

def write_points3D_text(points3D, path):
    """Write points3D to text file."""
    with open(path, "w") as fid:
        fid.write("# 3D point list with one line of data per point:\n")
        fid.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        fid.write("# Number of points: {}, mean track length: {}\n".format(
            len(points3D), np.mean([len(pt['image_ids']) for pt in points3D.values()])))
        
        for point3D_id, point in points3D.items():
            xyz_str = " ".join(map(str, point['xyz']))
            rgb_str = " ".join(map(str, point['rgb'].astype(int)))
            
            track_str = ""
            for image_id, point2D_idx in zip(point['image_ids'], point['point2D_idxs']):
                track_str += "{} {} ".format(image_id, point2D_idx)
            
            fid.write("{} {} {} {} {}\n".format(
                point3D_id, xyz_str, rgb_str, point['error'], track_str.rstrip()))

# Camera model parameters count
class CameraModel:
    def __init__(self, model_id, model_name, num_params):
        self.model_id = model_id
        self.model_name = model_name
        self.num_params = num_params

CAMERA_MODEL_IDS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
    5: CameraModel(5, "OPENCV_FISHEYE", 8),
    6: CameraModel(6, "FULL_OPENCV", 12),
    7: CameraModel(7, "FOV", 5),
    8: CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    9: CameraModel(9, "RADIAL_FISHEYE", 5),
    10: CameraModel(10, "THIN_PRISM_FISHEYE", 12)
}

def convert_bin_to_txt(dataset_folder):
    """
    Convert COLMAP binary files to text format.
    
    Args:
        dataset_folder: Path to folder containing cameras.bin, images.bin, points3D.bin
    """
    dataset_path = Path(dataset_folder)
    
    if not dataset_path.exists():
        print(f"Error: Dataset folder '{dataset_folder}' does not exist!")
        return False
    
    # File paths
    cameras_bin = dataset_path / "cameras.bin"
    images_bin = dataset_path / "images.bin" 
    points3D_bin = dataset_path / "points3D.bin"
    
    cameras_txt = dataset_path / "cameras.txt"
    images_txt = dataset_path / "images.txt"
    points3D_txt = dataset_path / "points3D.txt"
    
    success = True
    
    # Convert cameras.bin
    if cameras_bin.exists():
        try:
            print(f"Converting {cameras_bin} to {cameras_txt}...")
            cameras = read_cameras_binary(str(cameras_bin))
            write_cameras_text(cameras, str(cameras_txt))
            print(f"Successfully converted {len(cameras)} cameras")
        except Exception as e:
            print(f"Error converting cameras.bin: {e}")
            success = False
    else:
        print(f"Warning: {cameras_bin} not found")
    
    # Convert images.bin
    if images_bin.exists():
        try:
            print(f"Converting {images_bin} to {images_txt}...")
            images = read_images_binary(str(images_bin))
            write_images_text(images, str(images_txt))
            print(f"Successfully converted {len(images)} images")
        except Exception as e:
            print(f"Error converting images.bin: {e}")
            success = False
    else:
        print(f"Warning: {images_bin} not found")
    
    # Convert points3D.bin
    if points3D_bin.exists():
        try:
            print(f"Converting {points3D_bin} to {points3D_txt}...")
            points3D = read_points3D_binary(str(points3D_bin))
            write_points3D_text(points3D, str(points3D_txt))
            print(f"Successfully converted {len(points3D)} 3D points")
        except Exception as e:
            print(f"Error converting points3D.bin: {e}")
            success = False
    else:
        print(f"Warning: {points3D_bin} not found")
    
    return success

def main():
    parser = argparse.ArgumentParser(description="Convert COLMAP binary files to text format")
    parser.add_argument("dataset_folder", nargs='?', 
                       default=r"D:\Data\360_v2\garden\sparse\0",
                       help="Path to folder containing COLMAP binary files (default: D:\\Data\\360_v2\\garden\\sparse\\0)")
    
    args = parser.parse_args()
    
    print(f"Using dataset folder: {args.dataset_folder}")
    
    if convert_bin_to_txt(args.dataset_folder):
        print("\nConversion completed successfully!")
    else:
        print("\nConversion completed with errors!")
        sys.exit(1)

if __name__ == "__main__":
    main()