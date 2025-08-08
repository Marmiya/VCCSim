import json
import os
import csv
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
import numpy as np
import math

# DJI Project Name
root_file_path = r"C:\Users\Admin\Documents\DJI\DJITerra"
account_name = "18530020772"
# project_name = "可见光项目(1)"
project_name = "PolyTech"
result_output_path = r"D:\GaussianTest\DJI_Terra_Result"

# Auto generate output CSV file path
list_json_path = os.path.join(
    root_file_path,
    account_name,
    project_name,
    "images",
    "survey",
    "image_list.json"
)
# Auto generate XML file path
xml_file_path = os.path.join(
    root_file_path,
    account_name,
    project_name,
    "AT",
    "BlocksExchangeUndistortAT_WithoutTiePoints.xml"
)
# Auto generate model metadata XML file path, this file is used to get the Origin in spatial reference system
model_metadata_xml_path = os.path.join(
    root_file_path,
    account_name,
    project_name,
    "models",
    "pc",
    "0",
    "terra_fbx",
    "metadata.xml"
)
# Main output directory
output_csv_path = os.path.join(
    # Main output directory
    result_output_path,
    # output file name
    "TerraToUE.csv"
)

# read model origin from metadata.xml
def read_model_origin_from_metadata(metadata_path):
    """
    Read model coordinate origin from metadata.xml file
    
    Parameters:
        metadata_path: Path to metadata.xml file
    
    Returns:
        tuple: (origin_x, origin_y, origin_z) coordinate origin values
    """
    try:
        # Parse metadata XML file
        tree = ET.parse(metadata_path)
        root = tree.getroot()
        
        # Find SRSOrigin element
        srs_origin_element = root.find("SRSOrigin")
        if srs_origin_element is not None:
            # Parse comma-separated origin coordinates
            origin_text = srs_origin_element.text
            origin_coords = [float(coord.strip()) for coord in origin_text.split(",")]
            
            if len(origin_coords) >= 3:
                origin_x, origin_y, origin_z = origin_coords[0], origin_coords[1], origin_coords[2]
                print(f"✅ Successfully read model origin from metadata.xml: X={origin_x}, Y={origin_y}, Z={origin_z}")
                
                return origin_x, origin_y, origin_z
            else:
                print(f"⚠️ Invalid SRSOrigin format in metadata.xml: {origin_text}")
                return 0, 0, 0
        else:
            print("⚠️ SRSOrigin element not found in metadata.xml")
            return 0, 0, 0
            
    except FileNotFoundError:
        print(f"⚠️ Metadata file not found: {metadata_path}")
        print("Using default origin values: (0, 0, 0)")
        return 0, 0, 0
    except ET.ParseError as e:
        print(f"⚠️ Error parsing metadata.xml: {e}")
        print("Using default origin values: (0, 0, 0)")
        return 0, 0, 0
    except Exception as e:
        print(f"⚠️ Unexpected error reading metadata.xml: {e}")
        print("Using default origin values: (0, 0, 0)")
        return 0, 0, 0

# convert Terra coordinates and OPK angles to UE format
def convert_terra_to_ue(position_terra, opk_deg):
    """
    Convert Terra format coordinates and Omega/Phi/Kappa angles to UE position and Yaw/Pitch/Roll
    
    Parameters:
        position_terra: Terra position vector [X, Y, Z]
        opk_deg: Terra attitude angles [Omega, Phi, Kappa] (in degrees)
    
    Returns:
        position_ue: Converted position [X, Y, Z] for UE coordinate system
        ypr_deg: Converted rotation angles [Yaw, Pitch, Roll] (in degrees)
    """
    # Unpack input parameters
    x, y, z = position_terra
   
    omega, phi, kappa = opk_deg

    # Step 1: Convert OPK angles (XYZ rotation order) to rotation matrix
    rot_terra = R.from_euler('XYZ', [omega, phi, kappa], degrees=True)
    mat_terra = rot_terra.as_matrix()

    # Step 2: Coordinate axis swap: X <-> Y (active transformation on rotation matrix)
    # swap_XY_matrix transforms Terra axes to UE axes
    swap_XY_matrix = np.array([
        [0, 1, 0],  # new X = old Y
        [1, 0, 0],  # new Y = old X
        [0, 0, 1],  # Z remains unchanged
    ])

    # Apply coordinate transformation to rotation matrix
    mat_ue = swap_XY_matrix @ mat_terra @ swap_XY_matrix.T

    # Step 3: Extract UE-style Yaw/Pitch/Roll angles (ZYX rotation order)
    rot_ue = R.from_matrix(mat_ue)
    yaw, pitch, roll = rot_ue.as_euler('ZYX', degrees=True)

    x *= 100  # Convert from meters to centimeters
    y *= 100  # Convert from meters to centimeters
    z *= 100  # Convert from meters to centimeters

    # Step 4: Transform position coordinates: X <-> Y swap
    position_ue = [y, x, z]  # Only swap X and Y coordinates

    return position_ue, [yaw, pitch, roll]

# ✅ Read list.json to build mapping from original image names to hash names (preserve order)
with open(list_json_path, "r", encoding="utf-8") as f:
    list_data = json.load(f)

# Build image name mapping list
image_mapping = []
for entry in list_data:
    # Get original image path/name
    original_name = entry.get("path", entry.get("origin_path", ""))
    # Generate hash-based filename
    hash_name = entry["id"] + ".jpg"
    image_mapping.append((original_name, hash_name))

# ✅ Parse BlocksExchange XML file to build hash name → pose information mapping
tree = ET.parse(xml_file_path)
root = tree.getroot()

# Find all Photo elements in XML
photos = root.findall(".//Photo")
pose_dict = {}

# Camera position origin offset (reference point adjustment)
origin_x, origin_y, origin_z = read_model_origin_from_metadata(model_metadata_xml_path)

# Extract pose information for each Photo
for photo in photos:
    # Get image path from XML
    image_path_node = photo.find("ImagePath")
    if image_path_node is None:
        continue
    image_path = image_path_node.text.replace("\\", "/")
    hash_name = os.path.basename(image_path)

    # Extract Pose information
    pose = photo.find("Pose")
    if pose is not None:
        center = pose.find("Center")
        rotation = pose.find("Rotation")

        # Terra coordinates
        terra_x = (float(center.find("x").text) - origin_x)
        terra_y = (float(center.find("y").text) - origin_y)
        terra_z = (float(center.find("z").text))

        # Terra rotation angles (OPK format)4781
    
        omega = float(rotation.find("Omega").text) if rotation is not None else 0.0
        phi = float(rotation.find("Phi").text) if rotation is not None else 0.0
        kappa = float(rotation.find("Kappa").text) if rotation is not None else 0.0

        # Use convert_terra_to_ue function to perform coordinate system conversion
        position_terra = [terra_x, terra_y, terra_z]
        opk_deg = [omega, phi, kappa]
        
        position_ue, ypr_ue = convert_terra_to_ue(position_terra, opk_deg)
        ue_x, ue_y, ue_z = position_ue
        yaw, pitch, roll = ypr_ue

        # >>> 在此处添加：绕 Z 轴旋转 90°
        theta = math.radians(-90)
        # 2D 坐标旋转
        x2 = ue_x * math.cos(theta) - ue_y * math.sin(theta)
        y2 = ue_x * math.sin(theta) + ue_y * math.cos(theta)
        ue_x, ue_y = x2, y2
        # 航向角加 90° 并归一化到 [-180,180]
        # yaw = (yaw + 90 + 180) % 360 - 180
        # <<< 添加结束

        # Apply additional angle adjustments for UE coordinate system alignment
        roll -= 180   # Roll angle offset correction
        pitch -= 90   # Pitch angle offset correction  
        yaw += 90    # Yaw angle offset correction

        # Store all information: Terra original coordinates+angles, UE converted coordinates+angles
        pose_dict[hash_name] = (terra_x, terra_y, terra_z, omega, phi, kappa, 
                               ue_x, ue_y, ue_z, yaw, pitch, roll)

# ✅ Output results to CSV file in original image order
with open(output_csv_path, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # Write CSV header row
    writer.writerow(["OriginalName", "HashName", 
                    "Terra_X", "Terra_Y", "Terra_Z", "Omega", "Phi", "Kappa",
                    "UE_X", "UE_Y", "UE_Z", "Yaw", "Pitch", "Roll"])

    # Write pose
    #  data for each image in original order
    for original_name, hash_name in image_mapping:
        pose = pose_dict.get(hash_name)
        if pose:
            writer.writerow([original_name, hash_name, *pose])
        else:
            # 这里增加：从 imagelist 得到的原文件名一并输出
            print(f"⚠️ 未找到该图片的位姿：{hash_name}，原文件名：{original_name}，已跳过")

# === Generate UE format pose.txt file ===
pose_txt_path = os.path.splitext(output_csv_path)[0] + "_pose.txt"
with open(output_csv_path, "r", encoding="utf-8") as csvfile, open(pose_txt_path, "w", encoding="utf-8") as txtfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Output UE format coordinates and rotation angles
        # Note: There seems to be a coordinate swap in the original code (Yaw<->Pitch)
        ue_x = row["UE_X"]
        ue_y = row["UE_Y"] 
        ue_z = row["UE_Z"]
        yaw = row["Pitch"]    # Using Pitch value as Yaw (coordinate swap)
        pitch = row["Yaw"]    # Using Yaw value as Pitch (coordinate swap)
        roll = row["Roll"]
        # Write pose data in space-separated format
        txtfile.write(f"{ue_x} {ue_y} {ue_z} {yaw} {pitch} {roll}\n")

# Print completion messages
print(f"\n✅ Successfully exported pose mapping CSV: {output_csv_path}")
print(f"✅ Successfully exported UE format pose text: {pose_txt_path}")
print(f"⚠️ ⚠️ ⚠️ Please increase Z value by {origin_z * 100} to let model float above ground!!")