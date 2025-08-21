import json
import os
import csv
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
import numpy as np
import math

# DJI Project Configuration
root_file_path = r"C:\Users\Admin\Documents\DJI\DJITerra"
account_name = "18530020772"
project_name = "PolyTech"
result_output_path = r"D:\GaussianTest\DJI_Terra_Result"

# Generate file paths
list_json_path = os.path.join(root_file_path, account_name, project_name, "images", "survey", "image_list.json")
xml_file_path = os.path.join(root_file_path, account_name, project_name, "AT", "BlocksExchangeUndistortAT_WithoutTiePoints.xml")
model_metadata_xml_path = os.path.join(root_file_path, account_name, project_name, "models", "pc", "0", "terra_fbx", "metadata.xml")
output_csv_path = os.path.join(result_output_path, "TerraToUE.csv")

def read_model_origin_from_metadata(metadata_path):
    """Read model coordinate origin from metadata.xml file"""
    try:
        tree = ET.parse(metadata_path)
        root = tree.getroot()
        
        srs_origin_element = root.find("SRSOrigin")
        if srs_origin_element is not None:
            origin_text = srs_origin_element.text
            origin_coords = [float(coord.strip()) for coord in origin_text.split(",")]
            
            if len(origin_coords) >= 3:
                origin_x, origin_y, origin_z = origin_coords[0], origin_coords[1], origin_coords[2]
                print(f"Model origin: X={origin_x}, Y={origin_y}, Z={origin_z}")
                return origin_x, origin_y, origin_z
            else:
                print(f"Invalid SRSOrigin format: {origin_text}")
                return 0, 0, 0
        else:
            print("SRSOrigin element not found")
            return 0, 0, 0
            
    except FileNotFoundError:
        print(f"Metadata file not found: {metadata_path}")
        return 0, 0, 0
    except ET.ParseError as e:
        print(f"Error parsing metadata.xml: {e}")
        return 0, 0, 0
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 0, 0, 0

def convert_terra_to_ue(position_terra, opk_deg):
    """Convert Terra coordinates and OPK angles to UE format"""
    x, y, z = position_terra
    omega, phi, kappa = opk_deg

    # Convert OPK angles (XYZ rotation order) to rotation matrix
    rot_terra = R.from_euler('XYZ', [omega, phi, kappa], degrees=True)
    mat_terra = rot_terra.as_matrix()

    # Coordinate axis swap: X <-> Y
    swap_XY_matrix = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ])

    mat_ue = swap_XY_matrix @ mat_terra @ swap_XY_matrix.T

    # Extract UE-style Yaw/Pitch/Roll angles (ZYX rotation order)
    rot_ue = R.from_matrix(mat_ue)
    yaw, pitch, roll = rot_ue.as_euler('ZYX', degrees=True)

    # Convert to centimeters and swap coordinates
    x *= 100
    y *= 100
    z *= 100
    position_ue = [y, x, z]

    return position_ue, [yaw, pitch, roll]

# Read image list and build mapping
with open(list_json_path, "r", encoding="utf-8") as f:
    list_data = json.load(f)

image_mapping = []
for entry in list_data:
    original_name = entry.get("path", entry.get("origin_path", ""))
    hash_name = entry["id"] + ".jpg"
    image_mapping.append((original_name, hash_name))

# Parse XML file for pose information
tree = ET.parse(xml_file_path)
root = tree.getroot()
photos = root.findall(".//Photo")
pose_dict = {}

origin_x, origin_y, origin_z = read_model_origin_from_metadata(model_metadata_xml_path)

for photo in photos:
    image_path_node = photo.find("ImagePath")
    if image_path_node is None:
        continue
    
    image_path = image_path_node.text.replace("\\", "/")
    hash_name = os.path.basename(image_path)

    pose = photo.find("Pose")
    if pose is not None:
        center = pose.find("Center")
        rotation = pose.find("Rotation")

        terra_x = (float(center.find("x").text) - origin_x)
        terra_y = (float(center.find("y").text) - origin_y)
        terra_z = (float(center.find("z").text))

        omega = float(rotation.find("Omega").text) if rotation is not None else 0.0
        phi = float(rotation.find("Phi").text) if rotation is not None else 0.0
        kappa = float(rotation.find("Kappa").text) if rotation is not None else 0.0

        position_terra = [terra_x, terra_y, terra_z]
        opk_deg = [omega, phi, kappa]
        
        position_ue, ypr_ue = convert_terra_to_ue(position_terra, opk_deg)
        ue_x, ue_y, ue_z = position_ue
        yaw, pitch, roll = ypr_ue

        # Rotate around Z-axis by -90 degrees
        theta = math.radians(-90)
        x2 = ue_x * math.cos(theta) - ue_y * math.sin(theta)
        y2 = ue_x * math.sin(theta) + ue_y * math.cos(theta)
        ue_x, ue_y = x2, y2

        # Apply UE coordinate system adjustments
        roll -= 180
        pitch -= 90
        yaw += 90

        pose_dict[hash_name] = (terra_x, terra_y, terra_z, omega, phi, kappa, 
                               ue_x, ue_y, ue_z, yaw, pitch, roll)

# Output to CSV file
with open(output_csv_path, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["OriginalName", "HashName", 
                    "Terra_X", "Terra_Y", "Terra_Z", "Omega", "Phi", "Kappa",
                    "UE_X", "UE_Y", "UE_Z", "Yaw", "Pitch", "Roll"])

    for original_name, hash_name in image_mapping:
        pose = pose_dict.get(hash_name)
        if pose:
            writer.writerow([original_name, hash_name, *pose])
        else:
            print(f"Pose not found for {hash_name} (original: {original_name})")

# Generate UE format pose.txt file
pose_txt_path = os.path.splitext(output_csv_path)[0] + "_pose.txt"
with open(output_csv_path, "r", encoding="utf-8") as csvfile, open(pose_txt_path, "w", encoding="utf-8") as txtfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        ue_x = row["UE_X"]
        ue_y = row["UE_Y"] 
        ue_z = row["UE_Z"]
        yaw = row["Pitch"]    # Coordinate swap
        pitch = row["Yaw"]    # Coordinate swap
        roll = row["Roll"]
        txtfile.write(f"{ue_x} {ue_y} {ue_z} {yaw} {pitch} {roll}\n")

print(f"Exported CSV: {output_csv_path}")
print(f"Exported pose text: {pose_txt_path}")
print(f"Note: Increase Z value by {origin_z * 100} to position model above ground")