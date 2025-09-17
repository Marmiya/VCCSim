#!/usr/bin/env python3

# Paths configuration - modify these as needed
INPUT_POSE_FILE = r"C:\UEProjects\VCCSimDev\Saved\RuntimeLogs\20250917_193721\BP_PreciseDrone_1C_1D_L_FP_C_1\poses.txt"
OUTPUT_POSE_FILE = r"C:\UEProjects\VCCSimDev\Saved\RuntimeLogs\20250917_193721\BP_PreciseDrone_1C_1D_L_FP_C_1\poses_rh.txt"

import os
import sys

def convert_location_ue_to_rh(ue_x, ue_y, ue_z):
    """
    Convert UE location to right-handed coordinates
    UE: Left-handed, Z-up, centimeters
    RH: Right-handed, Z-up, meters
    Conversion: UE(X,Y,Z) -> RH(Y*0.01, X*0.01, Z*0.01)
    """
    rh_x = ue_y * 0.01  # Y_ue -> X_rh, cm to m
    rh_y = ue_x * 0.01  # X_ue -> Y_rh, cm to m
    rh_z = ue_z * 0.01  # Z_ue -> Z_rh, cm to m
    return rh_x, rh_y, rh_z

def convert_quaternion_ue_to_rh(ue_qx, ue_qy, ue_qz, ue_qw):
    """
    Convert UE quaternion to right-handed rotation
    Swap X and Y components to match coordinate transformation
    """
    rh_qx = ue_qy  # Y_ue -> X_rh
    rh_qy = ue_qx  # X_ue -> Y_rh
    rh_qz = ue_qz  # Z_ue -> Z_rh
    rh_qw = ue_qw  # W unchanged
    return rh_qx, rh_qy, rh_qz, rh_qw

def convert_pose_file():
    """Convert UE pose file to right-handed coordinate system"""

    if not os.path.exists(INPUT_POSE_FILE):
        print(f"Error: Input file not found: {INPUT_POSE_FILE}")
        print("Please update INPUT_POSE_FILE path at the beginning of this script")
        return False

    try:
        with open(INPUT_POSE_FILE, 'r') as infile:
            lines = infile.readlines()

        output_lines = []
        converted_count = 0

        # Add header comments
        output_lines.append("# Right-handed coordinate system poses (Z-up, meters)\n")
        output_lines.append("# Coordinate axes: +X right, +Y forward, +Z up\n")
        output_lines.append("# Format: Timestamp X Y Z Qx Qy Qz Qw\n")
        output_lines.append("# Converted from UE left-handed system\n")
        output_lines.append("#\n")

        for line in lines:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Parse pose line
            values = line.split()

            if len(values) == 8:
                # Parse UE pose: Timestamp X Y Z Qx Qy Qz Qw
                timestamp = float(values[0])
                ue_x = float(values[1])
                ue_y = float(values[2])
                ue_z = float(values[3])
                ue_qx = float(values[4])
                ue_qy = float(values[5])
                ue_qz = float(values[6])
                ue_qw = float(values[7])

                # Convert to right-handed system
                rh_x, rh_y, rh_z = convert_location_ue_to_rh(ue_x, ue_y, ue_z)
                rh_qx, rh_qy, rh_qz, rh_qw = convert_quaternion_ue_to_rh(ue_qx, ue_qy, ue_qz, ue_qw)

                # Write converted pose
                output_line = f"{timestamp:.1f} {rh_x:.6f} {rh_y:.6f} {rh_z:.6f} {rh_qx:.6f} {rh_qy:.6f} {rh_qz:.6f} {rh_qw:.6f}\n"
                output_lines.append(output_line)
                converted_count += 1

            else:
                print(f"Warning: Skipping invalid line (expected 8 values): {line}")

        # Write output file
        os.makedirs(os.path.dirname(OUTPUT_POSE_FILE), exist_ok=True)
        with open(OUTPUT_POSE_FILE, 'w') as outfile:
            outfile.writelines(output_lines)

        print(f"Conversion completed successfully!")
        print(f"Input file: {INPUT_POSE_FILE}")
        print(f"Output file: {OUTPUT_POSE_FILE}")
        print(f"Converted {converted_count} poses")
        print(f"Coordinate system: UE (left-handed, cm) -> Right-handed (meters)")

        return True

    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

if __name__ == "__main__":
    print("VCCSim Pose File Converter: UE -> Right-Handed")
    print("=" * 50)
    convert_pose_file()