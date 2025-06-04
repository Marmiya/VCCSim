import csv
import os
import math

def convert_rc_to_ue_coordinates(x, y, z, heading, pitch, roll):
    """
    Convert Reality Capture coordinates and rotations to Unreal Engine format
    
    Position conversion:
    - RC Y -> UE X 
    - RC X -> UE Y
    - RC Z -> UE Z
    
    Rotation conversion (verified working):
    - RC pitch -> UE pitch (with -90° offset)
    - RC heading -> UE yaw (direct mapping)
    - RC roll -> UE roll (with sign flip)
    """
    # Position conversion (meters to centimeters)
    ue_x = y * 100.0
    ue_y = x * 100.0
    ue_z = z * 100.0
    
    # Rotation conversion (working solution)
    ue_pitch = pitch - 90.0
    ue_yaw = heading
    ue_roll = -roll
    
    # Normalize angles to [-180, 180] range
    ue_pitch = ((ue_pitch + 180) % 360) - 180
    ue_yaw = ((ue_yaw + 180) % 360) - 180
    ue_roll = ((ue_roll + 180) % 360) - 180
    
    return ue_x, ue_y, ue_z, ue_pitch, ue_yaw, ue_roll


def convert_csv_to_pose_txt(input_csv_path, output_txt_path):
    """
    Convert Reality Capture camera poses CSV to pose.txt format
    Input CSV format expectation:
    Line 1: A comment or non-header line (e.g., '#cameras 904') - THIS LINE IS SKIPPED.
    Line 2: Header row (e.g., '#name,x,y,z,heading,pitch,roll') - USED BY DictReader.
    Remaining lines: Data.
    Output format: X Y Z P Y R (one pose per line)
    """
    poses = []
    
    try:
        with open(input_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            # Skip the first line (assuming it's a comment like '#cameras 904')
            try:
                next(csvfile) 
            except StopIteration:
                print("Error: CSV file is empty or has less than 2 lines (expected comment line then header).")
                return False
            
            # Read CSV with headers from the second line
            reader = csv.DictReader(csvfile)
            
            # Check if headers were found (DictReader.fieldnames)
            if not reader.fieldnames:
                print("Error: CSV DictReader could not find field names (headers).")
                print("Ensure the second line of your CSV contains comma-separated headers like: #name,x,y,z,heading,pitch,roll")
                return False

            expected_fields = ['x', 'y', 'z', 'heading', 'pitch', 'roll']
            missing_fields = [field for field in expected_fields if field not in reader.fieldnames]
            if missing_fields:
                print(f"Error: CSV headers are missing expected fields: {missing_fields}")
                print(f"Found headers: {reader.fieldnames}")
                print("Please ensure your CSV header (second line) contains these fields.")
                return False

            for row_number, row in enumerate(reader, start=1):
                try:
                    x = float(row['x'])
                    y = float(row['y'])
                    z = float(row['z'])
                    heading = float(row['heading'])
                    pitch = float(row['pitch'])
                    roll = float(row['roll'])
                    
                    ue_x, ue_y, ue_z, ue_pitch, ue_yaw, ue_roll = convert_rc_to_ue_coordinates(
                        x, y, z, heading, pitch, roll
                    )
                    
                    poses.append((ue_x, ue_y, ue_z, ue_pitch, ue_yaw, ue_roll))
                    
                except ValueError as e:
                    print(f"Error processing data in row {row_number}: {row}")
                    print(f"ValueError details: {e}. Check if all numerical values are valid.")
                    continue 
                except KeyError as e:
                    print(f"Error processing data in row {row_number}: {row}")
                    print(f"KeyError: Missing expected column '{e}'. Check CSV headers and data consistency.")
                    continue
    
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_csv_path}'")
        return False
    except Exception as e:
        print(f"General error reading CSV file: {e}")
        return False
    
    if not poses:
        print("No poses were processed. Output file will not be created.")
        return False

    try:
        with open(output_txt_path, 'w') as txtfile:
            for pose in poses:
                txtfile.write(f"{pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f} {pose[3]:.6f} {pose[4]:.6f} {pose[5]:.6f}\n")
        
        print(f"Successfully converted {len(poses)} camera poses.")
        print(f"Output saved to: {output_txt_path}")
        return True
        
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False


def convert_csv_to_pose_txt(input_csv_path, output_txt_path):
    """
    Convert Reality Capture camera poses CSV to pose.txt format
    Input CSV format expectation:
    Line 1: A comment or non-header line (e.g., '#cameras 1') - THIS LINE IS SKIPPED.
    Line 2: Header row (e.g., '#name,x,y,z,heading,pitch,roll') - USED BY DictReader.
    Remaining lines: Data.
    Output format: X Y Z P Y R (one pose per line)
    """
    poses = []
    
    try:
        with open(input_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            # Skip the first line (assuming it's a comment like '#cameras 1')
            try:
                next(csvfile) 
            except StopIteration:
                print("Error: CSV file is empty or has less than 2 lines (expected comment line then header).")
                return False
            
            # Read CSV with headers from the second line
            reader = csv.DictReader(csvfile)
            
            # Check if headers were found (DictReader.fieldnames)
            if not reader.fieldnames:
                print("Error: CSV DictReader could not find field names (headers).")
                print("Ensure the second line of your CSV contains comma-separated headers like: #name,x,y,z,heading,pitch,roll")
                return False

            expected_fields = ['x', 'y', 'z', 'heading', 'pitch', 'roll']
            missing_fields = [field for field in expected_fields if field not in reader.fieldnames]
            if missing_fields:
                print(f"Error: CSV headers are missing expected fields: {missing_fields}")
                print(f"Found headers: {reader.fieldnames}")
                print("Please ensure your CSV header (second line) contains these fields.")
                return False

            for row_number, row in enumerate(reader, start=1):
                try:
                    x = float(row['x'])
                    y = float(row['y'])
                    z = float(row['z'])
                    heading = float(row['heading'])
                    pitch = float(row['pitch'])
                    roll = float(row['roll'])
                    
                    ue_x, ue_y, ue_z, ue_pitch, ue_yaw, ue_roll = convert_rc_to_ue_coordinates(
                        x, y, z, heading, pitch, roll
                    )
                    
                    poses.append((ue_x, ue_y, ue_z, ue_pitch, ue_yaw, ue_roll))
                    
                except ValueError as e:
                    print(f"Error processing data in row {row_number}: {row}")
                    print(f"ValueError details: {e}. Check if all numerical values are valid.")
                    continue 
                except KeyError as e:
                    print(f"Error processing data in row {row_number}: {row}")
                    print(f"KeyError: Missing expected column '{e}'. Check CSV headers and data consistency.")
                    continue
    
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_csv_path}'")
        return False
    except Exception as e:
        print(f"General error reading CSV file: {e}")
        return False
    
    if not poses:
        print("No poses were processed. Output file will not be created.")
        return False

    try:
        with open(output_txt_path, 'w') as txtfile:
            for pose in poses:
                txtfile.write(f"{pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f} {pose[3]:.6f} {pose[4]:.6f} {pose[5]:.6f}\n")
        
        print(f"Successfully converted {len(poses)} camera poses.")
        print(f"Output saved to: {output_txt_path}")
        return True
        
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False


def main():
    # Configuration
    input_file = "D:/Data/L7/0528_filtered.csv" 
    output_file = "C:/UEProjects/VCCSimDev/Saved/pose.txt"
    
    print("Reality Capture to Unreal Engine Camera Pose Converter")
    print("=" * 55)
    print("Using verified rotation conversion:")
    print("- RC pitch -> UE pitch (with -90° offset)")
    print("- RC heading -> UE yaw (direct mapping)")  
    print("- RC roll -> UE roll (with sign flip)")
    print()
    
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' not found!")
        return
    
    success = convert_csv_to_pose_txt(input_file, output_file)
    
    if success:
        print("\nConversion completed successfully!")
        print(f"You can now use '{output_file}' with your Unreal Engine cameras.")
        print("\nRotation conversion verified working for drone camera poses.")
    else:
        print("\nConversion failed. Please check error messages.")


# Keep the test function available for future use
def test_all_rotation_modes():
    """Test function to try different rotation modes if needed in the future"""
    input_file = "E:/Scene/0526.csv" 
    
    for mode in range(1, 6):
        output_file = f"C:\\UEProjects\\VCCSimDev\\Saved\\pose_mode{mode}.txt"
        print(f"\nTesting rotation mode {mode}...")
        # This would require the old function with modes
        # convert_csv_to_pose_txt_with_modes(input_file, output_file, mode)


# Alternative single-mode function removed since not needed anymore


if __name__ == "__main__":
    main()