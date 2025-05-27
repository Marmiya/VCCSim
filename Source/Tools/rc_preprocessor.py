import csv
import math
import os
import numpy as np

def normalize_angle(angle):
    """Normalize angle to [-180, 180] range"""
    return ((angle + 180) % 360) - 180

def euler_to_rotation_matrix(heading, pitch, roll):
    """Convert Euler angles to rotation matrix (ZYX order)"""
    h, p, r = math.radians(heading), math.radians(pitch), math.radians(roll)
    
    # Individual rotation matrices
    cos_h, sin_h = math.cos(h), math.sin(h)
    cos_p, sin_p = math.cos(p), math.sin(p)  
    cos_r, sin_r = math.cos(r), math.sin(r)
    
    # Combined rotation matrix (ZYX order: Roll * Pitch * Heading)
    R = np.array([
        [cos_h*cos_p, cos_h*sin_p*sin_r - sin_h*cos_r, cos_h*sin_p*cos_r + sin_h*sin_r],
        [sin_h*cos_p, sin_h*sin_p*sin_r + cos_h*cos_r, sin_h*sin_p*cos_r - cos_h*sin_r],
        [-sin_p, cos_p*sin_r, cos_p*cos_r]
    ])
    
    return R

def rotation_matrix_to_euler(R):
    """Convert rotation matrix back to Euler angles (ZYX order)"""
    # Extract angles from rotation matrix
    sin_pitch = -R[2, 0]
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)  # Avoid numerical errors
    
    pitch = math.degrees(math.asin(sin_pitch))
    
    if abs(sin_pitch) < 0.99999:  # Not in gimbal lock
        heading = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    else:  # Gimbal lock case
        heading = math.degrees(math.atan2(-R[0, 1], R[1, 1]))
        roll = 0.0
    
    return normalize_angle(heading), normalize_angle(pitch), normalize_angle(roll)

def fix_large_roll_transform_1(heading, pitch, roll):
    """
    Transform 1: 180° rotation around Z-axis (heading)
    Good for poses where camera is "upside down"
    """
    new_heading = normalize_angle(heading + 180)
    new_pitch = normalize_angle(-pitch)
    new_roll = normalize_angle(roll + 180)
    return new_heading, new_pitch, new_roll

def fix_large_roll_transform_2(heading, pitch, roll):
    """
    Transform 2: 180° rotation around Y-axis (pitch)
    Good for poses where camera is "backwards"
    """
    new_heading = normalize_angle(heading + 180)
    new_pitch = normalize_angle(180 - pitch)
    new_roll = normalize_angle(-roll)
    return new_heading, new_pitch, new_roll

def fix_large_roll_transform_3(heading, pitch, roll):
    """
    Transform 3: Fix gimbal lock by swapping confused axes
    For high pitch + high roll, this might be RC confusing pitch/roll
    """
    if abs(pitch) > 85 and abs(roll) > 90:  # Both high - likely axis confusion
        # Try swapping pitch and roll (common RC error)
        new_heading = heading
        new_pitch = normalize_angle(roll - 90)  # Adjust for typical drone orientation
        new_roll = normalize_angle(pitch - 90)
        return new_heading, new_pitch, new_roll
    return heading, pitch, roll

def fix_large_roll_transform_4(heading, pitch, roll):
    """
    Transform 4: Use rotation matrix to find minimal roll equivalent
    """
    # Convert to rotation matrix
    R = euler_to_rotation_matrix(heading, pitch, roll)
    
    # Try different equivalent rotations and pick one with smallest |roll|
    alternatives = []
    
    # Original
    h, p, r = rotation_matrix_to_euler(R)
    alternatives.append((abs(r), h, p, r))
    
    # 180° flip around different axes
    for axis in ['x', 'y', 'z']:
        if axis == 'x':
            flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        elif axis == 'y':
            flip = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        else:  # z
            flip = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        
        R_flipped = R @ flip
        h, p, r = rotation_matrix_to_euler(R_flipped)
        alternatives.append((abs(r), h, p, r))
    
    # Return the one with smallest absolute roll
    _, best_h, best_p, best_r = min(alternatives, key=lambda x: x[0])
    return best_h, best_p, best_r

def analyze_and_fix_pose(heading, pitch, roll, roll_threshold=45.0):
    """
    Analyze a pose and try different transforms to minimize roll angle
    """
    original_roll_abs = abs(roll)
    
    if original_roll_abs <= roll_threshold:
        return heading, pitch, roll, "no_fix_needed"
    
    print(f"  Problematic pose: h={heading:.1f}°, p={pitch:.1f}°, r={roll:.1f}°")
    
    # Try different transform methods
    transforms = [
        ("transform_1", fix_large_roll_transform_1),
        ("transform_2", fix_large_roll_transform_2), 
        ("transform_3", fix_large_roll_transform_3),
        ("transform_4", fix_large_roll_transform_4),
    ]
    
    best_result = (heading, pitch, roll, "original")
    best_roll_abs = original_roll_abs
    
    for transform_name, transform_func in transforms:
        try:
            new_h, new_p, new_r = transform_func(heading, pitch, roll)
            new_roll_abs = abs(new_r)
            
            print(f"    {transform_name}: h={new_h:.1f}°, p={new_p:.1f}°, r={new_r:.1f}° (|r|={new_roll_abs:.1f}°)")
            
            if new_roll_abs < best_roll_abs:
                best_result = (new_h, new_p, new_r, transform_name)
                best_roll_abs = new_roll_abs
                
        except Exception as e:
            print(f"    {transform_name}: Failed - {e}")
    
    final_h, final_p, final_r, method = best_result
    print(f"    -> Best: {method} -> h={final_h:.1f}°, p={final_p:.1f}°, r={final_r:.1f}°")
    
    return final_h, final_p, final_r, method

def preprocess_csv_poses(input_csv_path, output_csv_path, roll_threshold=45.0):
    """
    Preprocess Reality Capture CSV to fix problematic poses with large roll angles
    """
    if not os.path.exists(input_csv_path):
        print(f"Error: Input file '{input_csv_path}' not found!")
        return False
    
    poses_data = []
    header_lines = []
    fixed_count = 0
    total_count = 0
    
    print(f"Reading CSV: {input_csv_path}")
    print(f"Roll threshold: ±{roll_threshold}°")
    print("=" * 60)
    
    try:
        with open(input_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            # Read first line (comment)
            first_line = csvfile.readline().strip()
            header_lines.append(first_line)
            
            # Read and parse CSV data
            reader = csv.DictReader(csvfile)
            
            expected_fields = ['x', 'y', 'z', 'heading', 'pitch', 'roll']
            missing_fields = [field for field in expected_fields if field not in reader.fieldnames]
            if missing_fields:
                print(f"Error: Missing fields: {missing_fields}")
                return False
            
            for row_number, row in enumerate(reader, start=1):
                try:
                    name = row.get('#name', f'Camera_{row_number}')
                    x = float(row['x'])
                    y = float(row['y'])
                    z = float(row['z'])
                    heading = float(row['heading'])
                    pitch = float(row['pitch'])
                    roll = float(row['roll'])
                    
                    total_count += 1
                    
                    # Analyze and potentially fix the pose
                    print(f"\n{name}:")
                    new_heading, new_pitch, new_roll, method = analyze_and_fix_pose(
                        heading, pitch, roll, roll_threshold
                    )
                    
                    if method != "no_fix_needed":
                        fixed_count += 1
                    
                    # Store the (potentially fixed) pose
                    poses_data.append({
                        '#name': name,
                        'x': x,
                        'y': y, 
                        'z': z,
                        'heading': new_heading,
                        'pitch': new_pitch,
                        'roll': new_roll,
                        'original_roll': roll,
                        'fix_method': method
                    })
                    
                except (ValueError, KeyError) as e:
                    print(f"Error processing row {row_number}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False
    
    # Write the preprocessed CSV
    print(f"\n" + "=" * 60)
    print(f"Summary:")
    print(f"Total poses: {total_count}")
    print(f"Fixed poses: {fixed_count}")
    print(f"Unchanged poses: {total_count - fixed_count}")
    
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Write comment line
            csvfile.write(f"{header_lines[0]}\n")
            
            # Write CSV data
            fieldnames = ['#name', 'x', 'y', 'z', 'heading', 'pitch', 'roll']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for pose in poses_data:
                writer.writerow({
                    '#name': pose['#name'],
                    'x': pose['x'],
                    'y': pose['y'],
                    'z': pose['z'],
                    'heading': f"{pose['heading']:.6f}",
                    'pitch': f"{pose['pitch']:.6f}",
                    'roll': f"{pose['roll']:.6f}"
                })
        
        print(f"Preprocessed CSV saved: {output_csv_path}")
        
        # Create a summary report
        report_path = output_csv_path.replace('.csv', '_fix_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Reality Capture Pose Preprocessing Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input file: {input_csv_path}\n")
            f.write(f"Output file: {output_csv_path}\n")
            f.write(f"Roll threshold: ±{roll_threshold}°\n\n")
            f.write(f"Total poses: {total_count}\n")
            f.write(f"Fixed poses: {fixed_count}\n")
            f.write(f"Unchanged poses: {total_count - fixed_count}\n\n")
            
            f.write("Detailed fixes:\n")
            for pose in poses_data:
                if pose['fix_method'] != "no_fix_needed":
                    f.write(f"{pose['#name']}: {pose['original_roll']:.1f}° -> {pose['roll']:.1f}° ({pose['fix_method']})\n")
        
        print(f"Fix report saved: {report_path}")
        return True
        
    except Exception as e:
        print(f"Error writing output: {e}")
        return False

def main():
    """
    Main function for CSV pose preprocessing
    """
    print("Reality Capture CSV Pose Preprocessor")
    print("=" * 40)
    
    # Configuration
    input_file = "E:/Scene/0527.csv"  # Updated to match your file
    output_file = "E:/Scene/0527_fixed.csv"
    roll_threshold = 10.0  # Poses with |roll| > this will be fixed
    
    print(f"This tool will:")
    print(f"1. Read camera poses from: {input_file}")
    print(f"2. Identify poses with |roll| > {roll_threshold}°")
    print(f"3. Try mathematical transforms to minimize roll angles")
    print(f"4. Save fixed poses to: {output_file}")
    print()
    
    success = preprocess_csv_poses(input_file, output_file, roll_threshold)
    
    if success:
        print(f"\n✓ Preprocessing completed!")
        print(f"You can now use the fixed CSV file: {output_file}")
        print(f"Then run your RC->UE conversion script on the fixed file.")
    else:
        print(f"\nX Preprocessing failed.")

if __name__ == "__main__":
    main()