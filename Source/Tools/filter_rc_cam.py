import csv
import os
import sys

def filter_cameras_by_roll(input_csv_path, output_csv_path, roll_threshold=45.0, 
                          also_filter_pitch=False, pitch_threshold=85.0):
    """
    Filter Reality Capture CSV to remove cameras with problematic roll angles
    
    Parameters:
    - input_csv_path: Path to original RC CSV file
    - output_csv_path: Path for filtered CSV file
    - roll_threshold: Remove cameras with |roll| > this value (default: 45°)
    - also_filter_pitch: If True, also filter extreme pitch angles
    - pitch_threshold: Remove cameras with |pitch| > this value (default: 85°)
    """
    
    if not os.path.exists(input_csv_path):
        print(f"Error: Input file '{input_csv_path}' not found!")
        return False
    
    kept_cameras = []
    removed_cameras = []
    header_lines = []
    
    print(f"Reading CSV: {input_csv_path}")
    print(f"Roll threshold: ±{roll_threshold}°")
    if also_filter_pitch:
        print(f"Pitch threshold: ±{pitch_threshold}°")
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
                print(f"Found fields: {reader.fieldnames}")
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
                    
                    # Check if camera should be kept
                    keep_camera = True
                    removal_reason = []
                    
                    # Check roll angle
                    if abs(roll) > roll_threshold:
                        keep_camera = False
                        removal_reason.append(f"roll={roll:.1f}° (>{roll_threshold}°)")
                    
                    # Check pitch angle if enabled
                    if also_filter_pitch and abs(pitch) > pitch_threshold:
                        keep_camera = False
                        removal_reason.append(f"pitch={pitch:.1f}° (>{pitch_threshold}°)")
                    
                    if keep_camera:
                        # Keep this camera
                        kept_cameras.append({
                            '#name': name,
                            'x': x,
                            'y': y,
                            'z': z,
                            'heading': heading,
                            'pitch': pitch,
                            'roll': roll
                        })
                        print(f"✓ {name}: h={heading:.1f}°, p={pitch:.1f}°, r={roll:.1f}°")
                    else:
                        # Remove this camera
                        removed_cameras.append({
                            '#name': name,
                            'x': x,
                            'y': y,
                            'z': z,
                            'heading': heading,
                            'pitch': pitch,
                            'roll': roll,
                            'reason': ', '.join(removal_reason)
                        })
                        print(f"✗ {name}: h={heading:.1f}°, p={pitch:.1f}°, r={roll:.1f}° - REMOVED ({', '.join(removal_reason)})")
                    
                except (ValueError, KeyError) as e:
                    print(f"Error processing row {row_number}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False
    
    # Summary statistics
    total_cameras = len(kept_cameras) + len(removed_cameras)
    removal_percentage = (len(removed_cameras) / total_cameras * 100) if total_cameras > 0 else 0
    
    print(f"\n" + "=" * 60)
    print(f"FILTERING SUMMARY:")
    print(f"Total cameras: {total_cameras}")
    print(f"Kept cameras: {len(kept_cameras)}")
    print(f"Removed cameras: {len(removed_cameras)} ({removal_percentage:.1f}%)")
    
    if len(kept_cameras) == 0:
        print(f"\nWARNING: No cameras remain after filtering!")
        print(f"Consider increasing the roll threshold.")
        return False
    
    if len(removed_cameras) > total_cameras * 0.5:
        print(f"\nWARNING: More than 50% of cameras were removed.")
        print(f"This suggests significant alignment issues in Reality Capture.")
        print(f"Consider re-running alignment with better settings.")
    
    # Write the filtered CSV
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Write comment line with updated count
            original_count = first_line.split()[1] if len(first_line.split()) > 1 else str(total_cameras)
            new_comment = f"#cameras {len(kept_cameras)}"
            csvfile.write(f"{new_comment}\n")
            
            # Write CSV data
            fieldnames = ['#name', 'x', 'y', 'z', 'heading', 'pitch', 'roll']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for camera in kept_cameras:
                writer.writerow({
                    '#name': camera['#name'],
                    'x': f"{camera['x']:.6f}",
                    'y': f"{camera['y']:.6f}",
                    'z': f"{camera['z']:.6f}",
                    'heading': f"{camera['heading']:.6f}",
                    'pitch': f"{camera['pitch']:.6f}",
                    'roll': f"{camera['roll']:.6f}"
                })
        
        print(f"\nFiltered CSV saved: {output_csv_path}")
        
        # Save removed cameras list for reference
        if removed_cameras:
            removed_list_path = output_csv_path.replace('.csv', '_removed_cameras.txt')
            with open(removed_list_path, 'w', encoding='utf-8') as f:
                f.write("Removed Cameras Report\n")
                f.write("=" * 30 + "\n\n")
                f.write(f"Input file: {input_csv_path}\n")
                f.write(f"Output file: {output_csv_path}\n")
                f.write(f"Roll threshold: ±{roll_threshold}°\n")
                if also_filter_pitch:
                    f.write(f"Pitch threshold: ±{pitch_threshold}°\n")
                f.write(f"\nTotal removed: {len(removed_cameras)} cameras\n\n")
                
                f.write("Removed cameras:\n")
                f.write("-" * 50 + "\n")
                for camera in removed_cameras:
                    f.write(f"{camera['#name']}: {camera['reason']}\n")
                    f.write(f"  Position: ({camera['x']:.2f}, {camera['y']:.2f}, {camera['z']:.2f})\n")
                    f.write(f"  Orientation: h={camera['heading']:.1f}°, p={camera['pitch']:.1f}°, r={camera['roll']:.1f}°\n\n")
            
            print(f"Removed cameras list: {removed_list_path}")
        
        return True
        
    except Exception as e:
        print(f"Error writing output: {e}")
        return False

def analyze_roll_distribution(csv_path):
    """
    Analyze the distribution of roll angles in the CSV to help choose threshold
    """
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found!")
        return
    
    roll_angles = []
    
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            # Skip first line
            next(csvfile)
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                try:
                    roll = float(row['roll'])
                    roll_angles.append(abs(roll))
                except (ValueError, KeyError):
                    continue
    
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    if not roll_angles:
        print("No roll angles found in CSV")
        return
    
    roll_angles.sort()
    n = len(roll_angles)
    
    print(f"\nROLL ANGLE DISTRIBUTION ANALYSIS:")
    print(f"=" * 40)
    print(f"Total cameras: {n}")
    print(f"Min |roll|: {min(roll_angles):.1f}°")
    print(f"Max |roll|: {max(roll_angles):.1f}°")
    print(f"Median |roll|: {roll_angles[n//2]:.1f}°")
    print(f"Mean |roll|: {sum(roll_angles)/n:.1f}°")
    
    # Show distribution at different thresholds
    thresholds = [15, 30, 45, 60, 90]
    print(f"\nCameras that would be REMOVED at different thresholds:")
    for threshold in thresholds:
        removed_count = sum(1 for r in roll_angles if r > threshold)
        percentage = (removed_count / n * 100)
        print(f"  |roll| > {threshold:2d}°: {removed_count:3d} cameras ({percentage:4.1f}%)")

def main():
    """
    Main function
    """
    print("Reality Capture CSV Camera Filter")
    print("=" * 40)
    
    # Configuration
    input_file = "D:/Data/L7/0528.csv"
    output_file = "D:/Data/L7/0528_filtered.csv"
    roll_threshold = 2.0  # Cameras with |roll| > this will be removed
    
    # Option to also filter extreme pitch angles
    filter_pitch = False  # Set to False to only filter roll
    pitch_threshold = 85.0
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print()
    
    # First, analyze the roll distribution to help user choose threshold
    if os.path.exists(input_file):
        analyze_roll_distribution(input_file)
        print()
        
        # Ask user if they want to continue with default threshold
        response = input(f"Continue with roll threshold of ±{roll_threshold}°? (y/n): ").lower().strip()
        if response not in ['y', 'yes', '']:
            new_threshold = input(f"Enter new roll threshold (current: {roll_threshold}): ").strip()
            try:
                roll_threshold = float(new_threshold)
                print(f"Using roll threshold: ±{roll_threshold}°")
            except ValueError:
                print(f"Invalid threshold, using default: ±{roll_threshold}°")
        print()
    
    # Perform the filtering
    success = filter_cameras_by_roll(
        input_file, 
        output_file, 
        roll_threshold=roll_threshold,
        also_filter_pitch=filter_pitch,
        pitch_threshold=pitch_threshold
    )
    
    if success:
        print(f"\n✓ Filtering completed successfully!")
        print(f"\nNext steps:")
        print(f"1. Use the filtered CSV: {output_file}")
        print(f"2. Run your RC->UE conversion script on the filtered file")
        print(f"3. Check if the remaining cameras provide good coverage")
        print(f"4. If too many cameras were removed, consider:")
        print(f"   - Increasing the roll threshold")
        print(f"   - Re-aligning in Reality Capture with better settings")
        print(f"   - Adding more images to improve alignment")
    else:
        print(f"\n✗ Filtering failed.")

if __name__ == "__main__":
    main()