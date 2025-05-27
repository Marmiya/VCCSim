import csv
import os
import shutil
import sys
from pathlib import Path

def get_camera_names_from_csv(csv_path):
    """
    Extract camera names from filtered CSV file
    """
    camera_names = []
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found!")
        return []
    
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            # Skip first line (comment)
            next(csvfile)
            
            reader = csv.DictReader(csvfile)
            
            if '#name' not in reader.fieldnames:
                print(f"Error: '#name' column not found in CSV")
                print(f"Available columns: {reader.fieldnames}")
                return []
            
            for row in reader:
                name = row['#name'].strip()
                if name:
                    camera_names.append(name)
        
        print(f"Found {len(camera_names)} camera names in CSV")
        return camera_names
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

def find_image_files(source_dir, camera_names):
    """
    Find image files in source directory that match camera names
    Handles different file extensions (.jpg, .JPG, .jpeg, .JPEG)
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' not found!")
        return {}
    
    # Common image extensions to check
    image_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tif', '.TIF', '.tiff', '.TIFF']
    
    found_files = {}
    missing_files = []
    
    print(f"\nSearching for images in: {source_dir}")
    print("-" * 50)
    
    for camera_name in camera_names:
        found = False
        
        # Try different extensions
        for ext in image_extensions:
            # Remove extension from camera name if present, then add new extension
            base_name = os.path.splitext(camera_name)[0]
            test_filename = base_name + ext
            test_path = os.path.join(source_dir, test_filename)
            
            if os.path.exists(test_path):
                found_files[camera_name] = test_path
                print(f"✓ Found: {test_filename}")
                found = True
                break
        
        if not found:
            missing_files.append(camera_name)
            print(f"✗ Missing: {camera_name}")
    
    print(f"\nSummary:")
    print(f"Found images: {len(found_files)}")
    print(f"Missing images: {len(missing_files)}")
    
    if missing_files:
        print(f"\nMissing image files:")
        for missing in missing_files:
            print(f"  - {missing}")
    
    return found_files, missing_files

def copy_filtered_images(source_dir, output_dir, camera_names, copy_mode='copy'):
    """
    Copy images corresponding to filtered camera names
    
    Parameters:
    - source_dir: Directory containing original images
    - output_dir: Directory to copy filtered images to
    - camera_names: List of camera names from filtered CSV
    - copy_mode: 'copy' or 'move' files
    """
    
    # Find matching image files
    found_files, missing_files = find_image_files(source_dir, camera_names)
    
    if not found_files:
        print("No matching image files found!")
        return False
    
    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return False
    
    # Copy/move files
    copied_count = 0
    failed_count = 0
    
    print(f"\nCopying images...")
    print("-" * 50)
    
    for camera_name, source_path in found_files.items():
        try:
            filename = os.path.basename(source_path)
            dest_path = os.path.join(output_dir, filename)
            
            if copy_mode == 'move':
                shutil.move(source_path, dest_path)
                action = "Moved"
            else:
                shutil.copy2(source_path, dest_path)  # copy2 preserves metadata
                action = "Copied"
            
            copied_count += 1
            print(f"{action}: {filename}")
            
        except Exception as e:
            failed_count += 1
            print(f"Failed to copy {camera_name}: {e}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"COPY OPERATION SUMMARY:")
    print(f"Total cameras in CSV: {len(camera_names)}")
    print(f"Images found: {len(found_files)}")
    print(f"Images copied: {copied_count}")
    print(f"Images failed: {failed_count}")
    print(f"Images missing: {len(missing_files)}")
    
    if missing_files:
        print(f"\nWARNING: {len(missing_files)} images were not found in source directory")
        print(f"This might indicate:")
        print(f"- Different file naming convention")
        print(f"- Images in subdirectories")
        print(f"- Missing image files")
    
    # Create a summary report
    try:
        report_path = os.path.join(output_dir, 'copy_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Filtered Images Copy Report\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Source directory: {source_dir}\n")
            f.write(f"Output directory: {output_dir}\n")
            f.write(f"CSV file: {csv_path}\n\n")
            f.write(f"Total cameras in CSV: {len(camera_names)}\n")
            f.write(f"Images found: {len(found_files)}\n")
            f.write(f"Images copied: {copied_count}\n")
            f.write(f"Images failed: {failed_count}\n")
            f.write(f"Images missing: {len(missing_files)}\n\n")
            
            if copied_count > 0:
                f.write("Successfully copied images:\n")
                f.write("-" * 40 + "\n")
                for camera_name, source_path in found_files.items():
                    filename = os.path.basename(source_path)
                    f.write(f"{filename}\n")
            
            if missing_files:
                f.write(f"\nMissing images:\n")
                f.write("-" * 40 + "\n")
                for missing in missing_files:
                    f.write(f"{missing}\n")
        
        print(f"\nDetailed report saved: {report_path}")
        
    except Exception as e:
        print(f"Warning: Could not create report file: {e}")
    
    return copied_count > 0

def preview_operation(csv_path, source_dir):
    """
    Preview what would be copied without actually copying
    """
    print("PREVIEW MODE - No files will be copied")
    print("=" * 50)
    
    camera_names = get_camera_names_from_csv(csv_path)
    if not camera_names:
        return
    
    found_files, missing_files = find_image_files(source_dir, camera_names)
    
    total_size = 0
    if found_files:
        print(f"\nCalculating total size...")
        for source_path in found_files.values():
            try:
                size = os.path.getsize(source_path)
                total_size += size
            except:
                pass
        
        total_size_mb = total_size / (1024 * 1024)
        print(f"Total size to copy: {total_size_mb:.1f} MB")

def main():
    """
    Main function
    """
    print("Copy Filtered Camera Images")
    print("=" * 30)
    
    # Configuration - Update these paths
    csv_path = "E:/Scene/0527_filtered.csv"  # Your filtered CSV
    source_dir = "E:/Scene/Imgs"  # Directory with original images  
    output_dir = "E:/Scene/Filtered_Images"  # Output directory for filtered images
    
    print(f"CSV file: {csv_path}")
    print(f"Source images: {source_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if files exist
    if not os.path.exists(csv_path):
        print(f"\nError: CSV file not found!")
        print(f"Please update csv_path in the script to point to your filtered CSV file")
        return
    
    if not os.path.exists(source_dir):
        print(f"\nError: Source directory not found!")
        print(f"Please update source_dir in the script to point to your images directory")
        return
    
    # Get camera names from CSV
    camera_names = get_camera_names_from_csv(csv_path)
    if not camera_names:
        return
    
    print(f"\nFirst few camera names:")
    for i, name in enumerate(camera_names[:5]):
        print(f"  {i+1}. {name}")
    if len(camera_names) > 5:
        print(f"  ... and {len(camera_names) - 5} more")
    
    # Ask user what to do
    print(f"\nOptions:")
    print(f"1. Preview (see what would be copied)")
    print(f"2. Copy images")
    print(f"3. Move images")
    
    choice = input(f"\nChoose option (1-3): ").strip()
    
    if choice == '1':
        preview_operation(csv_path, source_dir)
    elif choice == '2':
        print(f"\nCopying images...")
        success = copy_filtered_images(source_dir, output_dir, camera_names, copy_mode='copy')
        if success:
            print(f"\n✓ Copy operation completed!")
            print(f"Filtered images are now in: {output_dir}")
        else:
            print(f"\n✗ Copy operation failed!")
    elif choice == '3':
        print(f"\nMoving images...")
        confirm = input(f"This will MOVE files from source. Are you sure? (y/n): ").lower().strip()
        if confirm in ['y', 'yes']:
            success = copy_filtered_images(source_dir, output_dir, camera_names, copy_mode='move')
            if success:
                print(f"\n✓ Move operation completed!")
                print(f"Filtered images are now in: {output_dir}")
            else:
                print(f"\n✗ Move operation failed!")
        else:
            print("Move operation cancelled")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()