import os
import shutil
from pathlib import Path

def get_image_files(directory):
    """Get all image files from a directory, sorted alphabetically."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.exr'}
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist!")
        return []
    
    files = []
    for file in os.listdir(directory):
        if Path(file).suffix.lower() in image_extensions:
            files.append(file)
    
    return sorted(files)

def rename_images_to_match(target_dir, reference_names):
    """Rename images in target_dir to match the base names from reference_names, keeping original extensions."""
    target_files = get_image_files(target_dir)
    
    if len(target_files) != len(reference_names):
        print(f"Warning: Number of files in {target_dir} ({len(target_files)}) doesn't match reference ({len(reference_names)})")
        return False
    
    # Create temporary directory to avoid naming conflicts
    temp_dir = os.path.join(target_dir, "temp_rename")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Move files to temp directory with new names
        for old_file, ref_name in zip(target_files, reference_names):
            old_path = os.path.join(target_dir, old_file)
            
            # Get base name from reference and keep original extension
            ref_base_name = Path(ref_name).stem
            original_extension = Path(old_file).suffix
            new_name = ref_base_name + original_extension
            
            temp_path = os.path.join(temp_dir, new_name)
            shutil.move(old_path, temp_path)
        
        # Move them back to original directory
        for old_file, ref_name in zip(target_files, reference_names):
            ref_base_name = Path(ref_name).stem
            original_extension = Path(old_file).suffix
            new_name = ref_base_name + original_extension
            
            temp_path = os.path.join(temp_dir, new_name)
            final_path = os.path.join(target_dir, new_name)
            shutil.move(temp_path, final_path)
        
        os.rmdir(temp_dir)
        return True
        
    except Exception as e:
        print(f"Error during renaming: {e}")
        # Restore files from temp directory
        try:
            temp_files = os.listdir(temp_dir)
            for temp_file in temp_files:
                temp_path = os.path.join(temp_dir, temp_file)
                restore_path = os.path.join(target_dir, temp_file)
                shutil.move(temp_path, restore_path)
            os.rmdir(temp_dir)
            print("Files restored to target directory")
        except Exception as restore_error:
            print(f"Failed to restore files: {restore_error}")
        return False

def main():
    base_dir = "./Source/Tools/DGE/data"
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist!")
        return
    
    # Define all target directories
    real_images_dir = os.path.join(base_dir, "real_images")
    depth_images_dir = os.path.join(base_dir, "depth_images")
    rendered_images_dir = os.path.join(base_dir, "rendered_images")
    normal_images_dir = os.path.join(base_dir, "normal_images")
    
    # Verify all directories exist
    directories = [
        ("real_images", real_images_dir),
        ("depth_images", depth_images_dir), 
        ("rendered_images", rendered_images_dir),
        ("normal_images", normal_images_dir)
    ]
    
    for dir_name, dir_path in directories:
        if not os.path.exists(dir_path):
            print(f"Directory {dir_path} does not exist!")
            return
    
    # Get reference names from real_images directory
    reference_names = get_image_files(real_images_dir)
    
    if not reference_names:
        print("No image files found in real_images directory!")
        return
    
    print(f"Found {len(reference_names)} reference images")
    
    # Rename images in all target directories
    target_dirs = [
        ("depth_images", depth_images_dir),
        ("rendered_images", rendered_images_dir),
        ("normal_images", normal_images_dir)
    ]
    
    for dir_name, dir_path in target_dirs:
        print(f"Renaming images in {dir_name}...")
        if rename_images_to_match(dir_path, reference_names):
            print(f"✓ Successfully renamed {dir_name}")
        else:
            print(f"✗ Failed to rename {dir_name}")
    
    print("Renaming complete!")

if __name__ == "__main__":
    main()