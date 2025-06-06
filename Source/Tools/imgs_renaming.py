import os
import shutil
from pathlib import Path

def get_image_files(directory):
    """Get all image files from a directory, sorted alphabetically."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist!")
        return []
    
    files = []
    for file in os.listdir(directory):
        if Path(file).suffix.lower() in image_extensions:
            files.append(file)
    
    return sorted(files)

def rename_images_to_match(target_dir, reference_names):
    """
    Rename images in target_dir to match the base names from reference_names.
    Preserves the original file extensions (e.g., .png files stay .png, .jpg files stay .jpg).
    """

def rename_images_to_match(target_dir, reference_names):
    """Rename images in target_dir to match the base names from reference_names, keeping original extensions."""
    target_files = get_image_files(target_dir)
    
    if len(target_files) != len(reference_names):
        print(f"Warning: Number of files in {target_dir} ({len(target_files)}) doesn't match reference ({len(reference_names)})")
        return False
    
    # Create a temporary directory to avoid naming conflicts
    temp_dir = os.path.join(target_dir, "temp_rename")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # First, move all files to temp directory with new names (keeping original extensions)
        for i, (old_file, ref_name) in enumerate(zip(target_files, reference_names)):
            old_path = os.path.join(target_dir, old_file)
            
            # Get the base name from reference (without extension) and keep original extension
            ref_base_name = Path(ref_name).stem
            original_extension = Path(old_file).suffix
            new_name = ref_base_name + original_extension
            
            temp_path = os.path.join(temp_dir, new_name)
            
            print(f"Renaming: {old_file} -> {new_name}")
            shutil.move(old_path, temp_path)
        
        # Then move them back to the original directory
        for old_file, ref_name in zip(target_files, reference_names):
            ref_base_name = Path(ref_name).stem
            original_extension = Path(old_file).suffix
            new_name = ref_base_name + original_extension
            
            temp_path = os.path.join(temp_dir, new_name)
            final_path = os.path.join(target_dir, new_name)
            shutil.move(temp_path, final_path)
        
        # Remove temporary directory
        os.rmdir(temp_dir)
        return True
        
    except Exception as e:
        print(f"Error during renaming: {e}")
        # Try to restore files from temp directory if something went wrong
        try:
            temp_files = os.listdir(temp_dir)
            for temp_file in temp_files:
                temp_path = os.path.join(temp_dir, temp_file)
                # Move back to target directory with temp name, user can fix manually
                restore_path = os.path.join(target_dir, temp_file)
                shutil.move(temp_path, restore_path)
                print(f"Restored: {temp_file}")
            os.rmdir(temp_dir)
            print("Files restored to target directory with temporary names")
        except Exception as restore_error:
            print(f"Failed to restore files: {restore_error}")
            print(f"Please check temp directory: {temp_dir}")
        return False

def main():
    # Try to find the DGE/data directory automatically
    possible_paths = [
        "Source/Tools/DGE/data",    # From VCCSim root
        "Tools/DGE/data",           # From Source directory
        "DGE/data",                 # From Tools directory
        "data",                     # From DGE directory
        "./Source/Tools/DGE/data",  # Current directory variation
        "../DGE/data",              # If running from a subdirectory
        "../../DGE/data",           # If running from deeper subdirectory
    ]
    
    base_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            base_dir = path
            break
    
    if base_dir is None:
        print("Could not find DGE/data directory. Please specify the correct path.")
        print("Looking for directories containing: real_images, depth_maps, rendered_images")
        return
    
    print(f"Using base directory: {base_dir}")
    
    real_images_dir = os.path.join(base_dir, "real_images")
    depth_maps_dir = os.path.join(base_dir, "depth_images")  # Fixed typo
    rendered_images_dir = os.path.join(base_dir, "rendered_images")
    
    # Verify all directories exist
    for dir_name, dir_path in [("real_images", real_images_dir), 
                               ("depth_images", depth_maps_dir), 
                               ("rendered_images", rendered_images_dir)]:
        if not os.path.exists(dir_path):
            print(f"Directory {dir_path} does not exist!")
            return
    
    # Get reference names from real_images directory
    print("Getting reference names from real_images...")
    reference_names = get_image_files(real_images_dir)
    
    if not reference_names:
        print("No image files found in real_images directory!")
        return
    
    print(f"Found {len(reference_names)} reference images:")
    for name in reference_names[:5]:  # Show first 5 as example
        print(f"  - {name}")
    if len(reference_names) > 5:
        print(f"  ... and {len(reference_names) - 5} more")
    
    # Rename depth_maps (PNG files will keep .png extension)
    print(f"\nRenaming images in depth_maps...")
    if rename_images_to_match(depth_maps_dir, reference_names):
        print("✓ Successfully renamed depth_maps images")
    else:
        print("✗ Failed to rename depth_maps images")
    
    # Rename rendered_images (PNG files will keep .png extension)
    print(f"\nRenaming images in rendered_images...")
    if rename_images_to_match(rendered_images_dir, reference_names):
        print("✓ Successfully renamed rendered_images")
    else:
        print("✗ Failed to rename rendered_images")
    
    print("\nRenaming complete!")

if __name__ == "__main__":
    main()