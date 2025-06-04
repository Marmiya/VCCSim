import cv2
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

def downscale_real_images(input_dir, output_dir, target_width=1216, target_height=912, quality=95):
    """
    Downscale real images to match rendered image dimensions.
    
    Args:
        input_dir: Path to directory containing original real images
        output_dir: Path to directory where downscaled images will be saved
        target_width: Target width (default: 1216)
        target_height: Target height (default: 912)
        quality: JPEG quality for output images (default: 95)
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Target resolution: {target_width}x{target_height}")
    
    # Process each image
    for img_file in tqdm(image_files, desc="Downscaling images"):
        try:
            # Read image
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"Warning: Could not read {img_file}")
                continue
            
            original_height, original_width = img.shape[:2]
            print(f"Processing {img_file.name}: {original_width}x{original_height} -> {target_width}x{target_height}")
            
            # Resize image using high-quality interpolation
            resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Determine output filename and format
            output_filename = img_file.stem + '.jpg'  # Convert all to JPG for consistency
            output_filepath = output_path / output_filename
            
            # Save with high quality
            cv2.imwrite(str(output_filepath), resized_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    print(f"\nDownscaling completed! Processed images saved to: {output_dir}")

def verify_image_dimensions(directory, expected_width=1216, expected_height=912):
    """
    Verify that all images in a directory have the expected dimensions.
    """
    dir_path = Path(directory)
    image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(dir_path.glob(f"*{ext}"))
    
    print(f"\nVerifying dimensions in {directory}:")
    print(f"Expected: {expected_width}x{expected_height}")
    
    mismatched = []
    
    for img_file in image_files[:10]:  # Check first 10 images
        try:
            img = cv2.imread(str(img_file))
            if img is not None:
                height, width = img.shape[:2]
                if width != expected_width or height != expected_height:
                    mismatched.append((img_file.name, width, height))
                    print(f"  {img_file.name}: {width}x{height} ❌")
                else:
                    print(f"  {img_file.name}: {width}x{height} ✅")
        except Exception as e:
            print(f"  Error reading {img_file.name}: {e}")
    
    if mismatched:
        print(f"\nFound {len(mismatched)} mismatched images")
        return False
    else:
        print(f"\nAll checked images have correct dimensions!")
        return True

# Usage
if __name__ == "__main__":
    # Paths - updated to match your directory structure when running from VCCSim root
    real_images_input = "./Source/Tools/DGE/data/real_images"
    real_images_output = "./Source/Tools/DGE/data/real_images_downscaled"
    rendered_images_dir = "./Source/Tools/DGE/data/rendered_images"
    
    print("=== Image Downscaling Script ===\n")
    
    # First, let's verify the dimensions of rendered images to confirm target size
    print("Checking rendered image dimensions...")
    verify_image_dimensions(rendered_images_dir, 1216, 912)
    
    # Downscale real images
    print("\nStarting downscaling process...")
    downscale_real_images(
        input_dir=real_images_input,
        output_dir=real_images_output,
        target_width=1216,
        target_height=912,
        quality=95  # High quality to preserve image fidelity
    )
    
    # Verify the output
    print("\nVerifying downscaled images...")
    verify_image_dimensions(real_images_output, 1216, 912)
    
    print("\n=== Process Complete ===")
    print(f"Original images: {real_images_input}")
    print(f"Downscaled images: {real_images_output}")
    print("\nTo use the downscaled images in your analysis, update the data path in difference_analysis.py")