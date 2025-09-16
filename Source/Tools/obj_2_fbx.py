#!/usr/bin/env python3
"""
OBJ to FBX converter with material and texture support
Requires Blender to be installed and accessible via command line
"""

# Configure paths here
OBJ_PATH = r"E:\BaoAn\model4\terra_obj\BlockX\BlockX.obj"
FBX_PATH = r"E:\BaoAn\model4\fbx\BlockX.fbx"

import os
import sys
import subprocess
import tempfile
from pathlib import Path

def create_blender_script(obj_path, fbx_path):
    """Create Blender Python script for conversion"""
    script_content = f"""
import bpy
import os

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import OBJ with materials
bpy.ops.import_scene.obj(
    filepath=r"{obj_path}",
    use_split_objects=False,
    use_split_groups=False,
    use_groups_as_vgroups=False,
    use_image_search=True,
    split_mode='OFF'
)

# Export as FBX
bpy.ops.export_scene.fbx(
    filepath=r"{fbx_path}",
    use_selection=False,
    use_active_collection=False,
    global_scale=1.0,
    apply_unit_scale=True,
    apply_scale_options='FBX_SCALE_NONE',
    use_space_transform=True,
    bake_space_transform=False,
    object_types={{'MESH', 'OTHER', 'ARMATURE'}},
    use_mesh_modifiers=True,
    use_mesh_modifiers_render=True,
    mesh_smooth_type='OFF',
    use_subsurf=False,
    use_mesh_edges=False,
    use_tspace=False,
    use_custom_props=False,
    add_leaf_bones=True,
    primary_bone_axis='Y',
    secondary_bone_axis='X',
    use_armature_deform_only=False,
    armature_nodetype='NULL',
    bake_anim=True,
    bake_anim_use_all_bones=True,
    bake_anim_use_nla_strips=True,
    bake_anim_use_all_actions=True,
    bake_anim_force_startend_keying=True,
    bake_anim_step=1.0,
    bake_anim_simplify_factor=1.0,
    path_mode='AUTO',
    embed_textures=False,
    batch_mode='OFF',
    use_batch_own_dir=True,
    use_metadata=True,
    axis_forward='-Z',
    axis_up='Y'
)

print("Conversion completed successfully")
"""
    return script_content

def find_blender_executable():
    """Find Blender executable in common locations"""
    common_paths = [
        "blender",  # If in PATH
        "/usr/bin/blender",  # Linux
        "/usr/local/bin/blender",  # Linux
        "/Applications/Blender.app/Contents/MacOS/Blender",  # macOS
        "C:\\Program Files\\Blender Foundation\\Blender 4.0\\blender.exe",  # Windows
        "C:\\Program Files\\Blender Foundation\\Blender 3.6\\blender.exe",  # Windows
        "C:\\Program Files\\Blender Foundation\\Blender 3.3\\blender.exe",  # Windows
    ]
    
    for path in common_paths:
        if os.path.isfile(path) or subprocess.run(['which', path], 
                                                capture_output=True).returncode == 0:
            return path
    
    return None

def convert_obj_to_fbx(obj_file, output_fbx=None):
    """
    Convert OBJ file to FBX format with materials and textures
    
    Args:
        obj_file (str): Path to input OBJ file
        output_fbx (str, optional): Path to output FBX file
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    
    # Validate input file
    obj_path = Path(obj_file)
    if not obj_path.exists():
        print(f"Error: OBJ file not found: {obj_file}")
        return False
    
    if not obj_path.suffix.lower() == '.obj':
        print(f"Error: Input file must be .obj format")
        return False
    
    # Set output path
    if output_fbx is None:
        output_fbx = obj_path.with_suffix('.fbx')
    else:
        output_fbx = Path(output_fbx)
    
    # Find Blender executable
    blender_exe = find_blender_executable()
    if not blender_exe:
        print("Error: Blender not found. Please install Blender and ensure it's in your PATH")
        print("Download from: https://www.blender.org/download/")
        return False
    
    try:
        # Create temporary script file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
            script_content = create_blender_script(str(obj_path.absolute()), 
                                                 str(output_fbx.absolute()))
            temp_script.write(script_content)
            temp_script_path = temp_script.name
        
        # Run Blender in background mode
        cmd = [
            blender_exe,
            "--background",
            "--python", temp_script_path
        ]
        
        print(f"Converting {obj_file} to {output_fbx}...")
        print(f"Using Blender: {blender_exe}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Clean up temporary script
        os.unlink(temp_script_path)
        
        if result.returncode == 0:
            if output_fbx.exists():
                print(f"✓ Conversion successful: {output_fbx}")
                return True
            else:
                print("✗ Conversion failed: Output file not created")
                return False
        else:
            print(f"✗ Blender execution failed:")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Conversion timed out")
        return False
    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        return False

def main():
    """Main function"""
    input_obj = OBJ_PATH
    output_fbx = FBX_PATH
    
    # Check for associated files
    obj_path = Path(input_obj)
    mtl_path = obj_path.with_suffix('.mtl')
    
    # Create output directory if it doesn't exist
    output_path = Path(output_fbx)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Input OBJ: {input_obj}")
    if mtl_path.exists():
        print(f"Found MTL: {mtl_path}")
    
    # Find texture files
    texture_files = list(obj_path.parent.glob(f"{obj_path.stem}_*.jpg"))
    texture_files.extend(obj_path.parent.glob(f"{obj_path.stem}_*.png"))
    
    if texture_files:
        print(f"Found {len(texture_files)} texture files:")
        for tex in texture_files:
            print(f"  - {tex.name}")
    
    # Perform conversion
    success = convert_obj_to_fbx(input_obj, output_fbx)
    
    if success:
        print("\n🎉 Conversion completed successfully!")
    else:
        print("\n❌ Conversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()