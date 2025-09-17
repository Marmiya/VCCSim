#!/usr/bin/env python3
"""
OBJ to FBX converter with material and texture support
Requires Blender to be installed and accessible via command line
"""

# Configure paths here
OBJ_PATH = r"E:\BaoAn\model4\terra_obj\BlockA\BlockA.obj"
FBX_PATH = r"E:\BaoAn\model4\fbx\BlockA.fbx"

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def create_blender_script(obj_path, fbx_path):
    """Create Blender Python script for conversion"""
    script_content = f"""
import bpy
import os

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import OBJ with materials (use new operator for Blender 4.x)
try:
    # Try new Blender 4.x operator first
    bpy.ops.wm.obj_import(
        filepath=r"{obj_path}",
        use_split_objects=False,
        use_split_groups=False
    )
    print("Used new OBJ import operator (Blender 4.x)")
except AttributeError:
    # Fallback to old operator for Blender 3.x and earlier
    bpy.ops.import_scene.obj(
        filepath=r"{obj_path}",
        use_split_objects=False,
        use_split_groups=False,
        use_groups_as_vgroups=False,
        use_image_search=True,
        split_mode='OFF'
    )
    print("Used legacy OBJ import operator (Blender 3.x)")

# Reset origin to model center
bpy.ops.object.select_all(action='SELECT')
bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]

# Calculate the bounding box center for all selected objects
if bpy.context.selected_objects:
    # Get all mesh objects
    mesh_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    
    if mesh_objects:
        # Calculate combined bounding box
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
        for obj in mesh_objects:
            # Get world matrix
            matrix_world = obj.matrix_world
            
            # Get mesh vertices in world space
            for vertex in obj.data.vertices:
                world_vertex = matrix_world @ vertex.co
                min_x = min(min_x, world_vertex.x)
                max_x = max(max_x, world_vertex.x)
                min_y = min(min_y, world_vertex.y)
                max_y = max(max_y, world_vertex.y)
                min_z = min(min_z, world_vertex.z)
                max_z = max(max_z, world_vertex.z)
        
        # Calculate center point
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2
        
        # Move all objects to center the model at origin
        for obj in mesh_objects:
            obj.location.x -= center_x
            obj.location.y -= center_y
            obj.location.z -= center_z
        
        print(f"Model centered: moved by ({{-center_x:.3f}}, {{-center_y:.3f}}, {{-center_z:.3f}})")
        print(f"Bounding box: X[{{min_x:.3f}}, {{max_x:.3f}}] Y[{{min_y:.3f}}, {{max_y:.3f}}] Z[{{min_z:.3f}}, {{max_z:.3f}}]")
    else:
        print("No mesh objects found to center")
else:
    print("No objects selected for centering")

# Export as FBX (use new operator for Blender 4.x)
try:
    # Try new Blender 4.x operator first
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
    print("Used FBX export operator")
except Exception as e:
    print(f"FBX export failed: {{e}}")
    raise e

print("Conversion completed successfully")
"""
    return script_content

def find_blender_executable():
    """Find Blender executable in common locations"""
    # First check if blender is in PATH
    if shutil.which("blender"):
        return "blender"
    
    # Check common paths based on OS
    if os.name == 'nt':  # Windows
        # Since Blender is in PATH, we don't need to check specific paths
        pass
    else:  # Unix-like systems
        unix_paths = [
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            "/Applications/Blender.app/Contents/MacOS/Blender",  # macOS
        ]
        for path in unix_paths:
            if os.path.isfile(path):
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
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Clean up temporary script
        os.unlink(temp_script_path)
        
        if result.returncode == 0:
            if output_fbx.exists():
                print(f"✓ Conversion successful: {output_fbx}")
                return True
            else:
                print("✗ Conversion failed: Output file not created")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
        else:
            print(f"✗ Blender execution failed (return code: {result.returncode})")
            print(f"STDOUT: {result.stdout}")
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