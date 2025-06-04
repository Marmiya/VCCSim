import numpy as np
import os
import sys

def read_ply_file(filepath):
    """
    Read PLY file and return vertex data
    Supports both ASCII and binary formats
    """
    vertices = []
    other_data = []
    header_lines = []
    vertex_count = 0
    is_binary = False
    vertex_properties = []
    
    try:
        with open(filepath, 'rb') as f:
            # Read header
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            
            if line != 'ply':
                raise ValueError("Not a valid PLY file")
            
            # Parse header
            while True:
                line = f.readline().decode('ascii').strip()
                header_lines.append(line)
                
                if line.startswith('format'):
                    if 'binary' in line:
                        is_binary = True
                        if 'little_endian' in line:
                            endian = '<'
                        else:
                            endian = '>'
                    else:
                        is_binary = False
                        
                elif line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                    
                elif line.startswith('property'):
                    if 'vertex' in header_lines[-2] or len(vertex_properties) < 10:  # Assume first properties are vertex properties
                        parts = line.split()
                        if len(parts) >= 3:
                            prop_type = parts[1]
                            prop_name = parts[2]
                            vertex_properties.append((prop_name, prop_type))
                            
                elif line == 'end_header':
                    break
            
            # Find x, y, z indices
            x_idx = y_idx = z_idx = -1
            for i, (name, _) in enumerate(vertex_properties):
                if name == 'x':
                    x_idx = i
                elif name == 'y':
                    y_idx = i
                elif name == 'z':
                    z_idx = i
            
            if x_idx == -1 or y_idx == -1 or z_idx == -1:
                raise ValueError("Could not find x, y, z coordinates in PLY file")
            
            print(f"Found vertex properties: {[name for name, _ in vertex_properties]}")
            print(f"X index: {x_idx}, Y index: {y_idx}, Z index: {z_idx}")
            
            # Read vertex data
            if is_binary:
                # Binary format
                dtype_list = []
                for name, prop_type in vertex_properties:
                    if prop_type == 'float' or prop_type == 'float32':
                        dtype_list.append((name, f'{endian}f4'))
                    elif prop_type == 'double' or prop_type == 'float64':
                        dtype_list.append((name, f'{endian}f8'))
                    elif prop_type == 'uchar' or prop_type == 'uint8':
                        dtype_list.append((name, 'u1'))
                    elif prop_type == 'int' or prop_type == 'int32':
                        dtype_list.append((name, f'{endian}i4'))
                    else:
                        dtype_list.append((name, f'{endian}f4'))  # Default to float
                
                vertex_dtype = np.dtype(dtype_list)
                vertices = np.frombuffer(f.read(vertex_dtype.itemsize * vertex_count), dtype=vertex_dtype)
                
            else:
                # ASCII format
                vertices = []
                for i in range(vertex_count):
                    line = f.readline().decode('ascii').strip()
                    if line:
                        values = line.split()
                        # Convert to appropriate types
                        vertex_data = []
                        for j, (name, prop_type) in enumerate(vertex_properties):
                            if j < len(values):
                                if 'float' in prop_type or 'double' in prop_type:
                                    vertex_data.append(float(values[j]))
                                elif 'int' in prop_type:
                                    vertex_data.append(int(values[j]))
                                elif 'uchar' in prop_type:
                                    vertex_data.append(int(values[j]))
                                else:
                                    vertex_data.append(float(values[j]))
                            else:
                                vertex_data.append(0.0)
                        vertices.append(tuple(vertex_data))
                
                # Convert to structured array
                dtype_list = [(name, 'f4') for name, _ in vertex_properties]
                vertices = np.array(vertices, dtype=dtype_list)
    
    except Exception as e:
        print(f"Error reading PLY file: {e}")
        return None, None, None, None
    
    return vertices, header_lines, vertex_properties, is_binary


def filter_positive_z(vertices, vertex_properties):
    """
    Filter vertices to keep only those with positive z coordinates
    """
    # Find z coordinate
    z_field = None
    for name, _ in vertex_properties:
        if name == 'z':
            z_field = name
            break
    
    if z_field is None:
        raise ValueError("No z coordinate found in vertex data")
    
    # Filter vertices where z >= 0
    positive_z_mask = vertices[z_field] >= 0
    filtered_vertices = vertices[positive_z_mask]
    
    return filtered_vertices


def write_ply_file(filepath, vertices, vertex_properties, original_header, is_binary=False):
    """
    Write filtered vertices to a new PLY file
    Always writes in ASCII format for better compatibility
    """
    try:
        with open(filepath, 'w', encoding='ascii') as f:
            # Write header
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            
            # Write comments from original header
            for line in original_header:
                if line.startswith('comment'):
                    f.write(f'{line}\n')
            
            # Add our own comment
            f.write('comment Filtered to remove negative Z coordinates\n')
            
            # Write vertex element and properties
            f.write(f'element vertex {len(vertices)}\n')
            
            for name, prop_type in vertex_properties:
                # Ensure property types are compatible
                if prop_type in ['float', 'float32', 'double', 'float64']:
                    f.write(f'property float {name}\n')
                elif prop_type in ['uchar', 'uint8']:
                    f.write(f'property uchar {name}\n')
                elif prop_type in ['int', 'int32']:
                    f.write(f'property int {name}\n')
                else:
                    f.write(f'property float {name}\n')  # Default to float
            
            f.write('end_header\n')
            
            # Write vertex data in ASCII format
            for vertex in vertices:
                # Format each value properly
                formatted_values = []
                for i, (name, prop_type) in enumerate(vertex_properties):
                    val = vertex[i] if i < len(vertex) else vertex[name]
                    
                    if prop_type in ['uchar', 'uint8', 'int', 'int32']:
                        # Integer types
                        formatted_values.append(f'{int(val)}')
                    else:
                        # Float types - limit precision to avoid huge files
                        formatted_values.append(f'{float(val):.6f}')
                
                vertex_line = ' '.join(formatted_values)
                f.write(f'{vertex_line}\n')
                    
    except Exception as e:
        print(f"Error writing PLY file: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def filter_ply_positive_z(input_filepath, output_filepath=None):
    """
    Main function to filter PLY file and remove points with negative z coordinates
    """
    if not os.path.exists(input_filepath):
        print(f"Error: Input file '{input_filepath}' not found!")
        return False
    
    if output_filepath is None:
        # Generate output filename
        base, ext = os.path.splitext(input_filepath)
        output_filepath = f"{base}_filtered{ext}"
    
    print(f"Reading PLY file: {input_filepath}")
    
    # Read the original PLY file
    vertices, header_lines, vertex_properties, is_binary = read_ply_file(input_filepath)
    
    if vertices is None:
        print("Failed to read PLY file")
        return False
    
    original_count = len(vertices)
    print(f"Original point count: {original_count}")
    
    # Filter out negative z coordinates
    print("Filtering points with negative z coordinates...")
    filtered_vertices = filter_positive_z(vertices, vertex_properties)
    
    filtered_count = len(filtered_vertices)
    removed_count = original_count - filtered_count
    
    print(f"Filtered point count: {filtered_count}")
    print(f"Removed points: {removed_count} ({removed_count/original_count*100:.1f}%)")
    
    if filtered_count == 0:
        print("Warning: No points remain after filtering!")
        return False
    
    # Write the filtered PLY file
    print(f"Writing filtered PLY file: {output_filepath}")
    
    success = write_ply_file(output_filepath, filtered_vertices, vertex_properties, header_lines, is_binary=False)
    
    if success:
        print(f"Successfully created filtered point cloud: {output_filepath}")
        return True
    else:
        print("Failed to write filtered PLY file")
        return False


def main():
    """
    Main function with example usage
    """
    print("PLY Point Cloud Z-Filter")
    print("=" * 30)
    
    # Configuration - change these paths as needed
    input_file = "D:/Data/L7/Pointcloud/pcc.ply"
    output_file = "D:/Data/L7/Pointcloud/filtered_pointcloud.ply"
    
    # Check if command line arguments provided
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        print(f"Using command line input: {input_file}")
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        print(f"Using command line output: {output_file}")
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Debug file path
    print(f"\nDebugging file path:")
    print(f"Raw path: {repr(input_file)}")
    print(f"Absolute path: {os.path.abspath(input_file)}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Detect if this is a Windows path on Linux/Mac
    if ':' in input_file and not os.path.exists(input_file):
        print(f"\n⚠️  WARNING: You appear to be using a Windows-style path on a Unix system!")
        print(f"Windows path: {input_file}")
        print(f"This won't work on Linux/Mac systems.")
        print(f"\nTo find your file, try these commands in terminal:")
        print(f"1. Find PLY files: find ~ -name '*.ply' -type f 2>/dev/null")
        print(f"2. Find by name: find ~ -name 'pcc.ply' -type f 2>/dev/null")
        print(f"3. Check current directory: ls -la *.ply")
        print(f"4. Check if mounted: ls /mnt/ or ls /media/")
        
        # Try to find PLY files in common locations
        common_paths = [
            os.path.expanduser("~/"),
            os.getcwd(),
            "/mnt/",
            "/media/",
            os.path.expanduser("~/Downloads/"),
            os.path.expanduser("~/Documents/")
        ]
        
        print(f"\nSearching for PLY files in common locations...")
        found_files = []
        
        for path in common_paths:
            if os.path.exists(path):
                try:
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            if file.lower().endswith('.ply'):
                                full_path = os.path.join(root, file)
                                found_files.append(full_path)
                                if len(found_files) >= 10:  # Limit to first 10 files
                                    break
                        if len(found_files) >= 10:
                            break
                except PermissionError:
                    continue
        
        if found_files:
            print(f"Found PLY files:")
            for i, filepath in enumerate(found_files[:10], 1):
                print(f"  {i}. {filepath}")
            
            if 'pcc.ply' in str(found_files):
                matching = [f for f in found_files if 'pcc.ply' in f]
                if matching:
                    print(f"\n✅ Found potential match(es):")
                    for match in matching:
                        print(f"  {match}")
                    print(f"\nTry updating your script with one of these paths.")
        else:
            print(f"No PLY files found in common locations.")
        
        return
    
    # Regular file existence checks
    print(f"Directory exists: {os.path.exists(os.path.dirname(input_file))}")
    
    # List files in directory to help debug
    input_dir = os.path.dirname(input_file)
    if os.path.exists(input_dir):
        print(f"\nFiles in directory '{input_dir}':")
        try:
            files = os.listdir(input_dir)
            for f in files:
                if f.lower().endswith('.ply'):
                    print(f"  {f}")
        except PermissionError:
            print("  Permission denied to list directory")
    else:
        print(f"Directory does not exist: {input_dir}")
    
    # Check file existence with different methods
    file_exists_os = os.path.exists(input_file)
    file_exists_isfile = os.path.isfile(input_file)
    
    print(f"\nFile existence checks:")
    print(f"os.path.exists(): {file_exists_os}")
    print(f"os.path.isfile(): {file_exists_isfile}")
    
    # Try to open file to check permissions
    try:
        with open(input_file, 'rb') as f:
            first_bytes = f.read(10)
            print(f"File can be opened, first 10 bytes: {first_bytes}")
            file_accessible = True
    except FileNotFoundError:
        print("File not found when trying to open")
        file_accessible = False
    except PermissionError:
        print("Permission denied when trying to open file")
        file_accessible = False
    except Exception as e:
        print(f"Other error when trying to open file: {e}")
        file_accessible = False
    
    if not file_accessible:
        print(f"\nCannot access the input file.")
        return
    
    # Process the file
    success = filter_ply_positive_z(input_file, output_file)
    
    if success:
        print("\nFiltering completed successfully!")
        print("The filtered point cloud contains only points with z >= 0")
    else:
        print("\nFiltering failed. Please check error messages.")


if __name__ == "__main__":
    main()