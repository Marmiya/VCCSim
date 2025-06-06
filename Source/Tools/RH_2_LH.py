def convert_ply_right_to_left(input_path, output_path, axis='z'):
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    header = []
    vertex_lines = []
    face_lines = []
    num_vertices = 0
    num_faces = 0
    header_ended = False
    vertex_counted = 0
    face_counted = 0

    # Parse header and find counts
    for i, line in enumerate(lines):
        header.append(line)
        if line.startswith('element vertex'):
            num_vertices = int(line.split()[-1])
        if line.startswith('element face'):
            num_faces = int(line.split()[-1])
        if line.strip() == 'end_header':
            header_ended = True
            vertex_start = i + 1
            break

    # Get vertices and faces
    for i in range(vertex_start, vertex_start + num_vertices):
        vertex_lines.append(lines[i])
    for i in range(vertex_start + num_vertices, vertex_start + num_vertices + num_faces):
        face_lines.append(lines[i])

    # Flip the axis (default: Z)
    flipped_vertices = []
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    for v in vertex_lines:
        parts = v.strip().split()
        parts[axis_idx] = str(-float(parts[axis_idx]))
        flipped_vertices.append(' '.join(parts) + '\n')

    # Reverse face winding
    flipped_faces = []
    for f in face_lines:
        parts = f.strip().split()
        n = int(parts[0])
        indices = parts[1:n+1]
        rest = parts[n+1:]
        indices.reverse()
        flipped_faces.append(' '.join([str(n)] + indices + rest) + '\n')

    # Write output
    with open(output_path, 'w') as outfile:
        for line in header:
            outfile.write(line)
        for v in flipped_vertices:
            outfile.write(v)
        for f in flipped_faces:
            outfile.write(f)

    print(f"Converted {input_path} to left-handed coordinate system in {output_path}")


convert_ply_right_to_left("C:/UEProjects/VCCSimDev/Saved/TestScene.ply", "C:/UEProjects/VCCSimDev/Saved/output_left.ply", axis='z')
