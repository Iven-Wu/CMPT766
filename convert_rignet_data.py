import numpy as np
from collections import defaultdict
import open3d as o3d
import pdb

# Parse the file content (simulated for now)
file_path = 'input_meshes_new/17607.txt'

# Initialize data structures
joints = {}
root = None
skin_data = defaultdict(list)
joint_index_map = {}
vertex_count = 0

# Process the file
with open(file_path, 'r') as f:
    for line in f:
        if line.startswith("joints"):
            _, joint_name, x, y, z = line.split()
            # joints[joint_name] = {'position': (float(x), float(y), float(z)), 'children': []}
            joints[joint_name] = {'head': np.array([float(x), float(y), float(z)]),'tail': np.array([float(x), float(y), float(z)]), 'children': []}
        elif line.startswith("root"):
            _, root_name = line.split()
            root = root_name
        elif line.startswith("skin"):
            parts = line.split()
            vertex_id = int(parts[1])
            vertex_count = max(vertex_count, vertex_id + 1)
            for i in range(2, len(parts), 2):
                joint_name = parts[i]
                weight = float(parts[i + 1])
                skin_data[vertex_id].append((joint_name, weight))
        elif line.startswith("hier"):
            parts = line.split()
            parent_name = parts[1]
            child_name = parts[2]
            joints[parent_name]['children'].append(child_name)

# Build the tree structure
for joint_name, joint_info in joints.items():
    if joint_name != root:
        for parent_name in joints:
            if joint_name.startswith(parent_name) and joint_name != parent_name:
                joints[parent_name]['children'].append(joint_name)
                break

# Assign indices to joints
joint_index_map = {name: idx for idx, name in enumerate(joints.keys())}

# pdb.set_trace()

# Create the weight matrix
weight_matrix = np.zeros((len(joints), vertex_count))
for vertex_id, joint_weights in skin_data.items():
    for joint_name, weight in joint_weights:
        joint_idx = joint_index_map[joint_name]
        weight_matrix[joint_idx, vertex_id] = weight

# Encapsulate in a class or dictionary
tree_structure = {'root': root, 'joints': joints, 'joint_index_map': joint_index_map}
weight_matrix_output = weight_matrix.T

np.save('W1.npy',weight_matrix_output)
np.save('skeleton_all_frames.npy',tree_structure)



mesh = o3d.io.read_triangle_mesh("/home/fuyang/code/CMPT766/dataset/simplified_meshes/dinasour/remesh.obj")

# Load the skinning weight matrix (vertices x bones)
# Assume weights is a NumPy array of shape (num_vertices, num_bones)
weights = weight_matrix_output  # Shape: (num_vertices, num_bones)

# Function to visualize weights for a specific bone
def visualize_weights_for_bone(mesh, weights, bone_index):
    num_vertices = np.asarray(mesh.vertices).shape[0]

    # Ensure weights array matches the number of vertices
    assert weights.shape[0] == num_vertices, "Mismatch between mesh vertices and weights array!"

    # Extract weights for the selected bone
    bone_weights = weights[:, bone_index]

    # Normalize weights to use as colors (range: 0 to 1)
    max_weight = bone_weights.max()
    bone_weights_normalized = bone_weights / max_weight if max_weight > 0 else bone_weights

    # Create color map for vertices (e.g., red for high weight, blue for low weight)
    vertex_colors = np.zeros((num_vertices, 3))
    vertex_colors[:, 0] = bone_weights_normalized  # Red channel

    # Apply vertex colors to the mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    return mesh

# Create wireframe representation of the mesh
def create_wireframe_from_mesh(mesh):
    line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    line_set.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]] * len(line_set.lines))  # Gray lines
    return line_set

# def apply_transparency(mesh, alpha=0.3):
#     """
#     Modify the mesh to appear transparent by adjusting the rendering material.
#     Open3D doesn't directly support per-vertex transparency, but we can use
#     shading materials to mimic the effect.
#     """
#     mesh.compute_vertex_normals()  # Ensure normals are calculated
#     material = o3d.visualization.rendering.MaterialRecord()
#     material.shader = "defaultLitTransparency"
#     material.base_color = [1.0, 1.0, 1.0, alpha]  # RGBA, with alpha for transparency
#     return material

# Visualize
bone_index = 4  # Select a bone index
print(list(tree_structure['joint_index_map'].keys())[bone_index])
mesh_with_weights = visualize_weights_for_bone(mesh, weights, bone_index)
wireframe = create_wireframe_from_mesh(mesh)

# Use Open3D Visualizer
# vis = o3d.visualization.Visualizer()
# vis.create_window()

# # Add both the mesh with weights and the wireframe
# vis.add_geometry(mesh_with_weights)
# vis.add_geometry(wireframe)

o3d.visualization.draw_geometries([mesh_with_weights,wireframe])





(tree_structure, weight_matrix_output)