import numpy as np
from collections import defaultdict
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
            joints[joint_name] = {'head': (float(x), float(y), float(z)),'tail': (float(x), float(y), float(z)), 'children': []}
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

# Build the tree structure
for joint_name, joint_info in joints.items():
    if joint_name != root:
        for parent_name in joints:
            if joint_name.startswith(parent_name) and joint_name != parent_name:
                joints[parent_name]['children'].append(joint_name)
                break

# Assign indices to joints
joint_index_map = {name: idx for idx, name in enumerate(joints.keys())}

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




(tree_structure, weight_matrix_output)