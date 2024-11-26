import open3d as o3d
import numpy as np

# Provided functions
def drawSphere(center, radius, color=[0.0, 0.0, 0.0]):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    transform_mat = np.eye(4)
    transform_mat[0:3, -1] = center
    mesh_sphere.transform(transform_mat)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere

def drawCone(bottom_center, top_position, color=[0.6, 0.6, 0.9]):
    cone = o3d.geometry.TriangleMesh.create_cone(radius=0.007, height=np.linalg.norm(top_position - bottom_center) + 1e-6)
    line1 = np.array([0.0, 0.0, 1.0])
    line2 = (top_position - bottom_center) / (np.linalg.norm(top_position - bottom_center) + 1e-6)
    v = np.cross(line1, line2)
    c = np.dot(line1, line2) + 1e-8
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + k + np.matmul(k, k) * (1 / (1 + c))
    if np.abs(c + 1.0) < 1e-4:  # the above formula doesn't apply when cos(∠(𝑎,𝑏))=−1
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    T = bottom_center + 5e-3 * line2
    cone.transform(np.concatenate((np.concatenate((R, T[:, np.newaxis]), axis=1), np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0))
    cone.paint_uniform_color(color)
    return cone

# Sample bone dictionary

ske_infos = np.load('dataset/simplified_meshes/dinasour/skeleton_all_frames.npy',allow_pickle=True).item()
bones = ske_infos['joints']

ske_infos1 = np.load('/data/planetzoo/arctic_wolf_juvenile/skeleton/skeleton_all_frames.npy',allow_pickle=True).item()

bones1 = ske_infos1['frame_000001']

# Visualization components
objects = []

# Draw joints (spheres) and bones (cones)
for bone_name, bone_data in bones.items():
    head = np.array(bone_data['head'])
    tail = np.array(bone_data['tail'])

    # Add spheres for the head and tail
    objects.append(drawSphere(head, radius=0.01, color=[0.0, 1.0, 0.0]))  # Green for joints
    objects.append(drawSphere(tail, radius=0.01, color=[0.0, 1.0, 0.0]))

    # Add cone for the bone
    objects.append(drawCone(head, tail, color=[1.0, 0.0, 0.0]))  # Red for bones


# for bone_name, bone_data in bones1.items():
#     head = np.array(bone_data['head'])
#     tail = np.array(bone_data['tail'])

#     head[[1]] *= -1
#     tail[[1]] *= -1
#     head, tail = head[[0, 2, 1]], tail[[0, 2, 1]]

#     # Add spheres for the head and tail
#     objects.append(drawSphere(head, radius=0.01, color=[0.0, 1.0, 0.0]))  # Green for joints
#     objects.append(drawSphere(tail, radius=0.01, color=[0.0, 1.0, 0.0]))

#     # Add cone for the bone
#     objects.append(drawCone(head, tail, color=[1.0, 0.0, 0.0]))  # Red for bones

mesh = o3d.io.read_triangle_mesh('/home/fuyang/code/CMPT766/dataset/simplified_meshes/dinasour/remesh.obj')
# mesh.compute_vertex_normals()
# objects.append(mesh)
wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
wireframe.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]] * len(wireframe.lines))  # Gray color

objects.append(wireframe)


# Visualize
o3d.visualization.draw_geometries(objects)
