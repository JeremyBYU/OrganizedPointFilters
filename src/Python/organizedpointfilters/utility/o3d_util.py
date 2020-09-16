from copy import deepcopy
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.spatial.transform import Rotation as R
from polylidar import MatrixDouble, MatrixInt, create_tri_mesh_copy


COLOR_PALETTE = list(map(colors.to_rgb, plt.rcParams['axes.prop_cycle'].by_key()['color']))

def flatten(l): return [item for sublist in l for item in sublist]

def open_3d_mesh_to_tri_mesh(mesh: o3d.geometry.TriangleMesh):
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    vertices_mat = MatrixDouble(vertices)
    triangles_mat = MatrixInt(triangles)
    triangles_mat_np = np.asarray(triangles_mat)

    # print(triangles, triangles.dtype)
    # print(triangles_mat_np, triangles_mat_np.dtype)

    tri_mesh = create_tri_mesh_copy(vertices_mat, triangles_mat)
    return tri_mesh

def get_colors(inp, colormap=plt.cm.viridis, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))

def create_open_3d_pcd(points, clusters=None, cmap=plt.cm.tab20):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if clusters is not None:
        colors = get_colors(clusters, colormap=cmap)
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd

def create_open_3d_mesh_from_tri_mesh(tri_mesh):
    """Create an Open3D Mesh given a Polylidar TriMesh"""
    triangles = np.asarray(tri_mesh.triangles)
    vertices = np.asarray(tri_mesh.vertices)
    return create_open_3d_mesh(triangles, vertices)

# def create_open_3d_mesh(triangles, points, triangle_normals=None, color=COLOR_PALETTE[0]):
#     """Create an Open3D Mesh given triangles vertices

#     Arguments:
#         triangles {ndarray} -- Triangles array
#         points {ndarray} -- Points array

#     Keyword Arguments:
#         color {list} -- RGB COlor (default: {[1, 0, 0]})

#     Returns:
#         mesh -- Open3D Mesh
#     """
#     mesh_2d = o3d.geometry.TriangleMesh()
#     if points.ndim == 1:
#         points = points.reshape((int(points.shape[0] / 3), 3))
#     if triangles.ndim == 1:
#         triangles = triangles.reshape((int(triangles.shape[0] / 3), 3))
#         # Open 3D expects triangles to be counter clockwise
#         triangles = np.ascontiguousarray(np.flip(triangles, 1))
#     mesh_2d.triangles = o3d.utility.Vector3iVector(triangles)

#     mesh_2d.vertices = o3d.utility.Vector3dVector(points)
#     if triangle_normals is None:
#         mesh_2d.compute_vertex_normals()
#         mesh_2d.compute_triangle_normals()
#     elif triangle_normals.ndim == 1:
#         triangle_normals_ = triangle_normals.reshape((int(triangle_normals.shape[0] / 3), 3))
#         # triangles = np.ascontiguousarray(np.flip(triangles, 1))
#         mesh_2d.triangle_normals = o3d.utility.Vector3dVector(triangle_normals_)
#     mesh_2d.paint_uniform_color(color)
#     return mesh_2d

def create_open_3d_mesh(triangles, points, triangle_normals=None, color=COLOR_PALETTE[0], counter_clock_wise=True):
    """Create an Open3D Mesh given triangles vertices

    Arguments:
        triangles {ndarray} -- Triangles array
        points {ndarray} -- Points array

    Keyword Arguments:
        color {list} -- RGB COlor (default: {[1, 0, 0]})

    Returns:
        mesh -- Open3D Mesh
    """
    mesh_2d = o3d.geometry.TriangleMesh()
    if points.ndim == 1:
        points = points.reshape((int(points.shape[0] / 3), 3))
    if triangles.ndim == 1:
        triangles = triangles.reshape((int(triangles.shape[0] / 3), 3))
        # Open 3D expects triangles to be counter clockwise
    if not counter_clock_wise:
        triangles = np.ascontiguousarray(np.flip(triangles, 1))
    mesh_2d.triangles = o3d.utility.Vector3iVector(triangles)
    mask = np.isnan(points).any(axis=1) # I think that we need this with open3d 0.10.0
    points_ = np.copy(points)
    points_[mask, :] = [0,0,0]
    mesh_2d.vertices = o3d.utility.Vector3dVector(points_)
    if triangle_normals is None:
        mesh_2d.compute_vertex_normals()
        mesh_2d.compute_triangle_normals()
    elif triangle_normals.ndim == 1:
        triangle_normals_ = triangle_normals.reshape((int(triangle_normals.shape[0] / 3), 3))
        mesh_2d.triangle_normals = o3d.utility.Vector3dVector(triangle_normals_)
    mesh_2d.paint_uniform_color(color)
    mesh_2d.compute_vertex_normals()
    return mesh_2d




def split_triangles(mesh):
    """
    Split the mesh in independent triangles    
    """
    triangles = np.asarray(mesh.triangles).copy()
    vertices = np.asarray(mesh.vertices).copy()

    triangles_3 = np.zeros_like(triangles)
    vertices_3 = np.zeros((len(triangles) * 3, 3), dtype=vertices.dtype)

    for index_triangle, t in enumerate(triangles):
        index_vertex = index_triangle * 3
        vertices_3[index_vertex] = vertices[t[0]]
        vertices_3[index_vertex + 1] = vertices[t[1]]
        vertices_3[index_vertex + 2] = vertices[t[2]]

        triangles_3[index_triangle] = np.arange(index_vertex, index_vertex + 3)

    mesh_return = deepcopy(mesh)
    mesh_return.triangles = o3d.utility.Vector3iVector(triangles_3)
    mesh_return.vertices = o3d.utility.Vector3dVector(vertices_3)
    mesh_return.paint_uniform_color([0.5, 0.5, 0.5])
    return mesh_return


def assign_vertex_colors(mesh, normal_colors, mask=None):
    """Assigns vertex colors by given normal colors
    NOTE: New mesh is returned

    Arguments:
        mesh {o3d:TriangleMesh} -- Mesh
        normal_colors {ndarray} -- Normals Colors

    Returns:
        o3d:TriangleMesh -- New Mesh with painted colors
    """
    split_mesh = split_triangles(mesh)
    vertex_colors = np.asarray(split_mesh.vertex_colors)
    triangles = np.asarray(split_mesh.triangles)
    if mask is not None:
        triangles = triangles[mask, :]
    for i in range(triangles.shape[0]):
        # import ipdb; ipdb.set_trace()
        color = normal_colors[i, :]
        p_idx = triangles[i, :]
        vertex_colors[p_idx] = color

    return split_mesh


def translate_meshes(mesh_list, current_translation=0.0, axis=0):
    translate_amt = None
    translate_meshes = []
    for mesh_ in mesh_list:
        mesh_ = deepcopy(mesh_)

        bbox = mesh_.get_axis_aligned_bounding_box()
        x_extent = bbox.get_extent()[axis]
        translate_amt = [0, 0, 0]
        translate_amt[axis] = current_translation + x_extent / 2.0
        translate_meshes.append(mesh_.translate(translate_amt, relative=False))
        current_translation += x_extent + 0.5
    return translate_meshes

def plot_meshes(*meshes, shift=True):
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    axis.translate([-2.0, 0, 0])
    translate_meshes = []
    current_x = 0.0
    if shift:
        for i, mesh in enumerate(meshes):
            inner_meshes = [mesh]
            if isinstance(mesh, list):
                inner_meshes = mesh
            translate_amt = None
            for mesh_ in inner_meshes:
                mesh_ = deepcopy(mesh_)
                if translate_amt is not None:
                    translate_meshes.append(mesh_.translate(translate_amt, relative=True))
                else:
                    bbox = mesh_.get_axis_aligned_bounding_box()
                    x_extent = bbox.get_extent()[0]
                    translate_amt = [current_x + x_extent / 2.0, 0, 0]
                    translate_meshes.append(mesh_.translate(translate_amt, relative=True))
                    current_x += x_extent + 0.5
    else:
        translate_meshes = meshes

    o3d.visualization.draw_geometries([axis, *translate_meshes])


def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec (): 
    """
    magnitude = np.sqrt(np.sum(vec**2))
    return(magnitude)


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle

def create_arrow(scale=1, cylinder_radius=None, **kwargs):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale*0.2
    cylinder_height = scale*0.8
    cone_radius = cylinder_radius if cylinder_radius else scale/10
    cylinder_radius = cylinder_radius if cylinder_radius else scale/20
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=cone_radius,
        cone_height=cone_height,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height)
    return(mesh_frame)

def get_arrow(origin=[0,0,0],end=None,vec=None, **kwargs):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    # print(end)
    scale = 10; beta = 0; gamma = 0
    T = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    T[:3,-1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        scale = vector_magnitude(vec)
        mesh = create_arrow(scale, **kwargs)
        axis, angle = align_vector_to_another(b=vec/ scale)
        if axis is None:
            axis_a = axis
        else:
            axis_a = axis * angle
            rotation_3x3 = mesh.get_rotation_matrix_from_axis_angle(axis_a)
    # mesh.transform(T)
    if axis is not None:
        mesh = mesh.rotate(rotation_3x3, center=False)
    mesh.translate(origin)
    return(mesh)