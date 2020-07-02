"""
This is scratch. You can ignore this file
"""
from os import path
import open3d as o3d
import numpy as np
import sys
import time
# from examples.python.utility.helper import (load_pcd_file, load_pcd_and_meshes, create_mesh_from_organized_point_cloud, create_open_3d_mesh)
from examples.python.utility.o3d_util import plot_meshes, create_open_3d_mesh_from_tri_mesh
from examples.python.utility.helper import (load_pcd_file, load_pcd_and_meshes, create_mesh_from_organized_point_cloud)

THIS_DIR = path.dirname(path.realpath(__file__))
PCD_DIR = path.join(THIS_DIR, '..', '..', 'fixtures', 'pcd')


a =  np.load(path.join(PCD_DIR, 'rgbd_opc.npy'))
tri_mesh, tri_map = create_mesh_from_organized_point_cloud(a, stride=1)
tri_mesh_o3d = create_open_3d_mesh_from_tri_mesh(tri_mesh)

print(a.shape)
print(np.asarray(tri_mesh_o3d.triangles).shape)

# pc_raw, pc_image, pcd_o3d, tri_mesh, tri_mesh_o3d = load_pcd_and_meshes(path.join(PCD_DIR, 'pc_01.pcd'), stride=1)

tri_mesh_o3d = o3d.io.read_triangle_mesh('test.ply')
tri_mesh_o3d = tri_mesh_o3d.compute_adjacency_list()
tri_mesh_o3d.compute_triangle_normals()
# o3d.visualization.draw_geometries([tri_mesh_o3d])
print(tri_mesh_o3d.has_adjacency_list())
scope = o3d.geometry.FilterScope.All 
t0 = time.perf_counter()
smoothed = tri_mesh_o3d.filter_smooth_laplacian(200, 1.0, scope)
t1 = time.perf_counter()

smoothed.compute_triangle_normals()
smoothed.paint_uniform_color([1, 0, 0])
plot_meshes(tri_mesh_o3d, smoothed)

print((t1 - t0) * 1000)
# not as much mesh shrinkage as I thought there would be
# import ipdb; ipdb.set_trace()