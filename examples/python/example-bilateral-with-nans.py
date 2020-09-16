"""Demonsrates Bilateral Normal Filtering using CPU and GPU Acceleration
"""
from os import path
import numpy as np
import logging
import open3d as o3d
import organizedpointfilters as opf
from organizedpointfilters import Matrix3f, Matrix3fRef

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PPB")
logger.setLevel(logging.INFO)

from .utility.helper import (load_pcd_file, laplacian_opc, bilateral_opc, bilateral_opc_cuda,
                             create_mesh_from_organized_point_cloud_with_o3d)
from .utility.o3d_util import plot_meshes

def pick_valid_normals(opc_normals):
    mask = ~np.isnan(opc_normals).any(axis=1) # I think that we need this with open3d 0.10.0
    tri_norms = np.ascontiguousarray(opc_normals[mask,:])
    return tri_norms

def main():
    THIS_DIR = path.dirname(path.realpath(__file__))
    PCD_DIR = path.join(THIS_DIR, '..', '..', 'fixtures', 'pcd')
    # mesh_file = path.join(PCD_DIR, 'pc_01.pcd')
    mesh_file = path.join(PCD_DIR, 'unreal_lidar.npy')
    laplacian_loops = 1

    if '.npy' in mesh_file:
        pc_image = np.load(mesh_file).astype('f8')
        print(pc_image.shape)
        # pc_image = np.copy(pc_image[:25, :35, :])
        pc = np.reshape(pc_image, (pc_image.shape[0] * pc_image.shape[1], 3))
    else:
        pc, pc_image = load_pcd_file(
            mesh_file, stride=2)
    # import ipdb; ipdb.set_trace()
    # Not Smooth Mesh
    tri_mesh_noisy, tri_mesh_noisy_o3d = create_mesh_from_organized_point_cloud_with_o3d(pc_image)

    # Smooth VERTICES of Mesh, must do this first
    kwargs_laplacian = dict(loops=laplacian_loops, _lambda=1.0, kernel_size=3)
    opc_smooth = laplacian_opc(pc_image, **kwargs_laplacian)
    tri_mesh_opc, tri_mesh_opc_o3d = create_mesh_from_organized_point_cloud_with_o3d(opc_smooth)

    # Bilateral Filter on NORMALS [CPU](using the already smoothed VERTICES)
    kwargs_bilateral = dict(loops=2, sigma_length=0.1, sigma_angle=0.1)
    opc_normals_smooth = bilateral_opc(opc_smooth, **kwargs_bilateral)
    print(opc_normals_smooth.shape, opc_normals_smooth.dtype)
    # opc_normals_smooth must be reshaped from an "image" to a flat NX3 array
    total_triangles = int(opc_normals_smooth.size / 3)
    opc_normals_smooth = opc_normals_smooth.reshape((total_triangles, 3))
    opc_normals_smooth = pick_valid_normals(opc_normals_smooth)

    tri_mesh_smoothed_normal_o3d = o3d.geometry.TriangleMesh(tri_mesh_opc_o3d)
    tri_mesh_smoothed_normal_o3d.triangle_normals = o3d.utility.Vector3dVector(opc_normals_smooth)

    # Bilateral Filter on NORMALS [GPU](using the already smoothed VERTICES)
    opc_normals_smooth_gpu = bilateral_opc_cuda(opc_smooth, **kwargs_bilateral)
    total_triangles = int(opc_normals_smooth_gpu.size / 3)
    opc_normals_smooth_gpu = opc_normals_smooth_gpu.reshape((total_triangles, 3))
    opc_normals_smooth_gpu = pick_valid_normals(opc_normals_smooth_gpu)

    tri_mesh_smoothed_normal_o3d_gpu = o3d.geometry.TriangleMesh(tri_mesh_opc_o3d)
    tri_mesh_smoothed_normal_o3d_gpu.triangle_normals = o3d.utility.Vector3dVector(opc_normals_smooth_gpu)

    # print("Meshes from left to right - Input Noisy Mesh, Smoothed with CPU Laplacian, Smoothed with CPU Bilateral, Smoothed with GPU Bilateral")
    plot_meshes(tri_mesh_noisy_o3d, tri_mesh_opc_o3d, tri_mesh_smoothed_normal_o3d, tri_mesh_smoothed_normal_o3d_gpu)


if __name__ == "__main__":
    main()
