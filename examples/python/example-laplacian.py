"""
Demonstrates Laplacian Smoothing using CPU and GPU acceleration
"""
from os import path
import numpy as np
import logging

import organizedpointfilters as opf
from organizedpointfilters import Matrix3f, Matrix3fRef

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PPB")
logger.setLevel(logging.INFO)

from .utility.helper import load_pcd_file, laplacian_opc, create_mesh_from_organized_point_cloud_with_o3d, laplacian_opc_cuda
from .utility.o3d_util import plot_meshes


def main():
    THIS_DIR = path.dirname(path.realpath(__file__))
    PCD_DIR = path.join(THIS_DIR, '..', '..', 'fixtures', 'pcd')
    mesh_file = path.join(PCD_DIR, 'pc_01.pcd')
    pc, pc_image = load_pcd_file(
        mesh_file, stride=2)

    # Not Smooth Mesh
    tri_mesh_noisy, tri_mesh_noisy_o3d = create_mesh_from_organized_point_cloud_with_o3d(
        np.ascontiguousarray(pc[:, :3]))

    kwargs = dict(loops=5, _lambda=1.0, kernel_size=3)
    opc_smooth = laplacian_opc(pc_image, **kwargs, max_dist=0.25)
    tri_mesh_opc, tri_mesh_opc_o3d = create_mesh_from_organized_point_cloud_with_o3d(opc_smooth)

    opc_smooth_gpu = laplacian_opc_cuda(pc_image, **kwargs)
    tri_mesh_opc_gpu, tri_mesh_opc_gpu_o3d = create_mesh_from_organized_point_cloud_with_o3d(opc_smooth_gpu)

    kwargs['kernel_size'] = 5
    opc_smooth_gpu_k5 = laplacian_opc_cuda(pc_image, **kwargs)
    tri_mesh_opc_gpu_k5, tri_mesh_opc_gpu_o3d_k5 = create_mesh_from_organized_point_cloud_with_o3d(opc_smooth_gpu_k5)

    print("Meshes from left to right - Input Noisy Mesh, Smoothed with CPU Laplacian, Smoothed with GPU Laplacian, Smoothed with GPU Laplacian with kernel=5")

    plot_meshes(tri_mesh_noisy_o3d, tri_mesh_opc_o3d, tri_mesh_opc_gpu_o3d, tri_mesh_opc_gpu_o3d_k5)


if __name__ == "__main__":
    main()
