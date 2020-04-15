import numpy as np
import logging
import open3d as o3d
import organizedpointfilters as opf
from organizedpointfilters import Matrix3f, Matrix3fRef

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PPB")
logger.setLevel(logging.INFO)

from .utility.helper import (load_pcd_file, DEFAULT_PPB_FILE, load_pcd_and_meshes, laplacian_opc, bilateral_opc, bilateral_opc_cuda,
                             create_mesh_from_organized_point_cloud_with_o3d, laplacian_opc_cuda, compute_normals_opc,
                             compute_normals_and_centroids_opc)
from .utility.o3d_util import create_open_3d_pcd, plot_meshes, get_arrow, create_open_3d_mesh, flatten


def main():
    kwargs = dict(loops=5, stride=2, _lambda=1.0)
    pc, pc_image, pcd_o3d, tri_mesh, tri_mesh_o3d = load_pcd_and_meshes(
        '/home/jeremy/Documents/UMICH/Research/polylidar-plane-benchmark/data/synpeb/train/var4/pc_01.pcd', **kwargs)

    del kwargs['stride']

    # Not Smooth Mesh
    tri_mesh_noisy, tri_mesh_noisy_o3d = create_mesh_from_organized_point_cloud_with_o3d(
        np.ascontiguousarray(pc[:, :3]))

    # Smooth VERTICES of Mesh
    opc_smooth, pcd_smooth = laplacian_opc(pc_image, **kwargs, max_dist=0.25, kernel_size=3)
    tri_mesh_opc, tri_mesh_opc_o3d = create_mesh_from_organized_point_cloud_with_o3d(opc_smooth)


    # Bilateral Filter on NORMALS [CPU](using the already smoothed VERTICES)
    opc_normals_smooth = bilateral_opc(opc_smooth, loops=10)

    total_triangles = int(opc_normals_smooth.size / 3)
    print("Total Triangles: ", total_triangles)
    opc_normals_smooth = opc_normals_smooth.reshape((total_triangles, 3))

    tri_mesh_smoothed_normal_o3d = o3d.geometry.TriangleMesh(tri_mesh_opc_o3d)
    tri_mesh_smoothed_normal_o3d.triangle_normals = o3d.utility.Vector3dVector(opc_normals_smooth)

    # Bilateral Filter on NORMALS [CPU](using the already smoothed VERTICES)
    opc_normals_smooth_gpu = bilateral_opc_cuda(opc_smooth, loops=10)

    total_triangles = int(opc_normals_smooth_gpu.size / 3)
    print("Total Triangles: ", total_triangles)
    opc_normals_smooth_gpu = opc_normals_smooth_gpu.reshape((total_triangles, 3))
    
    tri_mesh_smoothed_normal_o3d_gpu = o3d.geometry.TriangleMesh(tri_mesh_opc_o3d)
    tri_mesh_smoothed_normal_o3d_gpu.triangle_normals = o3d.utility.Vector3dVector(opc_normals_smooth_gpu)
    

    plot_meshes(tri_mesh_noisy_o3d, tri_mesh_opc_o3d, tri_mesh_smoothed_normal_o3d, tri_mesh_smoothed_normal_o3d_gpu)


if __name__ == "__main__":
    main()
