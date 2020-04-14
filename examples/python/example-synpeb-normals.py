import numpy as np
import logging
import open3d as o3d
import organizedpointfilters as opf
from organizedpointfilters import Matrix3f, Matrix3fRef

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PPB")
logger.setLevel(logging.INFO)

from .utility.helper import (load_pcd_file, DEFAULT_PPB_FILE, load_pcd_and_meshes, laplacian_opc, smooth_normals_opc,
                             create_mesh_from_organized_point_cloud_with_o3d, laplacian_opc_cuda, compute_normals_opc)
from .utility.o3d_util import create_open_3d_pcd, plot_meshes, get_arrow, create_open_3d_mesh, flatten


def main():
    kwargs = dict(loops=5, stride=2, _lambda=1.0)
    pc, pc_image, pcd_o3d, tri_mesh, tri_mesh_o3d = load_pcd_and_meshes(
        '/home/jeremy/Documents/UMICH/Research/polylidar-plane-benchmark/data/synpeb/train/var2/pc_01.pcd', **kwargs)

    del kwargs['stride']

    # Not Smooth Mesh
    tri_mesh_noisy, tri_mesh_noisy_o3d = create_mesh_from_organized_point_cloud_with_o3d(
        np.ascontiguousarray(pc[:, :3]))

    # Smooth Mesh
    opc_smooth, pcd_smooth = laplacian_opc(pc_image, **kwargs, max_dist=0.25, kernel_size=3)
    tri_mesh_opc, tri_mesh_opc_o3d = create_mesh_from_organized_point_cloud_with_o3d(opc_smooth)

    # normals_opc = compute_normals_opc(opc_smooth)
    # print("Before: ")
    # print(normals_opc[1, 1, :, :])
    opc_normals_smooth = smooth_normals_opc(opc_smooth, loops=5)
    # print("After: ")
    # print(opc_normals_smooth [1, 1, :, :])
    # import ipdb; ipdb.set_trace()

    total_triangles = int(opc_normals_smooth.size / 3)
    print("Total Triangles: ", total_triangles)
    opc_normals_smooth = opc_normals_smooth.reshape((total_triangles, 3))
    # print(opc_normals_smooth.shape)
    # print(opc_normals_smooth)
    
    normals_o3d = np.asarray(tri_mesh_opc_o3d.triangle_normals)
    # print(normals_o3d.shape)
    # print(normals_o3d)

    tri_mesh_smoothed_normal_o3d = o3d.geometry.TriangleMesh(tri_mesh_opc_o3d)
    tri_mesh_smoothed_normal_o3d.triangle_normals = o3d.utility.Vector3dVector(opc_normals_smooth)
    
    # I don't think this is necessary
    # vertex_normals = np.asarray(tri_mesh_smoothed_normal_o3d.vertex_normals)
    # vertex_normals_zero = np.zeros_like(vertex_normals)
    # tri_mesh_smoothed_normal_o3d.vertex_normals = o3d.utility.Vector3dVector(vertex_normals_zero)

    # print(opc_normals_smooth.shape)
    # print(opc_normals_smooth)


    # vertices_o3d = np.asarray(tri_mesh_opc_o3d.vertices)

    # print(opc_smooth[:2, :2, :])
    # print(normals_opc[0, 0, :, :])
    # # print("")

    # # print(vertices_o3d[:4, :])
    # # print(normals_o3d[:3,:])
    # # print("")
    # # import ipdb; ipdb.set_trace()

    # total_triangles = int(normals_opc.size / 3)
    # normals_opc = normals_opc.reshape((total_triangles, 3))
    # tri_mesh_opc_o3d.triangle_normals = o3d.utility.Vector3dVector(normals_opc)

    # print(normals_o3d.shape)
    # print(normals_opc.shape)

    # print(normals_o3d[:10,:])

    plot_meshes(tri_mesh_noisy_o3d, tri_mesh_opc_o3d, tri_mesh_smoothed_normal_o3d)


if __name__ == "__main__":
    main()
