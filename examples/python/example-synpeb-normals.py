import numpy as np
import logging
import open3d as o3d
import organizedpointfilters as opf
from organizedpointfilters import Matrix3f, Matrix3fRef

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PPB")
logger.setLevel(logging.INFO)

from .utility.helper import (load_pcd_file, DEFAULT_PPB_FILE, load_pcd_and_meshes, laplacian_opc,
                             create_mesh_from_organized_point_cloud_with_o3d, laplacian_opc_cuda, compute_normals_opc)
from .utility.o3d_util import create_open_3d_pcd, plot_meshes, get_arrow, create_open_3d_mesh, flatten


def main():
    kwargs = dict(loops=10, stride=2, _lambda=1.0, kernel_size=3)
    pc, pc_image, pcd_o3d, tri_mesh, tri_mesh_o3d = load_pcd_and_meshes(
        '/home/jeremy/Documents/UMICH/Research/polylidar-plane-benchmark/data/synpeb/train/var4/pc_01.pcd', **kwargs)

    del kwargs['stride']

    # Not Smooth Mesh
    tri_mesh_noisy, tri_mesh_noisy_o3d = create_mesh_from_organized_point_cloud_with_o3d(
        np.ascontiguousarray(pc[:, :3]))

    # Smooth Mesh
    opc_smooth, pcd_smooth = laplacian_opc(pc_image, **kwargs, max_dist=0.25)
    tri_mesh_opc, tri_mesh_opc_o3d = create_mesh_from_organized_point_cloud_with_o3d(opc_smooth)
    normals_opc = compute_normals_opc(opc_smooth)
    normals_o3d = np.asarray(tri_mesh_opc_o3d.triangle_normals)
    vertices_o3d = np.asarray(tri_mesh_opc_o3d.vertices)

    # print(opc_smooth[:2, :2, :])
    # print(normals_opc[0, 0, :, :])
    # print("")

    # print(vertices_o3d[:4, :])
    # print(normals_o3d[:3,:])
    # print("")
    # import ipdb; ipdb.set_trace()

    total_triangles = int(normals_opc.size / 3)
    normals_opc = normals_opc.reshape((total_triangles, 3))
    tri_mesh_opc_o3d.triangle_normals = o3d.utility.Vector3dVector(normals_opc)

    print(normals_o3d.shape)
    print(normals_opc.shape)

    # print(normals_o3d[:10,:])

    plot_meshes(tri_mesh_noisy_o3d, tri_mesh_opc_o3d)


if __name__ == "__main__":
    main()
