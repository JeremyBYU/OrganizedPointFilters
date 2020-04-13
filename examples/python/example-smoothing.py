import logging
import copy

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from organizedpointfilters import Matrix3f, Matrix3fRef
import organizedpointfilters as opf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PPB")
logger.setLevel(logging.INFO)

from .utility.helper import load_pcd_file, DEFAULT_PPB_FILE, load_pcd_and_meshes, laplacian_opc, create_mesh_from_organized_point_cloud_with_o3d, laplacian_opc_cuda
from .utility.o3d_util import create_open_3d_pcd, plot_meshes, get_arrow, create_open_3d_mesh, flatten


	




def main():
    kwargs = dict(loops=0, stride=2, _lambda=1.0, kernel_size=3)
    pc, pc_image, pcd_o3d, tri_mesh, tri_mesh_o3d = load_pcd_and_meshes('/home/jeremy/Documents/UMICH/Research/polylidar-plane-benchmark/data/synpeb/train/var3/pc_02.pcd', **kwargs)

    tri_mesh_o3d:o3d.geometry.TriangleMesh
    tri_mesh_o3d.compute_triangle_normals()

    # opc_smooth, pcd_smooth = laplacian_opc(pc_image, **kwargs, max_dist=0.25)
    # tri_mesh_opc, tri_mesh_opc_o3d = create_mesh_from_organized_point_cloud_with_o3d(opc_smooth)
    tri_mesh_o3d_copy = copy.deepcopy(tri_mesh_o3d)


    main._lambda = 1.0
    main.loops = 0
    main.kernel_size = 3

    def smooth_mesh(vis:o3d.visualization.Visualizer):
        print("Current Params - Loops: {:d}; Lambda: {:.2f}; Kernel Size: {:d}".format(main.loops, main._lambda, main.kernel_size))
        if main.loops == 0:
            tri_mesh_o3d_temp = tri_mesh_o3d
        else:
            # opc_smooth, pcd_smooth = laplacian_opc_cuda(pc_image, loops=main.loops, _lambda=main._lambda, max_dist=0.25)
            opc_smooth, pcd_smooth = laplacian_opc(pc_image, loops=main.loops, _lambda=main._lambda, max_dist=0.25, kernel_size=main.kernel_size)
            tri_mesh_opc, tri_mesh_o3d_temp = create_mesh_from_organized_point_cloud_with_o3d(opc_smooth)
            # tri_mesh_o3d_temp = tri_mesh_o3d.filter_smooth_laplacian(main.loops, main._lambda)
        tri_mesh_o3d_copy.vertices = tri_mesh_o3d_temp.vertices
        tri_mesh_o3d_copy.compute_triangle_normals()

        vis.update_geometry(tri_mesh_o3d_copy)
        # vis.update_renderer()

    def increase_lambda(vis):
        main._lambda += .1
        main._lambda = min(main._lambda, 1.0)
        smooth_mesh(vis)
        return False

    def decrease_lambda(vis):
        main._lambda -= .1
        main._lambda = max(main._lambda, 0.1)
        smooth_mesh(vis)
        return False

    def increase_loops(vis):
        main.loops += 1
        main.loops = min(main.loops, 20)
        smooth_mesh(vis)
        return False

    def decrease_loops(vis):
        main.loops -= 1
        main.loops = max(main.loops, 0)
        smooth_mesh(vis)
        return False

    def increase_kernel(vis):
        main.kernel_size += 2
        main.kernel_size = min(main.kernel_size, 5)
        smooth_mesh(vis)
        return False

    def decrease_kernel(vis):
        main.kernel_size -= 2
        main.kernel_size = max(main.kernel_size, 3)
        smooth_mesh(vis)
        return False

    key_to_callback = {}
    key_to_callback[ord("E")] = increase_loops
    key_to_callback[ord("D")] = decrease_loops

    key_to_callback[ord("R")] = increase_lambda
    key_to_callback[ord("F")] = decrease_lambda

    key_to_callback[ord("T")] = increase_kernel
    key_to_callback[ord("G")] = decrease_kernel

    # main.vis = o3d.visualization.Visualizer()
    # vis = main.vis
    # vis.create_window()
    # vis.add_geometry(tri_mesh_o3d_copy)

    o3d.visualization.draw_geometries_with_key_callbacks([tri_mesh_o3d_copy], key_to_callback)



if __name__ == "__main__":
    main()