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

from .utility.helper import load_pcd_file, DEFAULT_PPB_FILE, load_pcd_and_meshes, laplacian_opc, create_mesh_from_organized_point_cloud_with_o3d, laplacian_opc_cuda, laplacian_then_bilateral_opc
from .utility.o3d_util import create_open_3d_pcd, plot_meshes, get_arrow, create_open_3d_mesh, flatten


def main():
    kwargs = dict(loops=0, stride=2, _lambda=1.0, kernel_size=3)
    pc, pc_image, pcd_o3d, tri_mesh, tri_mesh_o3d = load_pcd_and_meshes(
        '/home/jeremy/Documents/UMICH/Research/polylidar-plane-benchmark/data/synpeb/train/var4/pc_02.pcd', **kwargs)

    tri_mesh_o3d: o3d.geometry.TriangleMesh
    tri_mesh_o3d.compute_triangle_normals()

    # opc_smooth, pcd_smooth = laplacian_opc(pc_image, **kwargs, max_dist=0.25)
    # tri_mesh_opc, tri_mesh_opc_o3d = create_mesh_from_organized_point_cloud_with_o3d(opc_smooth)
    tri_mesh_o3d_copy = copy.deepcopy(tri_mesh_o3d)

    main._lambda = 1.0
    main.lap_loops = 0
    main.kernel_size = 3
    main.bl_loops = 0
    main.bl_sigma_angle = 0.261

    def smooth_mesh(vis: o3d.visualization.Visualizer):
        print("Current Params - Laplacian Loops: {:d}; Lambda: {:.2f}; Kernel Size: {:d}; Bilateral Loops: {:d}; Sigma Angle {:.2f}".format(
            main.lap_loops, main._lambda, main.kernel_size, main.bl_loops, main.bl_sigma_angle))
        if main.lap_loops == 0:
            tri_mesh_o3d_temp = tri_mesh_o3d
        else:
            # opc_smooth, pcd_smooth = laplacian_opc_cuda(pc_image, loops=main.lap_loops, _lambda=main._lambda, max_dist=0.25)
            # opc_smooth, pcd_smooth = laplacian_opc(pc_image, loops=main.lap_loops, _lambda=main._lambda, max_dist=0.25, kernel_size=main.kernel_size)
            opc_smooth, opc_normals_smooth, pcd_smooth = laplacian_then_bilateral_opc(
                pc_image, loops_laplacian=main.lap_loops, _lambda=main._lambda, max_dist=0.25, kernel_size=main.kernel_size,
                loops_bilateral=main.bl_loops, sigma_angle=main.bl_sigma_angle)
            tri_mesh_opc, tri_mesh_o3d_temp = create_mesh_from_organized_point_cloud_with_o3d(opc_smooth)

            # tri_mesh_o3d_temp = tri_mesh_o3d.filter_smooth_laplacian(main.lap_loops, main._lambda)
        tri_mesh_o3d_copy.vertices = tri_mesh_o3d_temp.vertices
        try:
            tri_mesh_o3d_copy.triangle_normals = o3d.utility.Vector3dVector(opc_normals_smooth)
        except Exception as e:
            tri_mesh_o3d_copy.compute_triangle_normals()
            print(e)

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

    def increase_sigma_angle(vis):
        main.bl_sigma_angle += .05
        main.bl_sigma_angle = min(main.bl_sigma_angle, 0.5)
        smooth_mesh(vis)
        return False

    def decrease_sigma_angle(vis):
        main.bl_sigma_angle -= .05
        main.bl_sigma_angle = max(main.bl_sigma_angle, 0.1)
        smooth_mesh(vis)
        return False

    def increase_loops(vis):
        main.lap_loops += 1
        main.lap_loops = min(main.lap_loops, 20)
        smooth_mesh(vis)
        return False

    def decrease_loops(vis):
        main.lap_loops -= 1
        main.lap_loops = max(main.lap_loops, 0)
        smooth_mesh(vis)
        return False

    def increase_loops_bl(vis):
        main.bl_loops += 1
        main.bl_loops = min(main.bl_loops, 20)
        smooth_mesh(vis)
        return False

    def decrease_loops_bl(vis):
        main.bl_loops -= 1
        main.bl_loops = max(main.bl_loops, 0)
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

    key_to_callback[ord("R")] = increase_loops_bl
    key_to_callback[ord("F")] = decrease_loops_bl

    key_to_callback[ord("U")] = increase_lambda
    key_to_callback[ord("J")] = decrease_lambda

    key_to_callback[ord("Y")] = increase_sigma_angle
    key_to_callback[ord("H")] = decrease_sigma_angle

    key_to_callback[ord("T")] = increase_kernel
    key_to_callback[ord("G")] = decrease_kernel

    # main.vis = o3d.visualization.Visualizer()
    # vis = main.vis
    # vis.create_window()
    # vis.add_geometry(tri_mesh_o3d_copy)

    o3d.visualization.draw_geometries_with_key_callbacks([tri_mesh_o3d_copy], key_to_callback)


if __name__ == "__main__":
    main()
