import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PPB")
logger.setLevel(logging.INFO)

from .utility.helper import load_pcd_file, DEFAULT_PPB_FILE, load_pcd_and_meshes, laplacian_opc, create_mesh_from_organized_point_cloud_with_o3d
import organizedpointfilters as opf
from organizedpointfilters import Matrix3f, Matrix3fRef
from .utility.o3d_util import create_open_3d_pcd, plot_meshes, get_arrow, create_open_3d_mesh, flatten


def main():
    kwargs = dict(loops=5, stride=2, _lambda=0.5, kernel_size=3)
    pc, pc_image, pcd_o3d, tri_mesh, tri_mesh_o3d = load_pcd_and_meshes('/home/jeremy/Documents/UMICH/Research/polylidar-plane-benchmark/data/synpeb/train/var4/pc_01.pcd', **kwargs)

    opc_smooth, pcd_smooth = laplacian_opc(pc_image, **kwargs)
    tri_mesh_opc, tri_mesh_opc_o3d = create_mesh_from_organized_point_cloud_with_o3d(opc_smooth)

    plot_meshes(tri_mesh_o3d, tri_mesh_opc_o3d)



if __name__ == "__main__":
    main()