import logging
from pathlib import Path
import time

import numpy as np
import pypcd.pypcd as pypcd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import open3d as o3d

from polylidar import extract_point_cloud_from_float_depth, extract_tri_mesh_from_organized_point_cloud, MatrixDouble
import organizedpointfilters as opf
import organizedpointfilters.cuda as opf_cuda
from organizedpointfilters import Matrix3f, Matrix3fRef

from .o3d_util import create_open_3d_pcd, plot_meshes, get_arrow, create_open_3d_mesh, open_3d_mesh_to_tri_mesh, assign_vertex_colors, get_colors

TEST_FIXTURES_DIR = Path('../polylidar-plane-benchmark/data').resolve()
SYNPEB_DIR = TEST_FIXTURES_DIR / "synpeb"
SYNPEB_MESHES_DIR = TEST_FIXTURES_DIR / "synpeb_meshes"
SYNPEB_DIR_TEST = SYNPEB_DIR / "test"
SYNPEB_DIR_TRAIN = SYNPEB_DIR / "train"

SYNPEB_DIR_TEST_GT = SYNPEB_DIR_TEST / "gt"
SYNPEB_DIR_TRAIN_GT = SYNPEB_DIR_TRAIN / "gt"

SYNPEB_DIR_TEST_ALL = [SYNPEB_DIR_TEST / "var{}".format(i) for i in range(1, 5)]
SYNPEB_DIR_TRAIN_ALL = [SYNPEB_DIR_TRAIN / "var{}".format(i) for i in range(1, 5)]

DEFAULT_PPB_FILE = SYNPEB_DIR_TRAIN_ALL[0] / "pc_01.pcd"
DEFAULT_PPB_FILE_SECONDARY = SYNPEB_DIR_TRAIN_ALL[0] / "pc_02.pcd"


SYNPEB_ALL_FNAMES = ["pc_{:02}.pcd".format(i) for i in range(1, 31)]



logger = logging.getLogger("PPB")


def tab40():
    """A discrete colormap with 40 unique colors"""
    colors_ = np.vstack([plt.cm.tab20c.colors, plt.cm.tab20b.colors])
    return colors.ListedColormap(colors_)

def load_pcd_and_meshes(input_file, stride=2, loops=5, _lambda=0.5, **kwargs):
    """Load PCD and Meshes
    """
    pc_raw, pc_image = load_pcd_file(input_file, stride)
    logger.info("Visualizing Point Cloud - Size: %dX%d ; # Points: %d",
                pc_image.shape[0], pc_image.shape[1], pc_raw.shape[0])

    # Get just the points, no intensity
    pc_points = np.ascontiguousarray(pc_raw[:, :3])
    # Create Open3D point cloud
    cmap = tab40()
    pcd_o3d = create_open_3d_pcd(pc_raw[:, :3], pc_raw[:, 3], cmap=cmap)

    tri_mesh, tri_mesh_o3d = create_meshes(pc_points, stride=stride, loops=loops, _lambda=_lambda)

    return pc_raw, pc_image, pcd_o3d, tri_mesh, tri_mesh_o3d

def create_meshes(pc_points, stride=2, loops=5, _lambda=0.5, **kwargs):
    tri_mesh = create_mesh_from_organized_point_cloud(pc_points, stride=stride)
    tri_mesh_o3d = create_open_3d_mesh(np.asarray(tri_mesh.triangles), pc_points)

    # Perform Smoothing
    if loops > 1:
        filter_scope = o3d.geometry.FilterScope(0)
        kwargs = dict(filter_scope=filter_scope)
        kwargs['lambda'] = _lambda
        t1 = time.perf_counter()
        tri_mesh_o3d = tri_mesh_o3d.filter_smooth_laplacian(loops, **kwargs)
        t2 = time.perf_counter()
        logger.info("Mesh Smoothing Took (ms): %.2f", (t2 - t1) * 1000)

    tri_mesh_o3d = tri_mesh_o3d.compute_triangle_normals()
    tri_mesh = open_3d_mesh_to_tri_mesh(tri_mesh_o3d)

    return tri_mesh, tri_mesh_o3d

def laplacian_opc(opc, loops=5, _lambda=0.5, kernel_size=3, **kwargs):
    opc_float = (np.ascontiguousarray(opc[:, :, :3])).astype(np.float32)

    a_ref = Matrix3fRef(opc_float)

    t1 = time.perf_counter()
    if kernel_size == 3:
        b_cp = opf.filter.laplacian_K3(a_ref, _lambda=_lambda, iterations=loops, **kwargs)
    else:
        b_cp = opf.filter.laplacian_K5(a_ref, _lambda=_lambda, iterations=loops, **kwargs)
    t2 = time.perf_counter()
    logger.info("OPC Mesh Smoothing Took (ms): %.2f", (t2 - t1) * 1000)

    opc_float_out = np.asarray(b_cp)

    opc_out = opc_float_out.astype(np.float64)

    num_points = opc_out.shape[0] * opc_out.shape[1]
    opc_out_flat = opc_out.reshape((num_points, 3))

    classes = opc[:,:, 3].reshape((num_points, ))
    cmap = tab40()
    pcd_out = create_open_3d_pcd(opc_out_flat, classes, cmap)

    return opc_out, pcd_out

def compute_normals_opc(opc, **kwargs):
    opc_float = (np.ascontiguousarray(opc[:, :, :3])).astype(np.float32)

    a_ref = Matrix3fRef(opc_float)

    t1 = time.perf_counter()
    normals = opf.filter.compute_normals(a_ref)
    t2 = time.perf_counter()
    logger.info("OPC Compute Normals Took (ms): %.2f", (t2 - t1) * 1000)
    normals_float_out = np.asarray(normals)
    normals_out = normals_float_out.astype(np.float64)

    return normals_out

def smooth_normals_opc(opc, loops=5, sigma_angle=0.17453, **kwargs):
    opc_float = (np.ascontiguousarray(opc[:, :, :3])).astype(np.float32)

    a_ref = Matrix3fRef(opc_float)

    t1 = time.perf_counter()
    normals = opf.filter.bilateral_K3(a_ref, iterations=loops, sigma_angle=sigma_angle)
    t2 = time.perf_counter()
    logger.info("OPC Bilateral Smooth Normals Took (ms): %.2f", (t2 - t1) * 1000)
    normals_float_out = np.asarray(normals)
    normals_out = normals_float_out.astype(np.float64)

    return normals_out


def laplacian_opc_cuda(opc, loops=5, _lambda=0.5, **kwargs):

    opc_float = (np.ascontiguousarray(opc[:, :, :3])).astype(np.float32)

    t1 = time.perf_counter()
    opc_float_out = opf_cuda.kernel.laplacian_K3_cuda(opc_float, loops=loops, _lambda=_lambda, **kwargs)
    t2 = time.perf_counter()

    logger.info("OPC CUDA Laplacian Mesh Smoothing Took (ms): %.2f", (t2 - t1) * 1000)

    # only for visualization purposes here
    opc_out = opc_float_out.astype(np.float64)
    num_points = opc_out.shape[0] * opc_out.shape[1]
    opc_out_flat = opc_out.reshape((num_points, 3))

    classes = opc[:,:, 3].reshape((num_points, ))
    cmap = tab40()
    pcd_out = create_open_3d_pcd(opc_out_flat, classes, cmap)


    return opc_out, pcd_out

def create_mesh_from_organized_point_cloud_with_o3d(pcd:np.ndarray, rows=500, cols=500, stride=2):
    pcd_ = pcd
    if pcd.ndim == 3:
        rows = pcd.shape[0]
        cols = pcd.shape[1]
        stride = 1
        pcd_ = pcd.reshape((rows*cols, 3))

    pcd_mat = MatrixDouble(pcd_)
    pcd_mat_np = np.asarray(pcd_mat)
    tri_mesh = extract_tri_mesh_from_organized_point_cloud(pcd_mat, rows, cols, stride)

    tri_mesh_o3d = create_open_3d_mesh(np.asarray(tri_mesh.triangles), pcd_)

    return tri_mesh, tri_mesh_o3d
    
    
def create_mesh_from_organized_point_cloud(pcd, rows=500, cols=500, stride=2):
    pcd_mat = MatrixDouble(pcd)
    pcd_mat_np = np.asarray(pcd_mat)
    tri_mesh = extract_tri_mesh_from_organized_point_cloud(pcd_mat, rows, cols, stride)
    return tri_mesh

def get_np_buffer_ptr(a):
    pointer, read_only_flag = a.__array_interface__['data']
    return hex(pointer)

def load_pcd_file(fpath, stride=2):
    pc = pypcd.PointCloud.from_path(fpath)
    x = pc.pc_data['x']
    y = pc.pc_data['y']
    z = pc.pc_data['z']
    i = pc.pc_data['intensity']

    width = int(pc.get_metadata()['width'])
    height = int(pc.get_metadata()['height'])

    # Flat Point Cloud
    pc_np = np.column_stack((x, y, z, i))
    # Image Point Cloud (organized)
    pc_np_image = np.reshape(pc_np, (width, height, 4))
    # I'm expecting the "image" to have the rows/y-axis going down
    pc_np_image = np.ascontiguousarray(np.flip(pc_np_image, 0))

    if stride is not None:
        pc_np_image = pc_np_image[::stride, ::stride, :]
        total_points_ds = pc_np_image.shape[0] * pc_np_image.shape[1]
        pc_np = np.reshape(pc_np_image, (total_points_ds, 4))
    
    pc_np = pc_np.astype(np.float64)
    pc_np_image = pc_np_image.astype(np.float64)

    return pc_np, pc_np_image