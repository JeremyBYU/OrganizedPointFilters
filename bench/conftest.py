from os import path
import pytest
import numpy as np
import sys
sys.path.insert(0, path.dirname(path.dirname(path.abspath(__file__))))

from examples.python.utility.helper import (load_pcd_file, load_pcd_and_meshes)
from examples.python.utility.o3d_util import create_open_3d_mesh_from_tri_mesh
from organizedpointfilters.utility.helper import create_mesh_from_organized_point_cloud

THIS_DIR = path.dirname(path.realpath(__file__))
PCD_DIR = path.join(THIS_DIR, '../', 'fixtures', 'pcd')

@pytest.fixture
def pcd1():
    pc, pc_image = load_pcd_file(path.join(PCD_DIR, 'pc_01.pcd'), stride=2)
    return np.ascontiguousarray(pc_image[:, :, :3].astype('f4'))

@pytest.fixture
def pcd1_mesh():
    pc_raw, pc_image, pcd_o3d, tri_mesh, tri_mesh_o3d = load_pcd_and_meshes(path.join(PCD_DIR, 'pc_01.pcd'), stride=2)
    tri_mesh_o3d = tri_mesh_o3d.compute_adjacency_list()
    return tri_mesh_o3d

@pytest.fixture
def rgbd1():
    a =  np.load(path.join(PCD_DIR, 'rgbd_opc.npy'))
    return np.ascontiguousarray(a[:, :, :3].astype('f4'))

@pytest.fixture
def rgbd1_mesh():
    a =  np.load(path.join(PCD_DIR, 'rgbd_opc.npy'))
    tri_mesh, tri_map = create_mesh_from_organized_point_cloud(a, stride=2)
    tri_mesh_o3d = create_open_3d_mesh_from_tri_mesh(tri_mesh)
    tri_mesh_o3d = tri_mesh_o3d.compute_adjacency_list()

    return tri_mesh_o3d

