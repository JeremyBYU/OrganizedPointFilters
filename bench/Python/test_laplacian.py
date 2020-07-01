import pytest

import numpy as np
import open3d as o3d
import organizedpointfilters as opf
import organizedpointfilters.cuda as opf_cuda
from organizedpointfilters import Matrix3fRef

from examples.python.utility.helper import compute_normals_and_centroids_opc

####### Laplacian ############

######## PCD1 Tests ##########

@pytest.mark.parametrize("loops", [1, 5])
def test_cuda_laplacian_k3_pcd1(pcd1, loops, benchmark):
    """
    Will benchmark laplacian cuda
    """
    # seems the memory transfer penalty is only REALLY high on the first iteration  
    b = benchmark(opf_cuda.kernel.laplacian_K3_cuda, pcd1, loops, 1.0)

@pytest.mark.parametrize("loops", [1, 5])
def test_cpu_laplacian_k3_pcd1(pcd1, loops, benchmark):
    """
    Will benchmark laplacian cpu
    """
    a = Matrix3fRef(pcd1)
    b = benchmark(opf.filter.laplacian_K3, a, 1.0, loops)

@pytest.mark.parametrize("loops", [1, 5])
def test_o3d_laplacian_k3_pcd1(pcd1_mesh, loops, benchmark):
    """
    Will benchmark laplacian o3d
    """
    b = benchmark(pcd1_mesh.filter_smooth_laplacian, loops, 1.0, o3d.geometry.FilterScope.Vertex )

def test_o3d_adjacency_pcd1(pcd1_mesh, benchmark):
    """
    Will benchmark adjacency list creation
    """
    b = benchmark(pcd1_mesh.compute_adjacency_list)

# RGBD 1 Test

@pytest.mark.parametrize("loops", [1, 5])
def test_cuda_laplacian_k3_rgbd1(rgbd1, loops, benchmark):
    """
    Will benchmark laplacian cuda
    """
    # seems the memory transfer penalty is only REALLY high on the first iteration  
    b = benchmark(opf_cuda.kernel.laplacian_K3_cuda, rgbd1, loops, 1.0)

@pytest.mark.parametrize("loops", [1, 5])
def test_cpu_laplacian_k3_rgbd1(rgbd1, loops, benchmark):
    """
    Will benchmark laplacian cpu
    """
    a = Matrix3fRef(rgbd1)
    # looks like i swapped order in when I defined my function arguments....
    b = benchmark(opf.filter.laplacian_K3, a, 1.0, loops)


@pytest.mark.parametrize("loops", [1, 5])
def test_o3d_laplacian_k3_rgbd1(rgbd1_mesh, loops, benchmark):
    """
    Will benchmark laplacian o3d
    """
    b = benchmark(rgbd1_mesh.filter_smooth_laplacian, loops, 1.0, o3d.geometry.FilterScope.Vertex )

def test_o3d_adjacency_rgbd1(rgbd1_mesh, benchmark):
    """
    Will benchmark adjacency list creation
    """
    b = benchmark(rgbd1_mesh.compute_adjacency_list)
