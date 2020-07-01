import pytest

import numpy as np
import open3d as o3d
import organizedpointfilters as opf
import organizedpointfilters.cuda as opf_cuda
from organizedpointfilters import Matrix3fRef

from examples.python.utility.helper import compute_normals_and_centroids_opc


####### Bilateral ############

def bench_cuda_opc(opc, loops):
    a_ref = Matrix3fRef(opc)
    normals, centroids = opf.filter.compute_normals_and_centroids(a_ref)
    normals_float = np.asarray(normals)
    centroid_float = np.asarray(centroids)
    # normals_opc, centroids_opc = compute_normals_and_centroids_opc(opc, convert_f64=False)
    b = opf_cuda.kernel.bilateral_K3_cuda(normals_float, centroid_float, loops)
    return b

@pytest.mark.parametrize("loops", [1, 5])
def test_cuda_bilateral_k3_pcd1(pcd1, loops, benchmark):
    """
    Will benchmark bilateral cuda
    """
    # seems the memory transfer penalty is REALLY high on the first iteration 
    # normals_opc, centroids_opc = compute_normals_and_centroids_opc(pcd1, convert_f64=False)
    b = benchmark(bench_cuda_opc, pcd1, loops)

@pytest.mark.parametrize("loops", [1, 5])
def test_cpu_bilateral_k3_pcd1(pcd1, loops, benchmark):
    """
    Will benchmark bilateral cpu
    """
    a = Matrix3fRef(pcd1)
    b = benchmark(opf.filter.bilateral_K3, a, loops)


@pytest.mark.parametrize("loops", [1, 5])
def test_cuda_bilateral_k3_rgbd1(rgbd1, loops, benchmark):
    """
    Will benchmark bilateral cuda
    """
    # seems the memory transfer penalty is REALLY high on the first iteration 
    # normals_opc, centroids_opc = compute_normals_and_centroids_opc(pcd1, convert_f64=False)
    b = benchmark(bench_cuda_opc, rgbd1, loops)

@pytest.mark.parametrize("loops", [1, 5])
def test_cpu_bilateral_k3_rgbd1(rgbd1, loops, benchmark):
    """
    Will benchmark bilateral cpu
    """
    a = Matrix3fRef(rgbd1)
    b = benchmark(opf.filter.bilateral_K3, a, loops)

