import numpy as np
import organizedpointfilters.cuda as opf_cuda
import pytest
from examples.python.utility.helper import compute_normals_and_centroids_opc

@pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("size", [250, 500])
def test_cuda_laplacian_k3(loops, size, benchmark):
    """
    Will benchmark laplacian cuda
    """
    # seems the memory transfer penalty is only REALLY high on the first iteration
    a = np.random.rand(size, size, 3).astype(np.float32)
    b = benchmark(opf_cuda.kernel.laplacian_K3_cuda, a, loops, 0.5)

@pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("size", [250, 500])
def test_cuda_laplacian_k5(loops, size, benchmark):
    """
    Will benchmark laplacian cuda
    """
    # seems the memory transfer penalty is only REALLY high on the first iteration
    a = np.random.rand(size, size, 3).astype(np.float32)
    b = benchmark(opf_cuda.kernel.laplacian_K5_cuda, a, loops, 0.5)

@pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("size", [250, 500])
def test_cuda_bilateral_k3(loops, size, benchmark):
    """
    Will benchmark laplacian cuda
    """
    # seems the memory transfer penalty is only REALLY high on the first iteration
    a = np.random.rand(size, size, 3).astype(np.float32)
    normals_opc, centroids_opc = compute_normals_and_centroids_opc(a, convert_f64=False)
    b = benchmark(opf_cuda.kernel.bilateral_K3_cuda, normals_opc, centroids_opc, loops)

