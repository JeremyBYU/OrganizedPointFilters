import numpy as np
import organizedpointfilters.cuda as opf_cuda
def test_cuda_laplacian_k3(benchmark):
    """
    Will benchmark laplacian cuda
    """
    # seems the memory transfer penalty is only REALLY high on the first iteration
    a = np.random.rand(250, 250, 3).astype(np.float32)
    b = benchmark(opf_cuda.kernel.laplacian_K3_cuda, a, 5, 0.5)

def test_cuda_laplacian_k5(benchmark):
    """
    Will benchmark laplacian cuda
    """
    # seems the memory transfer penalty is only REALLY high on the first iteration
    a = np.random.rand(250, 250, 3).astype(np.float32)
    b = benchmark(opf_cuda.kernel.laplacian_K5_cuda, a, 5, 0.5)

