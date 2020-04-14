
import logging
import os

import cupy as cp
import numpy as np
import time

path_to_kernel_laplacian = os.path.join(os.path.dirname(__file__), "laplacian.cu") 
with open(path_to_kernel_laplacian, 'r') as file:
    kernel_code = file.read()

logging.debug("Trying to open %s kernel source code", path_to_kernel_laplacian)
module = cp.RawModule(code=kernel_code)

ker_sum = module.get_function('test_sum')
laplacian_K3 = module.get_function('LaplacianLoopK3')

BLOCK_SIZE = 32
KERNEL_SIZE = 3


def run_cuda_once():
    N = 10
    x1 = cp.arange(N**2, dtype=cp.float32).reshape(N, N)
    x2 = cp.ones((N, N), dtype=cp.float32)
    y = cp.zeros((N, N), dtype=cp.float32)
    ker_sum((N,), (N,), (x1, x2, y, N**2, 0.1))   # y = x1 + x2
    assert(cp.allclose(y, x1 + x2))


def laplacian_K3_cuda(opc_float, loops=5, _lambda=0.5, **kwargs):
    """Perform laplacian smoothing on point cloud using CUDA
    
    Arguments:
        opc_float {ndarray} -- Numpy array of size MXNX3 dtype==np.float32
    
    Keyword Arguments:
        loops {int} -- Number of loop interations of smoothing (default: {5})
        _lambda {float} -- [0-1.0] (default: {0.5})
    
    Returns:
        ndarray -- MXNX3 smoothed point cloud, dtype==np.float32
    """
    assert opc_float.dtype == np.float32, "Numpy array must be float32"
    assert opc_float.ndim == 3, "Numpy array must have 3 dimensions, an organized point cloud"
    assert opc_float.shape[2] == 3, "Numpy last dimension must be size 3, representing a point (x,y,z)"

    t1 = time.perf_counter()
    # These device memory allocation take about 1.4 (ms) on my 2070 Super (250X250)
    opc_float_gpu_a = cp.asarray(opc_float)  # move the data to the current device.
    opc_float_gpu_b = cp.copy(opc_float_gpu_a) # make copy for the data (really only needed for ghost cells on second iter)

    opc_width = opc_float.shape[0]
    opc_height = opc_float.shape[1]
    
    num_points = opc_width * opc_height
    block_size = (BLOCK_SIZE, BLOCK_SIZE)
    grid_size = (int((opc_width - 1) / (BLOCK_SIZE - KERNEL_SIZE + 1)), int((opc_height - 1) / (BLOCK_SIZE - KERNEL_SIZE + 1)))

    # One iteration takes about 0.18 ms, 5 iterations takes 0.21 ms, must be start up cost? 
    use_b = True
    for i in range(loops):
        if (i % 2) == 0:
            laplacian_K3(grid_size, block_size, (opc_float_gpu_a, opc_float_gpu_b, opc_height, opc_width, np.float32(_lambda)))   # y = x1 + x2
            use_b = True
        else:
            laplacian_K3(grid_size, block_size, (opc_float_gpu_b, opc_float_gpu_a, opc_height, opc_width, np.float32(_lambda)))   # y = x1 + x2
            use_b = False

    opc_float_out = cp.asnumpy(opc_float_gpu_b) if use_b else cp.asnumpy(opc_float_gpu_a)
    t2 = time.perf_counter()

    # print((t2-t1) * 1000)
    cp.cuda.Stream.null.synchronize()


    return opc_float_out