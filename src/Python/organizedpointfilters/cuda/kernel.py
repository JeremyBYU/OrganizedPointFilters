
import logging
import os

import cupy as cp
import numpy as np
import time

path_to_kernel_laplacian = os.path.join(os.path.dirname(__file__), "laplacian.cu")
with open(path_to_kernel_laplacian, 'r') as file:
    laplacian_kernel_code = file.read()

path_to_kernel_bilateral = os.path.join(os.path.dirname(__file__), "bilateral.cu")
with open(path_to_kernel_bilateral, 'r') as file:
    bilateral_kernel_code = file.read()

bilateral_module = cp.RawModule(code=bilateral_kernel_code)
laplacian_module = cp.RawModule(code=laplacian_kernel_code)


laplacian_K3 = laplacian_module.get_function('LaplacianLoopK3')
laplacian_K5 = laplacian_module.get_function('LaplacianLoopK5')
bilateral_K3 = bilateral_module.get_function('BilateralLoopK3')

BLOCK_SIZE_LAPLACIAN = 32

# def run_cuda_once():
#     N = 10
#     x1 = cp.arange(N**2, dtype=cp.float32).reshape(N, N)
#     x2 = cp.ones((N, N), dtype=cp.float32)
#     y = cp.zeros((N, N), dtype=cp.float32)
#     ker_sum((N,), (N,), (x1, x2, y, N**2, 0.1))   # y = x1 + x2
#     assert(cp.allclose(y, x1 + x2))


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

    kernel_size = 3
    t1 = time.perf_counter()
    # These device memory allocation take about 1.4 (ms) on my 2070 Super (250X250)
    opc_float_gpu_a = cp.asarray(opc_float)  # move the data to the current device.
    # make copy for the data (really only needed for ghost cells on second iter)
    opc_float_gpu_b = cp.copy(opc_float_gpu_a)

    opc_width = opc_float.shape[0]
    opc_height = opc_float.shape[1]

    num_points = opc_width * opc_height
    block_size = (BLOCK_SIZE_LAPLACIAN, BLOCK_SIZE_LAPLACIAN)
    grid_size = (int((opc_width - 1) / (BLOCK_SIZE_LAPLACIAN - kernel_size + 1)),
                 int((opc_height - 1) / (BLOCK_SIZE_LAPLACIAN - kernel_size + 1)))

    # One iteration takes about 0.18 ms, 5 iterations takes 0.21 ms, must be start up cost?
    use_b = True
    for i in range(loops):
        if (i % 2) == 0:
            laplacian_K3(grid_size, block_size, (opc_float_gpu_a, opc_float_gpu_b,
                                                 opc_height, opc_width, np.float32(_lambda)))   # y = x1 + x2
            use_b = True
        else:
            laplacian_K3(grid_size, block_size, (opc_float_gpu_b, opc_float_gpu_a,
                                                 opc_height, opc_width, np.float32(_lambda)))   # y = x1 + x2
            use_b = False

    opc_float_out = cp.asnumpy(opc_float_gpu_b) if use_b else cp.asnumpy(opc_float_gpu_a)
    t2 = time.perf_counter()

    # print((t2-t1) * 1000)
    cp.cuda.Stream.null.synchronize()

    return opc_float_out


def laplacian_K5_cuda(opc_float, loops=5, _lambda=0.5, **kwargs):
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

    kernel_size = 5
    t1 = time.perf_counter()

    opc_float_gpu_a = cp.asarray(opc_float)  # move the data to the current device.
    # make copy for the data (really only needed for ghost cells on second iter)
    opc_float_gpu_b = cp.copy(opc_float_gpu_a)

    opc_width = opc_float.shape[0]
    opc_height = opc_float.shape[1]

    num_points = opc_width * opc_height
    block_size = (BLOCK_SIZE_LAPLACIAN, BLOCK_SIZE_LAPLACIAN)
    grid_size = (int((opc_width - 1) / (BLOCK_SIZE_LAPLACIAN - kernel_size + 1)),
                 int((opc_height - 1) / (BLOCK_SIZE_LAPLACIAN - kernel_size + 1)))

    # One iteration takes about 0.18 ms, 5 iterations takes 0.21 ms, must be start up cost?
    use_b = True
    for i in range(loops):
        if (i % 2) == 0:
            laplacian_K5(grid_size, block_size, (opc_float_gpu_a, opc_float_gpu_b,
                                                 opc_height, opc_width, np.float32(_lambda)))
            use_b = True
        else:
            laplacian_K5(grid_size, block_size, (opc_float_gpu_b, opc_float_gpu_a,
                                                 opc_height, opc_width, np.float32(_lambda)))
            use_b = False

    opc_float_out = cp.asnumpy(opc_float_gpu_b) if use_b else cp.asnumpy(opc_float_gpu_a)
    t2 = time.perf_counter()

    cp.cuda.Stream.null.synchronize()

    return opc_float_out


def bilateral_K3_cuda(normals, centroids, loops=5, sigma_length=0.1, sigma_angle=0.261, **kwargs):
    """Performs bilateral filtering on an organized point cloud

    Arguments:
        normals {ndarray} -- Numpy array of MXNX2X3
        centroids {ndarray} -- Numpy array of MXNX2X3

    Keyword Arguments:
        loops {int} -- Number of iterations (default: {5})
        sigma_length {float} -- scaling for centroid distance component (default: {0.1})
        sigma_angle {float} -- scaling for normal angle component (default: {0.261})

    Returns:
        ndarray -- normals, MXNX2X3
    """
    assert normals.dtype == np.float32, "Numpy array must be float32"
    assert normals.ndim == 4, "Numpy array must have 4 dimensions, created from organized point cloud. MXNX2X3"
    assert normals.shape[2] == 2, "Numpy second to last dimension must be size 2, representing two triangles in a cell. MXNX2X3"
    assert normals.shape[3] == 3, "Numpy last dimension must be size 3, representing a triangle normal (x,y,z). MXNX2X3"

    assert centroids.dtype == np.float32, "Numpy array must be float32"
    assert centroids.ndim == 4, "Numpy array must have 4 dimensions, created from organized point cloud. MXNX2X3"
    assert centroids.shape[2] == 2, "Numpy second to last dimension must be size 2, representing two triangles in a cell. MXNX2X3"
    assert centroids.shape[
        3] == 3, "Numpy last dimension must be size 3, representing a triangle centroid (x,y,z). MXNX2X3"

    kernel_size = 3
    block_size_single = 16
    t1 = time.perf_counter()
    # These device memory allocation take about X (ms) on my 2070 Super (250X250)
    normals_float_gpu_a = cp.asarray(normals)  # move the data to the current device.
    # make copy for the data (really only needed for ghost cells on second iter)
    normals_float_gpu_b = cp.copy(normals_float_gpu_a)

    centroids_float_gpu_a = cp.asarray(centroids)  # move the data to the current device.

    rows = opc_float.shape[0]
    cols = opc_float.shape[1]

    num_points = rows * cols

    block_size = (block_size_single, block_size_single)
    grid_size = (int((cols - 1) / (block_size_single - kernel_size + 1)),
                 int((rows - 1) / (block_size_single - kernel_size + 1)))

    use_b = True
    for i in range(loops):
        if (i % 2) == 0:
            bilateral_K3(grid_size, block_size, (normals_float_gpu_a, centroids_float_gpu_a, normals_float_gpu_b,
                                                 rows, cols, np.float32(sigma_length), np.float32(sigma_angle)))
            use_b = True
        else:
            bilateral_K3(grid_size, block_size, (normals_float_gpu_b, centroids_float_gpu_a, normals_float_gpu_a,
                                                 rows, cols, np.float32(sigma_length), np.float32(sigma_angle)))
            use_b = False

    normals_float_out = cp.asnumpy(normals_float_gpu_b) if use_b else cp.asnumpy(normals_float_gpu_a)
    t2 = time.perf_counter()

    # print((t2-t1) * 1000)
    cp.cuda.Stream.null.synchronize()
    return normals_float_out
