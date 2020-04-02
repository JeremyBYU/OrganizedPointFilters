loaded_from_source = r'''
extern "C" 
{
    // Dummy kernel to kickstart CUDA, not sure if necessary...
    __global__ void test_sum(const float* x1, const float* x2, float* y, \
                            unsigned int N)
    {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < N)
        {
            y[tid] = x1[tid] + x2[tid];
        }
    }

    //////// Start Helper Functions /////////
    __device__ void LoadPoint(float* out, const float* in, int outIdx, int inIdx)
    {
        out[outIdx * 3 + 0] = in[inIdx * 3 + 0];          // x cordinate
        out[outIdx * 3 + 1] = in[inIdx * 3 + 1];          // y cordinate
        out[outIdx * 3 + 2] = in[inIdx * 3 + 2];          // z cordinate
    }

    __device__ void ScalePointInPlace(float *point, const float &scale)
    {
        point[0] = point[0] * scale;
        point[1] = point[1] * scale;
        point[2] = point[2] * scale;
    }

    __device__ void AddPointsInPlace(float *point1, const float *point2)
    {
        point1[0] = point1[0] + point2[0];
        point1[1] = point1[1] + point2[1];
        point1[2] = point1[2] + point2[2];
    }

    __device__ void SubtractPointsInPlace(float *point1, const float *point2)
    {
        point1[0] = point1[0] - point2[0];
        point1[1] = point1[1] - point2[1];
        point1[2] = point1[2] - point2[2];
    }

    __device__ float PointDistance(const float *point1, const float *point2)
    {
        float x3 = (point1[0] - point2[0]);
        float y3 = (point1[1] - point2[1]);
        float z3 = (point1[2] - point2[2]);
        return norm3df(x3, y3, z3);
    }

    __device__ void Print2DPointArray(float *point, int rows, int cols)
    {
        printf("Shared Memory Size: %d\n ", rows);
        for(int i = 0; i < rows; ++i)
        {
            for(int j = 0; j < cols; ++j)
            {
                int idx = i * cols + j;
                // printf("(%.2f,%.2f,%.2f), ", point[idx*3], point[idx*3 + 1], point[idx*3 + 2]);
                printf("(%.4f), ", point[idx*3 + 1]);
            }
            printf("\n\n");

        }
    }
    //////// End Helper Functions /////////

    #define EPS 1e-12f
    #define BLOCK_WIDTH 32
    #define KERNEL_SIZE 3
    #define SHM_SIZE (BLOCK_WIDTH + KERNEL_SIZE -1)
    #define HALF_RADIUS KERNEL_SIZE/2
    __device__ void IntegeratePoint(int &nbr_shmIdx, float* SHM_POINTS, float *point_temp, float *point, float &total_weight, float* sum_point)
    {
        LoadPoint(point_temp, SHM_POINTS, 0, nbr_shmIdx);
        float dist = PointDistance(point, point_temp);
        float weight = 1.0f / (dist + EPS);
        total_weight += weight;
        ScalePointInPlace(point_temp, weight);
        AddPointsInPlace(sum_point, point_temp);
    }

    __device__ void ReadBlockAndHalo(float *SHM_POINTS, const float *opc, const int &shmRow_y, const int &shmCol_x, const int &srcRow_y, const int &srcCol_x, const int &cols)
    {
        // Copy Halo Cells, will have LOTS of branch divergence here
        int shmIdx = 0;
        int srcIdx_temp = 0;
        if (threadIdx.x == 0)
        {
            // Left Column, inner border
            shmIdx = shmRow_y * SHM_SIZE + (shmCol_x - 1);
            srcIdx_temp = srcRow_y * cols + (srcCol_x - 1);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
            if (threadIdx.y == 0)
            {
                // Top Left Corner 
                shmIdx = (shmRow_y - 1) * SHM_SIZE + (shmCol_x - 1);
                srcIdx_temp = (srcRow_y - 1) * cols + (srcCol_x - 1);
                LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
            }
            if (threadIdx.y == (blockDim.y - 1))
            {
                // Bottom Left Corner
                shmIdx = (shmRow_y + 1) * SHM_SIZE + (shmCol_x - 1);
                srcIdx_temp = (srcRow_y + 1) * cols + (srcCol_x - 1);
                LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
            }
        }
        if (threadIdx.x == (blockDim.x - 1))
        {
            // Right Column, inner border
            shmIdx = shmRow_y * SHM_SIZE + (shmCol_x + 1);
            srcIdx_temp = srcRow_y * cols + (srcCol_x + 1);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
            if (threadIdx.y == 0)
            {
                // Top Right Corner 
                shmIdx = (shmRow_y - 1) * SHM_SIZE + (shmCol_x + 1);
                srcIdx_temp = (srcRow_y - 1) * cols + (srcCol_x + 1);
                LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
            }
            if (threadIdx.y == (blockDim.y - 1))
            {
                // Bottom Right Corner
                shmIdx = (shmRow_y + 1) * SHM_SIZE + (shmCol_x + 1);
                srcIdx_temp = (srcRow_y + 1) * cols + (srcCol_x + 1);
                LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
            }
        }
        if (threadIdx.y == 0)
        {
            // Top Row
            shmIdx = (shmRow_y - 1) * SHM_SIZE + (shmCol_x);
            srcIdx_temp = (srcRow_y - 1) * cols + (srcCol_x);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
            // printf("Top Row; Row,Col: %d,%d; Point: (%.4f, %.4f, %.4f); Top Point: (%.4f, %.4f, %.4f)\n", srcRow_y, srcCol_x, point[0], point[1], point[2], point_temp[0], point_temp[1], point_temp[2]);
        }
        if (threadIdx.y == (blockDim.y - 1))
        {
            // Bottom Row
            shmIdx = (shmRow_y + 1) * SHM_SIZE + (shmCol_x);
            srcIdx_temp = (srcRow_y + 1) * cols + (srcCol_x);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
        }

    }

    __global__ void SmoothPoint(float* SHM_POINTS, const float *opc, float *opc_out, const int &srcIdx, 
                                const int &shmRow_y, const int &shmCol_x, const int &srcRow_y, 
                                const int &srcCol_x, const int &cols, const float &lambda)
    {

        int this_shmIdx = shmRow_y * SHM_SIZE + shmCol_x;  // shm point index for this points row/col

        float point[3];                     // will contain x,y,z point of this row/col
        float point_temp[3];                // a temporary for a point in stencil
        float sum_point[3] = {0,0,0};       // the weighted sum of points in stencil
        float dist = 0.0;                   // distance between point and neighbor
        float weight = 0.0;                 // weighting for point
        float total_weight = 0.0;           // total weight for scaling new point
        int nbr_shmIdx = 0;                 // nbr shared index

        LoadPoint(point, SHM_POINTS, 0, this_shmIdx);

        // I manually unwrapped the 3X3 Kernel loop below
        // Mostly because the center point should not occur in the kernel
        // and I didn't want an if statement for branch divergence

        //////// Left ////////////
        //////// Left Top ////////
        nbr_shmIdx = (shmRow_y - 1) * SHM_SIZE + (shmCol_x - 1);
        LoadPoint(point_temp, SHM_POINTS, 0, nbr_shmIdx);
        dist = PointDistance(point, point_temp);
        weight = 1.0f / (dist + EPS);
        total_weight += weight;
        ScalePointInPlace(point_temp, weight);
        AddPointsInPlace(sum_point, point_temp);
      
        //////// Left Center ////////
        nbr_shmIdx = (shmRow_y) * SHM_SIZE + (shmCol_x - 1);
        IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

        //////// Left Bottom ////////
        nbr_shmIdx = (shmRow_y + 1) * SHM_SIZE + (shmCol_x - 1);
        IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);


        //////// Center  ////////////
        //////// Center Top /////////
        nbr_shmIdx = (shmRow_y - 1) * SHM_SIZE + (shmCol_x);
        IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

        //////// Center Bottom /////////
        nbr_shmIdx = (shmRow_y + 1) * SHM_SIZE + (shmCol_x);
        IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);


        //////// Right  ////////////
        //////// Right Top /////////
        nbr_shmIdx = (shmRow_y - 1) * SHM_SIZE + (shmCol_x + 1);
        IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

        //////// Right Mid /////////
        nbr_shmIdx = (shmRow_y) * SHM_SIZE + (shmCol_x + 1);
        IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

        //////// Right Bottom /////////
        nbr_shmIdx = (shmRow_y + 1) * SHM_SIZE + (shmCol_x + 1);
        IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

        //////// Combine into New Point /////////
        /// This they very convoluted way of doing this -> opc_out(i, j) = point + lambda * (sum_point / total_weight - point);
        float new_scale = 1.0f / total_weight;
        ScalePointInPlace(sum_point, new_scale);
        SubtractPointsInPlace(sum_point, point);
        ScalePointInPlace(sum_point, lambda);
        AddPointsInPlace(sum_point, point);
    
        // store in global memory
        LoadPoint(opc_out, sum_point, srcIdx, 0);

    }

    __global__ void LaplacianLoopK3(float* opc, float* opc_out, int rows, int cols, float lambda)
    {
        __shared__ float SHM_POINTS[SHM_SIZE*3 * SHM_SIZE*3];  // block of 3D Points in shared memory

 		int shmRow_y = threadIdx.y + HALF_RADIUS;             // row of shared memory, interior
 		int shmCol_x = threadIdx.x + HALF_RADIUS;             // col of shared memory, interior
        int shmIdx = shmRow_y * SHM_SIZE + shmCol_x;          // index into shared memory, interior

        int srcRow_y = blockIdx.y * blockDim.y + threadIdx.y; // row in global memory
        int srcCol_x = blockIdx.x * blockDim.x + threadIdx.x; // col in global memory
        int srcIdx = srcRow_y * cols + srcCol_x;              // idx in global memory

        LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx);           // Copy global memory point (x,y,z) into shared memory
        if (srcRow_y > 0 && srcRow_y < (rows - 1) && srcCol_x > 0 && srcCol_x < (cols - 1))
        // Only perform smoothing in interior of image/opc, branch divergence occurs here
        {
            // Copy Halo Cells, will have LOTS of branch divergence here
            // After this call SHM_POINTS is completely filled 
            ReadBlockAndHalo(SHM_POINTS, opc, shmRow_y, shmCol_x, srcRow_y, srcCol_x, cols);
            __syncthreads();

            ////// Smooth Point Operation //////
            SmoothPoint(SHM_POINTS, opc, opc_out, srcIdx, shmRow_y, shmCol_x, srcRow_y, srcCol_x, cols, lambda);

        }

    }
}

'''


import logging

import cupy as cp
import numpy as np
import time


module = cp.RawModule(code=loaded_from_source)
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

    cp.cuda.Stream.null.synchronize()


    return opc_float_out