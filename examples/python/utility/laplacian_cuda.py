loaded_from_source = r'''
extern "C" 
{
    #define EPS 1e-12f
    __global__ void test_sum(const float* x1, const float* x2, float* y, \
                            unsigned int N, double echo)
    {
        // printf("echo: %.2f", echo);
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < N)
        {
            y[tid] = x1[tid] + x2[tid];
        }
    }
    __global__ void test_multiply(const float* x1, const float* x2, float* y, \
                                unsigned int N)
    {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < N)
        {
            y[tid] = x1[tid] * x2[tid];
        }
    }

    __device__ void LoadPoint(float* out, float* in, int outIdx, int inIdx)
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

    __device__ void SmoothPointFunction(int &nbr_shmIdx, float* SHM_POINTS, float *point_temp, float *point, float &total_weight, float* sum_point)
    {
        LoadPoint(point_temp, SHM_POINTS, 0, nbr_shmIdx);
        float dist = PointDistance(point, point_temp);
        float weight = 1.0f / (dist + EPS);
        total_weight += weight;
        ScalePointInPlace(point_temp, weight);
        AddPointsInPlace(sum_point, point_temp);
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





    #define BLOCK_WIDTH 32
    #define KERNEL_SIZE 3
    #define SHM_SIZE (BLOCK_WIDTH + KERNEL_SIZE -1)
    #define HALF_RADIUS KERNEL_SIZE/2
    #define IMG_IDX(row, col, Col) row * Col + row
    #define IMG_ROW(idx, Row, Col) 
    __global__ void LaplacianLoopK3(float* opc, float* opc_out, int rows, int cols, float lambda)
    {
        __shared__ float SHM_POINTS[SHM_SIZE*3 * SHM_SIZE*3];  // block of 3D Points in shared memory

        

 		int shmRow_y = threadIdx.y + HALF_RADIUS;             // row of shared memory, interior
 		int shmCol_x = threadIdx.x + HALF_RADIUS;             // col of shared memory, interior
        int shmIdx = shmRow_y * SHM_SIZE + shmCol_x;          // index into shared memory, interior

        int srcRow_y = blockIdx.y * blockDim.y + threadIdx.y; // row in global memory
        int srcCol_x = blockIdx.x * blockDim.x + threadIdx.x; // col in global memory
        int srcIdx = srcRow_y * cols + srcCol_x;              // idx in global memory

        int srcIdx_temp = srcIdx;

        LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx);           // Copy Point(x,y,z) into shared memory
        if (srcRow_y > 0 && srcRow_y < (rows - 1) && srcCol_x > 0 && srcCol_x < (cols - 1))
        // Branch divergence near borders of image
        {
            // Copy inside portion of shared memory

            // float dummy[3];
            // LoadPoint(dummy, opc, 0, srcIdx);
            // printf("All; Row,Col: %d,%d; Point: (%.4f, %.4f, %.4f);\n", srcRow_y, srcCol_x, dummy[0], dummy[1], dummy[2]);

            // Copy Halo Cells, will have LOTS of branch divergence here
            if (threadIdx.x == 0)
            {
                // Left Column, inner border
                shmIdx = shmRow_y * SHM_SIZE + (shmCol_x - 1);
                srcIdx_temp = srcRow_y * cols + (srcCol_x - 1);
                LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
                float point[3];
                float point_temp[3];
                LoadPoint(point, opc, 0, srcIdx);
                LoadPoint(point_temp, SHM_POINTS, 0, shmIdx);
                // printf("Left Column; Row,Col: %d,%d; Point: (%.4f, %.4f, %.4f); Left Point: (%.4f, %.4f, %.4f)\n", srcRow_y, srcCol_x, point[0], point[1], point[2], point_temp[0], point_temp[1], point_temp[2]);
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

                float point[3];
                float point_temp[3];
                LoadPoint(point, opc, 0, srcIdx);
                LoadPoint(point_temp, SHM_POINTS, 0, shmIdx);
                // printf("Top Row; Row,Col: %d,%d; Point: (%.4f, %.4f, %.4f); Top Point: (%.4f, %.4f, %.4f)\n", srcRow_y, srcCol_x, point[0], point[1], point[2], point_temp[0], point_temp[1], point_temp[2]);
            }
            if (threadIdx.y == (blockDim.y - 1))
            {
                // Bottom Row
                shmIdx = (shmRow_y + 1) * SHM_SIZE + (shmCol_x);
                srcIdx_temp = (srcRow_y + 1) * cols + (srcCol_x);
                LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
            }

            __syncthreads();

            //if(blockIdx.y == 1 && blockIdx.x == 1 && threadIdx.y == 0 && threadIdx.x == 0)
            //{
            //    printf("Block (1,1); Row,Col: %d,%d; \n", srcRow_y, srcCol_x);
            //    printf("block dim: %d \n", blockDim.x);
            //    Print2DPointArray(SHM_POINTS, SHM_SIZE, SHM_SIZE);
            //}


            // TODO - Try and use Eigen instead of manual math
            // Smooth Point
            int this_shmIdx = shmRow_y * SHM_SIZE + shmCol_x;  // shm point index for this i,j
            float total_weight = 0.0;

            float point[3];                     // will contain x,y,z point of this row/col
            float point_temp[3];                // a temporary for a point in stencil
            float sum_point[3] = {0,0,0};                 // the weighted sum of points in stencil

            LoadPoint(point, SHM_POINTS, 0, this_shmIdx);


            float dist = 0.0;
            float weight = 0.0;
            int nbr_shmIdx = 0;


            //////// Left ////////////
            //////// Left Top ////////
            nbr_shmIdx = (shmRow_y - 1) * SHM_SIZE + (shmCol_x - 1);
            LoadPoint(point_temp, SHM_POINTS, 0, nbr_shmIdx);
            dist = PointDistance(point, point_temp);
            weight = 1.0f / (dist + EPS);
            total_weight += weight;
            ScalePointInPlace(point_temp, weight);
            AddPointsInPlace(sum_point, point_temp);
            // printf("Row,Col: %d,%d; Old Point: (%.4f, %.4f, %.4f); Weight: %.4f, Left Top Point Weighted: (%.4f, %.4f, %.4f); Dist: %.4f \n", srcRow_y, srcCol_x, point[0], point[1], point[2], weight,  point_temp[0], point_temp[1], point_temp[2], dist);


            //////// Left Center ////////
            nbr_shmIdx = (shmRow_y) * SHM_SIZE + (shmCol_x - 1);
            SmoothPointFunction(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);
            // printf("Row,Col: %d,%d; Old Point: (%.4f, %.4f, %.4f); Weight: %.4f, Left Center Point Weighted: (%.4f, %.4f, %.4f); Dist: %.4f \n", srcRow_y, srcCol_x, point[0], point[1], point[2], weight,  point_temp[0], point_temp[1], point_temp[2], dist);

            //////// Left Bottom ////////
            nbr_shmIdx = (shmRow_y + 1) * SHM_SIZE + (shmCol_x - 1);
            SmoothPointFunction(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);


            //////// Center  ////////////
            //////// Center Top /////////
            nbr_shmIdx = (shmRow_y - 1) * SHM_SIZE + (shmCol_x);
            SmoothPointFunction(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

            //////// Center Bottom /////////
            nbr_shmIdx = (shmRow_y + 1) * SHM_SIZE + (shmCol_x);
            SmoothPointFunction(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);


            //////// Right  ////////////
            //////// Right Top /////////
            nbr_shmIdx = (shmRow_y - 1) * SHM_SIZE + (shmCol_x + 1);
            SmoothPointFunction(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

            //////// Right Mid /////////
            nbr_shmIdx = (shmRow_y) * SHM_SIZE + (shmCol_x + 1);
            SmoothPointFunction(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

            //////// Right Bottom /////////
            nbr_shmIdx = (shmRow_y + 1) * SHM_SIZE + (shmCol_x + 1);
            SmoothPointFunction(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

            float new_scale = 1.0f / total_weight;
            ScalePointInPlace(sum_point, new_scale);
            SubtractPointsInPlace(sum_point, point);
            ScalePointInPlace(sum_point, lambda);
            AddPointsInPlace(sum_point, point);
            //printf("Row,Col: %d,%d; Old Point: (%.4f, %.4f, %.4f); New Point (%.4f, %.4f, %.4f); lambda: %.2f\n", srcRow_y, srcCol_x, point[0], point[1], point[2], sum_point[0], sum_point[1], sum_point[2], lambda);

            // store in global memory
            LoadPoint(opc_out, sum_point, srcIdx, 0);

        }

    }
}

'''


        # int blockId = blockIdx.x + blockIdx.y * gridDim.x;
        # int threadId = blockId * (blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x)+ threadIdx.x;
        # // multiply by 3 for every threadID to get start of first float in x for (x,y,z) of point
        # int pointId = 3 * threadId;

        # int shmIdx = threadIdx.y * BLOCK_WIDTH + threadIdx.x;
 		# int shmRow_y = shmIdx / SHM_SIZE;  // row of shared memory
 		# int shmCol_x = shmIdx % SHM_SIZE;  // col of shared memory

        # int srcRow_y = blockIdx.y * BLOCK_WIDTH + shmRow_y - HALF_RADIUS
        # int srcCol_x = blockIdx.x * BLOCK_WIDTH + shmCol_x - HALF_RADIUS;

        # int srcIdx = (srcRow_y * rows + srcCol_x) * 3;   // index of start of float for point

 		# if(srcY>= 0 && srcY < height && srcX>=0 && srcX < width)
 		# 	N_ds[destY][destX] = InputImageData[src];  // copy element of image in shared memory
 		# else
 		# 	N_ds[destY][destX] = 0;

import logging

import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from .o3d_util import create_open_3d_pcd

BLOCK_SIZE = 32
KERNEL_SIZE = 3

logger = logging.getLogger("PPB")

cp.cuda.Stream.null.synchronize()

module = cp.RawModule(code=loaded_from_source)
ker_sum = module.get_function('test_sum')
laplacian_K3 = module.get_function('LaplacianLoopK3')
N = 10
x1 = cp.arange(N**2, dtype=cp.float32).reshape(N, N)
x2 = cp.ones((N, N), dtype=cp.float32)
y = cp.zeros((N, N), dtype=cp.float32)
ker_sum((N,), (N,), (x1, x2, y, N**2, 0.1))   # y = x1 + x2
assert(cp.allclose(y, x1 + x2))

cp.cuda.Stream.null.synchronize()


def tab40():
    """A discrete colormap with 40 unique colors"""
    colors_ = np.vstack([plt.cm.tab20c.colors, plt.cm.tab20b.colors])
    return colors.ListedColormap(colors_)



def laplacian_opc_cuda(opc, loops=5, _lambda=0.5, kernel_size=3, **kwargs):
    opc_float = (np.ascontiguousarray(opc[:, :, :3])).astype(np.float32)

    opc_float_gpu_a = cp.asarray(opc_float)  # move the data to the current device.
    opc_float_gpu_b = cp.copy(opc_float_gpu_a)

    opc_width = opc_float.shape[0]
    opc_height = opc_float.shape[1]
    
    num_points = opc_width * opc_height
    block_size = (BLOCK_SIZE, BLOCK_SIZE)
    grid_size = (int((opc_width - 1) / (BLOCK_SIZE - KERNEL_SIZE + 1)), int((opc_height - 1) / (BLOCK_SIZE - KERNEL_SIZE + 1)))
    # print(num_points)
    # print(grid_size)
    t1 = time.perf_counter()


    use_b = True
    for i in range(loops):
        if (i % 2) == 0:
            laplacian_K3(grid_size, block_size, (opc_float_gpu_a, opc_float_gpu_b, opc_height, opc_width, np.float32(0.5)))   # y = x1 + x2
            use_b = True
        else:
            laplacian_K3(grid_size, block_size, (opc_float_gpu_b, opc_float_gpu_a, opc_height, opc_width, np.float32(0.5)))   # y = x1 + x2
            use_b = False

    cp.cuda.Stream.null.synchronize()
    # print(opc_float[32, 1:32])
    # print(opc_float[31, 1:32])
    # print(opc_float[31:65, 31:65, 1])
    # print(opc_float[208, 64, :], opc_float[208, 63, :])

    opc_float_out = cp.asnumpy(opc_float_gpu_b) if use_b else cp.asnumpy(opc_float_gpu_a)
    t2 = time.perf_counter()
    logger.info("OPC CUDA Mesh Smoothing Took (ms): %.2f", (t2 - t1) * 1000)


    # opc_float_out = cp.asnumpy(opc_float_gpu_b)
    opc_out = opc_float_out.astype(np.float64)

    num_points = opc_out.shape[0] * opc_out.shape[1]
    opc_out_flat = opc_out.reshape((num_points, 3))

    classes = opc[:,:, 3].reshape((num_points, ))
    cmap = tab40()
    pcd_out = create_open_3d_pcd(opc_out_flat, classes, cmap)


    return opc_out, pcd_out
