/** These are NAIVE CUDA kerenels for laplacian smoothing of organized point clouds
 *  There are probably a few bugs in here. I am not as familiar with GPU programming but tried my best.
 *  I had a hard time debugging GPU programs so I kept everything as simple and verbose as possible. 
*/

extern "C" {

#define EPS 1e-12f
#define BLOCK_WIDTH 32

//////// Start Helper Functions /////////
__device__ void LoadPoint(float* out, const float* in, int outIdx, int inIdx)
{
    out[outIdx * 3 + 0] = in[inIdx * 3 + 0]; // x cordinate
    out[outIdx * 3 + 1] = in[inIdx * 3 + 1]; // y cordinate
    out[outIdx * 3 + 2] = in[inIdx * 3 + 2]; // z cordinate
}

__device__ void ScalePointInPlace(float* point, const float& scale)
{
    point[0] = point[0] * scale;
    point[1] = point[1] * scale;
    point[2] = point[2] * scale;
}

__device__ void AddPointsInPlace(float* point1, const float* point2)
{
    point1[0] = point1[0] + point2[0];
    point1[1] = point1[1] + point2[1];
    point1[2] = point1[2] + point2[2];
}

__device__ void SubtractPointsInPlace(float* point1, const float* point2)
{
    point1[0] = point1[0] - point2[0];
    point1[1] = point1[1] - point2[1];
    point1[2] = point1[2] - point2[2];
}

__device__ float PointDistance(const float* point1, const float* point2)
{
    float x3 = (point1[0] - point2[0]);
    float y3 = (point1[1] - point2[1]);
    float z3 = (point1[2] - point2[2]);
    return norm3df(x3, y3, z3);
}

__device__ void Print2DPointArray(float* point, int rows, int cols)
{
    printf("Shared Memory Size: %d\n ", rows);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            int idx = i * cols + j;
            // printf("(%.2f,%.2f,%.2f), ", point[idx*3], point[idx*3 + 1], point[idx*3 + 2]);
            printf("(%.4f), ", point[idx * 3 + 1]);
        }
        printf("\n\n");
    }
}

__device__ void IntegeratePoint(int& nbr_shmIdx, float* SHM_POINTS, float* point_temp, float* point,
                                float& total_weight, float* sum_point, float max_dist = 0.25)
{
    LoadPoint(point_temp, SHM_POINTS, 0, nbr_shmIdx);
    float dist = PointDistance(point, point_temp);
    // branch divergence
    if (dist > max_dist || isnan(dist)) return;
    float weight = 1.0f / (dist + EPS);
    total_weight += weight;
    ScalePointInPlace(point_temp, weight);
    AddPointsInPlace(sum_point, point_temp);
}
//////// End Helper Functions /////////

// K3 Constants
#define KERNEL_SIZE_K3 3
#define SHM_SIZE_K3 (BLOCK_WIDTH + KERNEL_SIZE_K3 - 1)
#define HALF_RADIUS_K3 KERNEL_SIZE_K3 / 2


//////////////////// K3 Functions /////////////////////////
__device__ void ReadBlockAndHaloK3(float* SHM_POINTS, const float* opc, const int& shmRow_y, const int& shmCol_x,
                                   const int& srcRow_y, const int& srcCol_x, const int& cols)
{
    // Copy Halo Cells, will have LOTS of branch divergence here
    int shmIdx = 0;
    int srcIdx_temp = 0;
    if (threadIdx.x == 0)
    {
        // Left Column, inner border
        shmIdx = shmRow_y * SHM_SIZE_K3 + (shmCol_x - 1);
        srcIdx_temp = srcRow_y * cols + (srcCol_x - 1);
        LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
        if (threadIdx.y == 0)
        {
            // Top Left Corner
            shmIdx = (shmRow_y - 1) * SHM_SIZE_K3 + (shmCol_x - 1);
            srcIdx_temp = (srcRow_y - 1) * cols + (srcCol_x - 1);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
        }
        if (threadIdx.y == (blockDim.y - 1))
        {
            // Bottom Left Corner
            shmIdx = (shmRow_y + 1) * SHM_SIZE_K3 + (shmCol_x - 1);
            srcIdx_temp = (srcRow_y + 1) * cols + (srcCol_x - 1);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
        }
    }
    if (threadIdx.x == (blockDim.x - 1))
    {
        // Right Column, inner border
        shmIdx = shmRow_y * SHM_SIZE_K3 + (shmCol_x + 1);
        srcIdx_temp = srcRow_y * cols + (srcCol_x + 1);
        LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
        if (threadIdx.y == 0)
        {
            // Top Right Corner
            shmIdx = (shmRow_y - 1) * SHM_SIZE_K3 + (shmCol_x + 1);
            srcIdx_temp = (srcRow_y - 1) * cols + (srcCol_x + 1);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
        }
        if (threadIdx.y == (blockDim.y - 1))
        {
            // Bottom Right Corner
            shmIdx = (shmRow_y + 1) * SHM_SIZE_K3 + (shmCol_x + 1);
            srcIdx_temp = (srcRow_y + 1) * cols + (srcCol_x + 1);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
        }
    }
    if (threadIdx.y == 0)
    {
        // Top Row
        shmIdx = (shmRow_y - 1) * SHM_SIZE_K3 + (shmCol_x);
        srcIdx_temp = (srcRow_y - 1) * cols + (srcCol_x);
        LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
        // printf("Top Row; Row,Col: %d,%d; Point: (%.4f, %.4f, %.4f); Top Point: (%.4f, %.4f, %.4f)\n", srcRow_y,
        // srcCol_x, point[0], point[1], point[2], point_temp[0], point_temp[1], point_temp[2]);
    }
    if (threadIdx.y == (blockDim.y - 1))
    {
        // Bottom Row
        shmIdx = (shmRow_y + 1) * SHM_SIZE_K3 + (shmCol_x);
        srcIdx_temp = (srcRow_y + 1) * cols + (srcCol_x);
        LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
    }
}

__global__ void SmoothPointK3(float* SHM_POINTS, const float* opc, float* opc_out, const int& srcIdx,
                              const int& shmRow_y, const int& shmCol_x, const int& srcRow_y, const int& srcCol_x,
                              const int& cols, const float& lambda)
{

    int this_shmIdx = shmRow_y * SHM_SIZE_K3 + shmCol_x; // shm point index for this points row/col

    float point[3];                 // will contain x,y,z point of this row/col
    float point_temp[3];            // a temporary for a point in stencil
    float sum_point[3] = {0, 0, 0}; // the weighted sum of points in stencil
    float dist = 0.0;               // distance between point and neighbor
    float weight = 0.0;             // weighting for point
    float total_weight = 0.0;       // total weight for scaling new point
    int nbr_shmIdx = 0;             // nbr shared index

    LoadPoint(point, SHM_POINTS, 0, this_shmIdx);

    // I manually unwrapped the 3X3 Kernel loop below
    // Mostly because the center point should not occur in the kernel
    // and I didn't want an if statement for branch divergence

    //////// Left ////////////
    //////// Left Top ////////
    nbr_shmIdx = (shmRow_y - 1) * SHM_SIZE_K3 + (shmCol_x - 1);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    //////// Left Center ////////
    nbr_shmIdx = (shmRow_y)*SHM_SIZE_K3 + (shmCol_x - 1);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    //////// Left Bottom ////////
    nbr_shmIdx = (shmRow_y + 1) * SHM_SIZE_K3 + (shmCol_x - 1);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    //////// Center  ////////////
    //////// Center Top /////////
    nbr_shmIdx = (shmRow_y - 1) * SHM_SIZE_K3 + (shmCol_x);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    //////// Center Bottom /////////
    nbr_shmIdx = (shmRow_y + 1) * SHM_SIZE_K3 + (shmCol_x);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    //////// Right  ////////////
    //////// Right Top /////////
    nbr_shmIdx = (shmRow_y - 1) * SHM_SIZE_K3 + (shmCol_x + 1);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    //////// Right Mid /////////
    nbr_shmIdx = (shmRow_y)*SHM_SIZE_K3 + (shmCol_x + 1);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    //////// Right Bottom /////////
    nbr_shmIdx = (shmRow_y + 1) * SHM_SIZE_K3 + (shmCol_x + 1);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    //////// Combine into New Point /////////
    /// This is a very convoluted way of doing this -> opc_out(i, j) = point + lambda * (sum_point / total_weight -
    /// point);
    // if total_weight == 0.0 we have no updates
    if (total_weight == 0.0) return;

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
    __shared__ float SHM_POINTS[SHM_SIZE_K3 * 3 * SHM_SIZE_K3 * 3]; // block of 3D Points in shared memory

    int shmRow_y = threadIdx.y + HALF_RADIUS_K3;    // row of shared memory, interior
    int shmCol_x = threadIdx.x + HALF_RADIUS_K3;    // col of shared memory, interior
    int shmIdx = shmRow_y * SHM_SIZE_K3 + shmCol_x; // index into shared memory, interior

    int srcRow_y = blockIdx.y * blockDim.y + threadIdx.y; // row in global memory
    int srcCol_x = blockIdx.x * blockDim.x + threadIdx.x; // col in global memory
    int srcIdx = srcRow_y * cols + srcCol_x;              // idx in global memory

    LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx); // Copy global memory point (x,y,z) into shared memory
    if (srcRow_y > 0 && srcRow_y < (rows - 1) && srcCol_x > 0 && srcCol_x < (cols - 1))
    // Only perform smoothing in interior of image/opc, branch divergence occurs here
    {
        // Copy Halo Cells, will have LOTS of branch divergence here
        // After this call SHM_POINTS is completely filled
        ReadBlockAndHaloK3(SHM_POINTS, opc, shmRow_y, shmCol_x, srcRow_y, srcCol_x, cols);
        __syncthreads();

        ////// Smooth Point Operation //////
        SmoothPointK3(SHM_POINTS, opc, opc_out, srcIdx, shmRow_y, shmCol_x, srcRow_y, srcCol_x, cols, lambda);
    }
}

///////////////////////////////////////////////////////////////////
/////////////////////// K5 Functions //////////////////////////////
///////////////////////////////////////////////////////////////////

// K5 Constants
#define KERNEL_SIZE_K5 5
#define SHM_SIZE_K5 (BLOCK_WIDTH + KERNEL_SIZE_K5 - 1)
#define HALF_RADIUS_K5 KERNEL_SIZE_K5 / 2


__device__ void ReadBlockAndHaloK5(float* SHM_POINTS, const float* opc, const int& shmRow_y, const int& shmCol_x,
                                   const int& srcRow_y, const int& srcCol_x, const int& cols)
{
    // Copy Halo Cells, will have LOTS of branch divergence here
    int shmIdx = 0;
    int srcIdx_temp = 0;
    if (threadIdx.x == 0)
    {
        // Left Column, inner border
        shmIdx = shmRow_y * SHM_SIZE_K5 + (shmCol_x - 1);
        srcIdx_temp = srcRow_y * cols + (srcCol_x - 1);
        LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);

        shmIdx = shmRow_y * SHM_SIZE_K5 + (shmCol_x - 2);
        srcIdx_temp = srcRow_y * cols + (srcCol_x - 2);
        LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);

        if (threadIdx.y == 0)
        {
            // Top Left Corner
            shmIdx = (shmRow_y - 1) * SHM_SIZE_K5 + (shmCol_x - 1);
            srcIdx_temp = (srcRow_y - 1) * cols + (srcCol_x - 1);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);

            shmIdx = (shmRow_y - 2) * SHM_SIZE_K5 + (shmCol_x - 2);
            srcIdx_temp = (srcRow_y - 2) * cols + (srcCol_x - 2);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);

            // 1 Up, 2 Left
            shmIdx = (shmRow_y - 1) * SHM_SIZE_K5 + (shmCol_x - 2);
            srcIdx_temp = (srcRow_y - 1) * cols + (srcCol_x - 2);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);

            // 2 Up, 1 Left
            shmIdx = (shmRow_y - 2) * SHM_SIZE_K5 + (shmCol_x - 1);
            srcIdx_temp = (srcRow_y - 2) * cols + (srcCol_x - 1);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
        }
        if (threadIdx.y == (blockDim.y - 1))
        {
            // Bottom Left Corner
            shmIdx = (shmRow_y + 1) * SHM_SIZE_K5 + (shmCol_x - 1);
            srcIdx_temp = (srcRow_y + 1) * cols + (srcCol_x - 1);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);

            shmIdx = (shmRow_y + 2) * SHM_SIZE_K5 + (shmCol_x - 2);
            srcIdx_temp = (srcRow_y + 2) * cols + (srcCol_x - 2);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);

            // 1 Down, 2 Left
            shmIdx = (shmRow_y + 1) * SHM_SIZE_K5 + (shmCol_x - 2);
            srcIdx_temp = (srcRow_y + 1) * cols + (srcCol_x - 2);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);

            // 2 Down, 1 left
            shmIdx = (shmRow_y + 2) * SHM_SIZE_K5 + (shmCol_x - 1);
            srcIdx_temp = (srcRow_y + 2) * cols + (srcCol_x - 1);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
        }
    }
    if (threadIdx.x == (blockDim.x - 1))
    {
        // Right Column, inner border
        shmIdx = shmRow_y * SHM_SIZE_K5 + (shmCol_x + 1);
        srcIdx_temp = srcRow_y * cols + (srcCol_x + 1);
        LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);

        shmIdx = shmRow_y * SHM_SIZE_K5 + (shmCol_x + 2);
        srcIdx_temp = srcRow_y * cols + (srcCol_x + 2);
        LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);

        if (threadIdx.y == 0)
        {
            // Top Right Corner
            shmIdx = (shmRow_y - 1) * SHM_SIZE_K5 + (shmCol_x + 1);
            srcIdx_temp = (srcRow_y - 1) * cols + (srcCol_x + 1);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);

            shmIdx = (shmRow_y - 2) * SHM_SIZE_K5 + (shmCol_x + 2);
            srcIdx_temp = (srcRow_y - 2) * cols + (srcCol_x + 2);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);

            // 1 Up, 2 Right
            shmIdx = (shmRow_y - 1) * SHM_SIZE_K5 + (shmCol_x + 2);
            srcIdx_temp = (srcRow_y - 1) * cols + (srcCol_x + 2);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);

            // 2 Up, 1 Right
            shmIdx = (shmRow_y - 2) * SHM_SIZE_K5 + (shmCol_x + 1);
            srcIdx_temp = (srcRow_y - 2) * cols + (srcCol_x + 1);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
        }
        if (threadIdx.y == (blockDim.y - 1))
        {
            // Bottom Right Corner
            shmIdx = (shmRow_y + 1) * SHM_SIZE_K5 + (shmCol_x + 1);
            srcIdx_temp = (srcRow_y + 1) * cols + (srcCol_x + 1);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);

            shmIdx = (shmRow_y + 2) * SHM_SIZE_K5 + (shmCol_x + 2);
            srcIdx_temp = (srcRow_y + 2) * cols + (srcCol_x + 2);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);

            // 1 Down, 2 Right
            shmIdx = (shmRow_y + 1) * SHM_SIZE_K5 + (shmCol_x + 2);
            srcIdx_temp = (srcRow_y + 1) * cols + (srcCol_x + 2);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);

            // 2 Down, 1 Right
            shmIdx = (shmRow_y + 2) * SHM_SIZE_K5 + (shmCol_x + 1);
            srcIdx_temp = (srcRow_y + 2) * cols + (srcCol_x + 1);
            LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
        }
    }
    if (threadIdx.y == 0)
    {
        // Top Row
        shmIdx = (shmRow_y - 1) * SHM_SIZE_K5 + (shmCol_x);
        srcIdx_temp = (srcRow_y - 1) * cols + (srcCol_x);
        LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);

        shmIdx = (shmRow_y - 2) * SHM_SIZE_K5 + (shmCol_x);
        srcIdx_temp = (srcRow_y - 2) * cols + (srcCol_x);
        LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
    }
    if (threadIdx.y == (blockDim.y - 1))
    {
        // Bottom Row
        shmIdx = (shmRow_y + 1) * SHM_SIZE_K5 + (shmCol_x);
        srcIdx_temp = (srcRow_y + 1) * cols + (srcCol_x);
        LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);

        shmIdx = (shmRow_y + 2) * SHM_SIZE_K5 + (shmCol_x);
        srcIdx_temp = (srcRow_y + 2) * cols + (srcCol_x);
        LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx_temp);
    }
}

__global__ void SmoothPointK5(float* SHM_POINTS, const float* opc, float* opc_out, const int& srcIdx,
                              const int& shmRow_y, const int& shmCol_x, const int& srcRow_y, const int& srcCol_x,
                              const int& cols, const float& lambda)
{

    int this_shmIdx = shmRow_y * SHM_SIZE_K5 + shmCol_x; // shm point index for this points row/col

    float point[3];                 // will contain x,y,z point of this row/col
    float point_temp[3];            // a temporary for a point in stencil
    float sum_point[3] = {0, 0, 0}; // the weighted sum of points in stencil
    float total_weight = 0.0;       // total weight for scaling new point
    int nbr_shmIdx = 0;             // nbr shared index

    LoadPoint(point, SHM_POINTS, 0, this_shmIdx);

    // I manually unwrapped the 3X3 Kernel loop below
    // Mostly because the center point should not occur in the kernel
    // and I didn't want an if statement for branch divergence

    //////// Left ////////////
    //////// Left Top ////////
    nbr_shmIdx = (shmRow_y - 1) * SHM_SIZE_K5 + (shmCol_x - 1);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    //////// Left Center ////////
    nbr_shmIdx = (shmRow_y)*SHM_SIZE_K5 + (shmCol_x - 1);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    //////// Left Bottom ////////
    nbr_shmIdx = (shmRow_y + 1) * SHM_SIZE_K5 + (shmCol_x - 1);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    /////// Far Left ///////////
    nbr_shmIdx = (shmRow_y - 2) * SHM_SIZE_K5 + (shmCol_x - 2);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    nbr_shmIdx = (shmRow_y - 1) * SHM_SIZE_K5 + (shmCol_x - 2);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    nbr_shmIdx = (shmRow_y - 0) * SHM_SIZE_K5 + (shmCol_x - 2);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    nbr_shmIdx = (shmRow_y + 1) * SHM_SIZE_K5 + (shmCol_x - 2);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    nbr_shmIdx = (shmRow_y + 2) * SHM_SIZE_K5 + (shmCol_x - 2);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    //// Far Left Extras ////
    nbr_shmIdx = (shmRow_y - 2) * SHM_SIZE_K5 + (shmCol_x - 1);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    nbr_shmIdx = (shmRow_y + 2) * SHM_SIZE_K5 + (shmCol_x - 1);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    //////// Center  ////////////
    //////// Center Top /////////
    nbr_shmIdx = (shmRow_y - 1) * SHM_SIZE_K5 + (shmCol_x);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    //////// Center Bottom /////////
    nbr_shmIdx = (shmRow_y + 1) * SHM_SIZE_K5 + (shmCol_x);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    ///////// Far Center ////////
    nbr_shmIdx = (shmRow_y - 2) * SHM_SIZE_K5 + (shmCol_x);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    nbr_shmIdx = (shmRow_y + 2) * SHM_SIZE_K5 + (shmCol_x);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    //////// Right  ////////////
    //////// Right Top /////////
    nbr_shmIdx = (shmRow_y - 1) * SHM_SIZE_K5 + (shmCol_x + 1);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    //////// Right Mid /////////
    nbr_shmIdx = (shmRow_y)*SHM_SIZE_K5 + (shmCol_x + 1);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    //////// Right Bottom /////////
    nbr_shmIdx = (shmRow_y + 1) * SHM_SIZE_K5 + (shmCol_x + 1);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    /////// Far Rigth ///////////
    nbr_shmIdx = (shmRow_y - 2) * SHM_SIZE_K5 + (shmCol_x + 2);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    nbr_shmIdx = (shmRow_y - 1) * SHM_SIZE_K5 + (shmCol_x + 2);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    nbr_shmIdx = (shmRow_y - 0) * SHM_SIZE_K5 + (shmCol_x + 2);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    nbr_shmIdx = (shmRow_y + 1) * SHM_SIZE_K5 + (shmCol_x + 2);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    nbr_shmIdx = (shmRow_y + 2) * SHM_SIZE_K5 + (shmCol_x + 2);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    //// Far Rigth Extras ////
    nbr_shmIdx = (shmRow_y - 2) * SHM_SIZE_K5 + (shmCol_x + 1);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    nbr_shmIdx = (shmRow_y + 2) * SHM_SIZE_K5 + (shmCol_x + 1);
    IntegeratePoint(nbr_shmIdx, SHM_POINTS, point_temp, point, total_weight, sum_point);

    //////// Combine into New Point /////////
    /// This is a very convoluted way of doing this -> opc_out(i, j) = point + lambda * (sum_point / total_weight -
    /// point);
    // if total_weight == 0.0 we have no updates
    if (total_weight == 0.0) return;

    float new_scale = 1.0f / total_weight;
    ScalePointInPlace(sum_point, new_scale);
    SubtractPointsInPlace(sum_point, point);
    ScalePointInPlace(sum_point, lambda);
    AddPointsInPlace(sum_point, point);

    // store in global memory
    LoadPoint(opc_out, sum_point, srcIdx, 0);
}

__global__ void LaplacianLoopK5(float* opc, float* opc_out, int rows, int cols, float lambda)
{
    __shared__ float SHM_POINTS[SHM_SIZE_K5 * 3 * SHM_SIZE_K5]; // block of 3D Points in shared memory

    int shmRow_y = threadIdx.y + HALF_RADIUS_K5;    // row of shared memory, interior
    int shmCol_x = threadIdx.x + HALF_RADIUS_K5;    // col of shared memory, interior
    int shmIdx = shmRow_y * SHM_SIZE_K5 + shmCol_x; // index into shared memory, interior

    int srcRow_y = blockIdx.y * blockDim.y + threadIdx.y; // row in global memory
    int srcCol_x = blockIdx.x * blockDim.x + threadIdx.x; // col in global memory
    int srcIdx = srcRow_y * cols + srcCol_x;              // idx in global memory

    LoadPoint(SHM_POINTS, opc, shmIdx, srcIdx); // Copy global memory point (x,y,z) into shared memory
    if (srcRow_y >= HALF_RADIUS_K5 && srcRow_y < (rows - HALF_RADIUS_K5) && srcCol_x >= HALF_RADIUS_K5 &&
        srcCol_x < (cols - HALF_RADIUS_K5))
    // Only perform smoothing in interior of image/opc, branch divergence occurs here
    {
        // Copy Halo Cells, will have LOTS of branch divergence here
        // After this call SHM_POINTS is completely filled
        ReadBlockAndHaloK5(SHM_POINTS, opc, shmRow_y, shmCol_x, srcRow_y, srcCol_x, cols);
        __syncthreads();

        ////// Smooth Point Operation //////
        SmoothPointK5(SHM_POINTS, opc, opc_out, srcIdx, shmRow_y, shmCol_x, srcRow_y, srcCol_x, cols, lambda);
    }
}
}