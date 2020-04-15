extern "C" {

#define EPS 1e-12f
#define BLOCK_WIDTH 16
#define FPC 6

//////// Start Helper Functions /////////
__device__ void LoadPoint(float* out, const float* in, int outIdx, int inIdx)
{
    // First Point
    out[outIdx * FPC + 0] = in[inIdx * FPC + 0]; // x cordinate
    out[outIdx * FPC + 1] = in[inIdx * FPC + 1]; // y cordinate
    out[outIdx * FPC + 2] = in[inIdx * FPC + 2]; // z cordinate

    // Second Point
    out[outIdx * FPC + 3] = in[inIdx * FPC + 3]; // x cordinate
    out[outIdx * FPC + 4] = in[inIdx * FPC + 4]; // y cordinate
    out[outIdx * FPC + 5] = in[inIdx * FPC + 5]; // z cordinate
}

__device__ void LoadDoublePoint(float* out1, float* out2, const float* in, int inIdx)
{
    // First Point
    out1[0] = in[inIdx * FPC + 0]; // x cordinate
    out1[1] = in[inIdx * FPC + 1]; // y cordinate
    out1[2] = in[inIdx * FPC + 2]; // z cordinate

    // Second Point
    out2[0] = in[inIdx * FPC + 3]; // x cordinate
    out2[1] = in[inIdx * FPC + 4]; // y cordinate
    out2[2] = in[inIdx * FPC + 5]; // z cordinate
}

__device__ void LoadSinglePoint(float* out1, int idx, const float* in, int inIdx)
{
    // inIdx is the shared memory index, inIdx * FPC should point to a vector of 6
    // idx should either be 0 or 3. 0 to pont to first triangle, 3 to point to second triangle
    out1[0] = in[inIdx * FPC + idx + 0]; // x cordinate
    out1[1] = in[inIdx * FPC + idx + 1]; // y cordinate
    out1[2] = in[inIdx * FPC + idx + 2]; // z cordinate
}

__device__ void LoadSinglePointBackIntoMemory(float* out, const float* in, int outIdx, int outStartIdx)
{
    // inIdx is the shared memory index, inIdx * FPC should point to a vector of 6
    // idx should either be 0 or 3. 0 to pont to first triangle, 3 to point to second triangle
    out[outIdx * FPC + outStartIdx + 0] = in[0]; // x cordinate
    out[outIdx * FPC + outStartIdx + 1] = in[1]; // y cordinate
    out[outIdx * FPC + outStartIdx + 2] = in[2]; // z cordinate
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

__device__ void Print2DArrayWithOffset(float* points, int rows, int cols, int points_per_cell = 6, int offset = 0)
{
    printf("Shared Memory Size: %d\n ", rows);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            int idx = i * cols + j;
            // printf("(%.2f,%.2f,%.2f), ", point[idx*3], point[idx*3 + 1], point[idx*3 + 2]);
            float val1 = points[idx * points_per_cell + offset + 0];
            float val2 = points[idx * points_per_cell + offset + 1];
            float val3 = points[idx * points_per_cell + offset + 2];
            printf("(%.2f,%.2f,%.2f),", val1, val2, val3);
        }
        printf("\n");
    }
    printf("\n\n");
}

__device__ void PrintVec3f(float* point) { printf("(%.2f, %.2f, %.2f)\n", point[0], point[1], point[2]); }

__device__ float GaussianWeight(float value, float sigma_squared) { return __expf(-(value * value) / sigma_squared); }

__device__ void IntegerateNormal(int& nbr_shmIdx, float* SHM_NORMALS, float* SHM_CENTROIDS, float* temp_normal,
                                 float* temp_centroid, float* normal, float* centroid, float& total_weight,
                                 float* sum_normal, const float& sls, const float& sas, int nbr_starting_idx)
{
    // Get First Nbr Triangle Normal and Centroid
    LoadSinglePoint(temp_normal, nbr_starting_idx, SHM_NORMALS, nbr_shmIdx);
    LoadSinglePoint(temp_centroid, nbr_starting_idx, SHM_CENTROIDS, nbr_shmIdx);

    float dist_normal = PointDistance(normal, temp_normal);
    float dist_centroid = PointDistance(centroid, temp_centroid);

    // branch divergence ---   : (
    if (isnan(dist_centroid) || isnan(dist_normal)) return;

    float weight = GaussianWeight(dist_normal, sas) * GaussianWeight(dist_centroid, sls);

    total_weight += weight;

    ScalePointInPlace(temp_normal, weight);
    AddPointsInPlace(sum_normal, temp_normal);
}

//////// End Helper Functions /////////

// K3 Constants
#define KERNEL_SIZE_K3 3
#define SHM_SIZE_K3 (BLOCK_WIDTH + KERNEL_SIZE_K3 - 1)
#define HALF_RADIUS_K3 KERNEL_SIZE_K3 / 2

//////////////////// K3 FUnctions /////////////////////////
__device__ void ReadBlockAndHaloK3(float* SHM_POINTS, const float* normals_in, const int& shmRow_y, const int& shmCol_x,
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
        LoadPoint(SHM_POINTS, normals_in, shmIdx, srcIdx_temp);
        if (threadIdx.y == 0)
        {
            // Top Left Corner
            shmIdx = (shmRow_y - 1) * SHM_SIZE_K3 + (shmCol_x - 1);
            srcIdx_temp = (srcRow_y - 1) * cols + (srcCol_x - 1);
            LoadPoint(SHM_POINTS, normals_in, shmIdx, srcIdx_temp);
        }
        if (threadIdx.y == (blockDim.y - 1))
        {
            // Bottom Left Corner
            shmIdx = (shmRow_y + 1) * SHM_SIZE_K3 + (shmCol_x - 1);
            srcIdx_temp = (srcRow_y + 1) * cols + (srcCol_x - 1);
            LoadPoint(SHM_POINTS, normals_in, shmIdx, srcIdx_temp);
        }
    }
    if (threadIdx.x == (blockDim.x - 1))
    {
        // Right Column, inner border
        shmIdx = shmRow_y * SHM_SIZE_K3 + (shmCol_x + 1);
        srcIdx_temp = srcRow_y * cols + (srcCol_x + 1);
        LoadPoint(SHM_POINTS, normals_in, shmIdx, srcIdx_temp);
        if (threadIdx.y == 0)
        {
            // Top Right Corner
            shmIdx = (shmRow_y - 1) * SHM_SIZE_K3 + (shmCol_x + 1);
            srcIdx_temp = (srcRow_y - 1) * cols + (srcCol_x + 1);
            LoadPoint(SHM_POINTS, normals_in, shmIdx, srcIdx_temp);
        }
        if (threadIdx.y == (blockDim.y - 1))
        {
            // Bottom Right Corner
            shmIdx = (shmRow_y + 1) * SHM_SIZE_K3 + (shmCol_x + 1);
            srcIdx_temp = (srcRow_y + 1) * cols + (srcCol_x + 1);
            LoadPoint(SHM_POINTS, normals_in, shmIdx, srcIdx_temp);
        }
    }
    if (threadIdx.y == 0)
    {
        // Top Row
        shmIdx = (shmRow_y - 1) * SHM_SIZE_K3 + (shmCol_x);
        srcIdx_temp = (srcRow_y - 1) * cols + (srcCol_x);
        LoadPoint(SHM_POINTS, normals_in, shmIdx, srcIdx_temp);
    }
    if (threadIdx.y == (blockDim.y - 1))
    {
        // Bottom Row
        shmIdx = (shmRow_y + 1) * SHM_SIZE_K3 + (shmCol_x);
        srcIdx_temp = (srcRow_y + 1) * cols + (srcCol_x);
        LoadPoint(SHM_POINTS, normals_in, shmIdx, srcIdx_temp);
    }
}

// Each thread is responsible for ONE cell whcih has TWO triangles
// I call them the first and second respsectively
// They will integrate normals from neighboring cells (which also have 2 triangles first/second)
__global__ void SmoothPointK3(float* SHM_NORMALS, float* SHM_CENTROIDS, float* normals_out, const int& srcIdx,
                              const int& shmRow_y, const int& shmCol_x, const int& srcRow_y, const int& srcCol_x,
                              const int& cols, const float& sls, const float& sas)
{

    int this_shmIdx = shmRow_y * SHM_SIZE_K3 + shmCol_x; // shm point index for this points row/col

    float first_normal[3];    // will contain x,y,z normal of first triangle of this row/col
    float second_normal[3];   // will contain x,y,z normal of second triangle of this row/col
    float first_centroid[3];  // will contain x,y,z centroid of first triangle of this row/col
    float second_centroid[3]; // will contain x,y,z centroid of second triangle of this row/col

    float temp_normal[3];   // a temporary normal in stencil
    float temp_centroid[3]; // a temporary centroid in stencil

    float first_sum_normal[3] = {0, 0, 0}; // first normal sum, MUST BE ZERO INITIALIZED
    float second_sum_normal[3] = {0, 0, 0}; // first normal sum, MUST BE ZERO INITIALIZED

    float first_total_weight = 0.0;  // total weight (scaling) of first triangle
    float second_total_weight = 0.0; // total weight (scaling) of second triangle

    int nbr_shmIdx = 0; // nbr shared index

    // Load from shared memory to local registers, will be used multiple times
    LoadDoublePoint(first_normal, second_normal, SHM_NORMALS, this_shmIdx);
    LoadDoublePoint(first_centroid, second_centroid, SHM_CENTROIDS, this_shmIdx);

    // I manually unwrapped the 3X3 Kernel loop below
    // Mostly because the center point should not occur in the kernel
    // and I didn't want an if statement for branch divergence
    // sometimes I regret my decisions....

    //////// Left ////////////
    //////// Left Top ////////
    nbr_shmIdx = (shmRow_y - 1) * SHM_SIZE_K3 + (shmCol_x - 1);
    // First Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, first_normal, first_centroid,
                     first_total_weight, first_sum_normal, sls, sas, 0);
    // Second Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, first_normal, first_centroid,
                     first_total_weight, first_sum_normal, sls, sas, 3);

    // Second Triangle
    // First Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, second_normal, second_centroid,
                     second_total_weight, second_sum_normal, sls, sas, 0);
    // Second Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, second_normal, second_centroid,
                     second_total_weight, second_sum_normal, sls, sas, 3);

    //////// Left Center ////////
    nbr_shmIdx = (shmRow_y)*SHM_SIZE_K3 + (shmCol_x - 1);
    // First Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, first_normal, first_centroid,
                     first_total_weight, first_sum_normal, sls, sas, 0);
    // Second Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, first_normal, first_centroid,
                     first_total_weight, first_sum_normal, sls, sas, 3);

    // Second Triangle
    // First Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, second_normal, second_centroid,
                     second_total_weight, second_sum_normal, sls, sas, 0);
    // Second Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, second_normal, second_centroid,
                     second_total_weight, second_sum_normal, sls, sas, 3);

    //////// Left Bottom ////////
    nbr_shmIdx = (shmRow_y + 1) * SHM_SIZE_K3 + (shmCol_x - 1);
    // First Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, first_normal, first_centroid,
                     first_total_weight, first_sum_normal, sls, sas, 0);
    // Second Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, first_normal, first_centroid,
                     first_total_weight, first_sum_normal, sls, sas, 3);

    // Second Triangle
    // First Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, second_normal, second_centroid,
                     second_total_weight, second_sum_normal, sls, sas, 0);
    // Second Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, second_normal, second_centroid,
                     second_total_weight, second_sum_normal, sls, sas, 3);

    //////// Center  ////////////
    //////// Center Top /////////
    nbr_shmIdx = (shmRow_y - 1) * SHM_SIZE_K3 + (shmCol_x);
    // First Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, first_normal, first_centroid,
                     first_total_weight, first_sum_normal, sls, sas, 0);
    // Second Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, first_normal, first_centroid,
                     first_total_weight, first_sum_normal, sls, sas, 3);

    // Second Triangle
    // First Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, second_normal, second_centroid,
                     second_total_weight, second_sum_normal, sls, sas, 0);
    // Second Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, second_normal, second_centroid,
                     second_total_weight, second_sum_normal, sls, sas, 3);

    //////// Center Bottom /////////
    nbr_shmIdx = (shmRow_y + 1) * SHM_SIZE_K3 + (shmCol_x);
    // First Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, first_normal, first_centroid,
                     first_total_weight, first_sum_normal, sls, sas, 0);
    // Second Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, first_normal, first_centroid,
                     first_total_weight, first_sum_normal, sls, sas, 3);

    // Second Triangle
    // First Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, second_normal, second_centroid,
                     second_total_weight, second_sum_normal, sls, sas, 0);
    // Second Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, second_normal, second_centroid,
                     second_total_weight, second_sum_normal, sls, sas, 3);

    //////// Right  ////////////
    //////// Right Top /////////
    nbr_shmIdx = (shmRow_y - 1) * SHM_SIZE_K3 + (shmCol_x + 1);
    // First Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, first_normal, first_centroid,
                     first_total_weight, first_sum_normal, sls, sas, 0);
    // Second Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, first_normal, first_centroid,
                     first_total_weight, first_sum_normal, sls, sas, 3);

    // Second Triangle
    // First Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, second_normal, second_centroid,
                     second_total_weight, second_sum_normal, sls, sas, 0);
    // Second Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, second_normal, second_centroid,
                     second_total_weight, second_sum_normal, sls, sas, 3);

    //////// Right Mid /////////
    nbr_shmIdx = (shmRow_y)*SHM_SIZE_K3 + (shmCol_x + 1);
    // First Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, first_normal, first_centroid,
                     first_total_weight, first_sum_normal, sls, sas, 0);
    // Second Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, first_normal, first_centroid,
                     first_total_weight, first_sum_normal, sls, sas, 3);

    // Second Triangle
    // First Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, second_normal, second_centroid,
                     second_total_weight, second_sum_normal, sls, sas, 0);
    // Second Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, second_normal, second_centroid,
                     second_total_weight, second_sum_normal, sls, sas, 3);

    //////// Right Bottom /////////
    nbr_shmIdx = (shmRow_y + 1) * SHM_SIZE_K3 + (shmCol_x + 1);
    // First Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, first_normal, first_centroid,
                     first_total_weight, first_sum_normal, sls, sas, 0);
    // Second Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, first_normal, first_centroid,
                     first_total_weight, first_sum_normal, sls, sas, 3);

    // Second Triangle
    // First Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, second_normal, second_centroid,
                     second_total_weight, second_sum_normal, sls, sas, 0);
    // Second Neighbor Triangle
    IntegerateNormal(nbr_shmIdx, SHM_NORMALS, SHM_CENTROIDS, temp_normal, temp_centroid, second_normal, second_centroid,
                     second_total_weight, second_sum_normal, sls, sas, 3);

    //////// Combine into TWO new triangle normals /////////
    // if total_weight == 0.0 we have no updates
    if (first_total_weight > 0.0)
    {
        float new_scale = 1.0f / first_total_weight;
        ScalePointInPlace(first_sum_normal, new_scale);
        // store in global memory
        LoadSinglePointBackIntoMemory(normals_out, first_sum_normal, srcIdx, 0);
    }
    if (second_total_weight > 0.0)
    {
        float new_scale = 1.0f / second_total_weight;
        ScalePointInPlace(second_sum_normal, new_scale);
        // store in global memory
        LoadSinglePointBackIntoMemory(normals_out, second_sum_normal, srcIdx, 3);
    }
}

__global__ void BilateralLoopK3(float* normals_in, float* centroids_in, float* normals_out, int rows, int cols,
                                const float sls, const float sas)
{
    __shared__ float SHM_NORMALS[SHM_SIZE_K3 * FPC * SHM_SIZE_K3];   // block of 3D Triangle Normals in shared memory
    __shared__ float SHM_CENTROIDS[SHM_SIZE_K3 * FPC * SHM_SIZE_K3]; // block of 3D Triangle Centroids in shared memory

    int shmRow_y = threadIdx.y + HALF_RADIUS_K3;    // row of shared memory, interior
    int shmCol_x = threadIdx.x + HALF_RADIUS_K3;    // col of shared memory, interior
    int shmIdx = shmRow_y * SHM_SIZE_K3 + shmCol_x; // index into shared memory, interior

    int srcRow_y = blockIdx.y * blockDim.y + threadIdx.y; // row in global memory
    int srcCol_x = blockIdx.x * blockDim.x + threadIdx.x; // col in global memory
    int srcIdx = srcRow_y * cols + srcCol_x;              // idx in global memory

    LoadPoint(SHM_NORMALS, normals_in, shmIdx, srcIdx);     // Copy global memory point (x,y,z) into shared memory
    LoadPoint(SHM_CENTROIDS, centroids_in, shmIdx, srcIdx); // Copy global memory point (x,y,z) into shared memory

    if (srcRow_y > 0 && srcRow_y < (rows - 1) && srcCol_x > 0 && srcCol_x < (cols - 1))
    // Only perform smoothing in interior of image/opc, branch divergence occurs here
    {
        // Copy Halo Cells, will have LOTS of branch divergence here
        // After this call SHM_NORMALS and SHM_CENTROIDS are completely filled, no more global memory reads
        ReadBlockAndHaloK3(SHM_NORMALS, normals_in, shmRow_y, shmCol_x, srcRow_y, srcCol_x, cols);
        ReadBlockAndHaloK3(SHM_CENTROIDS, centroids_in, shmRow_y, shmCol_x, srcRow_y, srcCol_x, cols);
        __syncthreads();

        ////// Smooth Normal Operation //////
        SmoothPointK3(SHM_NORMALS, SHM_CENTROIDS, normals_out, srcIdx, shmRow_y, shmCol_x, srcRow_y, srcCol_x, cols,
                      sls, sas);
    }
}
}