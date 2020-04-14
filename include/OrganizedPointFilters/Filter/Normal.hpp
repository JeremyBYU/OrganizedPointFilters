
#ifndef ORGANIZEDPOINTFILTERS_KERNEL_NORMAL
#define ORGANIZEDPOINTFILTERS_KERNEL_NORMAL
#include <iostream>

#include "OrganizedPointFilters/Types.hpp"

#define OPF_NORMAL_OMP_MAX_THREAD 16

namespace OrganizedPointFilters {

namespace Filter {

namespace NormalCore

{
inline void ComputeNormal(Eigen::Ref<RowMatrixXVec3f> opc, Eigen::Ref<OrganizedTriangleMatrix> normals,
                          const int row_tri, const int col_tri)
{
    // All points involved in the TWO triangles in this cell
    auto& p1 = opc(row_tri, col_tri);         // top left
    auto& p2 = opc(row_tri, col_tri + 1);     // top right
    auto& p3 = opc(row_tri + 1, col_tri + 1); // bottom right
    auto& p4 = opc(row_tri + 1, col_tri);     // bottom left

    auto& cell_normals = normals(row_tri, col_tri);

    // Triangle one is - p3, p2, p1
    // Triangle two is - p1, p4, p3

    // I'm concerned that eigen is actually worse than my hand written optimized normal calculation in polylidar
    // what you see is about 470 us for 250X250 opc
    // basically is (p2 - p3) doing a malloc? is stack allocated only once in the the calling function
    // the normalization creates new memory and then assigns, its faster just to normalize in place
    cell_normals.block<1, 3>(0, 0) = (p2 - p3).cross(p1 - p2).normalized();
    cell_normals.block<1, 3>(1, 0) = (p4 - p1).cross(p3 - p4).normalized();
}

inline void ComputeCentroid(Eigen::Ref<RowMatrixXVec3f> opc, Eigen::Ref<OrganizedTriangleMatrix> centroids,
                            const int row_tri, const int col_tri)
{
    // All points involved in the TWO triangles in this cell
    auto& p1 = opc(row_tri, col_tri);         // top left
    auto& p2 = opc(row_tri, col_tri + 1);     // top right
    auto& p3 = opc(row_tri + 1, col_tri + 1); // bottom right
    auto& p4 = opc(row_tri + 1, col_tri);     // bottom left

    auto& cell_centroid = centroids(row_tri, col_tri);

    // Triangle one is - p3, p2, p1
    // Triangle two is - p1, p4, p3

    // I'm concerned that eigen is actually worse than my hand written optimized normal calculation in polylidar
    // what you see is about 470 us for 250X250 opc
    // basically is (p2 - p3) doing a malloc? is stack allocated only once in the the calling function
    // the normalization creates new memory and then assigns, its faster just to normalize in place
    cell_centroid.block<1, 3>(0, 0) = (p3 + p2 + p1) / 3.0f;
    cell_centroid.block<1, 3>(1, 0) = (p1 + p4 + p3) / 3.0f;
}

} // namespace NormalCore

// inline void ComputeNormalLoop(Eigen::Ref<RowMatrixXVec3f> opc, Eigen::Ref<OrganizedTriangleMatrix> normals)
// {
//     // Rows and cols in OPC (POINTS)
//     const int rows_points = static_cast<int>(opc.rows());
//     const int cols_points = static_cast<int>(opc.cols());

//     // Rows and columns in MESH (Triangels)
//     // Each cell is composied of two triangles fromed by a right earcut
//     const int rows_tris = rows_points - 1;
//     const int cols_tris = cols_points - 1;

// #if defined(_OPENMP)
//     int num_threads = std::min(omp_get_max_threads(), OPF_KERNEL_OMP_MAX_THREAD);
//     num_threads = std::max(num_threads, 1);
// #pragma omp parallel for schedule(guided) num_threads(num_threads)
// #endif
//     for (int row_tri = 0; row_tri < rows_tris; ++row_tri)
//     {
//         for (int col_tri = 0; col_tri < cols_tris; ++col_tri)
//         {
//             NormalCore::ComputeNormal(opc, normals, row_tri, col_tri);
//         }
//     }
// }

// inline void ComputeCentroidLoop(Eigen::Ref<RowMatrixXVec3f> opc, Eigen::Ref<OrganizedTriangleMatrix> centroids)
// {
//     // Rows and cols in OPC (POINTS)
//     const int rows_points = static_cast<int>(opc.rows());
//     const int cols_points = static_cast<int>(opc.cols());

//     // Rows and columns in MESH (Triangels)
//     // Each cell is composied of two triangles fromed by a right earcut
//     const int rows_tris = rows_points - 1;
//     const int cols_tris = cols_points - 1;

// #if defined(_OPENMP)
//     int num_threads = std::min(omp_get_max_threads(), OPF_KERNEL_OMP_MAX_THREAD);
//     num_threads = std::max(num_threads, 1);
// #pragma omp parallel for schedule(guided) num_threads(num_threads)
// #endif
//     for (int row_tri = 0; row_tri < rows_tris; ++row_tri)
//     {
//         for (int col_tri = 0; col_tri < cols_tris; ++col_tri)
//         {
//             NormalCore::ComputeCentroid(opc, centroids, row_tri, col_tri);
//         }
//     }
// }

OrganizedTriangleMatrix ComputeNormals(Eigen::Ref<RowMatrixXVec3f> opc);

OrganizedTriangleMatrix ComputeCentroids(Eigen::Ref<RowMatrixXVec3f> opc);

std::tuple<OrganizedTriangleMatrix, OrganizedTriangleMatrix> ComputeNormalsAndCentroids(Eigen::Ref<RowMatrixXVec3f> opc);

} // namespace Filter
} // namespace OrganizedPointFilters

#endif
