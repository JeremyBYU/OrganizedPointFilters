
#ifndef ORGANIZEDPOINTFILTERS_KERNEL_NORMAL
#define ORGANIZEDPOINTFILTERS_KERNEL_NORMAL
#include <iostream>

#include "OrganizedPointFilters/Types.hpp"

#define OPF_BILATERAL_DEFAULT_ITER 1
#define OPF_BILATERAL_DEFAULT_SIGMA_LENGTH 0.1f     // 10 centimeters
#define OPF_BILATERAL_DEFAULT_SIGMA_ANGLE 0.174533f // 10 degrees

namespace OrganizedPointFilters {

namespace Filter {

inline void ComputeNormal(Eigen::Ref<RowMatrixXVec3f>& opc, Eigen::Ref<TriangleNormalMatrix>& normals,
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

inline void ComputeNormalLoop(Eigen::Ref<RowMatrixXVec3f> opc, Eigen::Ref<TriangleNormalMatrix> normals)
{
    // Rows and cols in OPC (POINTS)
    const int rows_points = static_cast<int>(opc.rows());
    const int cols_points = static_cast<int>(opc.cols());

    // Rows and columns in MESH (Triangels)
    // Each cell is composied of two triangles fromed by a right earcut
    const int rows_tris = rows_points - 1;
    const int cols_tris = cols_points - 1;

#if defined(_OPENMP)
    int num_threads = std::min(omp_get_max_threads(), OPF_KERNEL_OMP_MAX_THREAD);
    num_threads = std::max(num_threads, 1);
#pragma omp parallel for schedule(guided) num_threads(num_threads)
#endif
    for (int row_tri = 0; row_tri < rows_tris; ++row_tri)
    {
        for (int col_tri = 0; col_tri < cols_tris; ++col_tri)
        {
            ComputeNormal(opc, normals, row_tri, col_tri);
        }
    }
}

inline TriangleNormalMatrix ComputeNormals(Eigen::Ref<RowMatrixXVec3f> opc)
{
    // Rows and cols in OPC (POINTS)
    const int rows_points = static_cast<int>(opc.rows());
    const int cols_points = static_cast<int>(opc.cols());

    const int rows_tris = rows_points - 1;
    const int cols_tris = cols_points - 1;

    TriangleNormalMatrix normals(rows_tris, cols_tris);
    ComputeNormalLoop(opc, normals);

    return normals;
}

inline TriangleNormalMatrix BilateralFilterNormals(Eigen::Ref<RowMatrixXVec3f> opc,
                                                   Eigen::Ref<TriangleNormalMatrix> normals,
                                                   int loops = OPF_BILATERAL_DEFAULT_ITER,
                                                   float sigma_length = OPF_BILATERAL_DEFAULT_SIGMA_LENGTH,
                                                   float sigma_angle = OPF_BILATERAL_DEFAULT_SIGMA_ANGLE)

{
    TriangleNormalMatrix new_normals(normals.rows(), normals.cols());

    return new_normals;
}

} // namespace Filter
} // namespace OrganizedPointFilters

#endif
