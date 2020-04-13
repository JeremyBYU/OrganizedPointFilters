
#ifndef ORGANIZEDPOINTFILTERS_KERNEL_NORMAL
#define ORGANIZEDPOINTFILTERS_KERNEL_NORMAL
#include <iostream>

#include "OrganizedPointFilters/Types.hpp"

namespace OrganizedPointFilters {

namespace Kernel {


inline void ComputeNormal(Eigen::Ref<RowMatrixXVec3f>& opc, Eigen::Ref<TriangleNormalMatrix>& normals, const int row_tri,
                         const int col_tri)
{
    // All points involved in the TWO triangles in this cell

    auto& p1 = opc(row_tri, col_tri); // top left
    auto& p2 = opc(row_tri, col_tri + 1); // top right
    auto& p3 = opc(row_tri + 1, col_tri + 1); // bottom right
    auto& p4 = opc(row_tri + 1, col_tri); // bottom left

    // Triangle one is - p3, p2, p1
    // Triangle two is - p1, p4, p3

    auto triangle_one_normal = (p2 - p3).cross(p1 - p2).normalized();
    auto triangle_two_normal = (p4 - p1).cross(p3 - p4).normalized();



    auto &double_normal = normals(row_tri, col_tri);
    double_normal.block<1,3>(0,0) = triangle_one_normal;
    double_normal.block<1,3>(1,0) = triangle_two_normal;

    if (row_tri == 0 && col_tri == 0)
    {
        std::cout << "Normal: " << triangle_one_normal << std::endl;
        std::cout << "Normal: " << triangle_two_normal << std::endl;
        std::cout << "Combined: " << double_normal << std::endl;
    }

    
}

inline void NormalLoop(Eigen::Ref<RowMatrixXVec3f> opc, Eigen::Ref<TriangleNormalMatrix> normals)
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

    std::cout << "Rows Tris: " << rows_tris << "; Cols Tris: " << cols_tris << std::endl;
    TriangleNormalMatrix normals(rows_tris, cols_tris);
    NormalLoop(opc, normals);

    return normals;
}

} // namespace Kernel
} // namespace OrganizedPointFilters

#endif
