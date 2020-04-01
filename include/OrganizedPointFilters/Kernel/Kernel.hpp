
#ifndef ORGANIZEDPOINTFILTERS_KERNEL
#define ORGANIZEDPOINTFILTERS_KERNEL

#include "OrganizedPointFilters/Types.hpp"
#include <iostream>

#define eps 1e-12f
#define OPF_KERNEL_DEFAULT_LAMBDA 0.5f
#define OPF_KERNEL_DEFAULT_ITER 1
#define OPF_KERNEL_DEFAULT_KERNEL_SIZE 3

namespace OrganizedPointFilters {

namespace Kernel {

inline void smooth_point(Eigen::Ref<RowMatrixXVec3f>& opc, Eigen::Ref<RowMatrixXVec3f>& opc_out, const int i,
                         const int j, const float lambda = OPF_KERNEL_DEFAULT_LAMBDA, const int kernel_size = 3)
{
    const int shift = static_cast<int>(kernel_size / 2);
    double total_weight = 0.0;
    auto& point = opc(i, j);
    Eigen::Vector3f sum_vertex(0, 0, 0);
    // std::cout << i << ", " << j << ", " << shift << ", " << std::endl;
    for (auto row = i - shift; row <= i + shift; ++row)
    {
        for (auto col = j - shift; col <= j + shift; ++col)
        {
            if (i == row && j == col)
                continue;
            float dist = (point - opc(row, col)).norm();
            // std::cout << "dist:" << dist << std::endl;
            float weight = 1. / (dist + eps);
            total_weight += weight;
            sum_vertex += weight * opc(row, col);
        }
    }
    opc_out(i, j) = point + lambda * (sum_vertex / total_weight - point);
    // std::cout << "sum_vertex " << sum_vertex << std::endl;
    // std::cout << "Previous Point: " << point << std::endl;
    // std::cout << "New Point Point: " << opc_out(i,j) << std::endl;
}

// template<typename kernel_size>
inline void LaplacianLoop(Eigen::Ref<RowMatrixXVec3f> opc, Eigen::Ref<RowMatrixXVec3f> opc_out,
                          const float lambda = OPF_KERNEL_DEFAULT_LAMBDA, const int kernel_size = 3)
{
    auto rows = opc.rows();
    auto cols = opc.cols();
    const int shift = static_cast<int>(kernel_size / 2);

    for (auto row = shift; row < rows - shift; ++row)
    {
        for (auto col = shift; col < cols - shift; ++col)
        {
            smooth_point(opc, opc_out, row, col, lambda, kernel_size);
        }
    }
}

inline RowMatrixXVec3f Laplacian(Eigen::Ref<RowMatrixXVec3f> opc, 
                      float lambda = OPF_KERNEL_DEFAULT_LAMBDA, int iterations = OPF_KERNEL_DEFAULT_ITER, int kernel_size = 3)
{
    // TODO - Only really need to copy the ghost/halo cells on border
    RowMatrixXVec3f opc_out(opc);
    bool need_copy = false;
    for(int i = 0; i < iterations; ++i)
    {
        if (i %2 == 0)
        {
            LaplacianLoop(opc, opc_out, lambda, kernel_size);
            need_copy = false;
        }
        else
        {
            LaplacianLoop(opc_out, opc, lambda, kernel_size); 
            need_copy = true;
        }
        
    }
    if (need_copy)
    {
        opc_out = opc;
    }
    return opc_out;
}

// for (int iter = 0; iter < number_of_iterations; ++iter) {
//     FilterSmoothLaplacianHelper(mesh, prev_vertices, prev_vertex_normals,
//                                 prev_vertex_colors, mesh->adjacency_list_,
//                                 lambda, filter_vertex, filter_normal,
//                                 filter_color);
//     if (iter < number_of_iterations - 1) {
//         std::swap(mesh->vertices_, prev_vertices);
//         std::swap(mesh->vertex_normals_, prev_vertex_normals);
//         std::swap(mesh->vertex_colors_, prev_vertex_colors);
//     }
// }

} // namespace Kernel
} // namespace OrganizedPointFilters

#endif
