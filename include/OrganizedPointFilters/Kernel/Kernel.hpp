
#ifndef ORGANIZEDPOINTFILTERS_KERNEL
#define ORGANIZEDPOINTFILTERS_KERNEL

#include "OrganizedPointFilters/Types.hpp"
#include <iostream>

#define eps 1e-12f
#define OPF_KERNEL_DEFAULT_LAMBDA 0.5f
#define OPF_KERNEL_DEFAULT_ITER 1
#define OPF_KERNEL_DEFAULT_KERNEL_SIZE 3
#define OPF_KERNEL_OMP_MAX_THREAD 16

namespace OrganizedPointFilters {

namespace Kernel {

inline void smooth_point(Eigen::Ref<RowMatrixXVec3f>& opc, Eigen::Ref<RowMatrixXVec3f>& opc_out, const int i,
                         const int j, const float lambda = OPF_KERNEL_DEFAULT_LAMBDA, const int kernel_size = 3)
{
    const int shift = static_cast<const int>(kernel_size / 2);
    double total_weight = 0.0;
    auto& point = opc(i, j);
    Eigen::Vector3f sum_vertex(0, 0, 0);
    for (auto row = i - shift; row <= i + shift; ++row)
    {
        for (auto col = j - shift; col <= j + shift; ++col)
        {
            if (i == row && j == col) continue;
            float dist = (point - opc(row, col)).norm();
            float weight = 1. / (dist + eps);
            total_weight += weight;
            sum_vertex += weight * opc(row, col);
        }
    }
    auto new_point = sum_vertex / total_weight;
    opc_out(i, j) = point + lambda * (new_point - point);
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

inline RowMatrixXVec3f Laplacian(Eigen::Ref<RowMatrixXVec3f> opc, float lambda = OPF_KERNEL_DEFAULT_LAMBDA,
                                 int iterations = OPF_KERNEL_DEFAULT_ITER, int kernel_size = 3)
{
    // TODO - Only really need to copy the ghost/halo cells on border
    RowMatrixXVec3f opc_out(opc);
    bool need_copy = false;
    for (int i = 0; i < iterations; ++i)
    {
        if (i % 2 == 0)
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

template<int kernel_size = 3>
inline void SmoothPointT(Eigen::Ref<RowMatrixXVec3f>& opc, Eigen::Ref<RowMatrixXVec3f>& opc_out, const int i,
                         const int j, const float lambda = OPF_KERNEL_DEFAULT_LAMBDA)
{
    constexpr int shift = static_cast<const int>(kernel_size / 2);

    double total_weight = 0.0;
    auto& point = opc(i, j);
    Eigen::Vector3f sum_vertex(0, 0, 0);

    #pragma unroll
    for (auto row = 0; row < kernel_size; ++row)
    {
        int row_ = i - shift + row;
        #pragma unroll
        for (auto col = 0; col < kernel_size; ++col)
        {
            int col_ = j - shift + col;

            if (i == row_ && j == col_) continue;
            float dist = (point - opc(row_, col_)).norm();
            float weight = 1. / (dist + eps);
            total_weight += weight;
            sum_vertex += weight * opc(row_, col_);
        }
    }
    // if (i == 246 && j == 248)
    // {
    //     auto new_point = sum_vertex / total_weight;
    //     std::cout << "Row,Col: " << i << ", " << j <<  "; Old Point: " << point <<  "; New Point:" << new_point << std::endl;
    // }
    opc_out(i, j) = point + lambda * (sum_vertex / total_weight - point);
}

template<int kernel_size = 3>
inline void LaplacianLoopT(Eigen::Ref<RowMatrixXVec3f> opc, Eigen::Ref<RowMatrixXVec3f> opc_out,
                          const float lambda = OPF_KERNEL_DEFAULT_LAMBDA)
{
    const int rows = opc.rows();
    const int cols = opc.cols();
    constexpr int shift = static_cast<const int>(kernel_size / 2);
    const int rows_max = rows - shift;
    const int cols_max = cols - shift;

    #if defined(_OPENMP)
        int num_threads = std::min(omp_get_max_threads(), OPF_KERNEL_OMP_MAX_THREAD);
        num_threads = std::max(num_threads, 1);
        #pragma omp parallel for schedule(guided) num_threads(num_threads)
    #endif
    for (int row = shift; row < rows_max; ++row)
    {
        for (int col = shift; col < cols_max; ++col)
        {
            SmoothPointT<kernel_size>(opc, opc_out, row, col, lambda);
        }
    }
}

template<int kernel_size = 3>
inline RowMatrixXVec3f LaplacianT(Eigen::Ref<RowMatrixXVec3f> opc, float lambda = OPF_KERNEL_DEFAULT_LAMBDA,
                                 int iterations = OPF_KERNEL_DEFAULT_ITER)
{
    // TODO - Only really need to copy the ghost/halo cells on border
    RowMatrixXVec3f opc_out(opc);
    bool need_copy = false;
    for (int i = 0; i < iterations; ++i)
    {
        if (i % 2 == 0)
        {
            LaplacianLoopT<kernel_size>(opc, opc_out, lambda);
            need_copy = false;
        }
        else
        {
            LaplacianLoopT<kernel_size>(opc_out, opc, lambda);
            need_copy = true;
        }
    }
    if (need_copy)
    {
        opc_out = opc;
    }
    return opc_out;
}



} // namespace Kernel
} // namespace OrganizedPointFilters

#endif
