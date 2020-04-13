
#ifndef ORGANIZEDPOINTFILTERS_KERNEL
#define ORGANIZEDPOINTFILTERS_KERNEL
#include <iostream>

#include "OrganizedPointFilters/Types.hpp"

#define eps 1e-12f
#define OPF_KERNEL_DEFAULT_LAMBDA 0.5f
#define OPF_KERNEL_DEFAULT_ITER 1
#define OPF_KERNEL_DEFAULT_KERNEL_SIZE 3
#define OPF_KERNEL_OMP_MAX_THREAD 16
#define OPF_KERNEL_MAX_FLOAT 1000000.0

namespace OrganizedPointFilters {

namespace Kernel {

inline void smooth_point(Eigen::Ref<RowMatrixXVec3f>& opc, Eigen::Ref<RowMatrixXVec3f>& opc_out, const int i,
                         const int j, const float lambda = OPF_KERNEL_DEFAULT_LAMBDA, const int kernel_size = 3)
{
    const int shift = static_cast<const int>(kernel_size / 2);
    float total_weight = 0.0;
    auto& point = opc(i, j);
    Eigen::Vector3f sum_vertex(0, 0, 0);
    for (auto row = i - shift; row <= i + shift; ++row)
    {
        for (auto col = j - shift; col <= j + shift; ++col)
        {
            if (i == row && j == col) continue;
            float dist = (point - opc(row, col)).norm();
            float weight = 1.0f / (dist + eps);
            total_weight += weight;
            sum_vertex += weight * opc(row, col);
        }
    }
    opc_out(i, j) = point + lambda * (sum_vertex / total_weight - point);
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

inline void OppositePointK3(const int& row, const int& col, int& new_row, int& new_col)
{
    new_row = row;
    new_col = col;
    if (col == 0 || col == 2)
    {
        new_col = col == 0 ? 2 : 0;
        new_row = row;
    }
    else
    {
        new_col = col;
        new_row = row == 0 ? 2 : 0;
    }
    
}

template <int kernel_size = 3>
inline void SmoothPointT(Eigen::Ref<RowMatrixXVec3f>& opc, Eigen::Ref<RowMatrixXVec3f>& opc_out, const int i,
                         const int j, const float lambda = OPF_KERNEL_DEFAULT_LAMBDA,
                         float max_dist = OPF_KERNEL_MAX_FLOAT)
{
    constexpr int shift = static_cast<const int>(kernel_size / 2);

    float total_weight = 0.0;
    auto& point = opc(i, j);
    Eigen::Vector3f sum_vertex(0, 0, 0);
    float weight = 0.0;

    int opp_row = 0;
    int opp_col = 0;
    Eigen::Vector3f synthetic_point(0,0,0);

#pragma unroll
    for (auto row = 0; row < kernel_size; ++row)
    {
        int row_ = i - shift + row;
#pragma unroll
        for (auto col = 0; col < kernel_size; ++col)
        {
            int col_ = j - shift + col;
            if (i == row_ && j == col_) continue;

            auto nbr_point = opc(row_, col_);
            float dist = (point - nbr_point).norm();

            // if dist> max_dist this indicantes this neighbor is very far away and should not be included in vertex sum
            // simply removing this neighbor leaves the smoothign "unbalanced"
            // basically the center point (i,j) will be disprorionately dragged to one side
            // a better strategy would be to try and find a 'good' replacement synthetic point
            // the below code just looks for an opposite nbr point e.g. (0,0) -> (0, 2)
            // a vector is constructed from the center point and added to the center point
            // This makes the function about 20% slower then just simply not adding the nbr
            // if (dist > max_dist)
            // {
            //     OppositePointK3(row, col, opp_row, opp_col);
            //     auto global_row = i - shift + opp_row;
            //     auto global_col = j - shift + opp_col;
            //     synthetic_point = opc(global_row, global_col);
            //     dist = (point - synthetic_point).norm();
            //     if (dist > max_dist) continue;
            //     weight = 1. / (2.0f * dist + eps);
            //     nbr_point = (point - synthetic_point) + point;  
            // }
            // else
            // {
            //     weight = 1. / (dist + eps);
            // }

            // this is faster than above but leads to really small triangles near edges of surfaces
            if (dist > max_dist || std::isnan(dist)) continue;
            weight = 1.0f / (dist + eps);


            total_weight += weight;
            sum_vertex += weight * nbr_point;
        }
    }
    opc_out(i, j) = point + lambda * (sum_vertex / total_weight - point);
}

template <int kernel_size = 3>
inline void LaplacianLoopT(Eigen::Ref<RowMatrixXVec3f> opc, Eigen::Ref<RowMatrixXVec3f> opc_out,
                           const float lambda = OPF_KERNEL_DEFAULT_LAMBDA, float max_dist = OPF_KERNEL_MAX_FLOAT)
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
            SmoothPointT<kernel_size>(opc, opc_out, row, col, lambda, max_dist);
        }
    }
}

template <int kernel_size = 3>
inline RowMatrixXVec3f LaplacianT(Eigen::Ref<RowMatrixXVec3f> opc, float lambda = OPF_KERNEL_DEFAULT_LAMBDA,
                                  int iterations = OPF_KERNEL_DEFAULT_ITER, float max_dist = OPF_KERNEL_MAX_FLOAT)
{
    // TODO - Only really need to copy the ghost/halo cells on border
    RowMatrixXVec3f opc_out(opc);
    bool need_copy = false;
    for (int i = 0; i < iterations; ++i)
    {
        if (i % 2 == 0)
        {
            LaplacianLoopT<kernel_size>(opc, opc_out, lambda, max_dist);
            need_copy = false;
        }
        else
        {
            LaplacianLoopT<kernel_size>(opc_out, opc, lambda, max_dist);
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
