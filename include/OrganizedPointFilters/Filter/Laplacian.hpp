
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

namespace Filter {

namespace LaplacianCore

{

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
__attribute__((optimize("unroll-loops"))) inline void
SmoothPointT(Eigen::Ref<RowMatrixXVec3f>& opc, Eigen::Ref<RowMatrixXVec3f>& opc_out, const int i, const int j,
             const float lambda = OPF_KERNEL_DEFAULT_LAMBDA, float max_dist = OPF_KERNEL_MAX_FLOAT)
{
    constexpr int shift = static_cast<const int>(kernel_size / 2);

    float total_weight = 0.0;
    auto& point = opc(i, j);
    Eigen::Vector3f sum_vertex(0, 0, 0);
    float weight = 0.0;

    // int opp_row = 0;
    // int opp_col = 0;
    Eigen::Vector3f synthetic_point(0, 0, 0);

    for (auto row = 0; row < kernel_size; ++row)
    {
        int row_ = i - shift + row;
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

    if (total_weight <= 0.0) return;

    opc_out(i, j) = point + lambda * (sum_vertex / total_weight - point);
}

template <int kernel_size = 3>
inline void LaplacianLoopT(Eigen::Ref<RowMatrixXVec3f> opc, Eigen::Ref<RowMatrixXVec3f> opc_out,
                           const float lambda = OPF_KERNEL_DEFAULT_LAMBDA, float max_dist = OPF_KERNEL_MAX_FLOAT)
{
    const int rows = static_cast<int>(opc.rows());
    const int cols = static_cast<int>(opc.cols());
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

} // namespace LaplacianCore

/**
 * @brief Laplacian filtering to an organized point cloud (OPC).
 * An implicit mesh is defined by OPC wherein each 2X2 quad of the OPC creates two right-cut triangles.
 * Will perform laplacain filtering for an organized point cloud and return *smoothed* triangle **vertices**.
 * 
 * O = Point
 *
 *                  O----------------------O
 *                  |                    XX|
 *                  |  TRI 0          XXX  |
 *                  |              XXXX    |
 *                  |            XXX       |
 *                  |         XXX          |
 *                  |       XXX            |
 *                  |     XXX       TRI 1  |
 *                  |   XXX                |
 *                  |XXX                   |
 *                  OX---------------------O
 * 
 * @tparam 3                    Kernel size. Increasing the kernel size will make smoother but have higher computation costs.
 * @param opc                   Organized Point Cloud. M X N X 3 Eigen Matrix
 * @param lambda                Weighting for each iteration update    
 * @param iterations            Number of iterations
 * @param max_dist              Maximum distance a neighbor can be to be interagrated for smoothing.
 * @return RowMatrixXVec3f      M X N X 3. Smoothed vertices.    
 */
template <int kernel_size = 3>
RowMatrixXVec3f LaplacianT(Eigen::Ref<RowMatrixXVec3f> opc, float lambda = OPF_KERNEL_DEFAULT_LAMBDA,
                                  int iterations = OPF_KERNEL_DEFAULT_ITER, float max_dist = OPF_KERNEL_MAX_FLOAT);

} // namespace Filter
} // namespace OrganizedPointFilters

#endif
