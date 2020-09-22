
#ifndef ORGANIZEDPOINTFILTERS_FILTER_DECIMATE
#define ORGANIZEDPOINTFILTERS_FILTER_DECIMATE
#include <iostream>

#include "OrganizedPointFilters/Types.hpp"

namespace OrganizedPointFilters {

namespace Filter {

namespace DecimateCore {

template <int kernel_size = 2>
inline void DecimateColumnPointT(Eigen::Ref<RowMatrixXVec3f> opc, Eigen::Ref<RowMatrixXVec3f> opc_out,
                                 const int out_row, const int out_col)
{
    // std::cout << "out_row and out_col: " << out_row << ", " << out_col << std::endl;
    Eigen::Vector3f sum_point(0, 0, 0);
    const int in_col = out_col * kernel_size;
    for (int n = 0; n < kernel_size; ++n)
    {
        auto& nbr_point = opc(out_row, in_col + n);
        sum_point += nbr_point;
    }
    auto& out_point = opc_out(out_row, out_col);
    out_point = sum_point / (float) kernel_size;
}

template <int kernel_size = 2>
void DecimateColumnTLoop(Eigen::Ref<RowMatrixXVec3f> opc, Eigen::Ref<RowMatrixXVec3f> opc_out,
                                    int num_threads = 1)
{
    const int rows = static_cast<int>(opc.rows());
    const int cols = static_cast<int>(opc.cols());
    const int real_height = rows;
    const int real_width = cols / kernel_size;
#if defined(_OPENMP)
#pragma omp parallel for schedule(guided) num_threads(num_threads)
#endif
    for (int out_row = 0; out_row < real_height; ++out_row)
    {
        for (int out_col = 0; out_col < real_width; ++out_col)
        {
            DecimateColumnPointT<kernel_size>(opc, opc_out, out_row, out_col);
        }
    }
}

} // namespace DecimateCore

/**
 * @brief Decimate filtering to an organized point cloud (OPC).
 *
 * @tparam 2                    Kernel size. Increasing the kernel size will make make the OPC smoother, but will lose data
 * costs.
 * @param opc                   Organized Point Cloud. M X N X 3 Eigen Matrix
 * @return RowMatrixXVec3f      M X (N/kenrel_size) X 3. Smoothed Points.
 */
template <int kernel_size = 2>
RowMatrixXVec3f DecimateColumnT(Eigen::Ref<RowMatrixXVec3f> opc, int num_threads = 1);
} // namespace Filter
} // namespace OrganizedPointFilters
#endif