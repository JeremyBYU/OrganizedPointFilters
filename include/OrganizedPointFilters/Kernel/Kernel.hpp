
#ifndef ORGANIZEDPOINTFILTERS_KERNEL
#define ORGANIZEDPOINTFILTERS_KERNEL

#include "OrganizedPointFilters/Types.hpp"
#include <iostream>

namespace OrganizedPointFilters {

namespace Kernel {

// inline void smooth_point(Eigen::Ref<RowMatrixXVec3f> opc, Eigen::Ref<RowMatrixXVec3f> opc_out, const int i, const int j,
//                          int kernel_size = 3)
// {
//     const int shift = static_cast<int>(kernel_size / 2);
//     double total_weight = 0;
//     for (auto row = i - shift; row < i + shift; ++row)
//     {
//         for (auto col = j - shift; col < j + shift; ++col)
//         {

//         }
//     }
// }

// template<typename kernel_size>
inline void Laplacian(Eigen::Ref<RowMatrixXVec3f> opc, Eigen::Ref<RowMatrixXVec3f> opc_out, int kernel_size = 3)
{
    auto rows = opc.rows();
    auto cols = opc.cols();

    std::cout << "Inside Kernel:" << opc.data() << std::endl;

    const int shift = static_cast<int>(kernel_size / 2);

    for (auto row = shift; row < rows - shift; ++row)
    {
        for (auto col = shift; col < cols - shift; ++col)
        {
            std::cout << "Point: "; 
            std::cout << opc(row, col) << std::endl;
            // smooth_point(opc, smooth_opc, row, col, kernel_size);
        }
    }
}

} // namespace Kernel
} // namespace OrganizedPointFilters

#endif
