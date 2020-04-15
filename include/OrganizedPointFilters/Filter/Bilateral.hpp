
#ifndef ORGANIZEDPOINTFILTERS_FILTER_BILATERAL
#define ORGANIZEDPOINTFILTERS_FILTER_BILATERAL
#include <iostream>

#include "OrganizedPointFilters/Types.hpp"
#include "OrganizedPointFilters/Filter/Normal.hpp"

#include "FastExp/fastexp.h"

#define OPF_BILATERAL_DEFAULT_ITER 1
#define OPF_BILATERAL_DEFAULT_SIGMA_LENGTH 0.1f  // 10 centimeters
// Technically a default should be scaling by distance of unit normals, but angle (rad) and dist are similar for <90 degrees
// Not: user can pass in parameter that is a better scaling parameter for Gaussian Weight
#define OPF_BILATERAL_DEFAULT_SIGMA_ANGLE 0.261f // 15 degrees
#define OPF_BILATERAL_OMP_MAX_THREAD 16

namespace OrganizedPointFilters {

namespace Filter {

namespace BilateralCore {
// inline float GaussianWeight(float value, float sigma_squared) { return std::exp(-value / sigma_squared); }

// inline float GaussianWeight(float value, float sigma_squared) { return 1.0f / value; }

inline float GaussianWeight(float value, float sigma_squared)
{
    return fastexp::exp<float, fastexp::Product, 10>(-(value * value) / sigma_squared);
}

inline void IntegrateTriangle(Eigen::Block<EigenDoubleVector3f, 1, 3>& normal,
                              Eigen::Block<EigenDoubleVector3f, 1, 3>& centroid,
                              Eigen::Block<EigenDoubleVector3f, 1, 3>& nbr_normal,
                              Eigen::Block<EigenDoubleVector3f, 1, 3>& nbr_centroid, float& total_weight,
                              Eigen::Vector3f& sum_normal, float& sas, float& sls)
{
    auto normal_dist = (nbr_normal - normal).norm();
    auto centroid_dist = (nbr_centroid - centroid).norm();

    auto weight = GaussianWeight(normal_dist, sas) * GaussianWeight(centroid_dist, sls);
    total_weight += weight;
    sum_normal += weight * nbr_normal;
}

// Note loop unrolling is actually WORSE. Dont unroll loops here
template <int kernel_size = 3>
inline void SmoothNormal(Eigen::Ref<OrganizedTriangleMatrix> normals_in, Eigen::Ref<OrganizedTriangleMatrix> centroids,
                         Eigen::Ref<OrganizedTriangleMatrix> normals_out, int i, int j, float sls, float sas)
{
    constexpr int shift = static_cast<const int>(kernel_size / 2);

    float first_total_weight = 0.0;
    float second_total_weight = 0.0;
    // These are 2X3 matrices, each cell has two triangles
    auto& both_normals = normals_in(i, j);
    auto& both_centroids = centroids(i, j);

    auto first_normal = both_normals.block<1, 3>(0, 0);
    auto second_normal = both_normals.block<1, 3>(1, 0);

    auto first_centroid = both_centroids.block<1, 3>(0, 0);
    auto second_centroid = both_centroids.block<1, 3>(1, 0);

    // This will store the final normals
    // Wondering if we should just merge the normals together
    // meaning we average in the beginning and only have to do one normal integration
    Eigen::Vector3f first_sum_normal(0, 0, 0);
    Eigen::Vector3f second_sum_normal(0, 0, 0);

    // float first_weight = 0.0f;
    // float second_weight = 0.0f;

    // float temp_weight = 0.0f;

    for (auto row = 0; row < kernel_size; ++row)
    {
        int row_ = i - shift + row;
        for (auto col = 0; col < kernel_size; ++col)
        {
            int col_ = j - shift + col;
            if (i == row_ && j == col_) continue;

            auto& nbr_normals = normals_in(row_, col_);
            auto& nbr_centroids = centroids(row_, col_);

            auto nbr_first_normal = nbr_normals.block<1, 3>(0, 0);
            auto nbr_second_normal = nbr_normals.block<1, 3>(1, 0);

            auto nbr_first_centroid = nbr_centroids.block<1, 3>(0, 0);
            auto nbr_second_centroid = nbr_centroids.block<1, 3>(1, 0);

            // First Triangle vs nbr first
            BilateralCore::IntegrateTriangle(first_normal, first_centroid, nbr_first_normal, nbr_first_centroid,
                                             first_total_weight, first_sum_normal, sas, sls);
            // First Triangle vs nbr second
            BilateralCore::IntegrateTriangle(first_normal, first_centroid, nbr_second_normal, nbr_second_centroid,
                                             first_total_weight, first_sum_normal, sas, sls);

            // Second Triangle vs nbr first
            BilateralCore::IntegrateTriangle(second_normal, second_centroid, nbr_first_normal, nbr_first_centroid,
                                             second_total_weight, second_sum_normal, sas, sls);
            // Second Triangle vs nbr second
            BilateralCore::IntegrateTriangle(second_normal, second_centroid, nbr_second_normal, nbr_second_centroid,
                                             second_total_weight, second_sum_normal, sas, sls);
        }
    }

    // Write the average normals into "normal_out"
    auto& both_normals_out = normals_out(i, j);
    auto first_normal_out = both_normals_out.block<1, 3>(0, 0);
    auto second_normal_out = both_normals_out.block<1, 3>(1, 0);

    first_normal_out = first_sum_normal / first_total_weight;
    second_normal_out = second_sum_normal / second_total_weight;
}

template <int kernel_size = 3>
inline void BilateralNormalLoop(Eigen::Ref<OrganizedTriangleMatrix> normals_in,
                                Eigen::Ref<OrganizedTriangleMatrix> centroids,
                                Eigen::Ref<OrganizedTriangleMatrix> normals_out, float sls, float sas)
{
    const int rows = static_cast<int>(normals_in.rows());
    const int cols = static_cast<int>(normals_in.cols());

    constexpr int shift = static_cast<const int>(kernel_size / 2);
    const int rows_max = rows - shift;
    const int cols_max = cols - shift;

#if defined(_OPENMP)
    int num_threads = std::min(omp_get_max_threads(), OPF_BILATERAL_OMP_MAX_THREAD);
    num_threads = std::max(num_threads, 1);
#pragma omp parallel for schedule(guided) num_threads(num_threads)
#endif
    for (int row = shift; row < rows_max; ++row)
    {
        for (int col = shift; col < cols_max; ++col)
        {
            SmoothNormal<kernel_size>(normals_in, centroids, normals_out, row, col, sls, sas);
        }
    }
}

} // namespace BilateralCore

template <int kernel_size = 3>
OrganizedTriangleMatrix BilateralFilterNormals(Eigen::Ref<RowMatrixXVec3f> opc,
                                               int iterations = OPF_BILATERAL_DEFAULT_ITER,
                                               float sigma_length = OPF_BILATERAL_DEFAULT_SIGMA_LENGTH,
                                               float sigma_angle = OPF_BILATERAL_DEFAULT_SIGMA_ANGLE);

// {

//     OrganizedTriangleMatrix normals;
//     OrganizedTriangleMatrix centroids;

//     float sigma_length_squared = 2.0f * sigma_length * sigma_length;
//     float sigma_angle_squared = 2.0f * sigma_angle * sigma_angle;

//     std::tie(normals, centroids) = ComputeNormalsAndCentroids(opc);
//     // allocation
//     // OrganizedTriangleMatrix new_normals(normals.rows(), normals.cols());
//     OrganizedTriangleMatrix new_normals(normals);

//     bool need_copy = false;
//     for (int i = 0; i < iterations; ++i)
//     {
//         if (i % 2 == 0)
//         {
//             BilateralCore::BilateralNormalLoop<kernel_size>(normals, centroids, new_normals, sigma_length_squared,
//                                              sigma_angle_squared);
//             need_copy = false;
//         }
//         else
//         {
//             BilateralCore::BilateralNormalLoop<kernel_size>(new_normals, centroids, normals, sigma_length_squared,
//                                              sigma_angle_squared);
//             need_copy = true;
//         }
//     }

//     if (need_copy)
//     {
//         new_normals = normals;
//     }

//     return new_normals;
// }

} // namespace Filter
} // namespace OrganizedPointFilters

#endif