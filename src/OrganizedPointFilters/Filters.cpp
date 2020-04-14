#include "OrganizedPointFilters/Filter/Bilateral.hpp"
#include "OrganizedPointFilters/Filter/Normal.hpp"

using namespace OrganizedPointFilters;

template <int kernel_size = 3>
OrganizedTriangleMatrix Filter::BilateralFilterNormals(Eigen::Ref<RowMatrixXVec3f> opc, int iterations,
                                                       float sigma_length, float sigma_angle)

{

    OrganizedTriangleMatrix normals;
    OrganizedTriangleMatrix centroids;

    float sigma_length_squared = 2.0f * sigma_length * sigma_length;
    float sigma_angle_squared = 2.0f * sigma_angle * sigma_angle;

    std::tie(normals, centroids) = ComputeNormalsAndCentroids(opc);
    // allocation
    // OrganizedTriangleMatrix new_normals(normals.rows(), normals.cols());
    OrganizedTriangleMatrix new_normals(normals);

    bool need_copy = false;
    for (int i = 0; i < iterations; ++i)
    {
        if (i % 2 == 0)
        {
            BilateralCore::BilateralNormalLoop<kernel_size>(normals, centroids, new_normals, sigma_length_squared,
                                                            sigma_angle_squared);
            need_copy = false;
        }
        else
        {
            BilateralCore::BilateralNormalLoop<kernel_size>(new_normals, centroids, normals, sigma_length_squared,
                                                            sigma_angle_squared);
            need_copy = true;
        }
    }

    if (need_copy)
    {
        new_normals = normals;
    }

    return new_normals;
}

OrganizedTriangleMatrix Filter::ComputeNormals(Eigen::Ref<RowMatrixXVec3f> opc)
{
    // Rows and cols in OPC (POINTS)
    const int rows_points = static_cast<int>(opc.rows());
    const int cols_points = static_cast<int>(opc.cols());

    const int rows_tris = rows_points - 1;
    const int cols_tris = cols_points - 1;

    OrganizedTriangleMatrix normals(rows_tris, cols_tris);

#if defined(_OPENMP)
    int num_threads = std::min(omp_get_max_threads(), OPF_NORMAL_OMP_MAX_THREAD);
    num_threads = std::max(num_threads, 1);
#pragma omp parallel for schedule(guided) num_threads(num_threads)
#endif
    for (int row_tri = 0; row_tri < rows_tris; ++row_tri)
    {
        for (int col_tri = 0; col_tri < cols_tris; ++col_tri)
        {
            NormalCore::ComputeNormal(opc, normals, row_tri, col_tri);
        }
    }

    return normals;
}

OrganizedTriangleMatrix Filter::ComputeCentroids(Eigen::Ref<RowMatrixXVec3f> opc)
{
    // Rows and cols in OPC (POINTS)
    const int rows_points = static_cast<int>(opc.rows());
    const int cols_points = static_cast<int>(opc.cols());

    const int rows_tris = rows_points - 1;
    const int cols_tris = cols_points - 1;

    OrganizedTriangleMatrix centroids(rows_tris, cols_tris);
#if defined(_OPENMP)
    int num_threads = std::min(omp_get_max_threads(), OPF_NORMAL_OMP_MAX_THREAD);
    num_threads = std::max(num_threads, 1);
#pragma omp parallel for schedule(guided) num_threads(num_threads)
#endif
    for (int row_tri = 0; row_tri < rows_tris; ++row_tri)
    {
        for (int col_tri = 0; col_tri < cols_tris; ++col_tri)
        {
            NormalCore::ComputeCentroid(opc, centroids, row_tri, col_tri);
        }
    }

    return centroids;
}

std::tuple<OrganizedTriangleMatrix, OrganizedTriangleMatrix>
Filter::ComputeNormalsAndCentroids(Eigen::Ref<RowMatrixXVec3f> opc)
{
    // Rows and cols in OPC (POINTS)
    const int rows_points = static_cast<int>(opc.rows());
    const int cols_points = static_cast<int>(opc.cols());

    const int rows_tris = rows_points - 1;
    const int cols_tris = cols_points - 1;

    OrganizedTriangleMatrix normals(rows_tris, cols_tris);
    OrganizedTriangleMatrix centroids(rows_tris, cols_tris);

#if defined(_OPENMP)
    int num_threads = std::min(omp_get_max_threads(), OPF_NORMAL_OMP_MAX_THREAD);
    num_threads = std::max(num_threads, 1);
#pragma omp parallel for schedule(guided) num_threads(num_threads)
#endif
    for (int row_tri = 0; row_tri < rows_tris; ++row_tri)
    {
        for (int col_tri = 0; col_tri < cols_tris; ++col_tri)
        {
            NormalCore::ComputeNormal(opc, normals, row_tri, col_tri);
            NormalCore::ComputeCentroid(opc, centroids, row_tri, col_tri);
        }
    }

    return std::make_tuple(std::move(normals), std::move(centroids));
}



template OrganizedTriangleMatrix Filter::BilateralFilterNormals<3>(Eigen::Ref<RowMatrixXVec3f> opc, int iterations,
                                                                float sigma_length, float sigma_angle);
