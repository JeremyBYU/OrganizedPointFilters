#include "OrganizedPointFilters/Filter/Laplacian.hpp"
#include "OrganizedPointFilters/Filter/Decimate.hpp"
#include "OrganizedPointFilters/Filter/Normal.hpp"
#include "OrganizedPointFilters/Filter/Bilateral.hpp"

#include "organizedpointfilters_pybind/filter/filter.hpp"
#include "organizedpointfilters_pybind/docstring/docstring.hpp"

using namespace OrganizedPointFilters;

void pybind_filter(py::module& m)
{
    py::module_ m_submodule = m.def_submodule("filter");

    // m_submodule.def("laplacian", &Filter::Laplacian, "opc"_a, "_lambda"_a = OPF_KERNEL_DEFAULT_LAMBDA,
    //                 "iterations"_a = OPF_KERNEL_DEFAULT_ITER, "kernel_size"_a = OPF_KERNEL_DEFAULT_KERNEL_SIZE,
    //                 "Performs Laplacian Smoothing on Organized Point Cloud");
    // m_submodule.def("laplacian",
    //                 [](RowMatrixXVec3f& a, float lambda, int iterations, int kernel_size) {
    //                     return Filter::Laplacian(a, lambda, iterations, kernel_size);
    //                 },
    //                 "opc"_a, "_lambda"_a = OPF_KERNEL_DEFAULT_LAMBDA, "iterations"_a = OPF_KERNEL_DEFAULT_ITER,
    //                 "kernel_size"_a = OPF_KERNEL_DEFAULT_KERNEL_SIZE,
    //                 "Performs Laplacian Smoothing on Organized Point Cloud");

    // Templated (Faster) Laplacian
    m_submodule.def("decimate_column_K2", &Filter::DecimateColumnT<2>, "opc"_a, "num_threads"_a = 1,
                    "Decimates an organized point cloud column wise");

    m_submodule.def("decimate_column_K3", &Filter::DecimateColumnT<3>, "opc"_a, "num_threads"_a = 1,
                    "Decimates an organized point cloud column wise");

    m_submodule.def("decimate_column_K4", &Filter::DecimateColumnT<4>, "opc"_a, "num_threads"_a = 1,
                    "Decimates an organized point cloud column wise");


    // Templated (Faster) Laplacian
    m_submodule.def("laplacian_K3", &Filter::LaplacianT<3>, "opc"_a, "_lambda"_a = OPF_KERNEL_DEFAULT_LAMBDA,
                    "iterations"_a = OPF_KERNEL_DEFAULT_ITER, "max_dist"_a = OPF_KERNEL_MAX_FLOAT,
                    "Performs Laplacian Smoothing w/ a Filter Size of 3 on an Organized Point Cloud");

    docstring::FunctionDocInject(m_submodule, "laplacian_K3", {
            {"opc", "Organized Point Cloud. M X N X 3 Eigen Matrix"},
            {"_lambda", "Weighting for each iteration update"},
            {"iterations", "Number of iterations"},
            {"max_dist", "Maximum distance a neighbor can be to be interagrated for smoothing."}
    });

    m_submodule.def("laplacian_K5", &Filter::LaplacianT<5>, "opc"_a, "_lambda"_a = OPF_KERNEL_DEFAULT_LAMBDA,
                    "iterations"_a = OPF_KERNEL_DEFAULT_ITER, "max_dist"_a = OPF_KERNEL_MAX_FLOAT,
                    "Performs Laplacian Smoothing w/ a Filter Size of 5 on an Organized Point Cloud");

    docstring::FunctionDocInject(m_submodule, "laplacian_K5", {
            {"opc", "Organized Point Cloud. M X N X 3 Eigen Matrix"},
            {"_lambda", "Weighting for each iteration update"},
            {"iterations", "Number of iterations"},
            {"max_dist", "Maximum distance a neighbor can be to be interagrated for smoothing."}
    });

    // Normal Computation
    m_submodule.def("compute_normals", &Filter::ComputeNormals, "opc"_a,
                    "Computes Normals for an Organized Point Cloud");

    docstring::FunctionDocInject(m_submodule, "compute_normals", {
            {"opc", "Organized Point Cloud. M X N X 3 Eigen Matrix"},
    });

    m_submodule.def("compute_normals_and_centroids", &Filter::ComputeNormalsAndCentroids, "opc"_a,
                    "Computes Normals and Centroids for an Organized Point Cloud");

    docstring::FunctionDocInject(m_submodule, "compute_normals_and_centroids", {
            {"opc", "Organized Point Cloud. M X N X 3 Eigen Matrix"},
    });

    // Bilateral Filtering
    m_submodule.def("bilateral_K3", &Filter::BilateralFilterNormals<3>, "opc"_a,
                    "iterations"_a = OPF_BILATERAL_DEFAULT_ITER, "sigma_length"_a = OPF_BILATERAL_DEFAULT_SIGMA_LENGTH,
                    "sigma_angle"_a = OPF_BILATERAL_DEFAULT_SIGMA_ANGLE,
                    "Performs Bilateral Smoothing w/ a Filter Size of 3 on an Organized Point Cloud Normals");

    docstring::FunctionDocInject(m_submodule, "bilateral_K3", {
            {"opc", "Organized Point Cloud. M X N X 3 Eigen Matrix"},
            {"iterations", "Number of iterations"},
            {"sigma_length", "The standard deviation for exponential decay based on centroid difference between neighboring triangles."},
            {"sigma_angle", "The standard deviation for exponential decay based on surface normal difference between between neighboring triangles."}
    });


}