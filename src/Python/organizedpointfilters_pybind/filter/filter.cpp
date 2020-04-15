#include "OrganizedPointFilters/Filter/Laplacian.hpp"
#include "OrganizedPointFilters/Filter/Normal.hpp"
#include "OrganizedPointFilters/Filter/Bilateral.hpp"

#include "organizedpointfilters_pybind/filter/filter.hpp"

using namespace OrganizedPointFilters;

void pybind_filter(py::module& m)
{
    py::module m_submodule = m.def_submodule("filter");

    m_submodule.def("laplacian", &Filter::Laplacian, "opc"_a, "_lambda"_a = OPF_KERNEL_DEFAULT_LAMBDA,
                    "iterations"_a = OPF_KERNEL_DEFAULT_ITER, "kernel_size"_a = OPF_KERNEL_DEFAULT_KERNEL_SIZE,
                    "Performs Laplacian Smoothing on Organized Point Cloud");
    m_submodule.def("laplacian",
                    [](RowMatrixXVec3f& a, float lambda, int iterations, int kernel_size) {
                        return Filter::Laplacian(a, lambda, iterations, kernel_size);
                    },
                    "opc"_a, "_lambda"_a = OPF_KERNEL_DEFAULT_LAMBDA, "iterations"_a = OPF_KERNEL_DEFAULT_ITER,
                    "kernel_size"_a = OPF_KERNEL_DEFAULT_KERNEL_SIZE,
                    "Performs Laplacian Smoothing on Organized Point Cloud");

    // Templated (Faster) Laplacian
    m_submodule.def("laplacian_K3", &Filter::LaplacianT<3>, "opc"_a, "_lambda"_a = OPF_KERNEL_DEFAULT_LAMBDA,
                    "iterations"_a = OPF_KERNEL_DEFAULT_ITER, "max_dist"_a = OPF_KERNEL_MAX_FLOAT,
                    "Performs Laplacian Smoothing w/ a Filter Size of 3 on an Organized Point Cloud");

    m_submodule.def("laplacian_K5", &Filter::LaplacianT<5>, "opc"_a, "_lambda"_a = OPF_KERNEL_DEFAULT_LAMBDA,
                    "iterations"_a = OPF_KERNEL_DEFAULT_ITER, "max_dist"_a = OPF_KERNEL_MAX_FLOAT,
                    "Performs Laplacian Smoothing w/ a Filter Size of 5 on an Organized Point Cloud");

    // Normal Computation
    m_submodule.def("compute_normals", &Filter::ComputeNormals, "opc"_a,
                    "Computes Normals for an Organized Point Cloud");
    m_submodule.def("compute_normals_and_centroids", &Filter::ComputeNormalsAndCentroids, "opc"_a,
                    "Computes Normals and Centroids for an Organized Point Cloud");

    // Bilateral Filtering
    m_submodule.def("bilateral_K3", &Filter::BilateralFilterNormals<3>, "opc"_a,
                    "iterations"_a = OPF_BILATERAL_DEFAULT_ITER, "sigma_length"_a = OPF_BILATERAL_DEFAULT_SIGMA_LENGTH,
                    "sigma_angle"_a = OPF_BILATERAL_DEFAULT_SIGMA_ANGLE,
                    "Performs Bilateral Smoothing w/ a Filter Size of 3 on an Organized Point Cloud Normals");


    // m.def("create_tri_mesh_copy", py::overload_cast<Eigen::Ref<RowMatrixXVec3f>, Eigen::Ref<RowMatrixXVec3f>,
    // >(&MeshHelper::CreateTriMeshCopy),
    //       "Creates a copy of a tri mesh, triangles of int dtype", "vertices"_a, "triangles"_a,
    //       "calc_normals"_a=PL_DEFAULT_CALC_NORMALS);
}