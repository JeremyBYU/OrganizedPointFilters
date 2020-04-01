#include "OrganizedPointFilters/Kernel/Kernel.hpp"

#include "organizedpointfilters_pybind/kernel/kernel.hpp"

using namespace OrganizedPointFilters;

void pybind_kernel(py::module& m)
{
    py::module m_submodule = m.def_submodule("kernel");

    m_submodule.def("laplacian", &Kernel::Laplacian, "opc"_a, "_lambda"_a = OPF_KERNEL_DEFAULT_LAMBDA,
                    "iterations"_a = OPF_KERNEL_DEFAULT_ITER, "kernel_size"_a = OPF_KERNEL_DEFAULT_KERNEL_SIZE,
                    "Performs Laplacian Smoothing on Organized Point Cloud");

    m_submodule.def("laplacian",
                    [](RowMatrixXVec3f& a, float lambda, int iterations, int kernel_size) {
                        return Kernel::Laplacian(a, lambda, iterations, kernel_size);
                    },
                    "opc"_a, "_lambda"_a = OPF_KERNEL_DEFAULT_LAMBDA, "iterations"_a = OPF_KERNEL_DEFAULT_ITER,
                    "kernel_size"_a = OPF_KERNEL_DEFAULT_KERNEL_SIZE,
                    "Performs Laplacian Smoothing on Organized Point Cloud");

    // m.def("create_tri_mesh_copy", py::overload_cast<Eigen::Ref<RowMatrixXVec3f>, Eigen::Ref<RowMatrixXVec3f>,
    // >(&MeshHelper::CreateTriMeshCopy),
    //       "Creates a copy of a tri mesh, triangles of int dtype", "vertices"_a, "triangles"_a,
    //       "calc_normals"_a=PL_DEFAULT_CALC_NORMALS);
}