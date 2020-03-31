#include "OrganizedPointFilters/Kernel/Kernel.hpp"

#include "organizedpointfilters_pybind/kernel/kernel.hpp"

using namespace OrganizedPointFilters;

void pybind_kernel(py::module& m)
{
    py::module m_submodule = m.def_submodule("kernel");

    m_submodule.def("laplacian", &Kernel::Laplacian, "opc"_a, "opc_out"_a, "kernel_size"_a,
                    "Performs Laplacian Smoothing on Organized Point Cloud");

    m_submodule.def("laplacian", [] (RowMatrixXVec3f &a, RowMatrixXVec3f &b, int stride) {return Kernel::Laplacian(a, b, stride);}, "opc"_a, "opc_out"_a, "kernel_size"_a,
                    "Performs Laplacian Smoothing on Organized Point Cloud, pass");

    // m.def("create_tri_mesh_copy", py::overload_cast<Eigen::Ref<RowMatrixXVec3f>, Eigen::Ref<RowMatrixXVec3f>, >(&MeshHelper::CreateTriMeshCopy),
    //       "Creates a copy of a tri mesh, triangles of int dtype", "vertices"_a, "triangles"_a, "calc_normals"_a=PL_DEFAULT_CALC_NORMALS);
}