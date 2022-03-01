
#include "organizedpointfilters_pybind/organizedpointfilters_pybind.hpp"
#include "organizedpointfilters_pybind/docstring/docstring.hpp"
#include "organizedpointfilters_pybind/filter/filter.hpp"
#include "organizedpointfilters_pybind/types/bind_types.hpp"

using namespace OrganizedPointFilters;


PYBIND11_MODULE(organizedpointfilters_pybind, m)
{
    m.doc() = "Python binding of OrganizedPointFilters";

    // Will bind eigen matrix data types
    pybind_matrix_types(m);

    // Submodules
    pybind_filter(m);

    m.def("get_opf_version", &GetOrganizedPointFiltersVersion, "Get OrganizedPointFilters Version");
}