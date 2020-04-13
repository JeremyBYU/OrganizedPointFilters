
#include "organizedpointfilters_pybind/organizedpointfilters_pybind.hpp"
#include "organizedpointfilters_pybind/docstring/docstring.hpp"
#include "organizedpointfilters_pybind/filter/filter.hpp"
#include "organizedpointfilters_pybind/types/bind_types.hpp"

using namespace OrganizedPointFilters;


PYBIND11_MODULE(organizedpointfilters, m)
{
    m.doc() = "Python binding of OrganizedPointFilters";

    // Will bind eigen matrix data types
    pybind_matrix_types(m);

    m.def("hello", &OrganizedPointFilters::Hello, "name"_a, "Says hello to name");
    docstring::FunctionDocInject(m, "hello", {{"name", "The name to say hello with"}});

    // Submodules
    pybind_filter(m);
}