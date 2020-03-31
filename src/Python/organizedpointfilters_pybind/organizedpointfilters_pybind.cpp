
#include "organizedpointfilters_pybind/organizedpointfilters_pybind.hpp"
#include "organizedpointfilters_pybind/docstring/docstring.hpp"

using namespace OrganizedPointFilters;

PYBIND11_MODULE(organizedpointfilters, m)
{
    m.doc() = "Python binding of OrganizedPointFilters";

    m.def("hello", &OrganizedPointFilters::Hello, "name"_a, "Says hello to name");
    docstring::FunctionDocInject(m, "hello", {{"name", "The name to say hello with"}});

}