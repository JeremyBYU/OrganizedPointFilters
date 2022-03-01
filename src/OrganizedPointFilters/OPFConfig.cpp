#include "OPFConfig.hpp"
#include <string>

namespace OrganizedPointFilters {

std::string GetOrganizedPointFiltersVersion() { return std::string("OrganizedPointFilters ") + ORGANIZEDPOINTFILTERS_VERSION; }

} // namespace OrganizedPointFilters