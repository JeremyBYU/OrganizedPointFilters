#ifndef ORGANIZEDPOINTFILTERS_HELPER
#define ORGANIZEDPOINTFILTERS_HELPER

#include "OrganizedPointFilters/Types.hpp"

namespace OrganizedPointFilters {

namespace Helper {
std::tuple<ImgDetails, DepthInfo> get_depth_metadata(std::string fpath);
Z16_BUFFER get_depth_image(std::string fpath, int w = 848, int h = 480);
} // namespace Helper

} // namespace OrganizedPointFilters

#endif