#ifndef ORGANIZEDPOINTFILTERS_TYPES
#define ORGANIZEDPOINTFILTERS_TYPES

#include <string>
#include <vector>
#include <tuple>
#include <limits>

#include "eigen3/Eigen/Dense"


namespace OrganizedPointFilters {

struct ImgDetails
{
    size_t h;
    size_t w;
    size_t bpp;
    size_t stride;
};

struct DepthInfo
{
    float stereo_baseline_mm;
    float focal_length_x_mm;
    float depth_units_m;
    float d2d_convert_factor;
};

using Z16_BUFFER = std::vector<unsigned short>;


template<typename T>
using RowMatrixXVec3X = Eigen::Matrix<Eigen::Matrix<T, 3, 1> , Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

typedef Eigen::Matrix<float, 2, 3, Eigen::RowMajor> EigenDoubleVector3f;
using OrganizedTriangleMatrix = Eigen::Matrix<EigenDoubleVector3f, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using RowMatrixXVec3f = Eigen::Matrix<Eigen::Vector3f, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixX3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

// std::tuple<ImgDetails, DepthInfo> get_depth_metadata(std::string fpath);
// Z16_BUFFER get_depth_image(std::string fpath, int w=848, int h=480);

} // namespace OrganizedPointFilters

#endif