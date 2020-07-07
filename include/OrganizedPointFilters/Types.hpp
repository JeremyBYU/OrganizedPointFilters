#ifndef ORGANIZEDPOINTFILTERS_TYPES
#define ORGANIZEDPOINTFILTERS_TYPES

#include <string>
#include <vector>
#include <tuple>
#include <limits>

// #define EIGEN_RUNTIME_NO_MALLOC // Define this symbol to enable runtime tests for allocations
#include "Eigen/Dense"


namespace OrganizedPointFilters {

/**
 * @brief Provides details ona range image
 * This all needs to be fleshed out much better later
 */
struct ImgDetails
{
    /** @brief height of image  */
    size_t h;
    /** @brief width of image  */
    size_t w;
    /** @brief bytes per pixel  */
    size_t bpp;
    /** @brief stride of image if downsampling  */
    size_t stride;
};


/**
 * @brief Provides details on a camera sensor for a depth image
 * This all needs to be fleshed out much better later
 * 
 */
struct DepthInfo
{
    /** @brief stereo baseline in mm  */
    float stereo_baseline_mm;
    /** @brief focal legnth baseline in mm  */
    float focal_length_x_mm;
    /** @brief Depth unit in meters (0.001)  */
    float depth_units_m;
    /** @brief depth to disparity convert factor  */
    float d2d_convert_factor;
};

/** @brief Describes memory buffer for a range image in Z16 format  */
using Z16_BUFFER = std::vector<unsigned short>;


/** @brief Eigen Matrix (NX3) that is Row Major  */
template<typename T>
using RowMatrixXVec3X = Eigen::Matrix<Eigen::Matrix<T, 3, 1> , Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/** @brief 2 X 3 Eigen Matrix, Row Major. Think of it as two triangle (rows), with each triangle having an R^3 unit normal (3 columns)*/
typedef Eigen::Matrix<float, 2, 3, Eigen::RowMajor> EigenDoubleVector3f;
/** @brief M X N X 2 X 3 Eigen Matrix where M = rows and N=columns, Row Major.*/
using OrganizedTriangleMatrix = Eigen::Matrix<EigenDoubleVector3f, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/** @brief N X N X 3 Eigen Matrix in Row Major. This is basically the organized point cloud*/
using RowMatrixXVec3f = Eigen::Matrix<Eigen::Vector3f, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
/** @brief N X N Eigen Matrix in Row Major. Float image*/
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/** @brief N X 3 Eigen Matrix in Row Major.*/
// using RowMatrixX3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

// std::tuple<ImgDetails, DepthInfo> get_depth_metadata(std::string fpath);
// Z16_BUFFER get_depth_image(std::string fpath, int w=848, int h=480);

} // namespace OrganizedPointFilters

#endif