#include <iostream>
#include <sstream>
#include <vector>

#include <benchmark/benchmark.h>
#include "OrganizedPointFilters/OrganizedPointFilters.hpp"
#include "OrganizedPointFilters/Types.hpp"
#include "OrganizedPointFilters/RS/RS.hpp"

const std::string DEFAULT_DEPTH_IMAGE_SMALL = "./fixtures/realsense/001_small_depth.raw";
const std::string DEFAULT_DEPTH_IMAGE_SMALL_MD = "./fixtures/realsense/001_small_depth_metadata.csv";

const std::string DEFAULT_DEPTH_IMAGE_LARGE = "./fixtures/realsense/001_big_depth.raw";
const std::string DEFAULT_DEPTH_IMAGE_LARGE_MD = "./fixtures/realsense/001_big_depth_metadata.csv";

using namespace OrganizedPointFilters;


class SmallImage : public benchmark::Fixture
{
  public:
    ImgDetails details;
    DepthInfo info;
    Z16_BUFFER data;
    std::vector<float> data_float;
    float alpha = 0.35, delta=10.0;
    uint8_t hole_filling_mode = 1;
    int iterations = 3;
    void SetUp(const ::benchmark::State& state)
    {
        std::tie(details, info) = Helper::get_depth_metadata(DEFAULT_DEPTH_IMAGE_SMALL_MD);
        data = Helper::get_depth_image(DEFAULT_DEPTH_IMAGE_SMALL, details.w, details.h);
        data_float = RS::disparity_transform<unsigned short, float>(data, details, info);
    }
};

class BigImage : public benchmark::Fixture
{
  public:
    ImgDetails details;
    DepthInfo info;
    Z16_BUFFER data;
    std::vector<float> data_float;
    float alpha = 0.35, delta=10.0;
    uint8_t hole_filling_mode = 1;
    int iterations = 3;
    void SetUp(const ::benchmark::State& state)
    {
        std::tie(details, info) = Helper::get_depth_metadata(DEFAULT_DEPTH_IMAGE_LARGE_MD);
        data = Helper::get_depth_image(DEFAULT_DEPTH_IMAGE_LARGE, details.w, details.h);
        data_float = RS::disparity_transform<unsigned short, float>(data, details, info);
    }
};



BENCHMARK_DEFINE_F(SmallImage, BM_RSDisparity)
(benchmark::State& st)
{
    for (auto _ : st)
    {
        auto disparity = RS::disparity_transform<unsigned short, float>(data, details, info);
    }
}

BENCHMARK_DEFINE_F(BigImage, BM_RSDisparity)
(benchmark::State& st)
{
    for (auto _ : st)
    {
        auto disparity = RS::disparity_transform<unsigned short, float>(data, details, info);
    }
}

BENCHMARK_DEFINE_F(SmallImage, BM_RSSpatialZ16)
(benchmark::State& st)
{
    for (auto _ : st)
    {
        auto rs_filtered_image = RS::spatial_filter(data, details, alpha, delta, iterations, hole_filling_mode);
    }
}

BENCHMARK_DEFINE_F(BigImage, BM_RSSpatialZ16)
(benchmark::State& st)
{
    for (auto _ : st)
    {
        auto rs_filtered_image = RS::spatial_filter(data, details, alpha, delta, iterations, hole_filling_mode);
    }
}

BENCHMARK_DEFINE_F(SmallImage, BM_RSSpatialFloat)
(benchmark::State& st)
{
    for (auto _ : st)
    {
        auto rs_filtered_image = RS::spatial_filter(data_float, details, alpha, delta, iterations, hole_filling_mode);
    }
}

BENCHMARK_DEFINE_F(BigImage, BM_RSSpatialFloat)
(benchmark::State& st)
{
    for (auto _ : st)
    {
        auto rs_filtered_image = RS::spatial_filter(data_float, details, alpha, delta, iterations, hole_filling_mode);
    }
}


BENCHMARK_REGISTER_F(SmallImage, BM_RSDisparity)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(BigImage, BM_RSDisparity)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(SmallImage, BM_RSSpatialZ16)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(BigImage, BM_RSSpatialZ16)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(SmallImage, BM_RSSpatialFloat)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(BigImage, BM_RSSpatialFloat)->UseRealTime()->Unit(benchmark::kMicrosecond);
