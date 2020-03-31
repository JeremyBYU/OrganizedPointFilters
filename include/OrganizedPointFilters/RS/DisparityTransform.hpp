// Disparity transformation block is responsible to convert stereoscopic depth to disparity data
// and vice versa
// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#pragma once

#include "OrganizedPointFilters/Types.hpp"

namespace OrganizedPointFilters {
namespace RS {

template <typename Tin, typename Tout>
void disparity_convert(void* in_data, void* out_data, int width, int height, float d2d_convert_factor)
{
    static_assert((std::is_arithmetic<Tin>::value), "disparity transform requires numeric type for input data");
    static_assert((std::is_arithmetic<Tout>::value), "disparity transform requires numeric type for output data");

    auto in = reinterpret_cast<const Tin*>(in_data);
    auto out = reinterpret_cast<Tout*>(out_data);

    bool fp = (std::is_floating_point<Tin>::value);
    const float round = fp ? 0.5f : 0.f;

    float input{};
    // TODO SSE optimize
    for (auto i = 0; i < height; i++)
        for (auto j = 0; j < width; j++)
        {
            input = *in;
            if (std::isnormal(input))
                *out++ = static_cast<Tout>((d2d_convert_factor / input) + round);
            else
                *out++ = 0;
            in++;
        }
}

template <typename Tin, typename Tout>
std::vector<Tout> disparity_transform(std::vector<Tin>& source, ImgDetails& details, DepthInfo& info)
{
    size_t num_pixels = details.w * details.h;
    auto tgt = std::vector<Tout>(num_pixels);

    // Get conversion factor for disparity
    float stereo_baseline_meter = info.stereo_baseline_mm * .001f;
    float focal_length_meter = info.focal_length_x_mm * .001f;
    const uint8_t fractional_bits = 5;
    const uint8_t fractions = 1 << fractional_bits;
    info.d2d_convert_factor = (stereo_baseline_meter * info.focal_length_x_mm * fractions) / info.depth_units_m;
    // info.d2d_convert_factor = (stereo_baseline_meter * info.focal_length_x_mm) / info.depth_units_m;

    // update_transformation_profile(f);

    bool fp = (std::is_floating_point<Tin>::value);

    disparity_convert<Tin, Tout>(static_cast<void*>(source.data()), static_cast<void*>(tgt.data()), details.w,
                                 details.h, info.d2d_convert_factor);

    return tgt;
}

} // namespace RS
} // namespace img_filters