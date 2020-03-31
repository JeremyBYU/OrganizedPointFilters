// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved
//// In-place Domain Transform Edge-preserving filter based on
// http://inf.ufrgs.br/~eslgastal/DomainTransform/Gastal_Oliveira_SIGGRAPH2011_Domain_Transform.pdf
// The filter also allows to apply holes filling extention that due to implementation constrains can be applied
// horizontally only

#ifndef ORGANIZEDPOINTFILTERS_RS_SF
#define ORGANIZEDPOINTFILTERS_RS_SF

#include <map>
#include <vector>
#include <cmath>
#include <functional>

#include "OrganizedPointFilters/RS/SpatialFilter.hpp"
#include "OrganizedPointFilters/Types.hpp"

namespace OrganizedPointFilters {

namespace RS {

template <typename T>
std::vector<T> spatial_filter(std::vector<T>& source, ImgDetails& details, float spatial_alpha_param,
                              float spatial_edge_threshold, int spatial_iterations, uint8_t holes_filling_mode);

// std::vector<unsigned short> spatial_filter(std::vector<unsigned short> &source, ImgDetails &details, float
// spatial_alpha_param, float spatial_edge_threshold, int spatial_iterations, uint8_t holes_filling_mode);
template <typename T>
std::vector<T> prepare_target_frame(std::vector<T>& source, ImgDetails& details);
void recursive_filter_horizontal_fp(void* image_data, float alpha, float deltaZ, ImgDetails& details);
void recursive_filter_vertical_fp(void* image_data, float alpha, float deltaZ, ImgDetails& details);

template <typename T>
void recursive_filter_horizontal(void* image_data, float alpha, float deltaZ, uint8_t holes_filling_radius,
                                 ImgDetails& details)
{
    size_t v{}, u{};

    // Handle conversions for invalid input data
    bool fp = (std::is_floating_point<T>::value);

    auto _height = details.h;
    auto _width = details.w;

    // Filtering integer values requires round-up to the nearest discrete value
    const float round = fp ? 0.f : 0.5f;
    // define invalid inputs
    const T valid_threshold = fp ? static_cast<T>(std::numeric_limits<T>::epsilon()) : static_cast<T>(1);
    const T delta_z = static_cast<T>(deltaZ);

    auto image = reinterpret_cast<T*>(image_data);
    size_t cur_fill = 0;

    for (v = 0; v < _height; v++)
    {
        // left to right
        T* im = image + v * _width;
        T val0 = im[0];
        cur_fill = 0;

        for (u = 1; u < _width - 1; u++)
        {
            T val1 = im[1];

            if (fabs(val0) >= valid_threshold)
            {
                if (fabs(val1) >= valid_threshold)
                {
                    cur_fill = 0;
                    T diff = static_cast<T>(fabs(val1 - val0));

                    if (diff >= valid_threshold && diff <= delta_z)
                    {
                        float filtered = val1 * alpha + val0 * (1.0f - alpha);
                        val1 = static_cast<T>(filtered + round);
                        im[1] = val1;
                    }
                }
                else // Only the old value is valid - appy holes filling
                {
                    if (holes_filling_radius)
                    {
                        if (++cur_fill < holes_filling_radius) im[1] = val1 = val0;
                    }
                }
            }

            val0 = val1;
            im += 1;
        }

        // right to left
        im = image + (v + 1) * _width - 2; // end of row - two pixels
        T val1 = im[1];
        cur_fill = 0;

        for (u = _width - 1; u > 0; u--)
        {
            T val0 = im[0];

            if (val1 >= valid_threshold)
            {
                if (val0 > valid_threshold)
                {
                    cur_fill = 0;
                    T diff = static_cast<T>(fabs(val1 - val0));

                    if (diff <= delta_z)
                    {
                        float filtered = val0 * alpha + val1 * (1.0f - alpha);
                        val0 = static_cast<T>(filtered + round);
                        im[0] = val0;
                    }
                }
                else // 'inertial' hole filling
                {
                    if (holes_filling_radius)
                    {
                        if (++cur_fill < holes_filling_radius) im[0] = val0 = val1;
                    }
                }
            }

            val1 = val0;
            im -= 1;
        }
    }
}

template <typename T>
void recursive_filter_vertical(void* image_data, float alpha, float deltaZ, uint8_t holes_filling_radius,
                               ImgDetails& details)
{
    size_t v{}, u{};

    // Handle conversions for invalid input data
    bool fp = (std::is_floating_point<T>::value);

    auto _height = details.h;
    auto _width = details.w;

    // Filtering integer values requires round-up to the nearest discrete value
    const float round = fp ? 0.f : 0.5f;
    // define invalid range
    const T valid_threshold = fp ? static_cast<T>(std::numeric_limits<T>::epsilon()) : static_cast<T>(1);
    const T delta_z = static_cast<T>(deltaZ);

    auto image = reinterpret_cast<T*>(image_data);

    // we'll do one row at a time, top to bottom, then bottom to top

    // top to bottom

    T* im = image;
    T im0{};
    T imw{};
    for (v = 1; v < _height; v++)
    {
        for (u = 0; u < _width; u++)
        {
            im0 = im[0];
            imw = im[_width];

            // if ((fabs(im0) >= valid_threshold) && (fabs(imw) >= valid_threshold))
            {
                T diff = static_cast<T>(fabs(im0 - imw));
                if (diff < delta_z)
                {
                    float filtered = imw * alpha + im0 * (1.f - alpha);
                    im[_width] = static_cast<T>(filtered + round);
                }
            }
            im += 1;
        }
    }

    // bottom to top
    im = image + (_height - 2) * _width;
    for (v = 1; v < _height; v++, im -= (_width * 2))
    {
        for (u = 0; u < _width; u++)
        {
            im0 = im[0];
            imw = im[_width];

            if ((fabs(im0) >= valid_threshold) && (fabs(imw) >= valid_threshold))
            {
                T diff = static_cast<T>(fabs(im0 - imw));
                if (diff < delta_z)
                {
                    float filtered = im0 * alpha + imw * (1.f - alpha);
                    im[0] = static_cast<T>(filtered + round);
                }
            }
            im += 1;
        }
    }
}

template <typename T>
inline void intertial_holes_fill(T* image_data, uint8_t holes_filling_radius, ImgDetails& details)
{
    std::function<bool(T*)> fp_oper = [](T* ptr) { return !*((int*)ptr); };
    std::function<bool(T*)> uint_oper = [](T* ptr) { return !(*ptr); };
    auto empty = (std::is_floating_point<T>::value) ? fp_oper : uint_oper;

    auto _height = details.h;
    auto _width = details.w;

    size_t cur_fill = 0;

    T* p = image_data;
    for (int j = 0; j < _height; ++j)
    {
        ++p;
        cur_fill = 0;

        // Left to Right
        for (size_t i = 1; i < _width; ++i)
        {
            if (empty(p))
            {
                if (++cur_fill < holes_filling_radius) *p = *(p - 1);
            }
            else
                cur_fill = 0;

            ++p;
        }

        --p;
        cur_fill = 0;
        // Right to left
        for (size_t i = 1; i < _width; ++i)
        {
            if (empty(p))
            {
                if (++cur_fill < holes_filling_radius)
                {
                }
            }
            else
                cur_fill = 0;
            --p;
        }
        p += _width;
    }
}

template <typename T>
void dxf_smooth(void* frame_data, float alpha, float delta, int iterations, uint8_t holes_filling_mode,
                uint8_t holes_filling_radius, ImgDetails& details)
{
    static_assert((std::is_arithmetic<T>::value), "Spatial filter assumes numeric types");
    bool fp = (std::is_floating_point<T>::value);

    for (int i = 0; i < iterations; i++)
    {
        if (fp)
        {
            recursive_filter_horizontal_fp(frame_data, alpha, delta, details);
            recursive_filter_vertical_fp(frame_data, alpha, delta, details);
        }
        else
        {
            recursive_filter_horizontal<T>(frame_data, alpha, delta, holes_filling_radius, details);
            recursive_filter_vertical<T>(frame_data, alpha, delta, holes_filling_radius, details);
        }
    }

    // Disparity domain hole filling requires a second pass over the frame data
    // For depth domain a more efficient in-place hole filling is performed
    if (holes_filling_mode && fp) intertial_holes_fill<T>(static_cast<T*>(frame_data), holes_filling_radius, details);
}


} // namespace RS
} // namespace OrganizedPointFilters

#endif
