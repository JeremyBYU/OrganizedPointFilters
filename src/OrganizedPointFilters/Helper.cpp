#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <sys/stat.h>

#include "OrganizedPointFilters/OrganizedPointFilters.hpp"
#include "OrganizedPointFilters/Types.hpp"
#include "OrganizedPointFilters/Helper.hpp"

namespace OrganizedPointFilters {

namespace Helper {

// CONSOLE = spdlog::stdout_logger_st("console");
Z16_BUFFER get_depth_image(std::string fpath, int w, int h)
{
    // auto logger = spdlog::get("console");
    Z16_BUFFER img_data(w * h);
    struct stat results;
    size_t num_pixels = w * h;
    size_t num_bytes = num_pixels * 2;
    if (stat(fpath.c_str(), &results) == 0)
    {
        // std::cout << "size" << results.st_size << ", " << num_bytes  <<std::endl;
        // if (logger)
        // {
        //     logger->debug("Size is: {}; Expected: {}", results.st_size, num_bytes);
        // }
    }
    else
    {
        // if (logger)
        //     logger->error("Error occurred reading file", results.st_size, num_bytes);
    }
    std::ifstream myFile(fpath.c_str(), std::ios::in | std::ios::binary);
    // fread(img_data.data(), 2, num_pixels, myFile);
    myFile.read(reinterpret_cast<char*>(img_data.data()), num_bytes);
    return img_data;
}

std::tuple<ImgDetails, DepthInfo> get_depth_metadata(std::string fpath)
{
    // auto logger = spdlog::get("console");
    ImgDetails details;
    DepthInfo info{50.0021f, 425.849243f, 0.001f, 0};

    std::ifstream file(fpath.c_str());
    std::string str;
    int counter = 0;
    while (std::getline(file, str))
    {
        std::stringstream ss(str);
        std::vector<std::string> result;

        while (ss.good())
        {
            std::string substr;
            std::getline(ss, substr, ',');
            result.push_back(substr);
        }

        if (counter == 5)
        {
            details.w = std::stoi(result[1]);
        }
        else if (counter == 6)
        {
            details.h = std::stoi(result[1]);
        }
        else if (counter == 7)
        {
            details.bpp = std::stoi(result[1]);
        }
        else if (counter == 11)
        {
            info.focal_length_x_mm = std::stof(result[1]);
        }
        counter++;
    }
    return std::make_tuple(details, info);
}

} // namespace Helper

} // namespace OrganizedPointFilters