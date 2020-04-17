#ifndef ORGANIZEDPOINTFILTERS_HELPER
#define ORGANIZEDPOINTFILTERS_HELPER
#include <chrono>

#include "OrganizedPointFilters/Types.hpp"

namespace OrganizedPointFilters {




namespace Helper {

class Timer
{
    typedef std::chrono::high_resolution_clock high_resolution_clock;
    typedef std::chrono::microseconds microseconds;

  public:
    explicit Timer(bool run = false):_start()
    {
        if (run) Reset();
    }
    void Reset() { _start = high_resolution_clock::now(); }
    microseconds Elapsed() const
    {
        return std::chrono::duration_cast<microseconds>(high_resolution_clock::now() - _start);
    }
    template <typename T, typename Traits>
    friend std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& out, const Timer& timer)
    {
        return out << timer.Elapsed().count();
    }

  private:
    high_resolution_clock::time_point _start;
};


std::tuple<ImgDetails, DepthInfo> get_depth_metadata(std::string fpath);
Z16_BUFFER get_depth_image(std::string fpath, int w = 848, int h = 480);
} // namespace Helper

} // namespace OrganizedPointFilters

#endif