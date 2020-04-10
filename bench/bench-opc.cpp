#include <iostream>
#include <sstream>
#include <vector>

#include <benchmark/benchmark.h>
#include "OrganizedPointFilters/Kernel/Kernel.hpp"
#include "OrganizedPointFilters/Types.hpp"

using namespace OrganizedPointFilters;

void InitRandom(Eigen::Ref<RowMatrixXVec3f> a)
{
    for (auto i = 0; i < a.rows(); ++i)
    {
        for (auto j = 0; j < a.cols(); ++j)
        {
            a(i,j) = Eigen::Vector3f::Random();
        }
    }
}

static void BM_Laplacian(benchmark::State& st)
{
    RowMatrixXVec3f a(st.range(0), st.range(0));
    InitRandom(a);
    // std::cout << a.rows() << std::endl;
    int iterations = 5;
    int kernel_size = 3;
    int lambda = 0.5;
    for (auto _ : st)
    {
        auto result = Kernel::Laplacian(a, lambda, iterations, kernel_size);
        benchmark::DoNotOptimize(result.data());
    }
}

static void BM_LaplacianT(benchmark::State& st)
{
    RowMatrixXVec3f a(st.range(0), st.range(0));
    InitRandom(a);
    // std::cout << a.rows() << std::endl;
    int iterations = 5;
    int kernel_size = 3;
    int lambda = 0.5;
    for (auto _ : st)
    {
        auto result = Kernel::LaplacianT<3>(a, lambda, iterations, 0.25);
        benchmark::DoNotOptimize(result.data());
    }
}

BENCHMARK(BM_Laplacian)->UseRealTime()->DenseRange(250, 500, 250)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_LaplacianT)->UseRealTime()->DenseRange(250, 500, 250)->Unit(benchmark::kMillisecond);
