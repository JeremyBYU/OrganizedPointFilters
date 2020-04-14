#include <iostream>
#include <sstream>
#include <vector>

#include <benchmark/benchmark.h>
#include "OrganizedPointFilters/Filter/Laplacian.hpp"
#include "OrganizedPointFilters/Filter/Normal.hpp"
#include "OrganizedPointFilters/Filter/Bilateral.hpp"
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
    float lambda = 1.0;
    for (auto _ : st)
    {
        auto result = Filter::Laplacian(a, lambda, iterations, kernel_size);
        benchmark::DoNotOptimize(result.data());
    }
}

static void BM_ComputeNormals(benchmark::State& st)
{
    RowMatrixXVec3f a(st.range(0), st.range(0));
    InitRandom(a);
    // std::cout << a.rows() << std::endl;
    for (auto _ : st)
    {
        auto result = Filter::ComputeNormals(a);
        benchmark::DoNotOptimize(result.data());
    }
}

static void BM_ComputeCentroid(benchmark::State& st)
{
    RowMatrixXVec3f a(st.range(0), st.range(0));
    InitRandom(a);
    // std::cout << a.rows() << std::endl;
    for (auto _ : st)
    {
        auto result = Filter::ComputeCentroids(a);
        benchmark::DoNotOptimize(result.data());
    }
}

static void BM_ComputeNormalsAndCentroids(benchmark::State& st)
{
    RowMatrixXVec3f a(st.range(0), st.range(0));
    InitRandom(a);
    // std::cout << a.rows() << std::endl;
    for (auto _ : st)
    {
        OrganizedTriangleMatrix normals;
        OrganizedTriangleMatrix centroid;
        std::tie(normals, centroid) = Filter::ComputeNormalsAndCentroids(a);
        benchmark::DoNotOptimize(normals.data());
    }
}

template<int kernel_size = 3>
static void BM_LaplacianT(benchmark::State& st)
{
    RowMatrixXVec3f a(st.range(0), st.range(0));
    InitRandom(a);
    // std::cout << a.rows() << std::endl;
    int iterations = 5;
    float lambda = 1.0;
    for (auto _ : st)
    {
        auto result = Filter::LaplacianT<kernel_size>(a, lambda, iterations, 0.25);
        benchmark::DoNotOptimize(result.data());
    }
}

// template<int kernel_size = 3>
static void BM_BilateralNormalFiltering(benchmark::State& st)
{
    RowMatrixXVec3f a(st.range(0), st.range(0));
    InitRandom(a);
    // std::cout << a.rows() << std::endl;
    int iterations = 5;
    for (auto _ : st)
    {
        auto result = Filter::BilateralFilterNormals<3>(a, iterations);
        benchmark::DoNotOptimize(result.data());
    }
}

BENCHMARK(BM_Laplacian)->UseRealTime()->DenseRange(250, 500, 250)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_LaplacianT, 3)->UseRealTime()->DenseRange(250, 500, 250)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_LaplacianT, 5)->UseRealTime()->DenseRange(250, 500, 250)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BilateralNormalFiltering)->UseRealTime()->DenseRange(250, 500, 250)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ComputeNormals)->UseRealTime()->DenseRange(250, 500, 250)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ComputeCentroid)->UseRealTime()->DenseRange(250, 500, 250)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ComputeNormalsAndCentroids)->UseRealTime()->DenseRange(250, 500, 250)->Unit(benchmark::kMillisecond);
