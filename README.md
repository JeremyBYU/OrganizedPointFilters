# Organized Point Filters

For organized point clouds as well as images (depth images).

## Benchmarks

### C++

Benchmarks C++ Code. RealSense Image Filters, OPC filters, etc.

1. `./cmake-build/bin/run-bench --benchmark_filter=Laplacian --benchmark_repetitions=3 --benchmark_report_aggregates_only=true` 

### Python

Benchmarks CUDA Kernels

1. `pytest bench/Python/test_cuda.py --benchmark-sort=name`
