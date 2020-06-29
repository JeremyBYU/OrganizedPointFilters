# Organized Point Filters

This module is a collection of filters for use on **organized** point clouds (OPC).  It's very alpha-quality and should not really be used in production.  The filters:

* Laplacian Mesh Smoothing applied to an implicit fully connected right cut triangular mesh of an OPC.
    * Single threaded, CPU Multi-threaded, and GPU accelerated.
* Bilateral Mesh Normal Smoothing applied to an implicit fully connected right cut triangular mesh of an OPC.
    * Single threaded, CPU Multi-threaded, and GPU accelerated.
* Intel RealSense Bilateral Spatial and Disparity Transform filters used on depth images.
    * I thought it would be useful to pull this code out of the Intel SDK such that it can be used by others who are not using realsense cameras. Apache 2.0 License.

Here is an example GIF of Laplacian and Bilateral Filtering of a noisy organized pont cloud of stairs. The colors indicate the triangle normals of the mesh. The more uniform the colors, the smoother the surface

![Smoothing of OPC](https://raw.githubusercontent.com/JeremyBYU/OrganizedPointFilters/master/src_docs/_static/smoothing_example.gif)


## Documentation

Please see [documentation website](https://jeremybyu.github.io/OrganizedPointFilters/) for more details.





