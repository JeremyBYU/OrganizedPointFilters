#include "OrganizedPointFilters/Types.hpp"

#include "organizedpointfilters_pybind/types/bind_types.hpp"

using namespace OrganizedPointFilters;

// Convert Numpy Array to PolylidarMatrix
template <typename T>
Eigen::Ref<RowMatrixXVec3X<T>> py_array_to_matrix(py::array_t<T, py::array::c_style | py::array::forcecast> array)
{
    // return std::vector<std::array<T, dim>>();
    // std::cout << "Calling py_array_to_matrix" << std::endl;
    if (array.ndim() != 3)
    {
        throw py::cast_error(
            "Numpy array must have exactly 3 Dimensions to be transformed to RowMatrixXVec3X<T>, MXNX3");
    }
    if (array.shape(2) != 3)
    {
        throw py::cast_error("Numpy array last dimension must be 3, e.g., MXNX3");
    }
    size_t rows = array.shape(0);
    size_t cols = array.shape(1);
    size_t channels = array.shape(2);

    auto info = array.request();
    auto ptr = reinterpret_cast<Eigen::Matrix<T, 3, 1>*>(info.ptr);
    auto new_matrix = Eigen::Map<RowMatrixXVec3X<T>>(ptr, rows, cols);
    return new_matrix;
}

template <typename T>
RowMatrixXVec3X<T> py_array_to_matrix_copy(py::array_t<T, py::array::c_style | py::array::forcecast> array)
{
    // return std::vector<std::array<T, dim>>();
    // std::cout << "Calling py_array_to_matrix" << std::endl;
    if (array.ndim() != 3)
    {
        throw py::cast_error(
            "Numpy array must have exactly 3 Dimensions to be transformed to RowMatrixXVec3X<T>, MXNX3");
    }
    if (array.shape(2) != 3)
    {
        throw py::cast_error("Numpy array last dimension must be 3, e.g., MXNX3");
    }
    size_t rows = array.shape(0);
    size_t cols = array.shape(1);
    size_t channels = array.shape(2);

    RowMatrixXVec3X<T> new_matrix(rows, cols);
    size_t element_counter = 0;
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            new_matrix(i, j) = {array.at(i, j, 0), array.at(i, j, 1), array.at(i, j, 2)};
        }
    }
    return new_matrix;
}

TriangleNormalMatrix py_array_to_triangle_matrix_copy(py::array_t<float, py::array::c_style | py::array::forcecast> array)
{
    // return std::vector<std::array<T, dim>>();
    // std::cout << "Calling py_array_to_matrix" << std::endl;
    if (array.ndim() != 4)
    {
        throw py::cast_error(
            "Numpy array must have exactly 4 Dimensions to be transformed to RowMatrixXVec3X<T>, MXNX2X3");
    }
    if (array.shape(3) != 3 || array.shape(2) != 2)
    {
        throw py::cast_error("Numpy array last dimension must be 3, e.g., MXNX2X3");
    }
    size_t rows = array.shape(0);
    size_t cols = array.shape(1);
    size_t triangles_per_cell = array.shape(2);
    size_t channels = array.shape(3);

    TriangleNormalMatrix new_matrix(rows, cols);
    return new_matrix;
}



void pybind_matrix_types(py::module& m)
{

    py::class_<Eigen::Ref<RowMatrixXVec3X<float>>>(m, "Matrix3fRef", py::buffer_protocol())
        .def(py::init<>(&py_array_to_matrix<float>), "points"_a)
        .def_buffer([](Eigen::Ref<RowMatrixXVec3X<float>>& m) -> py::buffer_info {
            size_t rows = m.rows();
            size_t cols = m.cols();
            size_t channels = 3UL;
            return py::buffer_info(m.data(),                               /* Pointer to buffer */
                                   sizeof(float),                          /* Size of one scalar */
                                   py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
                                   3UL,                                    /* Number of dimensions */
                                   {rows, cols, channels},                 /* Buffer dimensions */
                                   {sizeof(float) * cols * channels,       /* Strides (in bytes) for each index */
                                    sizeof(float) * channels, sizeof(float)});
        });

    py::class_<RowMatrixXVec3X<float>>(m, "Matrix3f", py::buffer_protocol())
        .def(py::init<>(&py_array_to_matrix_copy<float>), "points"_a)
        .def_buffer([](RowMatrixXVec3X<float>& m) -> py::buffer_info {
            size_t rows = m.rows();
            size_t cols = m.cols();
            size_t channels = 3UL;
            return py::buffer_info(m.data(),                               /* Pointer to buffer */
                                   sizeof(float),                          /* Size of one scalar */
                                   py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
                                   3UL,                                    /* Number of dimensions */
                                   {rows, cols, channels},                 /* Buffer dimensions */
                                   {sizeof(float) * cols * channels,       /* Strides (in bytes) for each index */
                                    sizeof(float) * channels, sizeof(float)});
        });

    py::class_<TriangleNormalMatrix>(m, "TriangleNormalMatrix", py::buffer_protocol())
        .def(py::init<>(&py_array_to_triangle_matrix_copy), "triangle_normals"_a)
        .def_buffer([](TriangleNormalMatrix& m) -> py::buffer_info {
            size_t rows = m.rows();
            size_t cols = m.cols();
            size_t tris_per_cell = 2UL;
            size_t channels = 3UL;
            return py::buffer_info(m.data(),                               /* Pointer to buffer */
                                   sizeof(float),                          /* Size of one scalar */
                                   py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
                                   4UL,                                    /* Number of dimensions */
                                   {rows, cols, tris_per_cell, channels},                 /* Buffer dimensions */
                                   {sizeof(float) * cols * channels * tris_per_cell,       /* Strides (in bytes) for each index */
                                   sizeof(float) * channels * tris_per_cell, 
                                    sizeof(float) * channels, sizeof(float)});
        });

}
