include(FetchContent)
# Put any LIBRARY dependencies in here
FetchContent_Declare(
  extern_pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.9.1
)

FetchContent_Declare(
  extern_eigen
  GIT_REPOSITORY https://github.com/eigenteam/eigen-git-mirror.git
  GIT_TAG        3.3.7
)

# After the following call, the library will be installed
FetchContent_MakeAvailable(extern_eigen extern_pybind11)