# Set the Source, Header Files
set(ORGANIZEDPOINTFILTERS_SRC "${CMAKE_CURRENT_SOURCE_DIR}")
set(INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(ORGANIZEDPOINTFILTERS_INCLUDE_DIR "${INCLUDE_DIR}/OrganizedPointFilters")
set(ORGANIZEDPOINTFILTERS_SOURCE_FILES ${ORGANIZEDPOINTFILTERS_SRC}/OrganizedPointFilters.cpp ${ORGANIZEDPOINTFILTERS_SRC}/SpatialFilter.cpp
                                        ${ORGANIZEDPOINTFILTERS_SRC}/Helper.cpp)
set(ORGANIZEDPOINTFILTERS_HEADER_FILES ${ORGANIZEDPOINTFILTERS_INCLUDE_DIR}/OrganizedPointFilters.hpp ${ORGANIZEDPOINTFILTERS_INCLUDE_DIR}/Helper.hpp
                                        ${ORGANIZEDPOINTFILTERS_INCLUDE_DIR}/Types.hpp ${ORGANIZEDPOINTFILTERS_INCLUDE_DIR}/RS/SpatialFilter.hpp
                                        ${ORGANIZEDPOINTFILTERS_INCLUDE_DIR}/RS/DisparityTransform.hpp)

# Create the CPP Library
set(LIB_TYPE STATIC)
add_library(${PROJECT_NAME} ${LIB_TYPE} ${ORGANIZEDPOINTFILTERS_SOURCE_FILES})
add_library(OrganizedPointFilters::OPF ALIAS ${PROJECT_NAME})

# Set Compile Options for ONLY this target.
target_compile_options(${PROJECT_NAME}  PRIVATE
$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
${DESIRED_WARNINGS}>
$<$<CXX_COMPILER_ID:MSVC>:
/W4>)

target_link_libraries(${PROJECT_NAME} PRIVATE eigen)

# OPTION Build with OpenMP Support
if (OPF_WITH_OPENMP)
  find_package(OpenMP)
  target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

# Set Properties
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${ORGANIZEDPOINTFILTERS_HEADER_FILES}")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION} POSITION_INDEPENDENT_CODE ON)
# Set Public and Private Include Directories
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR})
# TODO - This is something that I still do not fully understand, Build vs Install?
target_include_directories(${PROJECT_NAME} PUBLIC
$<BUILD_INTERFACE:${INCLUDE_DIR}>
$<INSTALL_INTERFACE:include>
PRIVATE src)
