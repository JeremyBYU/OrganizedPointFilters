############ Start Version Parsing ##################

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" ORGANIZEDPOINTFILTERS_VERSION_READ)
foreach(ver ${ORGANIZEDPOINTFILTERS_VERSION_READ})
    if (ver MATCHES "(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(ORGANIZEDPOINTFILTERS_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
string(CONCAT ORGANIZEDPOINTFILTERS_VERSION
    "${ORGANIZEDPOINTFILTERS_VERSION_MAJOR}"
    ".${ORGANIZEDPOINTFILTERS_VERSION_MINOR}"
    ".${ORGANIZEDPOINTFILTERS_VERSION_PATCH}"
)

set(PROJECT_VERSION "${ORGANIZEDPOINTFILTERS_VERSION}")

message("Version is ${PROJECT_VERSION}")

############# End Version Parsing ##################


add_subdirectory(OrganizedPointFilters)

if (OPF_BUILD_PYMODULE)
    add_subdirectory(Python)
endif ()
