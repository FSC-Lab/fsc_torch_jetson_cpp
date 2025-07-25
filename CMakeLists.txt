cmake_minimum_required(VERSION 3.15)
project(cppflow_test VERSION 1.0.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
# append cmake to module path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
# add custom utility functions
# include(utils)

# find Eigen lib
find_package(Eigen3 CONFIG REQUIRED)

if (Eigen3_FOUND)
    message(STATUS "eigen3 found")
endif()

# find tensorflow lib
# find_package(tensorflow REQUIRED)

# if (tensorflow_FOUND)
#     message(STATUS "tensorflow found")
# endif()

# set compile options
string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE_)
if (${BUILD_TYPE_} STREQUAL "debug")
    message(STATUS "build type is debug...")
    # -Werror is not used !
    set(COMPILE_FLAGS "-Wall;-Wextra;-O0;-g;-fno-omit-frame-pointer;-fsanitize=address")
    set(LASAN -lasan)
elseif(${BUILD_TYPE_} STREQUAL "release")
    message(STATUS "build type is release...")
    set(COMPILE_FLAGS "-O3")
    set(LASAN "") # no lasan lib is add in release mode
else()
    # if the build type does not match debug or release, throw an error
    message(FATAL_ERROR "Invalid build type: " ${CMAKE_BUILD_TYPE})
endif()
# find tensorflow lib
# find_package(tensorflow REQUIRED)

# if (tensorflow_FOUND)
#     message(STATUS "tensorflow found")
# endif()

# set compile options
string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE_)
if (${BUILD_TYPE_} STREQUAL "debug")
    message(STATUS "build type is debug...")
    # -Werror is not used !
    set(COMPILE_FLAGS "-Wall;-Wextra;-O0;-g;-fno-omit-frame-pointer;-fsanitize=address")
    set(LASAN -lasan)
elseif(${BUILD_TYPE_} STREQUAL "release")
    message(STATUS "build type is release...")
    set(COMPILE_FLAGS "-O3")
    set(LASAN "") # no lasan lib is add in release mode
else()
    # if the build type does not match debug or release, throw an error
    message(FATAL_ERROR "Invalid build type: " ${CMAKE_BUILD_TYPE})
endif()


# # add cpp_flow_dir
# set(CPP_FLOW_HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cppflow_v3/include)

# message(${CPP_FLOW_HEADER_DIR})


# need to manualy add torch cmake file to the prefix path so that find_package can function
# properly
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/miniconda3/envs/py39/lib/python3.9/site-packages/torch/share/cmake")
find_package(Torch REQUIRED)
if (Torch_FOUND)
    message(STATUS "Torch found")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# # add src directory
add_subdirectory(src)