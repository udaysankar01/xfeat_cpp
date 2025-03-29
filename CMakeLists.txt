cmake_minimum_required(VERSION 3.5)
project(xfeat_cpp)

# Build type:
# - Set to RELEASE to benchmark the code
# - Set to DEBUG to debug the code
set(CMAKE_BUILD_TYPE Debug)

# enable folder organization in IDE
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# for libtorch
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/thirdparty/pytorch/torch")

# find libraries
set(CMAKE_PREFIX_PATH "/usr/lib/libtorch")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# set compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

include_directories(
    ${OpenCV_INCLUDE_DIR}

    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/thirdparty/
)

set(THIRD_PARTY_LIBS
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

add_subdirectory(src)
add_subdirectory(examples)