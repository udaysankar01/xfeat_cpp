cmake_minimum_required(VERSION 3.22.1)
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
find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)

# libtorch config
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(
    ${EIGEN_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}

    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/thirdparty/
)

set(THIRD_PARTY_LIBS
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    ${TORCH_LIBRARIES}
)

add_subdirectory(src)
add_subdirectory(examples)