
cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)

# Using the vcpkg submodule
set(CMAKE_TOOLCHAIN_FILE third_party/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "")

# Default to Debug
if(NOT CMAKE_BUILD_TYPE)
    message("Build type not specified, using Debug")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type not specified, using Debug" FORCE)
endif(NOT CMAKE_BUILD_TYPE)

project(nbody LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -lineinfo --use-local-env")

include_directories(.)

add_executable(nbody
    common/nbody_plugin.cpp
    common/utils.cpp
    cpu/nbody_dummy.cpp
    cpu/nbody_openmp.cpp
    cpu/nbody_threadpool.cpp
    cuda/nbody_cuda.cu
    cuda/nbody_cuda_rsqrt.cu
    cuda/nbody_cuda_soa.cu
    cuda/nbody_cuda_vbo.cu
    cuda/nbody_cuda_tiled.cu
    cuda/nbody_cuda_draw.cu
    nbody.cpp)

set_property(TARGET nbody PROPERTY CUDA_ARCHITECTURES 75)

find_package(OpenGL REQUIRED)
target_link_libraries(nbody PRIVATE OpenGL::GL OpenGL::GLU)

find_package(GLEW REQUIRED)
target_link_libraries(nbody PRIVATE GLEW::GLEW)

find_package(OpenMP REQUIRED)
target_link_libraries(nbody PRIVATE OpenMP::OpenMP_CXX)

find_package(imgui CONFIG REQUIRED)
target_link_libraries(nbody PRIVATE imgui::imgui)

find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(nbody PRIVATE glfw)

find_package(argparse CONFIG REQUIRED)
target_link_libraries(nbody PRIVATE argparse::argparse)
