cmake_minimum_required(VERSION 3.18)

# Force CUDA architectures before any project(CUDA) call
set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA arch list" FORCE)

# Tell CMake exactly where to find nvcc
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "Path to nvcc" FORCE)
# Force CMake to use g++-11 for host compilation
set(CMAKE_CXX_COMPILER "/usr/bin/g++-11" CACHE FILEPATH "C++ compiler" FORCE)

project(particle_filter_localization LANGUAGES CXX CUDA)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -DTHRUST_DEBUG=1")

# Find CUDA and add include directories
include_directories(/usr/local/cuda/include)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} include)

# Define source file groups
set(PARTICLE_FILTER_SOURCES
        src/particle_filter/particle_filter.cu
        src/particle_filter/motion_model.cu
        src/particle_filter/measurement_model.cu
        src/particle_filter/resampling.cu
)

set(DATA_STRUCTURE_SOURCES
        src/data_structures/linked_list.cu
        src/data_structures/circular_queue.cu
)

# Main executable
add_executable(particle_filter_localization
        src/main.cpp
        ${PARTICLE_FILTER_SOURCES}
        ${DATA_STRUCTURE_SOURCES}
)

target_link_libraries(particle_filter_localization
        PRIVATE ${OpenCV_LIBS}
)

# Enable separable compilation for CUDA device calls
set_target_properties(particle_filter_localization PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# Add CUDA include directories to the target
target_include_directories(particle_filter_localization PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})