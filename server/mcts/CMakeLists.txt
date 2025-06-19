cmake_minimum_required(VERSION 3.14)
project(mcts_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

# Set CUDA
enable_language(CUDA)
find_package(CUDA REQUIRED)

# Add compile options
add_compile_options(-O3 -march=native -fopenmp)

# Include directories
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# Source files
set(SOURCES
    mcts.cpp
    kalah_game.cpp
    neural_network.cpp
    python_bindings.cpp
)

# Create Python module
pybind11_add_module(mcts_cpp ${SOURCES})

# Link libraries
target_link_libraries(mcts_cpp PRIVATE
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    Threads::Threads
    ${Python_LIBRARIES}
)

# Set properties
set_target_properties(mcts_cpp PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 14
    POSITION_INDEPENDENT_CODE ON
)

# Installation
install(TARGETS mcts_cpp DESTINATION .)
