cmake_minimum_required(VERSION 3.3)
project(matrix_multiplication)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

set(SOURCE_FILES main.cpp)
add_executable(matrix_multiplication ${SOURCE_FILES})