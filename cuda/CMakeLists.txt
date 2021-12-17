cmake_minimum_required(VERSION 3.10)

project(cuda_tutorial)

add_subdirectory(../3rd/fmt EXCLUDE_FROM_ALL)

add_executable(main main.cpp)
target_link_libraries(main fmt-header-only)