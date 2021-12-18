cmake_minimum_required(VERSION 3.10)

project(cuda_tutorial)

add_subdirectory(../3rd/fmt-8.0.1 fmt.out EXCLUDE_FROM_ALL)

include_directories(../3rd/fmt-8.0.1/include)

add_executable(main main.cpp)
target_link_libraries(main fmt-header-only)