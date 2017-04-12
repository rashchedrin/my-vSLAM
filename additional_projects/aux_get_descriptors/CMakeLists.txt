cmake_minimum_required(VERSION 3.6)
project(aux_get_descriptors)

find_package( OpenCV REQUIRED )
set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(aux_get_descriptors ${SOURCE_FILES})
target_link_libraries( aux_get_descriptors ${OpenCV_LIBS} )