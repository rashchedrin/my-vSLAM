cmake_minimum_required(VERSION 3.6)
project(Orb_keypoints)

find_package( OpenCV REQUIRED )
set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(Orb_keypoints ${SOURCE_FILES})
target_link_libraries( Orb_keypoints ${OpenCV_LIBS} )