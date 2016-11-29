cmake_minimum_required (VERSION 2.8)

project(fydp)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(fydp_1 fydp.cpp)

target_link_libraries(fydp_1 ${OpenCV_LIBS})

