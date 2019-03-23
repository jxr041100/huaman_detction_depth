cmake_minimum_required(VERSION 2.8.8)

# ---[ Solution name
project(DepthHuman)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")

# ---[ find dependencies

find_package(OpenCV)



include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

#include_directories(pcl_util/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})


file(GLOB SOURCES "main.cpp")
add_executable(DepthHuman ${SOURCES})
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
	)
