cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(model_push)

find_packge(catkin REQUIRED COMPONENTS
  roscpp
  gazebo_ros
)

find_package(gazebo REQUIRED)

link_directories(${GAZEBO_LIBRARY_DIRS})
include_directories(${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})


catkin_package(
  DEPENDS
    roscpp
    gazebo_ros
)
