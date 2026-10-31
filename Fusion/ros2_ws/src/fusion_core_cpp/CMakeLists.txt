cmake_minimum_required(VERSION 3.16)
project(fusion_core_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  add_compile_options(/O2 /fp:fast /W3)
else()
  add_compile_options(-O3 -ffast-math -Wall -Wextra)
endif()

# 1. Standalone C++20 Fast Kernel Shared Library (usable from both rclcpp and Python ctypes)
add_library(fusion_fast_kernels SHARED
  src/fusion_fast_kernels.cpp
)
target_include_directories(fusion_fast_kernels PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# 2. Optional ROS 2 ament_cmake Composable Node build when rclcpp is present
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(rclcpp_components QUIET)
find_package(sensor_msgs QUIET)
find_package(nav_msgs QUIET)
find_package(fusion_interfaces QUIET)

if(ament_cmake_FOUND AND rclcpp_FOUND AND rclcpp_components_FOUND AND sensor_msgs_FOUND AND fusion_interfaces_FOUND)
  add_library(ouster_lidar_cpp_component SHARED
    src/fusion_composable_node.cpp
  )
  target_include_directories(ouster_lidar_cpp_component PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(ouster_lidar_cpp_component
    fusion_fast_kernels
  )
  ament_target_dependencies(ouster_lidar_cpp_component
    rclcpp
    rclcpp_components
    sensor_msgs
    nav_msgs
    fusion_interfaces
  )
  rclcpp_components_register_node(ouster_lidar_cpp_component
    PLUGIN "fusion_core_cpp::OusterLidarComposableNode"
    EXECUTABLE ouster_lidar_cpp_node
  )

  install(TARGETS fusion_fast_kernels ouster_lidar_cpp_component
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )
  install(DIRECTORY include/
    DESTINATION include
  )
  ament_package()
else()
  install(TARGETS fusion_fast_kernels
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )
endif()
