cmake_minimum_required(VERSION 3.10)
project(aimdk_dataplane)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(nlohmann_json REQUIRED)

# The rules (no ROS, no vendor types): what the node applies and the tests check.
add_library(x2_cbf_core STATIC
  cbf/src/command_adapter.cpp cbf/src/distance_gradient.cpp cbf/src/proxy_model.cpp
  cbf/src/kinematics.cpp cbf/src/safety_filter.cpp)
target_include_directories(x2_cbf_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cbf/include)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # The real-time kernel must not inherit the workspace's default -O0 dev build.
  # Deliberately no fast-math: preserve finite checks and distance tolerances.
  target_compile_options(x2_cbf_core PRIVATE -O3)
endif()

add_library(aimdk_dataplane_core STATIC src/projection.cpp src/core.cpp src/cbf_safety.cpp)
target_include_directories(aimdk_dataplane_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(aimdk_dataplane_core nlohmann_json::nlohmann_json x2_cbf_core)
set_target_properties(aimdk_dataplane_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# One-shot offline diagnostics only: no ROS node or actuator connection.
add_executable(cbf_pose_probe tools/cbf_pose_probe.cpp)
target_link_libraries(cbf_pose_probe aimdk_dataplane_core)
install(TARGETS cbf_pose_probe DESTINATION lib/${PROJECT_NAME})

# aimdk_msgs comes from the AimDK vendor overlay, not from this repository. A
# workspace built without it (CI, hosts that never drive an X2) builds and tests
# the rules but not the node, as aimdk_robot imports the vendor IDL only at runtime.
find_package(aimdk_msgs QUIET)
if(aimdk_msgs_FOUND)
  find_package(rclcpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(ibrobot_msgs REQUIRED)
  add_executable(aimdk_dataplane src/dataplane_node.cpp)
  target_link_libraries(aimdk_dataplane aimdk_dataplane_core)
  ament_target_dependencies(aimdk_dataplane rclcpp std_msgs sensor_msgs geometry_msgs ibrobot_msgs aimdk_msgs)
  install(TARGETS aimdk_dataplane DESTINATION lib/${PROJECT_NAME})
else()
  message(WARNING "aimdk_msgs not found (AimDK overlay not sourced): the aimdk_dataplane node is not built")
endif()

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_projection test/test_projection.cpp)
  target_link_libraries(test_projection aimdk_dataplane_core)
  ament_add_gtest(test_core test/test_core.cpp)
  target_link_libraries(test_core aimdk_dataplane_core)
  ament_add_gtest(test_cbf_kernel test/test_cbf_kernel.cpp)
  target_link_libraries(test_cbf_kernel x2_cbf_core)
  ament_add_gtest(test_cbf_safety test/test_cbf_safety.cpp)
  target_link_libraries(test_cbf_safety aimdk_dataplane_core)
endif()

ament_package()
