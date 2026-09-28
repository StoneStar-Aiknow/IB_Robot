cmake_minimum_required(VERSION 3.10)
project(lekiwi_hardware)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-vla)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(feetech_sdk REQUIRED)
find_package(so101_sdk REQUIRED)

# ===== Hardware interface plugin (thin adapter over the SDKs) =====
add_library(lekiwi_hardware_interface SHARED
  src/lekiwi_system_hardware.cpp
)
target_include_directories(lekiwi_hardware_interface PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(lekiwi_hardware_interface
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  feetech_sdk
  so101_sdk
)

# Pluginlib export
pluginlib_export_plugin_description_file(hardware_interface lekiwi_hardware_plugin.xml)

# ===== Config Files =====
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# ===== Tools =====
# scan_motors is a bus diagnostic, rewritten on feetech_sdk after the vendored
# FTServo FetchContent block was removed.
add_executable(scan_motors tools/scan_motors.cpp)
ament_target_dependencies(scan_motors feetech_sdk)
install(TARGETS scan_motors DESTINATION lib/${PROJECT_NAME})

# ===== Testing =====
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Ruff is this repository's Python linter (see pyproject.toml): line-length 120
  # with E501 ignored, and no pydocstyle rules selected. ament_flake8 (99 columns)
  # and ament_pep257 (D rules) therefore fail permanently on conforming code, and
  # ament_copyright enforces a header convention the repository does not follow.
  # Skipping them here keeps `colcon test` output limited to real failures.
  set(ament_cmake_flake8_FOUND TRUE)
  set(ament_cmake_pep257_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  # cpplint's hook always runs with bare defaults: the legal/copyright filter,
  # which this repository has no convention for (3% of sources carry a header and
  # there is no LICENSE file), and a 100-column limit against the 120 configured
  # in pyproject.toml. It cannot be reconfigured through ament_lint_auto, so skip
  # it here as robot_moveit and robot_description already do. uncrustify and
  # cppcheck stay enabled and continue to check this C++.
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_lekiwi_conversions test/test_lekiwi_conversions.cpp)
  if(TARGET test_lekiwi_conversions)
    target_include_directories(test_lekiwi_conversions PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  endif()

  ament_add_gtest(test_lekiwi_lifecycle test/test_lekiwi_lifecycle.cpp)
  if(TARGET test_lekiwi_lifecycle)
    target_link_libraries(test_lekiwi_lifecycle lekiwi_hardware_interface)
    target_include_directories(test_lekiwi_lifecycle PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  endif()
endif()

# ===== Install =====
install(TARGETS lekiwi_hardware_interface
  EXPORT lekiwi_hardware_export
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

ament_export_include_directories(include)
ament_export_libraries(lekiwi_hardware_interface)
ament_export_dependencies(hardware_interface pluginlib rclcpp rclcpp_lifecycle feetech_sdk so101_sdk)
ament_package()
