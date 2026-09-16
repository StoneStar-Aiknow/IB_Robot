cmake_minimum_required(VERSION 3.10)
project(lekiwi_sdk)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

find_package(ament_cmake REQUIRED)
find_package(feetech_sdk REQUIRED)
find_package(so101_sdk REQUIRED)

add_library(${PROJECT_NAME} STATIC
  src/mobile_manipulator.cpp
  src/omni_base.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

ament_target_dependencies(${PROJECT_NAME} feetech_sdk so101_sdk)

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

ament_export_targets(export_${PROJECT_NAME})
ament_export_include_directories(include)
ament_export_dependencies(feetech_sdk so101_sdk)

# ---------------------------------------------------------------------------
# Python bindings (pybind11): exposes OmniBase for Python runtime nodes and tools.
# Required, not optional: lekiwi_robot's base node imports lekiwi_sdk_py
# unconditionally, so a build that silently skipped the bindings would succeed
# and then fail at launch with ModuleNotFoundError.
# ---------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
pybind11_add_module(lekiwi_sdk_py python/bindings.cpp)
target_link_libraries(lekiwi_sdk_py PRIVATE ${PROJECT_NAME})
target_include_directories(lekiwi_sdk_py PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
install(TARGETS lekiwi_sdk_py
  DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_omni_base test/test_omni_base.cpp)
  target_link_libraries(test_omni_base ${PROJECT_NAME})
  ament_add_gtest(test_mobile_manipulator test/test_mobile_manipulator.cpp)
  target_link_libraries(test_mobile_manipulator ${PROJECT_NAME})
endif()

ament_package()
