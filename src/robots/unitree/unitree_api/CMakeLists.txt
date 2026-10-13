cmake_minimum_required(VERSION 3.5)
project(unitree_api)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
        "msg/Request.msg"
        "msg/RequestHeader.msg"
        "msg/RequestIdentity.msg"
        "msg/RequestLease.msg"
        "msg/RequestPolicy.msg"
        "msg/Response.msg"
        "msg/ResponseHeader.msg"
        "msg/ResponseStatus.msg"

  DEPENDENCIES geometry_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Vendored upstream sources carry no copyright headers and stay
  # byte-identical to upstream (see PROVENANCE.md), so skip the
  # copyright/cpplint linters.
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
