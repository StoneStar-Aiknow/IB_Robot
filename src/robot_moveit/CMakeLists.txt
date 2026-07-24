cmake_minimum_required(VERSION 3.10)
project(robot_moveit)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)


install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/moveit_gateway.py
    scripts/so101_placo_servo_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Pure-Python helper module imported by so101_placo_servo_node.py at runtime.
# Installed (not as a PROGRAM) so it sits next to the node in lib/<pkg> and is
# importable via the node's sys.path insertion.
install(
  FILES
    scripts/so101_placo_kinematics.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  # Register the pure-Python regression tests so `colcon test` runs them.
  # Disable plugin autoload because ROS launch_testing plugins are incompatible
  # with pytest 9.x in the project virtual environment.
  ament_add_pytest_test(
    test_moveit_gateway_motion_lifecycle
    test/test_moveit_gateway_motion_lifecycle.py
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
  )
  ament_add_pytest_test(
    test_so101_placo_home
    test/test_so101_placo_home.py
    TIMEOUT 60
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
  )
  ament_add_pytest_test(
    test_so101_ik_workers_launch
    test/test_so101_ik_workers_launch.py
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
  )
endif()

ament_package()
