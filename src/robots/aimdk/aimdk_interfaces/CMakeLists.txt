cmake_minimum_required(VERSION 3.10)
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()
project(aimdk_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LocomotionProfile.msg"
  "srv/GetMotionState.srv"
  "srv/GetLocomotionProfiles.srv"
  "action/ExecuteSimulationPolicy.action"
  DEPENDENCIES action_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
