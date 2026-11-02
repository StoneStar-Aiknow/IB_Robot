cmake_minimum_required(VERSION 3.10)
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()
project(aimdk_grasp_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)

# The X2 grasp pipeline and its grasp head on another host (aimdk_grasp): an
# internal protocol of the X2's two-host deployment, not a robot-agnostic contract.
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GraspPlanStep.msg"
  "msg/GraspPlan.msg"
  "msg/WatchStatus.msg"
  "srv/CaptureGraspObservation.srv"
  "srv/CompleteGraspScene.srv"
  "srv/RefineGraspPlan.srv"
  "srv/WatchGraspTarget.srv"
  "action/PlanGraspCandidates.action"
  DEPENDENCIES action_msgs builtin_interfaces sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
