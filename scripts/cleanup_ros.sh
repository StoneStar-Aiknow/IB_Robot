#!/bin/bash
# Clean up ROS 2 processes, shared memory and residual daemon state

set -euo pipefail

# shellcheck disable=SC2059
log() { printf "[cleanup_ros] $1\n" "${@:2}"; }

log "Stopping ROS 2 launch parents..."
# 1. Graceful shutdown of launch/process group leaders first
pkill -SIGINT -f "ros2 launch" 2>/dev/null || true
pkill -SIGINT -f "move_group" 2>/dev/null || true
pkill -SIGINT -f "ign gazebo" 2>/dev/null || true
pkill -SIGINT -f "gz sim" 2>/dev/null || true
pkill -SIGINT -f "mujoco_ros2_control/ros2_control_node" 2>/dev/null || true
pkill -SIGINT -f "controller_manager/ros2_control_node" 2>/dev/null || true
sleep 2

log "Force killing remaining ROS 2 nodes..."
# 2. Force kill every known robot/ROS node class
pkill -9 -f "ros2 launch" 2>/dev/null || true
pkill -9 -f "move_group" 2>/dev/null || true
pkill -9 -f "rviz2" 2>/dev/null || true
pkill -9 -f "ign gazebo" 2>/dev/null || true
pkill -9 -f "gz sim" 2>/dev/null || true
pkill -9 -f "mujoco_ros2_control/ros2_control_node" 2>/dev/null || true
pkill -9 -f "controller_manager/ros2_control_node" 2>/dev/null || true
pkill -9 -f "robot_state_publisher" 2>/dev/null || true
pkill -9 -f "realsense2_camera" 2>/dev/null || true

# IB-Robot application nodes
pkill -9 -f "lerobot_policy_node" 2>/dev/null || true
pkill -9 -f "action_dispatcher_node" 2>/dev/null || true
pkill -9 -f "moveit_gateway" 2>/dev/null || true
pkill -9 -f "safety_guard_node" 2>/dev/null || true
pkill -9 -f "skill_executor_node" 2>/dev/null || true
pkill -9 -f "task_entry_node" 2>/dev/null || true
pkill -9 -f "vlm_task_planner_node" 2>/dev/null || true
pkill -9 -f "task_executor_node" 2>/dev/null || true
pkill -9 -f "perception_service_node" 2>/dev/null || true
pkill -9 -f "geometric_grasp_node" 2>/dev/null || true
pkill -9 -f "act_inference_node" 2>/dev/null || true

# MCP bridge (robot_mcp_server) — killed so Hermes will respawn from current config
pkill -9 -f "robot_mcp_server" 2>/dev/null || true

# Auxiliary TF/relay/camera helper nodes
pkill -9 -f "static_tf_" 2>/dev/null || true
pkill -9 -f "interactive_marker" 2>/dev/null || true
pkill -9 -f "so101_joint_current" 2>/dev/null || true
pkill -9 -f "camera_info_relay" 2>/dev/null || true
pkill -9 -f "depth_image_relay" 2>/dev/null || true
pkill -9 -f "aligned_depth" 2>/dev/null || true
pkill -9 -f "color_image_relay" 2>/dev/null || true
pkill -9 -f "front_color_image_relay" 2>/dev/null || true
pkill -9 -f "front_aligned" 2>/dev/null || true
sleep 1

# 3. Stop ROS 2 daemon so stale graph data is flushed
log "Stopping ROS 2 daemon..."
ros2 daemon stop 2>/dev/null || true

# 4. Clean shared memory (ROS 2 Humble uses /dev/shm)
log "Cleaning shared memory..."
rm -f /dev/shm/ros2_humble_* 2>/dev/null || true
rm -f /dev/shm/ros_humble_* 2>/dev/null || true

log "Done."
echo ""
echo "You can now run:"
echo "  source .shrc_local && export ROS_DOMAIN_ID=<your_id> && source install/setup.zsh && ros2 launch robot_config robot.launch.py ..."
