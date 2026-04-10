# TaskStep.msg - A single step in a task execution plan.
#
# Used by task_dispatch to describe sequential operations
# (move arm, control gripper, wait) in a hardware-agnostic way.
# Planners (visual_grasp, VoxPoser, etc.) build a sequence of these
# steps and send them as an ExecuteTaskPlan action goal.

# Step types
uint8 MOVE_TO_POSE = 0
uint8 GRIPPER = 1
uint8 WAIT = 2

uint8 type

# Human-readable label for logging and feedback
string label

# --- MOVE_TO_POSE fields ---
geometry_msgs/Pose target_pose
float64 velocity_scaling        # 0.0 = use default, otherwise 0.0-1.0

# --- GRIPPER fields ---
float64 gripper_position        # Normalized: 0.0 (closed) to 1.0 (open)

# --- WAIT fields ---
float64 wait_duration_s
