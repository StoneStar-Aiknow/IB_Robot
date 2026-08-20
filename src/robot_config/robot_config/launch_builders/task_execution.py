"""Task execution launch builder.

This module generates the task_executor_node for task-level execution
in moveit_planning-based control modes (visual_grasp, VoxPoser, etc.).

The task_executor_node provides an ExecuteTaskPlan action server.
Planners send a sequence of TaskSteps; the executor delegates arm
motion to moveit_gateway and gripper control to ros2_control.
"""

from ament_index_python.packages import PackageNotFoundError, get_package_prefix
from launch_ros.actions import Node

from robot_config.utils import parse_bool


def generate_task_executor_node(robot_config, control_mode, use_sim=False):
    """Generate task executor node for task-level control modes.

    Args:
        robot_config: Robot configuration dict
        control_mode: Active control mode
        use_sim: Simulation mode flag

    Returns:
        Node action for task_executor, or None if not applicable
    """
    is_sim = parse_bool(use_sim, default=False)

    # Only launch for modes that use task-level execution.  A hybrid robot may
    # start in navigation mode while keeping MoveIt online for an automatic
    # transition to its configured manipulation mode.  In that case the task
    # executor is still a required manipulation dependency and must be started
    # before the first skill requests the transition.
    control_modes = robot_config.get("control_modes", {})
    mode_config = control_modes.get(control_mode, {})
    motion_mode = robot_config.get("motion_mode", {})
    manipulation_control_mode = ""
    if isinstance(motion_mode, dict) and bool(motion_mode.get("enabled", False)):
        manipulation_control_mode = str(motion_mode.get("manipulation_control_mode", "")).strip()
    manipulation_mode_config = control_modes.get(manipulation_control_mode, {})

    # Task executor is relevant for moveit-based modes with task_dispatch enabled
    executor_config = mode_config.get("executor", {})
    candidate_modes = [(control_mode, mode_config)]
    if manipulation_control_mode and control_mode != manipulation_control_mode:
        candidate_modes.append((manipulation_control_mode, manipulation_mode_config))
        executor_config = manipulation_mode_config.get("executor", {})

    # Auto-detect: launch task_executor when either the active mode or the
    # configured manipulation mode has MoveIt/task-dispatch semantics.
    needs_task_executor = any(
        "moveit" in mode_name.lower()
        or "visual_grasp" in mode_name.lower()
        or "voxposer" in mode_name.lower()
        or mode.get("executor", {}).get("task_dispatch", False)
        for mode_name, mode in candidate_modes
    )

    if not needs_task_executor:
        print(f"[robot_config] Task executor not needed for mode '{control_mode}'")
        return None

    try:
        get_package_prefix("task_dispatch")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "MoveIt task execution requires the ROS package 'task_dispatch', but it is not present in the "
            "sourced install space. Build robot_config with its runtime dependencies and source install/setup.sh."
        ) from exc

    robot_config_path = robot_config.get("_config_path", "")
    if not robot_config_path:
        raise ValueError("robot_config dict is missing '_config_path'. Ensure loader.py injects this correctly.")

    print("[robot_config] ========== Generating Task Executor ==========")
    print(f"[robot_config] Control mode: {control_mode}")

    node = Node(
        package="task_dispatch",
        executable="task_executor_node",
        name="task_executor",
        parameters=[
            {
                "robot_config_path": str(robot_config_path),
                "use_sim_time": is_sim,
                "skip_redundant_gripper_open": bool(executor_config.get("skip_redundant_gripper_open", False)),
                "gripper_open_position": float(executor_config.get("gripper_open_position", 1.0)),
                "gripper_position_tolerance": max(float(executor_config.get("gripper_position_tolerance", 0.05)), 0.0),
                "joint_state_max_age_s": max(float(executor_config.get("joint_state_max_age_s", 0.25)), 0.0),
            }
        ],
        output="screen",
    )

    print("[robot_config] ✓ Task executor node configured")
    return node
