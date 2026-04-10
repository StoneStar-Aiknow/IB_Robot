"""Task execution launch builder.

This module generates the task_executor_node for task-level execution
in moveit_planning-based control modes (visual_grasp, VoxPoser, etc.).

The task_executor_node provides an ExecuteTaskPlan action server.
Planners send a sequence of TaskSteps; the executor delegates arm
motion to moveit_gateway and gripper control to ros2_control.
"""

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

    # Only launch for modes that use task-level execution
    control_modes = robot_config.get("control_modes", {})
    mode_config = control_modes.get(control_mode, {})

    # Task executor is relevant for moveit-based modes with task_dispatch enabled
    executor_config = mode_config.get("executor", {})

    # Auto-detect: launch task_executor when mode has moveit semantics
    needs_task_executor = (
        "moveit" in control_mode.lower()
        or "visual_grasp" in control_mode.lower()
        or "voxposer" in control_mode.lower()
        or executor_config.get("task_dispatch", False)
    )

    if not needs_task_executor:
        print(f"[robot_config] Task executor not needed for mode '{control_mode}'")
        return None

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
            }
        ],
        output="screen",
    )

    print("[robot_config] ✓ Task executor node configured")
    return node
