"""Forwarding entry for the teleoperation-only operator launch.

The real entry moved to its owning package together with the teleop launch
builder (openspec change ``migrate-business-launch-builders``, design
decision 5): use ``ros2 launch robot_teleop teleop_operator.launch.py``.
This file keeps the historical ``robot_config`` path working — it declares
the same arguments, forwards them verbatim and prints one deprecation
notice. It deliberately imports NO ``robot_teleop`` Python module; the
target launch file is located through the package share directory at launch
time, so an environment without robot_teleop fails with a clear error
instead of an import failure at description-generation time.
"""

from ament_index_python.packages import PackageNotFoundError
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

DEPRECATION_NOTICE = (
    "teleop_operator.launch.py moved to the robot_teleop package; "
    "prefer: ros2 launch robot_teleop teleop_operator.launch.py"
)


def _operator_launch_source():
    from pathlib import Path

    from ament_index_python.packages import get_package_share_directory

    try:
        share = get_package_share_directory("robot_teleop")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "robot_config's teleop_operator.launch.py now forwards to the robot_teleop package, "
            "which is not installed in this environment. Install robot_teleop (e.g. "
            "colcon build --packages-select robot_teleop) or launch the operator host from a "
            "workspace that builds it."
        ) from exc
    launch_file = Path(share) / "launch" / "teleop_operator.launch.py"
    if not launch_file.is_file():
        raise RuntimeError(
            f"robot_teleop is installed but {launch_file} is missing; the installed package "
            "predates the teleop launch migration. Rebuild robot_teleop from a current source "
            "tree (stale egg-info from the old layout is the usual cause)."
        )
    return PythonLaunchDescriptionSource(str(launch_file))


def generate_launch_description():
    """Forward to robot_teleop's teleop_operator.launch.py, passing the arguments through."""
    return LaunchDescription(
        [
            LogInfo(msg=DEPRECATION_NOTICE),
            DeclareLaunchArgument(
                "robot_config",
                default_value="so101_single_arm",
                description="Robot configuration name (without .yaml extension)",
            ),
            DeclareLaunchArgument(
                "config_path",
                default_value="",
                description="Optional: Full path to robot config file (overrides robot_config)",
            ),
            IncludeLaunchDescription(
                _operator_launch_source(),
                launch_arguments={
                    "robot_config": LaunchConfiguration("robot_config"),
                    "config_path": LaunchConfiguration("config_path"),
                }.items(),
            ),
        ]
    )
