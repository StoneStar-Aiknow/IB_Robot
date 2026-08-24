"""Launch target tracking lifecycle nodes wired to real localization and Nav2.

The tracker consumes the FAST-LIO bridged odometry (``/odometry/filtered``) and
the SLAM TF tree; the follower drives Nav2 ``ComputePathToPose``/``FollowPath``
actions. The ``mock_slam_nav_interfaces`` node is intentionally not launched:
it exists only for bench tests without a localization stack.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    enabled = LaunchConfiguration("enabled")
    follower_enabled = LaunchConfiguration("follower_enabled")
    tracker_params = {
        "rgb_topic": LaunchConfiguration("rgb_topic"),
        "aligned_depth_topic": LaunchConfiguration("aligned_depth_topic"),
        "camera_info_topic": LaunchConfiguration("camera_info_topic"),
        "odometry_topic": LaunchConfiguration("odometry_topic"),
        "semantic_database_path": LaunchConfiguration("semantic_database_path"),
    }
    follower_params = {
        "enabled": follower_enabled,
        "compute_path_action": LaunchConfiguration("compute_path_action"),
        "follow_path_action": LaunchConfiguration("follow_path_action"),
    }
    return LaunchDescription(
        [
            DeclareLaunchArgument("enabled", default_value="false"),
            DeclareLaunchArgument("follower_enabled", default_value="false"),
            DeclareLaunchArgument("rgb_topic", default_value="/camera/front/image_raw"),
            DeclareLaunchArgument(
                "aligned_depth_topic", default_value="/camera/front/aligned_depth_to_color/image_raw"
            ),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera/front/camera_info"),
            DeclareLaunchArgument("odometry_topic", default_value="/odometry/filtered"),
            DeclareLaunchArgument("semantic_database_path", default_value=""),
            DeclareLaunchArgument("compute_path_action", default_value="/compute_path_to_pose"),
            DeclareLaunchArgument("follow_path_action", default_value="/follow_path"),
            Node(
                package="object_tracker",
                executable="target_tracker_node",
                name="target_tracker",
                condition=IfCondition(enabled),
                parameters=[tracker_params],
                output="screen",
            ),
            Node(
                package="object_tracker",
                executable="dynamic_target_follower_node",
                name="dynamic_target_follower",
                condition=IfCondition(follower_enabled),
                parameters=[follower_params],
                output="screen",
            ),
        ]
    )
