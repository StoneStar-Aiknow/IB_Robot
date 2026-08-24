import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_dir = FindPackageShare(package="semantic_mapping").find("semantic_mapping")
    default_rviz_config = os.path.join(package_dir, "rviz", "lekiwi_semantic_capture.rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    use_sim_time = LaunchConfiguration("use_sim_time")

    preview_decoder = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "robot_calibration",
            "calib_preview_decode",
            "--input-topic",
            "/semantic_mapping/preview/image/compressed",
            "--output-topic",
            "/semantic_mapping/preview/image",
            "--node-name",
            "semantic_mapping_preview_decoder",
        ],
        output="screen",
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="lekiwi_semantic_capture_rviz",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz_config,
                description="Full path to the semantic capture RViz config",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation clock if true",
            ),
            preview_decoder,
            rviz,
        ]
    )
