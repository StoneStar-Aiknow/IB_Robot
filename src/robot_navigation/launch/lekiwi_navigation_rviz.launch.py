import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_dir = FindPackageShare(package="robot_navigation").find("robot_navigation")
    default_rviz_config = os.path.join(pkg_dir, "config", "lekiwi_navigation.rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    use_sim_time = LaunchConfiguration("use_sim_time")

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz_config,
                description="Full path to the RViz config file",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation clock if true",
            ),
            rviz_node,
        ]
    )
