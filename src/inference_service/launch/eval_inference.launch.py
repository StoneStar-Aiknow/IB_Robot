from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_config_arg = DeclareLaunchArgument("robot_config_path", description="Path to robot_config YAML")
    policy_path_arg = DeclareLaunchArgument("policy_path", description="Path to pretrained policy directory")
    device_arg = DeclareLaunchArgument("device", default_value="auto")
    name_arg = DeclareLaunchArgument("name", default_value="lerobot_policy")
    node_name_arg = DeclareLaunchArgument("node_name", default_value="lerobot_policy_node")
    frequency_arg = DeclareLaunchArgument("frequency", default_value="10.0")
    use_header_time_arg = DeclareLaunchArgument("use_header_time", default_value="true")
    request_timeout_arg = DeclareLaunchArgument("request_timeout", default_value="30.0")
    lerobot_norm_mode_arg = DeclareLaunchArgument("lerobot_norm_mode", default_value="range_m100_100")

    policy_node = Node(
        package="inference_service",
        executable="lerobot_policy_node",
        name=LaunchConfiguration("node_name"),
        output="screen",
        parameters=[
            {
                "name": LaunchConfiguration("name"),
                "node_name": LaunchConfiguration("node_name"),
                "repo_id": LaunchConfiguration("policy_path"),
                "robot_config_path": LaunchConfiguration("robot_config_path"),
                "device": LaunchConfiguration("device"),
                "frequency": LaunchConfiguration("frequency"),
                "use_header_time": LaunchConfiguration("use_header_time"),
                "execution_mode": "monolithic",
                "request_timeout": LaunchConfiguration("request_timeout"),
                "lerobot_norm_mode": LaunchConfiguration("lerobot_norm_mode"),
            }
        ],
    )

    return LaunchDescription(
        [
            robot_config_arg,
            policy_path_arg,
            device_arg,
            name_arg,
            node_name_arg,
            frequency_arg,
            use_header_time_arg,
            request_timeout_arg,
            lerobot_norm_mode_arg,
            policy_node,
        ]
    )
