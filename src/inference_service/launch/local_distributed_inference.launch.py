from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("robot_config_path", description="Absolute path to the robot configuration YAML"),
        DeclareLaunchArgument("model_path", description="Absolute path to the unified policy bundle"),
        DeclareLaunchArgument("deployment", description="Named deployment from inference_manifest.json"),
        DeclareLaunchArgument("pipeline_id", default_value="policy"),
        DeclareLaunchArgument("request_timeout", default_value="5.0"),
        DeclareLaunchArgument("default_task", default_value=""),
        DeclareLaunchArgument("runtime_options_json", default_value="{}"),
        DeclareLaunchArgument("use_sim", default_value="false"),
        DeclareLaunchArgument(
            "action_server", default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/dispatch"]
        ),
        DeclareLaunchArgument(
            "reset_service", default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/reset"]
        ),
        DeclareLaunchArgument(
            "health_topic", default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/health"]
        ),
        DeclareLaunchArgument("action_topic", default_value=["/actions/", LaunchConfiguration("pipeline_id")]),
        DeclareLaunchArgument(
            "request_topic", default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/request"]
        ),
        DeclareLaunchArgument(
            "result_topic", default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/result"]
        ),
        DeclareLaunchArgument(
            "heartbeat_topic", default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/heartbeat"]
        ),
    ]
    shared = {
        "pipeline_id": LaunchConfiguration("pipeline_id"),
        "model_path": LaunchConfiguration("model_path"),
        "deployment": LaunchConfiguration("deployment"),
        "request_timeout": LaunchConfiguration("request_timeout"),
        "runtime_options_json": LaunchConfiguration("runtime_options_json"),
        "request_topic": LaunchConfiguration("request_topic"),
        "result_topic": LaunchConfiguration("result_topic"),
        "heartbeat_topic": LaunchConfiguration("heartbeat_topic"),
    }
    edge = Node(
        package="inference_service",
        executable="pipeline_policy_node",
        name=["inference_", LaunchConfiguration("pipeline_id")],
        output="screen",
        parameters=[
            {
                **shared,
                "execution_mode": "distributed",
                "robot_config_path": LaunchConfiguration("robot_config_path"),
                "default_task": LaunchConfiguration("default_task"),
                "use_sim": LaunchConfiguration("use_sim"),
                "use_sim_time": False,
                "node_name": ["inference_", LaunchConfiguration("pipeline_id")],
                "action_server": LaunchConfiguration("action_server"),
                "reset_service": LaunchConfiguration("reset_service"),
                "health_topic": LaunchConfiguration("health_topic"),
                "action_topic": LaunchConfiguration("action_topic"),
            }
        ],
    )
    cloud = Node(
        package="inference_service",
        executable="pure_inference_node",
        name=["inference_", LaunchConfiguration("pipeline_id"), "_cloud"],
        output="screen",
        parameters=[
            {
                **shared,
                "node_name": ["inference_", LaunchConfiguration("pipeline_id"), "_cloud"],
                "use_sim_time": False,
            }
        ],
    )
    return LaunchDescription([*arguments, edge, cloud])
