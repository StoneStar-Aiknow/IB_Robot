from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("pipeline_id", default_value="policy"),
        DeclareLaunchArgument("model_path", description="Absolute path to the unified policy bundle"),
        DeclareLaunchArgument("deployment", description="Named deployment from inference_manifest.json"),
        DeclareLaunchArgument("request_timeout", default_value="5.0"),
        DeclareLaunchArgument("runtime_options_json", default_value="{}"),
        DeclareLaunchArgument(
            "node_name",
            default_value=["inference_", LaunchConfiguration("pipeline_id"), "_cloud"],
        ),
        DeclareLaunchArgument(
            "request_topic",
            default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/request"],
        ),
        DeclareLaunchArgument(
            "result_topic",
            default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/result"],
        ),
        DeclareLaunchArgument(
            "heartbeat_topic",
            default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/heartbeat"],
        ),
    ]
    cloud_node = Node(
        package="inference_service",
        executable="pure_inference_node",
        name=LaunchConfiguration("node_name"),
        output="screen",
        parameters=[
            {
                "pipeline_id": LaunchConfiguration("pipeline_id"),
                "model_path": LaunchConfiguration("model_path"),
                "deployment": LaunchConfiguration("deployment"),
                "request_timeout": LaunchConfiguration("request_timeout"),
                "runtime_options_json": LaunchConfiguration("runtime_options_json"),
                "node_name": LaunchConfiguration("node_name"),
                "request_topic": LaunchConfiguration("request_topic"),
                "result_topic": LaunchConfiguration("result_topic"),
                "heartbeat_topic": LaunchConfiguration("heartbeat_topic"),
                "use_sim_time": False,
            }
        ],
    )
    return LaunchDescription([*arguments, cloud_node])
