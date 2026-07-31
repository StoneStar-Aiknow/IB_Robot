import json

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_config_arg = DeclareLaunchArgument("robot_config_path", description="Path to robot_config YAML")
    model_path_arg = DeclareLaunchArgument("model_path", description="Path to unified policy bundle")
    deployment_arg = DeclareLaunchArgument("deployment", default_value="cpu")
    pipeline_id_arg = DeclareLaunchArgument("pipeline_id", default_value="policy")
    execution_mode_arg = DeclareLaunchArgument("inference_execution_mode", default_value="monolithic")
    node_name_arg = DeclareLaunchArgument("node_name", default_value="inference_policy")
    request_timeout_arg = DeclareLaunchArgument("request_timeout", default_value="30.0")
    default_task_arg = DeclareLaunchArgument("default_task", default_value="")
    use_sim_arg = DeclareLaunchArgument("use_sim", default_value="false")
    action_server_arg = DeclareLaunchArgument(
        "action_server", default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/dispatch"]
    )
    reset_service_arg = DeclareLaunchArgument(
        "reset_service", default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/reset"]
    )
    health_topic_arg = DeclareLaunchArgument(
        "health_topic", default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/health"]
    )
    action_topic_arg = DeclareLaunchArgument(
        "action_topic", default_value=["/actions/", LaunchConfiguration("pipeline_id")]
    )
    request_topic_arg = DeclareLaunchArgument(
        "request_topic", default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/request"]
    )
    result_topic_arg = DeclareLaunchArgument(
        "result_topic", default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/result"]
    )
    heartbeat_topic_arg = DeclareLaunchArgument(
        "heartbeat_topic", default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/heartbeat"]
    )
    video_descriptor_topic_arg = DeclareLaunchArgument(
        "video_descriptor_topic",
        default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/video/descriptors"],
    )
    video_status_topic_arg = DeclareLaunchArgument(
        "video_status_topic", default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/video/status"]
    )

    policy_node = Node(
        package="inference_service",
        executable="pipeline_policy_node",
        name=LaunchConfiguration("node_name"),
        output="screen",
        parameters=[
            {
                "pipeline_id": LaunchConfiguration("pipeline_id"),
                "node_name": LaunchConfiguration("node_name"),
                "model_path": LaunchConfiguration("model_path"),
                "deployment": LaunchConfiguration("deployment"),
                "execution_mode": LaunchConfiguration("inference_execution_mode"),
                "robot_config_path": LaunchConfiguration("robot_config_path"),
                "request_timeout": LaunchConfiguration("request_timeout"),
                "default_task": LaunchConfiguration("default_task"),
                "runtime_options_json": json.dumps({}),
                "use_sim": LaunchConfiguration("use_sim"),
                "action_server": LaunchConfiguration("action_server"),
                "reset_service": LaunchConfiguration("reset_service"),
                "health_topic": LaunchConfiguration("health_topic"),
                "action_topic": LaunchConfiguration("action_topic"),
                "request_topic": LaunchConfiguration("request_topic"),
                "result_topic": LaunchConfiguration("result_topic"),
                "heartbeat_topic": LaunchConfiguration("heartbeat_topic"),
                "video_descriptor_topic": LaunchConfiguration("video_descriptor_topic"),
                "video_status_topic": LaunchConfiguration("video_status_topic"),
            }
        ],
    )

    return LaunchDescription(
        [
            robot_config_arg,
            model_path_arg,
            deployment_arg,
            pipeline_id_arg,
            execution_mode_arg,
            node_name_arg,
            request_timeout_arg,
            default_task_arg,
            use_sim_arg,
            action_server_arg,
            reset_service_arg,
            health_topic_arg,
            action_topic_arg,
            request_topic_arg,
            result_topic_arg,
            heartbeat_topic_arg,
            video_descriptor_topic_arg,
            video_status_topic_arg,
            policy_node,
        ]
    )
