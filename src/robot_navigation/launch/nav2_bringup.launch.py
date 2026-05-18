import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    # Get directories
    pkg_dir = FindPackageShare(package="robot_navigation").find("robot_navigation")
    lekiwi_description_dir = FindPackageShare(package="lekiwi_description").find("lekiwi_description")
    nav2_bringup_dir = FindPackageShare(package="nav2_bringup").find("nav2_bringup")
    default_model_path = os.path.join(lekiwi_description_dir, "urdf", "lekiwi_assembled.urdf.xacro")

    # Launch configurations
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_file = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")
    autostart = LaunchConfiguration("autostart")
    use_composition = LaunchConfiguration("use_composition")
    use_respawn = LaunchConfiguration("use_respawn")
    namespace = LaunchConfiguration("namespace")
    log_level = LaunchConfiguration("log_level")

    stdout_linebuf_envvar = SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1")

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites={
            "use_sim_time": use_sim_time,
            "yaml_filename": map_file,
        },
        convert_types=True,
    )

    # ==================== Robot Specific Nodes ====================
    # Joint State Publisher
    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        parameters=[
            {
                "robot_description": ParameterValue(Command(["xacro ", default_model_path]), value_type=str),
            }
        ],
    )
    # Robot State Publisher
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[
            {
                "robot_description": ParameterValue(Command(["xacro ", default_model_path]), value_type=str),
            }
        ],
    )

    # ==================== RViz2 ====================
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(pkg_dir, "config", "config.rviz")],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    # ==================== Nav2 Goal Client ====================
    nav2_goal_client_node = Node(
        package="robot_navigation",
        executable="nav2_goal_client",
        name="nav2_goal_client",
        parameters=[
            {
                "use_sim_time": use_sim_time,
            }
        ],
        output="screen",
    )

    localization_nodes = [
        Node(
            condition=IfCondition(PythonExpression(["not ", use_composition])),
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            respawn=use_respawn,
            respawn_delay=2.0,
            parameters=[configured_params],
            arguments=["--ros-args", "--log-level", log_level],
            remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
        ),
        Node(
            condition=IfCondition(PythonExpression(["not ", use_composition])),
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_localization",
            output="screen",
            arguments=["--ros-args", "--log-level", log_level],
            parameters=[
                {"use_sim_time": use_sim_time},
                {"autostart": autostart},
                {"node_names": ["map_server"]},
            ],
        ),
    ]

    navigation_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, "launch", "navigation_launch.py")),
        launch_arguments={
            "namespace": namespace,
            "use_sim_time": use_sim_time,
            "autostart": autostart,
            "params_file": params_file,
            "use_composition": use_composition,
            "use_respawn": use_respawn,
            "container_name": "nav2_container",
            "log_level": log_level,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time", default_value="false", description="Use simulation (Gazebo) clock if true"
            ),
            DeclareLaunchArgument(
                "map",
                default_value=os.path.expanduser("~/.ros/ibrobot/maps/rtabmap.yaml"),
                description="Full path to map yaml file",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=os.path.join(pkg_dir, "config", "nav2_params.yaml"),
                description="Full path to the Nav2 parameters file",
            ),
            DeclareLaunchArgument("namespace", default_value=""),
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument("use_composition", default_value="False"),
            DeclareLaunchArgument("use_respawn", default_value="False"),
            DeclareLaunchArgument("log_level", default_value="info"),
            stdout_linebuf_envvar,
            # Robot specific nodes
            robot_state_publisher_node,
            joint_state_publisher_node,
            *localization_nodes,
            navigation_bringup,
            # Nav2 Goal Client (with voice control)
            nav2_goal_client_node,
            # RViz2
            rviz_node,
        ]
    )
