"""Launch the mock runtime (conformance reference implementation)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.launch_context import LaunchContext
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from robot_runtime.launch_support import resolve_sim_selection
from robot_runtime.mock_runtime_node import default_profile
from robot_runtime.profile import validate_profile


def _launch_setup(context: LaunchContext, *_args, **_kwargs):
    profile = default_profile(True)
    validate_profile(profile, "<mock default>")
    use_sim, sim_platform = resolve_sim_selection(
        profile,
        LaunchConfiguration("use_sim").perform(context),
        LaunchConfiguration("sim_platform").perform(context),
        LaunchConfiguration("simulated").perform(context),
        source="<mock default>",
    )
    sim_platform_params = [{"sim_platform": sim_platform}] if sim_platform else []
    return [
        Node(
            package="robot_runtime",
            executable="mock_runtime",
            name="mock_runtime",
            output="screen",
            parameters=[{"use_sim": use_sim}, *sim_platform_params],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim",
                default_value="",
                description="Run simulated (true) or physical (false); default: the profile's selection",
            ),
            DeclareLaunchArgument(
                "sim_platform",
                default_value="",
                description="Simulation platform (sdk); implies use_sim:=true when given alone",
            ),
            DeclareLaunchArgument(
                "simulated",
                default_value="",
                description="Deprecated alias of use_sim (a warning names use_sim)",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
