"""Launch the mock runtime (conformance reference implementation)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.launch_context import LaunchContext
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from robot_runtime.launch_support import resolve_profile_argument, resolve_sim_selection
from robot_runtime.mock_runtime_node import default_profile
from robot_runtime.profile import load_profile, validate_profile


def _launch_setup(context: LaunchContext, *_args, **_kwargs):
    # The profile argument selects the runtime profile like every provider:
    # a path or a shipped name. Empty (and the historical name "mock", which
    # robot_runtime does not ship as a file) keeps today's default — the
    # in-process reference profile the node regenerates itself. The launch's
    # simulation selection and the node always use this same actual profile.
    profile_argument = LaunchConfiguration("profile").perform(context).strip()
    if profile_argument and profile_argument != "mock":
        profile_path = resolve_profile_argument("robot_runtime", profile_argument)
        profile = load_profile(profile_path)
        profile_value = str(profile_path)
    else:
        profile = default_profile(True)
        validate_profile(profile, "<mock default>")
        profile_value = ""
    use_sim, sim_platform = resolve_sim_selection(
        profile,
        LaunchConfiguration("use_sim").perform(context),
        LaunchConfiguration("sim_platform").perform(context),
        LaunchConfiguration("simulated").perform(context),
        source=profile_value or "<mock default>",
    )
    # The deployment fragment is passed through to the node, which merges it
    # against the profile at initialisation (D9): name and value errors
    # surface as a node-initialisation failure, the provider process exits,
    # the waiter never succeeds and no bound consumer is released.
    peripherals_file = LaunchConfiguration("peripherals").perform(context).strip()
    sim_platform_params = [{"sim_platform": sim_platform}] if sim_platform else []
    return [
        Node(
            package="robot_runtime",
            executable="mock_runtime",
            name="mock_runtime",
            output="screen",
            parameters=[
                {"use_sim": use_sim},
                *sim_platform_params,
                {"peripherals": peripherals_file},
                {"profile": profile_value},
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "profile",
                default_value="",
                description="Runtime profile path or shipped profile name (empty or 'mock': the in-process "
                "reference profile)",
            ),
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
            DeclareLaunchArgument(
                "peripherals",
                default_value="",
                description="Deployment peripherals fragment {peripherals: [...]} "
                "(overrides profile defaults by (type, name))",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
