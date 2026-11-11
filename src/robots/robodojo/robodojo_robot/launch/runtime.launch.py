"""RoboDojo runtime launch entry (robot-runtime-packaging spec).

    ros2 launch robodojo_robot runtime.launch.py [profile:=<path-or-name>] [instance_id:=<id>] [policy_port:=<port>]

The X5 exists only on the ``robodojo`` simulation platform. The selection resolves through
the shared rules and dispatches like every physics platform: the registry
(``ibrobot.sim_platforms``) names the platform's embodiment, which starts one process, the
direct-control runtime hosting RoboDojo's policy server. RoboDojo's eval client (Isaac Sim, its
own conda env) is started separately; see robodojo_robot/README.md. No simulator is started here.
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

from robot_runtime.launch_support import resolve_profile_argument, resolve_sim_selection, sim_embodiment_actions
from robot_runtime.profile import load_profile
from robot_runtime.sim_platforms import is_physics_selection


def apply_policy_port(profile: dict, policy_port: str) -> dict:
    """``policy_port:=N`` replaces the profile's policy server port (in memory, for this launch)."""
    port = policy_port.strip()
    if port:
        section = profile.setdefault("simulation", {}).setdefault("robodojo", {})
        section.setdefault("adapter_config", {})["port"] = int(port)
    return profile


def _launch_setup(context, *_args, **_kwargs):
    profile_path = resolve_profile_argument("robodojo_robot", LaunchConfiguration("profile").perform(context))
    profile_path = profile_path.resolve()
    profile = load_profile(profile_path)
    use_sim, sim_platform = resolve_sim_selection(
        profile,
        LaunchConfiguration("use_sim").perform(context),
        LaunchConfiguration("sim_platform").perform(context),
        LaunchConfiguration("simulated").perform(context),
        source=str(profile_path),
    )
    if not is_physics_selection(use_sim, sim_platform):
        raise RuntimeError(
            f"robodojo_robot runs only on the robodojo simulation platform (got use_sim={use_sim}, "
            f"sim_platform={sim_platform!r}): RoboDojo's Isaac process is the robot"
        )
    apply_policy_port(profile, LaunchConfiguration("policy_port").perform(context))
    return sim_embodiment_actions(
        profile,
        profile_path,
        str(sim_platform),
        LaunchConfiguration("peripherals").perform(context).strip(),
        LaunchConfiguration("instance_id").perform(context).strip(),
    )


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("profile", default_value=""),
            DeclareLaunchArgument("instance_id", default_value=""),
            DeclareLaunchArgument("policy_port", default_value="", description="Override the policy server port"),
            DeclareLaunchArgument("use_sim", default_value=""),
            DeclareLaunchArgument("sim_platform", default_value=""),
            DeclareLaunchArgument("simulated", default_value="", description="Deprecated alias of use_sim"),
            # Accepted for the shared provider contract; RoboDojo owns its world and spawn pose.
            DeclareLaunchArgument("sim_world", default_value=""),
            DeclareLaunchArgument("spawn_x", default_value="0.0"),
            DeclareLaunchArgument("spawn_y", default_value="0.0"),
            DeclareLaunchArgument("spawn_z", default_value="0.0"),
            DeclareLaunchArgument("spawn_yaw", default_value="0.0"),
            DeclareLaunchArgument("peripherals", default_value=""),
            DeclareLaunchArgument("initial_mode", default_value=""),
            OpaqueFunction(function=_launch_setup),
        ]
    )
