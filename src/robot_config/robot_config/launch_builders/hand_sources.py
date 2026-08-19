"""Launch builders for target-independent human-hand data sources."""

from pathlib import Path

from launch.actions import EmitEvent
from launch.events import Shutdown
from launch_ros.actions import Node

from robot_config.launch_builders.control import runtime_component_enabled
from robot_config.utils import parse_bool, resolve_mhandpro_sdk_path

FAILURE_POLICIES = {"require_all", "allow_available"}
STARTUP_P_POSE_MODES = {"manual", "interactive"}


def apply_hand_profile(robot_config: dict, requested_profile: str = "") -> str | None:
    """Apply one right/left/dual hand selection from the robot SSOT."""
    selection = robot_config.get("hand_profiles")
    if selection is None:
        if requested_profile:
            raise ValueError("hand_profile was provided but the selected robot config defines no hand_profiles")
        return None
    if not isinstance(selection, dict):
        raise ValueError("hand_profiles must be an object")

    profiles = selection.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("hand_profiles.profiles must be a non-empty object")
    profile_name = requested_profile or str(selection.get("default_profile", "")).strip()
    if not profile_name:
        raise ValueError("hand_profiles.default_profile must select a profile")
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        raise ValueError(f"unknown hand_profile {profile_name!r}; available profiles: {sorted(profiles)}")

    sides = profile.get("sides")
    if not isinstance(sides, list) or not sides or not set(sides).issubset({"left", "right"}):
        raise ValueError(f"hand_profiles.profiles.{profile_name}.sides must contain left and/or right")
    if len(set(sides)) != len(sides):
        raise ValueError(f"hand_profiles.profiles.{profile_name}.sides must not contain duplicates")

    source_name = str(profile.get("hand_source", "")).strip()
    hand_sources = robot_config.get("hand_sources", {}) or {}
    if not source_name or not isinstance(hand_sources, dict) or not isinstance(hand_sources.get(source_name), dict):
        raise ValueError(f"hand profile {profile_name!r} must reference an existing hand_source")
    hand_sources[source_name]["sides"] = list(sides)

    active_actuators = _profile_names(profile, profile_name, "active_actuators")
    actuators = robot_config.get("auxiliary_actuators", {}) or {}
    if not isinstance(actuators, dict):
        raise ValueError("auxiliary_actuators must be an object for hand profile selection")
    unknown_actuators = sorted(set(active_actuators) - set(actuators))
    if unknown_actuators:
        raise ValueError(f"hand profile {profile_name!r} references unknown actuators: {unknown_actuators}")
    for name, config in actuators.items():
        if not isinstance(config, dict):
            raise ValueError(f"auxiliary_actuators.{name} must be an object")
        if parse_bool(config.get("profile_managed", False), default=False):
            config["enabled"] = name in active_actuators

    active_devices = _profile_names(profile, profile_name, "active_devices")
    teleoperation = robot_config.get("teleoperation", {}) or {}
    if not isinstance(teleoperation, dict):
        raise ValueError("teleoperation must be an object for hand profile selection")
    known_devices = {
        str(device.get("name", "")) for device in teleoperation.get("devices", []) if isinstance(device, dict)
    }
    unknown_devices = sorted(set(active_devices) - known_devices)
    if unknown_devices:
        raise ValueError(f"hand profile {profile_name!r} references unknown teleop devices: {unknown_devices}")
    teleoperation["active_devices"] = active_devices

    active_joints = []
    for actuator_name in active_actuators:
        active_joints.extend(str(joint) for joint in actuators[actuator_name].get("joint_names", []))
    if not active_joints or len(set(active_joints)) != len(active_joints):
        raise ValueError(f"hand profile {profile_name!r} actuators must provide unique joint_names")
    robot_config.setdefault("joints", {})["hand"] = active_joints
    robot_config["active_hand_profile"] = profile_name
    return profile_name


def _profile_names(profile: dict, profile_name: str, key: str) -> list[str]:
    values = profile.get(key)
    if not isinstance(values, list) or not values or not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"hand_profiles.profiles.{profile_name}.{key} must be a non-empty string list")
    if len(set(values)) != len(values):
        raise ValueError(f"hand_profiles.profiles.{profile_name}.{key} must not contain duplicates")
    return list(values)


def confirm_interactive_startup_p_pose(
    robot_config: dict,
    *,
    use_sim: bool = False,
    control_mode: str | None = None,
) -> None:
    """Confirm the operator is holding P-pose before real source nodes start."""
    interactive_sources = []
    for config in _named_configs(robot_config.get("hand_sources", {}) or {}, "hand_sources"):
        if not runtime_component_enabled(config, use_sim=use_sim, control_mode=control_mode):
            continue
        if str(config.get("type", "")).strip() != "mhandpro" or parse_bool(config.get("mock", False), default=False):
            continue
        startup_p_pose = _startup_p_pose_mode(config)
        if startup_p_pose == "interactive" and parse_bool(config.get("require_p_pose", True), default=True):
            interactive_sources.append(str(config.get("name", "mhandpro")))

    if not interactive_sources:
        return

    names = ", ".join(interactive_sources)
    prompt = (
        f"\n[{names}] Hold the P-pose before hardware startup:\n"
        "  arm forward and level, palm down, wrist and fingers straight,\n"
        "  thumb straight about 45 degrees from the index finger.\n"
        "Press Enter when stable (Ctrl+C to cancel): "
    )
    try:
        input(prompt)
    except EOFError as exc:
        raise RuntimeError(
            "Interactive startup P-pose requires a terminal. "
            "Run ros2 launch in the foreground or set startup_p_pose: manual."
        ) from exc


def generate_hand_source_nodes(
    robot_config: dict,
    *,
    use_sim: bool = False,
    control_mode: str | None = None,
) -> list[Node]:
    """Generate shared hand-source nodes declared in the robot SSOT."""
    nodes = []
    names = set()
    state_topics = set()
    for index, config in enumerate(_named_configs(robot_config.get("hand_sources", {}) or {}, "hand_sources")):
        if not runtime_component_enabled(config, use_sim=use_sim, control_mode=control_mode):
            continue
        name = str(config.get("name", "")).strip()
        source_type = str(config.get("type", "")).strip()
        if not name or name in names:
            raise ValueError(f"hand_sources[{index}].name must be non-empty and unique")

        driver = config.get("driver", {}) or {}
        if not isinstance(driver, dict):
            raise ValueError(f"Hand source {name!r} driver must be an object")
        if source_type == "mhandpro":
            package = str(driver.get("package", "robot_teleop")).strip()
            executable = str(driver.get("executable", "mhandpro_source_node")).strip()
        else:
            package = str(driver.get("package", "")).strip()
            executable = str(driver.get("executable", "")).strip()
        if not package or not executable:
            raise ValueError(f"Hand source {name!r} requires driver.package and driver.executable")

        parameters = dict(config.get("parameters", {}) or {})
        if source_type == "mhandpro":
            sides = list(dict.fromkeys(str(side) for side in config.get("sides", ["right"])))
            if not sides or not set(sides).issubset({"left", "right"}):
                raise ValueError(f"Hand source {name!r} sides must contain left and/or right")
            mock = parse_bool(config.get("mock", False), default=False)
            failure_policy = str(config.get("failure_policy", "require_all")).strip().lower()
            if failure_policy not in FAILURE_POLICIES:
                raise ValueError(f"Hand source {name!r} failure_policy must be one of {sorted(FAILURE_POLICIES)}")
            startup_p_pose = _startup_p_pose_mode(config)
            lib_path_raw = str(config.get("lib_path", ""))
            lib_path = resolve_mhandpro_sdk_path(lib_path_raw)
            if not mock:
                if not lib_path:
                    raise RuntimeError(
                        "mHandPro SDK library path resolved to an empty value. "
                        "Set hand_sources.<name>.lib_path or MHANDPRO_SDK_LIB to an external vendor library."
                    )
                if not Path(lib_path).is_file():
                    raise RuntimeError(f"mHandPro SDK library not found: {lib_path} (configured as {lib_path_raw!r})")
            topic_prefix = str(config.get("topic_prefix", f"/hand_sources/{name}")).rstrip("/")
            for side in sides:
                state_topic = f"{topic_prefix}/{side}/state"
                if state_topic in state_topics:
                    raise ValueError(f"Hand source state topic is published more than once: {state_topic}")
                state_topics.add(state_topic)
            parameters.update(
                {
                    "source_name": str(config.get("source_name", name)),
                    "lib_path": lib_path,
                    "sides": sides,
                    "mock": mock,
                    "require_p_pose": parse_bool(config.get("require_p_pose", True), default=True),
                    "calibrate_p_pose_on_startup": startup_p_pose == "interactive",
                    "publish_frequency": float(config.get("publish_frequency", 50.0)),
                    "publish_raw_frame": parse_bool(config.get("publish_raw_frame", False), default=False),
                    "stale_timeout": float(config.get("stale_timeout", 0.2)),
                    "startup_timeout": float(config.get("startup_timeout", 30.0)),
                    "calibration_timeout": float(config.get("calibration_timeout", 30.0)),
                    "p_pose_quality_frames": int(config.get("p_pose_quality_frames", 5)),
                    "p_pose_max_openness": float(config.get("p_pose_max_openness", 0.7)),
                    "calibration_service": str(
                        config.get("calibration_service", f"/hand_sources/{name}/calibrate_p_pose")
                    ),
                    "topic_prefix": topic_prefix,
                    "replay_rate_hz": float(config.get("replay_rate_hz", 50.0)),
                    "replay_segment_seconds": float(config.get("replay_segment_seconds", 0.7)),
                    "failure_policy": failure_policy,
                    "auto_reconnect": parse_bool(config.get("auto_reconnect", True), default=True),
                    "reconnect_initial_delay": float(config.get("reconnect_initial_delay", 1.0)),
                    "reconnect_max_delay": float(config.get("reconnect_max_delay", 10.0)),
                    "reconnect_max_attempts": int(config.get("reconnect_max_attempts", 0)),
                }
            )

        names.add(name)
        on_exit = None
        if source_type == "mhandpro" and not parse_bool(config.get("mock", False), default=False):
            on_exit = EmitEvent(event=Shutdown(reason=f"required mHandPro source {name!r} exited"))
        nodes.append(
            Node(
                package=package,
                executable=executable,
                name=name,
                output="screen",
                parameters=[parameters],
                on_exit=on_exit,
            )
        )
    return nodes


def _startup_p_pose_mode(config: dict) -> str:
    mode = str(config.get("startup_p_pose", "manual")).strip().lower()
    if mode not in STARTUP_P_POSE_MODES:
        raise ValueError(f"mHandPro startup_p_pose must be one of {sorted(STARTUP_P_POSE_MODES)}")
    return mode


def _named_configs(raw_configs, label: str) -> list[dict]:
    if isinstance(raw_configs, dict):
        configs = []
        for name, value in raw_configs.items():
            if not isinstance(value, dict):
                raise ValueError(f"{label}.{name} must be an object")
            configs.append({"name": name, **value})
        return configs
    if isinstance(raw_configs, list):
        if not all(isinstance(value, dict) for value in raw_configs):
            raise ValueError(f"{label} list entries must be objects")
        return raw_configs
    raise ValueError(f"{label} must be an object or list")
