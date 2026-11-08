"""Provider simulation-platform and deployment-toggle contract (group 3).

Openspec ``so101-gazebo-runtime-transport`` tasks 3.1, 3.1b and 3.3: the
shared provider/simulation-platform predicate used by loader validation and
launch preparation, toggle-only deployment peripheral overrides on the
provider path, and the simulation selection generic launch hands the
provider (with the world name and the full spawn pose).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from launch.actions import IncludeLaunchDescription

from robot_config.launch_registry.registry import SimulationStartup
from robot_config.loader import load_robot_config_dict, validate_runtime_provider_config
from robot_config.utils import provider_sim_platform_error

ROBOTS = Path(__file__).resolve().parents[1] / "config" / "robots"


def _provider_config(**simulation):
    return {
        "name": "consumer",
        "runtime": {"provider": "so101_robot", "profile": "so101_single_arm", "require_model": True},
        "simulation": {"platform": "sdk", **simulation},
    }


# ---------------------------------------------------------------------------
# 3.1: the shared provider/simulation-platform predicate
# ---------------------------------------------------------------------------


def test_shared_predicate_accepts_provider_platforms_and_names_everything_otherwise():
    assert provider_sim_platform_error("sdk", "so101_robot") == ""
    assert provider_sim_platform_error("gazebo", "so101_robot") == ""
    assert provider_sim_platform_error("mujoco", "so101_robot") == ""
    for platform in ("mock", "webots"):
        error = provider_sim_platform_error(platform, "so101_robot")
        assert platform in error and "so101_robot" in error and "gazebo" in error


def test_providerless_predicate_rejects_mujoco_only():
    from robot_config.utils import providerless_sim_platform_error

    for platform in ("sdk", "gazebo", "mock", ""):
        assert providerless_sim_platform_error(platform) == ""
    error = providerless_sim_platform_error("mujoco")
    assert "mujoco" in error and "runtime provider" in error


def test_loader_accepts_provider_config_with_gazebo_platform():
    errors = validate_runtime_provider_config(_provider_config(platform="gazebo"))
    assert not [error for error in errors if "simulation.platform" in error]


def test_loader_accepts_provider_config_with_mujoco_platform():
    errors = validate_runtime_provider_config(_provider_config(platform="mujoco"))
    assert not [error for error in errors if "simulation.platform" in error]


@pytest.mark.parametrize("platform", ["mock"])
def test_loader_rejects_other_platforms_naming_platform_and_provider(platform):
    errors = validate_runtime_provider_config(_provider_config(platform=platform))
    assert any(platform in error and "so101_robot" in error for error in errors)


@pytest.mark.parametrize("platform", ["mock"])
def test_preparation_rejects_other_platforms_through_the_same_predicate(platform):
    from launch import LaunchContext

    from robot_config.launch_registry.builtin_registry import build_builtin_registry
    from robot_config.launch_registry.preparation import PreparationError, prepare_launch

    context = LaunchContext()
    context.launch_configurations.update({"use_sim": "true", "sim_platform": platform})
    with pytest.raises(PreparationError, match=f"runtime.provider=so101_robot.*{platform}"):
        prepare_launch(context, build_builtin_registry(), loaded_config=_provider_config())


# ---------------------------------------------------------------------------
# D1 (so101-mujoco-runtime-platform): mujoco needs a provider
# ---------------------------------------------------------------------------


def _providerless_config(**simulation):
    return {"name": "legacy", "simulation": {"platform": "gazebo", **simulation}}


def test_loader_rejects_providerless_mujoco_naming_the_platform():
    errors = validate_runtime_provider_config(_providerless_config(platform="mujoco"))
    assert any("mujoco" in error and "runtime provider" in error for error in errors)


def test_loader_keeps_accepting_providerless_gazebo_and_sdk():
    for platform in ("gazebo", "sdk"):
        errors = validate_runtime_provider_config(_providerless_config(platform=platform))
        assert not [error for error in errors if "simulation.platform" in error]


def test_preparation_rejects_providerless_mujoco_cli_override():
    from launch import LaunchContext

    from robot_config.launch_registry.builtin_registry import build_builtin_registry
    from robot_config.launch_registry.preparation import PreparationError, prepare_launch

    context = LaunchContext()
    context.launch_configurations.update({"use_sim": "", "sim_platform": "mujoco"})
    with pytest.raises(PreparationError, match=r"simulation.platform='mujoco'.*runtime provider"):
        prepare_launch(context, build_builtin_registry(), loaded_config=_providerless_config())


def test_preparation_rejects_providerless_mujoco_from_the_config():
    from launch import LaunchContext

    from robot_config.launch_registry.builtin_registry import build_builtin_registry
    from robot_config.launch_registry.preparation import PreparationError, prepare_launch

    context = LaunchContext()
    context.launch_configurations.update({"use_sim": "true", "sim_platform": ""})
    with pytest.raises(PreparationError, match=r"simulation.platform='mujoco'.*runtime provider"):
        prepare_launch(context, build_builtin_registry(), loaded_config=_providerless_config(platform="mujoco"))


def test_provider_mujoco_loads_and_prepares_on_sim_time():
    from launch import LaunchContext

    from robot_config.launch_registry.builtin_registry import build_builtin_registry
    from robot_config.launch_registry.preparation import prepare_launch

    context = LaunchContext()
    context.launch_configurations.update({"use_sim": "true", "sim_platform": "mujoco"})
    prepared = prepare_launch(context, build_builtin_registry(), loaded_config=_provider_config())
    assert prepared.sim_platform == "mujoco"
    # the provider's simulator provides the clock; bound consumers run on sim time
    assert prepared.node_use_sim_time is True
    assert prepared.context.node_use_sim_time is True
    assert prepared.context.sim_backend_needs_ros2_control is False


# ---------------------------------------------------------------------------
# 7.3: generic launch applies the same selection rules as runtime entries
# ---------------------------------------------------------------------------


def _prepared(use_sim_cli: str, platform_cli: str, robot_config=None):
    from launch import LaunchContext

    from robot_config.launch_registry.builtin_registry import build_builtin_registry
    from robot_config.launch_registry.preparation import prepare_launch

    context = LaunchContext()
    context.launch_configurations.update({"use_sim": use_sim_cli, "sim_platform": platform_cli})
    return prepare_launch(context, build_builtin_registry(), loaded_config=robot_config or _provider_config())


@pytest.mark.parametrize(
    ("use_sim_cli", "platform_cli", "config_platform", "expected_use_sim", "expected_platform"),
    [
        # CLI platform alone selects simulation and the provider keeps it
        ("", "gazebo", "sdk", True, "gazebo"),
        ("true", "gazebo", "sdk", True, "gazebo"),
        # explicit use_sim resolves as before; config-only platform never
        # forces simulation (the runtime target stays exactly as before)
        ("true", "", "sdk", True, "sdk"),
        ("false", "", "sdk", False, "sdk"),
        ("", "", "sdk", False, "sdk"),
        ("", "", "gazebo", False, "gazebo"),
    ],
)
def test_selection_matrix_resolves_like_the_runtime_entries(
    use_sim_cli, platform_cli, config_platform, expected_use_sim, expected_platform
):
    from robot_config.launch_registry.registry import RuntimeTarget

    prepared = _prepared(use_sim_cli, platform_cli, _provider_config(platform=config_platform))
    assert prepared.use_sim is expected_use_sim
    assert prepared.sim_platform == expected_platform
    assert prepared.runtime_target is (RuntimeTarget.SIMULATION if expected_use_sim else RuntimeTarget.HARDWARE)

    arguments = _include_arguments(
        _provider_config(platform=expected_platform),
        use_sim=expected_use_sim,
        **({"sim_platform": expected_platform} if expected_use_sim else {}),
    )
    if expected_use_sim:
        assert arguments["use_sim"] == "true"
        assert arguments["sim_platform"] == expected_platform
    else:
        assert arguments["use_sim"] == "false"
        assert "sim_platform" not in arguments, "physical execution passes no platform"
    assert "simulated" not in arguments


def test_cli_platform_with_explicit_physical_fails_naming_both():
    from robot_config.launch_registry.preparation import PreparationError

    with pytest.raises(PreparationError, match=r"use_sim:=false.*sim_platform:='gazebo'"):
        _prepared("false", "gazebo")


def test_cli_platform_still_honours_an_explicit_hardware_target():
    from robot_config.runtime_target import RuntimeTargetError

    # A platform implying simulation must not silently bypass an explicit
    # runtime.target=hardware; the existing target/use_sim consistency rule fires.
    config = _provider_config()
    config["runtime"] = {**config["runtime"], "target": "hardware"}
    with pytest.raises(RuntimeTargetError, match="runtime target 'hardware' requires use_sim=false"):
        _prepared("", "gazebo", config)


def test_preparation_sets_gazebo_world_caps_for_a_provider():
    from launch import LaunchContext

    from robot_config.launch_registry.builtin_registry import build_builtin_registry
    from robot_config.launch_registry.preparation import prepare_launch

    context = LaunchContext()
    context.launch_configurations.update({"use_sim": "true", "sim_platform": ""})
    prepared = prepare_launch(context, build_builtin_registry(), loaded_config=_provider_config(platform="gazebo"))
    assert prepared.sim_platform == "gazebo"
    # the world provides the clock; generic never spawns ros2_control for a provider
    assert prepared.context.node_use_sim_time is True
    assert prepared.context.sim_backend_needs_ros2_control is False


# ---------------------------------------------------------------------------
# runtime-peripheral-overrides: configuration overrides on the provider path
# ---------------------------------------------------------------------------


def test_toggle_entries_are_accepted_for_require_model_providers():
    config = _provider_config()
    config["peripherals"] = [
        {"type": "camera", "name": "front", "disabled": True},
        {"type": "camera", "name": "top", "enabled": False},
    ]
    errors = validate_runtime_provider_config(config)
    assert not errors


def test_device_fields_are_accepted_for_require_model_providers():
    # revision 2: an entry may set the enabled state and device settings of a
    # runtime-declared peripheral; only driver stays runtime-owned.
    config = _provider_config()
    config["peripherals"] = [
        {
            "type": "camera",
            "name": "top",
            "width": 1280,
            "height": 720,
            "fps": 15,
            "pixel_format": "yuyv",
            "depth_width": 640,
            "depth_height": 480,
            "depth_fps": 30,
        },
        {"type": "camera", "name": "front", "disabled": True},
    ]
    errors = validate_runtime_provider_config(config)
    assert not errors, errors


def test_driver_is_rejected_for_require_model_providers():
    config = _provider_config()
    config["peripherals"] = [{"type": "camera", "name": "front", "driver": "opencv"}]
    errors = validate_runtime_provider_config(config)
    assert any(
        "may not set driver: driver is runtime-owned; change the runtime profile" in error for error in errors
    ), errors


def test_ros2_control_still_rejected_for_provider_configs():
    config = _provider_config()
    config["ros2_control"] = {"urdf_path": "x"}
    errors = validate_runtime_provider_config(config)
    assert any("runtime.profile" in error for error in errors)


def test_agent_gazebo_overlay_loads_with_toggles_and_gazebo_platform():
    config = load_robot_config_dict(ROBOTS / "so101_single_arm_agent_gazebo.yaml", defer_interface_binding=True)
    assert config["simulation"]["platform"] == "gazebo"
    # the disabled camera toggles survive loading and reach the runtime fragment
    toggles = {(entry["name"], bool(entry.get("disabled"))) for entry in config["peripherals"]}
    assert ("top", True) in toggles and ("wrist", True) in toggles and ("front", True) in toggles
    errors = validate_runtime_provider_config(config)
    assert not [error for error in errors if "peripheral" in error or "platform" in error]


def test_pick_banana_overlay_loads_and_validates():
    config = load_robot_config_dict(ROBOTS / "so101_single_arm_pick_banana.yaml", defer_interface_binding=True)
    assert config["simulation"]["platform"] == "gazebo"
    assert config["simulation"]["scene"] == "pick_banana"
    assert config["simulation"]["robot_spawn"] == {"x": 0.0, "y": 0.30, "z": -0.03}
    assert not validate_runtime_provider_config(config)


def test_undeclared_toggle_is_rejected_by_the_runtime_merge():
    from robot_runtime.interface_description import InterfaceDescriptionError, merge_peripherals

    with pytest.raises(InterfaceDescriptionError, match=r"\('camera', 'ghost'\) is not declared"):
        merge_peripherals(
            [{"type": "camera", "name": "front", "driver": "opencv"}],
            [{"type": "camera", "name": "ghost", "disabled": True}],
        )


def test_rtp_distributed_launches_the_provider_on_gazebo_pick_banana(monkeypatch, tmp_path):
    """Task 4.2: the migrated so101_rtp_distributed overlay composes the
    provider on the Gazebo platform in the pick_banana world, with its
    application deltas intact (name, skill profile, distributed inference)."""
    from launch import LaunchContext

    import robot_config.launch_builders.runtime as runtime_module
    from robot_config.launch_builders.readiness import load_launch_robot_config
    from robot_config.launch_registry.builtin_registry import build_launch_registry
    from robot_config.launch_registry.orchestrator import build_prepared_launch
    from robot_config.launch_registry.preparation import prepare_launch

    provider_launch = tmp_path / "runtime.launch.py"
    provider_launch.write_text("# provider is not executed by this construction test\n", encoding="utf-8")
    monkeypatch.setattr(runtime_module, "resolve_runtime_launch", lambda _provider: str(provider_launch))

    loaded = load_launch_robot_config("so101_rtp_distributed", None, defer_interface_binding=True)
    assert loaded["name"] == "so101_rtp_distributed"
    assert loaded["embodied"]["skill_catalog_profile"] == "so101_rtp_distributed"
    policy = loaded["control_modes"]["model_inference"]["inference"]["pipelines"]["policy"]
    assert (policy["deployment"], policy["execution_mode"]) == ("ascend_310b1", "distributed")

    context = LaunchContext()
    context.launch_configurations.update({"use_sim": "true", "sim_platform": ""})
    prepared = prepare_launch(context, build_launch_registry(entry_points=[]), loaded_config=loaded)
    assert prepared.sim_platform == "gazebo"
    actions = build_prepared_launch(prepared)

    from launch.actions import IncludeLaunchDescription

    includes = [a for a in actions if isinstance(a, IncludeLaunchDescription)]

    def arguments(include):
        def text(value):
            if isinstance(value, list | tuple):
                return "".join(text(item) for item in value)
            return str(getattr(value, "text", value))

        return {text(k): text(v) for k, v in include.launch_arguments}

    world = next(i for i in includes if "gz_args" in arguments(i))
    provider = next(i for i in includes if "profile" in arguments(i))
    assert "sim_models_pick_banana.world" in arguments(world)["gz_args"]
    provider_args = arguments(provider)
    assert provider_args["use_sim"] == "true"
    assert provider_args["sim_platform"] == "gazebo"
    assert "simulated" not in provider_args
    assert provider_args["gazebo_world"] == "demo"
    assert (provider_args["spawn_x"], provider_args["spawn_y"], provider_args["spawn_z"]) == ("0.0", "0.3", "-0.03")
    # the deployment peripheral toggle reaches the provider fragment
    import yaml as _yaml

    fragment = _yaml.safe_load(Path(provider_args["peripherals"]).read_text(encoding="utf-8"))
    assert fragment == {"peripherals": [{"name": "front", "type": "camera", "disabled": True}]}


# ---------------------------------------------------------------------------
# 3.3: one selection vocabulary (use_sim + sim_platform), world name and full spawn pose
# ---------------------------------------------------------------------------


def _include_arguments(robot_config, **kwargs):
    from robot_config.launch_builders.runtime import generate_runtime_provider_actions

    actions, _waiter = generate_runtime_provider_actions(robot_config, use_sim=kwargs.pop("use_sim", True), **kwargs)

    def text(value):
        if isinstance(value, list | tuple):
            return "".join(text(item) for item in value)
        return str(getattr(value, "text", value))

    include = next(action for action in actions if isinstance(action, IncludeLaunchDescription))
    return {text(key): text(value) for key, value in include.launch_arguments}


@pytest.fixture
def cbf_provider_config(monkeypatch, tmp_path):
    import robot_config.launch_builders.runtime as runtime_module

    provider_launch = tmp_path / "runtime.launch.py"
    provider_launch.write_text("# construction only\n", encoding="utf-8")
    monkeypatch.setattr(runtime_module, "resolve_runtime_launch", lambda _provider: str(provider_launch))
    return {
        "runtime": {"provider": "aimdk_robot", "profile": "unused_while_disabled"},
        "mirror_session": {"control_mode": "mirror"},
        "control_modes": {"mirror": {"runtime_mode": "configured_mirror_stream"}},
    }


@pytest.mark.parametrize("enabled", [None, False])
def test_cbf_default_and_off_keep_include_arguments_identical(cbf_provider_config, monkeypatch, enabled):
    from robot_runtime import profile

    baseline = _include_arguments(cbf_provider_config, use_sim=False)

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("disabled CBF must not load a profile or calibration mapping")

    monkeypatch.setattr(profile, "load_profile", unexpected_load)
    cbf_provider_config["cbf_safety"] = {
        **({"enabled": enabled} if enabled is not None else {}),
        "mode": "invalid_but_disabled",
        "deadline_us": "invalid_but_disabled",
        "deadline_guard_us": -1,
        "thumb_mapping": {"verified": False, "command_scale": object()},
    }
    cbf_provider_config["mirror_session"] = {}
    cbf_provider_config["motion_imitation_stream"] = {"mock_playback": "invalid_but_disabled"}
    assert _include_arguments(cbf_provider_config, use_sim=False) == baseline
    assert "cbf_safety_json" not in baseline


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_cbf_enabled_passes_one_json_argument_with_derived_mode(cbf_provider_config, mode):
    baseline = _include_arguments(cbf_provider_config, use_sim=False)
    mapping = {
        "verified": False,
        "command_scale": [None] * 4,
        "command_offset": [None] * 4,
        "feedback_scale": [None] * 4,
        "feedback_offset": [None] * 4,
    }
    cbf_provider_config["cbf_safety"] = {
        "enabled": True,
        "mode": mode,
        "thumb_mapping": mapping,
        "feedback_max_age_s": 0.08,
        "protected_mirror_mode": "must_not_override_the_configured_mode",
    }
    arguments = _include_arguments(cbf_provider_config, use_sim=False)
    payload = json.loads(arguments.pop("cbf_safety_json"))
    assert arguments == baseline
    assert payload == {
        "enabled": True,
        "mode": mode,
        "deadline_us": 10000,
        "deadline_guard_us": 500,
        "protected_mirror_mode": "configured_mirror_stream",
        "thumb_mapping": mapping,
        "feedback_max_age_s": 0.08,
    }


def test_cbf_defaults_do_not_fabricate_thumb_calibration(cbf_provider_config):
    cbf_provider_config["cbf_safety"] = {"enabled": True}
    payload = json.loads(_include_arguments(cbf_provider_config, use_sim=False)["cbf_safety_json"])
    assert payload["mode"] == "shadow"
    assert "thumb_mapping" not in payload


def test_cbf_feature_profile_enforces_with_explicit_candidate_mapping():
    import yaml

    config = yaml.safe_load((ROBOTS / "aimdk_x2_voice_mirror_all.yaml").read_text(encoding="utf-8"))["robot"]
    assert config["cbf_safety"] == {
        "enabled": True,
        "mode": "enforce",
        "hands_installed": [True, False],
        "deadline_us": 10000,
        "deadline_guard_us": 500,
        "feedback_limit_tolerance_rad": 0.017453292519943295,
        "check_arm_feedback_goal_range": False,
        "thumb_mapping": {
            "verified": False,
            "command_scale": [-1.0, 1.0, 1.0, -1.0],
            "command_offset": [0.0, -1.6424, -0.576, -1.6424],
            "feedback_scale": [-1.0, 1.0, 1.0, -1.0],
            "feedback_offset": [0.0, -1.6424, -0.576, -1.6424],
            "native_lower": [-1.117, -0.05, -0.03, -1.64],
            "native_upper": [0.03, 1.64, 0.87, 0.05],
        },
    }


@pytest.mark.parametrize("config_path", sorted(ROBOTS.glob("*.yaml")), ids=lambda path: path.stem)
def test_cbf_only_voice_mirror_all_emits_configuration_across_robot_profiles(config_path):
    from robot_config.launch_builders.runtime import _cbf_safety_json
    from robot_config.loader import _load_robot_section

    # Use launch's inheritance loader without materializing unrelated model or
    # hardware interfaces. Checking raw YAML would miss an inherited enable.
    _, config = _load_robot_section(config_path)
    payload = _cbf_safety_json(config)
    if config_path.stem == "aimdk_x2_voice_mirror_all":
        assert json.loads(payload)["enabled"] is True
    else:
        assert payload == "", f"{config_path.name} unexpectedly enables the mirror-only CBF"


@pytest.mark.usefixtures("cbf_provider_config")
def test_cbf_grasp_profile_omits_the_provider_launch_argument():
    from robot_config.loader import _load_robot_section

    _, config = _load_robot_section(ROBOTS / "aimdk_x2_voice_mirror_grasp.yaml")
    assert config["cbf_safety"]["enabled"] is False
    assert "cbf_safety_json" not in _include_arguments(config, use_sim=False)


def test_cbf_enabled_requires_the_aimdk_provider(cbf_provider_config):
    cbf_provider_config["cbf_safety"] = {"enabled": True}
    cbf_provider_config["runtime"]["provider"] = "so101_robot"
    with pytest.raises(ValueError, match="runtime.provider=aimdk_robot"):
        _include_arguments(cbf_provider_config, use_sim=False)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"enabled": "false"}, "enabled"),
        ({"mode": "invalid"}, "mode"),
        ({"deadline_us": 0}, "deadline_us"),
        ({"deadline_us": True}, "deadline_us"),
        ({"deadline_us": 10000.5}, "deadline_us"),
        ({"deadline_guard_us": -1}, "deadline_guard_us"),
        ({"deadline_guard_us": 10000}, "deadline_guard_us"),
        ({"deadline_guard_us": False}, "deadline_guard_us"),
    ],
)
def test_cbf_enabled_rejects_invalid_mode_or_budget(cbf_provider_config, overrides, error):
    cbf_provider_config["cbf_safety"] = {"enabled": True, **overrides}
    with pytest.raises(ValueError, match=error):
        _include_arguments(cbf_provider_config, use_sim=False)


def test_cbf_enabled_requires_a_derived_protected_mode(cbf_provider_config):
    cbf_provider_config["cbf_safety"] = {"enabled": True}
    cbf_provider_config["mirror_session"]["control_mode"] = "missing"
    with pytest.raises(ValueError, match="mirror_session.control_mode"):
        _include_arguments(cbf_provider_config, use_sim=False)


def test_cbf_limits_use_profile_arm_order_and_preserve_raw_bands(cbf_provider_config, tmp_path):
    import yaml

    # Use an intentionally non-sorted profile and reverse the input mappings;
    # JSON must follow the existing arm channel order, never mapping order.
    joints = [f"arm_{index}" for index in range(14)]
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "runtime": {"name": "aimdk_robot", "version": "0.1.0"},
                "modes": {"idle": {"controllers": []}},
                "capabilities": {"joint.state": {}},
                "joints": joints,
                "arm_joints": joints,
            }
        ),
        encoding="utf-8",
    )
    cbf_provider_config["runtime"]["profile"] = str(profile)
    cbf_provider_config["cbf_safety"] = {"enabled": True}
    mock = {
        "arm_velocity_limits": {joint: float(index + 1) for index, joint in reversed(list(enumerate(joints)))},
        "arm_limits": {joint: [-index - 0.5, index + 0.5] for index, joint in reversed(list(enumerate(joints)))},
    }
    cbf_provider_config["motion_imitation_stream"] = {"mock_playback": mock}
    payload = json.loads(_include_arguments(cbf_provider_config, use_sim=False)["cbf_safety_json"])
    assert payload["arm_velocity_limits"] == [float(index + 1) for index in range(14)]
    assert payload["arm_lower_limits"] == [-index - 0.5 for index in range(14)]
    assert payload["arm_upper_limits"] == [index + 0.5 for index in range(14)]
    assert "arm_lower_limits" not in mock
    mock["retarget"] = {"vel_scale": 0.125}
    scaled = json.loads(_include_arguments(cbf_provider_config, use_sim=False)["cbf_safety_json"])
    assert scaled["arm_velocity_limits"] == [0.125 * (index + 1) for index in range(14)]
    assert mock["arm_velocity_limits"][joints[0]] == 1.0


def test_hardware_and_sdk_pass_use_sim_and_platform_without_the_alias():
    hardware = _include_arguments(_provider_config(), use_sim=False)
    assert hardware["use_sim"] == "false"
    assert "sim_platform" not in hardware
    assert "simulated" not in hardware, "the legacy alias must never be passed"
    sdk = _include_arguments(_provider_config(), use_sim=True, sim_platform="sdk")
    assert sdk["use_sim"] == "true"
    assert sdk["sim_platform"] == "sdk"
    assert "simulated" not in sdk


def test_gazebo_emits_platform_world_and_full_spawn_pose_only():
    startup = SimulationStartup(
        entity_action=None,
        world_name="demo",
        spawn_pose=(0.5, 0.3, -0.03, 1.25),
    )
    arguments = _include_arguments(
        _provider_config(platform="gazebo"), use_sim=True, sim_platform="gazebo", simulation_startup=startup
    )
    assert arguments["use_sim"] == "true"
    assert arguments["sim_platform"] == "gazebo"
    assert "simulated" not in arguments, "the legacy alias must never be passed"
    assert arguments["gazebo_world"] == "demo"
    assert (arguments["spawn_x"], arguments["spawn_y"], arguments["spawn_z"]) == ("0.5", "0.3", "-0.03")
    assert arguments["spawn_yaw"] == "1.25"


def test_gazebo_without_a_started_world_spawns_into_an_empty_world():
    arguments = _include_arguments(
        _provider_config(platform="gazebo"), use_sim=True, sim_platform="gazebo", simulation_startup=None
    )
    assert arguments["use_sim"] == "true"
    assert arguments["sim_platform"] == "gazebo"
    assert arguments["gazebo_world"] == ""
    assert arguments["spawn_yaw"] == "0.0"


def test_gazebo_readiness_timeout_uses_the_policy_with_a_gazebo_tier():
    from robot_config.launch_builders.readiness import resolve_controller_startup_timeout

    config = {"controller_startup_timeout": {"gazebo": 300.0, "sim": 120.0, "hardware": 30.0}}
    assert resolve_controller_startup_timeout(config, True, "gazebo") == 300.0
    assert resolve_controller_startup_timeout(config, True, "sdk") == 120.0
    assert resolve_controller_startup_timeout(config, False) == 30.0
    # without a gazebo tier the sim tier applies
    config = {"controller_startup_timeout": {"sim": 120.0, "hardware": 30.0}}
    assert resolve_controller_startup_timeout(config, True, "gazebo") == 120.0


# ---------------------------------------------------------------------------
# so101-mujoco-runtime-platform D3/D7: the generic world-side hand-off
# ---------------------------------------------------------------------------


def _mujoco_include_arguments(robot_config, *, sim_platform="mujoco", simulation_startup=None):
    from robot_config.launch_builders.runtime import generate_runtime_provider_actions

    def text(value):
        if isinstance(value, list | tuple):
            return "".join(text(item) for item in value)
        return str(getattr(value, "text", value))

    actions, _waiter = generate_runtime_provider_actions(
        robot_config, use_sim=True, sim_platform=sim_platform, simulation_startup=simulation_startup
    )
    include = next(action for action in actions if isinstance(action, IncludeLaunchDescription))
    return {text(key): text(value) for key, value in include.launch_arguments}


def test_mujoco_adapter_through_the_real_factory_resolves_world_only():
    from robot_config.launch_builders.sim_backend import get_sim_backend

    adapter = get_sim_backend("mujoco")
    config = {"simulation": {"scene": "pick_banana", "robot_spawn": {"x": 1.0, "y": 2.0, "z": -0.03, "yaw": 0.5}}}
    actions, scene_name, spawn_pose = adapter.start_world_only(config)
    assert actions == []
    assert scene_name == "pick_banana"
    assert spawn_pose == (1.0, 2.0, -0.03, 0.5)
    # without robot_spawn the scene layout's spawn applies
    actions, scene_name, spawn_pose = adapter.start_world_only({"simulation": {"scene": "pick_banana"}})
    assert actions == []
    assert spawn_pose == (0.0, 0.30, -0.03, 0.0)
    # no scene: the robot model runs alone
    actions, scene_name, spawn_pose = adapter.start_world_only({"simulation": {}})
    assert scene_name == "" and spawn_pose == (0.0, 0.0, 0.0, 0.0)


def test_mujoco_adapter_start_backend_raises_the_provider_error():
    from robot_config.launch_builders.sim_backend import get_sim_backend

    adapter = get_sim_backend("mujoco")
    with pytest.raises(RuntimeError, match=r"mujoco.*runtime provider"):
        adapter.start_backend({"simulation": {"platform": "mujoco"}})


def test_mujoco_hand_off_passes_scene_and_spawn_by_name_only():
    from robot_config.launch_registry.registry import SimulationStartup

    arguments = _mujoco_include_arguments(
        _provider_config(platform="mujoco"),
        simulation_startup=SimulationStartup(
            entity_action=None, scene_name="pick_banana", spawn_pose=(0.5, 0.3, -0.03, 1.25)
        ),
    )
    assert arguments["use_sim"] == "true"
    assert arguments["sim_platform"] == "mujoco"
    assert "simulated" not in arguments, "the legacy alias must never be passed"
    assert arguments["mujoco_scene"] == "pick_banana"
    assert (arguments["spawn_x"], arguments["spawn_y"], arguments["spawn_z"]) == ("0.5", "0.3", "-0.03")
    assert arguments["spawn_yaw"] == "1.25"
    assert "gazebo_world" not in arguments


def test_mujoco_without_a_started_world_hands_an_empty_scene():
    arguments = _mujoco_include_arguments(_provider_config(platform="mujoco"), simulation_startup=None)
    assert arguments["mujoco_scene"] == ""
    assert arguments["spawn_yaw"] == "0.0"


@pytest.mark.parametrize(
    ("config_name", "expected_scene", "expected_spawn"),
    [
        ("so101_single_arm", "", ("0.0", "0.0", "0.0")),
        ("so101_single_arm_pick_banana", "pick_banana", ("0.0", "0.3", "-0.03")),
    ],
)
def test_mujoco_binding_branch_hands_off_the_resolved_world(
    monkeypatch, tmp_path, config_name, expected_scene, expected_spawn
):
    """The real binding branch: prepare → pre-binding world bootstrap → include.

    The include carries use_sim/sim_platform with the resolved scene (empty
    for the plain profile, the overlay's scene for pick_banana) and the
    overlay/layout spawn; no generic simulator or controller-manager node
    exists and the world-only step runs exactly once."""
    from launch import LaunchContext

    import robot_config.launch_builders.runtime as runtime_module
    from robot_config.launch_builders.readiness import load_launch_robot_config
    from robot_config.launch_registry.builtin_registry import build_launch_registry
    from robot_config.launch_registry.orchestrator import build_prepared_launch
    from robot_config.launch_registry.preparation import prepare_launch

    provider_launch = tmp_path / "runtime.launch.py"
    provider_launch.write_text("# provider is not executed by this construction test\n", encoding="utf-8")
    monkeypatch.setattr(runtime_module, "resolve_runtime_launch", lambda _provider: str(provider_launch))

    loaded = load_launch_robot_config(config_name, None, defer_interface_binding=True)
    context = LaunchContext()
    context.launch_configurations.update({"use_sim": "true", "sim_platform": "mujoco"})
    prepared = prepare_launch(context, build_launch_registry(entry_points=[]), loaded_config=loaded)
    assert prepared.sim_platform == "mujoco"

    world_only_calls = []
    import robot_config.launch_builders.sim_backend as sim_backend_module

    original_factory = sim_backend_module.get_sim_backend

    def factory_spy(platform):
        adapter = original_factory(platform)
        if platform == "mujoco":
            original_start_world_only = adapter.start_world_only

            def spy_world_only(robot_config):
                world_only_calls.append(robot_config.get("simulation", {}).get("scene"))
                return original_start_world_only(robot_config)

            adapter.start_world_only = spy_world_only
        return adapter

    monkeypatch.setattr(sim_backend_module, "get_sim_backend", factory_spy)

    actions = build_prepared_launch(prepared)
    # the world-only step ran exactly once: the pre-binding bootstrap resolved
    # it and the consumer pass reused the pre-seeded SimulationStartup
    assert world_only_calls == [expected_scene or None]

    from launch.actions import IncludeLaunchDescription

    includes = [a for a in actions if isinstance(a, IncludeLaunchDescription)]
    assert includes, "the provider include must exist"

    def arguments(include):
        def text(value):
            if isinstance(value, list | tuple):
                return "".join(text(item) for item in value)
            return str(getattr(value, "text", value))

        return {text(k): text(v) for k, v in include.launch_arguments}

    provider = next(i for i in includes if "profile" in arguments(i))
    provider_args = arguments(provider)
    assert provider_args["use_sim"] == "true"
    assert provider_args["sim_platform"] == "mujoco"
    assert "simulated" not in provider_args
    assert provider_args["mujoco_scene"] == expected_scene
    assert (provider_args["spawn_x"], provider_args["spawn_y"], provider_args["spawn_z"]) == expected_spawn
    # no generic simulator server and no generic controller-manager node
    from launch_ros.actions import Node

    nodes = [a for a in actions if isinstance(a, Node)]
    assert not [n for n in nodes if n._Node__package == "controller_manager"]
    assert not any("gz_args" in arguments(i) for i in includes), "no Gazebo world on the mujoco platform"


# ---------------------------------------------------------------------------
# 3.1c: bound consumers follow the platform clock (D7)
# ---------------------------------------------------------------------------


def _prepared_for_platform(monkeypatch, *, use_sim, sim_platform, platform):
    from launch import LaunchContext

    from robot_config.launch_registry.builtin_registry import build_builtin_registry
    from robot_config.launch_registry.preparation import prepare_launch

    context = LaunchContext()
    context.launch_configurations.update({"use_sim": use_sim, "sim_platform": sim_platform})
    config = _provider_config(platform=platform)
    return prepare_launch(context, build_builtin_registry(), loaded_config=config)


@pytest.mark.parametrize(
    ("use_sim", "sim_platform", "platform", "expected"),
    [
        ("false", "", "sdk", False),  # hardware
        ("true", "", "sdk", False),  # sdk: no /clock, wall time as before
        ("true", "", "gazebo", True),  # gazebo: the world owns /clock
    ],
)
def test_bound_consumers_follow_the_platform_clock(monkeypatch, use_sim, sim_platform, platform, expected):
    prepared = _prepared_for_platform(monkeypatch, use_sim=use_sim, sim_platform=sim_platform, platform=platform)
    # the final PreparedLaunch carries the resolved clock policy
    assert prepared.node_use_sim_time is expected
    assert prepared.context.node_use_sim_time is expected
    # and the execution builder hands exactly that to the generated consumers
    import robot_config.launch_builders.execution as execution_module
    from robot_config.launch_builders.execution import build_execution

    captured = {}

    def spy(robot_config, control_mode, use_sim, use_sim_time=None, runtime_target=None, *, scheduler_enabled=None):
        captured["use_sim_time"] = use_sim_time
        captured["scheduler_enabled"] = scheduler_enabled
        return []

    monkeypatch.setattr(execution_module, "generate_execution_nodes", spy)
    build_execution(prepared.context)
    assert captured["use_sim_time"] is expected
    assert captured["scheduler_enabled"] is prepared.scheduler_enabled


# ---------------------------------------------------------------------------
# 5.1: the scene task builder (so101-mujoco-runtime-platform D8)
# ---------------------------------------------------------------------------


def _scene_task_registry():
    from robot_config.launch_builders.scene_task import SCENE_TASK_BUILDER
    from robot_config.launch_builders.simulation import SIMULATION_BUILDER
    from robot_config.launch_registry.registry import LaunchBuilderRegistry

    registry = LaunchBuilderRegistry()
    registry.add_all([SIMULATION_BUILDER, SCENE_TASK_BUILDER])
    return registry


def _with_bound_trajectory_interfaces(config):
    """Bind the trajectory interfaces the way the live consumer pass sees them."""
    import copy

    bound = copy.deepcopy(config)
    bound.setdefault("runtime", {})["interface_description"] = {
        "interfaces": {
            "joint.trajectory_arm": {
                "endpoint": "/arm_trajectory_controller/follow_joint_trajectory",
                "joint_names": ["1", "2", "3", "4", "5"],
            },
            "joint.trajectory_gripper": {
                "endpoint": "/gripper_trajectory_controller/follow_joint_trajectory",
                "joint_names": ["6"],
            },
        }
    }
    return bound


def _consumer_context(config, *, sim_platform):
    from robot_config.launch_registry.registry import LaunchBuildContext

    return LaunchBuildContext(
        robot_config=config,
        use_sim=True,
        sim_platform=sim_platform,
        node_use_sim_time=True,
        provider_started=True,
    )


def _execute_scene_task_plan(context):
    """Run the frozen simulation+scene_task plan exactly as the consumer pass does."""
    from robot_config.launch_registry.registry import execute_plan

    plan = _scene_task_registry().plan(context)
    return execute_plan(plan, context)


def _node_parameters(node):
    from launch import LaunchContext
    from launch_ros.utilities import evaluate_parameters

    return evaluate_parameters(LaunchContext(), node._Node__parameters)[0]


def _pick_banana_config():
    from robot_config.launch_builders.readiness import load_launch_robot_config

    return load_launch_robot_config("so101_single_arm_pick_banana", None, defer_interface_binding=True)


@pytest.mark.parametrize("sim_platform", ["mujoco", "gazebo"])
def test_scene_task_emits_exactly_one_node_per_launch(sim_platform):
    """5.1: on both physics platforms the plan emits one task node after readiness."""
    from launch_ros.actions import Node

    config = _with_bound_trajectory_interfaces(_pick_banana_config())
    context = _consumer_context(config, sim_platform=sim_platform)
    outputs = _execute_scene_task_plan(context)

    task_nodes = [n for n in outputs.actions if isinstance(n, Node) and n._Node__package == "sim_models"]
    assert len(task_nodes) == 1, "exactly one task node per launch"
    assert task_nodes[0].node_executable == "pick_banana_task_node"
    parameters = _node_parameters(task_nodes[0])
    assert parameters["sim_platform"] == sim_platform
    assert parameters["use_sim_time"] is True
    assert parameters["trajectory_mode"] == "trajectory"
    assert parameters["arm_trajectory_endpoint"] == "/arm_trajectory_controller/follow_joint_trajectory"
    assert parameters["arm_joint_names"] == ("1", "2", "3", "4", "5")
    assert parameters["gripper_trajectory_endpoint"] == "/gripper_trajectory_controller/follow_joint_trajectory"
    assert parameters["gripper_joint_names"] == ("6",)

    if sim_platform == "mujoco":
        # issue #195: no model path crosses the boundary — the backend places
        # through ResetWorld state overrides and observes the scene through
        # the provider-configured FreeJointStatePublisherPlugin topic.
        assert "scene_model_path" not in parameters
        assert "world_name" not in parameters
        assert not [n for n in outputs.actions if isinstance(n, Node) and n._Node__package == "ros_gz_bridge"]
    else:
        assert parameters["world_name"] == "demo"
        assert "scene_model_path" not in parameters

        def raw_arguments(node):
            return [
                "".join(str(getattr(part, "text", part)) for part in arg) if isinstance(arg, list) else str(arg)
                for arg in getattr(node, "_Node__arguments", []) or []
            ]

        # the /clock bridge belongs to the world prelude; the scene task adds
        # exactly one further bridge carrying both object channels
        bridges = [
            n
            for n in outputs.actions
            if isinstance(n, Node)
            and n._Node__package == "ros_gz_bridge"
            and any("set_pose" in arg for arg in raw_arguments(n))
        ]
        assert len(bridges) == 1, "one bridge carries both Gazebo channels"
        bridge_arguments = raw_arguments(bridges[0])
        assert "/world/demo/set_pose@ros_gz_interfaces/srv/SetEntityPose" in bridge_arguments
        assert any("dynamic_pose/info" in arg and "gz.msgs.Pose_V" in arg for arg in bridge_arguments)

        def text(value):
            if isinstance(value, list | tuple):
                return "".join(text(part) for part in value)
            return str(getattr(value, "text", value))

        remappings = {(text(source), text(target)) for source, target in bridges[0]._Node__remappings}
        assert ("/world/demo/dynamic_pose/info", "/sim/pick_banana/object_poses") in remappings


def test_scene_task_never_emits_in_the_pre_binding_prelude(monkeypatch, tmp_path):
    """5.1: the pre-binding bootstrap resolves the world only; the task waits for readiness."""
    from launch import LaunchContext
    from launch_ros.actions import Node

    import robot_config.launch_builders.runtime as runtime_module
    from robot_config.launch_registry.builtin_registry import build_launch_registry
    from robot_config.launch_registry.orchestrator import build_prepared_launch
    from robot_config.launch_registry.preparation import prepare_launch

    provider_launch = tmp_path / "runtime.launch.py"
    provider_launch.write_text("# provider is not executed by this construction test\n", encoding="utf-8")
    monkeypatch.setattr(runtime_module, "resolve_runtime_launch", lambda _provider: str(provider_launch))

    context = LaunchContext()
    context.launch_configurations.update({"use_sim": "true", "sim_platform": "mujoco"})
    prepared = prepare_launch(context, build_launch_registry(entry_points=[]), loaded_config=_pick_banana_config())
    actions = build_prepared_launch(prepared)
    assert not [n for n in actions if isinstance(n, Node) and n._Node__package == "sim_models"], (
        "the task node must not start before the runtime is ready"
    )


def test_scene_task_omissions_are_logged(caplog):
    """5.1: non-binding, provider-less, undeclared-platform and sceneless launches log the omission."""
    import logging

    from robot_config.launch_builders.scene_task import build_scene_task
    from robot_config.launch_registry.registry import LaunchBuildContext

    config = _with_bound_trajectory_interfaces(_pick_banana_config())

    def context(**overrides):
        base = {
            "robot_config": config,
            "use_sim": True,
            "sim_platform": "mujoco",
            "node_use_sim_time": True,
            "provider_started": True,
        }
        base.update(overrides)
        return LaunchBuildContext(**base)

    with caplog.at_level(logging.INFO, logger="robot_config.launch_builders.scene_task"):
        # not the live-binding consumer pass (pre-binding bootstrap, non-binding launch)
        assert build_scene_task(context(provider_started=False)).actions == []
        assert "not on the live-binding consumer pass" in caplog.text
        caplog.clear()
        # provider-less launch
        providerless = _with_bound_trajectory_interfaces({"simulation": {"scene": "pick_banana", "platform": "mujoco"}})
        assert build_scene_task(context(robot_config=providerless)).actions == []
        assert "no runtime provider" in caplog.text
        caplog.clear()
        # platform the layout does not declare (sdk)
        assert build_scene_task(context(sim_platform="sdk")).actions == []
        assert "not declared by scene 'pick_banana'" in caplog.text
        caplog.clear()
        # no scene at all
        plain = _with_bound_trajectory_interfaces(
            _pick_banana_config().__class__({**_pick_banana_config(), "simulation": {"platform": "mujoco"}})
        )
        assert build_scene_task(context(robot_config=plain)).actions == []
        assert "no scene" in caplog.text


def test_scene_task_omitted_when_the_scene_declares_no_task(caplog):
    """5.1: a scene without a task declaration (empty) gets no task node."""
    import logging

    from robot_config.launch_builders.scene_task import build_scene_task
    from robot_config.launch_registry.registry import LaunchBuildContext, SimulationStartup

    config = _with_bound_trajectory_interfaces({"runtime": {"provider": "so101_robot"}})
    context = LaunchBuildContext(
        robot_config=config,
        use_sim=True,
        sim_platform="mujoco",
        node_use_sim_time=True,
        provider_started=True,
        simulation_startup=SimulationStartup(entity_action=None, scene_name="empty"),
    )
    with caplog.at_level(logging.INFO, logger="robot_config.launch_builders.scene_task"):
        assert build_scene_task(context).actions == []
    assert "declares no task" in caplog.text


def test_scene_task_omitted_without_bound_trajectory_interfaces(caplog):
    """5.1: a runtime that does not serve trajectory actions cannot host the task."""
    import logging

    from robot_config.launch_builders.scene_task import build_scene_task
    from robot_config.launch_registry.registry import LaunchBuildContext, SimulationStartup

    config = _pick_banana_config()  # real config, never bound: no interface description
    context = LaunchBuildContext(
        robot_config=config,
        use_sim=True,
        sim_platform="mujoco",
        node_use_sim_time=True,
        provider_started=True,
        simulation_startup=SimulationStartup(entity_action=None, scene_name="pick_banana"),
    )
    with caplog.at_level(logging.WARNING, logger="robot_config.launch_builders.scene_task"):
        assert build_scene_task(context).actions == []
    assert "joint.trajectory_arm" in caplog.text
