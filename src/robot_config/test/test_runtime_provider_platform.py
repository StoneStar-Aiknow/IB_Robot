"""Provider simulation-platform and deployment-toggle contract (group 3).

Openspec ``so101-gazebo-runtime-transport`` tasks 3.1, 3.1b and 3.3: the
shared provider/simulation-platform predicate used by loader validation and
launch preparation, toggle-only deployment peripheral overrides on the
provider path, and the single transport selection generic launch hands the
provider (with the world name and the full spawn pose).
"""

from __future__ import annotations

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


def test_shared_predicate_accepts_sdk_and_gazebo_and_names_everything_otherwise():
    assert provider_sim_platform_error("sdk", "so101_robot") == ""
    assert provider_sim_platform_error("gazebo", "so101_robot") == ""
    for platform in ("mujoco", "mock", "webots"):
        error = provider_sim_platform_error(platform, "so101_robot")
        assert platform in error and "so101_robot" in error and "gazebo" in error


def test_loader_accepts_provider_config_with_gazebo_platform():
    errors = validate_runtime_provider_config(_provider_config(platform="gazebo"))
    assert not [error for error in errors if "simulation.platform" in error]


@pytest.mark.parametrize("platform", ["mujoco", "mock"])
def test_loader_rejects_other_platforms_naming_platform_and_provider(platform):
    errors = validate_runtime_provider_config(_provider_config(platform=platform))
    assert any(platform in error and "so101_robot" in error for error in errors)


@pytest.mark.parametrize("platform", ["mujoco", "mock"])
def test_preparation_rejects_other_platforms_through_the_same_predicate(platform):
    from launch import LaunchContext

    from robot_config.launch_registry.builtin_registry import build_builtin_registry
    from robot_config.launch_registry.preparation import PreparationError, prepare_launch

    context = LaunchContext()
    context.launch_configurations.update({"use_sim": "true", "sim_platform": platform})
    with pytest.raises(PreparationError, match=f"runtime.provider=so101_robot.*{platform}"):
        prepare_launch(context, build_builtin_registry(), loaded_config=_provider_config())


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
# 3.1b: toggle-only deployment peripherals on the provider path
# ---------------------------------------------------------------------------


def test_toggle_only_peripherals_are_accepted_for_require_model_providers():
    config = _provider_config()
    config["peripherals"] = [
        {"type": "camera", "name": "front", "disabled": True},
        {"type": "camera", "name": "top", "enabled": False},
    ]
    errors = validate_runtime_provider_config(config)
    assert not errors


@pytest.mark.parametrize(
    "entry",
    [
        {"type": "camera", "name": "front", "driver": "opencv"},
        {"type": "camera", "name": "front", "index": 2},
        {"type": "camera", "name": "front", "transform": {"parent_frame": "base"}},
    ],
)
def test_non_toggle_peripheral_keys_are_rejected(entry):
    config = _provider_config()
    config["peripherals"] = [entry]
    errors = validate_runtime_provider_config(config)
    assert any("may only toggle" in error for error in errors), errors


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
    provider on the Gazebo transport in the pick_banana world, with its
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
    assert provider_args["transport"] == "gazebo"
    assert "simulated" not in provider_args
    assert provider_args["gazebo_world"] == "demo"
    assert (provider_args["spawn_x"], provider_args["spawn_y"], provider_args["spawn_z"]) == ("0.0", "0.3", "-0.03")
    # the deployment peripheral toggle reaches the provider fragment
    import yaml as _yaml

    fragment = _yaml.safe_load(Path(provider_args["peripherals"]).read_text(encoding="utf-8"))
    assert fragment == {"peripherals": [{"name": "front", "type": "camera", "disabled": True}]}


# ---------------------------------------------------------------------------
# 3.3: exactly one transport selection, world name and full spawn pose
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


def test_hardware_and_sdk_keep_the_legacy_simulated_selection():
    hardware = _include_arguments(_provider_config(), use_sim=False)
    assert hardware["simulated"] == "false"
    assert "transport" not in hardware
    sdk = _include_arguments(_provider_config(), use_sim=True, sim_platform="sdk")
    assert sdk["simulated"] == "true"
    assert "transport" not in sdk


def test_gazebo_emits_transport_world_and_full_spawn_pose_only():
    startup = SimulationStartup(
        entity_action=None,
        world_name="demo",
        spawn_pose=(0.5, 0.3, -0.03, 1.25),
    )
    arguments = _include_arguments(
        _provider_config(platform="gazebo"), use_sim=True, sim_platform="gazebo", simulation_startup=startup
    )
    assert arguments["transport"] == "gazebo"
    assert "simulated" not in arguments, "the boolean alias must not accompany an explicit transport"
    assert arguments["gazebo_world"] == "demo"
    assert (arguments["spawn_x"], arguments["spawn_y"], arguments["spawn_z"]) == ("0.5", "0.3", "-0.03")
    assert arguments["spawn_yaw"] == "1.25"


def test_gazebo_without_a_started_world_spawns_into_an_empty_world():
    arguments = _include_arguments(
        _provider_config(platform="gazebo"), use_sim=True, sim_platform="gazebo", simulation_startup=None
    )
    assert arguments["transport"] == "gazebo"
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
# 3.1c: bound consumers follow the transport clock (D7)
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
def test_bound_consumers_follow_the_transport_clock(monkeypatch, use_sim, sim_platform, platform, expected):
    prepared = _prepared_for_platform(monkeypatch, use_sim=use_sim, sim_platform=sim_platform, platform=platform)
    # the final PreparedLaunch carries the resolved clock policy
    assert prepared.node_use_sim_time is expected
    assert prepared.context.node_use_sim_time is expected
    # and the execution builder hands exactly that to the generated consumers
    import robot_config.launch_builders.execution as execution_module
    from robot_config.launch_builders.execution import build_execution

    captured = {}

    def spy(robot_config, control_mode, use_sim, use_sim_time=None, runtime_target=None):
        captured["use_sim_time"] = use_sim_time
        return []

    monkeypatch.setattr(execution_module, "generate_execution_nodes", spy)
    build_execution(prepared.context)
    assert captured["use_sim_time"] is expected
