"""Generic launch through the platform registry (openspec sim-platform-registry 3.4).

The generic-launch fake-platform coverage (D10 extensibility list): a fake
physics platform with ``world=None`` flows through preparation, the
pre-binding bootstrap and the provider include arguments with **no
processes**; a configuration whose default platform is ``gazebo`` prepares
on an empty (or raising) registry when launched physical or on ``sdk``; the
scene task starts after binding on gazebo and mujoco with a stubbed bound
description; and the readiness timeout resolves on hardware without the
registry.
"""

from __future__ import annotations

import sys
import types
from importlib.metadata import EntryPoint

import pytest

from robot_runtime.sim_platforms import (
    SIM_PLATFORM_ENTRY_POINT_GROUP,
    SimPlatform,
    SimSensorTraits,
    SimWorldStartup,
    clear_sim_platform_cache,
    discover_sim_platforms,
)

PLATFORM = "testsim"
SPEC_REF = "testsim_platform:SPEC"


def _entry_point(name: str, value: str) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=SIM_PLATFORM_ENTRY_POINT_GROUP)


@pytest.fixture
def fake_platform(monkeypatch) -> SimPlatform:
    """A registered fake physics platform with ``world=None`` (no processes).

    Seeded through the ``entry_points=`` seam and the cached discovery path
    the preparation/bootstrap code reads, exactly like a third-party plugin.
    """
    module = types.ModuleType("testsim_platform")
    module.SPEC = SimPlatform(
        name=PLATFORM,
        package="testsim_pkg",
        embodiment="robot_runtime.launch_support:render_robot_description",  # unused on these paths
        world=None,
        scene_task=None,
        sensor_traits=SimSensorTraits(),
    )
    monkeypatch.setitem(sys.modules, "testsim_platform", module)
    clear_sim_platform_cache()
    platforms = discover_sim_platforms(entry_points=[_entry_point(PLATFORM, SPEC_REF)])
    assert list(platforms) == [PLATFORM]

    import robot_runtime.sim_platforms as sim_platforms

    def _discover(*, entry_points=None):
        if entry_points is None:
            return {PLATFORM: module.SPEC}
        return {str(getattr(entry, "name", "")): module.SPEC for entry in entry_points}

    monkeypatch.setattr(sim_platforms, "discover_sim_platforms", _discover)
    return module.SPEC


@pytest.fixture
def raising_registry(monkeypatch):
    """Discovery that raises: proves a path never consults the registry."""

    def _raise(*, entry_points=None):
        raise AssertionError("the platform registry must not be consulted on this path")

    import robot_runtime.sim_platforms as sim_platforms

    monkeypatch.setattr(sim_platforms, "discover_sim_platforms", _raise)
    clear_sim_platform_cache()


def _provider_config(platform: str = "sdk", **simulation) -> dict:
    return {
        "name": "test_robot",
        "runtime": {"provider": "test_robot", "profile": "p"},
        "simulation": {"platform": platform, **simulation},
    }


def _launch_context(**arguments):
    from types import SimpleNamespace

    return SimpleNamespace(launch_configurations={"robot_config": "test_robot", **arguments})


def _prepare(config, **arguments):
    from robot_config.launch_registry.builtin_registry import build_launch_registry
    from robot_config.launch_registry.preparation import prepare_launch

    return prepare_launch(_launch_context(**arguments), build_launch_registry(entry_points=[]), loaded_config=config)


# ---------------------------------------------------------------------------
# world=None hand-off: preparation, pre-binding bootstrap, include arguments
# ---------------------------------------------------------------------------


def test_fake_platform_prepares_with_registry_caps(fake_platform):
    """A registered fake physics platform prepares: physics caps apply."""
    prepared = _prepare(_provider_config(PLATFORM), use_sim="true", sim_platform=PLATFORM)
    assert prepared.sim_platform == PLATFORM
    assert prepared.use_sim is True
    # a physics selection: the world provides the clock, ros2_control is the
    # provider's (needs_ros2_control False) and generic never spawns it
    assert prepared.node_use_sim_time is True
    assert prepared.sim_backend_needs_ros2_control is False
    assert prepared.context.node_use_sim_time is True


def test_fake_platform_world_none_bootstraps_without_processes(fake_platform, monkeypatch, tmp_path):
    """The pre-binding bootstrap resolves world=scene and starts nothing (D5).

    ``world=None`` means generic launch starts nothing world-side and the
    default world resolution applies: ``sim_world`` is the scene name and no
    process is launched ahead of the provider readiness wait.
    """
    from launch import LaunchContext
    from launch.actions import IncludeLaunchDescription
    from launch_ros.actions import Node

    import robot_config.launch_builders.runtime as runtime_module
    from robot_config.launch_registry.orchestrator import build_prepared_launch

    provider_launch = tmp_path / "runtime.launch.py"
    provider_launch.write_text("# provider is not executed by this construction test\n", encoding="utf-8")
    monkeypatch.setattr(runtime_module, "resolve_runtime_launch", lambda _provider: str(provider_launch))

    config = _provider_config(PLATFORM, scene="pick_banana")
    context = LaunchContext()
    context.launch_configurations.update({"use_sim": "true", "sim_platform": PLATFORM})
    prepared = _prepare(config, use_sim="true", sim_platform=PLATFORM)
    actions = build_prepared_launch(prepared)

    # the include carries the platform-neutral hand-off: the scene name as
    # sim_world (world=None default) and the resolved spawn pose
    includes = [a for a in actions if isinstance(a, IncludeLaunchDescription)]

    def text(value):
        if isinstance(value, list | tuple):
            return "".join(text(part) for part in value)
        return str(getattr(value, "text", value))

    provider = next(i for i in includes if "profile" in {text(k): text(v) for k, v in i.launch_arguments})
    arguments = {text(k): text(v) for k, v in provider.launch_arguments}
    assert arguments["use_sim"] == "true"
    assert arguments["sim_platform"] == PLATFORM
    assert arguments["sim_world"] == "pick_banana"
    assert (arguments["spawn_x"], arguments["spawn_y"], arguments["spawn_z"]) == ("0.0", "0.3", "-0.03")
    assert arguments["spawn_yaw"] == "0.0"
    # world=None: no world-side process at all ahead of the provider
    assert not [n for n in actions if isinstance(n, Node) and n._Node__package in ("ros_gz_sim", "ros_gz_bridge")]
    assert not any("gz_args" in {text(k): text(v) for k, v in i.launch_arguments} for i in includes)


def test_fake_platform_world_hook_receives_scene_and_spawn_once(fake_platform, monkeypatch, tmp_path):
    """A platform WITH a world hook: generic launch resolves the scene and
    spawn once (D7) before calling it, and the hook's actions start ahead of
    the provider readiness wait."""
    from launch import LaunchContext

    import robot_config.launch_builders.runtime as runtime_module
    from robot_config.launch_registry.orchestrator import build_prepared_launch

    provider_launch = tmp_path / "runtime.launch.py"
    provider_launch.write_text("# provider is not executed by this construction test\n", encoding="utf-8")
    monkeypatch.setattr(runtime_module, "resolve_runtime_launch", lambda _provider: str(provider_launch))

    hook_module = types.ModuleType("testsim_world_hooks")
    calls: list[dict] = []

    def start_world(robot_config, *, scene, spawn_pose):
        calls.append({"scene": scene, "spawn_pose": spawn_pose})
        return SimWorldStartup(actions=[], world=f"{scene}-world")

    hook_module.start_world = start_world
    monkeypatch.setitem(sys.modules, "testsim_world_hooks", hook_module)
    fake_platform.__dict__["world"] = "testsim_world_hooks:start_world"

    config = _provider_config(PLATFORM, scene="pick_banana", robot_spawn={"x": 1.0, "yaw": 0.5})
    context = LaunchContext()
    context.launch_configurations.update({"use_sim": "true", "sim_platform": PLATFORM})
    prepared = _prepare(config, use_sim="true", sim_platform=PLATFORM)
    build_prepared_launch(prepared)

    # resolved exactly once, before the provider include: the scene by name
    # and the D7 spawn rule (robot_spawn beats the layout)
    assert len(calls) == 1
    assert calls[0]["scene"] == "pick_banana"
    assert calls[0]["spawn_pose"] == (1.0, 0.0, 0.0, 0.5)


# ---------------------------------------------------------------------------
# an empty/raising registry never gates hardware or sdk launches (D2)
# ---------------------------------------------------------------------------


def test_gazebo_default_config_prepares_physical_with_raising_registry(raising_registry):
    """Config default gazebo, launched use_sim:=false: never consults the registry."""
    prepared = _prepare(_provider_config("gazebo"), use_sim="false")
    assert prepared.use_sim is False
    assert prepared.sim_platform == "gazebo"
    assert prepared.node_use_sim_time is False


def test_gazebo_default_config_prepares_on_sdk_with_raising_registry(raising_registry):
    """Config default gazebo, launched sim_platform:=sdk: never consults the registry."""
    prepared = _prepare(_provider_config("gazebo"), use_sim="true", sim_platform="sdk")
    assert prepared.use_sim is True
    assert prepared.sim_platform == "sdk"
    assert prepared.node_use_sim_time is False  # sdk provides no /clock


def test_unregistered_platform_fails_preparation_for_an_effective_selection(fake_platform, monkeypatch):
    """An effective simulated selection of an unregistered platform fails (D2).

    ``fake_platform`` registers ``testsim``; an *empty* registry makes the
    same selection fail before any process starts, naming the platform.
    """
    import robot_runtime.sim_platforms as sim_platforms
    from robot_config.launch_registry.preparation import PreparationError

    monkeypatch.setattr(sim_platforms, "discover_sim_platforms", lambda *, entry_points=None: {})
    clear_sim_platform_cache()
    with pytest.raises(PreparationError, match="testsim.*not installed"):
        _prepare(_provider_config("testsim"), use_sim="true", sim_platform="testsim")


# ---------------------------------------------------------------------------
# readiness timeout without registry access (D5)
# ---------------------------------------------------------------------------


def test_scene_task_none_platform_logs_and_skips(fake_platform, monkeypatch, caplog):
    """A platform with scene_task=None: the task is not started and the reason is logged."""
    import logging

    import sim_models.scene_compiler as scene_compiler
    from robot_config.launch_builders.scene_task import build_scene_task
    from robot_config.launch_registry.registry import LaunchBuildContext, SimulationStartup

    # a scene whose task declares the fake platform (the shipped layouts
    # declare only the first-party platforms)
    monkeypatch.setattr(
        scene_compiler,
        "get_scene_task",
        lambda scene: {"executable": "pick_banana_task_node", "platforms": [PLATFORM]},
    )
    config = _provider_config(PLATFORM, scene="pick_banana")
    # the bound trajectory interfaces the builder requires before the transport
    config["runtime"] = {
        **config["runtime"],
        "interface_description": {
            "interfaces": {
                "joint.trajectory_arm": {"endpoint": "/arm/follow_joint_trajectory", "joint_names": ["1"]},
                "joint.trajectory_gripper": {"endpoint": "/gripper/follow_joint_trajectory", "joint_names": ["2"]},
            }
        },
    }
    context = LaunchBuildContext(
        robot_config=config,
        use_sim=True,
        sim_platform=PLATFORM,
        node_use_sim_time=True,
        provider_started=True,
        simulation_startup=SimulationStartup(entity_action=None, world="pick_banana", scene_name="pick_banana"),
    )
    with caplog.at_level(logging.INFO, logger="robot_config.launch_builders.scene_task"):
        result = build_scene_task(context)
    assert result.actions == []
    assert "offers no scene-task transport" in caplog.text


def test_readiness_timeout_resolves_on_hardware_with_raising_registry(raising_registry):
    """A gazebo-default config launched physical resolves the hardware tier."""
    from robot_config.launch_builders.readiness import resolve_controller_startup_timeout

    config = {"controller_startup_timeout": {"gazebo": 300.0, "sim": 120.0, "hardware": 30.0}}
    assert resolve_controller_startup_timeout(config, False) == 30.0


def test_readiness_timeout_tiers_per_platform(fake_platform):
    """hardware -> hardware; sdk -> sim; physics <p> -> <p> -> sim -> default."""
    from robot_config.launch_builders.readiness import resolve_controller_startup_timeout

    config = {"controller_startup_timeout": {"gazebo": 300.0, "sim": 120.0, "hardware": 30.0}}
    assert resolve_controller_startup_timeout(config, True, "gazebo") == 300.0
    assert resolve_controller_startup_timeout(config, True, "sdk") == 120.0
    assert resolve_controller_startup_timeout(config, False) == 30.0
    # a physics tier that is not configured falls back to sim
    assert resolve_controller_startup_timeout(config, True, PLATFORM) == 120.0
    # and a platform-specific tier wins for that platform
    config2 = {"controller_startup_timeout": {PLATFORM: 45.0, "sim": 120.0}}
    assert resolve_controller_startup_timeout(config2, True, PLATFORM) == 45.0
    # the no-config defaults stay (120 s sim, 30 s hardware)
    assert resolve_controller_startup_timeout({}, True, PLATFORM) == 120.0
    assert resolve_controller_startup_timeout({}, False) == 30.0
