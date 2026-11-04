"""AimDK X2 runtime launch: the physics-platform branch (sim-platform-registry D3).

A physics selection dispatches to the effective embodiment and returns exactly
its actions — the vendor mock and the synthetic data plane are the ``sdk``
stand-ins and never start on a physics platform. No X2 physics route is
shipped; the branch is proven with a fake registered platform and a profile
copy that declares it.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import launch
import pytest

from robot_runtime.sim_platforms import SimPlatform, clear_sim_platform_cache, discover_sim_platforms

_LAUNCH = Path(__file__).resolve().parents[1] / "launch" / "runtime.launch.py"
_PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "x2_ultra.yaml"
_PLATFORM = "x2testsim"
_FAKE_ACTIONS = [f"{_PLATFORM}-embodiment-action"]


class _FakeEmbodiment:
    def __init__(self):
        self.calls: list[dict] = []

    def render_description(self, profile, profile_path, *, peripherals=None, spawn_pose=(0.0, 0.0, 0.0, 0.0), world=""):
        raise AssertionError("the physics branch does not render the description here")

    def actions(
        self,
        profile,
        profile_path,
        peripherals_file="",
        instance_id="",
        *,
        robot_description=None,
        interface_description=None,
        world="",
        spawn_pose=(0.0, 0.0, 0.0, 0.0),
    ):
        self.calls.append({"world": world, "spawn_pose": spawn_pose, "interface_description": interface_description})
        return list(_FAKE_ACTIONS)


def _runtime_launch_module():
    spec = importlib.util.spec_from_file_location("aimdk_robot_runtime_launch_under_test", _LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_platform(monkeypatch):
    """A fake physics platform discoverable through the registry seam."""
    module = types.ModuleType(f"{_PLATFORM}_platform")
    module.SPEC = SimPlatform(name=_PLATFORM, package="testsim_pkg", embodiment=f"{_PLATFORM}_platform:EMBODIMENT")
    module.EMBODIMENT = _FakeEmbodiment()
    monkeypatch.setitem(sys.modules, f"{_PLATFORM}_platform", module)
    clear_sim_platform_cache()
    entry = types.SimpleNamespace(name=_PLATFORM, value=f"{_PLATFORM}_platform:SPEC")
    platforms = discover_sim_platforms(entry_points=[entry])
    import robot_runtime.sim_platforms as sim_platforms

    monkeypatch.setattr(sim_platforms, "discover_sim_platforms", lambda *, entry_points=None: platforms)
    return module


@pytest.fixture
def physics_profile(tmp_path):
    """A profile copy that declares the fake platform as supported.

    The default platform stays ``sdk``: the physics selection comes from the
    launch argument (``sim_platform:=``), exactly how a deployment selects a
    platform the profile merely allows.
    """
    import yaml

    data = yaml.safe_load(_PROFILE.read_text(encoding="utf-8"))
    data["sim_platforms"] = ["sdk", _PLATFORM]
    path = tmp_path / "x2_physics.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _render(profile_path, sim_platform="", sim_world="", spawn=("0.0", "0.0", "0.0", "0.0")):
    context = launch.LaunchContext()
    context.launch_configurations.update(
        {
            "profile": str(profile_path),
            "use_sim": "true",
            "sim_platform": sim_platform,
            "simulated": "",
            "sim_world": sim_world,
            "spawn_x": spawn[0],
            "spawn_y": spawn[1],
            "spawn_z": spawn[2],
            "spawn_yaw": spawn[3],
            "instance_id": "",
            "initial_mode": "",
            "peripherals": "",
        }
    )
    return _runtime_launch_module()._launch_setup(context)


def test_a_physics_platform_returns_only_the_fake_embodiments_actions(fake_platform, physics_profile):
    actions = _render(physics_profile, _PLATFORM)
    assert actions == _FAKE_ACTIONS, "no vendor mock, no data plane, nothing else"


def test_the_embodiment_receives_a_description_carrying_the_selected_platform(fake_platform, physics_profile):
    """The description the physics branch hands the embodiment names the platform."""
    _render(physics_profile, _PLATFORM)
    description = fake_platform.EMBODIMENT.calls[0]["interface_description"]
    assert description["execution"] == "simulated"
    assert description["sim_platform"] == _PLATFORM


def test_the_physics_branch_hands_the_world_and_spawn_pose_to_the_embodiment(fake_platform, physics_profile):
    actions = _render(physics_profile, _PLATFORM, sim_world="x2_park", spawn=("0.4", "-1.5", "0.02", "1.57"))
    assert actions == _FAKE_ACTIONS
    call = fake_platform.EMBODIMENT.calls[-1]
    assert (call["world"], call["spawn_pose"]) == ("x2_park", (0.4, -1.5, 0.02, 1.57))


def test_the_sdk_path_keeps_the_vendor_mock_and_data_plane(fake_platform):
    """Regression pin: the physics branch must not disturb the sdk path."""
    from launch_ros.actions import Node

    def flatten(actions, out=None):
        out = out if out is not None else []
        for action in actions:
            out.append(action)
            deferred = getattr(action, "_TimerAction__actions", None)
            if deferred:
                flatten(list(deferred), out)
        return out

    actions = _render(_PROFILE, "")
    executables = {node._Node__node_executable for node in flatten(actions) if isinstance(node, Node)}
    assert "aimdk_vendor_mock" in executables and "aimdk_dataplane" in executables
