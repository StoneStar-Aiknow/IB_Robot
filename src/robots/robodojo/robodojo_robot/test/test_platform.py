"""The robodojo platform: registry entry, embodiment, launch dispatch and the policy port."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from robot_runtime.profile import load_profile
from robot_runtime.sim_platforms import (
    SimEmbodiment,
    clear_sim_platform_cache,
    get_sim_platform,
    is_physics_selection,
    resolve_sim_embodiment,
)

PKG = Path(__file__).resolve().parents[1]
PROFILE = PKG / "profiles" / "arx_x5_dual.yaml"


def _launch_module():
    spec = importlib.util.spec_from_file_location("robodojo_runtime_launch", PKG / "launch" / "runtime.launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _adapter_port(params: dict) -> int:
    from robodojo_robot.adapter import RoboDojoAdapter

    overrides = {**json.loads(params["adapter_overrides_json"]), "robodojo_root": "/unused"}
    return int(RoboDojoAdapter(load_profile(PROFILE), overrides)._cfg["port"])


@pytest.fixture(autouse=True)
def _fresh_registry():
    clear_sim_platform_cache()
    yield
    clear_sim_platform_cache()


def test_the_platform_is_registered_through_the_shared_registry():
    spec = get_sim_platform("robodojo")  # the ibrobot.sim_platforms entry point of this package
    assert spec.package == "robodojo_robot" and spec.world is None and spec.scene_task is None
    assert is_physics_selection(True, "robodojo")  # consumers run on the runtime's /clock


def test_the_embodiment_starts_the_direct_control_runtime_with_this_platforms_adapter():
    from robodojo_robot.embodiment import ADAPTER, ROBODOJO_EMBODIMENT, runtime_parameters

    profile = load_profile(PROFILE)
    embodiment = resolve_sim_embodiment(profile, PROFILE, get_sim_platform("robodojo"))
    assert embodiment is ROBODOJO_EMBODIMENT and isinstance(embodiment, SimEmbodiment)
    assert embodiment.render_description(profile, PROFILE) == ""  # no public model
    (node,) = embodiment.actions(profile, PROFILE, instance_id="x5_a")
    assert (node.node_package, node.node_executable) == ("robot_runtime_sim", "direct_control_runtime")
    params = runtime_parameters(profile, PROFILE, "x5_a")
    assert params["adapter"] == ADAPTER == "robodojo_robot.adapter:create_adapter"
    assert params["platform"] == "robodojo" and params["instance_id"] == "x5_a"
    assert params["use_sim_time"] is False  # the runtime is the clock source
    with pytest.raises(RuntimeError, match="peripherals"):
        embodiment.actions(profile, PROFILE, "fragment.yaml")


def test_policy_port_overrides_the_profile_port():
    from robodojo_robot.embodiment import runtime_parameters

    launch = _launch_module()
    profile = launch.apply_policy_port(load_profile(PROFILE), " 34960 ")
    assert _adapter_port(runtime_parameters(profile, PROFILE)) == 34960


def test_without_policy_port_the_profile_port_is_used():
    from robodojo_robot.embodiment import runtime_parameters

    launch = _launch_module()
    profile = launch.apply_policy_port(load_profile(PROFILE), "")
    port = int(load_profile(PROFILE)["simulation"]["robodojo"]["adapter_config"]["port"])
    assert _adapter_port(runtime_parameters(profile, PROFILE)) == port


def test_the_platform_spec_module_stays_import_light():
    code = (
        "import sys\n"
        "import robodojo_robot.platforms\n"
        "heavy = {'launch', 'launch_ros', 'rclpy', 'robodojo_robot.adapter', 'robot_runtime_sim.direct_control'}\n"
        "loaded = heavy & set(sys.modules)\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
