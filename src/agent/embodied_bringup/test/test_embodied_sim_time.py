"""The embodied nodes' simulation-clock pass-through (lekiwi-nav-grasp R3).

The runtime facade runs on simulation time on physics platforms and the
Gateway compares its own ROS clock against runtime-status stamps, so every
embodied node that exchanges stamped data must use the SAME clock decision
the base launch's ``task_executor`` uses — which the launch entry derives
through the base launch's OWN target/platform resolution
(``robot_config.launch_registry.preparation.resolve_target_and_backend``,
review r1 item 2), never a private re-parse of the raw ``use_sim`` argument.

These tests pin the builder output for sim, hardware and mock, and the
launch entry's resolution for the entry combinations: config-declared
simulation with EMPTY ``use_sim``, the providerless default platform, a
``sim_platform`` override alone, and hardware.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from embodied_bringup.launch_builders.embodied import generate_embodied_nodes

_LAUNCH = Path(__file__).resolve().parents[1] / "launch" / "embodied_pipeline.launch.py"


def _launch_module():
    spec = importlib.util.spec_from_file_location("embodied_pipeline_launch_under_test", _LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeLaunchContext:
    """The launch context face ``_resolve_embodied_clock`` reads: the launch
    configurations only, exactly what ``launch_setup`` would pass."""

    def __init__(self, configurations: dict[str, str]):
        self.launch_configurations = configurations


def _resolve(configurations: dict[str, str], config: dict):
    return _launch_module()._resolve_embodied_clock(_FakeLaunchContext(configurations), config)


def _full_closure_config() -> dict:
    """A providerless config whose closure contains every stamped consumer:
    the Gateway and the safety guard (common params), the perception
    service, the GraspGen planner and verifier, the pick executor, the
    place executor and the imitation executor."""
    return {
        "name": "sim_time_probe_robot",
        "embodied": {
            "enabled": True,
            "entry_mode": "hermes",
            "execution": {},
            "perception": {"enabled": True},
            "imitate_human_motion": {"enabled": True},
            "sound_orientation": {"enabled": False},
        },
        "grasp_execution": {
            "enabled": True,
            "auto_start_dependencies": True,
            "camera": {},
            "planner_node": {},
            "verifier_node": {},
        },
        "placement_execution": {"enabled": True},
        "joints": {"arm": ["1", "2", "3", "4", "5"], "gripper": ["6"]},
    }


def _nodes_by_name(nodes):
    """The nodes' first parameter mapping, with launch substitutions
    normalised to plain values (bools pass through, like the existing
    builder tests' ``_normalize_launch_param_mapping``)."""
    by_name = {}
    for node in nodes:
        raw_params = node.__dict__.get("_Node__parameters")[0]
        normalized = {}
        for raw_key, raw_value in raw_params.items():
            key = "".join(getattr(item, "text", str(item)) for item in raw_key)
            if isinstance(raw_value, tuple):
                value = "".join(getattr(item, "text", str(item)) for item in raw_value)
            else:
                value = raw_value
            normalized[key] = value
        by_name[node.__dict__.get("_Node__node_name")] = normalized
    return by_name


def _pipeline_config() -> dict:
    config = _full_closure_config()
    config["grasp_execution"] = {"enabled": True, "backend": "graspgen"}
    return config


@pytest.mark.parametrize(
    "use_sim,sim_platform,expected",
    [
        (True, "gazebo", True),
        (True, "mujoco", True),
        (True, "mock", False),
        (False, "", False),
    ],
)
def test_every_stamped_consumer_carries_the_clock_decision(use_sim, sim_platform, expected):
    nodes = generate_embodied_nodes(
        _full_closure_config(),
        active_control_mode="moveit_planning",
        use_sim=use_sim,
        use_sim_time=expected,
    )
    params = _nodes_by_name(nodes)
    stamped_consumers = (
        "skill_executor_node",  # the Gateway
        "safety_guard_node",
        "perception_service_node",
        "grasp_planner",
        "grasp_verifier",
        "pick_executor_node",
        "placement_executor_node",
        "imitate_human_motion_executor_node",
    )
    for name in stamped_consumers:
        assert name in params, f"the fixture config must start {name}"
        assert params[name].get("use_sim_time") is expected, f"{name} use_sim_time must be {expected}"


@pytest.mark.parametrize(
    "use_sim,sim_platform,expected",
    [(True, "gazebo", True), (True, "mock", False), (False, "", False)],
)
def test_the_grasp_pipeline_node_carries_the_clock_decision(use_sim, sim_platform, expected):
    nodes = generate_embodied_nodes(
        _pipeline_config(),
        active_control_mode="moveit_planning",
        use_sim=use_sim,
        use_sim_time=expected,
    )
    params = _nodes_by_name(nodes)
    assert "grasp_pipeline_node" in params
    assert params["grasp_pipeline_node"].get("use_sim_time") is expected
    # a robot that names a grasp backend runs the pipeline INSTEAD of the
    # primitive executor closure
    assert "pick_executor_node" not in params
    assert "grasp_planner" not in params


def test_the_default_clock_is_wall_time():
    """The builder's default (a caller that passes nothing) is wall time —
    hardware semantics; only an explicit decision turns sim time on."""
    params = _nodes_by_name(generate_embodied_nodes(_full_closure_config(), active_control_mode="moveit_planning"))
    assert params["skill_executor_node"]["use_sim_time"] is False
    assert params["pick_executor_node"]["use_sim_time"] is False


def test_the_sound_orientation_node_carries_the_clock_decision():
    config = _full_closure_config()
    config["embodied"]["idle_behaviors"] = {"sound_orientation": {"enabled": True}}
    nodes = generate_embodied_nodes(
        config,
        active_control_mode="moveit_planning",
        use_sim=True,
        use_sim_time=True,
    )
    params = _nodes_by_name(nodes)
    assert params["sound_orientation_node"]["use_sim_time"] is True


# ---------------------------------------------------------------------------
# Review r1 item 2: the launch entry resolves the clock through the base
# launch's own target/platform resolution — the entry combinations
# ---------------------------------------------------------------------------


def test_config_declared_simulation_with_empty_use_sim_runs_the_sim_clock():
    """EMPTY ``use_sim`` infers the target from the configuration (the base
    launch's rule): a config-declared simulation (runtime.target=simulation
    on the mujoco platform, runtime.provider present) runs the embodied
    nodes on the simulator's clock. The old raw parse read use_sim="" as
    false and gave the same launch wall-time nodes."""
    config = _full_closure_config()
    config["runtime"] = {"target": "simulation", "provider": "lekiwi_sdk"}
    config["simulation"] = {"platform": "mujoco"}
    target_state = _resolve({"use_sim": "", "sim_platform": "", "runtime_target": ""}, config)
    assert target_state["use_sim"] is True
    assert target_state["sim_platform"] == "mujoco"
    assert target_state["node_use_sim_time"] is True


def test_config_declared_simulation_with_explicit_false_is_hardware():
    """``use_sim:=false`` with the same config-declared simulation is the
    hardware combination the base launch rejects — the same error here
    (never a silent wall-time sim)."""
    from robot_config.runtime_target import RuntimeTargetError

    config = _full_closure_config()
    config["runtime"] = {"target": "simulation", "provider": "lekiwi_sdk"}
    config["simulation"] = {"platform": "mujoco"}
    with pytest.raises(RuntimeTargetError):
        _resolve({"use_sim": "false", "sim_platform": "", "runtime_target": ""}, config)


def test_the_providerless_default_platform_is_gazebo():
    """A providerless configuration without ``simulation.platform``:
    ``use_sim:=true`` alone resolves the platform to the providerless
    DEFAULT ``gazebo`` — a sim-clock launch. The old raw parse fell back to
    the empty platform and gave the nodes wall time while the base launch
    simulated gazebo."""
    target_state = _resolve({"use_sim": "true", "sim_platform": "", "runtime_target": ""}, _full_closure_config())
    assert target_state["use_sim"] is True
    assert target_state["sim_platform"] == "gazebo"
    assert target_state["node_use_sim_time"] is True


def test_a_sim_platform_override_alone_selects_simulation():
    """``sim_platform:=gazebo`` with EMPTY ``use_sim`` selects simulation on
    that platform (the base launch's rule): the embodied nodes follow the
    simulator's clock. The old raw parse read the empty use_sim as false."""
    target_state = _resolve({"use_sim": "", "sim_platform": "gazebo", "runtime_target": ""}, _full_closure_config())
    assert target_state["use_sim"] is True
    assert target_state["sim_platform"] == "gazebo"
    assert target_state["node_use_sim_time"] is True


def test_the_sim_platform_overrides_the_configured_platform():
    config = _full_closure_config()
    config["simulation"] = {"platform": "mock"}
    target_state = _resolve({"use_sim": "true", "sim_platform": "gazebo", "runtime_target": ""}, config)
    assert target_state["sim_platform"] == "gazebo"
    assert target_state["node_use_sim_time"] is True
    assert config["simulation"]["platform"] == "gazebo"  # the resolution the base launch also applies


def test_hardware_and_mock_stay_on_wall_time():
    """Hardware (``use_sim:=false``) and the mock platform (wall-clock
    backend) keep every embodied node on wall time — the task_executor
    decision."""
    target_state = _resolve({"use_sim": "false", "sim_platform": "", "runtime_target": ""}, _full_closure_config())
    assert target_state["use_sim"] is False
    assert target_state["node_use_sim_time"] is False
    config = _full_closure_config()
    config["simulation"] = {"platform": "mock"}
    target_state = _resolve({"use_sim": "true", "sim_platform": "", "runtime_target": ""}, config)
    assert target_state["use_sim"] is True
    assert target_state["sim_platform"] == "mock"
    assert target_state["node_use_sim_time"] is False


def test_the_provider_sdk_platform_provides_no_clock():
    """The provider transport (``sdk``) provides no ``/clock`` as far as the
    embodied nodes are concerned — wall time, exactly like task_executor."""
    config = _full_closure_config()
    config["runtime"] = {"provider": "lekiwi_sdk"}
    config["simulation"] = {"platform": "sdk"}
    target_state = _resolve({"use_sim": "true", "sim_platform": "", "runtime_target": ""}, config)
    assert target_state["use_sim"] is True
    assert target_state["sim_platform"] == "sdk"
    assert target_state["node_use_sim_time"] is False
