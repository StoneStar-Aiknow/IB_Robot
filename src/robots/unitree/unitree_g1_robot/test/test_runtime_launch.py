"""Unitree G1 runtime launch setup generation and peripheral fragments.

runtime-deployment-peripherals task 4.2: the launch setup must generate —
``build_description`` is called with ``use_sim=`` (the ``simulated=`` keyword
mismatched the signature since the use_sim rename and made every launch-setup
render raise ``TypeError``).

Task 4.3 (D7): a deployment fragment disabling the declared ``head_front``
RealSense camera removes it from the description the runtime node serves and
from the synthetic perception actions; an empty fragment is unchanged.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import launch
import yaml
from launch_ros.actions import Node

_PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "g1_29dof_dex3.yaml"
_LAUNCH = Path(__file__).resolve().parents[1] / "launch" / "runtime.launch.py"


def _runtime_launch_module():
    spec = importlib.util.spec_from_file_location("unitree_g1_runtime_launch_under_test", _LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render(peripherals_file: str, simulated: str = "true"):
    context = launch.LaunchContext()
    context.launch_configurations.update(
        {
            "profile": str(_PROFILE),
            "use_sim": "",
            "sim_platform": "",
            "simulated": simulated,
            "instance_id": "",
            "initial_mode": "",
            "network_interface": "",
            "peripherals": peripherals_file,
        }
    )
    return _runtime_launch_module()._launch_setup(context)


def _node_params(node):
    from launch import LaunchContext
    from launch_ros.utilities import evaluate_parameters

    params = {}
    for entry in evaluate_parameters(LaunchContext(), node._Node__parameters):
        params.update(entry)
    return params


def test_launch_setup_generates_on_the_simulated_path():
    """The pre-existing build_description(simulated=) keyword mismatch made
    every launch-setup render fail with TypeError; with use_sim= the setup
    generates (task 4.2)."""
    actions = _render("")
    executables = {node._Node__node_executable for node in actions if isinstance(node, Node)}
    assert "runtime_node" in executables
    assert "vendor_mock_node" in executables


def test_launch_setup_applies_the_fragment_to_the_description(tmp_path):
    """D7: a fragment disabling the declared head_front camera removes it
    from the description the runtime node serves and from the synthetic
    perception publishers; an empty fragment changes nothing."""
    empty = _render("")
    runtime = next(node for node in empty if isinstance(node, Node) and node._Node__node_executable == "runtime_node")
    empty_description = json.loads(_node_params(runtime)["interface_description_json"])
    assert any(name.startswith("camera.head_front.") for name in empty_description["interfaces"])
    empty_synthetic = next(
        node for node in empty if isinstance(node, Node) and node._Node__node_executable == "synthetic_perception"
    )
    assert any(
        name.startswith("camera.head_front.")
        for name in json.loads(_node_params(empty_synthetic)["description_json"])["interfaces"]
    )

    fragment = tmp_path / "deployment_peripherals.yaml"
    fragment.write_text(
        yaml.safe_dump({"peripherals": [{"type": "camera", "name": "head_front", "disabled": True}]}),
        encoding="utf-8",
    )
    disabled = _render(str(fragment))
    runtime = next(
        node for node in disabled if isinstance(node, Node) and node._Node__node_executable == "runtime_node"
    )
    description = json.loads(_node_params(runtime)["interface_description_json"])
    assert not any(name.startswith("camera.head_front.") for name in description["interfaces"])
    synthetic = next(
        node for node in disabled if isinstance(node, Node) and node._Node__node_executable == "synthetic_perception"
    )
    assert not any(
        name.startswith("camera.head_front.")
        for name in json.loads(_node_params(synthetic)["description_json"])["interfaces"]
    )
    # the lidar surface stays
    assert "lidar.chest_front.points" in description["interfaces"]


def test_launch_declares_the_peripherals_argument():
    """D7 (m3): the provider's real launch description declares the
    peripherals argument every provider contract requires."""
    from launch.actions import DeclareLaunchArgument

    def text(value):
        return "".join(part.text if hasattr(part, "text") else str(part) for part in value)

    description = _runtime_launch_module().generate_launch_description()
    names = {
        text(action._DeclareLaunchArgument__name)
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    assert "peripherals" in names
