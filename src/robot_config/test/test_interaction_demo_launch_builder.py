"""Logical endpoint binding and unified launch wiring for the interaction demo."""

import importlib.util
from pathlib import Path

import pytest
from launch import LaunchContext
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters

from robot_config.interface_binding import InterfaceBindingError, bind_robot_interfaces, required_interface_ids
from robot_config.launch_builders.interaction_demo import generate_interaction_demo_nodes
from robot_config.loader import load_robot_config_dict
from robot_runtime.interface_description import build_description
from robot_runtime.profile import load_profile

CONFIG = Path(__file__).resolve().parents[1] / "config/robots/aimdk_x2_interaction_demo.yaml"
PROFILE = Path(__file__).resolve().parents[2] / "robots/aimdk/aimdk_robot/profiles/x2_ultra.yaml"


def parameters(node):
    return evaluate_parameters(LaunchContext(), node._Node__parameters)[0]


def deferred_config():
    return load_robot_config_dict(CONFIG, defer_interface_binding=True)


def descriptor():
    return build_description(load_profile(PROFILE), simulated=True)


def test_demo_yaml_contains_only_deployment_and_logical_interface_binding():
    config = deferred_config()
    assert config["runtime"]["provider"] == "aimdk_robot"
    assert config["runtime"]["profile"] == "x2_ultra"
    demo = config["interaction_demo"]
    assert set(demo) == {"enabled", "service_name", "timeout_sec", "interfaces"}
    assert required_interface_ids(config) == ["speech.speak", "motion.named"]
    assert "commands" not in demo


def test_runtime_description_resolves_endpoints_before_launching_business_node():
    bound = bind_robot_interfaces(deferred_config(), descriptor(), require_ready=True)
    [node] = generate_interaction_demo_nodes(bound)
    assert node.node_package == "robot_interaction_demo"
    params = parameters(node)
    assert params == {
        "speech_service": "/speech/speak",
        "named_motion_action": "/motion/execute_named",
        "timeout_sec": 40.0,
        "service_name": "/interaction_demo/execute",
    }
    for role, logical_id in (("speech", "speech.speak"), ("named_motion", "motion.named")):
        source = bound["interaction_demo"]["interfaces"][role]["_interface_source"]
        assert source["id"] == logical_id
        assert source["direction"] == "serve"


def test_unbound_business_node_fails_closed():
    with pytest.raises(ValueError, match="not resolved"):
        generate_interaction_demo_nodes(deferred_config())


@pytest.mark.parametrize(
    ("role", "field", "value", "code"),
    [
        ("speech", "interface", "speech.missing", "unknown_interface"),
        ("speech", "kind", "action", "kind_mismatch"),
        ("speech", "type", "std_srvs/srv/Trigger", "type_mismatch"),
        ("named_motion", "interface", "speech.speak", "kind_mismatch"),
    ],
)
def test_invalid_logical_binding_is_rejected(role, field, value, code):
    config = deferred_config()
    config["interaction_demo"]["interfaces"][role][field] = value
    with pytest.raises(InterfaceBindingError) as error:
        bind_robot_interfaces(config, descriptor(), require_ready=True)
    assert error.value.code == code


def test_base_x2_does_not_launch_demo():
    assert generate_interaction_demo_nodes(load_robot_config_dict(CONFIG.with_name("aimdk_x2.yaml"))) == []


@pytest.mark.parametrize("timeout", [True, "40", 0, float("nan")])
def test_builder_rejects_invalid_timeout(timeout):
    config = bind_robot_interfaces(deferred_config(), descriptor(), require_ready=True)
    config["interaction_demo"]["timeout_sec"] = timeout
    with pytest.raises(ValueError, match="timeout_sec"):
        generate_interaction_demo_nodes(config)


def test_unified_launch_constructs_demo_from_live_bound_config(monkeypatch):
    path = CONFIG.parents[2] / "launch/robot.launch.py"
    spec = importlib.util.spec_from_file_location("interaction_demo_robot_launch", path)
    robot_launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(robot_launch)
    captured = {}
    marker = Node(package="robot_runtime", executable="wait_for_runtime")

    def fake_bound_actions(config, construct, **_kwargs):
        captured["required"] = required_interface_ids(config)
        captured["consumers"] = construct(bind_robot_interfaces(config, descriptor(), require_ready=True))
        return [marker]

    monkeypatch.setattr(robot_launch, "generate_bound_runtime_actions", fake_bound_actions)
    context = LaunchContext()
    context.launch_configurations.update(config_path=str(CONFIG), use_sim="true", record="false")
    assert robot_launch.launch_setup(context) == [marker]
    assert captured["required"] == ["speech.speak", "motion.named"]
    [demo] = [
        action
        for action in captured["consumers"]
        if isinstance(action, Node) and action.node_package == "robot_interaction_demo"
    ]
    assert parameters(demo)["speech_service"] == "/speech/speak"
