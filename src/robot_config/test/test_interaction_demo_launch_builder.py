"""Logical endpoint binding for the interaction demo's robot_config section.

The generator-side tests moved to robot_interaction_demo with the
interaction_demo entry (openspec change
``migrate-business-launch-builders``); this file keeps the config-shape and
binding-rejection tests, which only use robot_config APIs.
"""

from pathlib import Path

import pytest
from launch import LaunchContext
from launch_ros.utilities import evaluate_parameters

from robot_config.interface_binding import InterfaceBindingError, bind_robot_interfaces, required_interface_ids
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
    return build_description(load_profile(PROFILE), use_sim=True)


def test_demo_yaml_contains_only_deployment_and_logical_interface_binding():
    config = deferred_config()
    assert config["runtime"]["provider"] == "aimdk_robot"
    assert config["runtime"]["profile"] == "x2_ultra"
    demo = config["interaction_demo"]
    assert set(demo) == {"enabled", "service_name", "timeout_sec", "interfaces"}
    assert required_interface_ids(config) == ["speech.speak", "motion.named"]
    assert "commands" not in demo


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
