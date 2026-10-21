import json

import pytest
import yaml

from ibrobot_tracing.topology import bind_robot_topology
from robot_config import loader
from robot_config.loader import load_robot_section
from robot_config.tracing_topology import main


def test_manual_topology_matches_expanded_overlay(tmp_path, capsys):
    base = {
        "robot": {
            "name": "base",
            "default_control_mode": "model_inference",
            "contract": {
                "observations": [{"key": "observation.state", "topic": "/joint_states"}],
                "actions": [{"key": "action", "publish": {"topic": "/command"}}],
            },
            "control_modes": {
                "model_inference": {
                    "inference": {"pipelines": {"policy": {"execution_mode": "monolithic"}}},
                    "executor": {"inference_pipeline": "policy"},
                }
            },
        }
    }
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    child = tmp_path / "child.yaml"
    child.write_text(yaml.safe_dump({"robot": {"name": "child", "base_config": "base"}}))

    assert main([str(child), "--control-mode", "model_inference"]) == 0
    actual = json.loads(capsys.readouterr().out)
    config_path, config = load_robot_section(child)
    config["_config_path"] = str(config_path)
    expected = bind_robot_topology(config, control_mode="model_inference").to_dict()
    expected["metadata"].update(source="robot_config_yaml", provenance="declared", runtime_verified=False)
    assert actual == {**expected, "schema_version": 2}
    assert actual["robot_name"] == "child"
    assert actual["metadata"]["pipeline_id"] == "policy"
    assert any(edge["name"] == "/joint_states" for edge in actual["edges"])
    assert any(edge["name"] == "/command" for edge in actual["edges"])
    assert all(component["provenance"] == "declared" for component in actual["components"])
    assert "started_nodes" not in actual["metadata"]


def test_declaration_does_not_require_runtime_model_availability(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(yaml.safe_dump({"robot": {"name": "offline", "default_control_mode": "teleop"}}))

    def unavailable_model(_config):
        raise loader.PerceptionRuntimeConfigError("model bundle is unavailable")

    monkeypatch.setattr(loader, "parse_perception_runtime_config", unavailable_model)
    with pytest.raises(ValueError, match="model bundle is unavailable"):
        loader.load_robot_config_dict(config_path)

    assert main([str(config_path)]) == 0
    actual = json.loads(capsys.readouterr().out)
    assert actual["control_mode"] == "teleop"
    assert actual["metadata"]["runtime_verified"] is False
    assert actual["metadata"]["provenance"] == "declared"


@pytest.mark.parametrize("target", ["hardware", "simulation", "benchmark"])
@pytest.mark.parametrize("scheduler_enabled", [None, False, True])
@pytest.mark.parametrize("inference_enabled", [False, True])
def test_declared_dispatch_path_uses_robot_scheduling_rules(
    tmp_path, capsys, target, scheduler_enabled, inference_enabled
):
    """Configuration activation belongs here; tracing only tests rendering the resolved path."""
    config = {
        "name": "offline",
        "runtime": {"target": target},
        "control_modes": {"policy": {"inference": {"enabled": inference_enabled}}},
        "default_control_mode": "teleop",
    }
    if scheduler_enabled is not None:
        config["runtime_scheduler"] = {"enable": scheduler_enabled}
    if target == "benchmark":
        config["benchmark"] = {"type": "libero", "adapter": "libero"}
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(yaml.safe_dump({"robot": config}))

    assert main([str(config_path), "--control-mode", "policy"]) == 0
    topology = json.loads(capsys.readouterr().out)
    scheduled = scheduler_enabled is True and inference_enabled and target == "hardware"
    components = {component["component_id"]: component for component in topology["components"]}
    assert ("global_scheduler" in components) is scheduled
    assert (topology["metadata"].get("dispatch_path") == "scheduled") is scheduled
    assert components["action_dispatcher"]["executable"] == (
        "scheduled_action_dispatcher_node" if scheduled else "action_dispatcher_node"
    )
    assert topology["metadata"]["runtime_verified"] is False


def test_declared_topology_uses_target_without_consulting_yaml_use_sim(tmp_path, capsys):
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "robot": {
                    "name": "offline",
                    "runtime": {"target": "hardware"},
                    "use_sim": True,
                    "runtime_scheduler": {"enable": True},
                    "control_modes": {"model_inference": {"inference": {"enabled": True}}},
                }
            }
        )
    )
    assert main([str(config_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["metadata"]["dispatch_path"] == "scheduled"


@pytest.mark.parametrize("reference", ["missing", "child", "../outside"])
def test_invalid_base_fails_without_emitting_manifest(tmp_path, capsys, reference):
    child = tmp_path / "child.yaml"
    child.write_text(yaml.safe_dump({"robot": {"name": "child", "base_config": reference}}))
    assert main([str(child)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Error:" in output.err


def test_invalid_yaml_fails_without_emitting_manifest(tmp_path, capsys):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("robot: [")
    assert main([str(config_path)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Error:" in output.err
