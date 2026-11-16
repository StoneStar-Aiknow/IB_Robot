"""Transport-only projection: stable consumer binding without an interface snapshot."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_config.interface_binding import InterfaceBindingError, required_interface_ids
from robot_config.loader import (
    TRANSPORT_ENDPOINT_KEYS,
    load_robot_config_dict,
    load_transport_robot_config,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
G1_ARM_PRESETS = PACKAGE_ROOT / "config" / "robots" / "unitree_g1_arm_presets.yaml"


def _robot_file(tmp_path: Path, name: str, robot: dict) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump({"robot": robot}, sort_keys=False), encoding="utf-8")
    return path


def _logical_robot(name: str, **embodied) -> dict:
    payload = {
        "name": name,
        "runtime": {"provider": "unitree_g1_robot", "profile": "g1_29dof_arm_presets", "require_model": True},
        "embodied": {
            "enabled": True,
            "runtime_named_motion": {
                "enabled": True,
                "interface": "motion.named",
                "kind": "action",
                "type": "ibrobot_msgs/action/ExecuteNamedMotion",
            },
            **embodied,
        },
    }
    return payload


def test_repo_g1_presets_projects_without_a_snapshot() -> None:
    projection = load_transport_robot_config(config_path=G1_ARM_PRESETS)

    assert projection.name == "unitree_g1_arm_presets"
    assert projection.interfaces_deferred is True
    assert projection.config_path == str(G1_ARM_PRESETS.resolve())
    raw = yaml.safe_load(G1_ARM_PRESETS.read_text(encoding="utf-8"))["robot"]
    # The strict loader needs a snapshot for this config; the projection does not.
    assert required_interface_ids(raw) == ["motion.named"]
    assert "interface_description" not in projection.embodied
    assert "endpoint" not in projection.embodied["runtime_named_motion"]
    assert projection.embodied["skill_catalog_profile"] == "unitree_g1_arm_presets"


def test_strict_loader_still_requires_the_interface_description() -> None:
    with pytest.raises(InterfaceBindingError) as error:
        load_robot_config_dict(G1_ARM_PRESETS)

    assert error.value.code == "description_required"
    assert error.value.path == "runtime.interface_description"


def test_name_path_and_environment_selectors_project_identically(monkeypatch) -> None:
    monkeypatch.delenv("ROBOT_NAME", raising=False)
    by_path = load_transport_robot_config(config_path=G1_ARM_PRESETS)
    by_name = load_transport_robot_config(config_name="unitree_g1_arm_presets")
    monkeypatch.setenv("ROBOT_CONFIG", "unitree_g1_arm_presets")
    by_environment = load_transport_robot_config()

    for projection in (by_name, by_environment):
        assert projection.name == by_path.name
        assert projection.embodied == by_path.embodied
        assert projection.config_path == by_path.config_path
        assert projection.config_sources == by_path.config_sources


def test_explicit_config_name_wins_over_the_environment(monkeypatch) -> None:
    monkeypatch.delenv("ROBOT_NAME", raising=False)
    monkeypatch.setenv("ROBOT_CONFIG", "unitree_g1")

    projection = load_transport_robot_config(config_name="unitree_g1_arm_presets")

    assert projection.name == "unitree_g1_arm_presets"


def test_explicit_endpoint_overrides_are_preserved(tmp_path) -> None:
    config = _robot_file(
        tmp_path,
        "overlay.yaml",
        _logical_robot(
            "overlay_robot",
            skill_gateway_status_service="/custom/gateway_status",
            execute_program_action="/custom/execute_program",
        ),
    )

    projection = load_transport_robot_config(config_path=config)

    assert projection.embodied["skill_gateway_status_service"] == "/custom/gateway_status"
    assert projection.embodied["execute_program_action"] == "/custom/execute_program"
    assert projection.config_sources == (str(config.resolve()),)


def test_base_config_inheritance_and_source_chain(tmp_path) -> None:
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "robot": {
                    "name": "base_robot",
                    "embodied": {
                        "skill_gateway_status_service": "/base/status",
                        "get_execution_service": "/base/execution",
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    overlay = _robot_file(
        tmp_path,
        "overlay.yaml",
        {
            "base_config": "base.yaml",
            "name": "overlay_robot",
            "embodied": {"skill_gateway_status_service": "/overlay/status"},
        },
    )

    projection = load_transport_robot_config(config_path=overlay)

    assert projection.name == "overlay_robot"
    assert projection.embodied["skill_gateway_status_service"] == "/overlay/status"
    assert projection.embodied["get_execution_service"] == "/base/execution"
    assert projection.config_sources == (str((tmp_path / "base.yaml").resolve()), str(overlay.resolve()))


def test_projection_needs_no_model_profile_or_directory(tmp_path) -> None:
    # require_model=true and a profile that does not exist anywhere: the
    # projection never reads model artifacts, profiles or vendor resources.
    config = _robot_file(
        tmp_path,
        "model_robot.yaml",
        _logical_robot("model_robot", get_execution_service="/model/get_execution"),
    )

    projection = load_transport_robot_config(config_path=config)

    assert projection.name == "model_robot"
    assert projection.embodied["get_execution_service"] == "/model/get_execution"
    with pytest.raises(InterfaceBindingError):
        load_robot_config_dict(config)


def test_non_string_endpoint_is_rejected(tmp_path) -> None:
    config = _robot_file(
        tmp_path,
        "mapping_endpoint.yaml",
        _logical_robot("mapping_robot", skill_action_name={"interface": "motion.named"}),
    )

    with pytest.raises(ValueError, match="embodied.skill_action_name must be a non-empty endpoint string"):
        load_transport_robot_config(config_path=config)


def test_missing_robot_section_is_rejected(tmp_path) -> None:
    path = tmp_path / "no_robot.yaml"
    path.write_text("other: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing 'robot' section"):
        load_transport_robot_config(config_path=path)


def test_missing_name_is_rejected(tmp_path) -> None:
    path = _robot_file(tmp_path, "no_name.yaml", {"embodied": {}})

    with pytest.raises(ValueError, match="missing 'name'"):
        load_transport_robot_config(config_path=path)


def test_projection_endpoint_key_contract_is_complete() -> None:
    # Every key the neutral transport contract reads must be a supported
    # projection key; the projection declares them once for the config layer.
    assert len(TRANSPORT_ENDPOINT_KEYS) == len(set(TRANSPORT_ENDPOINT_KEYS)) == 18
    assert "skill_gateway_status_service" in TRANSPORT_ENDPOINT_KEYS
    assert "execute_program_action" in TRANSPORT_ENDPOINT_KEYS
