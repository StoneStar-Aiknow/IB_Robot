"""Consumer transport binding for the stable configuration selectors."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from robot_client.transport import ENDPOINT_CONFIG_KEYS
from robot_config.interface_binding import InterfaceBindingError
from robot_config.loader import TRANSPORT_ENDPOINT_KEYS, load_transport_robot_config
from robot_skill_cli import catalog, cli

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
G1_ARM_PRESETS = PACKAGE_ROOT.parent.parent / "robot_config" / "config" / "robots" / "unitree_g1_arm_presets.yaml"


def _expected_transport(name: str = "unitree_g1_arm_presets"):
    projection = load_transport_robot_config(config_name=name)
    return catalog._gateway_transport(projection.embodied, robot_id=projection.name)


def test_endpoint_key_contract_matches_the_neutral_transport() -> None:
    assert set(TRANSPORT_ENDPOINT_KEYS) == set(ENDPOINT_CONFIG_KEYS)


def test_transport_context_binds_the_repo_g1_config_without_a_snapshot(monkeypatch) -> None:
    monkeypatch.delenv("ROBOT_CONFIG", raising=False)
    monkeypatch.delenv("ROBOT_NAME", raising=False)

    context, transport = catalog.load_transport_context(config_name="unitree_g1_arm_presets")

    assert context.view["robot_name"] == "unitree_g1_arm_presets"
    assert context.view["timeout_policy"]["rpc_timeout_sec"] > 0
    assert transport == _expected_transport()
    assert transport.status_service == "/embodied/get_skill_gateway_status"
    assert transport.robot_id == "unitree_g1_arm_presets"


def test_name_and_path_selectors_bind_the_same_target(monkeypatch) -> None:
    monkeypatch.delenv("ROBOT_CONFIG", raising=False)
    monkeypatch.delenv("ROBOT_NAME", raising=False)

    _, by_name = catalog.load_transport_context(config_name="unitree_g1_arm_presets")
    _, by_path = catalog.load_transport_context(config_path=G1_ARM_PRESETS)

    assert by_name == by_path


def test_environment_selector_binds_the_same_target(monkeypatch) -> None:
    monkeypatch.delenv("ROBOT_NAME", raising=False)
    monkeypatch.setenv("ROBOT_CONFIG", str(G1_ARM_PRESETS))

    _, by_environment = catalog.load_transport_context()

    assert by_environment == _expected_transport()


def test_cli_status_uses_the_stable_config_selector(monkeypatch, capsys) -> None:
    monkeypatch.delenv("ROBOT_CONFIG", raising=False)
    monkeypatch.delenv("ROBOT_NAME", raising=False)
    captured: dict[str, object] = {}

    def runner(_args, context, transport):
        captured["transport"] = transport
        captured["robot_name"] = context.view["robot_name"]
        return {"control_plane_ready": True, "robot_name": context.view["robot_name"]}

    monkeypatch.setattr(cli, "_run_runtime_command", Mock(side_effect=runner))

    assert cli.main(["--config-name", "unitree_g1_arm_presets", "status"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["data"] == {"control_plane_ready": True, "robot_name": "unitree_g1_arm_presets"}
    assert captured["robot_name"] == "unitree_g1_arm_presets"
    assert captured["transport"] == _expected_transport()


def test_strict_catalog_loader_still_requires_the_interface_snapshot(monkeypatch) -> None:
    monkeypatch.delenv("ROBOT_CONFIG", raising=False)
    monkeypatch.delenv("ROBOT_NAME", raising=False)

    with pytest.raises(InterfaceBindingError) as error:
        catalog.load_catalog_context(config_name="unitree_g1_arm_presets")

    assert error.value.code == "description_required"
