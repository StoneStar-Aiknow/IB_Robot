"""The Web server binds one deployment target from the robot configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_client.transport import gateway_transport
from robot_config.loader import load_transport_robot_config
from robot_skill_web.server import WebServer, bound_client

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
G1_ARM_PRESETS = PACKAGE_ROOT.parent.parent / "robot_config" / "config" / "robots" / "unitree_g1_arm_presets.yaml"


def _clear_selectors(monkeypatch) -> None:
    monkeypatch.delenv("ROBOT_CONFIG", raising=False)
    monkeypatch.delenv("ROBOT_NAME", raising=False)


def _expected_transport():
    projection = load_transport_robot_config(config_name="unitree_g1_arm_presets")
    return gateway_transport(projection.embodied, robot_id=projection.name, robot_name=projection.name)


def _digest(client) -> str:
    try:
        return client.transport.identity_digest()
    finally:
        client.close()


def test_bound_client_binds_the_repo_g1_deployment(monkeypatch) -> None:
    _clear_selectors(monkeypatch)

    client = bound_client(config_name="unitree_g1_arm_presets")
    try:
        session = client.session
        assert session["robot_id"] == "unitree_g1_arm_presets"
        assert session["robot_name"] == "unitree_g1_arm_presets"
        assert session["caller_scope"] == "robot_skill_cli"
        assert session["target_key"].endswith(client.transport.identity_digest())
        assert client.transport == _expected_transport()
        # Binding the config must not open a ROS bridge, start a provider or
        # contact the Gateway.
        assert client._bridge is None
    finally:
        client.close()


def test_bound_client_name_path_and_environment_select_the_same_target(monkeypatch) -> None:
    _clear_selectors(monkeypatch)
    expected = _digest(bound_client(config_name="unitree_g1_arm_presets"))
    by_path = _digest(bound_client(config_path=G1_ARM_PRESETS))
    monkeypatch.setenv("ROBOT_CONFIG", str(G1_ARM_PRESETS))
    by_environment = _digest(bound_client())

    assert by_path == expected
    assert by_environment == expected


def test_bound_client_requires_a_deployment_config(monkeypatch) -> None:
    _clear_selectors(monkeypatch)

    with pytest.raises(ValueError, match="Robot configuration name or path is required"):
        bound_client()


def test_webserver_default_client_is_the_bound_target(monkeypatch, tmp_path) -> None:
    _clear_selectors(monkeypatch)

    server = WebServer(
        config_name="unitree_g1_arm_presets",
        port=8765,
        submission_directory=tmp_path / "submissions",
    )
    try:
        assert server.state.client.session["robot_name"] == "unitree_g1_arm_presets"
        assert server.state.client.transport.status_service == "/embodied/get_skill_gateway_status"
    finally:
        server.close()


def test_bound_client_rejects_a_non_string_endpoint(monkeypatch, tmp_path) -> None:
    _clear_selectors(monkeypatch)
    config = tmp_path / "bad_endpoint.yaml"
    config.write_text(
        "robot:\n  name: bad_robot\n  embodied:\n    skill_action_name:\n      interface: motion.named\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="embodied.skill_action_name"):
        bound_client(config_path=config)
