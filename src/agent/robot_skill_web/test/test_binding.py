"""The Web server binds one deployment target from the robot configuration."""

from __future__ import annotations

import pytest

from robot_skill_web.server import WebServer, bound_client


def test_bound_client_uses_the_resolved_deployment_config(monkeypatch, tmp_path):
    import robot_config.config_path
    import robot_config.loader

    resolved = tmp_path / "robot.yaml"
    monkeypatch.setattr(robot_config.config_path, "resolve_robot_config_path", lambda **_kwargs: resolved)
    monkeypatch.setattr(
        robot_config.loader,
        "load_robot_config_dict",
        lambda _path: {"name": "lab_g1", "embodied": {"validate_skill_service": "/lab/validate_skill"}},
    )
    client = bound_client(config_name="lab_g1")
    try:
        session = client.session
        assert session["robot_id"] == "lab_g1"
        assert session["robot_name"] == "lab_g1"
        assert session["caller_scope"] == "robot_skill_cli"
        assert session["target_key"].endswith(client.transport.identity_digest())
        assert client.transport.validate_skill_service == "/lab/validate_skill"
        # Starting the Web server must not open a ROS bridge or send anything.
        assert client._bridge is None
    finally:
        client.close()


def test_bound_client_requires_a_deployment_config(monkeypatch):
    import robot_config.config_path

    def missing(**_kwargs):
        raise ValueError("Robot configuration name or path is required")

    monkeypatch.setattr(robot_config.config_path, "resolve_robot_config_path", missing)
    with pytest.raises(ValueError, match="Robot configuration name or path is required"):
        bound_client()


def test_webserver_default_client_is_the_bound_target(monkeypatch, tmp_path):
    import robot_config.config_path
    import robot_config.loader

    monkeypatch.setattr(robot_config.config_path, "resolve_robot_config_path", lambda **_kwargs: tmp_path / "r.yaml")
    monkeypatch.setattr(robot_config.loader, "load_robot_config_dict", lambda _path: {"name": "bound_robot"})
    server = WebServer(config_name="bound_robot", port=8765)
    try:
        assert server.state.client.session["robot_name"] == "bound_robot"
    finally:
        server.state.close()
