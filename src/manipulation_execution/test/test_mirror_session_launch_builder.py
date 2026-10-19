"""Unit tests for the mirror-session launch builder contributed by this package.

Openspec change ``decentralize-launch-builder-registration`` (design decision
4): the ``mirror_session`` launch wiring moved verbatim from
``robot_config/launch_builders/mirror_session.py`` to this package's
``launch_builder.py`` and registers through the ``ibrobot.launch_builders``
entry point. These tests pin the migrated behavior itself; the entry point
discovery and the end-to-end plan participation are covered by
``test_mirror_session_entry_point.py``.

The expected node parameters mirror what the pre-migration module produced
for the same synthetic bound configuration (recorded against the f0aded6e
tree before the move).
"""

from __future__ import annotations

import json

import pytest
import yaml
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node

from manipulation_execution.launch_builders.mirror_session import (
    MIRROR_SESSION_BUILDER,
    build_mirror_session,
    generate_mirror_session_nodes,
)
from robot_config.launch_builders.registry import (
    BuildResult,
    LaunchBuildContext,
    resolve_builder_factory,
)

_BOUND_DESCRIPTION = {
    "interfaces": {
        "joint.arm_stream": {
            "topic": "/arm/stream",
            "kind": "topic",
            "direction": "subscribe",
            "message_type": "std_msgs/msg/Float64MultiArray",
        },
        "camera.head.color_compressed": {
            "endpoint": "/camera/head/image_compressed",
            "kind": "topic",
            "direction": "publish",
            "message_type": "sensor_msgs/msg/CompressedImage",
        },
        "runtime.get_status": {"kind": "service", "type": "ibrobot_msgs/srv/GetRuntimeStatus"},
    }
}


def _node_parameters(node: Node) -> dict:
    def text(value):
        return "".join(item.text if hasattr(item, "text") else str(item) for item in value)

    def decode(value):
        if not isinstance(value, tuple):
            return value
        if all(isinstance(item, list) for item in value):
            return [decode(tuple(item)) for item in value]
        try:
            return yaml.safe_load(text(value))
        except yaml.YAMLError:
            return text(value)

    return {text(key): decode(value) for key, value in node._Node__parameters[0].items()}


def _config(**mirror_overrides) -> dict:
    config = {
        "name": "mirror_probe",
        "skill_required_control_mode": "mirror_mode",
        "control_modes": {
            "mirror_mode": {
                "runtime_mode": "policy_stream",
                "inference": {"enabled": False},
                "executor": {"enabled": False},
            },
        },
        "runtime": {"interface_description": _BOUND_DESCRIPTION},
        "mirror_session": {
            "enabled": True,
            "publish_enabled": False,
            "control_mode": "mirror_mode",
            "stop_timeout_sec": 4.5,
            "publish_rate_hz": 50.0,
            "status_period_sec": 1.0,
            "interfaces": {},
        },
    }
    config["mirror_session"].update(mirror_overrides)
    return config


class TestSpecDeclaration:
    def test_spec_shape_is_unchanged_by_the_migration(self):
        """id / factory string / runtime_ready timing — and, deliberately,
        no ``provides`` (typed outputs are builtin-only), no arguments and
        no hooks: the entry is exactly the one that used to close the
        builtin table."""
        assert MIRROR_SESSION_BUILDER.id == "mirror_session"
        assert (
            MIRROR_SESSION_BUILDER.factory
            == "manipulation_execution.launch_builders.mirror_session:build_mirror_session"
        )
        assert MIRROR_SESSION_BUILDER.timing == "runtime_ready"
        assert MIRROR_SESSION_BUILDER.provides == ()
        assert MIRROR_SESSION_BUILDER.arguments == ()
        assert MIRROR_SESSION_BUILDER.hooks == ()
        assert MIRROR_SESSION_BUILDER.enabled is None
        assert MIRROR_SESSION_BUILDER.mock_skip_key is None

    def test_factory_string_resolves_lazily_to_the_source_module(self, monkeypatch):
        """The factory stays a ``module:attribute`` STRING resolved on every
        call, so patching the source module keeps working as a seam."""
        assert resolve_builder_factory(MIRROR_SESSION_BUILDER.factory) is build_mirror_session

        def replacement(_context):
            return BuildResult()

        monkeypatch.setattr(
            "manipulation_execution.launch_builders.mirror_session.build_mirror_session",
            replacement,
        )
        assert resolve_builder_factory(MIRROR_SESSION_BUILDER.factory) is replacement

    def test_build_wraps_the_generator_actions(self, monkeypatch):
        marker = Node(package="p", executable="e")
        monkeypatch.setattr(
            "manipulation_execution.launch_builders.mirror_session.generate_mirror_session_nodes",
            lambda robot_config: [marker],
        )
        result = build_mirror_session(LaunchBuildContext(robot_config={"mirror_session": {"enabled": False}}))
        assert isinstance(result, BuildResult)
        assert result.actions == [marker]


class TestGenerateNodes:
    def test_disabled_session_builds_no_actions(self):
        assert generate_mirror_session_nodes({"mirror_session": {"enabled": False}}) == []
        assert generate_mirror_session_nodes({}) == []

    def test_enabled_session_builds_node_and_release_handler(self):
        """The migrated builder produces exactly the pre-migration structure:
        one session node plus one process-exit handler running the release
        node, with the parameters recorded from the f0aded6e module."""
        actions = generate_mirror_session_nodes(_config())

        assert len(actions) == 2
        session, handler = actions
        assert isinstance(session, Node)
        # Private accessors: the public properties only resolve after launch
        # executes the action (same pattern as robot_config's builder tests).
        assert session._Node__package == "manipulation_execution"
        assert session._Node__node_executable == "mirror_session_node"
        assert session._Node__node_name == "mirror_session"
        assert isinstance(handler, RegisterEventHandler)
        exit_handler = handler._RegisterEventHandler__event_handler
        assert isinstance(exit_handler, OnProcessExit)
        # The handler is bound to the session node (release runs on its exit).
        assert exit_handler._OnActionEventBase__action_matcher is session

        parameters = _node_parameters(session)
        assert parameters["policy_mode"] == "policy_stream"
        assert parameters["publish_enabled"] is False
        assert parameters["stop_timeout_sec"] == 4.5
        assert parameters["publish_rate_hz"] == 50.0
        assert parameters["status_period_sec"] == 1.0
        assert parameters["runtime_description_json"] == json.dumps(_BOUND_DESCRIPTION, sort_keys=True)
        # No motion_imitation_stream block: the stream topics stay empty.
        assert parameters["pear_frame_topic"] == ""
        assert parameters["actor_state_topic"] == ""
        assert parameters["camera_topic"] == ""
        assert parameters["mock_preview_topic"] == ""
        assert parameters["mock_play_service"] == ""

    def test_stream_topics_are_wired_from_motion_imitation_stream(self):
        """With the stream and mock playback enabled, the session's input
        topics follow ``motion_imitation_stream`` exactly as before."""
        config = _config()
        config["motion_imitation_stream"] = {
            "enabled": True,
            "frame_topic": "/stream/pear",
            "status_topic": "/stream/status",
            "actor_state_topic": "/stream/actor",
            "frame_source": {"interface": "camera.head.color_compressed", "decoder": "mock_cpu"},
            "yolox_detect_service": "/perception/yolox",
            "pear_parameters_service": "/perception/pear",
            "mock_playback": {
                "enabled": True,
                "preview_topic": "/mock/preview",
                "publish_rate_hz": 50.0,
                "input_staleness_sec": 0.5,
                "loop": True,
                "status_period_sec": 5.0,
            },
        }

        session, _ = generate_mirror_session_nodes(config)
        parameters = _node_parameters(session)
        assert parameters["pear_frame_topic"] == "/stream/pear"
        assert parameters["actor_state_topic"] == "/stream/actor"
        assert parameters["camera_topic"] == "/camera/head/image_compressed"
        assert parameters["mock_preview_topic"] == "/mock/preview"
        assert parameters["mock_play_service"] == "/motion_imitation_mock/play"

    def test_publish_enabled_flows_through(self):
        session, _ = generate_mirror_session_nodes(_config(publish_enabled=True))
        assert _node_parameters(session)["publish_enabled"] is True


class TestGenerateNodesValidation:
    """The builder's config checks, migrated verbatim: each failure names
    the offending key instead of starting a half-wired node."""

    def test_unknown_keys_are_rejected(self):
        with pytest.raises(ValueError, match="unknown keys \\['nope'\\]"):
            generate_mirror_session_nodes(_config(nope=True))

    def test_control_mode_must_equal_skill_required_control_mode(self):
        with pytest.raises(ValueError, match="must equal skill_required_control_mode"):
            generate_mirror_session_nodes(_config(control_mode="other_mode"))

    def test_control_mode_must_be_a_control_modes_member(self):
        with pytest.raises(ValueError, match="is not a control_modes member"):
            generate_mirror_session_nodes(
                {
                    **_config(),
                    "skill_required_control_mode": "not_declared",
                    "mirror_session": {**_config()["mirror_session"], "control_mode": "not_declared"},
                }
            )

    def test_runtime_mode_must_be_explicit_and_not_idle(self):
        config = _config()
        config["control_modes"]["mirror_mode"] = {"runtime_mode": "idle"}
        with pytest.raises(ValueError, match="must be an explicit non-idle runtime mode"):
            generate_mirror_session_nodes(config)

    def test_sole_producer_requires_disabled_inference_and_executor(self):
        config = _config()
        config["control_modes"]["mirror_mode"] = {"runtime_mode": "policy_stream"}
        with pytest.raises(ValueError, match="inference.enabled must be false"):
            generate_mirror_session_nodes(config)

        config = _config()
        config["control_modes"]["mirror_mode"] = {
            "runtime_mode": "policy_stream",
            "inference": {"enabled": False},
        }
        with pytest.raises(ValueError, match="executor.enabled must be false"):
            generate_mirror_session_nodes(config)

    def test_missing_runtime_description_fails_closed(self):
        from robot_config.interface_binding import InterfaceBindingError

        config = _config()
        config["runtime"] = {}
        with pytest.raises(InterfaceBindingError, match="description_required"):
            generate_mirror_session_nodes(config)

    def test_missing_skill_required_control_mode_is_rejected(self):
        config = _config()
        config.pop("skill_required_control_mode")
        with pytest.raises(ValueError, match="needs skill_required_control_mode"):
            generate_mirror_session_nodes(config)


class TestRegistryIntegration:
    def test_builder_spec_is_addable_to_a_fresh_registry(self):
        from robot_config.launch_builders.registry import LaunchBuilderRegistry

        registry = LaunchBuilderRegistry()
        registry.add(MIRROR_SESSION_BUILDER)
        assert "mirror_session" in registry
        assert registry.snapshot()[-1].id == "mirror_session"

    def test_factory_reference_is_valid_for_the_resolver(self):
        # A malformed reference would fail at BuilderSpec construction; this
        # also guards against accidental callable factories after the move.
        resolve_builder_factory(MIRROR_SESSION_BUILDER.factory)
