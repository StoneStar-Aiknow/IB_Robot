"""Launch wiring for the resident motion-imitation frame stream.

This module is ``manipulation_execution``'s contribution to the
launch-builder registry (openspec change ``migrate-business-launch-builders``):
the ``motion_imitation_stream`` entry is declared as an entry point in the
``ibrobot.launch_builders`` group (see ``setup.py``), so the nodes' owning
package registers its own launch wiring without any robot_config change.
The content is a verbatim migration of the former
``robot_config/launch_builders/motion_imitation_stream.py``.

The two perception model services this stream calls are already started by
``generate_perception_model_nodes``, because they are plain
``perception_services.services`` entries. This builder adds the remaining
piece: the consumer node itself.

Validation lives here rather than in the loader, matching
``interaction_demo.py``: a malformed block raises with the offending key named
instead of silently starting a node wired to a dead endpoint.

Runtime endpoints are never written in the robot YAML. The camera stream and
the arm channel's joint order (which the mock's output follows) are both
looked up by logical ID in the bound runtime interface description, the way
action_dispatch's PolicyAdmission binds its runtime interfaces, so the builder
fails closed on a config that was never bound instead of guessing a topic.

The mock never publishes to the robot. ``mirror_session`` is the only outlet
and reads the mock's preview through its gate; ``mock_playback_endpoints`` is
how its builder finds the preview topic and the ``~/play`` service.
"""

import json
import math
from collections.abc import Mapping
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from robot_config.interface_binding import InterfaceBindingError
from robot_config.launch_builders.registry import BuilderSpec, BuildResult, LaunchBuildContext

_COMPRESSED_TYPE = "sensor_msgs/msg/CompressedImage"
# The runtime publishes the camera; the stream subscribes it.
_CAMERA_SHAPE = ("topic", "publish", _COMPRESSED_TYPE)
_FRAME_SOURCE_KEYS = frozenset({"interface", "topic", "decoder", "decoder_params"})
# Superseded by frame_source; refused rather than silently ignored.
_LEGACY_STREAM_KEYS = ("rgb_compressed_topic", "rgb_compressed_type")
_DEFAULT_DECODER = "mock_cpu"
_ARM_JOINT_COUNT = 14
# The runtime's arm command channel, by logical ID. The mock's output reaches
# it through mirror_session, in its joint order, so the shape is checked the
# way action_dispatch's PolicyAdmission checks its runtime interfaces:
# (kind, direction, message_type) must match exactly.
_ARM_STREAM = "joint.arm_stream"
_ARM_STREAM_SHAPE = ("topic", "subscribe", "std_msgs/msg/Float64MultiArray")
_ARM_STREAM_MODE = "policy_stream"
# Shipped with the mock; see that directory's README for provenance.
_DEFAULT_ANIMATION = "data/x2_mock_imitation/rec_20260914_212729_delivery.json"
MOCK_NODE_NAME = "motion_imitation_mock"
MOCK_PLAY_SERVICE = f"/{MOCK_NODE_NAME}/play"
_DEFAULT_MOCK_PREVIEW = f"/{MOCK_NODE_NAME}/arm_command_preview"


def _required_bool(config: Mapping, key: str) -> bool:
    value = config.get(key, False)
    if type(value) is not bool:
        raise ValueError(f"motion_imitation_stream.{key} must be a boolean")
    return value


def _required_topic(config: Mapping, key: str, default: str = "") -> str:
    value = config.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"motion_imitation_stream.{key} must be a non-empty string")
    topic = value.strip()
    if not topic.startswith("/"):
        raise ValueError(f"motion_imitation_stream.{key} must be an absolute ROS name")
    return topic


def _positive(config: Mapping, key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"motion_imitation_stream.{key} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"motion_imitation_stream.{key} must be positive and finite")
    return number


def _non_negative(config: Mapping, key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"motion_imitation_stream.{key} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"motion_imitation_stream.{key} must be non-negative and finite")
    return number


def _unit_interval(config: Mapping, key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"motion_imitation_stream.{key} must be a number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"motion_imitation_stream.{key} must be within [0, 1]")
    return number


def _stream_config(robot_config: Mapping) -> Mapping | None:
    """The block if the stream is enabled, else None."""
    config = robot_config.get("motion_imitation_stream", {})
    if not isinstance(config, Mapping):
        raise ValueError("motion_imitation_stream must be a mapping")
    return config if _required_bool(config, "enabled") else None


def motion_imitation_stream_parameters(robot_config: Mapping) -> dict | None:
    """The stream node's validated parameters, or None if the robot does not enable it."""
    config = _stream_config(robot_config)
    return None if config is None else _stream_parameters(config, robot_config)


def mock_playback_endpoints(robot_config: Mapping) -> tuple[str, str] | None:
    """``(preview_topic, play_service)`` of the mock node, or None if it is not launched."""
    config = _stream_config(robot_config)
    mock = None if config is None else _mock_config(config)
    if mock is None:
        return None
    return _required_topic(mock, "preview_topic", _DEFAULT_MOCK_PREVIEW), MOCK_PLAY_SERVICE


def generate_motion_imitation_stream_nodes(robot_config: dict) -> list[Node]:
    """Launch the head-camera -> YOLOX -> PEAR frame stream, if the robot enables it."""
    config = _stream_config(robot_config)
    if config is None:
        return []
    parameters = _stream_parameters(config, robot_config)
    nodes = [
        Node(
            package="manipulation_execution",
            executable="motion_imitation_stream_node",
            name="motion_imitation_stream",
            output="screen",
            parameters=[parameters],
        )
    ]
    nodes.extend(_mock_playback_nodes(config, parameters["frame_topic"], robot_config))
    return nodes


def _stream_parameters(config: Mapping, robot_config: Mapping) -> dict:
    for key in _LEGACY_STREAM_KEYS:
        if key in config:
            raise ValueError(
                f"motion_imitation_stream.{key} must not be set: the camera is configured by "
                "frame_source.interface (or frame_source.topic)"
            )

    rgb_topic, interface_id, decoder, decoder_params = _frame_source(config, robot_config)
    parameters = {
        "rgb_compressed_topic": rgb_topic,
        "frame_source_interface": interface_id,
        "decoder": decoder,
        "decoder_params_json": decoder_params,
        "frame_topic": _required_topic(config, "frame_topic", "/motion_imitation_stream/pear_frame"),
        "status_topic": _required_topic(config, "status_topic", "/motion_imitation_stream/status"),
        "actor_state_topic": _required_topic(config, "actor_state_topic", "/motion_imitation_stream/actor_state"),
        "yolox_detect_service": _required_topic(config, "yolox_detect_service", "/perception/hri/yolox_detect"),
        "pear_parameters_service": _required_topic(
            config, "pear_parameters_service", "/perception/hri/pear_parameters"
        ),
        "person_confidence_threshold": _unit_interval(config, "person_confidence_threshold", 0.30),
        "yolox_refresh_interval_sec": _positive(config, "yolox_refresh_interval_sec", 1.0),
        "rpc_timeout_sec": _positive(config, "rpc_timeout_sec", 2.0),
        "max_pear_hz": _non_negative(config, "max_pear_hz", 0.0),
        "publish_fallback_frames": _required_bool(config, "publish_fallback_frames")
        if "publish_fallback_frames" in config
        else True,
        "status_period_sec": _positive(config, "status_period_sec", 5.0),
        "actor_stale_sec": _positive(config, "actor_stale_sec", 3.0),
    }
    outputs = [parameters[key] for key in ("frame_topic", "status_topic", "actor_state_topic")]
    if len(set(outputs)) != len(outputs):
        raise ValueError("motion_imitation_stream.frame_topic, status_topic and actor_state_topic must differ")
    # Detections are refreshed once per interval; a shorter staleness bound
    # would make the actor unknown between every two refreshes.
    if parameters["actor_stale_sec"] <= parameters["yolox_refresh_interval_sec"]:
        raise ValueError("motion_imitation_stream.actor_stale_sec must exceed yolox_refresh_interval_sec")
    return parameters


def _described_interfaces(robot_config: Mapping, consumer: str) -> Mapping:
    """The bound description's interfaces, or a ``description_required`` failure.

    Launch binds the live runtime description before building application nodes;
    a config that was never bound has no endpoint to use and fails here rather
    than falling back to a guessed topic.
    """
    runtime = robot_config.get("runtime") or {}
    description = runtime.get("interface_description") if isinstance(runtime, Mapping) else None
    if not isinstance(description, Mapping) or not isinstance(description.get("interfaces"), Mapping):
        raise InterfaceBindingError(
            "description_required",
            "runtime.interface_description",
            f"{consumer}; bind the live description first",
        )
    return description["interfaces"]


def _frame_source(config: Mapping, robot_config: Mapping) -> tuple[str, str, str, str]:
    """Resolve the camera topic and the decoder for the stream.

    ``frame_source.interface`` names a compressed camera in the runtime
    description and is the normal form. ``frame_source.topic`` is an escape
    hatch for a stream the description does not carry (a resized stream from
    another compute unit, say); naming a described topic that way is refused,
    because it would bypass the shape check. Exactly one of the two is set.

    The decoder is only named here. Its registry lives with the node, which
    rejects an unknown name or parameter when it starts.
    """
    source = config.get("frame_source")
    if not isinstance(source, Mapping):
        raise ValueError("motion_imitation_stream.frame_source must be a mapping")
    unknown = sorted(set(source) - _FRAME_SOURCE_KEYS)
    if unknown:
        raise ValueError(f"motion_imitation_stream.frame_source has unknown keys {unknown}")
    if ("interface" in source) == ("topic" in source):
        raise ValueError("motion_imitation_stream.frame_source needs exactly one of interface or topic")
    interfaces = _described_interfaces(robot_config, "motion_imitation_stream.frame_source subscribes a camera")
    if "interface" in source:
        interface_id = source["interface"]
        if not isinstance(interface_id, str) or not interface_id or interface_id.strip() != interface_id:
            raise InterfaceBindingError(
                "invalid_interface",
                "motion_imitation_stream.frame_source.interface",
                "interface must be a non-empty logical ID",
            )
        item = interfaces.get(interface_id)
        if not isinstance(item, Mapping):
            raise InterfaceBindingError("unknown_interface", interface_id, "ID is not in the runtime description")
        if (item.get("kind"), item.get("direction"), item.get("message_type")) != _CAMERA_SHAPE:
            kind, direction, message_type = _CAMERA_SHAPE
            raise InterfaceBindingError(
                "invalid_interface", interface_id, f"requires {kind} {message_type} {direction}"
            )
        topic = item.get("endpoint")
        if not isinstance(topic, str) or not topic.startswith("/"):
            raise InterfaceBindingError("invalid_interface", interface_id, "endpoint must be an absolute ROS name")
    else:
        interface_id = ""
        topic = source["topic"]
        if not isinstance(topic, str) or not topic.startswith("/") or topic.strip() != topic:
            raise ValueError("motion_imitation_stream.frame_source.topic must be an absolute ROS name")
        described = sorted(
            name for name, item in interfaces.items() if isinstance(item, Mapping) and item.get("endpoint") == topic
        )
        if described:
            raise ValueError(
                f"motion_imitation_stream.frame_source.topic is described as {described[0]}; use frame_source.interface"
            )
    decoder = source.get("decoder", _DEFAULT_DECODER)
    if not isinstance(decoder, str) or not decoder:
        raise ValueError("motion_imitation_stream.frame_source.decoder must be a non-empty string")
    params = source.get("decoder_params", {})
    if not isinstance(params, Mapping):
        raise ValueError("motion_imitation_stream.frame_source.decoder_params must be a mapping")
    try:
        params_json = json.dumps(dict(params), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"motion_imitation_stream.frame_source.decoder_params must be plain data: {exc}") from exc
    return topic, interface_id, decoder, params_json


def _arm_stream(robot_config: Mapping) -> tuple[str, list[str]]:
    """Resolve the arm command endpoint and joint order from the bound description."""
    interfaces = _described_interfaces(
        robot_config, f"motion_imitation_stream.mock_playback commands {_ARM_STREAM} in its joint order"
    )
    item = interfaces.get(_ARM_STREAM)
    if not isinstance(item, Mapping):
        raise InterfaceBindingError("unknown_interface", _ARM_STREAM, "ID is not in the runtime description")
    if (item.get("kind"), item.get("direction"), item.get("message_type")) != _ARM_STREAM_SHAPE:
        kind, direction, message_type = _ARM_STREAM_SHAPE
        raise InterfaceBindingError("invalid_interface", _ARM_STREAM, f"requires {kind} {message_type} {direction}")
    if _ARM_STREAM_MODE not in (item.get("modes") or []):
        raise InterfaceBindingError("invalid_interface", _ARM_STREAM, f"must accept runtime mode {_ARM_STREAM_MODE}")
    endpoint = item.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.startswith("/"):
        raise InterfaceBindingError("invalid_interface", _ARM_STREAM, "endpoint must be an absolute ROS name")
    joint_names = [str(name) for name in item.get("joint_names") or []]
    if len(joint_names) != _ARM_JOINT_COUNT or len(set(joint_names)) != _ARM_JOINT_COUNT:
        raise InterfaceBindingError(
            "invalid_interface", _ARM_STREAM, f"must describe {_ARM_JOINT_COUNT} distinct joint_names"
        )
    return endpoint, joint_names


def _arm_limits(mock: Mapping, joint_names: list[str]) -> tuple[dict, dict]:
    """Read the arm travel band the mock safety clamps against.

    The limits are application configuration: they live in the robot YAML, not
    in the runtime profile. They are keyed by joint name; the order of the values
    the mock publishes is the channel's ``joint_names`` from the description.
    """
    limits = mock.get("arm_limits")
    velocity = mock.get("arm_velocity_limits")
    if not isinstance(limits, Mapping) or not isinstance(velocity, Mapping):
        raise ValueError("motion_imitation_stream.mock_playback needs arm_limits and arm_velocity_limits mappings")
    if set(limits) != set(joint_names):
        raise ValueError(f"motion_imitation_stream.mock_playback.arm_limits must name exactly the {_ARM_STREAM} joints")
    if set(velocity) != set(joint_names):
        raise ValueError(
            f"motion_imitation_stream.mock_playback.arm_velocity_limits must name exactly the {_ARM_STREAM} joints"
        )
    return dict(limits), dict(velocity)


def _mock_config(config: Mapping) -> Mapping | None:
    """The ``mock_playback`` block if enabled, else None; superseded keys are refused."""
    mock = config.get("mock_playback", {})
    if not isinstance(mock, Mapping):
        raise ValueError("motion_imitation_stream.mock_playback must be a mapping")
    if not _required_bool(mock, "enabled"):
        return None
    if "arm_command_topic" in mock:
        raise ValueError(
            "motion_imitation_stream.mock_playback.arm_command_topic must not be set: the mock publishes "
            "its preview only, and mirror_session resolves the arm channel from the runtime description"
        )
    if "publish_enabled" in mock:
        raise ValueError(
            "motion_imitation_stream.mock_playback.publish_enabled must not be set: the mock never "
            "publishes to the robot; the switch is mirror_session.publish_enabled"
        )
    return mock


def _mock_playback_nodes(config: Mapping, frame_topic: str, robot_config: Mapping) -> list[Node]:
    mock = _mock_config(config)
    if mock is None:
        return []
    _, joint_names = _arm_stream(robot_config)
    limits, velocity = _arm_limits(mock, joint_names)
    manifest = str(mock.get("animation_manifest", "")).strip()
    if not manifest:
        manifest = str(Path(get_package_share_directory("manipulation_execution")) / _DEFAULT_ANIMATION)
    return [
        Node(
            package="manipulation_execution",
            executable="motion_imitation_mock_node",
            name=MOCK_NODE_NAME,
            output="screen",
            parameters=[
                {
                    "animation_manifest": manifest,
                    "pear_frame_topic": frame_topic,
                    "preview_topic": _required_topic(mock, "preview_topic", _DEFAULT_MOCK_PREVIEW),
                    "status_topic": _required_topic(mock, "status_topic", "/motion_imitation_mock/status"),
                    "publish_rate_hz": _positive(mock, "publish_rate_hz", 50.0),
                    "input_staleness_sec": _positive(mock, "input_staleness_sec", 0.5),
                    "status_period_sec": _positive(mock, "status_period_sec", 5.0),
                    "loop": _required_bool(mock, "loop") if "loop" in mock else True,
                    "joint_names_json": json.dumps(joint_names),
                    "joint_limits_json": json.dumps({name: list(limits[name]) for name in joint_names}),
                    "velocity_limits_json": json.dumps({name: float(velocity[name]) for name in joint_names}),
                }
            ],
        )
    ]


__all__ = [
    "MOCK_NODE_NAME",
    "MOCK_PLAY_SERVICE",
    "generate_motion_imitation_stream_nodes",
    "mock_playback_endpoints",
    "motion_imitation_stream_parameters",
]


# ---------------------------------------------------------------------------
# Registry entry (openspec change ``decentralize-launch-builder-registration``):
# the BuilderSpec constant and build function moved here from the retired
# ``adapters.py``; the body is a verbatim migration. Openspec change
# ``migrate-business-launch-builders`` moved this module from
# ``robot_config/launch_builders/motion_imitation_stream.py`` to its owning
# package.
# ---------------------------------------------------------------------------


def build_motion_imitation_stream(context: LaunchBuildContext) -> BuildResult:
    from manipulation_execution.launch_builders.motion_imitation_stream import (
        generate_motion_imitation_stream_nodes,
    )

    return BuildResult(actions=list(generate_motion_imitation_stream_nodes(context.robot_config)))


MOTION_IMITATION_STREAM_BUILDER = BuilderSpec(
    id="motion_imitation_stream",
    factory=f"{__name__}:build_motion_imitation_stream",
    timing="runtime_ready",
)
