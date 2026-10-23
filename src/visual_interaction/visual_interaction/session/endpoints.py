"""Resolve the camera by ID from the bound runtime description, and nothing else.

Same rule as ``manipulation_execution.mirror_session.endpoints``: endpoints are
never configured, they come from ``runtime.interface_description.interfaces``
and each must have the expected ``(kind, direction, message_type)``.

This feature resolves exactly one interface. It subscribes to a camera and
publishes its own utterance and diagnostics; it holds no runtime mode, requests
no control authority and sends no joint command. :data:`FORBIDDEN_PREFIXES`
turns that into a check rather than a claim — :func:`assert_observation_only`
refuses a description-driven set that reaches a motion interface at all, so an
observation-only entry point cannot quietly grow one.

Audio focus is not control authority. Both are acquired, and a future speech
coordinator will need the first, which is exactly why the check names what it
forbids rather than matching on the verb.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# The default described camera: what the session resolves when a deployment
# names an interface rather than a topic. mirror reads the same stream.
#
# A deployment may instead point ``frame_source.topic`` at the robot's
# half-resolution relay (hri_halfres_relay, measured at 30.1 Hz and ~80 KB
# against this stream's 10.0 Hz and ~296 KB). That relay republishes a resized
# JPEG outside the HAL interface contract, so it is named as a topic rather
# than added here — the same way motion_imitation_stream reads it.
#
# Note the two publish with opposite QoS: this stream RELIABLE/TRANSIENT_LOCAL,
# the relay BEST_EFFORT/VOLATILE. Neither may be assumed.
CAMERA_COMPRESSED = "camera.stereo_head_front_left.color_compressed"
CAMERA_STEREO_LEFT_COMPRESSED = CAMERA_COMPRESSED

_SHAPES: Mapping[str, tuple[str, str, str]] = {
    CAMERA_COMPRESSED: ("topic", "publish", "sensor_msgs/msg/CompressedImage"),
    "camera.stereo_head_front_right.color_compressed": (
        "topic",
        "publish",
        "sensor_msgs/msg/CompressedImage",
    ),
    "camera.rgbd_head_front.color_compressed": ("topic", "publish", "sensor_msgs/msg/CompressedImage"),
}

# Interface families an observation-only entry point must never resolve. The
# runtime exposes a control plane whatever this feature does, so "we did not
# start an executor" is not evidence of no side effect: refusing to resolve the
# interface is.
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "joint.",
    "runtime.set_mode",
    "runtime.control",
    "motion.",
    "base.",
    "arm.",
    "hand.",
    "gripper.",
    "nav.",
)


class EndpointBindingError(ValueError):
    """A named binding failure; codes match ``robot_config.InterfaceBindingError``."""

    def __init__(self, code: str, path: str, detail: str) -> None:
        self.code = code
        self.path = path
        super().__init__(f"{code}: {path}: {detail}")


@dataclass(frozen=True)
class Endpoint:
    """One resolved interface: where it is, and how to talk to it."""

    endpoint: str
    qos: Mapping[str, object]


def assert_observation_only(ids: tuple[str, ...]) -> None:
    """Raise when a requested ID belongs to a motion or control family."""
    for interface_id in ids:
        for prefix in FORBIDDEN_PREFIXES:
            if interface_id == prefix.rstrip(".") or interface_id.startswith(prefix):
                raise EndpointBindingError(
                    "forbidden_interface",
                    interface_id,
                    "visual interaction is observation-only and resolves no motion interface",
                )


def resolve_camera(description: Mapping[str, object], interface_id: str = CAMERA_COMPRESSED) -> Endpoint:
    """Return the camera endpoint, or raise naming what is wrong with it."""
    assert_observation_only((interface_id,))
    if interface_id not in _SHAPES:
        raise EndpointBindingError(
            "unknown_interface", interface_id, f"not a known camera; expected one of {sorted(_SHAPES)}"
        )
    if not isinstance(description, Mapping) or not isinstance(description.get("interfaces"), Mapping):
        raise EndpointBindingError(
            "description_required", "runtime.interface_description", "bind the live description first"
        )

    item = description["interfaces"].get(interface_id)
    if not isinstance(item, Mapping):
        raise EndpointBindingError("unknown_interface", interface_id, "ID is not in the runtime description")

    kind, direction, message_type = _SHAPES[interface_id]
    if (item.get("kind"), item.get("direction"), item.get("message_type")) != _SHAPES[interface_id]:
        raise EndpointBindingError("invalid_interface", interface_id, f"requires {kind} {message_type} {direction}")

    endpoint = item.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.startswith("/"):
        raise EndpointBindingError("invalid_interface", interface_id, "endpoint must be an absolute ROS name")

    return Endpoint(endpoint=endpoint, qos=dict(item.get("qos") or {}))


__all__ = [
    "CAMERA_COMPRESSED",
    "FORBIDDEN_PREFIXES",
    "Endpoint",
    "EndpointBindingError",
    "assert_observation_only",
    "resolve_camera",
]
