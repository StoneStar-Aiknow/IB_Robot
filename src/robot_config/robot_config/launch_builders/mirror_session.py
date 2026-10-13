"""Launch wiring for the X2 mirror session (``manipulation_execution.mirror_session_node``).

The node is the only publisher of the arm and hand channels while it holds the
mirror's runtime mode, so this builder checks what the node cannot see from
inside one process:

* the runtime mode it takes over is ``control_modes.<control_mode>.runtime_mode``,
  explicit and not idle;
* nothing else produces that mode: no control mode mapped to it launches an
  inference pipeline or an action dispatcher;
* the bound runtime description is passed whole, so the node resolves every
  endpoint by interface ID; an unbound config fails here. ``interfaces``
  declares the runtime services the session and its release call, which is
  what makes launch wait for them and bind the live description
  (``robot_config.interface_binding``).

Two exits back to idle are wired here; the third (normal exit) is the node's
own ``finally``. When the node process exits while launch keeps running,
launch runs ``mirror_session_release``, which only requests idle while the
runtime is still in the mirror's mode with no stop latch, so running it after
a normal exit is a no-op. Launch does not start it while launch itself is
shutting down; interrupting the whole launch therefore relies on the node's
``finally``, which needs the runtime to still answer.

The perception inputs and the mock playback window pipeline follow
``motion_imitation_stream``: the PEAR frame, ActorState and camera topics are
the stream's, and with ``mock_playback`` enabled the session plays the mock's
animation through its gate. ``publish_enabled`` (default false) decides whether
the gate's output reaches the robot or only the session's preview topics.
"""

import json
import math
from collections.abc import Mapping

from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node

from robot_config.interface_binding import InterfaceBindingError
from robot_config.launch_builders.motion_imitation_stream import (
    mock_playback_endpoints,
    motion_imitation_stream_parameters,
)

_KEYS = frozenset(
    {
        "enabled",
        "publish_enabled",
        "control_mode",
        "stop_timeout_sec",
        "publish_rate_hz",
        "status_period_sec",
        "interfaces",
    }
)
_DEFAULT_CONTROL_MODE = "model_inference"
# Must match manipulation_execution.mirror_session.core.RELEASE_RESERVE_S.
_RELEASE_RESERVE_SEC = 2.0
_IDLE = "idle"
# The session node's name and its start/stop service (std_srvs/SetBool), which
# the Gateway's mirror_session executor calls for mirror_start / mirror_stop.
NODE_NAME = "mirror_session"
SESSION_SERVICE = f"/{NODE_NAME}/set_session"


def _bool(config: Mapping, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if type(value) is not bool:
        raise ValueError(f"mirror_session.{key} must be a boolean")
    return value


def _number(config: Mapping, key: str, default: float, minimum: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"mirror_session.{key} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= minimum:
        raise ValueError(f"mirror_session.{key} must be finite and greater than {minimum:g}")
    return number


def _policy_mode(robot_config: Mapping, control_mode: str) -> str:
    """The runtime mode of ``control_mode``, checked to have no other producer."""
    modes = robot_config.get("control_modes")
    mode = modes.get(control_mode) if isinstance(modes, Mapping) else None
    if not isinstance(mode, Mapping):
        raise ValueError(f"mirror_session.control_mode {control_mode!r} is not a control_modes member")
    runtime_mode = mode.get("runtime_mode")
    if not isinstance(runtime_mode, str) or not runtime_mode.strip() or runtime_mode.strip() == _IDLE:
        raise ValueError(f"control_modes.{control_mode}.runtime_mode must be an explicit non-idle runtime mode")
    runtime_mode = runtime_mode.strip()
    for name, other in modes.items():
        if not isinstance(other, Mapping) or str(other.get("runtime_mode", "")).strip() != runtime_mode:
            continue
        # Explicit opt-outs only: a missing switch is a producer that may start.
        inference = other.get("inference") or {}
        executor = other.get("executor") or {}
        if not isinstance(inference, Mapping) or inference.get("enabled") is not False:
            raise ValueError(
                f"control_modes.{name}.inference.enabled must be false: mirror_session is the only "
                f"producer of runtime mode {runtime_mode!r}"
            )
        if not isinstance(executor, Mapping) or executor.get("enabled") is not False:
            raise ValueError(
                f"control_modes.{name}.executor.enabled must be false: mirror_session is the only "
                f"producer of runtime mode {runtime_mode!r}"
            )
    return runtime_mode


def _runtime_description(robot_config: Mapping) -> str:
    runtime = robot_config.get("runtime") or {}
    description = runtime.get("interface_description") if isinstance(runtime, Mapping) else None
    if not isinstance(description, Mapping) or not isinstance(description.get("interfaces"), Mapping):
        raise InterfaceBindingError(
            "description_required",
            "runtime.interface_description",
            "mirror_session resolves its runtime endpoints by ID; bind the live description first",
        )
    return json.dumps(description, sort_keys=True)


def generate_mirror_session_nodes(robot_config: dict) -> list:
    """The session node and its process-exit release, if the robot enables the session."""
    config = robot_config.get("mirror_session", {})
    if not isinstance(config, Mapping):
        raise ValueError("mirror_session must be a mapping")
    if not _bool(config, "enabled", False):
        return []
    unknown = sorted(set(config) - _KEYS)
    if unknown:
        raise ValueError(f"mirror_session has unknown keys {unknown}")
    control_mode = config.get("control_mode", _DEFAULT_CONTROL_MODE)
    if not isinstance(control_mode, str) or not control_mode.strip():
        raise ValueError("mirror_session.control_mode must be a non-empty string")

    release_parameters = {
        "runtime_description_json": _runtime_description(robot_config),
        "policy_mode": _policy_mode(robot_config, control_mode.strip()),
        "stop_timeout_sec": _number(config, "stop_timeout_sec", 5.0, _RELEASE_RESERVE_SEC),
    }
    stream = motion_imitation_stream_parameters(robot_config) or {}
    mock = mock_playback_endpoints(robot_config) or ("", "")
    parameters = {
        **release_parameters,
        "publish_enabled": _bool(config, "publish_enabled", False),
        "publish_rate_hz": _number(config, "publish_rate_hz", 50.0, 0.0),
        "status_period_sec": _number(config, "status_period_sec", 1.0, 0.0),
        "pear_frame_topic": stream.get("frame_topic", ""),
        "actor_state_topic": stream.get("actor_state_topic", ""),
        "camera_topic": stream.get("rgb_compressed_topic", ""),
        "mock_preview_topic": mock[0],
        "mock_play_service": mock[1],
    }
    session = Node(
        package="manipulation_execution",
        executable="mirror_session_node",
        name=NODE_NAME,
        output="screen",
        parameters=[parameters],
    )
    release = Node(
        package="manipulation_execution",
        executable="mirror_session_release",
        name="mirror_session_release",
        output="screen",
        parameters=[release_parameters],
    )
    return [session, RegisterEventHandler(OnProcessExit(target_action=session, on_exit=[release]))]


__all__ = ["NODE_NAME", "SESSION_SERVICE", "generate_mirror_session_nodes"]
