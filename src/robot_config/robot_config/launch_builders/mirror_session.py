"""Launch wiring for the X2 mirror session (``manipulation_execution.mirror_session_node``).

The node is the only publisher of the arm and hand channels while it holds the
mirror's runtime mode, so this builder checks what the node cannot see from
inside one process:

* the runtime mode it takes over is ``control_modes.<control_mode>.runtime_mode``,
  explicit and not idle, and ``control_mode`` is ``skill_required_control_mode``:
  the Gateway switches to the skills' mode and the session takes over exactly
  that mode;
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

``business`` names the start/cancel rules (``hooks``), the cover judge and the
business config file both nodes read; ``config`` is resolved here to an
absolute path, whether the file exists is the nodes' check at startup. Only
``""`` and ``demo`` hooks wrap the mock; any other business class and the mock
are exclusive. ``operator.enabled`` appends the operator node (covered camera,
head touch, LED) as the last action; without an ``operator`` block it is not
launched. The operator's parameter names are listed here rather than imported:
robot_config does not depend on manipulation_execution, so
manipulation_execution's test_mirror_session_launch holds them against the
node's declaration. Its ``set_session(false)`` wait is the contract's stop
timeout plus margin, a constant: ``stop_timeout_sec`` can only be at most 4.5
(the Gateway's ``mirror_stop`` timeout must exceed it by 0.5), so it is not
passed.
"""

import json
import math
import os
from collections.abc import Mapping

from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node

from embodied_common.mirror_session_contract import (
    DEFAULT_STOP_TIMEOUT_SEC,
    NODE_NAME,
    OPERATOR_NODE_NAME,
    RELEASE_RESERVE_SEC,
)
from robot_config.interface_binding import InterfaceBindingError
from robot_config.launch_builders.motion_imitation_stream import (
    mock_playback_endpoints,
    mock_reset_pose,
    motion_imitation_stream_parameters,
)
from robot_config.utils import resolve_ros_path

_KEYS = frozenset(
    {
        "enabled",
        "publish_enabled",
        "control_mode",
        "stop_timeout_sec",
        "publish_rate_hz",
        "status_period_sec",
        "interfaces",
        "business",
        "operator",
    }
)
_BUSINESS_KEYS = frozenset({"hooks", "cover_judge", "config"})
_OPERATOR_KEYS = frozenset({"enabled", "send_enabled"})
# The hooks that wrap the mock instead of replacing it.
_MOCK_HOOKS = ("", "demo")
# The Gateway's names, read as robot_skill_cli's catalog reads them, with its defaults.
_GATEWAY_NAMES = (
    ("gateway_status_service", "skill_gateway_status_service", "/embodied/get_skill_gateway_status"),
    ("gateway_validate_service", "validate_skill_service", "/embodied/validate_skill"),
    ("gateway_skill_action", "skill_action_name", "/embodied/execute_skill"),
)
_IDLE = "idle"


def _bool(config: Mapping, key: str, default: bool, where: str = "mirror_session") -> bool:
    value = config.get(key, default)
    if type(value) is not bool:
        raise ValueError(f"{where}.{key} must be a boolean")
    return value


def _string(config: Mapping, key: str, where: str) -> str:
    value = config.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"{where}.{key} must be a string")
    return value


def _block(config: Mapping, key: str, keys: frozenset) -> Mapping | None:
    """``mirror_session.<key>``, its keys checked; None if it is absent."""
    block = config.get(key)
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(f"mirror_session.{key} must be a mapping")
    unknown = sorted(set(block) - keys)
    if unknown:
        raise ValueError(f"mirror_session.{key} has unknown keys {unknown}")
    return block


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


def _business(config: Mapping, mock: bool) -> dict:
    """``hooks``, ``cover_judge`` and the absolute ``config`` path of ``mirror_session.business``."""
    block = _block(config, "business", _BUSINESS_KEYS) or {}
    business = {key: _string(block, key, "mirror_session.business") for key in sorted(_BUSINESS_KEYS)}
    if business["config"]:
        business["config"] = str(resolve_ros_path(business["config"]))
        if not os.path.isabs(business["config"]):
            raise ValueError(
                f"mirror_session.business.config must resolve to an absolute path, got {business['config']!r}"
            )
    if mock and business["hooks"] not in _MOCK_HOOKS:
        raise ValueError(
            f"mirror_session.business.hooks {business['hooks']!r} replaces the mock: disable "
            "motion_imitation_stream.mock_playback or use the demo hooks"
        )
    return business


def _gateway_names(robot_config: Mapping) -> dict:
    embodied = robot_config.get("embodied") or {}
    if not isinstance(embodied, Mapping):
        raise ValueError("embodied must be a mapping")
    names = {}
    for parameter, key, default in _GATEWAY_NAMES:
        value = embodied.get(key, default)
        if not isinstance(value, str) or not value:
            raise ValueError(f"embodied.{key} must be a non-empty string")
        names[parameter] = value
    return names


def _operator_parameters(robot_config: Mapping, config: Mapping, session: Mapping, business: Mapping) -> dict | None:
    """The operator node's parameters, or None if ``mirror_session.operator`` does not enable it."""
    block = _block(config, "operator", _OPERATOR_KEYS)
    if block is None or not _bool(block, "enabled", False, "mirror_session.operator"):
        return None
    if not session["camera_topic"]:
        raise ValueError(
            "mirror_session.operator needs the camera: motion_imitation_stream must name its compressed frame source"
        )
    return {
        "runtime_description_json": session["runtime_description_json"],
        "camera_topic": session["camera_topic"],
        # frame_source only takes a compressed camera.
        "camera_kind": "compressed",
        "business_config": business["config"],
        "cover_judge": business["cover_judge"],
        "send_enabled": _bool(block, "send_enabled", False, "mirror_session.operator"),
        **_gateway_names(robot_config),
        "status_period_sec": session["status_period_sec"],
    }


def generate_mirror_session_nodes(robot_config: dict) -> list:
    """The session node, its process-exit release and, if enabled, the operator node."""
    config = robot_config.get("mirror_session", {})
    if not isinstance(config, Mapping):
        raise ValueError("mirror_session must be a mapping")
    if not _bool(config, "enabled", False):
        return []
    unknown = sorted(set(config) - _KEYS)
    if unknown:
        raise ValueError(f"mirror_session has unknown keys {unknown}")
    skill_mode = robot_config.get("skill_required_control_mode")
    if not isinstance(skill_mode, str) or not skill_mode.strip():
        raise ValueError("mirror_session needs skill_required_control_mode: its skills run through the Gateway")
    control_mode = config.get("control_mode", skill_mode)
    if not isinstance(control_mode, str) or not control_mode.strip():
        raise ValueError("mirror_session.control_mode must be a non-empty string")
    if control_mode.strip() != skill_mode.strip():
        raise ValueError(
            f"mirror_session.control_mode {control_mode!r} must equal skill_required_control_mode "
            f"{skill_mode!r}: the Gateway switches to the skills' mode and the session takes over exactly that mode"
        )

    release_parameters = {
        "runtime_description_json": _runtime_description(robot_config),
        "policy_mode": _policy_mode(robot_config, control_mode.strip()),
        "stop_timeout_sec": _number(config, "stop_timeout_sec", DEFAULT_STOP_TIMEOUT_SEC, RELEASE_RESERVE_SEC),
    }
    stream = motion_imitation_stream_parameters(robot_config) or {}
    mock = mock_playback_endpoints(robot_config) or ("", "")
    reset_pose = mock_reset_pose(robot_config)
    business = _business(config, mock=bool(mock[1]))
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
        "mock_reset_pose_json": json.dumps(reset_pose) if reset_pose else "",
        "business_hooks": business["hooks"],
        "business_config": business["config"],
    }
    operator = _operator_parameters(robot_config, config, parameters, business)
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
    actions = [session, RegisterEventHandler(OnProcessExit(target_action=session, on_exit=[release]))]
    if operator is not None:
        actions.append(
            Node(
                package="manipulation_execution",
                executable="mirror_operator_node",
                name=OPERATOR_NODE_NAME,
                output="screen",
                parameters=[operator],
            )
        )
    return actions


__all__ = ["generate_mirror_session_nodes"]
