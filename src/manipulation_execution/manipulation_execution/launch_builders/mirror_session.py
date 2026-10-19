"""Launch wiring for the X2 mirror session (``manipulation_execution.mirror_session_node``).

This module is ``manipulation_execution``'s contribution to the launch-builder
registry (openspec change ``decentralize-launch-builder-registration``, design
decision 4): ``MIRROR_SESSION_BUILDER`` is declared as an entry point in the
``ibrobot.launch_builders`` group (see ``setup.py``), so the node's owning
package registers its own launch wiring without any robot_config change. The
content is a verbatim migration of the former
``robot_config/launch_builders/mirror_session.py``; openspec change
``migrate-business-launch-builders`` moved it from
``manipulation_execution/launch_builder.py`` to this unified
``launch_builders/`` layout, together with the motion-imitation builder
whose stream helpers it shares.

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

Installing ``manipulation_execution`` does not enable the session:
participation is still decided by the config (``mirror_session.enabled``),
exactly as before the migration.
"""

import json
import math
from collections.abc import Mapping

from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node

from manipulation_execution.launch_builders.motion_imitation_stream import (
    mock_playback_endpoints,
    motion_imitation_stream_parameters,
)
from robot_config.interface_binding import InterfaceBindingError
from robot_config.launch_builders.registry import BuilderSpec, BuildResult, LaunchBuildContext

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
_IDLE = "idle"


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
    # Function-local import: keep registry construction free of the
    # embodied_common dependency chain.
    from embodied_common.mirror_session_contract import DEFAULT_STOP_TIMEOUT_SEC, NODE_NAME, RELEASE_RESERVE_SEC

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


__all__ = ["generate_mirror_session_nodes"]


# ---------------------------------------------------------------------------
# Registry entry (openspec change ``decentralize-launch-builder-registration``,
# design decision 4): the BuilderSpec constant and build function moved here
# from ``robot_config/launch_builders/mirror_session.py`` (a verbatim
# migration) and are contributed through the entry point group
# ``ibrobot.launch_builders`` declared in this package's ``setup.py``.
# ---------------------------------------------------------------------------


def build_mirror_session(context: LaunchBuildContext) -> BuildResult:
    from manipulation_execution.launch_builders.mirror_session import generate_mirror_session_nodes

    return BuildResult(actions=list(generate_mirror_session_nodes(context.robot_config)))


MIRROR_SESSION_BUILDER = BuilderSpec(
    id="mirror_session",
    factory=f"{__name__}:build_mirror_session",
    timing="runtime_ready",
)
