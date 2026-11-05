"""Launch wiring for the visual interaction session.

This module is ``visual_interaction``'s contribution to the launch-builder
registry: ``VISUAL_INTERACTION_BUILDER`` is declared as an entry point in the
``ibrobot.launch_builders`` group (see ``setup.py``), so the node's owning
package registers its own wiring without any robot_config change. Installing
the package does not enable it — participation comes from
``visual_interaction.enabled`` in the robot YAML, checked below.

The builder checks what the node cannot see from inside one process:

* the deployment really is observation-only. The node resolves one camera and
  refuses a motion interface, but a config could still ask for a control mode
  that launches an inference pipeline or an action dispatcher alongside it,
  which would put a motion producer in an entry point described as
  observation-only. That is refused here, where the whole config is visible.
* the bound runtime description is passed whole, so the node resolves the
  camera by interface ID. An unbound config fails here rather than at the
  first frame.
* the category whitelist stays inside the contract's own set. Widening it is a
  configuration error, not a deployment choice.

There is no release node and no exit handler, unlike the mirror session: this
feature takes over no runtime mode, so there is nothing to hand back.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from launch_ros.actions import Node

from robot_config.interface_binding import InterfaceBindingError
from robot_config.launch_registry.registry import BuilderSpec, BuildResult, LaunchBuildContext

# Keys the YAML may set. An unknown key is refused rather than ignored: it
# usually means a rename landed in one place only, and ignoring it would leave
# the deployment running on a default nobody chose.
_KEYS = frozenset(
    {
        "enabled",
        "frame_source",
        "vl",
        "session",
        "categories",
        "actor_state_topic",
        "mirror_state_topic",
        "mirror_state_stale_sec",
        "round_period_sec",
        "window_sec",
        "frames_per_window",
        "buffer_sec",
        "stale_stream_sec",
        "status_period_sec",
        "mode",
        "chatter",
        "voice_status_topic",
    }
)
# visual_interaction.chatter, read only with mode "chatter".
_CHATTER_KEYS = frozenset(
    {
        "gap_min_sec",
        "gap_max_sec",
        "fresh_sec",
        "action_cooldown_sec",
        "object_cooldown_sec",
        "latest_start_sec",
    }
)
_FRAME_SOURCE_KEYS = frozenset({"interface", "topic", "qos"})
_QOS_KEYS = frozenset({"reliability", "durability", "history", "depth"})
_VL_KEYS = frozenset({"implementation", "demo_root", "max_new_tokens"})
_SESSION_KEYS = frozenset(
    {
        "confirmation_count",
        "max_evidence_age_sec",
        "min_gap_sec",
        "absent_reset_sec",
        "max_session_sec",
    }
)

# Control modes whose presence would mean something else drives the robot in
# the same entry point. Checked against every mode the config launches, not
# only the default one.
_MOTION_PRODUCER_KEYS = ("inference", "executor")


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _unknown(section: Mapping[str, object], allowed: frozenset[str], path: str) -> None:
    extra = sorted(set(section) - allowed)
    if extra:
        raise ValueError(f"{path} has unknown keys: {extra}")


def _assert_observation_only(robot_config: Mapping[str, object]) -> None:
    """Refuse a config that also launches a motion producer.

    The node refuses to resolve a motion interface, which keeps it from
    sending a command itself. This is the other half: an entry point described
    as observation-only must not start an inference pipeline or an action
    dispatcher beside it.

    A missing ``enabled`` key counts as enabled, matching
    ``robot_config.launch_builders.execution._executor_enabled``, which reads
    ``executor.get("enabled", True)``. Treating absence as off would pass every
    inherited teleop and navigation mode, which are exactly the ones that drive
    the robot — and they are inherited here, through aimdk_x2_skills.yaml.
    """
    modes = robot_config.get("control_modes")
    if not isinstance(modes, Mapping):
        return
    for mode_name, mode in modes.items():
        if not isinstance(mode, Mapping):
            continue
        for key in _MOTION_PRODUCER_KEYS:
            section = mode.get(key)
            if isinstance(section, Mapping) and section.get("enabled", True) is not False:
                raise ValueError(
                    f"control_modes.{mode_name}.{key}.enabled must be false: "
                    "visual_interaction is observation-only and cannot share an entry point "
                    "with a motion producer"
                )


def _runtime_description(robot_config: Mapping[str, object]) -> str:
    """Read the bound description from where ``interface_binding`` writes it.

    ``robot_config["runtime"]["interface_description"]`` — set by
    ``robot_config.interface_binding`` and read the same way by the
    mirror-session builder. Reading a top-level key instead looks plausible and
    fails on every real config, because nothing ever writes one.
    """
    runtime = robot_config.get("runtime")
    description = runtime.get("interface_description") if isinstance(runtime, Mapping) else None
    if not isinstance(description, Mapping) or not isinstance(description.get("interfaces"), Mapping):
        raise InterfaceBindingError(
            "description_required",
            "runtime.interface_description",
            "visual_interaction resolves its camera by ID; bind the live description first",
        )
    return json.dumps(description, sort_keys=True)


def _categories(section: Mapping[str, object]) -> list[str]:
    from embodied_common.visual_interaction_contract import CATEGORIES

    raw = section.get("categories")
    if raw is None:
        return []
    if not isinstance(raw, list | tuple) or not all(isinstance(value, str) for value in raw):
        raise ValueError("visual_interaction.categories must be a list of strings")
    if not raw:
        raise ValueError("visual_interaction.categories cannot be empty: a session with no subject can never speak")
    outside = sorted(set(raw) - set(CATEGORIES))
    if outside:
        raise ValueError(
            f"visual_interaction.categories must stay inside the contract whitelist; {outside} is not in "
            f"{list(CATEGORIES)}"
        )
    return list(raw)


def generate_visual_interaction_nodes(robot_config: Mapping[str, object]) -> list:
    """Return the launch actions for the visual interaction session, or none."""
    from embodied_common.mirror_session_contract import STATE_STALE_SEC
    from embodied_common.visual_interaction_contract import (
        DEFAULT_ABSENT_RESET_SEC,
        DEFAULT_CONFIRMATION_COUNT,
        DEFAULT_MAX_EVIDENCE_AGE_SEC,
        DEFAULT_MAX_SESSION_SEC,
        DEFAULT_MIN_GAP_SEC,
    )
    from visual_interaction.session.endpoints import CAMERA_COMPRESSED

    section = robot_config.get("visual_interaction")
    if section is None:
        return []
    section = _mapping(section, "visual_interaction")
    _unknown(section, _KEYS, "visual_interaction")

    enabled = section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("visual_interaction.enabled must be a boolean")
    if not enabled:
        return []

    _assert_observation_only(robot_config)

    frame_source = _mapping(section.get("frame_source", {}), "visual_interaction.frame_source")
    _unknown(frame_source, _FRAME_SOURCE_KEYS, "visual_interaction.frame_source")
    # At most one of the two, with the same meaning motion_imitation_stream
    # gives them: ``interface`` names a camera the runtime description carries
    # and is the normal form; ``topic`` is the escape hatch for a stream the
    # description does not carry. The robot's half-resolution relay
    # (hri_halfres_relay) is one of those — it republishes a resized JPEG and
    # is not part of the HAL interface contract, so naming it needs the hatch
    # rather than a new profile entry. Neither set keeps the default camera.
    if "interface" in frame_source and "topic" in frame_source:
        raise ValueError("visual_interaction.frame_source takes at most one of interface or topic")
    camera_topic = ""
    interface = ""
    camera_qos_json = ""
    if "topic" in frame_source:
        camera_topic = frame_source["topic"]
        if not isinstance(camera_topic, str) or not camera_topic.startswith("/"):
            raise ValueError("visual_interaction.frame_source.topic must be an absolute ROS name")
        # A named topic has no description to take QoS from, so the config has
        # to state it. Without this key the node fell back to a hardcoded
        # default that happened to match the live relay — frames flowed, and a
        # publisher with different QoS would have delivered nothing, silently.
        qos = _mapping(frame_source.get("qos", {}), "visual_interaction.frame_source.qos")
        _unknown(qos, _QOS_KEYS, "visual_interaction.frame_source.qos")
        if not qos:
            raise ValueError(
                "visual_interaction.frame_source.qos is required with topic: a mismatched "
                "subscription receives nothing and reports nothing"
            )
        camera_qos_json = json.dumps(dict(qos))
    else:
        interface = frame_source.get("interface", CAMERA_COMPRESSED)
        if not isinstance(interface, str) or not interface:
            raise ValueError("visual_interaction.frame_source.interface must be an interface ID")
        if "qos" in frame_source:
            raise ValueError(
                "visual_interaction.frame_source.qos belongs with topic; an interface takes "
                "its QoS from the bound runtime description"
            )

    vl = _mapping(section.get("vl", {}), "visual_interaction.vl")
    _unknown(vl, _VL_KEYS, "visual_interaction.vl")
    implementation = vl.get("implementation", "fake")
    if implementation not in ("fake", "internvl"):
        raise ValueError(
            f"visual_interaction.vl.implementation {implementation!r} is not available; expected 'fake' or 'internvl'"
        )
    demo_root = vl.get("demo_root", "")
    if implementation == "internvl":
        # The deployment directory is not in this repository, so it has to be
        # named. Failing here rather than at the first round keeps a launch
        # from coming up as a node that can never answer.
        if not isinstance(demo_root, str) or not demo_root.startswith("/"):
            raise ValueError("visual_interaction.vl.demo_root must be an absolute path for 'internvl'")
    elif demo_root:
        raise ValueError("visual_interaction.vl.demo_root only applies to implementation 'internvl'")
    max_new_tokens = int(vl.get("max_new_tokens", 110))
    if max_new_tokens < 1:
        raise ValueError("visual_interaction.vl.max_new_tokens must be positive")

    session = _mapping(section.get("session", {}), "visual_interaction.session")
    _unknown(session, _SESSION_KEYS, "visual_interaction.session")

    actor_topic = section.get("actor_state_topic", "")
    if not isinstance(actor_topic, str) or (actor_topic and not actor_topic.startswith("/")):
        raise ValueError("visual_interaction.actor_state_topic must be empty or an absolute ROS name")

    # The mirror gate: the magicmirror entry points this at
    # /mirror_session/state, and the rounds suspend with MIRROR_INACTIVE
    # unless that topic currently reports MIRRORING. Empty keeps the feature
    # ungated, which is what a deployment without a mirror wants.
    mirror_state_topic = section.get("mirror_state_topic", "")
    if not isinstance(mirror_state_topic, str) or (mirror_state_topic and not mirror_state_topic.startswith("/")):
        raise ValueError("visual_interaction.mirror_state_topic must be empty or an absolute ROS name")
    mirror_state_stale_sec = section.get("mirror_state_stale_sec", STATE_STALE_SEC)
    if isinstance(mirror_state_stale_sec, bool) or not isinstance(mirror_state_stale_sec, int | float):
        raise ValueError("visual_interaction.mirror_state_stale_sec must be a positive number of seconds")
    if mirror_state_stale_sec <= 0:
        raise ValueError("visual_interaction.mirror_state_stale_sec must be a positive number of seconds")

    parameters = {
        "runtime_description_json": _runtime_description(robot_config),
        "camera_interface": interface,
        "camera_topic": camera_topic,
        "actor_state_topic": actor_topic,
        "mirror_state_topic": mirror_state_topic,
        "mirror_state_stale_sec": float(mirror_state_stale_sec),
        "vl_implementation": implementation,
        "vl_demo_root": demo_root,
        "vl_max_new_tokens": max_new_tokens,
        "round_period_sec": float(section.get("round_period_sec", 1.0)),
        "window_sec": float(section.get("window_sec", 2.0)),
        "frames_per_window": int(section.get("frames_per_window", 4)),
        "buffer_sec": float(section.get("buffer_sec", 6.0)),
        "stale_stream_sec": float(section.get("stale_stream_sec", 2.0)),
        "confirmation_count": int(session.get("confirmation_count", DEFAULT_CONFIRMATION_COUNT)),
        "max_evidence_age_sec": float(session.get("max_evidence_age_sec", DEFAULT_MAX_EVIDENCE_AGE_SEC)),
        "min_gap_sec": float(session.get("min_gap_sec", DEFAULT_MIN_GAP_SEC)),
        "absent_reset_sec": float(session.get("absent_reset_sec", DEFAULT_ABSENT_RESET_SEC)),
        "max_session_sec": float(session.get("max_session_sec", DEFAULT_MAX_SESSION_SEC)),
        # JSON, matching the node's declaration: an empty ROS string array has
        # no inferable element type, so the node would refuse it at startup.
        "categories_json": json.dumps(_categories(section)),
        "status_period_sec": float(section.get("status_period_sec", 5.0)),
    }

    if camera_qos_json:
        parameters["camera_topic_qos_json"] = camera_qos_json

    mode = section.get("mode", "observe")
    if mode not in ("observe", "chatter"):
        raise ValueError(f"visual_interaction.mode {mode!r} is not one of 'observe', 'chatter'")
    chatter = _mapping(section.get("chatter", {}), "visual_interaction.chatter")
    _unknown(chatter, _CHATTER_KEYS, "visual_interaction.chatter")
    voice_status_topic = section.get("voice_status_topic", "")
    if not isinstance(voice_status_topic, str) or (voice_status_topic and not voice_status_topic.startswith("/")):
        raise ValueError("visual_interaction.voice_status_topic must be empty or an absolute ROS name")
    parameters["mode"] = mode
    parameters["voice_status_topic"] = voice_status_topic
    for key in sorted(_CHATTER_KEYS):
        if key in chatter:
            parameters[f"chatter_{key}"] = float(chatter[key])

    return [_visual_interaction_node(robot_config, vl_implementation=implementation, parameters=parameters)]


def _visual_interaction_node(robot_config: Mapping[str, object], *, vl_implementation: str, parameters: dict) -> Node:
    """The session node, attached to RT-Infer when the robot enables it.

    The InternVL chain (vision + prefill + decode) runs inside this node
    process through the board deployment's ais_bench sessions
    (``visual_interaction.session.internvl_vision``), so under RT-Infer the
    whole node becomes the vendor's ``vl``-role client -- one priority for the
    whole chain, matching how the candidate10 validation measured it. The
    ``fake`` implementation touches no NPU and never wraps.
    """
    from embodied_common.visual_interaction_contract import NODE_NAME

    if vl_implementation != "internvl":
        return Node(
            package="visual_interaction",
            executable="visual_interaction_node",
            name=NODE_NAME,
            parameters=[parameters],
            output="screen",
        )
    from robot_config.rt_infer_config import RtInferConfigError, parse_rt_infer_config

    try:
        rt_infer = parse_rt_infer_config(robot_config)
    except RtInferConfigError as exc:
        raise ValueError(f"rt_infer configuration is invalid: {exc}") from exc
    if not rt_infer.enable:
        return Node(
            package="visual_interaction",
            executable="visual_interaction_node",
            name=NODE_NAME,
            parameters=[parameters],
            output="screen",
        )
    from inference_service.launch_builders.perception_models import write_rt_infer_snapshot

    return Node(
        package="inference_service",
        executable="rt_infer_client",
        sigterm_timeout=str(rt_infer.stop_timeout_sec + 1),
        name=NODE_NAME,
        output="screen",
        parameters=[parameters],
        arguments=[
            "--role",
            "vl",
            "--rt-infer-config",
            str(write_rt_infer_snapshot(dict(robot_config))),
            "--node-main",
            "visual_interaction.visual_interaction_node:main",
            "--",
        ],
    )


__all__ = ["generate_visual_interaction_nodes"]


def build_visual_interaction(context: LaunchBuildContext) -> BuildResult:
    from visual_interaction.launch_builders.visual_interaction import generate_visual_interaction_nodes

    return BuildResult(actions=list(generate_visual_interaction_nodes(context.robot_config)))


VISUAL_INTERACTION_BUILDER = BuilderSpec(
    id="visual_interaction",
    factory=f"{__name__}:build_visual_interaction",
    timing="runtime_ready",
    # Topological edge to the device owner (binding only when rt_infer is
    # enabled; dropped otherwise).
    after=("rt_infer_scheduler",),
)
