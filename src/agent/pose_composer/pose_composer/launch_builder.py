"""Launch wiring for the pose composer, contributed to robot.launch.py's builder registry.

Declared as an entry point in the ``ibrobot.launch_builders`` group (setup.py),
the way manipulation_execution contributes the mirror session. Participation is
decided by the robot config::

    pose_composer:
      enabled: true
      mjcf_path: ""        # default: the official model aimdk_fetch_x2_mjcf put in the cache
      llm:                 # OpenAI-compatible, like embodied.agent.planner
        provider: openai_compatible
        base_url: ...
        api_key_env: ...   # never a literal key
        model: ...
        timeout_sec: 60.0
        temperature: 0.3
        extra_body: {}

The runtime profile is the robot's own (``runtime.profile``); the poses run in
the runtime mode of ``skill_required_control_mode``, which the Gateway switches
to before a pose_composer skill runs.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

from launch_ros.actions import Node

from robot_config.launch_registry.registry import BuilderSpec, BuildResult, LaunchBuildContext

_KEYS = frozenset({"enabled", "mjcf_path", "llm"})
_LLM_KEYS = frozenset({"provider", "base_url", "api_key_env", "model", "timeout_sec", "temperature", "extra_body"})


def _llm_parameters(llm) -> dict:
    if not isinstance(llm, Mapping):
        raise ValueError("pose_composer.llm must be a mapping")
    unknown = sorted(set(llm) - _LLM_KEYS)
    if unknown:
        raise ValueError(f"pose_composer.llm has unknown keys {unknown} (a literal key is never accepted)")
    for key in ("base_url", "model"):
        if not isinstance(llm.get(key), str) or not llm[key].strip():
            raise ValueError(f"pose_composer.llm.{key} is required")
    provider = llm.get("provider", "openai_compatible")
    if provider not in {"openai_compatible", "kimicode"}:
        raise ValueError("pose_composer.llm.provider must be openai_compatible or kimicode")
    timeout = llm.get("timeout_sec", 60.0)
    temperature = llm.get("temperature", 0.3)
    for key, value in (("timeout_sec", timeout), ("temperature", temperature)):
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"pose_composer.llm.{key} must be a non-negative number")
    extra_body = llm.get("extra_body", {})
    if not isinstance(extra_body, Mapping):
        raise ValueError("pose_composer.llm.extra_body must be a mapping")
    return {
        "llm_provider": provider,
        "llm_base_url": llm["base_url"],
        "llm_api_key_env": str(llm.get("api_key_env", "")),
        "llm_model": llm["model"],
        "llm_timeout_sec": float(timeout),
        "llm_temperature": float(temperature),
        "llm_extra_body_json": json.dumps(dict(extra_body), sort_keys=True),
    }


def _trajectory_modes(robot_config: Mapping) -> list[str]:
    mode = str(robot_config.get("skill_required_control_mode", "") or "").strip()
    runtime_mode = str(((robot_config.get("control_modes") or {}).get(mode) or {}).get("runtime_mode", "") or "")
    if not runtime_mode.strip() or runtime_mode.strip() == "idle":
        raise ValueError(
            "pose_composer needs skill_required_control_mode with an explicit non-idle runtime_mode: "
            "its skills run through the Gateway in that mode"
        )
    return [runtime_mode.strip()]


def generate_pose_composer_nodes(robot_config: dict) -> list:
    from embodied_common.pose_composer_contract import NODE_NAME

    config = robot_config.get("pose_composer", {})
    if not isinstance(config, Mapping):
        raise ValueError("pose_composer must be a mapping")
    if config.get("enabled", False) is not True:
        return []
    unknown = sorted(set(config) - _KEYS)
    if unknown:
        raise ValueError(f"pose_composer has unknown keys {unknown}")
    runtime = robot_config.get("runtime") or {}
    if str(runtime.get("provider", "")) != "aimdk_robot":
        raise ValueError("pose_composer drives the X2 arms: runtime.provider must be aimdk_robot")
    from robot_runtime.launch_support import resolve_profile_argument

    profile = resolve_profile_argument("aimdk_robot", str(runtime.get("profile", "")))
    mjcf = str(config.get("mjcf_path", "") or "").strip()
    if not mjcf:
        from aimdk_robot.mock_physics import CACHE, PINNED_COMMIT

        mjcf = str(CACHE / PINNED_COMMIT / "X2_URDF-v1.3.0" / "scene.xml")
    if not Path(mjcf).expanduser().is_file():
        raise ValueError(f"pose_composer model not found: {mjcf} (fetch it: ros2 run aimdk_robot aimdk_fetch_x2_mjcf)")
    parameters = {
        "mjcf_path": str(Path(mjcf).expanduser()),
        "runtime_profile": str(profile),
        "trajectory_modes_json": json.dumps(_trajectory_modes(robot_config)),
        **_llm_parameters(config.get("llm")),
    }
    return [Node(package="pose_composer", executable="pose_composer_node", name=NODE_NAME, output="screen",
                 parameters=[parameters])]  # fmt: skip


def build_pose_composer(context: LaunchBuildContext) -> BuildResult:
    return BuildResult(actions=list(generate_pose_composer_nodes(context.robot_config)))


POSE_COMPOSER_BUILDER = BuilderSpec(
    id="pose_composer",
    factory=f"{__name__}:build_pose_composer",
    timing="runtime_ready",
)
