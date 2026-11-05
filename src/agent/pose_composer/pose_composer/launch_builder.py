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
      memory:              # optional; shared by every conversation, kept across restarts
        enabled: true
        path: ""           # default ~/.local/share/ibrobot/pose_memory/<robot name>.sqlite3
        max_poses: 500
        candidate_ttl_days: 7
        seeds: true        # load the package's pinned seed poses
      lookup:
        enabled: true      # default; query existing seeds/memory independently of writes

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

from embodied_common.pose_composer_contract import lookup_enabled
from robot_config.launch_registry.registry import BuilderSpec, BuildResult, LaunchBuildContext

_KEYS = frozenset(
    {
        "enabled",
        "mjcf_path",
        "llm",
        "memory",
        "lookup",
        "max_joint_delta_rad",
        "max_hand_displacement_m",
        "mode",
        "fixed_library",
        "fixed_library_sha256",
        "interfaces",
        "path_validation",
    }
)
_MEMORY_KEYS = frozenset({"enabled", "path", "max_poses", "candidate_ttl_days", "seeds"})
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


def memory_settings(robot_config: Mapping) -> dict | None:
    """The validated pose_composer.memory block, or None when the memory is off."""
    config = robot_config.get("pose_composer") or {}
    memory = config.get("memory") if isinstance(config, Mapping) else None
    if memory is None or (isinstance(memory, Mapping) and memory.get("enabled", False) is False):
        return None
    if not isinstance(memory, Mapping) or memory.get("enabled") is not True:
        raise ValueError("pose_composer.memory.enabled must be a boolean")
    unknown = sorted(set(memory) - _MEMORY_KEYS)
    if unknown:
        raise ValueError(f"pose_composer.memory has unknown keys {unknown}")
    name = str(robot_config.get("name", "robot") or "robot")
    path = str(memory.get("path", "") or "").strip() or f"~/.local/share/ibrobot/pose_memory/{name}.sqlite3"
    max_poses = memory.get("max_poses", 500)
    ttl = memory.get("candidate_ttl_days", 7)
    if isinstance(max_poses, bool) or not isinstance(max_poses, int) or max_poses <= 0:
        raise ValueError("pose_composer.memory.max_poses must be a positive integer")
    if isinstance(ttl, bool) or not isinstance(ttl, int | float) or not math.isfinite(ttl) or ttl <= 0:
        raise ValueError("pose_composer.memory.candidate_ttl_days must be a positive number")
    seeds = memory.get("seeds", False)
    if not isinstance(seeds, bool):
        raise ValueError("pose_composer.memory.seeds must be a boolean")
    return {"path": str(Path(path).expanduser()), "max_poses": max_poses, "candidate_ttl_days": float(ttl),
            "seeds": seeds}  # fmt: skip


def _memory_parameters(robot_config: Mapping) -> dict:
    memory = memory_settings(robot_config)
    if memory is None:
        return {"memory_path": ""}
    seeds_path = ""
    if memory["seeds"]:
        from ament_index_python.packages import get_package_share_directory

        seeds_path = str(Path(get_package_share_directory("pose_composer")) / "seeds" / "x2_poses.yaml")
    return {
        "memory_path": memory["path"],
        "memory_max_poses": memory["max_poses"],
        "memory_candidate_ttl_days": memory["candidate_ttl_days"],
        "memory_seeds_path": seeds_path,
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


def _hand_restore_parameters(robot_config: dict) -> str:
    """What poses that open hands need from the deployment, or "" when no configured pose opens one.

    The open preset belongs to the deployed grasp model, so ``grasp_execution`` is required; the hand and
    arm interfaces come from the bound runtime description, never from vendor topic names.
    """
    poses = (robot_config.get("task_poses") or {}).get("poses") or {}
    if not any(pose.get("hands") for pose in poses.values()):
        return ""
    runtime = robot_config.get("runtime") or {}
    description = runtime.get("interface_description")
    grasp_execution = robot_config.get("grasp_execution")
    if not isinstance(description, Mapping) or not description or not isinstance(grasp_execution, Mapping):
        raise ValueError(
            "task poses that open hands need the bound runtime.interface_description and grasp_execution "
            "(the open preset belongs to the deployed grasp model)"
        )
    embodied = robot_config.get("embodied") or {}
    return json.dumps(
        {
            "interface_description": description,
            "grasp_execution": grasp_execution,
            "gateway_status_service": embodied.get(
                "skill_gateway_status_service", "/embodied/get_skill_gateway_status"
            ),
        },
        sort_keys=True,
    )


def generate_pose_composer_nodes(robot_config: dict, *, use_sim: bool | None = None) -> list:
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
    bound_parameters = {"lookup_enabled": lookup_enabled(config)}
    for role, entry in (config.get("interfaces") or {}).items():
        parameter = {"trajectory": "trajectory_action", "status": "runtime_status_service"}.get(role)
        if parameter is None or not isinstance(entry, Mapping) or not entry.get("endpoint"):
            raise ValueError(f"pose_composer.interfaces.{role} requires a bound trajectory/status endpoint")
        bound_parameters[parameter] = entry["endpoint"]
    description = runtime.get("interface_description") or {}
    state = (description.get("interfaces") or {}).get("joint.state") or {}
    if state.get("endpoint"):
        bound_parameters["joint_state_topic"] = state["endpoint"]
    for key in ("max_joint_delta_rad", "max_hand_displacement_m"):
        value = config.get(key, 0.0)
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"pose_composer.{key} must be finite and non-negative")
    mode = config.get("mode", "generated")
    from robot_config.path_validation import resolve_path_validation

    resolve_path_validation(robot_config, robot_config.get("_config_path", "."))
    if mode not in {"generated", "fixed"}:
        raise ValueError("pose_composer.mode must be generated or fixed")
    if mode == "fixed":
        if config.get("llm") or memory_settings(robot_config) is not None:
            raise ValueError("fixed mode does not allow language generation or mutable pose memory")
        if config.get("max_hand_displacement_m", 0):
            raise ValueError("fixed mode does not implement Cartesian commissioning limits")
        from robot_config.fixed_poses import resolve_fixed_poses

        resolve_fixed_poses(robot_config, robot_config.get("_config_path", "."))
        hands = _hand_restore_parameters(robot_config)
        if not hands:
            raise ValueError("fixed X2 execution requires live articulated-hand checks")
        return [
            Node(
                package="pose_composer",
                executable="pose_composer_node",
                name=NODE_NAME,
                output="screen",
                parameters=[
                    {
                        "fixed_library": config["fixed_library"],
                        "fixed_library_sha256": config["fixed_library_sha256"],
                        "path_validation_json": json.dumps(config["path_validation"], sort_keys=True),
                        "runtime_profile": str(profile),
                        "arm_limits_json": json.dumps(robot_config.get("arm_limits")),
                        "task_poses_json": json.dumps(robot_config.get("task_poses") or {}, sort_keys=True),
                        "task_pose_simulation": bool(robot_config.get("use_sim", False))
                        if use_sim is None
                        else use_sim,
                        "hand_restore_json": hands,
                        "trajectory_modes_json": json.dumps(_trajectory_modes(robot_config)),
                        "max_joint_delta_rad": float(config.get("max_joint_delta_rad", 0.0)),
                        **bound_parameters,
                    }
                ],
            )
        ]
    mjcf = str(config.get("mjcf_path", "") or "").strip()
    if not mjcf:
        from aimdk_robot.mock_physics import CACHE, PINNED_COMMIT

        mjcf = str(CACHE / PINNED_COMMIT / "X2_URDF-v1.3.0" / "scene.xml")
    if not Path(mjcf).expanduser().is_file():
        raise ValueError(f"pose_composer model not found: {mjcf} (fetch it: ros2 run aimdk_robot aimdk_fetch_x2_mjcf)")
    parameters = {
        **bound_parameters,
        "task_poses_json": json.dumps(robot_config.get("task_poses") or {}, sort_keys=True),
        "task_pose_simulation": bool(robot_config.get("use_sim", False)) if use_sim is None else use_sim,
        "arm_limits_json": json.dumps(robot_config.get("arm_limits")),
        "mjcf_path": str(Path(mjcf).expanduser()),
        "runtime_profile": str(profile),
        "trajectory_modes_json": json.dumps(_trajectory_modes(robot_config)),
        **(
            _llm_parameters(config["llm"])
            if config.get("llm") is not None or not robot_config.get("task_poses")
            else {}
        ),
        **_memory_parameters(robot_config),
    }
    hands = _hand_restore_parameters(robot_config)
    if hands:
        parameters["hand_restore_json"] = hands
    for key in ("max_joint_delta_rad", "max_hand_displacement_m"):
        value = config.get(key, 0.0)
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"pose_composer.{key} must be finite and non-negative")
        parameters[key] = float(value)
    return [Node(package="pose_composer", executable="pose_composer_node", name=NODE_NAME, output="screen",
                 parameters=[parameters])]  # fmt: skip


def build_pose_composer(context: LaunchBuildContext) -> BuildResult:
    return BuildResult(actions=list(generate_pose_composer_nodes(context.robot_config, use_sim=context.use_sim)))


POSE_COMPOSER_BUILDER = BuilderSpec(
    id="pose_composer",
    factory=f"{__name__}:build_pose_composer",
    timing="runtime_ready",
)
