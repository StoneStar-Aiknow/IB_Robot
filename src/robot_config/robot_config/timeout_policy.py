"""Helpers for resolving embodied timeout policy from robot_config."""

from __future__ import annotations

import math
from typing import Any

DEFAULT_EMBODIED_TIMEOUT_POLICY: dict[str, float] = {
    "task_budget_sec": 180.0,
    "default_skill_timeout_sec": 120.0,
    "robot_state_freshness_sec": 0.5,
    "scene_freshness_sec": 0.5,
    "model_idle_timeout_sec": 120.0,
    "rpc_timeout_sec": 5.0,
    "gripper_settle_sec": 1.5,
}
# ROS 2 Humble generated Python float32 setters reject the exact IEEE-754 maximum.
_FLOAT32_MAX = 3.402823466e38
_GATEWAY_FLOAT32_TIMEOUTS = {
    "default_skill_timeout_sec",
    "task_budget_sec",
    "rpc_timeout_sec",
    "robot_state_freshness_sec",
}


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _finite_positive_timeout(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"embodied.timeouts.{name} must be a finite positive number")
    try:
        timeout = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"embodied.timeouts.{name} must be a finite positive number") from exc
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError(f"embodied.timeouts.{name} must be a finite positive number")
    if name in _GATEWAY_FLOAT32_TIMEOUTS and timeout > _FLOAT32_MAX:
        raise ValueError(f"embodied.timeouts.{name} must not exceed float32 maximum")
    return timeout


def resolve_embodied_timeout_policy(embodied_config: dict[str, Any]) -> dict[str, float]:
    """Resolve a compact timeout policy while keeping legacy fields compatible."""

    execution = _mapping(embodied_config.get("execution", {}), "embodied.execution")
    perception = _mapping(embodied_config.get("perception", {}), "embodied.perception")
    perception_scene_sources = _mapping(perception.get("scene_sources", {}), "embodied.perception.scene_sources")
    perception_vlm_api = _mapping(perception.get("vlm_api", {}), "embodied.perception.vlm_api")
    configured = _mapping(embodied_config.get("timeouts", {}), "embodied.timeouts")

    scene_freshness_fallback = perception_scene_sources.get(
        "max_scene_age_sec", DEFAULT_EMBODIED_TIMEOUT_POLICY["scene_freshness_sec"]
    )
    model_idle_fallback = perception_vlm_api.get(
        "timeout_sec", DEFAULT_EMBODIED_TIMEOUT_POLICY["model_idle_timeout_sec"]
    )

    policy = {
        "task_budget_sec": configured.get(
            "task_budget_sec",
            execution.get("skill_timeout_sec", DEFAULT_EMBODIED_TIMEOUT_POLICY["task_budget_sec"]),
        ),
        "default_skill_timeout_sec": configured.get(
            "default_skill_timeout_sec",
            execution.get("skill_timeout_sec", DEFAULT_EMBODIED_TIMEOUT_POLICY["default_skill_timeout_sec"]),
        ),
        "robot_state_freshness_sec": configured.get(
            "robot_state_freshness_sec",
            DEFAULT_EMBODIED_TIMEOUT_POLICY["robot_state_freshness_sec"],
        ),
        "scene_freshness_sec": configured.get("scene_freshness_sec", scene_freshness_fallback),
        "model_idle_timeout_sec": configured.get("model_idle_timeout_sec", model_idle_fallback),
        "rpc_timeout_sec": configured.get("rpc_timeout_sec", DEFAULT_EMBODIED_TIMEOUT_POLICY["rpc_timeout_sec"]),
        "gripper_settle_sec": configured.get(
            "gripper_settle_sec",
            execution.get("primitive_wait_sec", DEFAULT_EMBODIED_TIMEOUT_POLICY["gripper_settle_sec"]),
        ),
    }
    resolved_policy = {name: _finite_positive_timeout(name, value) for name, value in policy.items()}
    if resolved_policy["default_skill_timeout_sec"] > resolved_policy["task_budget_sec"]:
        raise ValueError("embodied.timeouts.default_skill_timeout_sec must be <= task_budget_sec")
    return resolved_policy
