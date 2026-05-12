"""Helpers for resolving embodied timeout policy from robot_config."""

from __future__ import annotations

from typing import Any

DEFAULT_EMBODIED_TIMEOUT_POLICY: dict[str, float] = {
    "task_budget_sec": 180.0,
    "scene_freshness_sec": 0.5,
    "model_idle_timeout_sec": 120.0,
    "rpc_timeout_sec": 5.0,
    "gripper_settle_sec": 1.5,
}


def resolve_embodied_timeout_policy(embodied_config: dict[str, Any]) -> dict[str, float]:
    """Resolve a compact timeout policy while keeping legacy fields compatible."""

    execution = embodied_config.get("execution", {})
    planner = embodied_config.get("planner", {})
    planner_scene_sources = planner.get("scene_sources", {})
    planner_vlm_api = planner.get("vlm_api", {})
    perception = embodied_config.get("perception", {})
    perception_scene_sources = perception.get("scene_sources", {})
    perception_vlm_api = perception.get("vlm_api", {})
    configured = embodied_config.get("timeouts", {})

    scene_freshness_fallback = planner_scene_sources.get(
        "max_scene_age_sec",
        perception_scene_sources.get("max_scene_age_sec", DEFAULT_EMBODIED_TIMEOUT_POLICY["scene_freshness_sec"]),
    )
    model_idle_fallback = planner_vlm_api.get(
        "timeout_sec",
        perception_vlm_api.get("timeout_sec", DEFAULT_EMBODIED_TIMEOUT_POLICY["model_idle_timeout_sec"]),
    )

    return {
        "task_budget_sec": float(
            configured.get(
                "task_budget_sec",
                execution.get("skill_timeout_sec", DEFAULT_EMBODIED_TIMEOUT_POLICY["task_budget_sec"]),
            )
        ),
        "scene_freshness_sec": float(configured.get("scene_freshness_sec", scene_freshness_fallback)),
        "model_idle_timeout_sec": float(configured.get("model_idle_timeout_sec", model_idle_fallback)),
        "rpc_timeout_sec": float(configured.get("rpc_timeout_sec", DEFAULT_EMBODIED_TIMEOUT_POLICY["rpc_timeout_sec"])),
        "gripper_settle_sec": float(
            configured.get(
                "gripper_settle_sec",
                execution.get("primitive_wait_sec", DEFAULT_EMBODIED_TIMEOUT_POLICY["gripper_settle_sec"]),
            )
        ),
    }
