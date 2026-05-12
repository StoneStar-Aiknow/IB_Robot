"""Fallback planner that reuses the existing rule-based parser."""

from embodied_agent.command_parser import parse_text_command
from vlm_task_planner.response_parser import PlannerResult


def fallback_plan_from_text(
    text: str,
    default_target_name: str,
    default_place_name: str,
    default_relative_motion_step_m: float,
) -> PlannerResult:
    plan = parse_text_command(
        text,
        default_target_name=default_target_name,
        default_place_name=default_place_name,
        default_relative_motion_step_m=default_relative_motion_step_m,
    )
    return PlannerResult(
        task_type=plan.task_type,
        target_name=plan.target_name,
        place_name=plan.place_name,
        motion_direction=plan.motion_direction,
        motion_distance=plan.motion_distance,
        skill_sequence=list(plan.skill_sequence),
        required_missing_skills=[],
        confidence=0.0,
        scene_summary="",
        planner_reason="",
        planner_source="rule_fallback",
    )
