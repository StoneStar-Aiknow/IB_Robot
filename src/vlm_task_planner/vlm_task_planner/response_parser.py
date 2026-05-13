"""Parse VLM planner responses into constrained task plans."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class PlannerResult:
    task_type: str
    target_name: str
    place_name: str
    motion_direction: str
    motion_distance: float
    skill_sequence: list[str]
    required_missing_skills: list[str]
    confidence: float
    scene_summary: str
    planner_reason: str
    planner_source: str


def _extract_json_blob(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("planner response is empty")

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text, i)
                return json.dumps(obj, ensure_ascii=False)
            except json.JSONDecodeError:
                continue
    raise ValueError("planner response does not contain a JSON object")


def _infer_task_type(skill_sequence: Sequence[str]) -> str:
    if list(skill_sequence) == ["inspect_scene"]:
        return "observe_scene"
    if list(skill_sequence) == ["observe_target_area"]:
        return "observe_target_area"
    if list(skill_sequence) == ["approach_named_target"]:
        return "approach_named_target"
    if list(skill_sequence) == ["hover_named_target"]:
        return "hover_named_target"
    if list(skill_sequence) == ["recover_safe_pose"]:
        return "recover_safe_pose"
    if list(skill_sequence) == ["pick_named_target"]:
        return "pick_only"
    if list(skill_sequence) == ["lift_named_target"]:
        return "lift_named_target"
    if list(skill_sequence) == ["retreat_from_target"]:
        return "retreat_from_target"
    if list(skill_sequence) == ["place_named_pose"]:
        return "place_only"
    if list(skill_sequence) == ["release_at_named_pose"]:
        return "release_at_named_pose"
    if list(skill_sequence) == ["move_relative_ee"]:
        return "relative_motion"
    if "pick_named_target" in skill_sequence and "place_named_pose" in skill_sequence:
        return "pick_and_place"
    return "planned_task"


def _string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a string list")
    return [str(item).strip() for item in value if str(item).strip()]


def _parse_confidence(value: Any) -> float:
    if value is None or value == "":
        confidence = 1.0
    elif isinstance(value, int | float):
        confidence = float(value)
    elif isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            confidence = 1.0
        else:
            keyword_mapping = {
                "low": 0.25,
                "medium": 0.6,
                "high": 0.85,
                "低": 0.25,
                "中": 0.6,
                "中等": 0.6,
                "高": 0.85,
            }
            if normalized in keyword_mapping:
                confidence = keyword_mapping[normalized]
            elif normalized.endswith("%"):
                confidence = float(normalized[:-1].strip()) / 100.0
            else:
                confidence = float(normalized)
    else:
        raise ValueError("confidence must be numeric or a supported string label")

    if confidence > 1.0 and confidence <= 100.0:
        confidence /= 100.0
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be in [0.0, 1.0]")
    return confidence


def parse_planner_response(
    raw_text: str,
    allowed_skills: Sequence[str],
    default_target_name: str,
    default_place_name: str,
    default_relative_motion_step_m: float,
) -> PlannerResult:
    payload = json.loads(_extract_json_blob(raw_text))
    skill_items = payload.get("skill_sequence", [])
    if not isinstance(skill_items, list):
        raise ValueError("planner response skill_sequence must be a list")

    allowed_skill_set = set(allowed_skills)
    skill_sequence: list[str] = []
    required_missing_skills = _string_list(
        payload.get("required_missing_skills"),
        "required_missing_skills",
    )
    target_name = str(payload.get("target_name", ""))
    place_name = str(payload.get("place_name", ""))
    motion_direction = str(payload.get("motion_direction", ""))
    motion_distance = float(payload.get("motion_distance", 0.0) or 0.0)
    planner_reason = str(payload.get("planner_reason", "")).strip()

    if not skill_items and not required_missing_skills:
        raise ValueError("planner response must contain a non-empty skill_sequence list")

    for skill_item in skill_items:
        if isinstance(skill_item, str):
            skill_name = skill_item
            args: dict[str, Any] = {}
        elif isinstance(skill_item, dict):
            skill_name = str(skill_item.get("skill_name", "")).strip()
            args = skill_item.get("args", {}) or {}
            if not isinstance(args, dict):
                raise ValueError(f"planner args for skill '{skill_name}' must be a JSON object")
        else:
            raise ValueError("each skill entry must be a string or object")

        if not skill_name:
            raise ValueError("planner skill entry is missing skill_name")
        if skill_name not in allowed_skill_set:
            raise ValueError(f"planner selected unsupported skill: {skill_name}")

        skill_sequence.append(skill_name)
        if (
            skill_name
            in {
                "observe_target_area",
                "approach_named_target",
                "hover_named_target",
                "pick_named_target",
                "lift_named_target",
                "retreat_from_target",
            }
            and not target_name
        ):
            target_name = str(args.get("target_name", default_target_name))
        if skill_name in {"place_named_pose", "release_at_named_pose"} and not place_name:
            place_name = str(args.get("place_name", default_place_name))
        if skill_name == "move_relative_ee":
            if not motion_direction:
                motion_direction = str(args.get("motion_direction", ""))
            if motion_distance <= 0.0:
                motion_distance = float(args.get("motion_distance", default_relative_motion_step_m))

    if (
        any(
            skill_name in skill_sequence
            for skill_name in (
                "observe_target_area",
                "approach_named_target",
                "hover_named_target",
                "pick_named_target",
                "lift_named_target",
                "retreat_from_target",
            )
        )
        and not target_name
    ):
        target_name = default_target_name
    if (
        any(skill_name in skill_sequence for skill_name in ("place_named_pose", "release_at_named_pose"))
        and not place_name
    ):
        place_name = default_place_name
    if "move_relative_ee" in skill_sequence and motion_distance <= 0.0:
        motion_distance = default_relative_motion_step_m

    confidence = _parse_confidence(payload.get("confidence", 1.0))
    scene_summary = str(payload.get("scene_summary", "")).strip()
    task_type = str(payload.get("intent", "")).strip() or _infer_task_type(skill_sequence)
    if required_missing_skills and not task_type:
        task_type = "unsupported_task"

    return PlannerResult(
        task_type=task_type,
        target_name=target_name,
        place_name=place_name,
        motion_direction=motion_direction,
        motion_distance=motion_distance,
        skill_sequence=skill_sequence,
        required_missing_skills=required_missing_skills,
        confidence=confidence,
        scene_summary=scene_summary,
        planner_reason=planner_reason,
        planner_source="vlm_api",
    )
