"""Parse scene-understanding model responses into structured analysis."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class SceneAnalysis:
    scene_summary: str
    visible_objects: list[str]
    robot_state_summary: str
    ee_pose_interpretation: str
    risks: list[str]
    confidence: float


def _extract_json_blob(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("scene analysis response is empty")

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("scene analysis response does not contain a JSON object")
    return match.group(0)


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
        confidence = 0.0
    elif isinstance(value, int | float):
        confidence = float(value)
    elif isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            confidence = 0.0
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


def parse_scene_analysis_response(raw_text: str) -> SceneAnalysis:
    payload = json.loads(_extract_json_blob(raw_text))
    confidence = _parse_confidence(payload.get("confidence", 0.0))

    return SceneAnalysis(
        scene_summary=str(payload.get("scene_summary", "")).strip(),
        visible_objects=_string_list(payload.get("visible_objects"), "visible_objects"),
        robot_state_summary=str(payload.get("robot_state_summary", "")).strip(),
        ee_pose_interpretation=str(payload.get("ee_pose_interpretation", "")).strip(),
        risks=_string_list(payload.get("risks"), "risks"),
        confidence=confidence,
    )
