"""Parse scene-understanding model responses into structured analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass

from embodied_common.json_utils import extract_json_blob, parse_confidence, string_list


@dataclass
class SceneAnalysis:
    scene_summary: str
    visible_objects: list[str]
    robot_state_summary: str
    ee_pose_interpretation: str
    risks: list[str]
    confidence: float


def parse_scene_analysis_response(raw_text: str) -> SceneAnalysis:
    payload = json.loads(extract_json_blob(raw_text, "scene analysis response"))
    confidence = parse_confidence(payload.get("confidence", 0.0))

    return SceneAnalysis(
        scene_summary=str(payload.get("scene_summary", "")).strip(),
        visible_objects=string_list(payload.get("visible_objects"), "visible_objects"),
        robot_state_summary=str(payload.get("robot_state_summary", "")).strip(),
        ee_pose_interpretation=str(payload.get("ee_pose_interpretation", "")).strip(),
        risks=string_list(payload.get("risks"), "risks"),
        confidence=confidence,
    )
