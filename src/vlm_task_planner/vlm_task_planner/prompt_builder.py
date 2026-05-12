"""Build constrained prompts for the VLM planner."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any


def build_chat_messages(
    task_text: str,
    scene_snapshot: dict[str, Any],
    scene_analysis: dict[str, Any] | None,
    allowed_skills: Sequence[str],
    named_poses: dict[str, Any],
    named_targets: dict[str, Any],
    workspace: dict[str, Any],
    relative_motion_reference_frame: str,
    relative_motion_direction_mapping: dict[str, Any],
) -> list[dict[str, Any]]:
    system_text = (
        "You are a robot task planner for IB_Robot.\n"
        "You must output JSON only.\n"
        "You must not output raw motor commands, joint angles, or cartesian poses.\n"
        "You may only choose skills from the allowed skill list.\n"
        "You must use the provided scene understanding result together with the live robot state, multi-view image "
        "context, and any RGB-D spatial summary.\n"
        "Decompose the task into the smallest reasonable sequence of allowed skills.\n"
        "If the task requires capabilities that are not present in the allowed skill list, "
        "do not invent a workaround. Return an empty skill_sequence and list them in required_missing_skills.\n"
        "If the camera image or state is insufficient, lower confidence or choose a conservative plan.\n"
        "The skill_sequence field must be a list of objects with skill_name and args.\n"
        "target_name must be chosen from named_targets when a pick skill is used.\n"
        "place_name must be chosen from named_poses when a place skill is used.\n"
        "IMPORTANT: When the user is asking a pure scene description or observation question "
        "(e.g. 'what do you see', 'describe the scene', 'what is in front of you'), "
        "you MUST use inspect_scene and MUST NOT use observe_target_area or any skill that moves the arm. "
        "observe_target_area is only for pre-manipulation positioning, not for answering questions.\n"
    )

    snapshot_text = json.dumps(
        {
            "task_text": task_text,
            "scene_understanding": scene_analysis or {},
            "allowed_skills": list(allowed_skills),
            "named_poses": named_poses,
            "named_targets": named_targets,
            "workspace": workspace,
            "relative_motion_reference_frame": relative_motion_reference_frame,
            "relative_motion_direction_mapping": relative_motion_direction_mapping,
            "ee_pose": scene_snapshot.get("ee_pose"),
            "joint_state": scene_snapshot.get("joint_state"),
            "camera_topic": scene_snapshot.get("camera_topic"),
            "camera_views": scene_snapshot.get("camera_views", {}),
            "rgbd_context": scene_snapshot.get("rgbd_context", {}),
        },
        ensure_ascii=False,
        indent=2,
    )

    user_text = (
        "Plan the robot task using the provided scene and command.\n"
        "Return a JSON object with fields: intent, required_missing_skills, planner_reason, "
        "skill_sequence, target_name, place_name, motion_direction, motion_distance, confidence, scene_summary.\n"
        "If the task cannot be completed with the allowed skills, required_missing_skills must be a non-empty list "
        "and skill_sequence must be [].\n"
        "If multiple camera views are provided, use them jointly. If RGB-D summaries are provided, use them to reason "
        "about spatial clearance, reachability, occlusion, and collision risk.\n"
        f"Scene:\n{snapshot_text}"
    )

    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    images = scene_snapshot.get("images", [])
    if isinstance(images, list) and images:
        for item in images:
            image_data_url = str(item.get("image_data_url", "")).strip()
            if image_data_url:
                user_content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    else:
        image_data_url = scene_snapshot.get("image_data_url", "")
        if image_data_url:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url,
                    },
                }
            )

    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": user_content},
    ]
