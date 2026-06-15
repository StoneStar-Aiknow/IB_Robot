"""Build constrained prompts for continuous multimodal scene understanding."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

_MAX_USER_TEXT_LEN = 2000


def _sanitize_user_text(text: str) -> str:
    """Truncate overly long user inputs to avoid token overflows."""
    text = (text or "").strip()
    if len(text) > _MAX_USER_TEXT_LEN:
        text = text[:_MAX_USER_TEXT_LEN]
    return text


def append_images(content: list[dict[str, Any]], scene_snapshot: dict[str, Any]) -> None:
    """Append image_url content blocks from a scene snapshot to a content list."""
    images = scene_snapshot.get("images", [])
    if isinstance(images, list) and images:
        for item in images:
            image_data_url = str(item.get("image_data_url", "")).strip()
            if image_data_url:
                content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    else:
        image_data_url = scene_snapshot.get("image_data_url", "")
        if image_data_url:
            content.append({"type": "image_url", "image_url": {"url": image_data_url}})


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _infer_analysis_preferences(user_text: str, user_context: dict[str, Any]) -> dict[str, Any]:
    normalized = (user_text or "").strip()
    preferences: dict[str, Any] = {
        "raw_user_intent": normalized,
        "response_style": "balanced",
        "focus_scope": "current_scene",
        "task_intent": "scene_understanding",
        "priority_aspects": [],
        "spatial_focus": [],
        "user_context_keys": sorted(user_context.keys()),
    }

    if _contains_any(normalized, ["简短", "简洁", "一句话", "只回答", "只回复"]):
        preferences["response_style"] = "concise"
    elif _contains_any(normalized, ["详细", "具体", "尽量详细", "完整"]):
        preferences["response_style"] = "detailed"

    if _contains_any(normalized, ["桌面", "桌上"]):
        preferences["focus_scope"] = "tabletop"
    if _contains_any(normalized, ["左边", "左侧", "左"]):
        preferences["spatial_focus"].append("left")
    if _contains_any(normalized, ["右边", "右侧", "右"]):
        preferences["spatial_focus"].append("right")
    if _contains_any(normalized, ["前面", "前方", "前"]):
        preferences["spatial_focus"].append("front")
    if _contains_any(normalized, ["后面", "后方", "后"]):
        preferences["spatial_focus"].append("back")

    if _contains_any(normalized, ["抓", "夹", "拿", "取", "放", "移动", "挪"]):
        preferences["task_intent"] = "manipulation_assessment"
        preferences["priority_aspects"] = [
            "target object identity and location",
            "graspability and reachability",
            "occlusions and blockers",
            "safety risks or uncertainty",
        ]
    elif _contains_any(normalized, ["在哪里", "找", "定位", "哪个"]):
        preferences["task_intent"] = "object_search"
        preferences["priority_aspects"] = [
            "candidate object locations",
            "visual evidence and uncertainty",
            "distinguishing features",
        ]
    elif _contains_any(normalized, ["风险", "危险", "安全", "碰撞", "会不会"]):
        preferences["task_intent"] = "risk_assessment"
        preferences["priority_aspects"] = [
            "potential collision or instability",
            "uncertainty sources",
            "safety recommendations",
        ]
    elif _contains_any(normalized, ["看看", "观察", "查看", "有什么"]):
        preferences["task_intent"] = "scene_overview"
        preferences["priority_aspects"] = [
            "main visible objects",
            "robot pose relevance",
            "notable uncertainty or occlusion",
        ]

    if not preferences["priority_aspects"]:
        preferences["priority_aspects"] = [
            "main visible objects",
            "robot state relevance",
            "uncertainty and risks",
        ]

    return preferences


def build_scene_analysis_messages(
    user_text: str,
    user_context: dict[str, Any],
    scene_snapshot: dict[str, Any],
    conversation_history: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    user_text = _sanitize_user_text(user_text)
    analysis_preferences = _infer_analysis_preferences(user_text, user_context)
    system_text = (
        "You are a continuous multimodal scene understanding assistant for IB_Robot.\n"
        "You receive one or more live robot camera images, current robot state, optional RGB-D summaries, "
        "user question, and optional user context.\n"
        "The user question is the primary semantic objective, not generic background text.\n"
        "You must infer the user's analysis preference from the user question and optional context, "
        "then bias the scene understanding toward what the user actually cares about.\n"
        "For manipulation-related requests, prioritize target identity, target location, graspability, "
        "occlusion, blockers, and action risks over generic scene description.\n"
        "For search requests, prioritize where the requested object may be and what evidence supports it.\n"
        "For overview requests, summarize the scene globally but still respect any focus region or user hints.\n"
        "Return JSON only, in Chinese.\n"
        "The JSON object must contain fields: "
        "scene_summary, visible_objects, robot_state_summary, ee_pose_interpretation, risks, confidence.\n"
        "When the visible object location is useful, also include optional field objects. "
        "objects must be an array of objects with fields: label, bbox_2d, confidence, attributes. "
        "bbox_2d must be [x1, y1, x2, y2] in primary image pixel coordinates. "
        "Only output bbox_2d when you can localize the object from the image; do not guess.\n"
        "visible_objects and risks must be arrays of strings.\n"
        "Do not output code fences.\n"
    )

    snapshot_text = json.dumps(
        {
            "user_text": user_text,
            "user_context": user_context,
            "analysis_preferences": analysis_preferences,
            "camera_topic": scene_snapshot.get("camera_topic"),
            "camera_views": scene_snapshot.get("camera_views", {}),
            "rgbd_context": scene_snapshot.get("rgbd_context", {}),
            "ee_pose": scene_snapshot.get("ee_pose"),
            "joint_state": scene_snapshot.get("joint_state"),
        },
        ensure_ascii=False,
        indent=2,
    )

    messages: list[dict[str, Any]] = [{"role": "system", "content": [{"type": "text", "text": system_text}]}]

    for item in conversation_history:
        role = item.get("role", "").strip()
        text = item.get("text", "").strip()
        if role in {"user", "assistant"} and text:
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    user_text_block = (
        "请结合当前图像、机械臂状态、用户原始输入以及用户补充信息进行场景理解。\n"
        "不要把用户输入当作固定模板文本，而要把它理解为用户当前关注点、任务目标、回答偏好和判断标准。\n"
        "如果用户输入带有动作意图（如抓取、夹取、移动），请优先分析目标是否清晰可见、位于哪里、是否适合操作、"
        "有哪些遮挡/风险，以及还缺少哪些关键信息。\n"
        "如果提供了多路图像，请联合利用 primary 和 wrist 视角判断；如果提供了 RGB-D 摘要，请把深度/空间信息用于距离、"
        "遮挡、可达性和风险判断，但不要虚构未给出的几何细节。\n"
        "输出 JSON 字段：scene_summary, visible_objects, robot_state_summary, "
        "ee_pose_interpretation, risks, confidence。若能定位物体，请额外输出 objects，"
        "每个 object 包含 label、bbox_2d、confidence、attributes。bbox_2d 使用 primary 图像像素坐标 [x1,y1,x2,y2]。\n"
        f"当前输入:\n{snapshot_text}"
    )
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text_block}]
    append_images(user_content, scene_snapshot)

    messages.append({"role": "user", "content": user_content})
    return messages
