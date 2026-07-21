"""Rule-based parser for the embodied minimal closure."""

import json
import re
from dataclasses import dataclass
from typing import Any

from embodied_common.skill_templates import is_skill_disabled


@dataclass
class PlannedTask:
    task_type: str
    target_name: str
    place_name: str
    motion_direction: str
    motion_distance: float
    skill_sequence: list[str]
    message: str = ""


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _match_skill_alias(text: str, skill_aliases: dict[str, list[str]] | None) -> str:
    if not skill_aliases:
        return ""
    alias_items = [
        (skill_name, keyword) for skill_name, keywords in skill_aliases.items() for keyword in keywords if keyword
    ]
    matched: list[tuple[str, str, int]] = []
    for skill_name, keyword in alias_items:
        pos = text.find(keyword)
        if pos >= 0:
            matched.append((skill_name, keyword, pos))
    if not matched:
        return ""
    # Prefer longer keyword; for equal length, prefer earlier position in text.
    matched.sort(key=lambda item: (-len(item[1]), item[2]))
    return matched[0][0]


def extract_skill_aliases(skill_templates: dict[str, Any] | None) -> dict[str, list[str]]:
    """Extract ``{skill_name: [zh_keywords]}`` from SSOT skill templates.

    Chinese trigger keywords live in ``description.aliases_zh`` and are exposed
    to the rule parser only when ``description.rule_entry`` is true. This keeps
    legacy command contracts separate while letting launch wiring and the parser
    share the same SSOT. Returns an empty dict when no opted-in aliases exist.
    """
    aliases: dict[str, list[str]] = {}
    if not isinstance(skill_templates, dict):
        return aliases
    for skill_name, template in skill_templates.items():
        if not isinstance(template, dict) or is_skill_disabled(template):
            continue
        description = template.get("description")
        if not isinstance(description, dict):
            continue
        if description.get("rule_entry") is not True:
            continue
        if description.get("requires_motion_params") is True:
            continue
        raw = description.get("aliases_zh")
        if isinstance(raw, list) and raw:
            keywords = [str(item).strip() for item in raw if str(item).strip()]
            if keywords:
                aliases[skill_name] = keywords
    return aliases


def load_skill_aliases(raw_json: str) -> dict[str, list[str]]:
    """Parse the SSOT-derived ``skill_aliases_json`` parameter string.

    Empty or invalid JSON yields an empty dict, leaving the rule parser on its
    legacy hardcoded keywords (graceful degradation when launch wiring has not
    been updated or the config lacks aliases).
    """
    raw_json = (raw_json or "").strip()
    if not raw_json:
        return {}
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {
        str(skill): [str(kw) for kw in keywords] for skill, keywords in parsed.items() if isinstance(keywords, list)
    }


def _resolve_motion_direction(text: str) -> str:
    direction_keywords = [
        ("forward", ["往前", "向前", "前一点", "往前一点", "向前一点"]),
        ("backward", ["往后", "向后", "后一点", "往后一点", "向后一点"]),
        ("left", ["往左", "向左", "左一点", "往左一点", "向左一点"]),
        ("right", ["往右", "向右", "右一点", "往右一点", "向右一点"]),
        ("up", ["往上", "向上", "上一点", "往上一点", "向上一点"]),
        ("down", ["往下", "向下", "朝下", "下一点", "往下一点", "向下一点"]),
    ]
    for direction, keywords in direction_keywords:
        if _contains_any(text, keywords):
            return direction
    return ""


def _unsupported(text: str) -> PlannedTask:
    return PlannedTask(
        task_type="unknown",
        target_name="",
        place_name="",
        motion_direction="",
        motion_distance=0.0,
        skill_sequence=[],
        message=f"unsupported command: {text}",
    )


def parse_text_command(
    text: str,
    default_target_name: str = "demo_object",
    default_place_name: str = "tray_right",
    default_relative_motion_step_m: float = 0.03,
    skill_aliases: dict[str, list[str]] | None = None,
) -> PlannedTask:
    """Parse a free-form Chinese command into a minimal deterministic skill plan.

    Parameter-bearing commands are parsed first so direction and angle values
    are preserved. Existing observation, recovery, and gripper branches retain
    their public task types. Finally, ``skill_aliases`` built from descriptions
    explicitly marked ``rule_entry`` route direct no-parameter skills.
    """
    normalized = (text or "").strip().replace(" ", "")
    if not normalized:
        return PlannedTask(
            task_type="unknown",
            target_name="",
            place_name="",
            motion_direction="",
            motion_distance=0.0,
            skill_sequence=[],
            message="empty command",
        )

    # Pick/place/target-object keywords must be checked BEFORE parameter motion
    # parsing so compound commands like "把香蕉往上移动一点" or "抓住后顺时针旋转30度"
    # are rejected, not silently routed to a bare move_relative_ee / rotate.
    contains_pick = _contains_any(normalized, ["抓", "拿", "取"])
    contains_place = _contains_any(normalized, ["放下", "放到", "放在", "摆放"])
    contains_target = _contains_any(normalized, ["松开", "放开", "释放", "香蕉", "目标物"])
    if contains_pick or contains_place or contains_target:
        return _unsupported(text)

    cw_match = re.search(r"顺时针(?:旋转)?(\d+(?:\.\d+)?)度", normalized)
    if cw_match:
        angle = float(cw_match.group(1))
        return PlannedTask(
            task_type="rotate_gripper_cw",
            target_name="",
            place_name="",
            motion_direction="",
            motion_distance=angle,
            skill_sequence=["rotate_gripper_cw"],
        )

    ccw_match = re.search(r"逆时针(?:旋转)?(\d+(?:\.\d+)?)度", normalized)
    if ccw_match:
        angle = float(ccw_match.group(1))
        return PlannedTask(
            task_type="rotate_gripper_ccw",
            target_name="",
            place_name="",
            motion_direction="",
            motion_distance=angle,
            skill_sequence=["rotate_gripper_ccw"],
        )

    motion_direction = _resolve_motion_direction(normalized)
    if motion_direction and _contains_any(normalized, ["一点", "一点点", "移动", "挪", "往", "向", "夹爪", "末端"]):
        return PlannedTask(
            task_type="relative_motion",
            target_name="",
            place_name="",
            motion_direction=motion_direction,
            motion_distance=default_relative_motion_step_m,
            skill_sequence=["move_relative_ee"],
        )

    if _contains_any(
        normalized, ["观察点", "观察位置", "观察桌面", "看看桌面", "查看桌面", "扫描桌面", "观察场景", "看看场景"]
    ):
        return PlannedTask(
            task_type="observe_scene",
            target_name="",
            place_name="",
            motion_direction="",
            motion_distance=0.0,
            skill_sequence=["inspect_scene"],
        )

    if _contains_any(normalized, ["原位", "原点", "回到home", "回原位", "回安全位", "回安全位置", "返回home"]):
        return PlannedTask(
            task_type="recover_safe_pose",
            target_name="",
            place_name="",
            motion_direction="",
            motion_distance=0.0,
            skill_sequence=["recover_safe_pose"],
        )

    if _contains_any(normalized, ["零点", "零位", "回零点", "到零点"]):
        return PlannedTask(
            task_type="recover_zero_pose",
            target_name="",
            place_name="",
            motion_direction="",
            motion_distance=0.0,
            skill_sequence=["recover_zero_pose"],
        )

    if _contains_any(normalized, ["打开夹爪", "张开夹爪", "展开夹爪", "夹爪打开", "夹爪张开", "开爪"]):
        return PlannedTask(
            task_type="open_gripper",
            target_name="",
            place_name="",
            motion_direction="",
            motion_distance=0.0,
            skill_sequence=["open_gripper_skill"],
        )

    if _contains_any(normalized, ["关闭夹爪", "合拢夹爪", "夹爪关闭", "夹爪合拢", "夹紧", "闭爪"]):
        return PlannedTask(
            task_type="close_gripper",
            target_name="",
            place_name="",
            motion_direction="",
            motion_distance=0.0,
            skill_sequence=["close_gripper_skill"],
        )

    # SSOT-injected aliases handle explicitly opted-in no-parameter skills.
    # Longer aliases match first so "正式打招呼" is not shadowed by "打招呼".
    alias_skill = _match_skill_alias(normalized, skill_aliases)
    if alias_skill:
        return PlannedTask(
            task_type=alias_skill,
            target_name="",
            place_name="",
            motion_direction="",
            motion_distance=0.0,
            skill_sequence=[alias_skill],
        )

    return _unsupported(text)
