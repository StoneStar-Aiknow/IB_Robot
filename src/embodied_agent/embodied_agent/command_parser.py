"""Rule-based parser for the embodied minimal closure."""

import re
from dataclasses import dataclass


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
) -> PlannedTask:
    """Parse a free-form Chinese command into a minimal deterministic skill plan."""
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

    contains_pick = _contains_any(normalized, ["抓", "拿", "取"])
    contains_place = "放" in normalized
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

    if contains_pick or contains_place or _contains_any(normalized, ["松开", "放开", "释放", "香蕉", "目标物"]):
        return _unsupported(text)

    return _unsupported(text)
