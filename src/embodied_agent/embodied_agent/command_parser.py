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


def _resolve_target_name(text: str, default_target_name: str) -> str:
    if "香蕉" in text:
        return "banana"
    if _contains_any(text, ["目标物", "物体", "目标"]):
        return "demo_object"
    return default_target_name


def _resolve_place_name(text: str, default_place_name: str) -> str:
    if _contains_any(text, ["右侧托盘", "右边托盘", "右托盘", "右边"]):
        return "tray_right"
    return default_place_name


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

    if _contains_any(normalized, ["观察桌面", "看看桌面", "查看桌面", "扫描桌面", "观察场景", "看看场景"]):
        return PlannedTask(
            task_type="observe_scene",
            target_name="",
            place_name="",
            motion_direction="",
            motion_distance=0.0,
            skill_sequence=["inspect_scene"],
        )

    if _contains_any(normalized, ["观察香蕉", "看看香蕉", "查看香蕉", "观察目标区域", "看看目标区域"]):
        return PlannedTask(
            task_type="observe_target_area",
            target_name=_resolve_target_name(normalized, default_target_name),
            place_name="",
            motion_direction="",
            motion_distance=0.0,
            skill_sequence=["observe_target_area"],
        )

    if _contains_any(normalized, ["回到home", "回原位", "回安全位", "回安全位置", "返回home"]):
        return PlannedTask(
            task_type="recover_safe_pose",
            target_name="",
            place_name="",
            motion_direction="",
            motion_distance=0.0,
            skill_sequence=["recover_safe_pose"],
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

    target_name = _resolve_target_name(normalized, default_target_name)
    contains_target_reference = target_name != default_target_name or _contains_any(
        normalized,
        ["目标物", "物体", "目标"],
    )

    if contains_target_reference and _contains_any(normalized, ["后撤", "退后", "撤回来", "远离"]):
        return PlannedTask(
            task_type="retreat_from_target",
            target_name=target_name,
            place_name="",
            motion_direction="",
            motion_distance=0.0,
            skill_sequence=["retreat_from_target"],
        )

    if contains_target_reference and _contains_any(normalized, ["抬起", "抬高", "举起", "提起来"]):
        return PlannedTask(
            task_type="lift_named_target",
            target_name=target_name,
            place_name="",
            motion_direction="",
            motion_distance=0.0,
            skill_sequence=["lift_named_target"],
        )

    if contains_target_reference and _contains_any(normalized, ["上面", "上方", "悬停"]):
        return PlannedTask(
            task_type="hover_named_target",
            target_name=target_name,
            place_name="",
            motion_direction="",
            motion_distance=0.0,
            skill_sequence=["hover_named_target"],
        )

    if contains_target_reference and _contains_any(normalized, ["靠近", "接近"]):
        return PlannedTask(
            task_type="approach_named_target",
            target_name=target_name,
            place_name="",
            motion_direction="",
            motion_distance=0.0,
            skill_sequence=["approach_named_target"],
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

    if contains_pick and contains_place:
        return PlannedTask(
            task_type="pick_and_place",
            target_name=target_name,
            place_name=_resolve_place_name(normalized, default_place_name),
            motion_direction="",
            motion_distance=0.0,
            skill_sequence=["pick_named_target", "place_named_pose"],
        )

    if contains_pick:
        return PlannedTask(
            task_type="pick_only",
            target_name=target_name,
            place_name="",
            motion_direction="",
            motion_distance=0.0,
            skill_sequence=["pick_named_target"],
        )

    if _contains_any(normalized, ["松开", "放开", "释放"]):
        return PlannedTask(
            task_type="release_at_named_pose",
            target_name="",
            place_name=_resolve_place_name(normalized, default_place_name),
            motion_direction="",
            motion_distance=0.0,
            skill_sequence=["release_at_named_pose"],
        )

    if contains_place:
        return PlannedTask(
            task_type="place_only",
            target_name="",
            place_name=_resolve_place_name(normalized, default_place_name),
            motion_direction="",
            motion_distance=0.0,
            skill_sequence=["place_named_pose"],
        )

    return PlannedTask(
        task_type="unknown",
        target_name="",
        place_name="",
        motion_direction="",
        motion_distance=0.0,
        skill_sequence=[],
        message=f"unsupported command: {text}",
    )
