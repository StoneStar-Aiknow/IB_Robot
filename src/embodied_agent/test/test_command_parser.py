import pytest

from embodied_agent.command_parser import parse_text_command
from embodied_agent.task_context import load_task_context
from embodied_agent.task_entry_node import build_direct_planned_task


@pytest.mark.parametrize(
    "text",
    [
        "抓取目标物并放到右侧托盘",
        "抓香蕉",
        "观察香蕉",
        "把夹爪移动到香蕉的上面",
        "把夹爪往靠近香蕉的方向移动",
        "把香蕉抬起来",
        "从香蕉旁边后撤",
        "在右侧托盘松开夹爪",
        "放到右侧托盘",
    ],
)
def test_parse_target_and_grasp_commands_are_disabled(text):
    plan = parse_text_command(text)
    assert plan.task_type == "unknown"
    assert not plan.skill_sequence


def test_parse_observe_command():
    plan = parse_text_command("观察桌面")
    assert plan.task_type == "observe_scene"
    assert plan.skill_sequence == ["inspect_scene"]


@pytest.mark.parametrize(
    ("text", "task_type", "skill_name"),
    [
        ("原位", "recover_safe_pose", "recover_safe_pose"),
        ("观察点", "observe_scene", "inspect_scene"),
        ("零点", "recover_zero_pose", "recover_zero_pose"),
    ],
)
def test_parse_named_pose_keywords(text, task_type, skill_name):
    plan = parse_text_command(text)
    assert plan.task_type == task_type
    assert plan.skill_sequence == [skill_name]


@pytest.mark.parametrize(
    ("text", "direction"),
    [
        ("夹爪往前一点", "forward"),
        ("夹爪往后一点", "backward"),
        ("夹爪往左一点", "left"),
        ("夹爪往右一点", "right"),
        ("夹爪往上一点", "up"),
        ("夹爪往下一点", "down"),
    ],
)
def test_parse_relative_motion_command(text, direction):
    plan = parse_text_command(text, default_relative_motion_step_m=0.03)
    assert plan.task_type == "relative_motion"
    assert plan.motion_direction == direction
    assert plan.motion_distance == 0.03
    assert plan.skill_sequence == ["move_relative_ee"]


@pytest.mark.parametrize("text", ["夹爪向下", "夹爪朝下"])
def test_parse_down_orientation_command_does_not_emit_unsupported_skill(text):
    plan = parse_text_command(text)
    assert plan.task_type == "relative_motion"
    assert plan.motion_direction == "down"
    assert plan.skill_sequence == ["move_relative_ee"]


def test_parse_bare_down_orientation_command_is_not_unsupported_skill():
    plan = parse_text_command("朝下")
    assert plan.task_type == "unknown"
    assert "gripper_point_down" not in plan.skill_sequence


@pytest.mark.parametrize(
    ("text", "task_type", "skill_name"),
    [
        ("顺时针旋转45度", "rotate_gripper_cw", "rotate_gripper_cw"),
        ("逆时针旋转30度", "rotate_gripper_ccw", "rotate_gripper_ccw"),
    ],
)
def test_parse_rotate_gripper_command_uses_supported_skill(text, task_type, skill_name):
    plan = parse_text_command(text)
    assert plan.task_type == task_type
    assert plan.skill_sequence == [skill_name]


def test_parse_unknown_command():
    plan = parse_text_command("唱首歌")
    assert plan.task_type == "unknown"
    assert not plan.skill_sequence


def test_build_direct_planned_task_keeps_skill_sequence_in_context():
    plan = parse_text_command("回原位")
    task = build_direct_planned_task(
        task_id="task-demo",
        source="voice_asr",
        raw_command="回原位",
        plan=plan,
        timeout_sec=30.0,
    )
    assert task.task_id == "task-demo"
    assert task.task_type == "recover_safe_pose"
    assert task.target_name == ""
    context = load_task_context(task.context_json)
    assert context["skill_sequence"] == ["recover_safe_pose"]
    assert context["timeout_context"]["task_timeout_sec"] == 30.0
