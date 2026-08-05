import pytest

from embodied_agent.command_parser import parse_text_command, parse_text_workflow
from embodied_common.command_parser import extract_skill_aliases, load_skill_aliases


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
        "把香蕉往上移动一点",
        "把香蕉顺时针旋转30度",
        "把目标物往左挪一点",
        "抓住香蕉后顺时针旋转30度",
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


def test_social_gestures_require_ssot_alias_injection():
    """All seven social gestures are unreachable without SSOT alias injection.

    This enforces single-source-of-truth: the Chinese trigger keywords live
    only in ``description.aliases_zh`` (SSOT YAML) and are injected by the
    launch wiring. Without injection, gesture phrases must NOT silently match
    a parallel hardcoded keyword block.
    """
    for text in ["开心转圈", "撒娇", "卖萌", "挥手", "点头", "庆祝一下"]:
        plan = parse_text_command(text)
        assert not plan.skill_sequence, f"{text!r} should not match without SSOT alias injection"


def test_injected_skill_aliases_route_all_social_gestures():
    """SSOT-injected aliases (from description.aliases_zh) drive gesture routing.

    All seven social gestures are reachable only when the SSOT alias map is
    injected -- this mirrors how the launch wiring feeds ``skill_aliases``
    into the rule parser.
    """
    skill_aliases = {
        "wave_hello": ["打招呼", "挥手", "再见"],
        "nod_yes": ["点头", "同意"],
        "shake_no": ["摇头", "不是"],
        "celebrate": ["庆祝", "太棒了"],
        "greet_observe_raise": ["举手致意"],
        "act_cute": ["撒娇", "卖萌"],
        "happy_spin_upright": ["开心转圈"],
    }
    cases = {
        "和我打招呼": "wave_hello",
        "再见啦": "wave_hello",
        "点点头": "nod_yes",
        "摇摇头": "shake_no",
        "庆祝一下": "celebrate",
        "举手致意": "greet_observe_raise",
        "撒娇一下": "act_cute",
        "开心转圈": "happy_spin_upright",
    }
    for text, expected_skill in cases.items():
        plan = parse_text_command(text, skill_aliases=skill_aliases)
        assert plan.task_type == expected_skill, f"{text!r} -> {plan.task_type} != {expected_skill}"
        assert plan.skill_sequence == [expected_skill]


def test_injected_aliases_take_priority_over_hardcoded_keywords():
    """When injected, the SSOT alias is authoritative even for legacy gestures."""
    skill_aliases = {"happy_spin_upright": ["转圈圈"]}
    plan = parse_text_command("转圈圈", skill_aliases=skill_aliases)
    assert plan.skill_sequence == ["happy_spin_upright"]


@pytest.mark.parametrize(
    ("text", "expected_skill"),
    [
        ("正式打招呼", "greet_observe_raise"),
        ("举手打招呼", "greet_observe_raise"),
    ],
)
def test_injected_aliases_prefer_longer_specific_keywords(text, expected_skill):
    skill_aliases = {
        "wave_hello": ["打招呼", "挥手"],
        "greet_observe_raise": ["举手致意", "正式打招呼", "举手打招呼"],
    }
    plan = parse_text_command(text, skill_aliases=skill_aliases)
    assert plan.task_type == expected_skill
    assert plan.skill_sequence == [expected_skill]


def test_parameter_commands_are_not_shadowed_by_injected_aliases():
    skill_aliases = {
        "move_relative_ee": ["往前一点", "往上一点", "夹爪往左", "移动一点"],
        "rotate_gripper_cw": ["顺时针旋转", "顺时针转"],
        "rotate_gripper_ccw": ["逆时针旋转", "逆时针转"],
    }

    forward = parse_text_command("夹爪往前一点", default_relative_motion_step_m=0.03, skill_aliases=skill_aliases)
    assert forward.task_type == "relative_motion"
    assert forward.motion_direction == "forward"
    assert forward.motion_distance == 0.03
    assert forward.skill_sequence == ["move_relative_ee"]

    cw = parse_text_command("顺时针旋转30度", skill_aliases=skill_aliases)
    assert cw.task_type == "rotate_gripper_cw"
    assert cw.motion_distance == 30.0
    assert cw.skill_sequence == ["rotate_gripper_cw"]

    ccw = parse_text_command("逆时针旋转45度", skill_aliases=skill_aliases)
    assert ccw.task_type == "rotate_gripper_ccw"
    assert ccw.motion_distance == 45.0
    assert ccw.skill_sequence == ["rotate_gripper_ccw"]


def test_pick_and_place_commands_cannot_be_reenabled_by_aliases():
    skill_aliases = {
        "wave_hello": ["挥手"],
        "celebrate": ["庆祝一下"],
    }

    for text in ("抓香蕉然后挥手", "把目标物放下后庆祝一下"):
        plan = parse_text_command(text, skill_aliases=skill_aliases)
        assert plan.task_type == "unknown"
        assert plan.skill_sequence == []


def test_extract_skill_aliases_reads_description_aliases_zh():
    templates = {
        "wave_hello": {
            "description": {
                "rule_entry": True,
                "aliases_zh": ["打招呼", "挥手"],
                "requires_motion_params": False,
            },
            "primitive_sequence": [],
        },
        "move_relative_ee": {
            "description": {
                "aliases_zh": ["往前一点", "移动一点"],
                "requires_motion_params": True,
            },
            "primitive_sequence": [],
        },
        "inspect_scene": {"description": {"aliases_en": ["observe"]}, "primitive_sequence": []},
        "bare_skill": {"primitive_sequence": []},
    }
    aliases = extract_skill_aliases(templates)
    assert aliases == {"wave_hello": ["打招呼", "挥手"]}


def test_extract_skill_aliases_requires_rule_entry_opt_in():
    templates = {
        "wave_hello": {
            "description": {
                "rule_entry": True,
                "aliases_zh": ["挥手"],
                "requires_motion_params": False,
            }
        },
        "inspect_scene": {
            "description": {
                "aliases_zh": ["观察桌面"],
                "requires_motion_params": False,
            }
        },
        "move_relative_ee": {
            "description": {
                "rule_entry": True,
                "aliases_zh": ["往前一点"],
                "requires_motion_params": True,
            }
        },
    }

    assert extract_skill_aliases(templates) == {"wave_hello": ["挥手"]}


def test_extract_skill_aliases_skips_disabled_skills():
    templates = {
        "wave_hello": {
            "description": {
                "rule_entry": True,
                "aliases_zh": ["挥手"],
                "requires_motion_params": False,
            }
        },
        "disabled_nod": {
            "disabled": True,
            "description": {
                "rule_entry": True,
                "aliases_zh": ["点头"],
                "requires_motion_params": False,
            },
        },
    }

    assert extract_skill_aliases(templates) == {"wave_hello": ["挥手"]}


@pytest.mark.parametrize(
    ("text", "task_type", "skill_sequence"),
    [
        ("观察桌面", "observe_scene", ["inspect_scene"]),
        ("打开夹爪", "open_gripper", ["open_gripper_skill"]),
        ("回原位", "recover_safe_pose", ["recover_safe_pose"]),
    ],
)
def test_injected_aliases_do_not_change_legacy_task_types(text, task_type, skill_sequence):
    aliases = {
        "inspect_scene": ["观察桌面"],
        "open_gripper_skill": ["打开夹爪"],
        "recover_safe_pose": ["回原位"],
    }

    plan = parse_text_command(text, skill_aliases=aliases)

    assert plan.task_type == task_type
    assert plan.skill_sequence == skill_sequence


def test_load_skill_aliases_parses_valid_json():
    raw = '{"wave_hello": ["打招呼", "挥手"], "nod_yes": ["点头"]}'
    assert load_skill_aliases(raw) == {"wave_hello": ["打招呼", "挥手"], "nod_yes": ["点头"]}


def test_load_skill_aliases_handles_empty_and_invalid():
    assert load_skill_aliases("") == {}
    assert load_skill_aliases("   ") == {}
    assert load_skill_aliases("{invalid") == {}
    assert load_skill_aliases("[1, 2]") == {}
    assert load_skill_aliases('{"wave": "not_a_list"}') == {}


@pytest.mark.parametrize(
    "text",
    [
        "播放庆祝动作",
        "播放挥手动作",
        "放松地挥挥手",
    ],
)
def test_play_and_relax_phrases_are_not_blocked_by_place_check(text):
    """播放/放松等含'放'字的合法命令不应被放置保护拦截。"""
    skill_aliases = {
        "wave_hello": ["挥手"],
        "celebrate": ["庆祝"],
    }
    plan = parse_text_command(text, skill_aliases=skill_aliases)
    assert plan.task_type != "unknown"


def test_overlapping_aliases_resolve_by_position():
    """'不是的'同时匹配 nod_yes 的'是的'和 shake_no 的'不是'，
    应按文本位置优先解析为 shake_no 而非依赖 YAML 顺序。"""
    skill_aliases = {
        "nod_yes": ["是的"],
        "shake_no": ["不是"],
    }
    plan = parse_text_command("不是的", skill_aliases=skill_aliases)
    assert plan.task_type == "shake_no"


@pytest.mark.parametrize(
    "text",
    [
        "先点头，然后挥手",
        "点头，接着挥手",
        "first nod yes, then wave hello",
        "execute nod_yes then execute wave_hello",
        "nod_yes followed by wave_hello",
    ],
)
def test_parse_explicit_ordered_multi_skill_workflow(text):
    steps, error = parse_text_workflow(
        text,
        skill_aliases={
            "nod_yes": ["点头", "nod", "yes"],
            "wave_hello": ["挥手", "wave", "hello"],
        },
    )

    assert error == ""
    assert [step.skill_sequence[0] for step in steps] == ["nod_yes", "wave_hello"]


def test_multiple_skills_without_ordered_connector_are_rejected():
    steps, error = parse_text_workflow(
        "点头和挥手",
        skill_aliases={"nod_yes": ["点头"], "wave_hello": ["挥手"]},
    )

    assert steps == []
    assert error == "multiple skills require an explicit ordered connector"


def test_overlapping_short_english_alias_does_not_create_false_workflow_ambiguity():
    steps, error = parse_text_workflow(
        "nod",
        skill_aliases={"nod_yes": ["nod"], "shake_no": ["no"]},
    )

    assert error == ""
    assert [step.skill_sequence[0] for step in steps] == ["nod_yes"]


def test_unknown_workflow_step_rejects_the_entire_workflow():
    steps, error = parse_text_workflow(
        "先点头，然后唱首歌，再挥手",
        skill_aliases={"nod_yes": ["点头"], "wave_hello": ["挥手"]},
    )

    assert steps == []
    assert error == "unsupported command: 唱首歌"


def test_goodbye_alias_is_not_mistaken_for_the_chinese_then_connector():
    steps, error = parse_text_workflow("再见啦", skill_aliases={"wave_hello": ["再见"]})

    assert error == ""
    assert [step.skill_sequence[0] for step in steps] == ["wave_hello"]
