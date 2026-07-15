"""Tests for the entry-layer visual game helpers.

These cover the pure request-construction and trigger-matching logic used by
``task_entry_node``; they do not spin a ROS node.
"""

import json

import pytest

from embodied_agent.visual_games import (
    HOUSES,
    build_game_request,
    fix_stt_errors,
    match_game,
)

_SORTING_HAT_POLICY = {
    "sorting_hat": {
        "enabled": True,
        "trigger_aliases": ["分院帽", "奔月帽", "风月帽", "分月帽"],
    }
}


@pytest.mark.parametrize(
    "text",
    ["分院帽", "帮我分院帽一下", "奔月帽", "风月帽", "分月帽", "奔 月 帽"],
)
def test_trigger_and_stt_aliases_match_sorting_hat(text):
    assert match_game(text, _SORTING_HAT_POLICY) == "sorting_hat"


@pytest.mark.parametrize("text", ["向前移动", "观察桌面", "抓香蕉", ""])
def test_non_game_commands_do_not_match(text):
    assert match_game(text, _SORTING_HAT_POLICY) is None


def test_disabled_game_does_not_match():
    policy = {"sorting_hat": {"enabled": False, "trigger_aliases": ["分院帽"]}}
    assert match_game("分院帽", policy) is None


def test_unknown_game_name_in_policy_is_ignored():
    policy = {"palmistry": {"enabled": True, "trigger_aliases": ["看手相"]}}
    assert match_game("看手相", policy) is None


def test_fix_stt_errors_normalizes_spaced_and_unspaced():
    assert fix_stt_errors("奔 月 帽") == "分院帽"
    assert fix_stt_errors("风月帽") == "分院帽"


def test_build_game_request_fields():
    request = build_game_request("sorting_hat")

    assert request.request_id.startswith("game-")
    assert request.source == "game.sorting_hat"
    assert request.session_id.startswith("sorting-hat-")
    assert request.session_id.endswith(request.request_id)
    assert "分院帽" in request.user_text
    # Person-centric guard: the prompt must steer the model at the person and
    # away from the generic robot object framing, and still pin the house into
    # scene_summary. Protects against drifting back to object-centric wording.
    assert "忽略" in request.user_text  # ignore desk objects
    assert "visible_objects" in request.user_text  # person cues go here
    assert "scene_summary" in request.user_text
    assert "标准的场景分析 JSON" not in request.user_text  # the object-framing phrase is gone
    assert request.timeout_sec == 0.0

    context = json.loads(request.context_json)
    assert context["intent"] == "visual_game"
    assert context["game_name"] == "sorting_hat"
    assert context["entertainment_only"] is True
    assert context["required_inputs"] == ["primary_image"]
    contract = context["response_contract"]
    assert contract["field"] == "scene_summary"
    assert contract["kind"] == "enum"
    assert contract["allowed_values"] == HOUSES
