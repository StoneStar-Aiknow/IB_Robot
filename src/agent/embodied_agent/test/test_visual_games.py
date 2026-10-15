"""Tests for the pure visual-game request builder."""

import json

import pytest

from embodied_agent.visual_games import build_game_request
from embodied_common.visual_game_contracts import SORTING_HAT_HOUSES, SORTING_HAT_UNDETERMINED


def test_build_game_request_fields():
    request = build_game_request("sorting_hat", request_id="game-test-1")

    assert request.request_id == "game-test-1"
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
    assert SORTING_HAT_UNDETERMINED in request.user_text
    assert "标准的场景分析 JSON" not in request.user_text  # the object-framing phrase is gone
    assert request.timeout_sec == 0.0

    context = json.loads(request.context_json)
    assert context["intent"] == "visual_game"
    assert context["game_name"] == "sorting_hat"
    assert context["handler"] == "sorting_hat_v1"
    assert context["entertainment_only"] is True
    assert context["required_inputs"] == ["primary_image"]
    contract = context["response_contract"]
    assert contract["field"] == "scene_summary"
    assert contract["kind"] == "enum"
    assert contract["allowed_values"] == [*SORTING_HAT_HOUSES, SORTING_HAT_UNDETERMINED]


def test_build_game_request_requires_non_empty_caller_id():
    with pytest.raises(ValueError, match="request_id must be non-empty"):
        build_game_request("sorting_hat", request_id=" ")
