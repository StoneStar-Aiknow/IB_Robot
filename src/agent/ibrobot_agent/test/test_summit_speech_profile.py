"""The exhibition's spoken grasp lines pass the bridge's own start-up checks and promise no delivery."""

from __future__ import annotations

from pathlib import Path

import pytest
from ament_index_python.packages import get_package_share_directory

from embodied_common.control_phrases import STOP_PHRASES
from embodied_common.execution_detail import ExecutionDetail
from ibrobot_agent.speech_policy import GraspSpeechPolicy
from robot_config.loader import _load_robot_section, _validate_agent_voice_bridge

CONFIG = Path(get_package_share_directory("robot_config")) / "config" / "robots" / "aimdk_x2_summit.yaml"
RETRIED = "哎呀，这次没成功，我再试一下。"


@pytest.fixture(scope="module")
def bridge():
    return _load_robot_section(CONFIG)[1]["embodied"]["agent"]["voice_bridge"]


def _payload(state, detail):
    return {
        "state": state,
        "request_key": {"request_id": "r1"},
        "detail": {"current_skill": "pick_object", "execution_id": "e1", "execution_detail": detail},
    }


def _detail(phase, phase_state="RUNNING", **fields):
    fields.setdefault("recovery_state", "NOT_NEEDED")
    return ExecutionDetail(schema_version=1, phase=phase, phase_state=phase_state, **fields).to_dict()


def test_the_bridge_accepts_the_profile(bridge):
    assert _validate_agent_voice_bridge(bridge) == []
    assert bridge["speech_mode"] == "concise"


def test_the_default_stop_phrases_are_kept(bridge):
    assert set(STOP_PHRASES) <= set(bridge["stop_phrases"])
    assert {"不玩了", "不拿了"} <= set(bridge["stop_phrases"])


def test_no_line_uses_the_english_target_or_promises_a_box(bridge):
    for key, text in bridge["speech_messages"].items():
        assert "{target}" not in text, key
        assert "领取盒" not in text and "送到" not in text, key


def test_a_retry_and_the_result_speak_the_configured_lines(bridge):
    policy = GraspSpeechPolicy(bridge["speech_mode"], bridge["speech_messages"])
    assert policy.start("r1", "这个东西", 0.0).text == "好，我这就去拿，请稍等。"
    assert policy.observe(_payload("EXECUTING", _detail("perceive")), 1.0) == []
    retried = policy.observe(_payload("EXECUTING", _detail("perceive", retry_count=1, reason_code="EMPTY_GRASP")), 2.0)
    assert [notice.text for notice in retried] == [RETRIED]
    ended = policy.observe(_payload("SUCCEEDED", _detail("verify", "SUCCEEDED")), 3.0)
    assert ended[0].text == bridge["speech_messages"]["success"]


def test_recovery_is_silent_and_a_retry_is_announced_once_through_its_attempt(bridge):
    policy = GraspSpeechPolicy(bridge["speech_mode"], bridge["speech_messages"])
    policy.observe(_payload("EXECUTING", _detail("close")), 1.0)
    recovering = _detail("recover", recovery_state="IN_PROGRESS", reason_code="EMPTY_GRASP")
    assert policy.observe(_payload("EXECUTING", recovering), 2.0) == []
    recovered = _detail("recover", recovery_state="COMPLETED", reason_code="EMPTY_GRASP")
    assert policy.observe(_payload("EXECUTING", recovered), 3.0) == []

    (retried,) = policy.observe(_payload("EXECUTING", _detail("observe", retry_count=1)), 4.0)

    assert retried.text == RETRIED
    # Synthesis takes about 2 s; the attempt has moved on by the time the line can play.
    policy.observe(_payload("EXECUTING", _detail("perceive", retry_count=1)), 5.0)
    assert policy.is_current(retried, 6.5)
    assert not policy.is_current(retried, 30.0)


def test_a_retry_line_dies_with_the_pick(bridge):
    policy = GraspSpeechPolicy(bridge["speech_mode"], bridge["speech_messages"])
    policy.observe(_payload("EXECUTING", _detail("close")), 1.0)
    (retried,) = policy.observe(_payload("EXECUTING", _detail("observe", retry_count=1)), 2.0)
    policy.observe(_payload("FAILED", _detail("observe", "FAILED", retry_count=1, reason_code="EMPTY_GRASP")), 3.0)
    assert not policy.is_current(retried, 3.5)


def test_results_cannot_be_switched_off():
    with pytest.raises(ValueError):
        GraspSpeechPolicy("concise", {"failed": ""})


def test_a_pick_that_ends_after_the_put_back_speaks_the_same_result(bridge):
    assert bridge["speech_messages"]["completed"] == bridge["speech_messages"]["success"]
