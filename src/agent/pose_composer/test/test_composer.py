"""Prepared compositions: composed once with the plan, executed once, expiring."""

import json

import pytest

from pose_composer.composer import ComposeError, Composer
from pose_composer.model import load

FORWARD = {"kind": "pose", "left": {"upper": [1, 0, 0], "fore": [1, 0, 0]}, "right": {"keep": True},
           "user_message": "好的"}  # fmt: skip


def _composer(reply, **kw):
    calls = []

    def model(messages):
        calls.append(messages)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)

    return Composer(model, **kw), calls


def test_a_composition_is_kept_for_its_text_and_taken_once():
    composer, _ = _composer(FORWARD)
    prepared = composer.compose("左手向前伸直", dict(load().rest))
    assert composer.take("左手向前伸直").composition_id == prepared.composition_id
    assert composer.take("左手向前伸直") is None


def test_a_composition_can_be_taken_by_id():
    composer, _ = _composer(FORWARD)
    prepared = composer.compose("左手向前伸直", dict(load().rest))
    assert composer.take("other text", prepared.composition_id) is prepared


def test_an_expired_composition_is_not_executed():
    composer, _ = _composer(FORWARD, ttl_sec=0.0)
    composer.compose("左手向前伸直", dict(load().rest))
    assert composer.take("左手向前伸直") is None


def test_a_refusal_is_reported_and_not_kept():
    composer, _ = _composer({"kind": "rejected", "reason": "会打到自己", "user_message": "不做"})
    prepared = composer.compose("打自己的头", dict(load().rest))
    assert prepared.composition.kind == "rejected"
    assert composer.take("打自己的头") is None


@pytest.mark.parametrize(
    ("reply", "code"),
    [
        (RuntimeError("timeout"), "POSE_MODEL_FAILED"),
        ("好的我举手", "POSE_NOT_EXECUTABLE"),
        ({"kind": "pose", "left": {"hand_to": "kneecap"}}, "POSE_NOT_EXECUTABLE"),
    ],
)
def test_failures_carry_their_code(reply, code):
    composer, _ = _composer(reply)
    with pytest.raises(ComposeError) as raised:
        composer.compose("摆个姿势", dict(load().rest))
    assert raised.value.code == code


def test_an_empty_or_overlong_description_never_reaches_the_model():
    composer, calls = _composer(FORWARD)
    for text in ("  ", "举" * 201):
        with pytest.raises(ComposeError):
            composer.compose(text, dict(load().rest))
    assert calls == []


def test_a_reached_pose_is_context_for_the_next_instruction():
    composer, calls = _composer(FORWARD)
    prepared = composer.compose("双手向前伸直", dict(load().rest))
    composer.succeeded(prepared)
    composer.compose("再抬高一点", dict(load().rest))
    assert "双手向前伸直" in calls[-1][1]["content"]
