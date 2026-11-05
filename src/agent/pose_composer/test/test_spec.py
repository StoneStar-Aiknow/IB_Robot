"""The language model's description: parsing, refusal and conversion to safe joint frames."""

import json

import pytest

from pose_composer import kin, safety, spec
from pose_composer.model import load


def _compose(payload: dict, start=None):
    return spec.compose(spec.parse(json.dumps(payload)), start or dict(load().rest))


def test_directions_become_a_safe_frame_that_points_where_asked():
    out = _compose({"kind": "pose", "left": {"upper": [1, 0, 0], "fore": [1, 0, 0]}, "right": {"keep": True},
                    "user_message": "好"})  # fmt: skip
    pose = kin.fk(out.frames[0])
    assert kin.angle_deg(pose.fore("left"), [1, 0, 0]) < 10
    assert safety.check_pose(out.frames[0]).ok
    rest = load().rest
    assert all(abs(out.frames[0][n] - rest[n]) < 1e-6 for n in out.frames[0] if n.startswith("right_"))


def test_a_body_point_puts_the_hand_beside_it():
    out = _compose({"kind": "pose", "right": {"hand_to": "waist_right", "elbow": "out"}, "left": {"keep": True}})
    pose = kin.fk(out.frames[0])
    assert ((pose.points["right_hand"] - pose.points["waist_right"]) ** 2).sum() ** 0.5 < 0.16


def test_a_sequence_keeps_its_order():
    out = _compose({"kind": "sequence", "frames": [
        {"left": {"upper": [0, 0, 1], "fore": [0, 0, 1]}, "right": {"upper": [0, 0, 1], "fore": [0, 0, 1]}},
        {"left": {"upper": [0, 0, -1], "fore": [0.5, 0, -0.87]}, "right": {"upper": [0, 0, -1], "fore": [0.5, 0, -0.87]}},
    ]})  # fmt: skip
    assert len(out.frames) == 2
    assert kin.fk(out.frames[0]).upper("left")[2] > 0.8
    assert kin.fk(out.frames[1]).upper("left")[2] < -0.8


@pytest.mark.parametrize("kind", ["rejected", "no_motion"])
def test_refusals_and_answers_move_nothing(kind):
    out = _compose({"kind": kind, "reason": "x", "user_message": "不行"})
    assert out.kind == kind and out.frames == []


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"kind": "pose", "left": {"hand_to": "kneecap"}}, "unknown body point"),
        ({"kind": "pose", "left": {"upper": [0, 0, 0], "fore": [1, 0, 0]}}, "non-zero"),
        ({"kind": "pose", "left": {"hand_to": "chest", "offset_cm": [80, 0, 0]}}, "30 cm"),
        ({"kind": "pose", "left": {"bend": 1}}, "upper\\+fore"),
    ],
)
def test_a_description_that_cannot_be_executed_is_refused(payload, reason):
    with pytest.raises(spec.SpecError, match=reason):
        _compose(payload)


def test_a_reply_that_is_not_a_single_json_object_is_refused():
    with pytest.raises(spec.SpecError):
        spec.parse("好的，我会举手")
    with pytest.raises(spec.SpecError):
        spec.parse('{"kind": "dance"}')


def test_the_prompt_carries_the_current_pose_and_history():
    messages = spec.build_messages("再高一点", dict(load().rest), [("双手向前伸直", dict(load().rest))])
    assert "当前姿态" in messages[1]["content"] and "双手向前伸直" in messages[1]["content"]
    assert "hand_to" in messages[0]["content"]
