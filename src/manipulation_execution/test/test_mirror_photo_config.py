"""The photo section and the trigger message: what is accepted, what is refused, and the defaults' promises.

``parse_photo_section`` is what the launch builder and the photo node both run
on ``mirror_session.photo``; a wrong value must fail at launch with the path of
the value in the message, never be ignored. ``normalize_target`` is the one
place an upper PC address is read, from yaml or from the target file.
"""

import json

import pytest

from embodied_common.mirror_photo_contract import TRIGGER_TOPIC, TriggerRequest, parse_trigger, trigger_json
from manipulation_execution.mirror_photo import config
from manipulation_execution.mirror_photo.config import (
    DEFAULT_PORT,
    PhotoSection,
    from_json,
    normalize_target,
    parse_photo_section,
    to_json,
)

ENABLED = {"enabled": True, "photos_dir": "/root/run/photos", "target_file": "/root/run/photo_target"}


def _with(**overrides):
    section = json.loads(json.dumps(ENABLED))
    for path, value in overrides.items():
        block = section
        keys = path.split("__")
        for key in keys[:-1]:
            block = block.setdefault(key, {})
        block[keys[-1]] = value
    return section


def test_a_left_out_section_is_photo_off():
    assert parse_photo_section(None) == PhotoSection()
    assert parse_photo_section(None).enabled is False
    assert parse_photo_section({}).enabled is False


def test_off_needs_no_paths_on_needs_absolute_ones():
    assert parse_photo_section({"enabled": False}).photos_dir == ""
    with pytest.raises(ValueError, match=r"mirror_session\.photo\.photos_dir: must be an absolute path"):
        parse_photo_section({"enabled": True, "target_file": "/a"})
    with pytest.raises(ValueError, match=r"mirror_session\.photo\.target_file: must be an absolute path"):
        parse_photo_section({"enabled": True, "photos_dir": "/a", "target_file": "photo_target"})


def test_the_target_file_is_not_the_photo_dir():
    with pytest.raises(ValueError, match="target_file: must not be photos_dir"):
        parse_photo_section({"enabled": True, "photos_dir": "/a/b", "target_file": "/a/b/"})


@pytest.mark.parametrize(
    "section, path",
    [
        (_with(enable=True), "mirror_session.photo: unknown keys ['enable']"),
        (_with(triggers__still__v_stil=0.2), "mirror_session.photo.triggers.still: unknown keys ['v_stil']"),
        (_with(triggers__smile={}), "mirror_session.photo.triggers: unknown keys ['smile']"),
        (_with(limits__photos_per_round=1), "mirror_session.photo.limits: unknown keys ['photos_per_round']"),
        (_with(upload__retry_max_sec=0), "mirror_session.photo.upload: unknown keys ['retry_max_sec']"),
        (_with(weights__face=1.0), "mirror_session.photo.weights: unknown keys ['face']"),
    ],
)
def test_a_misspelt_or_retired_key_fails_with_its_path(section, path):
    with pytest.raises(ValueError) as caught:
        parse_photo_section(section)
    assert str(caught.value).startswith(path)


@pytest.mark.parametrize(
    "section, message",
    [
        (_with(enabled="yes"), "mirror_session.photo.enabled: must be true or false"),
        (_with(warmup_sec=-1), "mirror_session.photo.warmup_sec: must be at least 0"),
        (_with(warmup_sec=True), "mirror_session.photo.warmup_sec: must be a number"),
        (_with(warmup_sec=float("nan")), "mirror_session.photo.warmup_sec: must be a number"),
        (
            _with(triggers__uniform__period_sec=0),
            "mirror_session.photo.triggers.uniform.period_sec: must be greater than 0",
        ),
        (
            _with(triggers__still__v_still=0.5, triggers__still__v_move=0.4),
            "mirror_session.photo.triggers.still.v_move: must be at least v_still",
        ),
        (
            _with(triggers__gesture__labels=["Victory"]),
            "mirror_session.photo.triggers.gesture.labels: must map label to weight",
        ),
        (
            _with(triggers__gesture__labels={"Victory": "high"}),
            "mirror_session.photo.triggers.gesture.labels.Victory: must be a number",
        ),
        (
            _with(triggers__event__labels={"": 1.0}),
            "mirror_session.photo.triggers.event.labels: labels must be non-empty strings",
        ),
        (
            _with(limits__max_candidates_per_round=0),
            "mirror_session.photo.limits.max_candidates_per_round: must be an integer of at least 1",
        ),
        (
            _with(limits__max_candidates_per_round=6, limits__reserve_for_priority=6),
            "mirror_session.photo.limits.reserve_for_priority: must be less than",
        ),
        (
            _with(limits__max_priority_per_minute=2.5),
            "mirror_session.photo.limits.max_priority_per_minute: must be an integer",
        ),
        (_with(upload__default_target="192.168.3.10/x2"), "mirror_session.photo.upload.default_target:"),
        (
            _with(upload__outbox_max_files=0),
            "mirror_session.photo.upload.outbox_max_files: must be an integer of at least 1",
        ),
        (_with(upload__timeout_sec=0), "mirror_session.photo.upload.timeout_sec: must be greater than 0"),
        (_with(scoring_deadline_sec=0), "mirror_session.photo.scoring_deadline_sec: must be greater than 0"),
        (_with(triggers=[1]), "mirror_session.photo.triggers: must be a mapping"),
    ],
)
def test_a_bad_value_fails_with_its_path(section, message):
    with pytest.raises(ValueError) as caught:
        parse_photo_section(section)
    assert str(caught.value).startswith(message)


def test_values_given_override_the_defaults_one_by_one():
    section = parse_photo_section(
        _with(
            triggers__still__v_still=0.3,
            triggers__gesture__labels={"Victory": 4.0},
            triggers__event__labels={"jump": 2.0},
            weights__sharpness=0.0,
            upload__default_target="192.168.3.10",
        )
    )
    assert section.still.v_still == 0.3
    assert section.still.v_move == config.StillRule.v_move
    assert dict(section.gesture.labels) == {"Victory": 4.0}
    assert dict(section.event.labels) == {"jump": 2.0}
    assert section.sharpness_weight == 0.0
    assert section.stillness_weight == PhotoSection.stillness_weight
    assert section.upload.default_target == "192.168.3.10"
    assert section.upload.outbox_max_files == 200


def test_the_default_gestures_are_the_three_detected():
    assert dict(PhotoSection().gesture.labels) == {"Victory": 3.0, "Thumb_Up": 3.0, "Pointing_Up": 1.5}


def test_with_the_defaults_a_victory_or_thumb_up_beats_any_held_pose():
    """zll: a gesture wins over a pose. The ranks can add at most their weights' sum."""
    section = PhotoSection()
    worst_gesture = min(section.gesture.labels["Victory"], section.gesture.labels["Thumb_Up"])
    best_pose = section.still.weight + section.stillness_weight + section.sharpness_weight
    assert worst_gesture > best_pose
    assert section.still.weight > section.uniform.weight + section.stillness_weight + section.sharpness_weight


def test_the_json_parameter_reads_back_the_same_section():
    section = parse_photo_section(_with(triggers__event__labels={"jump": 2.0}, limits__reserve_for_priority=3))
    assert from_json(to_json(section)) == section
    assert from_json(to_json(PhotoSection())) == PhotoSection()


def test_a_broken_json_parameter_names_the_section():
    with pytest.raises(ValueError, match=r"^mirror_session\.photo: not a JSON section"):
        from_json("{")


@pytest.mark.parametrize(
    "text, url",
    [
        ("192.168.3.10", f"http://192.168.3.10:{DEFAULT_PORT}"),
        ("  192.168.3.10:9000 ", "http://192.168.3.10:9000"),
        ("http://192.168.3.10:9000/", "http://192.168.3.10:9000"),
        ("upper-pc.local", f"http://upper-pc.local:{DEFAULT_PORT}"),
    ],
)
def test_an_address_becomes_a_base_url(text, url):
    assert normalize_target(text) == url


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "https://192.168.3.10",
        "192.168.3.10/x2",
        "user@192.168.3.10",
        "192.168.3.10:0",
        "192.168.3.10:99999",
        ":8765",
        "a b",
        "192.168.3.10:port",
    ],
)
def test_anything_else_is_not_an_address(text):
    with pytest.raises(ValueError):
        normalize_target(text)


def test_the_trigger_message_reads_back():
    text = trigger_json("jump", source="choreo", score=0.8, stamp=12.5)
    assert parse_trigger(text) == TriggerRequest("choreo", "jump", 0.8, 12.5)
    assert parse_trigger(trigger_json("jump", source="choreo")) == TriggerRequest("choreo", "jump", 1.0, None)
    assert TRIGGER_TOPIC == "/mirror_photo/trigger"


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        json.dumps({"label": "jump"}),
        json.dumps({"source": "a", "label": ""}),
        json.dumps({"source": "a", "label": "jump", "score": 1.5}),
        json.dumps({"source": "a", "label": "jump", "stamp": "now"}),
        json.dumps({"source": "a", "label": "jump", "extra": 1}),
    ],
)
def test_a_bad_trigger_message_is_refused(text):
    with pytest.raises(ValueError):
        parse_trigger(text)


def test_a_zero_stamp_means_the_newest_frame():
    assert parse_trigger(json.dumps({"source": "a", "label": "jump", "stamp": 0})).stamp is None
