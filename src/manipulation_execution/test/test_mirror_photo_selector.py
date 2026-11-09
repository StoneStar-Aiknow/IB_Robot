"""One MIRRORING round through ``PhotoCore``: when it starts and ends, what it keeps, what it sends.

The first priority of the feature is that every round with a frame gives a
photo, so the short, empty and interrupted rounds are tested as carefully as
the normal one. Times are camera time; frames arrive at 15 fps; every frame is
the same small synthetic JPEG, so a candidate is told apart by its stamp.
"""

import hashlib
import json

import cv2
import numpy as np
import pytest

from embodied_common.mirror_photo_contract import TriggerRequest
from embodied_common.mirror_session_contract import STATE_MIRRORING, STATE_STALE_SEC, STATE_STANDBY, STATE_WAIT
from manipulation_execution.mirror_photo.config import EventRule, Limits, PhotoSection
from manipulation_execution.mirror_photo.selector import (
    BUFFER_SEC,
    SCHEMA,
    PhotoCore,
    choose_photo,
    emergency_photo,
    provisional_photo,
)
from manipulation_execution.mirror_photo.triggers import EVENT, GESTURE, STILL, UNIFORM, Trigger

FPS = 15
WALL = 1762651234.0
SECTION = PhotoSection(enabled=True, photos_dir="/photos", target_file="/target")


def _encode(width: int = 64, height: int = 48) -> bytes:
    image = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", image)[1].tobytes()


FRAME = _encode()
STILL_POSE = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), 52)


class Round:
    def __init__(self, section: PhotoSection = SECTION):
        self.finished = []
        self.provisional = []
        self.section = section
        self.core = PhotoCore(
            section,
            on_round_end=self.finished.append,
            on_provisional=lambda *args: self.provisional.append(args),
            wall=lambda: WALL,
            round_suffix=lambda: "abcd",
        )

    def state(self, state: str, now: float = 0.0) -> None:
        self.core.on_state(json.dumps({"state": state, "phase": "x"}), now)

    def frames(self, start: float, end: float) -> None:
        """Frames from ``start`` up to but not including ``end``."""
        for index in range(round((end - start) * FPS)):
            self.core.on_image(start + index / FPS, FRAME)

    def stop(self) -> None:
        self.state(STATE_STANDBY)
        assert len(self.finished) == 1
        return self.finished[0]

    @property
    def kept(self) -> list:
        return self.core._round.candidates

    def kept_stamps(self, source: str) -> list[float]:
        return [candidate.stamp for candidate in self.kept if candidate.trigger.source == source]


def started(section: PhotoSection = SECTION, *, warm: bool = True) -> Round:
    """A round in MIRRORING, past its warm-up if ``warm``, with the rhythm off unless asked."""
    one = Round(section)
    one.state(STATE_MIRRORING)
    if warm:
        one.frames(0.0, 2.0)
    return one


def quiet(**overrides) -> PhotoSection:
    """No steady rhythm: only the triggers a test fires itself become candidates."""
    from dataclasses import replace

    from manipulation_execution.mirror_photo.config import UniformRule

    return replace(SECTION, uniform=UniformRule(enabled=False), **overrides)


# -- when a round starts and ends ------------------------------------------------------


def test_nothing_is_kept_outside_mirroring():
    one = Round()
    one.state(STATE_STANDBY)
    one.frames(0.0, 5.0)
    one.core.on_event(TriggerRequest("a", "b"))
    assert not one.core.mirroring
    assert one.finished == []


def test_a_round_is_one_stretch_of_mirroring_however_many_heartbeats():
    one = Round()
    for second in range(3):
        one.state(STATE_MIRRORING, second)
    assert one.core.mirroring
    one.frames(0.0, 1.0)
    one.state(STATE_STANDBY, 3.0)
    one.state(STATE_WAIT, 4.0)
    assert not one.core.mirroring
    assert len(one.finished) == 1
    assert one.finished[0].reason == "state"
    assert one.finished[0].frames == FPS
    assert one.finished[0].round_id.endswith("-abcd")


def test_the_next_round_starts_clean():
    one = started()
    one.core.on_event(TriggerRequest("a", "b"))
    one.state(STATE_WAIT)
    one.state(STATE_MIRRORING)
    assert one.kept == []
    assert one.core._round.frame_count == 0


@pytest.mark.parametrize("text", ["not json", "[]", "{}", '{"state": 3}'])
def test_a_message_that_is_not_a_state_changes_nothing(text):
    one = started()
    one.core.on_state(text, 0.5)
    assert one.core.mirroring


def test_a_lost_heartbeat_ends_the_round_as_stale():
    one = Round()
    one.state(STATE_MIRRORING, 10.0)
    one.frames(0.0, 1.0)
    one.core.tick(10.0 + STATE_STALE_SEC)
    assert one.core.mirroring
    one.core.tick(10.0 + STATE_STALE_SEC + 0.01)
    assert not one.core.mirroring
    assert one.finished[0].reason == "stale"


def test_the_session_coming_back_after_stale_is_a_new_round():
    one = Round()
    one.state(STATE_MIRRORING, 10.0)
    one.frames(0.0, 1.0)
    one.core.tick(20.0)
    one.state(STATE_MIRRORING, 21.0)
    assert one.core.mirroring
    assert len(one.finished) == 1


def test_the_node_stopping_ends_the_round():
    one = started()
    one.core.shutdown()
    assert one.finished[0].reason == "shutdown"
    one.core.shutdown()
    assert len(one.finished) == 1


def test_the_tick_alone_does_nothing_outside_a_round():
    one = Round()
    one.core.tick(1000.0)
    assert one.finished == []


# -- every round with a frame gives a photo ----------------------------------------------


def test_a_round_too_short_for_any_candidate_sends_its_last_frame():
    one = Round()
    one.state(STATE_MIRRORING)
    one.frames(0.0, 0.5)
    finished = one.stop()
    assert finished.candidates == ()
    photo, choice = choose_photo(finished, SECTION)
    assert choice is None
    assert photo.jpeg == FRAME
    assert photo.meta["quality"] == "fallback"
    assert photo.meta["source"] is None
    assert photo.meta["stamp"] == finished.last_frame[0]


def test_a_round_with_every_trigger_off_still_sends():
    from dataclasses import replace

    from manipulation_execution.mirror_photo.config import GestureRule, StillRule, UniformRule

    section = replace(
        SECTION,
        uniform=UniformRule(enabled=False),
        still=StillRule(enabled=False),
        gesture=GestureRule(enabled=False),
        event=EventRule(enabled=False),
    )
    one = started(section)
    one.frames(2.0, 4.0)
    photo, _ = choose_photo(one.stop(), section)
    assert photo.meta["quality"] == "fallback"


def test_a_round_without_a_single_frame_sends_nothing_and_says_so():
    one = Round()
    one.state(STATE_MIRRORING)
    finished = one.stop()
    assert finished.frames == 0
    assert choose_photo(finished, SECTION) == (None, None)


def test_a_normal_round_sends_a_normal_photo():
    one = started()
    one.frames(2.0, 8.0)
    photo, choice = choose_photo(one.stop(), SECTION)
    assert choice is not None
    assert photo.meta["quality"] == "normal"
    assert photo.meta["source"] == UNIFORM


def test_a_round_ended_by_the_node_stopping_still_chooses():
    one = started()
    one.core.shutdown()
    photo, _ = choose_photo(one.finished[0], SECTION)
    assert photo.meta["end_reason"] == "shutdown"


def test_when_choosing_fails_the_heaviest_candidate_is_sent_unscored():
    one = started()
    one.frames(2.0, 8.0)
    round_now = one.core._round
    stamps = [frame[0] for frame in round_now.frames]
    one.core._offer(round_now, _gesture(stamps[-10]))
    one.core._offer(round_now, Trigger(STILL, STILL, 1.5, 1.0, stamps[-1]))
    photo = emergency_photo(one.stop(), "RuntimeError: boom")
    assert (photo.meta["source"], photo.meta["stamp"]) == (GESTURE, stamps[-10])
    assert (photo.meta["quality"], photo.meta["scoring"], photo.meta["error"]) == (
        "fallback",
        "error",
        "RuntimeError: boom",
    )


def test_when_choosing_fails_without_candidates_the_last_frame_is_sent():
    one = Round()
    one.state(STATE_MIRRORING)
    one.frames(0.0, 0.5)
    finished = one.stop()
    assert emergency_photo(finished, "x").meta["stamp"] == finished.last_frame[0]
    empty = Round()
    empty.state(STATE_MIRRORING)
    assert emergency_photo(empty.stop(), "x") is None


# -- warm-up and the rhythm --------------------------------------------------------------


def test_the_warm_up_has_no_candidates_then_the_rhythm_starts():
    one = Round()
    one.state(STATE_MIRRORING)
    one.frames(0.0, 1.4)
    assert one.kept == []
    one.frames(1.4, 1.6)
    assert one.kept_stamps(UNIFORM) == [pytest.approx(1.5, abs=1 / FPS)]


def test_the_warm_up_counts_from_the_first_frame_not_from_camera_zero():
    one = Round()
    one.state(STATE_MIRRORING)
    one.frames(1000.0, 1001.0)
    assert one.kept == []


def test_triggers_in_the_warm_up_are_not_candidates():
    one = started(warm=False)
    one.frames(0.0, 1.0)
    one.core.on_hand(0.9, [("Victory", 0.9)])
    one.core.on_event(TriggerRequest("a", "b"))
    assert one.kept == []
    assert one.core._round.refused["warmup"] == 1


def test_a_long_round_rolls_the_rhythm_and_keeps_room_for_the_triggers():
    one = started()
    one.frames(2.0, 182.0)  # 3 minutes at one candidate per 3 s
    limits = SECTION.limits
    stamps = one.kept_stamps(UNIFORM)
    assert len(stamps) == limits.max_candidates_per_round - limits.reserve_for_priority
    assert stamps == sorted(stamps)
    assert stamps[-1] > 178.0  # the newest are the ones kept


# -- triggers name a frame ---------------------------------------------------------------


def test_a_gesture_takes_the_frame_nearest_its_stamp():
    one = started(quiet())
    one.core.on_hand(1.6, [("Victory", 0.9)])
    one.core.on_hand(1.9, [("Victory", 0.9)])
    assert one.kept_stamps(GESTURE) == [pytest.approx(1.9, abs=1 / FPS)]
    assert one.kept[0].trigger.label == "Victory"


def test_a_trigger_ahead_of_the_camera_waits_for_its_frame():
    one = started(quiet())
    one.core.on_event(TriggerRequest("a", "b", stamp=2.5))
    assert one.kept == []
    one.frames(2.0, 2.6)
    assert one.kept_stamps(EVENT) == [pytest.approx(2.5, abs=1 / FPS)]


def test_a_trigger_between_two_frames_takes_the_nearer_once_both_are_in():
    one = started(quiet())
    # Both ahead of the camera (newest frame 1.933); 2.03 is nearer the frame before, 2.31 the one after.
    one.core.on_event(TriggerRequest("a", "b", stamp=2.03))
    one.core.on_event(TriggerRequest("a", "b", stamp=2.31))
    one.frames(2.0, 2.4)
    assert one.kept_stamps(EVENT) == [pytest.approx(2.0), pytest.approx(2.0 + 5 / FPS)]
    assert one.core._round.pending == []


def test_a_trigger_far_ahead_of_the_camera_is_dropped_an_event_takes_the_newest():
    one = started(quiet())
    newest = one.core._round.frames[-1][0]
    one.core._offer(one.core._round, Trigger(STILL, STILL, 1.5, 1.0, newest + 5.0))
    one.core.on_event(TriggerRequest("a", "b", stamp=newest + 5.0))
    assert one.core._round.pending == []
    assert one.core._round.refused["nomatch"] == 1
    assert one.kept_stamps(EVENT) == [newest]


def test_a_trigger_whose_frame_never_comes_is_given_up():
    one = started(quiet())
    one.core._offer(one.core._round, Trigger(STILL, STILL, 1.5, 1.0, 2.5))
    assert one.core._round.pending
    one.core.on_image(10.0, FRAME)
    assert not one.core._round.pending
    assert one.core._round.refused["nomatch"] == 1


def test_an_event_ahead_of_the_camera_falls_back_to_the_newest_frame():
    one = started(quiet())
    one.core.on_event(TriggerRequest("a", "b", stamp=2.5))
    one.core.on_image(10.0, FRAME)
    assert one.kept_stamps(EVENT) == [10.0]


def test_a_stamp_older_than_the_buffer_is_late_for_a_gesture_and_the_newest_for_an_event():
    one = started(quiet())
    one.frames(2.0, 2.0 + BUFFER_SEC + 2.0)
    newest = one.core._round.frames[-1][0]
    one.core._offer(one.core._round, Trigger(STILL, STILL, 1.5, 1.0, 0.5))
    assert one.kept == []
    assert one.core._round.refused["late"] == 1
    one.core.on_event(TriggerRequest("a", "b", stamp=0.5))
    assert one.kept_stamps(EVENT) == [newest]


def test_an_event_without_a_stamp_takes_the_newest_frame():
    one = started(quiet())
    newest = one.core._round.frames[-1][0]
    one.core.on_event(TriggerRequest("a", "b"))
    assert one.kept_stamps(EVENT) == [newest]
    assert one.kept[0].trigger.origin == "a"


def test_the_still_rule_runs_from_pear_frames():
    one = started(quiet())
    for index in range(20):
        stamp = 1.5 + index / 30
        one.core.on_pear(stamp, STILL_POSE, (10.0, 20.0, 30.0, 40.0))
    assert len(one.kept_stamps(STILL)) == 1


def test_a_malformed_pear_frame_is_ignored():
    one = started(quiet())
    one.core.on_pear(1.9, np.zeros(10), None)
    assert one.core.mirroring
    assert one.core._round.speeds == []


def test_the_best_person_box_near_the_frame_goes_with_the_photo():
    one = started()
    one.core.on_pear(1.9, STILL_POSE, (10.0, 20.0, 30.0, 40.0))
    one.core.on_pear(1.95, STILL_POSE, None)
    one.frames(2.0, 2.2)
    stamp = one.core._round.frames[-1][0]
    one.core.on_pear(stamp, STILL_POSE, (100.0, 200.0, 300.0, 400.0))
    one.core.on_event(TriggerRequest("a", "b"))
    photo, _ = choose_photo(one.stop(), SECTION)
    assert photo.meta["bbox"] == [100.0, 200.0, 300.0, 400.0]


def test_no_person_box_near_the_frame_is_sent_as_none():
    one = started()
    one.frames(2.0, 2.2)
    one.core.on_event(TriggerRequest("a", "b"))
    photo, _ = choose_photo(one.stop(), SECTION)
    assert photo.meta["bbox"] is None


# -- limits ------------------------------------------------------------------------------


def _gesture(stamp: float, weight: float = 3.0, label: str = "Victory") -> Trigger:
    return Trigger(GESTURE, label, weight, 0.9, stamp)


def _fill(one: Round, stamps, weight: float = 3.0) -> None:
    for stamp in stamps:
        one.core._offer(one.core._round, _gesture(stamp, weight))


def test_the_rate_limit_counts_triggers_per_minute_of_camera_time():
    from dataclasses import replace

    one = started(replace(quiet(), limits=Limits(max_priority_per_minute=3)))
    _fill(one, [1.6, 1.7, 1.8, 1.9])
    assert len(one.kept) == 3
    assert one.core._round.refused["rate"] == 1


def test_the_rate_limit_opens_again_after_the_minute():
    from dataclasses import replace

    one = started(replace(quiet(), limits=Limits(max_priority_per_minute=1)))
    _fill(one, [1.9])
    one.frames(65.0, 65.2)
    _fill(one, [65.1])
    assert len(one.kept) == 2


def test_the_rate_limit_leaves_the_rhythm_alone():
    from dataclasses import replace

    from manipulation_execution.mirror_photo.config import UniformRule

    section = replace(SECTION, uniform=UniformRule(period_sec=1.0), limits=Limits(max_priority_per_minute=1))
    one = started(section)
    one.frames(2.0, 10.0)
    assert len(one.kept_stamps(UNIFORM)) >= 8


def test_the_same_frame_twice_is_one_candidate_the_heavier_trigger_stays():
    one = started(quiet())
    one.core._offer(one.core._round, Trigger(STILL, STILL, 1.5, 1.0, 1.9))
    one.core._offer(one.core._round, _gesture(1.9))
    one.core._offer(one.core._round, Trigger(STILL, STILL, 1.5, 1.0, 1.9))
    assert [candidate.trigger.source for candidate in one.kept] == [GESTURE]
    assert one.core._round.refused["duplicate"] == 1


def test_a_trigger_finds_room_when_the_rhythm_has_filled_its_share():
    one = started()
    one.frames(2.0, 182.0)
    stamp = one.core._round.frames[-1][0]
    uniform_before = len(one.kept_stamps(UNIFORM))
    one.core._offer(one.core._round, _gesture(stamp))
    assert one.kept_stamps(GESTURE) == [stamp]
    assert len(one.kept_stamps(UNIFORM)) == uniform_before


def test_a_full_round_replaces_the_weakest_only_for_a_stronger_trigger():
    from dataclasses import replace

    section = replace(
        quiet(), limits=Limits(max_candidates_per_round=2, reserve_for_priority=0, max_priority_per_minute=100)
    )
    one = started(section)
    _fill(one, [1.7], weight=1.5)
    _fill(one, [1.8], weight=3.0)
    _fill(one, [1.9], weight=1.5)  # no stronger than the weakest: refused
    assert sorted(candidate.trigger.weight for candidate in one.kept) == [1.5, 3.0]
    assert one.core._round.refused["full"] == 1
    _fill(one, [1.85], weight=3.0)  # stronger: the 1.5 goes
    assert sorted(candidate.trigger.weight for candidate in one.kept) == [3.0, 3.0]


def test_a_cut_frame_is_not_a_candidate_but_the_round_goes_on():
    one = started(quiet())
    one.core.on_image(2.0, FRAME[: len(FRAME) // 2])
    one.core.on_event(TriggerRequest("a", "b"))
    assert one.kept == []
    assert one.core._round.refused["badjpeg"] == 1


def test_a_repeated_or_out_of_order_frame_is_dropped():
    one = started(quiet())
    count = one.core._round.frame_count
    newest = one.core._round.frames[-1][0]
    one.core.on_image(newest, FRAME)
    one.core.on_image(newest - 0.5, FRAME)
    assert one.core._round.frame_count == count


def test_the_buffer_holds_only_the_last_seconds():
    one = started(quiet())
    one.frames(2.0, 30.0)
    frames = one.core._round.frames
    assert frames[-1][0] - frames[0][0] <= BUFFER_SEC
    assert len(frames) <= BUFFER_SEC * FPS + 1


# -- the choice and its metadata ---------------------------------------------------------


def test_a_gesture_is_chosen_over_a_pose_and_the_rhythm():
    one = started()
    one.frames(2.0, 8.0)
    round_now = one.core._round
    stamps = [frame[0] for frame in round_now.frames]
    one.core._offer(round_now, Trigger(STILL, STILL, 1.5, 1.0, stamps[-1]))
    one.core._offer(round_now, _gesture(stamps[-10], label="Thumb_Up"))
    photo, _ = choose_photo(one.stop(), SECTION)
    assert photo.meta["source"] == GESTURE
    assert photo.meta["label"] == "Thumb_Up"
    assert photo.meta["candidate_sources"] == {UNIFORM: 3, STILL: 1, GESTURE: 1}


def test_the_metadata_describes_the_photo():
    image = np.random.default_rng(1).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    data = cv2.imencode(".jpg", image)[1].tobytes()
    one = Round(quiet())
    one.state(STATE_MIRRORING)
    for index in range(40):
        one.core.on_image(index / FPS, data)
    one.core.on_event(TriggerRequest("preset_motions", "heart", 0.7))
    photo, _ = choose_photo(one.stop(), SECTION)
    assert photo.name == one.finished[0].round_id
    meta = photo.meta
    assert meta["schema"] == SCHEMA
    assert meta["round_id"] == photo.name
    assert (meta["width"], meta["height"]) == (320, 240)
    assert meta["bytes"] == len(data)
    assert meta["sha256"] == hashlib.sha256(data).hexdigest()
    assert (meta["source"], meta["label"], meta["origin"], meta["score"]) == (EVENT, "heart", "preset_motions", 0.7)
    assert meta["scoring"] == "single"
    assert meta["end_reason"] == "state"
    json.dumps(meta)  # goes in a header and a file


def test_the_photo_is_the_frame_as_the_camera_sent_it():
    one = started()
    one.frames(2.0, 5.0)
    photo, _ = choose_photo(one.stop(), SECTION)
    assert photo.jpeg is FRAME


# -- the crash copy ----------------------------------------------------------------------


def test_the_provisional_best_follows_the_heaviest_then_the_latest():
    one = started(quiet())
    one.core._offer(one.core._round, Trigger(STILL, STILL, 1.5, 1.0, 1.7))
    one.core._offer(one.core._round, _gesture(1.8))
    one.core._offer(one.core._round, Trigger(STILL, STILL, 1.5, 1.0, 1.9))
    best = [args[2].trigger.source for args in one.provisional]
    assert best == [STILL, GESTURE]


def test_the_provisional_photo_says_so():
    one = started(quiet())
    one.core._offer(one.core._round, _gesture(1.8))
    round_id, started_at, candidate = one.provisional[0]
    photo = provisional_photo(round_id, started_at, candidate)
    assert photo.name == round_id
    assert photo.meta["quality"] == "provisional"
    assert photo.jpeg == FRAME


# -- status ------------------------------------------------------------------------------


def test_the_status_tells_a_round_and_the_last_one():
    one = started()
    status = one.core.status()
    assert status["mirroring"] is True
    assert status["round"]["candidates"] == {UNIFORM: 1}
    one.stop()
    status = one.core.status()
    assert status["mirroring"] is False
    assert status["rounds"] == 1
    assert status["last_round"]["reason"] == "state"
    assert "round" not in status
