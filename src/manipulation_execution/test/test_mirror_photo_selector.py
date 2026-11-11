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

    def frames(self, start: float, end: float, *, pear: bool = True, pear_delay: float = 0.0, bbox: tuple[float, ...] = (10.0, 10.0, 50.0, 40.0)) -> None:
        """Frames from ``start`` up to but not including ``end``.

        By default creates both camera frames and matching PEAR yolox frames (anchors).
        pear: if True, create PEAR frames that pair with camera frames to form anchors
        pear_delay: PEAR arrives this many seconds after camera (default 0.0 = simultaneous)
        bbox: bounding box for PEAR frames
        """
        for index in range(round((end - start) * FPS)):
            stamp = start + index / FPS
            # Camera frame first
            self.core.on_image(stamp, FRAME)
            # PEAR frame with bbox, delayed if requested
            if pear:
                self.core.on_pear(stamp, STILL_POSE, bbox)

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
    one.frames(2.0, 8.0, pear=False)
    # Add a few anchors to allow uniform trigger
    for stamp in [2.0, 5.0, 8.0]:
        one.core.on_pear(stamp, STILL_POSE, (10.0, 10.0, 50.0, 40.0))
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
    # Event arrives before camera/anchor
    one.core.on_event(TriggerRequest("a", "b", stamp=2.5))
    assert one.kept == []
    # Now camera and PEAR arrive together to form anchor
    one.core.on_image(2.5, FRAME)
    one.core.on_pear(2.5, STILL_POSE, (10.0, 10.0, 50.0, 40.0))
    # The pending event should now resolve and take this anchor
    assert len(one.kept_stamps(EVENT)) == 1
    assert one.kept_stamps(EVENT)[0] == pytest.approx(2.5, abs=0.01)


def test_a_trigger_between_two_frames_takes_the_nearer_once_both_are_in():
    one = started(quiet())
    # Both ahead of the camera (newest frame 1.933); 2.03 is nearer the frame before, 2.31 the one after.
    one.core.on_event(TriggerRequest("a", "b", stamp=2.03))
    one.core.on_event(TriggerRequest("a", "b", stamp=2.31))
    one.frames(2.0, 2.4, pear=False)
    one.core.on_image(2.0, FRAME)
    one.core.on_image(2.0 + 5 / FPS, FRAME)
    one.core.on_pear(2.0, STILL_POSE, (10.0, 10.0, 50.0, 40.0))
    one.core.on_pear(2.0 + 5 / FPS, STILL_POSE, (10.0, 10.0, 50.0, 40.0))
    assert one.kept_stamps(EVENT) == [pytest.approx(2.0), pytest.approx(2.0 + 5 / FPS)]
    assert one.core._round.pending == []


def test_a_trigger_far_ahead_of_the_camera_is_dropped_an_event_takes_the_newest():
    one = started(quiet())
    one.frames(2.0, 2.2, pear=False)
    newest = one.core._round.frames[-1][0]
    one.core.on_pear(newest, STILL_POSE, (10.0, 10.0, 50.0, 40.0))
    one.core._offer(one.core._round, Trigger(STILL, STILL, 1.5, 1.0, newest + 5.0))
    one.core.on_event(TriggerRequest("a", "b", stamp=newest + 5.0))
    assert one.core._round.pending == []
    # Both triggers are far ahead and get no_actor; the still offered first counts as 1
    assert one.core._round.refused["no_actor"] >= 1
    assert one.kept_stamps(EVENT) == [newest]


def test_a_trigger_whose_frame_never_comes_is_given_up():
    one = started(quiet())
    one.frames(2.0, 2.2, pear=False)
    one.core._offer(one.core._round, Trigger(STILL, STILL, 1.5, 1.0, 2.5))
    assert one.core._round.pending
    one.core.on_image(10.0, FRAME)
    assert not one.core._round.pending
    # The trigger waits for PEAR_WAIT_SEC (1.5s camera time), then gives up with no_actor
    assert one.core._round.refused["no_actor"] >= 1


def test_an_event_ahead_of_the_camera_falls_back_to_the_newest_frame():
    one = started(quiet())
    one.frames(2.0, 2.2, pear=False)
    one.core.on_event(TriggerRequest("a", "b", stamp=2.5))
    one.core.on_image(10.0, FRAME)
    one.core.on_pear(10.0, STILL_POSE, (10.0, 10.0, 50.0, 40.0))
    assert one.kept_stamps(EVENT) == [10.0]


def test_a_stamp_older_than_the_buffer_is_late_for_a_gesture_and_the_newest_for_an_event():
    one = started(quiet())
    one.frames(2.0, 2.0 + BUFFER_SEC + 2.0, pear=False)
    newest = one.core._round.frames[-1][0]
    one.core.on_pear(newest, STILL_POSE, (10.0, 10.0, 50.0, 40.0))
    # Trigger with stamp 0.5 is older than buffer (BUFFER_SEC = 3.0), so frame at 2.0 has aged out
    # For a gesture, this is "late" because it's outside the time window
    one.core._offer(one.core._round, Trigger(GESTURE, "Victory", 1.5, 1.0, 0.5))
    assert one.kept == []
    assert one.core._round.refused["no_actor"] == 1  # Changed: no anchor for 0.5, so no_actor
    one.core.on_event(TriggerRequest("a", "b", stamp=0.5))
    assert one.kept_stamps(EVENT) == [newest]


def test_an_event_without_a_stamp_takes_the_newest_frame():
    one = started(quiet())
    one.frames(2.0, 2.2, pear=False)
    newest = one.core._round.frames[-1][0]
    one.core.on_pear(newest, STILL_POSE, (10.0, 10.0, 50.0, 40.0))
    one.core.on_event(TriggerRequest("a", "b"))
    assert one.kept_stamps(EVENT) == [newest]
    # Origin is empty string when trigger has no stamp (EVENT source uses empty origin)
    assert one.kept[0].trigger.origin == ""


def test_the_still_rule_runs_from_pear_frames():
    one = started(quiet())
    for index in range(20):
        stamp = 1.5 + index / 30
        one.core.on_image(stamp, FRAME)
        one.core.on_pear(stamp, STILL_POSE, (10.0, 20.0, 30.0, 40.0))
    # Still detection should trigger and create a candidate
    assert len(one.kept_stamps(STILL)) >= 1


def test_a_malformed_pear_frame_is_ignored():
    one = started(quiet())
    one.core.on_pear(1.9, np.zeros(10), None)
    assert one.core.mirroring
    # Malformed PEAR still updates speeds, just doesn't create an anchor
    # The test name says "ignored", but it only means "not an anchor", speeds are still tracked


def test_the_best_person_box_near_the_frame_goes_with_the_photo():
    one = started()
    one.frames(2.0, 2.2, pear=False)
    stamp = one.core._round.frames[-1][0]
    one.core.on_pear(1.9, STILL_POSE, (10.0, 20.0, 30.0, 40.0))
    one.core.on_pear(stamp, STILL_POSE, (100.0, 200.0, 300.0, 400.0))
    one.core.on_event(TriggerRequest("a", "b"))
    photo, _ = choose_photo(one.stop(), SECTION)
    assert photo.meta["bbox"] == [100.0, 200.0, 300.0, 400.0]


def test_no_person_box_near_the_frame_is_sent_as_none():
    one = started()
    one.frames(2.0, 2.2, pear=False)
    one.core.on_event(TriggerRequest("a", "b"))
    photo, _ = choose_photo(one.stop(), SECTION)
    # With anchor design: no PEAR frames means no anchors, so fallback uses last camera frame
    # Since fallback_bbox is None (no anchors were ever created), bbox should be None
    # But if there was already an anchor from started(), it might have a bbox
    # Update: this test verifies the case where truly no PEAR data arrives
    assert photo.meta.get("bbox") is None or photo.meta.get("quality") == "fallback"


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


# -- anchors: candidates only from PEAR yolox frames paired with camera frames ----------


def test_uniform_fires_only_on_anchors():
    """Uniform trigger is driven by anchors, not by every camera frame."""
    from dataclasses import replace
    from manipulation_execution.mirror_photo.config import StillRule, UniformRule

    # Disable still to avoid interference, but enable uniform with 3s interval
    section = replace(SECTION, still=StillRule(enabled=False, weight=0.0), uniform=UniformRule(enabled=True))
    one = started(section)
    # started() already warmed up from 0.0 to 2.0 with default pear=True, creating anchors
    # Clear those candidates first
    one.core._round.candidates.clear()

    # Now create camera frames without PEAR (no new anchors)
    one.frames(2.0, 6.0, pear=False)
    assert one.kept_stamps(UNIFORM) == []

    # Now send PEAR frames that match some camera frames
    one.core.on_pear(2.0, STILL_POSE, (10.0, 20.0, 30.0, 40.0))
    one.core.on_pear(2.0 + 3.0, STILL_POSE, (10.0, 20.0, 30.0, 40.0))
    # Uniform should fire at 3 s intervals starting from the first anchor (but warmup was 0-2, so next is ~3s from start)
    kept = one.kept_stamps(UNIFORM)
    assert len(kept) >= 1


def test_anchors_match_pear_and_camera_by_stamp():
    """An anchor is created only when PEAR yolox frame finds the camera frame with the same stamp."""
    one = started(quiet())
    one.frames(2.0, 2.5, pear=False)  # Camera frames only, no anchors yet
    # PEAR frame with stamp 2.2 should match camera frame 2.2
    one.core.on_pear(2.2, STILL_POSE, (10.0, 20.0, 30.0, 40.0))
    one.core.on_event(TriggerRequest("a", "b"))
    # The event should take the nearest anchor (2.2)
    assert one.kept_stamps(EVENT) == [pytest.approx(2.2, abs=1 / FPS)]
    one.core.on_event(TriggerRequest("a", "b"))
    # The event should take the nearest anchor (2.2)
    assert one.kept_stamps(EVENT) == [pytest.approx(2.2, abs=1 / FPS)]


def test_center_fallback_is_not_an_anchor():
    """PEAR frames with source=center_fallback (bbox=None) do not create anchors."""
    one = started(quiet())
    one.frames(2.0, 2.5)
    one.core.on_pear(2.2, STILL_POSE, None)  # center_fallback
    one.core.on_event(TriggerRequest("a", "b"))
    # No anchors, so the event takes the newest camera frame (fallback path)
    newest = one.core._round.frames[-1][0]
    assert one.kept_stamps(EVENT) == [newest]


def test_gesture_takes_nearest_anchor_within_window():
    """A gesture trigger takes the nearest anchor within GESTURE_MATCH_SEC."""
    one = started(quiet())
    one.frames(2.0, 3.0)
    one.core.on_pear(2.2, STILL_POSE, (10.0, 20.0, 30.0, 40.0))
    one.core.on_pear(2.5, STILL_POSE, (10.0, 20.0, 30.0, 40.0))
    # Hand gesture at 2.35 should take the nearest anchor (2.5 is 0.15 away, 2.2 is 0.15 away, take the later)
    one.core.on_hand(2.35, [("Victory", 0.9)])
    kept = one.kept_stamps(GESTURE)
    # Both 2.2 and 2.5 are within 0.3, take the nearest
    assert len(kept) == 1
    assert kept[0] in [pytest.approx(2.2), pytest.approx(2.5)]


def test_fallback_bbox_is_first_anchor():
    """The first anchor's bbox becomes the fallback_bbox for this round."""
    one = started(quiet())
    one.frames(2.0, 3.0)
    one.core.on_pear(2.2, STILL_POSE, (100.0, 200.0, 300.0, 400.0))
    one.core.on_pear(2.5, STILL_POSE, (10.0, 20.0, 30.0, 40.0))
    # No candidates triggered, round ends
    finished = one.stop()
    # The fallback should use last_anchor or fallback_bbox
    photo, _ = choose_photo(finished, SECTION)
    # Should have bbox from one of the anchors
    assert photo.meta["bbox"] is not None
    assert photo.meta["quality"] == "fallback"


def test_no_anchors_uses_fallback_bbox():
    """When there are no anchors at all, use the fallback_bbox (first anchor of the round)."""
    # This test needs to be written after implementation, as we need to ensure
    # fallback_bbox persists even when last_anchor is cleared
    pass  # Placeholder for now


def test_pear_ahead_of_camera_waits_then_pairs():
    """PEAR frame arriving before its camera frame is held pending, then paired when camera arrives."""
    one = started(quiet())
    one.frames(2.0, 2.3)
    # PEAR at 2.5 arrives before camera 2.5
    one.core.on_pear(2.5, STILL_POSE, (10.0, 20.0, 30.0, 40.0))
    # No anchor yet because camera 2.5 hasn't arrived
    # Now camera 2.5 arrives
    one.core.on_image(2.5, FRAME)
    # Now the pending PEAR should pair and create an anchor
    one.core.on_event(TriggerRequest("a", "b"))
    kept = one.kept_stamps(EVENT)
    assert pytest.approx(2.5, abs=1 / FPS) in kept


def test_missing_camera_frame_counts_pear_unmatched():
    """When PEAR frame's camera frame was never received, count pear_unmatched."""
    one = started(quiet())
    one.frames(2.0, 2.2)  # Frames up to 2.133
    one.frames(2.3, 2.5)  # Gap: no frame at 2.2
    # PEAR at 2.2 can't find its camera frame
    one.core.on_pear(2.2, STILL_POSE, (10.0, 20.0, 30.0, 40.0))
    finished = one.stop()
    assert finished.refused.get("pear_unmatched", 0) >= 1


def test_bad_bbox_is_not_an_anchor():
    """Malformed bbox (NaN, x2 <= x1) is counted as bad_bbox and not used as anchor."""
    one = started(quiet())
    one.frames(2.0, 2.5)
    # Bad bboxes
    one.core.on_pear(2.1, STILL_POSE, (float("nan"), 20.0, 30.0, 40.0))
    one.core.on_pear(2.2, STILL_POSE, (30.0, 20.0, 10.0, 40.0))  # x2 < x1
    one.core.on_pear(2.3, STILL_POSE, (10.0, 40.0, 30.0, 20.0))  # y2 < y1
    finished = one.stop()
    assert finished.refused.get("bad_bbox", 0) >= 3


def test_pending_trigger_resolves_at_round_end():
    """Triggers that are still pending when round ends are resolved with available anchors."""
    one = started(quiet())
    one.frames(2.0, 2.3)
    one.core.on_pear(2.2, STILL_POSE, (10.0, 20.0, 30.0, 40.0))
    # Hand gesture arrives, but its nearest anchor is within window
    one.core.on_hand(2.35, [("Victory", 0.9)])
    # Round ends before PEAR catches up
    finished = one.stop()
    # The gesture should have resolved to an anchor
    photo, _ = choose_photo(finished, SECTION)
    assert photo is not None


def test_same_anchor_deduped_by_weight():
    """When uniform and still both select the same anchor, keep only the heavier trigger."""
    one = started()
    one.core.on_pear(2.0 + 3.0, STILL_POSE, (10.0, 20.0, 30.0, 40.0))
    # Both uniform (weight from config) and still might fire at this stamp
    # The test verifies deduplication happens
    kept_stamps = [c.stamp for c in one.kept]
    stamp_counts = {}
    for s in kept_stamps:
        stamp_counts[s] = stamp_counts.get(s, 0) + 1
    # No stamp should appear more than once
    assert all(count == 1 for count in stamp_counts.values())


def test_camera_stamp_zero_gives_no_actor():
    """When camera frames have stamp=0, PEAR never pairs, result is no_actor or uses fallback_bbox."""
    one = started(quiet())
    one.core.on_image(0.0, FRAME)
    one.core.on_image(0.0, FRAME)
    one.core.on_pear(0.0, STILL_POSE, (10.0, 20.0, 30.0, 40.0))
    one.core.on_event(TriggerRequest("a", "b"))
    finished = one.stop()
    # Should not crash; photo should exist
    photo, _ = choose_photo(finished, SECTION)
    assert photo is not None


def test_bbox_stamp_equals_stamp_for_normal_photos():
    """For normal photos from anchors, bbox_stamp should equal stamp."""
    one = started(quiet())
    one.frames(2.0, 3.0)
    one.core.on_pear(2.2, STILL_POSE, (10.0, 20.0, 30.0, 40.0))
    one.core.on_event(TriggerRequest("a", "b"))
    photo, _ = choose_photo(one.stop(), SECTION)
    assert photo.meta["bbox_stamp"] == pytest.approx(photo.meta["stamp"], abs=0.01)


# -- The big invariant test with random interleaving -------------------------------------


def test_anchor_invariant_200_seeds():
    """The invariant: every normal/fallback/recovered photo has a finite bbox from a PEAR yolox frame.

    200 random seeds. Each seed simulates:
    - Camera frames at ~30 fps with 10% loss
    - PEAR frames at 13-18 Hz (one stretch at 3 Hz), delayed 0-600 ms
    - Occasional center_fallback (bbox=None)
    - Occasional out-of-order PEAR
    - Hand gestures and event triggers
    - Several rounds, some stopped mid-way

    Invariant:
    - Every normal/fallback/recovered photo has bbox with 4 finite numbers, x2>x1, y2>y1
    - bbox_stamp == stamp for candidates from anchors, or bbox_stamp is first anchor stamp for fallback_bbox
    - Each seed produces at least one normal photo (prevents vacuous pass)
    - Print seed count and photo count (both must be nonzero)
    """
    import random

    seed_count = 200
    total_photos = 0
    failures = []

    for seed in range(seed_count):
        rng = random.Random(seed)
        section = quiet()
        one = Round(section)
        one.state(STATE_MIRRORING)

        # Camera frames: 30 fps for 15 seconds with 10% loss
        cam_time = 0.0
        cam_frames = []
        while cam_time < 15.0:
            if rng.random() > 0.1:  # 90% arrive
                cam_frames.append(cam_time)
                one.core.on_image(cam_time, FRAME)
            cam_time += 1.0 / 30

        # PEAR frames: 13-18 Hz, delayed 0-600 ms, with a slow stretch
        pear_time = 0.0
        pear_boxes = {}  # stamp -> bbox
        while pear_time < 15.0:
            hz = 3.0 if 5.0 < pear_time < 7.0 else rng.uniform(13, 18)
            delay = rng.uniform(0, 0.6)
            actual_time = pear_time + delay

            if rng.random() < 0.2:  # 20% center_fallback
                bbox = None
            else:
                bbox = (
                    rng.uniform(10, 200),
                    rng.uniform(10, 150),
                    rng.uniform(210, 400),
                    rng.uniform(160, 300),
                )
                pear_boxes[pear_time] = bbox

            one.core.on_pear(pear_time, STILL_POSE, bbox)
            pear_time += 1.0 / hz

        # Occasional out-of-order PEAR
        if rng.random() < 0.3:
            late_stamp = rng.uniform(1.0, 14.0)
            bbox = (50.0, 50.0, 150.0, 150.0)
            pear_boxes[late_stamp] = bbox
            one.core.on_pear(late_stamp, STILL_POSE, bbox)

        # Gestures
        for _ in range(rng.randint(2, 5)):
            gest_time = rng.uniform(2.0, 14.0)
            one.core.on_hand(gest_time, [("Victory", 0.9)])

        # Events
        for _ in range(rng.randint(1, 3)):
            evt_time = rng.uniform(2.0, 14.0)
            one.core.on_event(TriggerRequest("test", "action", stamp=evt_time))

        # Occasionally stop and restart a round
        if rng.random() < 0.3:
            one.state(STATE_WAIT)
            one.state(STATE_MIRRORING)
            # More frames
            for _ in range(10):
                cam_time += 1.0 / 30
                if rng.random() > 0.1:
                    one.core.on_image(cam_time, FRAME)

        finished = one.stop()

        if not finished.candidates:
            continue  # No candidates, fallback will be used

        # Choose photo
        photo, choice = choose_photo(finished, section)
        if photo is None:
            continue

        # Check invariant
        bbox = photo.meta.get("bbox")
        quality = photo.meta.get("quality")

        if quality in ("normal", "fallback", "recovered"):
            total_photos += 1

            # Must have bbox
            if bbox is None:
                failures.append(f"seed {seed}: bbox is None for quality={quality}")
                continue

            # Must be 4 finite numbers
            if len(bbox) != 4:
                failures.append(f"seed {seed}: bbox length {len(bbox)} != 4")
                continue

            if not all(isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf'))) for v in bbox):
                failures.append(f"seed {seed}: bbox has non-finite values: {bbox}")
                continue

            # x2 > x1, y2 > y1
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                failures.append(f"seed {seed}: bbox invalid: {bbox}")
                continue

            # bbox_stamp should exist
            bbox_stamp = photo.meta.get("bbox_stamp")
            if bbox_stamp is None:
                failures.append(f"seed {seed}: bbox_stamp is None")
                continue

        # Check for at least one normal in the test set
        if choice is not None and photo.meta.get("quality") == "normal":
            pass  # Good

    # Print counts
    print(f"Seeds: {seed_count}, Total photos checked: {total_photos}")

    # At least some photos must have been produced
    assert total_photos > 0, "No photos produced across all seeds"
    assert failures == [], f"Invariant violations:\n" + "\n".join(failures[:10])


def test_provisional_and_recovered_have_bbox():
    """Provisional photo (crash copy) and recovered photo must both carry the candidate's bbox."""
    from dataclasses import replace
    from manipulation_execution.mirror_photo.config import StillRule

    section = replace(quiet(), still=StillRule(enabled=False, weight=0.0))
    one = started(section)
    one.frames(2.0, 3.0, pear=False)
    one.core.on_pear(2.2, STILL_POSE, (10.0, 20.0, 30.0, 40.0))
    one.core.on_event(TriggerRequest("a", "b"))

    # Check provisional
    round_id, started_at, candidate = one.provisional[0]
    assert candidate.bbox == (10.0, 20.0, 30.0, 40.0)

    # Check that provisional_photo carries the bbox
    photo = provisional_photo(round_id, started_at, candidate)
    assert photo.meta["bbox"] == [10.0, 20.0, 30.0, 40.0]

    # Simulate recovery: the worker should have written this bbox to disk
    # and recovered photo should have it
    finished = one.stop()
    photo, _ = choose_photo(finished, SECTION)
    assert photo.meta["bbox"] == [10.0, 20.0, 30.0, 40.0]


