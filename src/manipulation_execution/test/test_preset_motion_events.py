"""A held pose creates one event and preserves source-frame timing."""

from types import SimpleNamespace

from manipulation_execution.preset_motions.events import match_event, match_state


def result(**changes):
    values = dict(
        entered=True,
        source_stamp_ns=12_345_678_901,
        template_name="overhead_heart",
        score=0.8,
        actor_id=3,
        session_epoch=4,
        sequence=7,
        active=True,
        reason="holding",
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_event_keeps_source_and_match_times_separate():
    event = match_event(result(), 13_000_000_009, "unique")
    assert event == {
        "schema_version": 1,
        "event_id": "unique",
        "template_name": "overhead_heart",
        "score": 0.8,
        "actor_id": 3,
        "source_stamp": {"sec": 12, "nanosec": 345_678_901},
        "matched_stamp": {"sec": 13, "nanosec": 9},
    }


def test_held_pose_and_missing_source_do_not_emit():
    assert match_event(result(entered=False), 20) is None
    assert match_event(result(source_stamp_ns=None), 20) is None


def test_separate_confirmations_get_different_event_ids():
    assert match_event(result(), 20)["event_id"] != match_event(result(), 20)["event_id"]


def test_live_state_is_separate_from_once_only_event():
    state = match_state(result(entered=False), 13_000_000_009, session_active=True)
    assert state["active"] is True and state["input_valid"] is True
    assert state["session_epoch"] == 4 and state["source_stamp_ns"] == 12_345_678_901
    assert "event_id" not in state


def test_normal_release_and_entry_confirmation_gap_remain_valid_input():
    for reason in ("pose_exited", "pose_not_matched", "confirmation_gap"):
        state = match_state(result(active=False, reason=reason), 20, session_active=True)
        assert state["active"] is False and state["input_valid"] is True


def test_lost_input_and_stopped_session_are_never_active_evidence():
    invalid = match_state(result(active=False, reason="pose_stale"), 20, session_active=True)
    assert not invalid["input_valid"]
    stopped = match_state(result(), 20, session_active=False)
    assert not stopped["active"] and not stopped["input_valid"]
