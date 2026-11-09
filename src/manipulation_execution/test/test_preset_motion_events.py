"""A held pose creates one event and preserves source-frame timing."""

from types import SimpleNamespace

from manipulation_execution.preset_motions.events import match_event


def result(**changes):
    values = dict(entered=True, source_stamp_ns=12_345_678_901, template_name="overhead_heart", score=0.8, actor_id=3)
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
