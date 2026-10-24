"""The standing-zone actor selector on real ROS messages (dev plan D1-D8).

``ZoneActorSelector`` replaces ``ActorSelector`` when a zone file is set, so it
must keep that selector's contract on the node's inputs: one answer per PEAR
frame, missing or stale detections are unknown and never present, and only a
new detections stamp advances anything. On top of that each refresh, and each
unknown frame that changes the state, produces one JSON record, and those
records alone must replay to the same answers.

The zone is built in the test, not read from a file: a trapezoid in a
1000x800 image, near edge at y=700, far edge at y=500. A person is a 400 px
tall box whose feet are at the box's bottom centre.
"""

import json

from sensor_msgs.msg import Image

from ibrobot_msgs.msg import Detection2D, DetectionArray
from manipulation_execution.actor_zone_config import TuningConfig, ZoneConfig
from manipulation_execution.actor_zone_core import (
    ACQUIRING,
    GRACE,
    HOLD,
    IDLE,
    LOCKED,
    ZoneCore,
    boxes_from_report,
)
from manipulation_execution.actor_zone_selector import ZoneActorSelector

ZONE = ZoneConfig(
    path="/test/actor_zone.yaml",
    sha256="z" * 64,
    image_size=(1000, 800),
    crop_x=(100.0, 900.0),
    floor_quad=((300.0, 700.0), (700.0, 700.0), (650.0, 500.0), (350.0, 500.0)),
    near_edge_margin_px=30,
    note="test",
)
STALE_SEC = 3.0


def _stamp(message, seconds):
    message.header.stamp.sec = int(seconds)
    message.header.stamp.nanosec = int(round((seconds - int(seconds)) * 1e9))
    return message


def _frame(seconds, width=1000, height=800):
    return _stamp(Image(width=width, height=height, encoding="bgr8", step=width * 3), seconds)


def _person(foot_x, foot_y, confidence=0.9, height=400.0, label="person", scale=1.0):
    half = 60.0
    bbox = [foot_x - half, foot_y - height, foot_x + half, foot_y]
    return Detection2D(label=label, confidence=confidence, bbox=[v * scale for v in bbox])


def _detections(seconds, *people):
    return _stamp(DetectionArray(detections=list(people)), seconds)


INSIDE = (500.0, 650.0)
OUTSIDE = (850.0, 650.0)


class _Sink:
    def __init__(self):
        self.reports = []
        self.warnings = []

    def report(self, text):
        self.reports.append(json.loads(text))

    def warning(self, text):
        self.warnings.append(text)


def _selector(tuning=None, sink=None):
    sink = sink or _Sink()
    selector = ZoneActorSelector(
        ZONE, tuning or TuningConfig(), 0.30, STALE_SEC, on_report=sink.report, on_warning=sink.warning
    )
    return selector, sink


def _lock(selector, start=10.0, foot=INSIDE, confidence=0.9):
    """Two refreshes with one person inside the zone: the default acquire_updates."""
    for i in range(2):
        decision = selector.update(_frame(start + i), _detections(start + i, _person(*foot, confidence=confidence)))
    assert selector.core.state == LOCKED
    return decision


def test_one_person_inside_is_locked_on_the_second_refresh():
    selector, _ = _selector()
    first = selector.update(_frame(10.0), _detections(10.0, _person(*INSIDE)))
    assert (selector.core.state, first.actor_present, first.reason) == (ACQUIRING, False, "zone_acquiring")
    second = selector.update(_frame(11.0), _detections(11.0, _person(*INSIDE)))
    assert second.actor_present and second.start_ready
    assert second.reason == "zone_locked"
    assert second.actor_id == 1
    assert list(second.bbox) == [440.0, 250.0, 560.0, 650.0]


def test_missing_detections_are_unknown_and_release_a_locked_actor():
    # D1
    selector, sink = _selector()
    _lock(selector)
    decision = selector.update(_frame(12.0), None)
    assert decision.reason == "no_detections"
    assert not decision.actor_present and not decision.start_ready
    assert decision.actor_id == -1 and decision.persons_count == 0
    assert decision.detections_stamp is None
    assert selector.core.state == HOLD
    assert sink.reports[-1]["kind"] == "unknown"
    assert sink.reports[-1]["events"] == ["release:unknown"]


def test_stale_detections_release_a_locked_actor():
    # D2, LOCKED -> HOLD
    selector, _ = _selector()
    _lock(selector)
    decision = selector.update(_frame(11.0 + STALE_SEC + 0.1), _detections(11.0, _person(*INSIDE)))
    assert decision.reason == "stale_detections"
    assert not decision.actor_present
    assert selector.core.state == HOLD


def test_stale_detections_reset_an_acquisition():
    # D2, ACQUIRING -> IDLE
    selector, sink = _selector()
    selector.update(_frame(10.0), _detections(10.0, _person(*INSIDE)))
    assert selector.core.state == ACQUIRING
    selector.update(_frame(10.0 + STALE_SEC + 0.1), _detections(10.0, _person(*INSIDE)))
    assert selector.core.state == IDLE
    assert sink.reports[-1]["events"] == ["acquire_reset:unknown"]


def test_repeated_unknown_frames_record_only_the_change_of_state():
    selector, sink = _selector()
    _lock(selector)
    before = len(sink.reports)
    for i in range(5):
        selector.update(_frame(20.0 + 0.1 * i), _detections(11.0, _person(*INSIDE)))
    unknown = [r for r in sink.reports[before:] if r.get("kind") == "unknown"]
    assert len(sink.reports) - before == 1
    assert unknown[0]["frame_stamp"] == 20.0
    assert unknown[0]["reason"] == "stale_detections"
    assert unknown[0]["zone_sha256"] == ZONE.sha256


def test_frames_between_refreshes_reuse_the_answer_without_advancing():
    # D3: YOLOX failing keeps the old stamp; under stale_sec nothing moves.
    selector, sink = _selector()
    locked = _lock(selector)
    reports = len(sink.reports)
    for i in range(1, 20):
        decision = selector.update(_frame(11.0 + 0.14 * i), _detections(11.0, _person(*INSIDE)))
        assert decision.actor_present
        assert decision.actor_id == locked.actor_id
        assert list(decision.bbox) == list(locked.bbox)
    assert len(sink.reports) == reports
    assert selector.core.last_stamp == 11.0


def test_other_labels_and_low_confidence_boxes_are_not_people():
    # D4
    selector, _ = _selector()
    for i in range(3):
        decision = selector.update(
            _frame(10.0 + i),
            _detections(10.0 + i, _person(*INSIDE, label="chair"), _person(*INSIDE, confidence=0.29)),
        )
    assert decision.persons_count == 0
    assert not decision.actor_present
    assert decision.reason == "no_person"
    assert selector.core.state == IDLE


def test_a_frame_twice_the_zone_size_scales_the_zone_and_warns_once():
    # D5: the same scene in a 2000x1600 frame reaches the same states.
    small, _ = _selector()
    large, sink = _selector()
    scenes = [INSIDE, INSIDE, OUTSIDE, OUTSIDE, OUTSIDE]
    for i, foot in enumerate(scenes):
        a = small.update(_frame(10.0 + i), _detections(10.0 + i, _person(*foot)))
        b = large.update(_frame(10.0 + i, 2000, 1600), _detections(10.0 + i, _person(*foot, scale=2.0)))
        assert (small.core.state, a.reason, a.actor_present) == (large.core.state, b.reason, b.actor_present)
        assert [2.0 * v for v in a.bbox] == list(b.bbox)
    assert large.core.state == HOLD
    scaled = [w for w in sink.warnings if "scaled" in w]
    assert len(scaled) == 1


def test_a_missed_refresh_in_grace_keeps_the_last_box_and_confidence():
    # D6: the hand node drops the hands under 0.5, so GRACE must not zero conf.
    selector, _ = _selector()
    locked = _lock(selector, confidence=0.8)
    decision = selector.update(_frame(12.0), _detections(12.0))
    assert selector.core.state == GRACE
    assert decision.reason == "zone_grace"
    assert decision.actor_present
    assert abs(decision.confidence - 0.8) < 1e-6
    assert list(decision.bbox) == list(locked.bbox)


def test_each_new_refresh_reports_once_as_json_with_raw_boxes():
    # D7
    selector, sink = _selector()
    detections = _detections(10.0, _person(*INSIDE), _person(*OUTSIDE, label="chair"))
    for i in range(4):
        selector.update(_frame(10.0 + 0.1 * i), detections)
    assert len(sink.reports) == 1
    report = sink.reports[0]
    assert "kind" not in report
    assert report["stamp"] == 10.0
    assert report["image_size"] == [1000, 800]
    assert [b["label"] for b in report["raw"]] == ["person", "chair"]
    assert report["raw"][0]["bbox"] == [440.0, 250.0, 560.0, 650.0]
    assert report["handoff_suspected"] is False
    assert report["zone_sha256"] == ZONE.sha256


def test_a_non_finite_raw_box_is_recorded_as_null():
    selector, sink = _selector()
    broken = Detection2D(label="person", confidence=0.9, bbox=[float("nan"), 0.0, 1.0, 1.0])
    selector.update(_frame(10.0), _detections(10.0, broken))
    assert sink.reports[0]["raw"][0]["bbox"][0] is None


def test_a_failing_report_callback_costs_only_a_warning():
    warnings = []

    def broken(_text):
        raise RuntimeError("publisher gone")

    selector = ZoneActorSelector(ZONE, TuningConfig(), 0.30, STALE_SEC, on_report=broken, on_warning=warnings.append)
    selector.update(_frame(10.0), _detections(10.0, _person(*INSIDE)))
    decision = selector.update(_frame(11.0), _detections(11.0, _person(*INSIDE)))
    assert decision.actor_present
    assert any("publisher gone" in w for w in warnings)


def _replay(reports, tuning):
    core = ZoneCore(ZONE, tuning, 0.30)
    answers = []
    for report in reports:
        if report.get("kind") == "unknown":
            events = core.unknown(report["frame_stamp"])
            assert events == report["events"]
            answers.append((core.state, None, None))
            continue
        replayed = core.step(report["stamp"], tuple(report["image_size"]), boxes_from_report(report))
        answers.append((replayed["state"], replayed["actor_id"], replayed["decision"]))
    return answers


def test_recorded_reports_replay_to_the_same_answers():
    # D8: in, out, back in after the hold, a gap of missing detections, in again.
    tuning = TuningConfig()
    selector, sink = _selector(tuning)
    t = 10.0
    script = [INSIDE] * 3 + [OUTSIDE] * 4 + [None] * 3 + [INSIDE] * 3 + ["gap"] + [INSIDE] * 4
    for step in script:
        if step == "gap":
            selector.update(_frame(t), None)
        elif step is None:
            selector.update(_frame(t), _detections(t))
        else:
            selector.update(_frame(t), _detections(t, _person(*step)))
        t += 1.0
    online = [
        (r["state"], None, None) if r.get("kind") == "unknown" else (r["state"], r["actor_id"], r["decision"])
        for r in sink.reports
    ]
    assert any(r.get("kind") == "unknown" for r in sink.reports)
    assert {r["state"] for r in sink.reports} >= {ACQUIRING, LOCKED, GRACE, HOLD}
    assert _replay(sink.reports, tuning) == online
