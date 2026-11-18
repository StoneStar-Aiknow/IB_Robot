"""The gesture runner (hand_perception/gesture_runner.py) with a fake om: messages, gating, fail silent."""

import json

import numpy as np
import pytest

from manipulation_execution.hand_perception import gesture_crop as gc
from manipulation_execution.hand_perception import gesture_om as go
from manipulation_execution.hand_perception import gesture_runner as gr

MS = 1_000_000
BOX = (100.0, 0.0, 500.0, 1000.0)
IMG = np.zeros((1080, 1920, 3), np.uint8)


def hand(cx, cy, size=40.0):
    lm = np.zeros((21, 2))
    lm[gc.WRIST] = (0.0, size / 2)
    lm[gc.MIDDLE_MCP] = (0.0, -size / 2)
    lm -= lm.mean(axis=0)
    return lm + (cx, cy)


TWO = [hand(250, 300), hand(330, 300)]


def heart_out(score=0.8):
    y = np.zeros(go.OUTPUT_SHAPE, np.float32)
    y[0, :4, 7] = (100, 100, 200, 200)
    y[0, 4 + 9, 7] = score
    return y


class FakeOm:
    def __init__(self, outs=None, raises=None, delay=None):
        self.outs, self.raises, self.delay = outs, raises, delay
        self.calls = 0
        self.closed = False
        self.path = "fake.om"

    def infer(self, x):
        self.calls += 1
        assert x.shape == (1, 3, 320, 320) and x.dtype == np.float32
        if self.delay:
            self.delay()
        if self.raises:
            raise self.raises
        return self.outs if self.outs is not None else heart_out()

    def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def runner(om=None, *, cfg=None, factory=None, clock=None):
    sent, logs = [], []
    om = om or FakeOm()
    r = gr.GestureRunner(
        gc.CropConfig(),
        cfg or gr.RunnerConfig(),
        model_path="x.om",
        device=0,
        publish=sent.append,
        log=logs.append,
        clock=clock or Clock(),
        om_factory=factory or (lambda _p, _d: om),
    )
    return r, om, sent, logs


def test_one_run_sends_the_v1_message():
    r, om, sent, _ = runner()
    assert r.load() and r.enabled
    msg = r.step(5 * MS, 7, IMG, BOX, TWO, busy=False)
    assert len(sent) == 1 and json.loads(sent[0]) == msg
    assert set(msg) == {"schema", "stamp_ns", "actor_id", "cls", "score", "crop", "roi", "infer_ms"}
    assert msg["schema"] == "x2_hand_gesture/v1" and msg["stamp_ns"] == 5 * MS and msg["actor_id"] == 7
    assert msg["cls"] == "heart" and msg["score"] == pytest.approx(0.8) and msg["crop"] == "both"
    plan = gc.plan_crop(TWO, BOX, gc.CropConfig())
    assert msg["roi"] == list(plan.roi)
    assert msg["infer_ms"] >= 0


def test_no_gesture_is_sent_as_none():
    r, _, sent, _ = runner(FakeOm(outs=np.zeros(go.OUTPUT_SHAPE, np.float32)))
    r.load()
    assert r.step(0, 1, IMG, BOX, TWO, False)["cls"] == "none"
    assert json.loads(sent[0])["score"] == 0.0


def test_busy_no_hands_far_and_rate_do_not_run():
    r, om, sent, _ = runner()
    r.load()
    assert r.step(0, 1, IMG, BOX, TWO, busy=True) is None
    assert r.step(0, 1, IMG, BOX, [], busy=False) is None
    assert r.step(0, 1, IMG, BOX, [hand(100, 300), hand(800, 300)], busy=False) is None
    assert om.calls == 0
    assert r.step(0, 1, IMG, BOX, TWO, False) is not None
    assert r.step(33 * MS, 1, IMG, BOX, TWO, False) is None  # 15 Hz
    assert r.step(67 * MS, 1, IMG, BOX, TWO, False) is not None
    assert om.calls == 2 and len(sent) == 2


def test_one_hand_runs_at_most_5hz():
    r, om, sent, _ = runner()
    r.load()
    one = [hand(300, 300)]
    stamps = [i * 33 * MS for i in range(13)]
    runs = [s for s in stamps if r.step(s, 1, IMG, BOX, one, False) is not None]
    assert runs == [0, 198 * MS, 396 * MS] and json.loads(sent[0])["crop"] == "one"


def test_load_failure_is_one_line_and_off():
    def boom(_p, _d):
        raise RuntimeError("no device")

    r, _, sent, logs = runner(factory=boom)
    assert not r.load() and not r.enabled and r.disabled_reason == "load"
    assert len(logs) == 1 and "load failed" in logs[0]
    assert r.step(0, 1, IMG, BOX, TWO, False) is None and not sent


def test_errors_drop_the_result_and_five_in_a_row_turn_it_off():
    om = FakeOm(raises=RuntimeError("acl"))
    clock = Clock()
    r, _, sent, logs = runner(om, clock=clock)
    r.load()
    for i in range(5):
        assert r.enabled
        assert r.step(i * 100 * MS, 1, IMG, BOX, TWO, False) is None
    assert not r.enabled and r.disabled_reason == "errors" and om.closed and not sent
    assert sum("dropped" in line for line in logs) == 1  # rate-limited
    assert r.step(10**9, 1, IMG, BOX, TWO, False) is None and om.calls == 5


def test_a_good_run_resets_the_error_count():
    om = FakeOm(raises=go.BadOutput("nan"))
    r, _, _, _ = runner(om)
    r.load()
    for i in range(4):
        r.step(i * 100 * MS, 1, IMG, BOX, TWO, False)
    om.raises = None
    assert r.step(400 * MS, 1, IMG, BOX, TWO, False) is not None
    om.raises = go.BadOutput("shape")
    for i in range(4):
        r.step((5 + i) * 100 * MS, 1, IMG, BOX, TWO, False)
    assert r.enabled


def test_three_slow_runs_in_a_row_turn_it_off():
    import time

    om = FakeOm(delay=lambda: time.sleep(0.025))
    r, _, sent, logs = runner(om)
    r.load()
    for i in range(3):
        assert r.enabled
        assert r.step(i * 100 * MS, 1, IMG, BOX, TWO, False) is not None  # a slow result still goes out
    assert not r.enabled and r.disabled_reason == "slow" and om.closed and len(sent) == 3
    assert any("over 20 ms" in line for line in logs)


def test_a_fast_run_resets_the_slow_count():
    import time

    om = FakeOm(delay=lambda: time.sleep(0.025))
    r, _, _, _ = runner(om)
    r.load()
    r.step(0, 1, IMG, BOX, TWO, False)
    r.step(100 * MS, 1, IMG, BOX, TWO, False)
    om.delay = None
    r.step(200 * MS, 1, IMG, BOX, TWO, False)
    om.delay = lambda: time.sleep(0.025)
    r.step(300 * MS, 1, IMG, BOX, TWO, False)
    r.step(400 * MS, 1, IMG, BOX, TWO, False)
    assert r.enabled


def test_publish_failure_counts_as_a_failure_and_never_raises():
    r, _, _, _ = runner()

    def bad(_text):
        raise RuntimeError("context gone")

    r._publish = bad
    r.load()
    for i in range(5):
        assert r.step(i * 100 * MS, 1, IMG, BOX, TWO, False) is None
    assert not r.enabled


def test_summary_line_is_numbers_only_and_rate_limited():
    clock = Clock()
    r, _, _, logs = runner(clock=clock)
    r.load()
    logs.clear()
    r.step(0, 1, IMG, BOX, TWO, False)
    r.step(10 * MS, 1, IMG, BOX, TWO, False)
    assert not logs
    clock.t = 1.0
    r.step(100 * MS, 1, IMG, BOX, TWO, True)
    assert len(logs) == 1
    line = logs[0]
    for key in ("runs both=1", "rate=1", "busy=1", "box_h 1000", "dist/h 0.08", "palm 40", "cls heart",
                "score 0.80", "npu_ms", "total_ms"):
        assert key in line, (key, line)


@pytest.mark.parametrize(
    "kw", [dict(conf=0.0), dict(conf=1.0), dict(slow_ms=0.0), dict(slow_limit=0), dict(fail_limit=1.5)]
)
def test_runner_config_is_checked(kw):
    with pytest.raises(ValueError):
        gr.RunnerConfig(**kw)
