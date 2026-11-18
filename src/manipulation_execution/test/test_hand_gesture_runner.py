"""The gesture runner (hand_perception/gesture_runner.py) with a fake om: messages, gating, fail silent, its thread."""

import json
import threading
import time

import numpy as np
import pytest

from manipulation_execution.hand_perception import gesture_crop as gc
from manipulation_execution.hand_perception import gesture_om as go
from manipulation_execution.hand_perception import gesture_runner as gr

MS = 1_000_000
BOX = (100.0, 0.0, 500.0, 1000.0)
IMG = np.zeros((1080, 1920, 3), np.uint8)
WAIT_S = 2.0  # upper bound for any wait on the gesture thread; the tests pass in far less


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
        self.close_thread = None
        self.infer_threads = set()
        self.path = "fake.om"

    def infer(self, x):
        self.calls += 1
        self.infer_threads.add(threading.current_thread().name)
        assert x.shape == (1, 3, 320, 320) and x.dtype == np.float32
        if self.delay:
            self.delay()
        if self.raises:
            raise self.raises
        return self.outs if self.outs is not None else heart_out()

    def close(self):
        self.closed = True
        self.close_thread = threading.current_thread().name


class BlockingOm(FakeOm):
    """``infer`` waits for ``release`` (bounded); ``entered`` is set while one is out."""

    def __init__(self):
        super().__init__()
        self.entered, self.release = threading.Event(), threading.Event()
        self.delay = self._block

    def _block(self):
        self.entered.set()
        assert self.release.wait(WAIT_S), "test never released the om"
        self.entered.clear()


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


def step(r, stamp, actor=1, lms=TWO, busy=False):
    """Synchronous: post one frame and run what it left (no gesture thread)."""
    if not r.post(stamp, actor, IMG, BOX, lms, busy):
        return None
    return r.run(r.take(0))


def wait_for(pred, timeout=WAIT_S):
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.002)
    return True


def test_one_run_sends_the_v1_message():
    r, om, sent, _ = runner()
    assert r.load() and r.enabled
    msg = step(r, 5 * MS, 7)
    assert len(sent) == 1 and json.loads(sent[0]) == msg
    assert set(msg) == {"schema", "stamp_ns", "actor_id", "cls", "score", "crop", "roi", "infer_ms"}
    assert msg["schema"] == "x2_hand_gesture/v1" and msg["stamp_ns"] == 5 * MS and msg["actor_id"] == 7
    assert msg["cls"] == "heart" and msg["score"] == pytest.approx(0.8) and msg["crop"] == "both"
    plan = gc.plan_crop(TWO, BOX, gc.CropConfig())
    assert msg["roi"] == list(plan.roi)
    assert msg["infer_ms"] >= 0


def test_the_posted_square_is_a_copy_of_the_frame():
    r, _, _, _ = runner()
    r.load()
    img = np.full((1080, 1920, 3), 50, np.uint8)
    assert r.post(0, 1, img, BOX, TWO, False)
    img[:] = 0  # the hand worker reuses or drops its frame
    job = r.take(0)
    assert job.square.shape == (job.plan.side, job.plan.side, 3) and job.square.min() == 50


def test_no_gesture_is_sent_as_none():
    r, _, sent, _ = runner(FakeOm(outs=np.zeros(go.OUTPUT_SHAPE, np.float32)))
    r.load()
    assert step(r, 0)["cls"] == "none"
    assert json.loads(sent[0])["score"] == 0.0


def test_busy_no_hands_far_and_rate_do_not_run():
    r, om, sent, _ = runner()
    r.load()
    assert step(r, 0, busy=True) is None
    assert step(r, 0, lms=[]) is None
    assert step(r, 0, lms=[hand(100, 300), hand(800, 300)]) is None
    assert step(r, 0, actor=None) is None
    assert om.calls == 0
    assert step(r, 0) is not None
    assert step(r, 33 * MS) is None  # 15 Hz
    assert step(r, 67 * MS) is not None
    assert om.calls == 2 and len(sent) == 2


def test_one_hand_runs_at_most_5hz():
    r, om, sent, _ = runner()
    r.load()
    one = [hand(300, 300)]
    stamps = [i * 33 * MS for i in range(13)]
    runs = [s for s in stamps if step(r, s, lms=one) is not None]
    assert runs == [0, 198 * MS, 396 * MS] and json.loads(sent[0])["crop"] == "one"


def test_nothing_is_posted_before_the_load():
    r, om, _, _ = runner()
    assert r.state == gr.LOADING and not r.post(0, 1, IMG, BOX, TWO, False) and r.take(0) is None


def test_load_failure_is_one_line_and_off():
    def boom(_p, _d):
        raise RuntimeError("no device")

    r, _, sent, logs = runner(factory=boom)
    assert not r.load() and r.off and r.disabled_reason == "load"
    assert len(logs) == 1 and "load failed" in logs[0]
    assert step(r, 0) is None and not sent


def test_errors_drop_the_result_and_five_in_a_row_turn_it_off():
    om = FakeOm(raises=RuntimeError("acl"))
    r, _, sent, logs = runner(om)
    r.load()
    for i in range(5):
        assert r.enabled
        assert step(r, i * 100 * MS) is None
    assert r.off and r.disabled_reason == "errors" and not sent
    assert sum("dropped" in line for line in logs) == 1  # rate-limited
    assert step(r, 10**9) is None and om.calls == 5


def test_a_good_run_resets_the_error_count():
    om = FakeOm(raises=go.BadOutput("nan"))
    r, _, _, _ = runner(om)
    r.load()
    for i in range(4):
        step(r, i * 100 * MS)
    om.raises = None
    assert step(r, 400 * MS) is not None
    om.raises = go.BadOutput("shape")
    for i in range(4):
        step(r, (5 + i) * 100 * MS)
    assert r.enabled


def slow():
    time.sleep(0.025)


def test_three_slow_runs_turn_it_off():
    r, _, sent, logs = runner(FakeOm(delay=slow))
    r.load()
    for i in range(3):
        assert r.enabled
        assert step(r, i * 100 * MS) is not None  # a slow result still goes out
    assert r.off and r.disabled_reason == "slow" and len(sent) == 3
    assert any("3 of the last 3 runs over 20 ms" in line for line in logs)


def test_alternating_fast_and_slow_runs_turn_it_off():
    """3 slow of the last 10 is enough; a fast run in between no longer saves it."""
    om = FakeOm()
    r, _, sent, logs = runner(om)
    r.load()
    for i in range(5):
        assert r.enabled
        om.delay = slow if i % 2 == 0 else None
        assert step(r, i * 100 * MS) is not None
    assert r.off and r.disabled_reason == "slow" and len(sent) == 5
    assert any("3 of the last 5 runs" in line for line in logs)


def test_two_slow_runs_in_every_ten_stay_on():
    om = FakeOm()
    r, _, sent, _ = runner(om)
    r.load()
    for i in range(25):
        om.delay = slow if i % 5 == 0 else None  # 2 of any 10 in a row
        step(r, i * 100 * MS)
    assert r.enabled and len(sent) == 25


def test_publish_failure_counts_as_a_failure_and_never_raises():
    r, _, _, _ = runner()

    def bad(_text):
        raise RuntimeError("context gone")

    r._publish = bad
    r.load()
    for i in range(5):
        assert step(r, i * 100 * MS) is None
    assert r.off


def test_summary_line_is_numbers_only_and_rate_limited():
    clock = Clock()
    r, _, _, logs = runner(clock=clock)
    r.load()
    logs.clear()
    step(r, 0)
    step(r, 10 * MS)
    assert not logs
    clock.t = 1.0
    step(r, 100 * MS, busy=True)
    assert len(logs) == 1
    line = logs[0]
    for key in (
        "runs both=1",
        "rate=1",
        "busy=1",
        "box_h 1000",
        "dist/h 0.08",
        "palm 40",
        "cls heart",
        "score 0.80",
        "npu_ms",
        "total_ms",
    ):
        assert key in line, (key, line)


@pytest.mark.parametrize(
    "kw",
    [
        dict(conf=0.0),
        dict(conf=1.0),
        dict(slow_ms=0.0),
        dict(slow_limit=0),
        dict(fail_limit=1.5),
        dict(slow_window=0),
        dict(slow_window=2, slow_limit=3),
        dict(watchdog_ms=0.0),
        dict(watchdog_ms=float("inf")),
    ],
)
def test_runner_config_is_checked(kw):
    with pytest.raises(ValueError):
        gr.RunnerConfig(**kw)


# -- the gesture thread -------------------------------------------------------------


def test_the_om_loads_runs_and_is_freed_on_the_gesture_thread():
    where = {}
    om = FakeOm()

    def factory(_p, _d):
        where["load"] = threading.current_thread().name
        return om

    r, _, sent, _ = runner(factory=factory)
    r.start()
    assert wait_for(lambda: r.enabled)
    assert where["load"] == "hand-gesture" != threading.current_thread().name
    assert r.post(0, 1, IMG, BOX, TWO, False)
    assert wait_for(lambda: len(sent) == 1)
    r.stop()
    assert not r.thread.is_alive()
    assert om.infer_threads == {"hand-gesture"} and om.closed and om.close_thread == "hand-gesture"


def test_a_slow_load_never_holds_the_hand_path():
    loaded = threading.Event()
    om = FakeOm()

    def factory(_p, _d):
        assert loaded.wait(WAIT_S)
        return om

    r, _, _, _ = runner(factory=factory)
    t0 = time.perf_counter()
    r.start()
    posted = [r.post(i * 100 * MS, 1, IMG, BOX, TWO, False) for i in range(5)]
    assert time.perf_counter() - t0 < 0.5 and not any(posted) and r.state == gr.LOADING
    loaded.set()
    assert wait_for(lambda: r.enabled)
    r.stop()


def test_posts_during_a_run_keep_only_the_latest():
    om = BlockingOm()
    r, _, sent, logs = runner(om)
    r.start()
    assert wait_for(lambda: r.enabled)
    assert r.post(0, 1, IMG, BOX, TWO, False)
    assert om.entered.wait(WAIT_S)  # the thread has taken stamp 0 and is in the om
    assert r.post(100 * MS, 1, IMG, BOX, TWO, False)
    assert r.post(200 * MS, 1, IMG, BOX, TWO, False)  # replaces 100 ms, not yet taken
    assert r._skips["replaced"] == 1
    om.release.set()
    assert wait_for(lambda: len(sent) == 2)
    assert [json.loads(s)["stamp_ns"] for s in sent] == [0, 200 * MS] and om.calls == 2
    r.stop()


def test_the_hand_path_stays_fast_while_a_run_is_blocked():
    om = BlockingOm()
    r, _, _, _ = runner(om)
    r.start()
    assert wait_for(lambda: r.enabled)
    assert r.post(0, 1, IMG, BOX, TWO, False) and om.entered.wait(WAIT_S)
    times = []
    for i in range(1, 31):
        t0 = time.perf_counter()
        r.post(i * 100 * MS, 1, IMG, BOX, TWO, False)
        times.append(time.perf_counter() - t0)
    assert max(times) < 0.05, max(times)  # a crop and a copy; the om is still out
    assert om.calls == 1 and r.enabled
    om.release.set()
    r.stop()


def test_the_watchdog_turns_a_hung_run_off_and_leaves_it():
    clock = Clock()
    om = BlockingOm()
    r, _, sent, logs = runner(om, clock=clock)
    r.start()
    assert wait_for(lambda: r.enabled)
    assert r.post(0, 1, IMG, BOX, TWO, False) and om.entered.wait(WAIT_S)
    clock.t = 0.15
    assert r.post(100 * MS, 1, IMG, BOX, TWO, False) and r.enabled  # under 200 ms: still on
    clock.t = 0.25
    t0 = time.perf_counter()
    assert not r.post(200 * MS, 1, IMG, BOX, TWO, False)
    assert time.perf_counter() - t0 < 0.05
    assert r.off and r.disabled_reason == "hung"
    assert sum("gesture model off" in line for line in logs) == 1 and "200 ms" in logs[-1]
    assert not r.post(300 * MS, 1, IMG, BOX, TWO, False)
    r.stop(timeout=0.0)  # the hung thread is not waited for
    om.release.set()  # it comes back late: nothing goes out, the om is freed
    r.thread.join(WAIT_S)
    assert not r.thread.is_alive() and not sent and om.closed and om.calls == 1


def test_a_gesture_thread_that_dies_turns_the_gesture_off():
    r, om, sent, logs = runner()

    def broken(_timeout):
        raise RuntimeError("boom")

    r.take = broken
    r.start()
    assert wait_for(lambda: r.off) and r.disabled_reason == "thread"
    assert wait_for(lambda: om.closed) and not r.post(0, 1, IMG, BOX, TWO, False)
    assert any("gesture thread ended (RuntimeError: boom)" in line for line in logs)
