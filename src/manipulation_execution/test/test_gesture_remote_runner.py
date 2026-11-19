"""No models or ROS graph: deterministic failures of the optional service client."""

import json
import threading
import time
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_common.hand_gesture_contract import ResultMeta, SourceState
from manipulation_execution.hand_perception.gesture_crop import CropConfig
from manipulation_execution.hand_perception.remote_gesture import RemoteConfig, RemoteGestureRunner

SERVER = "00000000-0000-4000-8000-000000000003"
ENGINE = "a" * 64
ROS0 = 1000.0
IMAGE = np.full((480, 640, 3), 33, np.uint8)
BOX = np.array((100, 10, 540, 470), np.float64)


def landmarks(x):
    lm = np.tile((x, 250.0), (21, 1))
    lm[0], lm[9] = (x, 270), (x, 230)
    return lm


HANDS = [landmarks(280), landmarks(360)]


class Clock:
    t = 0.0

    def __call__(self):
        return self.t

    def ros(self):
        return ROS0 + self.t


def response(meta, *, gesture="holy", idle=True, status="ok", request=None, server=SERVER):
    result = ResultMeta(
        request or meta,
        server,
        ENGINE,
        status=status,
        gesture=gesture,
        score=0.9 if gesture != "none" else 0.0,
        idle=idle,
        retired_through=meta.retire_through,
    )
    return SimpleNamespace(
        success=True,
        result_json=result.to_json(),
        inference_time_ms=5.0,
        model=SimpleNamespace(ready=True, deployment_fingerprint="b" * 64),
    )


class Transport:
    def __init__(self):
        self.available = True
        self.calls = []
        self.reaped = []
        self.control_response = lambda meta: response(meta)
        self.block, self.entered = None, threading.Event()
        self.raise_infer = False

    def ready(self):
        return self.available

    def begin(self, meta, pixels, frame_id):
        future = Future()
        self.calls.append((meta, pixels, frame_id, future))
        if meta.op != "infer":
            future.set_result(self.control_response(meta))
        else:
            if self.block is not None:
                self.entered.set()
                assert self.block.wait(2.0), "test must release the blocked sender"
            if self.raise_infer:
                raise RuntimeError("send may have reached the wire")
        return future

    def reap(self, future, *, cancel=False):
        self.reaped.append((future, cancel))
        if cancel and not future.done():
            future.cancel()

    @property
    def images(self):
        return [c for c in self.calls if c[0].op == "infer"]


def context(r, c, *, transition=42.0):
    r.observe_session(
        json.dumps(
            dict(
                seq=int(c() * 1e6) + 1,
                stamp=c.ros(),
                state="MIRRORING",
                phase="running",
                holding=True,
                releasing=False,
                last_transition={"from": "MIRRORING/approach", "to": "MIRRORING/running", "t": transition},
            )
        ),
        now_ros_s=c.ros(),
        now_mono_s=c(),
    )


def make(config=None):
    c, t, sent, sources = Clock(), Transport(), [], []
    r = RemoteGestureRunner(
        CropConfig(),
        config or RemoteConfig(ready_samples=1),
        transport=t,
        publish=sent.append,
        publish_source=sources.append,
        clock=c,
        ros_clock=c.ros,
    )
    context(r, c)
    return r, c, t, sent, sources


def boot(r):
    r.step()
    r.step()
    assert r._healthy


def post(r, c, **kw):
    context(r, c)
    stamp = int((c.ros() - 0.02) * 1e9)
    return r.post(
        stamp,
        kw.pop("actor_id", 3),
        IMAGE,
        BOX,
        kw.pop("hands", HANDS),
        pear_stamp_ns=stamp - 10_000_000,
        frame_id="camera",
        **kw,
    )


def test_service_missing_or_unready_never_processes_images_or_deletes_runner():
    r, c, t, sent, sources = make()
    t.available = False
    for _ in range(10):
        r.step()
        assert not post(r, c)
    assert not t.calls and not sent and not r.off
    assert not SourceState.from_json(sources[-1]).available


def test_post_only_hands_over_references_and_metadata(monkeypatch):
    r, c, t, _, _ = make()
    boot(r)
    from manipulation_execution.hand_perception import gesture_crop

    monkeypatch.setattr(gesture_crop, "extract_square", lambda *_: pytest.fail("crop ran on post"))
    assert post(r, c)
    assert r._job.image is IMAGE
    assert not np.shares_memory(r._job.landmarks[0], HANDS[0])
    assert not t.images


def test_model_result_keeps_original_context_and_publishes_a_true_none():
    r, c, t, sent, sources = make()
    boot(r)
    assert post(r, c)
    r.step()
    meta, pixels, frame_id, future = t.images[-1]
    assert pixels.shape == (320, 320, 3) and pixels.dtype == np.uint8
    assert frame_id == "camera" and meta.actor_id == 3
    c.t = 0.05
    future.set_result(response(meta, gesture="none"))
    r.step()
    message = json.loads(sent[-1])
    assert message["stamp_ns"] == meta.stamp_ns and message["roi"] == list(meta.roi)
    assert message["cls"] == "none" and message["score"] == 0.0
    assert message["session_token"] == meta.session_token
    assert SourceState.from_json(sources[-1]).source_key == meta.source_key


def test_timeout_keeps_image_slot_until_a_matched_retirement_finishes():
    r, c, t, sent, _ = make()
    boot(r)
    post(r, c)
    r.step()
    old = t.images[-1]
    c.t = 0.41
    post(r, c)
    r.step()
    assert r._call.expired and not old[-1].cancelled()
    assert len(t.images) == 1
    c.t = 1.01
    post(r, c)
    r.step()  # dispatch only retire, not another image
    assert t.calls[-1][0].op == "retire" and len(t.images) == 1
    c.t = 1.02
    r.step()
    assert old[-1].cancelled() and len(t.images) == 2
    assert not sent


def test_busy_or_unmatched_retirement_cannot_free_the_image_slot():
    r, c, t, sent, _ = make()
    boot(r)
    post(r, c)
    r.step()
    t.control_response = lambda meta: response(meta, idle=False, status="busy")
    c.t = 1.01
    post(r, c)
    r.step()
    r.step()
    assert r._call is not None and len(t.images) == 1 and not sent
    assert not t.images[0][-1].cancelled()


def test_locally_cancelled_future_is_not_a_real_terminal_response():
    r, c, t, sent, _ = make()
    boot(r)
    post(r, c)
    r.step()
    t.images[-1][-1].cancel()
    c.t = 0.01
    r.step()
    assert r._call is not None and r._call.expired and not sent
    assert len(t.images) == 1


def test_late_completion_only_reaps_it_never_becomes_a_vote():
    r, c, t, sent, _ = make()
    boot(r)
    post(r, c)
    r.step()
    meta, _, _, future = t.images[-1]
    c.t = 0.41
    future.set_result(response(meta))
    r.step()
    assert r._call is None and not sent
    assert r.stats["stale_result"] == 1


@pytest.mark.parametrize("changed", ["session", "echo", "engine", "server"])
def test_switched_or_corrupt_context_never_publishes(changed):
    r, c, t, sent, _ = make()
    boot(r)
    post(r, c)
    r.step()
    meta, _, _, future = t.images[-1]
    c.t = 0.05
    reply = response(meta)
    if changed == "session":
        context(r, c, transition=43.0)
    elif changed == "echo":
        reply = response(meta, request=replace(meta, request_id=meta.request_id + 10))
    elif changed == "engine":
        reply.result_json = reply.result_json.replace(ENGINE, "c" * 64)
    else:
        reply = response(meta, server="00000000-0000-4000-8000-000000000099")
    future.set_result(reply)
    r.step()
    assert not sent


def test_ready_probes_never_create_votes_and_source_generation_stays_stable():
    r, c, t, sent, sources = make()
    boot(r)
    generation = r.generation
    for n in range(1, 4):
        c.t = n + 0.01
        context(r, c)
        r.step()
        r.step()
    assert r.generation == generation and not sent
    assert all(c[0].op == "probe" for c in t.calls)
    assert all(SourceState.from_json(s).stream_generation == generation for s in sources)


def test_wrong_engine_stays_quiet_and_does_not_fall_back_to_local_inference():
    r, c, t, sent, _ = make(RemoteConfig(ready_samples=1, expected_engine_sha256="f" * 64))
    r.step()
    r.step()
    assert not r._healthy and not post(r, c) and not sent and not t.images


def test_uncertain_send_does_not_expose_an_empty_slot():
    r, c, t, sent, _ = make()
    boot(r)
    t.raise_infer = True
    post(r, c)
    r.step()
    assert r._call is not None and r._call.expired and r._call.future is None
    post(r, c)
    r.step()
    assert len(t.images) == 1 and not sent


def test_blocked_sender_does_not_block_hand_posts_or_stop():
    r, c, t, _, _ = make()
    boot(r)
    t.block = threading.Event()
    post(r, c)
    worker = threading.Thread(target=r.step, daemon=True)
    worker.start()
    assert t.entered.wait(1.0)
    try:
        start = time.perf_counter()
        assert post(r, c)
        r.stop()
        assert time.perf_counter() - start < 0.1
        assert len(t.images) == 1
    finally:
        t.block.set()
        worker.join(1.0)
    assert not worker.is_alive()


def test_stop_prevents_pending_jobs_and_completed_results_from_becoming_votes():
    r, c, t, sent, _ = make()
    boot(r)
    post(r, c)
    r.step()
    meta, _, _, future = t.images[-1]
    future.set_result(response(meta))
    r.stop()
    r.step()
    assert not sent and len(t.images) == 1


def test_failed_control_does_not_repeatedly_increment_a_degraded_source():
    r, c, t, _, _ = make()
    boot(r)
    t.available = False
    r.step()
    generation = r.generation
    for _ in range(5):
        r.step()
    assert r.generation == generation


@pytest.mark.parametrize("near_edge", [False, True])
def test_transmitted_crop_preserves_the_original_local_model_input(near_edge):
    from manipulation_execution.hand_perception import gesture_crop, gesture_om

    r, c, t, _, _ = make()
    boot(r)
    image = np.empty_like(IMAGE)
    image[:] = (17, 90, 230)  # asymmetric BGR values catch accidental color conversion
    hands = [landmarks(0), landmarks(50)] if near_edge else HANDS
    stamp = int((c.ros() - 0.02) * 1e9)
    assert r.post(stamp, 3, image, BOX, hands, pear_stamp_ns=stamp - 10_000_000)
    r.step()
    plan = gesture_crop.plan_crop(hands, BOX, r.crop, image.shape[:2])
    original = gesture_crop.extract_square(image, plan.x0, plan.y0, plan.side)
    transmitted = t.images[-1][1]
    np.testing.assert_array_equal(gesture_om.preprocess(original), gesture_om.preprocess(transmitted))
