"""hand_perception_node with the gesture model: off is the baseline, on runs after the HandFrame, faults stay silent.

Real rclpy, the node's real worker thread and the real gesture thread; the hand
NPU pipeline and the gesture om session are fakes. Publishers are swapped for
recorders so a test sees exactly what each thread sent. ``feed`` waits for the
gesture run a frame posted (bounded), so the runs are not replaced by the next
frame unless a test wants that.
"""

import json
import os
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import CompressedImage

from ibrobot_msgs.msg import PearFrame
from manipulation_execution import hand_perception_node as module
from manipulation_execution.hand_perception import gesture_om

MS = 1_000_000
WAIT_S = 3.0  # upper bound for any wait on a thread; the tests pass in far less
W, H = 640, 480
BOX = (150.0, 20.0, 490.0, 470.0)  # 450 px high
JPEG = cv2.imencode(".jpg", np.full((H, W, 3), 90, np.uint8))[1].tobytes()
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def hand_lm(cx, cy, size=40.0):
    lm = np.zeros((21, 2))
    lm[0] = (0.0, size / 2)
    lm[9] = (0.0, -size / 2)
    lm -= lm.mean(axis=0)
    return lm + (cx, cy)


class FakeHands:
    """The NPU hand pipeline: the hands it returns are set by the test."""

    description = "fake hands"

    def __init__(self):
        self.lms = [hand_lm(280, 250), hand_lm(360, 250)]

    def infer(self, crop, _box):
        return [
            {"lm": lm.copy(), "wlm": np.zeros((21, 3)), "handed_score": 0.9, "gesture": "", "gesture_score": 0.0}
            for lm in self.lms
        ]

    def reset(self):
        pass

    def close(self):
        pass


KEYPOINTS = SimpleNamespace(
    CURL_FINGERS=FINGERS,
    upper_body_box=lambda _box, w, h: (0, 0, w, h),
    assign_sides=lambda lms, _cx: {"left": 0 if lms else None, "right": 1 if len(lms) > 1 else None},
    finger_curls=lambda _wlm: np.full(5, 0.5),
)


class Session:
    """The gesture om's ais_bench session."""

    instances = []
    open_error = None
    infer_error = None
    delay_s = 0.0
    block = None  # an Event: infer waits on it (bounded); entered is set once it does
    entered = None

    def __init__(self, device, path):
        if Session.open_error:
            raise Session.open_error
        self.device, self.path, self.calls, self.freed, self.warmed = device, path, 0, False, False
        self.thread = threading.current_thread().name
        Session.instances.append(self)

    def infer(self, feeds, mode=None):
        if not self.warmed:  # GestureOm's warm-up run while loading: not one of the runs the tests count
            self.warmed = True
            return [np.zeros(gesture_om.OUTPUT_SHAPE, np.float32)]
        self.calls += 1
        if Session.delay_s:
            time.sleep(Session.delay_s)
        if Session.block is not None:
            Session.entered.set()
            assert Session.block.wait(WAIT_S), "test never released the session"
        if Session.infer_error:
            raise Session.infer_error
        y = np.zeros(gesture_om.OUTPUT_SHAPE, np.float32)
        y[0, :4, 3] = (50, 60, 200, 210)
        y[0, 4 + 2, 3] = 0.85  # holy
        return [y]

    def free_resource(self):
        self.freed = True


class Recorder:
    def __init__(self, on_publish=None):
        self.msgs, self.on_publish = [], on_publish

    def publish(self, msg):
        self.msgs.append(msg)
        if self.on_publish:
            self.on_publish(len(self.msgs))


@pytest.fixture
def env(monkeypatch, tmp_path):
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps({"endpoints": {s: {f: [0.0, 1.0] for f in FINGERS} for s in ("left", "right")}}))
    fake = FakeHands()
    monkeypatch.setattr(
        module,
        "npu_hands",
        SimpleNamespace(
            load_extern=lambda *_args: KEYPOINTS,
            model_paths=lambda _model_dir: {},
            extern_path=lambda _extern_dir, name: tmp_path / name,
            file_sha256=lambda _path: "stub",
            EXTERN_FILES=(),
            CALIB_MEDIAPIPE_JSON="calib.json",
            KEYPOINTS_PY="keypoints.py",
            NpuHands=lambda *_args: fake,
        ),
    )
    Session.instances = []
    monkeypatch.setattr(Session, "open_error", None)
    monkeypatch.setattr(Session, "infer_error", None)
    monkeypatch.setattr(Session, "delay_s", 0.0)
    monkeypatch.setattr(Session, "block", None)
    monkeypatch.setattr(Session, "entered", threading.Event())
    monkeypatch.setattr(gesture_om, "_open_session", lambda device, path: Session(device, path))
    model = tmp_path / "yolov10n_gestures_320_cut_v1.om"
    model.write_bytes(b"om")
    rclpy.init()
    nodes = []

    def make(**params):
        params.setdefault("calib_path", str(calib))
        params.setdefault("backend", "ais_bench")
        params.setdefault("log_period_s", 0.0)
        node = module.HandPerceptionNode(parameter_overrides=[Parameter(k, value=v) for k, v in params.items()])
        nodes.append(node)
        node.wait_ready(5.0)
        node.pub = Recorder()
        node.posted, node.ran = 0, threading.Semaphore(0)
        g = node._gesture
        if g is not None and not getattr(g, "is_remote", False):
            node.gesture_pub = Recorder()
            assert wait_for(lambda: g.state != "loading"), "the gesture thread did not load"
            post, run = g.post, g.run

            def posted(*args):
                ok = post(*args)
                node.posted += ok
                return ok

            def ran(job):
                try:
                    return run(job)
                finally:
                    node.ran.release()

            g.post, g.run = posted, ran
        done = threading.Semaphore(0)
        process = node._process

        def wrapped(*args):
            try:
                process(*args)
            finally:
                done.release()

        node._process = wrapped
        node.processed = done
        return node

    yield SimpleNamespace(make=make, model=str(model), hands=fake)
    for node in nodes:
        node.destroy_node()
    rclpy.shutdown()


def image(stamp):
    msg = CompressedImage()
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
    msg.format, msg.data = "jpeg", JPEG
    return msg


def pear(stamp, actor_id=7):
    msg = PearFrame()
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
    msg.source, msg.actor_id, msg.confidence = "yolox", actor_id, 0.9
    msg.bbox = np.asarray(BOX, np.float32)
    return msg


def wait_for(pred, timeout=WAIT_S):
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.002)
    return True


def feed(node, stamp, actor_id=7, n=1, wait_gesture=True):
    node._on_pear(pear(stamp, actor_id))
    node._on_image(image(stamp))
    for _ in range(n):
        assert node.processed.acquire(timeout=5.0), "the worker did not process the frame"
    posted, node.posted = node.posted, 0
    for _ in range(posted if wait_gesture else 0):
        assert node.ran.acquire(timeout=WAIT_S), "the gesture thread did not run the posted frame"


def run_frames(node, count, step_ms=100, base=1_000_000_000_000, wait_gesture=True):
    for i in range(count):
        feed(node, base + i * step_ms * MS, wait_gesture=wait_gesture)
    return base


def gestures(node):
    return [json.loads(m.data) for m in node.gesture_pub.msgs] if node.gesture_pub is not None else []


def frame_fields(msg):
    return (
        msg.frame_id, msg.timestamp_ns, msg.person_valid, list(msg.person_bbox_xyxy), msg.hands_in_frame,
        msg.left.detected, list(msg.left.image_landmarks), msg.right.detected, list(msg.right.curl_norm),
    )  # fmt: skip


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"gesture_enabled": False, "gesture_model_path": "MODEL"},
        {"gesture_enabled": True, "gesture_model_path": ""},
        {"gesture_enabled": True, "gesture_model_path": "  "},
    ],
)
def test_off_builds_nothing_and_publishes_the_baseline(env, params):
    params = {k: (env.model if v == "MODEL" else v) for k, v in params.items()}
    node = env.make(**params)
    assert node._gesture is None and node.gesture_pub is None
    assert node.count_publishers("/hri/hand/gesture") == 0
    run_frames(node, 4)
    assert len(node.pub.msgs) == 4 and all(m.hands_in_frame == 2 for m in node.pub.msgs)
    assert Session.instances == []  # no model opened


def test_on_and_off_publish_the_same_hand_frames(env):
    off = env.make()
    run_frames(off, 3)
    on = env.make(gesture_enabled=True, gesture_model_path=env.model)
    run_frames(on, 3)
    assert [frame_fields(m) for m in off.pub.msgs] == [frame_fields(m) for m in on.pub.msgs]
    assert len(gestures(on)) == 3


def test_on_runs_after_the_hand_frame_and_sends_v1(env):
    node = env.make(gesture_enabled=True, gesture_model_path=env.model, device=0)
    assert node.count_publishers("/hri/hand/gesture") == 1
    order = []
    node.pub.on_publish = lambda _n: order.append("hand")
    node.gesture_pub.on_publish = lambda _n: order.append("gesture")
    base = run_frames(node, 2)
    assert order == ["hand", "gesture", "hand", "gesture"]
    (session,) = Session.instances
    assert session.thread == "hand-gesture" and session.path == env.model
    g = gestures(node)[0]
    assert g["schema"] == "x2_hand_gesture/v1" and g["stamp_ns"] == base and g["actor_id"] == 7
    assert g["cls"] == "holy" and g["score"] == pytest.approx(0.85) and g["crop"] == "both"
    x0, y0, x1, y1 = g["roi"]
    assert x1 - x0 == y1 - y0 > 64


def test_gesture_topic_and_rates_are_parameters(env):
    node = env.make(gesture_enabled=True, gesture_model_path=env.model, gesture_topic="/t/g", gesture_max_rate_hz=5.0)
    assert node.count_publishers("/t/g") == 1
    run_frames(node, 6, step_ms=50)
    assert [g["stamp_ns"] for g in gestures(node)] == [1_000_000_000_000 + i * 50 * MS for i in (0, 4)]


def test_no_hands_or_no_person_does_not_run(env):
    node = env.make(gesture_enabled=True, gesture_model_path=env.model)
    env.hands.lms = []
    run_frames(node, 2)
    node._on_image(image(5_000_000_000_000))  # no PearFrame that old: person invalid
    assert node.processed.acquire(timeout=5.0)
    assert len(node.pub.msgs) == 3 and gestures(node) == [] and Session.instances[0].calls == 0


def test_mixed_backend_is_off_unless_allowed(env):
    node = env.make(gesture_enabled=True, gesture_model_path=env.model, backend="inference_service")
    assert node._gesture is None and node.gesture_pub is None
    node = env.make(
        gesture_enabled=True, gesture_model_path=env.model, backend="inference_service",
        gesture_allow_mixed_backend=True,
    )  # fmt: skip
    run_frames(node, 1)
    assert len(gestures(node)) == 1


@pytest.mark.parametrize(
    "params",
    [
        {"gesture_k": -1.0},
        {"gesture_k": "big"},
        {"gesture_slow_limit": 3.5},
        {"gesture_slow_ms": "fast"},
        {"gesture_enabled": "true"},
        {"gesture_enabled": 1},
        {"gesture_model_path": 5},
        {"gesture_topic": "bad topic"},
        {"gesture_topic": 7},
    ],
)
@pytest.mark.parametrize("enabled", [True, False])
def test_bad_gesture_parameter_is_off_not_a_crash(env, params, enabled):
    params = {"gesture_enabled": enabled, "gesture_model_path": env.model, **params}
    node = env.make(**params)  # wrong types too: the gesture_* parameters are declared untyped
    assert node._gesture is None and node.gesture_pub is None
    assert node.count_publishers("/hri/hand/gesture") == 0
    run_frames(node, 2)
    assert len(node.pub.msgs) == 2 and Session.instances == []


def test_a_number_given_as_an_integer_is_fine(env):
    node = env.make(gesture_enabled=True, gesture_model_path=env.model, gesture_slow_ms=20, gesture_k=2)
    run_frames(node, 1)
    assert len(gestures(node)) == 1


def test_load_failure_turns_it_off_and_hands_go_on(env, monkeypatch):
    monkeypatch.setattr(Session, "open_error", RuntimeError("acl 507011"))
    node = env.make(gesture_enabled=True, gesture_model_path=env.model)
    run_frames(node, 3)
    assert node._gesture is None and len(node.pub.msgs) == 3 and gestures(node) == []


def test_e2e_model_is_refused_at_load(env, tmp_path):
    e2e = tmp_path / "yolov10n_gestures_320_e2e.om"
    e2e.write_bytes(b"om")
    node = env.make(gesture_enabled=True, gesture_model_path=str(e2e))
    run_frames(node, 1)
    assert node._gesture is None and Session.instances == [] and len(node.pub.msgs) == 1


def test_inference_errors_five_in_a_row_turn_it_off_and_hands_go_on(env, monkeypatch):
    monkeypatch.setattr(Session, "infer_error", RuntimeError("aclmdlExecute 100000"))
    node = env.make(gesture_enabled=True, gesture_model_path=env.model)
    run_frames(node, 8)
    (session,) = Session.instances
    assert session.calls == 5 and wait_for(lambda: session.freed)
    assert node._gesture is None and len(node.pub.msgs) == 8 and gestures(node) == []
    assert [m.frame_id for m in node.pub.msgs] == list(range(8))


def test_three_slow_runs_turn_it_off_and_hands_go_on(env, monkeypatch):
    monkeypatch.setattr(Session, "delay_s", 0.025)
    node = env.make(gesture_enabled=True, gesture_model_path=env.model, gesture_slow_ms=20.0)
    run_frames(node, 6)
    assert Session.instances[0].calls == 3 and wait_for(lambda: Session.instances[0].freed)
    assert node._gesture is None and len(node.pub.msgs) == 6 and len(gestures(node)) == 3


def test_slow_runs_spread_over_the_window_turn_it_off(env, monkeypatch):
    node = env.make(gesture_enabled=True, gesture_model_path=env.model, gesture_slow_ms=20.0)
    for i in range(5):  # slow, fast, slow, fast, slow: 3 of the last 5
        monkeypatch.setattr(Session, "delay_s", 0.025 if i % 2 == 0 else 0.0)
        feed(node, 1_000_000_000_000 + i * 100 * MS)
    run_frames(node, 2, base=1_000_600_000_000)
    assert node._gesture is None and len(gestures(node)) == 5 and len(node.pub.msgs) == 7


def test_hand_frames_keep_flowing_while_the_gesture_run_is_blocked(env, monkeypatch):
    monkeypatch.setattr(Session, "block", threading.Event())
    node = env.make(gesture_enabled=True, gesture_model_path=env.model, gesture_watchdog_ms=5000.0)
    try:
        feed(node, 1_000_000_000_000, wait_gesture=False)
        assert Session.entered.wait(WAIT_S)
        t0 = time.perf_counter()
        run_frames(node, 10, base=1_000_100_000_000, wait_gesture=False)
        per_frame = (time.perf_counter() - t0) / 10
        assert len(node.pub.msgs) == 11 and Session.instances[0].calls == 1 and gestures(node) == []
        assert per_frame < 0.1, per_frame  # JPEG decode + fake hands + a crop copy; nothing waits on the om
        assert node._gesture is not None and node._gesture.enabled
    finally:
        Session.block.set()
    assert wait_for(lambda: len(gestures(node)) == 2)  # the blocked run, then the latest posted frame
    assert gestures(node)[-1]["stamp_ns"] == 1_000_100_000_000 + 9 * 100 * MS


def test_the_watchdog_turns_a_hung_run_off_and_hands_go_on(env, monkeypatch):
    monkeypatch.setattr(Session, "block", threading.Event())
    node = env.make(gesture_enabled=True, gesture_model_path=env.model, gesture_watchdog_ms=50.0)
    try:
        feed(node, 1_000_000_000_000, wait_gesture=False)
        assert Session.entered.wait(WAIT_S)
        time.sleep(0.08)  # the run is now past the 50 ms watchdog
        run_frames(node, 5, base=1_000_100_000_000, wait_gesture=False)
        assert node._gesture is None and len(node.pub.msgs) == 6
        assert [m.frame_id for m in node.pub.msgs] == list(range(6))
    finally:
        Session.block.set()
    assert wait_for(lambda: Session.instances[0].freed)  # it came back late: freed, nothing sent
    assert gestures(node) == [] and Session.instances[0].calls == 1


def test_a_raising_runner_is_dropped_silently(env):
    node = env.make(gesture_enabled=True, gesture_model_path=env.model)

    def boom(*_a, **_k):
        raise RuntimeError("bug")

    node._gesture.post = boom
    run_frames(node, 3)
    assert node._gesture is None and len(node.pub.msgs) == 3


def test_a_newer_frame_waiting_skips_the_gesture_only_down_to_the_busy_rate(env):
    node = env.make(gesture_enabled=True, gesture_model_path=env.model)
    base = 1_000_000_000_000
    feed(node, base)
    newer = {}

    def park(n):
        if n in newer:  # a newer frame arrives while this one is being published
            stamp = newer.pop(n)
            node._on_pear(pear(stamp))
            node._on_image(image(stamp))

    node.pub.on_publish = park
    newer[2] = base + 100 * MS
    feed(node, base + 50 * MS, n=2)  # busy, 50 ms after the last run: the hands go first
    newer[4] = base + 230 * MS
    feed(node, base + 200 * MS, n=2)  # busy, 100 ms after the last run: runs anyway (230: 15 Hz cap)
    ms = [(m.timestamp_ns - base) // MS for m in node.pub.msgs]
    assert ms == [0, 50, 100, 200, 230]
    assert [(g["stamp_ns"] - base) // MS for g in gestures(node)] == [0, 100, 200]


def test_actor_id_comes_from_the_pear_frame_the_box_came_from(env):
    node = env.make(gesture_enabled=True, gesture_model_path=env.model)
    feed(node, 1_000_000_000_000, actor_id=3)
    feed(node, 1_000_100_000_000, actor_id=11)
    assert [g["actor_id"] for g in gestures(node)] == [3, 11]


def _find(tree, key):
    if isinstance(tree, dict):
        if key in tree:
            yield tree
        for v in tree.values():
            yield from _find(v, key)
    elif isinstance(tree, list):
        for v in tree:
            yield from _find(v, key)


def test_deployed_yaml_has_the_gesture_off_with_node_types():
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[2] / "robot_config/config/robots/aimdk_x2_voice_mirror_all.yaml"
    (params,) = list(_find(yaml.safe_load(path.read_text()), "gesture_model_path"))
    gesture = {k: v for k, v in params.items() if k.startswith("gesture_")}
    assert gesture == {k: v for k, v in module.PARAMS.items() if k.startswith("gesture_")}
    for k, v in gesture.items():  # declared untyped, so a wrong type is only gesture off: keep them right
        assert type(v) is type(module.PARAMS[k]), k
    assert params["gesture_enabled"] is False and params["gesture_model_path"] == ""
