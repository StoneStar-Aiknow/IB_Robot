"""Hand perception for the X2 mirror: head camera + PEAR person box -> HandFrame.

Perception only, no motion output. One HandFrame per processed camera frame,
``header.stamp`` = the camera frame's stamp (the clock PearFrame carries).

Threads: the subscription callbacks only park the newest frame in a single
slot; one worker thread owns the NPU (ACL contexts are per thread), takes the
slot, and publishes. Frames that arrive while the worker is busy are replaced,
so there is never a backlog.

Per frame:
  1. person box = newest PearFrame not later than the frame; it must be a
     ``yolox`` detection with ``confidence >= min_detector_score`` and at most
     ``max_bbox_age_s`` old, else the frame is published with no hands and
     ``person_valid`` false, without inference;
  2. a stamp going backwards or a gap over ``reset_gap_s`` drops the tracked
     hands; a stamp equal to the previous one is published empty;
  3. decode, crop the upper body, run the NPU pipeline on the crop, move the
     keypoints back to source pixels, split into the person's left / right hand,
     curls from the world keypoints, normalised by the calibration file.

The hand pipeline and keypoint helper modules are not part of this package;
they are loaded from ``extern_dir`` at start (see npu_hands.load_extern).
"""

from __future__ import annotations

import collections
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage

from ibrobot_msgs.msg import HandFrame, HandObservation, PearFrame

from .hand_perception import npu_hands
from .imitation_retargeting.x2._vendor.hand_core import CALIB_MEDIAPIPE_PATH

SIDES = ("left", "right")
CAMERA_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)
# Reliable publisher: matches reliable and best-effort subscribers alike
FRAME_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=30,
)
PARAMS = {
    "camera_topic": "/aima/hal/sensor/rgb_head_front_center/rgb_image_half/compressed",
    "pear_frame_topic": "/motion_imitation_stream/pear_frame",
    "frame_topic": "/hri/hand/frame",
    "model_dir": "/root/x2-dexhand-run/models",
    "extern_dir": "/root/x2-dexhand-run/extern",
    "device": 0,
    "rule_labels": False,
    "calib_path": str(CALIB_MEDIAPIPE_PATH),
    "max_bbox_age_s": 0.5,
    "min_detector_score": 0.5,
    "reset_gap_s": 1.0,
    "log_period_s": 5.0,
}


def stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def empty_observation() -> HandObservation:
    o = HandObservation()
    o.detected = False
    o.score = float("nan")
    o.curl_rad = np.full(5, np.nan, dtype=np.float32)
    o.curl_norm = np.full(5, np.nan, dtype=np.float32)
    o.world_landmarks = np.full(63, np.nan, dtype=np.float32)
    o.image_landmarks = np.full(42, np.nan, dtype=np.float32)
    o.gesture, o.gesture_score = "", 0.0
    return o


class Stats:
    """Counters for one log period."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.frames_in = self.frames_out = self.no_person = 0
        self.work_ms, self.stamp_to_pub_ms = [], []
        self.seen = dict.fromkeys(SIDES, 0)

    def line(self, labels: dict, dropped: int, resets: int) -> str:
        dt = max(time.perf_counter() - self.t0, 1e-9)
        inferred = max(1, self.frames_out - self.no_person)

        def pct(xs, q):
            return float(np.percentile(xs, q)) if xs else float("nan")

        lab = " ".join(f"{s}={labels[s][0] or '-'}({labels[s][1]:.2f})" for s in SIDES)
        return (
            f"in {self.frames_in / dt:4.1f} fps, out {self.frames_out / dt:4.1f} fps, dropped {dropped} | "
            f"decode+infer ms p50 {pct(self.work_ms, 50):.1f} p95 {pct(self.work_ms, 95):.1f} | "
            f"stamp->publish ms p50 {pct(self.stamp_to_pub_ms, 50):.1f} max {pct(self.stamp_to_pub_ms, 100):.1f} | "
            f"hand seen left {self.seen['left'] / inferred:.2f} right {self.seen['right'] / inferred:.2f} | "
            f"no person {self.no_person} | tracking resets {resets} | {lab}"
        )


class HandPerceptionNode(Node):
    def __init__(self):
        super().__init__("hand_perception")
        p = {k: self.declare_parameter(k, v).value for k, v in PARAMS.items()}
        self.p = p
        self.keypoints = npu_hands.load_extern(p["extern_dir"], npu_hands.KEYPOINTS_PY)
        calib = json.loads(Path(p["calib_path"]).read_text())
        fingers = self.keypoints.CURL_FINGERS
        # per side: (5,2) [open, fist] in CURL_FINGERS order
        self.calib = {s: np.array([calib["endpoints"][s][f] for f in fingers], dtype=np.float64) for s in SIDES}
        for d, names in ((p["model_dir"], npu_hands.MODEL_FILES), (p["extern_dir"], npu_hands.EXTERN_FILES)):
            for n in names:
                f = Path(d) / n
                self.get_logger().info(f"{f} sha256 {npu_hands.file_sha256(f) if f.is_file() else 'MISSING'}")
        self.get_logger().info(f"calib {p['calib_path']} sha256 {npu_hands.file_sha256(p['calib_path'])}")

        self._slot = None
        self._slot_lock = threading.Lock()
        self._has_frame = threading.Event()
        self._stopping = threading.Event()
        self._ready = threading.Event()
        self._init_error = None
        self._boxes = collections.deque(maxlen=64)
        self._boxes_lock = threading.Lock()
        self._prev_stamp = None
        self.dropped = self.resets = self.frame_count = 0
        self.labels = {s: ("", 0.0) for s in SIDES}
        self.stats = Stats()

        self.pub = self.create_publisher(HandFrame, p["frame_topic"], FRAME_QOS)
        self.create_subscription(PearFrame, p["pear_frame_topic"], self._on_pear, qos_profile_sensor_data)
        self.create_subscription(CompressedImage, p["camera_topic"], self._on_image, CAMERA_QOS)
        self.worker = threading.Thread(target=self._run, name="hand-npu", daemon=True)
        self.worker.start()
        self.get_logger().info(
            f"camera {p['camera_topic']}, person {p['pear_frame_topic']} -> {p['frame_topic']}; "
            f"rule_labels={p['rule_labels']}"
        )

    def wait_ready(self, timeout: float = 120.0) -> None:
        if not self._ready.wait(timeout):
            raise RuntimeError(f"NPU backend not up after {timeout:g} s")
        if self._init_error:
            raise RuntimeError(f"NPU backend failed: {self._init_error}")
        self.get_logger().info(f"{self.backend.description} ready")

    # -- callbacks (executor thread) ----------------------------------------
    def _on_pear(self, msg: PearFrame):
        box = (stamp_ns(msg.header), np.asarray(msg.bbox, dtype=np.float64), msg.source, float(msg.confidence))
        with self._boxes_lock:
            self._boxes.append(box)

    def _on_image(self, msg: CompressedImage):
        stamp = stamp_ns(msg.header)
        with self._slot_lock:
            if self._slot is not None:
                self.dropped += 1
                if stamp < stamp_ns(self._slot[0].header):  # older than the parked one: keep the parked one
                    return
            self._slot = (msg, time.perf_counter())
        self.stats.frames_in += 1
        self._has_frame.set()

    # -- worker thread -------------------------------------------------------
    def _run(self):
        try:
            self.backend = npu_hands.NpuHands(
                self.p["model_dir"], self.p["extern_dir"], int(self.p["device"]), bool(self.p["rule_labels"])
            )
        except Exception as e:  # noqa: BLE001
            self._init_error = f"{type(e).__name__}: {e}"
            self._ready.set()
            return
        self._ready.set()
        try:
            while not self._stopping.is_set():
                if not self._has_frame.wait(0.5):
                    continue
                with self._slot_lock:
                    item, self._slot = self._slot, None
                    self._has_frame.clear()
                if item is None:
                    continue
                try:
                    self._process(*item)
                except Exception as e:  # noqa: BLE001  one bad frame must not stop the stream
                    self.get_logger().error(f"frame dropped: {type(e).__name__}: {e}")
                    self._reset_tracking()
                if not rclpy.ok():
                    break
        finally:
            self.backend.close()

    def _person_box(self, stamp: int):
        with self._boxes_lock:
            older = [b for b in self._boxes if b[0] <= stamp]
        if not older:
            return None, None, False
        ts, box, source, conf = max(older, key=lambda b: b[0])
        age = stamp - ts
        valid = (
            source == "yolox"
            and conf >= float(self.p["min_detector_score"])
            and age <= float(self.p["max_bbox_age_s"]) * 1e9
            and bool(np.all(np.isfinite(box)))
        )
        return box, age, valid

    def _reset_tracking(self):
        self.backend.reset()
        self.resets += 1
        self._prev_stamp = None

    def _process(self, msg: CompressedImage, t_arrival: float):
        import cv2

        stamp = stamp_ns(msg.header)
        box, age, valid = self._person_box(stamp)
        hands, n_hands, decode_ms, infer_ms = dict.fromkeys(SIDES), 0, 0.0, 0.0
        if valid:
            prev = self._prev_stamp
            if prev is not None and (stamp < prev or stamp - prev > float(self.p["reset_gap_s"]) * 1e9):
                self._reset_tracking()
            if stamp != self._prev_stamp:
                t = time.perf_counter()
                img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                decode_ms = (time.perf_counter() - t) * 1000
                if img is None:
                    self.get_logger().warning("JPEG decode failed; frame published without hands")
                else:
                    t = time.perf_counter()
                    hands, n_hands = self._hands(img, box)
                    infer_ms = (time.perf_counter() - t) * 1000
                    self._prev_stamp = stamp
        out = HandFrame()
        out.header = msg.header
        out.frame_id = self.frame_count
        out.timestamp_ns = stamp
        out.left = self._observation("left", hands["left"])
        out.right = self._observation("right", hands["right"])
        out.person_valid = bool(valid)
        out.person_bbox_xyxy = box if box is not None else np.full(4, np.nan)
        out.person_bbox_age_ms = float(age / 1e6) if age is not None else float("nan")
        out.hands_in_frame = int(n_hands)
        out.decode_ms, out.infer_ms = float(decode_ms), float(infer_ms)
        out.latency_ms = float((time.perf_counter() - t_arrival) * 1000)
        self.pub.publish(out)
        self.frame_count += 1
        self._account(out, valid, (time.time_ns() - stamp) / 1e6)

    def _hands(self, img, box):
        """Upper-body crop -> NPU -> {side: hand or None}, number of hands found."""
        kp = self.keypoints
        h, w = img.shape[:2]
        x0, y0, x1, y1 = kp.upper_body_box(box, w, h)
        crop = np.ascontiguousarray(img[y0:y1, x0:x1])
        found = self.backend.infer(crop, (box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0))
        for f in found:
            f["lm"] = f["lm"] + (x0, y0)  # crop -> source pixels
        pick = kp.assign_sides([f["lm"] for f in found], 0.5 * (box[0] + box[2]))
        return {s: None if pick[s] is None else found[pick[s]] for s in SIDES}, len(found)

    def _observation(self, side: str, hand) -> HandObservation:
        if hand is None:
            return empty_observation()
        curl = self.keypoints.finger_curls(hand["wlm"])
        if not np.all(np.isfinite(curl)):  # degenerate keypoints
            return empty_observation()
        ends = self.calib[side]
        o = HandObservation()
        o.detected = True
        o.score = float(hand["handed_score"])
        o.curl_rad = curl.astype(np.float32)
        o.curl_norm = ((curl - ends[:, 0]) / (ends[:, 1] - ends[:, 0])).astype(np.float32)
        o.world_landmarks = np.asarray(hand["wlm"], dtype=np.float32).ravel()
        o.image_landmarks = np.asarray(hand["lm"], dtype=np.float32).ravel()
        o.gesture, o.gesture_score = hand["gesture"], float(hand["gesture_score"])
        return o

    def _account(self, out: HandFrame, valid: bool, stamp_to_pub_ms: float):
        st = self.stats
        st.frames_out += 1
        st.stamp_to_pub_ms.append(stamp_to_pub_ms)
        if valid:
            st.work_ms.append(out.decode_ms + out.infer_ms)
            for s in SIDES:
                o = getattr(out, s)
                if o.detected:
                    st.seen[s] += 1
                    self.labels[s] = (o.gesture, o.gesture_score)
        else:
            st.no_person += 1
        period = float(self.p["log_period_s"])
        if period > 0 and time.perf_counter() - st.t0 >= period:
            self.get_logger().info(st.line(self.labels, self.dropped, self.resets))
            self.stats = Stats()

    def destroy_node(self):
        self._stopping.set()
        self._has_frame.set()
        if self.worker.is_alive() and self.worker is not threading.current_thread():
            self.worker.join(timeout=2.0)
        super().destroy_node()


def main(argv=None):
    # No rclpy signal handler: Ctrl-C becomes KeyboardInterrupt here, so the worker
    # is stopped before the context goes invalid (else it races a dead publisher)
    rclpy.init(args=list(sys.argv[1:] if argv is None else argv), signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = HandPerceptionNode()
        node.wait_ready()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
