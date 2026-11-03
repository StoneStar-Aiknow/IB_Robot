"""The X2 head RGB-D camera as the grasp pipeline's observation source.

A capture pairs each of the newest colour images with the depth image and
calibrations nearest it in time, and takes the newest such set that has
image-time joint feedback and TF; it registers its depth into the colour
camera on the planning host's GPU (Warp): registering every frame on the Orin
CPU took ~0.4 s each and left observations 0.6-0.8 s old when a request came
(2026-10-08). The camera pose is the ``torso_link`` TF at the image time.

Images, joint feedback and TF are taken when a capture needs them
(:mod:`robot_runtime.on_demand`), not by an executor:
run as callbacks they kept the idle pipeline's GIL busy enough on the Orin that
feedback and TF reached it late and no image could be associated. Each waits in
its reader queue (the newest few sets; as much feedback as the history holds;
seconds of TF) and a capture drains them first. The images are received in a
DDS participant of their own (a separate rclpy context), so their fragments
reach no other reader.

Times are source times: the HAL, the Orin and the camera share one PTP clock,
so image, TF and feedback stamps compare directly.

Migrated from the X2 pipeline's observation node (on-demand capture), its
observation adapter and the task node's capture.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

import numpy as np

from manipulation_execution.grasp_pipeline.contracts import Flag, GraspFailure, Observation

from .feedback import FeedbackHistory
from .frames import ImageTransform
from .observation import AdmittedState, LiveState, RegisteredObservation, copy_feedback
from .registration import CameraCalibration, calibration_signature, register_rgbd
from .revalidation import ReobserveRequired, RevalidationConfig, require_fresh
from .types import RGBDFrame

CAMERA = (
    ("camera.rgbd_head_front.color", "sensor_msgs/msg/Image"),
    ("camera.rgbd_head_front.depth", "sensor_msgs/msg/Image"),
    ("camera.rgbd_head_front.color_info", "sensor_msgs/msg/CameraInfo"),
    ("camera.rgbd_head_front.depth_info", "sensor_msgs/msg/CameraInfo"),
)
BASE_FRAME = "torso_link"
RECENT_SETS = 4
#: Samples of each camera stream kept between captures: enough for the newest
#: sets while a stream's newest part has not arrived yet.
IMAGE_QUEUE = RECENT_SETS + 2
SYNC_SLOP_S = 0.05
# A capture serves the newest of the last few sets that has image-time
# feedback and TF, waiting up to this long for one. The 500 Hz feedback can
# lose samples when its thread stalls, and TF can trail the newest image
# (live 2026-10-09: feedback gaps > 100 ms, TF 34-70 ms behind); sets are
# ~33 ms apart, and the planner still gates scene age. Live 2026-11-03 (48
# picks, 310P robot side): feedback reaching the Orin stopped for ~400 ms at
# times (nearest sample 356-372 ms before the image) and the vendor HAL for
# 120-190 ms; 0.2 s failed 5 picks before any motion. Feedback that arrives
# unhealthy or out of sync is waited out the same way.
CAPTURE_WAIT_S = 0.6
#: How long a capture waits for a feedback sample inside the freshness limit
#: before it reports the stale one (five 50 Hz runtime views), and how often it looks.
FEEDBACK_WAIT_S = 0.1
FEEDBACK_POLL_S = 0.005
OBSERVATION_UNAVAILABLE = "OBSERVATION_UNAVAILABLE"


def stamp_ns(header) -> int:
    return header.stamp.sec * 10**9 + header.stamp.nanosec


def image_to_rgb(message) -> np.ndarray:
    """Convert a ROS image into contiguous uint8 RGB."""

    encoding = message.encoding.lower()
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(encoding)
    if channels is None:
        raise ValueError(f"Unsupported RGB encoding: {message.encoding}")
    if message.step < message.width * channels:
        raise ValueError("RGB image step is smaller than packed row size")
    image = np.ndarray(
        (message.height, message.width, channels),
        dtype=np.uint8,
        buffer=bytes(message.data),
        strides=(message.step, channels, 1),
    )
    if encoding in ("bgr8", "bgra8"):
        image = image[:, :, [2, 1, 0]]
    else:
        image = image[:, :, :3]
    return np.ascontiguousarray(image, dtype=np.uint8)


def image_to_depth(message) -> np.ndarray:
    """Convert one supported ROS depth encoding without changing units."""

    encoding = message.encoding.lower()
    if encoding not in ("16uc1", "mono16", "32fc1", "64fc1"):
        raise ValueError(f"Unsupported depth encoding: {message.encoding}")
    native_dtype = {
        "16uc1": np.dtype(np.uint16),
        "mono16": np.dtype(np.uint16),
        "32fc1": np.dtype(np.float32),
        "64fc1": np.dtype(np.float64),
    }[encoding]
    dtype = native_dtype.newbyteorder(">" if message.is_bigendian else "<")
    if message.step < message.width * dtype.itemsize:
        raise ValueError("Depth image step is smaller than packed row size")
    depth = np.ndarray(
        (message.height, message.width),
        dtype=dtype,
        buffer=bytes(message.data),
        strides=(message.step, dtype.itemsize),
    )
    return np.asarray(depth, dtype=native_dtype).copy()


def effective_depth_scale(message, configured_scale: float) -> float:
    """Use encoding-specific defaults when configured scale is non-positive."""

    if configured_scale > 0:
        return float(configured_scale)
    if message.encoding.lower() in ("16uc1", "mono16"):
        return 0.001
    if message.encoding.lower() in ("32fc1", "64fc1"):
        return 1.0
    raise ValueError(f"Unsupported depth encoding: {message.encoding}")


def transform_matrix(transform) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    rotation, translation = transform.rotation, transform.translation
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).as_matrix()
    matrix[:3, 3] = [translation.x, translation.y, translation.z]
    return matrix


def lookup_matrix(buffer, target: str, source: str, stamp: int) -> np.ndarray:
    """``target_from_source`` at ``stamp``, without waiting: the capture polls instead."""
    if target == source:
        return np.eye(4)
    from rclpy.time import Time

    return transform_matrix(buffer.lookup_transform(target, source, Time(nanoseconds=stamp)).transform)


#: Capture timings kept in HeadCamera.calls.
CALLS_KEPT = 32
#: How often a wait for the GPU looks at its deadline and cancel flag.
GPU_POLL_S = 0.02


@contextmanager
def gpu_section(lock, *, deadline: float = float("inf"), cancelled: Flag = lambda: False, code: str = "SKILL_TIMEOUT"):
    """Hold the head's GPU lock (see :func:`aimdk_grasp.backend.head_parts`), waiting no longer than ``deadline``.

    A plan or refine holds it for seconds; a capture or scene stage waiting
    behind one gives up at its own deadline (a ``GraspFailure`` with ``code``)
    or on cancel instead of returning long after its caller stopped waiting.
    """
    acquire = getattr(lock, "acquire", None)
    if acquire is None:  # nullcontext: nothing else uses the GPU
        with lock:
            yield
        return
    while not acquire(timeout=GPU_POLL_S):
        if cancelled():
            raise GraspFailure("CANCELED", "cancelled waiting for the GPU")
        if time.monotonic() >= deadline:
            raise GraspFailure(code, "the GPU was busy (planning) past the deadline")
    try:
        yield
    finally:
        lock.release()


def register_set(
    rgb, depth, rgb_info, depth_info, rgb_from_depth, *, depth_scale=-1.0, max_skew_s=0.05, device="cuda"
) -> tuple[RGBDFrame, str]:
    """Register depth into the RGB camera; returns the frame and its calibration signature."""
    stamp = stamp_ns(rgb.header)
    if stamp <= 0:
        raise ValueError("RGB source timestamp required")
    for message in (depth, rgb_info, depth_info):
        if stamp_ns(message.header) <= 0 or abs(stamp_ns(message.header) - stamp) / 1e9 > max_skew_s:
            raise ValueError("RGB-D/calibration source timestamps exceed synchronization tolerance")
    calibrations = []
    for image, info in ((rgb, rgb_info), (depth, depth_info)):
        if image.header.frame_id != info.header.frame_id or (image.width, image.height) != (info.width, info.height):
            raise ValueError("Image frame/dimensions do not match CameraInfo")
        calibrations.append(
            CameraCalibration(
                info.width,
                info.height,
                np.asarray(info.k).reshape(3, 3),
                np.asarray(info.d),
                info.header.frame_id,
                info.distortion_model,
            )
        )
    rc, dc = calibrations
    matrix = np.asarray(rgb_from_depth, dtype=float).reshape(4, 4)
    frame = register_rgbd(
        image_to_rgb(rgb),
        image_to_depth(depth),
        rc,
        dc,
        matrix,
        depth_scale=effective_depth_scale(depth, depth_scale),
        device=device,
    )
    return frame, calibration_signature(rc, dc, matrix)


def _bound(interfaces: Mapping[str, Any], name: str, message_type: str) -> Mapping[str, Any]:
    interface = interfaces.get(name) or {}
    shape = (interface.get("kind"), interface.get("direction"), interface.get("message_type"))
    if shape != ("topic", "publish", message_type):
        raise ValueError(f"interface description must publish {name} as a {message_type} topic, got {shape}")
    return interface


class HeadCamera:
    """``ObservationSource`` over the runtime's head camera interfaces; fills ``history`` as it runs.

    A capture is admitted as the X2 pipeline admitted a request: the newest
    registered image and the current feedback, both fresh by ``revalidation``
    on the source ``clock`` (default the node's), with one more capture when
    the first is just stale. Call :meth:`close` before the node is destroyed.
    """

    def __init__(
        self,
        node,
        description: Mapping[str, Any],
        history: FeedbackHistory,
        *,
        device: str = "cuda",
        depth_scale: float = -1.0,
        calibration_verified: bool = True,
        revalidation: RevalidationConfig = RevalidationConfig(),
        clock: Callable[[], float] | None = None,
        gpu_lock: AbstractContextManager | None = None,
    ):
        """``gpu_lock`` is held while a capture registers on the GPU; see :func:`aimdk_grasp.backend.head_parts`."""
        import rclpy
        from rclpy.context import Context
        from sensor_msgs.msg import CameraInfo, Image

        from robot_config.contract_utils import qos_profile_from_dict
        from robot_runtime.on_demand import OnDemandSubscriptions

        from .feedback_ros import subscribe_feedback
        from .lazy_transforms import LazyTransforms

        interfaces = description.get("interfaces") or {}
        bound = [_bound(interfaces, name, message_type) for name, message_type in CAMERA]
        self._node, self.history = node, history
        self._device, self._depth_scale, self._verified = device, float(depth_scale), bool(calibration_verified)
        self._gpu = nullcontext() if gpu_lock is None else gpu_lock
        self._streams = [deque(maxlen=IMAGE_QUEUE) for _ in CAMERA]
        self._images_lock = threading.Lock()
        self.revalidation = revalidation
        self.clock = clock or (lambda: node.get_clock().now().nanoseconds * 1e-9)
        # The newest captures' timings (the watch captures too, ~7 Hz while it runs).
        self.calls: deque[dict] = deque(maxlen=CALLS_KEPT)
        self._side = OnDemandSubscriptions(node.context, node.get_name() + "_feedback")
        self._tf = LazyTransforms(self._side)
        subscribe_feedback(self._side, description, history)
        # The images get a DDS participant of their own: every best-effort
        # reader that shares one with them is handed each fragment of each
        # frame and rejects it (live 2026-10-21: ~2000 warnings/s, and the
        # camera's raw streams 0.4-0.6 s late for every subscriber on the Orin).
        self._camera_context = Context()
        # Not the process's arguments: a launch's -r __node:= would name its nodes like the owner.
        rclpy.init(args=[], context=self._camera_context, domain_id=node.context.get_domain_id())
        self._camera = OnDemandSubscriptions(self._camera_context, node.get_name() + "_camera")
        kinds = (Image, Image, CameraInfo, CameraInfo)
        for kind, interface, stream in zip(kinds, bound, self._streams, strict=True):
            self._camera.subscribe(
                kind,
                str(interface["endpoint"]),
                qos_profile_from_dict(interface.get("qos")),
                IMAGE_QUEUE,
                lambda message, _received, stream=stream: stream.append(message),
            )

    def recent_sets(self, profile: dict | None = None) -> list[tuple]:
        """The newest few (colour, depth, colour info, depth info) sets, newest first, parts within SYNC_SLOP_S.

        ``profile`` accumulates the camera drain's time and sample count.
        """
        with self._images_lock:
            began = time.monotonic()
            taken = self._camera.drain()
            if profile is not None:
                profile["camera_drain_s"] = profile.get("camera_drain_s", 0.0) + time.monotonic() - began
                profile["camera_taken"] = profile.get("camera_taken", 0) + taken
            colour, *others = (list(stream) for stream in self._streams)
        sets = []
        for rgb in reversed(colour):
            stamp = stamp_ns(rgb.header)
            parts = [
                min(stream, key=lambda message: abs(stamp_ns(message.header) - stamp), default=None)
                for stream in others
            ]
            if all(part is not None and abs(stamp_ns(part.header) - stamp) <= SYNC_SLOP_S * 1e9 for part in parts):
                sets.append((rgb, *parts))
                if len(sets) == RECENT_SETS:
                    break
        return sets

    def capture(self, *, deadline: float, cancelled: Flag) -> Observation:
        started, recaptured = time.monotonic(), False
        state = self.capture_state(deadline=deadline, cancelled=cancelled)
        try:
            require_fresh(state, self.clock(), self.revalidation)
        except ReobserveRequired:
            # A feedback sample just late (live 2026-10-14: 141 ms against the
            # 100 ms limit, 3 of 40 requests) failed the request before any
            # perception. capture_state now waits for a fresh sample first, so
            # this is left for an image past its age. One more capture; limits
            # are unchanged.
            state = self.capture_state(deadline=deadline, cancelled=cancelled)
            recaptured = True
            try:
                require_fresh(state, self.clock(), self.revalidation)
            except ReobserveRequired as exc:
                raise GraspFailure(OBSERVATION_UNAVAILABLE, str(exc), retryable=True) from None
        admitted = AdmittedState(state, self.clock(), started)
        registered = state.observation
        stamp = registered.transform.stamp_ns
        # This capture's own entry: the watch captures on another thread meanwhile.
        own = next((call for call in reversed(self.calls) if call.get("stamp_ns") == stamp), {})
        report = dict(own, capture_s=time.monotonic() - started, recaptured=recaptured)
        k = registered.frame.intrinsics
        return Observation(
            rgb=registered.frame.rgb,
            depth_m=registered.frame.depth,
            intrinsics=np.array([[k.fx, 0.0, k.cx], [0.0, k.fy, k.cy], [0.0, 0.0, 1.0]]),
            frame_id=registered.frame.frame_id,
            stamp_ns=registered.transform.stamp_ns,
            base_frame=BASE_FRAME,
            base_from_camera=registered.transform.base_from_camera,
            robot_state=admitted,
            report=report,
        )

    def capture_state(self, *, deadline: float = float("inf"), cancelled: Flag = lambda: False) -> LiveState:
        """The newest set registered now, with the current feedback; not checked for freshness.

        Waits up to CAPTURE_WAIT_S for a set with image-time feedback and TF and
        for feedback that is healthy and in sync, then fails retryable: a later
        capture may succeed.
        """
        from tf2_ros import TransformException

        started = time.monotonic()
        wait_until = min(deadline, started + CAPTURE_WAIT_S)
        # Where the search for a set goes: what each drain takes, and the matching.
        profile: dict = {"rounds": 0, "side_drain_s": 0.0, "side_taken": 0, "match_s": 0.0}
        while True:
            if cancelled():
                raise GraspFailure("CANCELED", "cancelled waiting for an observation")
            profile["rounds"] += 1
            began = time.monotonic()
            profile["side_taken"] += self._side.drain()
            profile["side_drain_s"] += time.monotonic() - began
            errors = []
            sets = self.recent_sets(profile)
            began = time.monotonic()
            for kept in sets:
                rgb, depth = kept[:2]
                stamp = stamp_ns(rgb.header)
                try:
                    feedback = self.history.at(stamp)
                    torso = lookup_matrix(self._tf, BASE_FRAME, rgb.header.frame_id, stamp)
                    rgb_from_depth = lookup_matrix(self._tf, rgb.header.frame_id, depth.header.frame_id, stamp)
                    break
                except (ValueError, TransformException) as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
            else:
                profile["match_s"] += time.monotonic() - began
                if time.monotonic() >= wait_until:
                    reason = errors[0] if errors else "no synchronized RGB-D set received yet"
                    raise GraspFailure(
                        OBSERVATION_UNAVAILABLE,
                        f"No recent RGB-D set has image-time feedback and TF: {reason}",
                        retryable=True,
                    )
                time.sleep(0.02)
                continue
            profile["match_s"] += time.monotonic() - began
            found = time.monotonic()
            rgb_info, depth_info = kept[2:]
            try:
                with gpu_section(self._gpu, deadline=deadline, cancelled=cancelled, code=OBSERVATION_UNAVAILABLE):
                    locked = time.monotonic()
                    frame, calibration_id = register_set(
                        rgb,
                        depth,
                        rgb_info,
                        depth_info,
                        rgb_from_depth,
                        depth_scale=self._depth_scale,
                        device=self._device,
                    )
                registered = RegisteredObservation(
                    frame,
                    ImageTransform(stamp, frame.frame_id, BASE_FRAME, torso, self._verified),
                    min(stamp, stamp_ns(depth.header)) * 1e-9,
                    feedback,
                    calibration_id,
                ).snapshot()
                registered_at = time.monotonic()
                current = self._current_feedback(deadline, cancelled)
            except ValueError as exc:
                # Unhealthy or unsynchronized feedback (live 2026-11-03: 3 of 48
                # picks) clears with the next samples; a newer set is taken.
                profile.setdefault("invalid", []).append(str(exc))
                if time.monotonic() >= wait_until:
                    raise GraspFailure(
                        OBSERVATION_UNAVAILABLE, f"Invalid RGB-D set or feedback: {exc}", retryable=True
                    ) from None
                time.sleep(0.02)
                continue
            break
        done = time.monotonic()
        self.calls.append(
            dict(
                stamp_ns=int(stamp),
                wait_s=found - started,
                **profile,
                gpu_wait_s=locked - found,
                registration_s=registered_at - locked,
                feedback_wait_s=done - registered_at,
                total_s=done - started,
                image_age_s=(self._node.get_clock().now().nanoseconds - stamp) * 1e-9,
            )
        )
        return LiveState(registered, current)

    def _current_feedback(self, deadline: float, cancelled: Flag):
        """The newest feedback, waiting up to FEEDBACK_WAIT_S for one inside the freshness limit.

        The current feedback is what has arrived by now (registration took its
        time since the image's drain, and nothing fills the history meanwhile).
        A sample just past the limit (a view the runtime published late or
        DDS dropped) is followed by the next one within a view period; waiting
        for it costs ~20 ms, where failing the freshness check made the caller
        take and register a whole new image (0.2-0.5 s). The limit is not
        relaxed: past FEEDBACK_WAIT_S the stale sample is returned and checked.
        """
        ended = min(deadline, time.monotonic() + FEEDBACK_WAIT_S)
        while True:
            self._side.drain()
            latest = self.history.latest()
            age = self.clock() - latest.received_at
            if 0 <= age <= self.revalidation.feedback_max_age_s or time.monotonic() >= ended or cancelled():
                return copy_feedback(latest)
            time.sleep(FEEDBACK_POLL_S)

    def close(self) -> None:
        import rclpy

        self._camera.close()
        rclpy.try_shutdown(context=self._camera_context)
        self._side.close()
