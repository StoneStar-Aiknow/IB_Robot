"""Perception started before a pick's goal, and the container handed to its place.

The Gateway knows a pick's target (and, for a pick followed by a place, its
container) once the Agent has planned, before the task-pose preparation runs.
:class:`PerceptionPrefetch` captures and proposes for that target on a thread
meanwhile; the goal takes the result only for the same target, when it is young
enough and the robot has not moved since the capture (``GraspBackends.posture``),
and observes again otherwise. A container detected on the same capture travels
with the held object to the place, which reuses its mask only when its own
capture shows the same camera and an unchanged container region
(:func:`container_change`).

Nothing here imports ROS; the message conversions take the message classes.
"""

from __future__ import annotations

import threading
import time
from array import array as _array
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .contracts import Flag, GraspFailure, Observation, Proposals

#: Pixels around the container mask kept for the place's comparison.
ROI_MARGIN_PX = 12
#: The place's camera may differ from the pick's by at most this (m, deg).
CAMERA_SHIFT_M = 0.002
CAMERA_TURN_DEG = 0.2
#: A region pixel counts as changed past these (RGB levels; depth metres), as the
#: grasp target's recheck does (aimdk_grasp RevalidationConfig rgb_change, depth_change_m).
RGB_CHANGE = 30.0
DEPTH_CHANGE_M = 0.01
#: The container region may change by at most this share of its pixels: the held
#: object and the arm enter the view between the two captures, and the container
#: may not be reused when more than this of it looks different.
MAX_CHANGED_FRACTION = 0.10


@dataclass(frozen=True)
class PrefetchedContainer:
    """A container mask on one capture, with what the place compares it by."""

    query: str
    stamp_ns: int
    base_from_camera: np.ndarray
    intrinsics: np.ndarray
    mask: np.ndarray
    roi: tuple[int, int, int, int]
    rgb: np.ndarray
    depth: np.ndarray


def container_snapshot(observation: Observation, mask, query: str) -> PrefetchedContainer:
    """The container ``mask`` on ``observation`` with its region's RGB and depth."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != observation.depth_m.shape or not mask.any():
        raise ValueError("a nonempty container mask over the whole image is required")
    rows, cols = np.nonzero(mask)
    height, width = mask.shape
    roi = (
        max(0, int(cols.min()) - ROI_MARGIN_PX),
        max(0, int(rows.min()) - ROI_MARGIN_PX),
        min(width, int(cols.max()) + 1 + ROI_MARGIN_PX),
        min(height, int(rows.max()) + 1 + ROI_MARGIN_PX),
    )
    x0, y0, x1, y1 = roi
    return PrefetchedContainer(
        query=str(query).strip(),
        stamp_ns=int(observation.stamp_ns),
        base_from_camera=np.array(observation.base_from_camera, dtype=float),
        intrinsics=np.array(observation.intrinsics, dtype=float),
        mask=mask.copy(),
        roi=roi,
        rgb=np.array(observation.rgb[y0:y1, x0:x1], dtype=np.uint8),
        depth=np.array(observation.depth_m[y0:y1, x0:x1], dtype=np.float32),
    )


def container_change(prefetched: PrefetchedContainer, observation: Observation, query: str) -> tuple[str, dict]:
    """Why ``prefetched`` cannot stand for the container in ``observation`` ('' when it can), and the measures."""
    report: dict = {"age_s": round((observation.stamp_ns - prefetched.stamp_ns) * 1e-9, 3)}
    if prefetched.query.casefold() != str(query).strip().casefold():
        return "another container was prefetched", report
    if observation.depth_m.shape != prefetched.mask.shape or not np.allclose(
        observation.intrinsics, prefetched.intrinsics
    ):
        return "the image geometry changed", report
    if observation.stamp_ns <= prefetched.stamp_ns:
        return "the place capture is not newer than the prefetched one", report
    shift = float(np.linalg.norm(observation.base_from_camera[:3, 3] - prefetched.base_from_camera[:3, 3]))
    relative = prefetched.base_from_camera[:3, :3].T @ observation.base_from_camera[:3, :3]
    turn = float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))
    report.update(camera_shift_m=round(shift, 4), camera_turn_deg=round(turn, 3))
    if shift > CAMERA_SHIFT_M or turn > CAMERA_TURN_DEG:
        return "the camera moved", report
    x0, y0, x1, y1 = prefetched.roi
    rgb = observation.rgb[y0:y1, x0:x1].astype(float)
    depth = observation.depth_m[y0:y1, x0:x1]
    rgb_changed = np.max(np.abs(rgb - prefetched.rgb.astype(float)), axis=2) > RGB_CHANGE
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(prefetched.depth) & (prefetched.depth > 0)
    depth_changed = valid & (np.abs(depth - prefetched.depth) > DEPTH_CHANGE_M)
    container = prefetched.mask[y0:y1, x0:x1]
    changed = float(np.mean((rgb_changed | depth_changed)[container]))
    report["changed_fraction"] = round(changed, 4)
    if changed > MAX_CHANGED_FRACTION:
        return "the container region changed", report
    return "", report


def container_to_msg(prefetched: PrefetchedContainer | None, message_class, *, encode_mask, image_class):
    """``ibrobot_msgs/PrefetchedContainer``; an empty one (query '') for None."""
    message = message_class()
    if prefetched is None:
        return message
    message.query = prefetched.query
    message.stamp_ns = int(prefetched.stamp_ns)
    message.base_from_camera = [float(v) for v in prefetched.base_from_camera.reshape(-1)]
    message.intrinsics = [float(v) for v in prefetched.intrinsics.reshape(-1)]
    message.mask = encode_mask(prefetched.mask)
    message.roi = [int(v) for v in prefetched.roi]
    message.rgb = _image(image_class, prefetched.rgb, "rgb8")
    message.depth = _image(image_class, prefetched.depth, "32FC1")
    return message


def container_from_msg(message, *, decode_mask) -> PrefetchedContainer | None:
    if not message.query:
        return None
    x0, y0, x1, y1 = (int(v) for v in message.roi)
    rgb = np.frombuffer(bytes(message.rgb.data), dtype=np.uint8).reshape(y1 - y0, x1 - x0, 3)
    depth = np.frombuffer(bytes(message.depth.data), dtype=np.float32).reshape(y1 - y0, x1 - x0)
    return PrefetchedContainer(
        query=message.query,
        stamp_ns=int(message.stamp_ns),
        base_from_camera=np.asarray(message.base_from_camera, dtype=float).reshape(4, 4),
        intrinsics=np.asarray(message.intrinsics, dtype=float).reshape(3, 3),
        mask=np.asarray(decode_mask(message.mask), dtype=bool),
        roi=(x0, y0, x1, y1),
        rgb=rgb.copy(),
        depth=depth.copy(),
    )


MASK_FORMAT = "mono8; png compressed"


def encode_mask(mask: np.ndarray):
    """A HxW boolean mask as a PNG ``sensor_msgs/CompressedImage``."""
    import cv2
    from sensor_msgs.msg import CompressedImage

    ok, data = cv2.imencode(".png", np.where(np.asarray(mask, dtype=bool), 255, 0).astype(np.uint8))
    if not ok:
        raise ValueError("PNG encoding failed")
    # array("B") skips the per-byte check rclpy runs on assigned bytes (70-80 ms per container ROI on x86).
    return CompressedImage(format=MASK_FORMAT, data=_array("B", data.tobytes()))


def decode_mask(message) -> np.ndarray:
    import cv2

    if message.format != MASK_FORMAT:
        raise ValueError(f"mask format {message.format!r} is not {MASK_FORMAT!r}")
    image = cv2.imdecode(np.frombuffer(bytes(message.data), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("mask does not decode")
    return image > 0


def _image(image_class, array: np.ndarray, encoding: str):
    image = image_class()
    image.height, image.width = int(array.shape[0]), int(array.shape[1])
    image.encoding = encoding
    image.is_bigendian = 0
    data = np.ascontiguousarray(array)
    image.step = int(data.strides[0])
    image.data = _array("B", data.tobytes())
    return image


@dataclass
class _Job:
    query: str
    container_query: str
    root_task_id: str
    began: float
    cancelled: threading.Event = field(default_factory=threading.Event)
    perceived: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    observation: Observation | None = None
    proposals: Proposals | None = None
    posture: np.ndarray | None = None
    captured_at: float | None = None
    container: PrefetchedContainer | None = None
    taken: bool = False
    error: str = ""
    container_error: str = ""
    report: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PrefetchedPerception:
    """What a goal takes over: the capture and its proposals."""

    observation: Observation
    proposals: Proposals
    report: dict


class PerceptionPrefetch:
    """One prefetch at a time; a new one replaces (cancels) the previous one,
    except a request for the same target and container while that one is young
    and unused: it joins it (and binds its task when that was still unknown).

    ``capture(deadline, cancelled)``, ``propose(observation, query, deadline,
    cancelled)`` and ``detect_container(observation, query, deadline,
    cancelled) -> mask`` are the robot's backends; ``posture() -> vector`` the
    robot's measured joints (None: nothing to compare, and no prefetch is used).
    """

    def __init__(
        self,
        *,
        capture: Callable,
        propose: Callable,
        posture: Callable | None,
        detect_container: Callable | None = None,
        budget_s: float = 10.0,
        max_age_s: float = 4.0,
        posture_tolerance_rad: float = 0.02,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._capture, self._propose, self._posture = capture, propose, posture
        self._detect_container = detect_container
        self.budget_s, self.max_age_s, self.posture_tolerance_rad = budget_s, max_age_s, posture_tolerance_rad
        self._clock = clock
        self._lock = threading.Lock()
        self._job: _Job | None = None

    def start(self, query: str, container_query: str = "", root_task_id: str = "") -> str:
        """Start a prefetch; '' when started, else why not."""
        query, container_query = str(query).strip(), str(container_query).strip()
        if not query:
            return "empty target query"
        if self._posture is None:
            return "the robot cannot report its posture; a prefetched capture could not be checked"
        root_task_id = str(root_task_id)
        with self._lock:
            current = self._job
            if (
                current is not None
                and not current.cancelled.is_set()
                and not current.taken
                and not current.error
                and current.query.casefold() == query.casefold()
                and current.container_query.casefold() == container_query.casefold()
                and current.root_task_id in ("", root_task_id)
                and self._clock() - current.began < self.max_age_s
            ):
                # The Agent starts it once the plan names the target, before the
                # Gateway knows the task; the Gateway's own request for the same
                # target then only binds the task (for the container handover).
                current.root_task_id = root_task_id or current.root_task_id
                current.report["joined_after_s"] = round(self._clock() - current.began, 3)
                return ""
            job = _Job(query, container_query, root_task_id, self._clock())
            if current is not None:
                current.cancelled.set()
            self._job = job
        threading.Thread(target=self._run, args=(job,), daemon=True, name="perception-prefetch").start()
        return ""

    def cancel(self) -> None:
        with self._lock:
            if self._job is not None:
                self._job.cancelled.set()
            self._job = None

    def _run(self, job: _Job) -> None:
        deadline = job.began + self.budget_s

        def stopped() -> bool:
            return job.cancelled.is_set() or self._clock() >= deadline

        try:
            began = self._clock()
            # Measured around the capture: a robot that moved during it left an image of no one posture.
            before = np.asarray(self._posture(), dtype=float)
            job.observation = self._capture(deadline=deadline, cancelled=stopped)
            job.posture = np.asarray(self._posture(), dtype=float)
            job.captured_at = self._clock()
            if before.shape != job.posture.shape or (
                len(before) and float(np.max(np.abs(before - job.posture))) > self.posture_tolerance_rad
            ):
                raise RuntimeError("the robot moved during the capture")
            job.report["capture_s"] = round(job.captured_at - began, 3)
            job.proposals = self._propose(job.observation, job.query, deadline=deadline, cancelled=stopped)
            job.report["perceive_s"] = round(self._clock() - job.captured_at, 3)
        except Exception as exc:  # noqa: BLE001 - the goal observes again
            job.error = f"{getattr(exc, 'code', type(exc).__name__)}: {exc}"[:300]
        finally:
            job.perceived.set()
        try:
            if job.container_query and self._detect_container is not None and job.observation is not None:
                began = self._clock()
                mask = self._detect_container(
                    job.observation, job.container_query, deadline=deadline, cancelled=stopped
                )
                if mask is not None:
                    job.container = container_snapshot(job.observation, mask, job.container_query)
                job.report["container_s"] = round(self._clock() - began, 3)
        except Exception as exc:  # noqa: BLE001 - the place detects again
            job.container_error = f"{getattr(exc, 'code', type(exc).__name__)}: {exc}"[:300]
        finally:
            job.finished.set()

    def take(self, query: str, *, deadline: float, cancelled: Flag) -> tuple[PrefetchedPerception | None, dict]:
        """The prefetched capture and proposals for ``query``, or None with the reason in the report.

        Waits for a running prefetch of the same target (it is ahead of a new
        capture); every other case answers at once.
        """
        with self._lock:
            job = self._job
        report: dict = {"used": False}
        if job is None:
            report["reason"] = "none started"
            return None, report
        report.update(job.report)
        if job.query.casefold() != str(query).strip().casefold():
            # Another target: stop it, the goal's own capture needs the camera and the models.
            job.cancelled.set()
            report["reason"] = f"prefetched {job.query!r}"
            return None, report
        if job.taken:
            report["reason"] = "already used by a goal"
            return None, report
        job.taken = True  # one goal only, whether or not it can use it
        waited = self._clock()
        while not job.perceived.wait(0.01):
            if cancelled() or self._clock() >= deadline:
                raise GraspFailure("CANCELED" if cancelled() else "SKILL_TIMEOUT", "waiting for the prefetched capture")
        report["wait_s"] = round(self._clock() - waited, 3)
        report.update(job.report)
        if job.error or job.proposals is None:
            report["reason"] = job.error or "no proposals"
            return None, report
        age = self._clock() - job.captured_at
        report["age_s"] = round(age, 3)
        if age > self.max_age_s:
            report["reason"] = f"older than {self.max_age_s:g} s"
            return None, report
        try:
            now = np.asarray(self._posture(), dtype=float)
        except Exception as exc:  # noqa: BLE001 - nothing proves the robot is still where it was
            report["reason"] = f"posture unavailable: {type(exc).__name__}: {exc}"[:300]
            return None, report
        if now.shape != job.posture.shape:
            report["reason"] = "posture changed shape"
            return None, report
        moved = float(np.max(np.abs(now - job.posture))) if len(now) else 0.0
        report["posture_change_rad"] = round(moved, 4)
        if not np.isfinite(moved) or moved > self.posture_tolerance_rad:
            report["reason"] = "the robot moved after the capture"
            return None, report
        report["used"] = True
        return PrefetchedPerception(job.observation, job.proposals, report), report

    def container(self, root_task_id: str, *, wait_s: float = 0.0) -> tuple[PrefetchedContainer | None, dict]:
        """The container prefetched for ``root_task_id`` (None when none or not ready within ``wait_s``)."""
        with self._lock:
            job = self._job
        if job is None or not job.container_query or job.root_task_id != str(root_task_id):
            return None, {"reason": "none prefetched for this task"}
        if not job.finished.wait(max(0.0, wait_s)):
            return None, {"reason": "still detecting"}
        if job.container is None:
            return None, {"reason": job.container_error or "not detected"}
        return job.container, {
            "query": job.container.query,
            **{k: v for k, v in job.report.items() if k == "container_s"},
        }
