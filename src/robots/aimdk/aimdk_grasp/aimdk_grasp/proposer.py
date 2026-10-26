"""The X2 grasp proposer: the target and its candidates from the models, its scene built alongside.

One target per request. The mask-independent scene stage (unproject, reach
crop, self-filter) starts before detection and overlaps the model round trips;
the mask-dependent stage and GraspGen-X share the request window on an
independent CUDA stream. A detection the scene finds no support beneath
(NoSupportError: live, a second "bottle" on the floor outscored the one on the
table) gives way to the next-best one, up to ``support_fallbacks`` times. The
finished scene goes to ``preload(start, scene)`` (the planner, with the
request's current feedback as the start) while GraspGen-X is still
inferring: 0.13 s of the planning call on the Orin (live 2026-10-13).

The scene is returned as ``Proposals.scene`` for the X2 planner. With
``remote_scene`` the planning host builds it instead, from the capture it
keeps: it starts the mask-independent stage when it captures, and the mask
goes to it while GraspGen-X infers here; ``Proposals.scene`` is then what
``remote_scene`` returns.

The detector and the segmenter see the image turned upright (``upright_turns``):
the X2 head camera is mounted upside down. Their boxes and masks are turned
back, so everything after them is in the camera's own image.

Migrated from the X2 pipeline's RGBDGraspPipeline.run_with_scene and the
perception half of GraspRequestProcessor.process.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from manipulation_execution.grasp_pipeline.contracts import Flag, GraspFailure, Observation, Proposals
from manipulation_execution.grasp_pipeline.perception import Detection, PerceptionServices

from .feedback import Feedback
from .joints import LEFT_ARM
from .observation import AdmittedState
from .rgbd_to_cloud import largest_cluster_mask, mask_depth_to_points
from .robot_spheres import RobotSphereModel
from .scene_estimation import NoSupportError, SceneEstimate, SceneEstimationConfig
from .types import CameraIntrinsics, RGBDFrame

NO_EXECUTABLE_CANDIDATE = "NO_EXECUTABLE_CANDIDATE"


def detection_prompt(target_query: str) -> str:
    """GroundingDINO expects '<phrase> .'; IB_Robot passes the bare target phrase."""
    query = " ".join(str(target_query).split())
    return query if query.endswith(".") else f"{query} ."


def box_iou(first, second):
    """Intersection over union of two xyxy boxes."""
    (a0, a1, a2, a3), (b0, b1, b2, b3) = ([float(value) for value in box] for box in (first, second))
    inter = max(0.0, min(a2, b2) - max(a0, b0)) * max(0.0, min(a3, b3) - max(a1, b1))
    union = (a2 - a0) * (a3 - a1) + (b2 - b0) * (b3 - b1) - inter
    return inter / union if union > 0 else 0.0


#: Below this the camera looks within ~15 degrees of straight up or down: its image has no up.
MIN_UP_IN_IMAGE = 0.25


def upright_turns(base_from_camera) -> int:
    """Counter-clockwise quarter turns (``np.rot90``) that put the base's up (+z) at the top of the image.

    GroundingDINO ranks objects as it learned them, upright. On the X2's
    upside-down head camera "an apple ." put a pear first in 6 of 8 frames and
    "a cup ." boxed the apple; turned upright, both were right in 8 of 8 (live
    2026-10-26). The camera frame is optical: x right, y down in the image.
    """
    up = np.asarray(base_from_camera, dtype=float)[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    direction = up[:2]
    if np.hypot(*direction) < MIN_UP_IN_IMAGE:
        return 0
    best, turns = -np.inf, 0
    for k in range(4):
        if -direction[1] > best:
            best, turns = -direction[1], k
        direction = np.array([direction[1], -direction[0]])  # where one more turn takes it
    return turns


def _turn_box(box, turns: int, width: int, height: int) -> np.ndarray:
    """An xyxy box (pixel edges) after ``turns`` counter-clockwise quarter turns of a width x height image."""
    x0, y0, x1, y1 = (float(value) for value in box)
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    for _ in range(turns % 4):
        corners = np.stack([corners[:, 1], width - corners[:, 0]], axis=1)
        width, height = height, width
    return np.array([*corners.min(axis=0), *corners.max(axis=0)], dtype=np.float32)


@dataclass(frozen=True)
class ImageTurn:
    """The image the models see, ``turns`` counter-clockwise quarter turns from the camera's, and back."""

    turns: int
    height: int
    width: int

    def image(self, array: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.rot90(array, self.turns)) if self.turns else array

    def mask_back(self, mask: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.rot90(mask, -self.turns)) if self.turns else mask

    def box_turned(self, box) -> np.ndarray:
        return _turn_box(box, self.turns, self.width, self.height)

    def box_back(self, box) -> np.ndarray:
        width, height = (self.height, self.width) if self.turns % 2 else (self.width, self.height)
        return _turn_box(box, -self.turns % 4, width, height)

    def detection_back(self, detection: Detection) -> Detection:
        if not self.turns:
            return detection
        mask = None if detection.mask is None else self.mask_back(np.asarray(detection.mask, dtype=bool))
        return replace(detection, box_xyxy=self.box_back(detection.box_xyxy), mask=mask)


@dataclass(frozen=True)
class PerceptionSettings:
    """The live X2 perception (run reports, 2026-10): 400 candidates, thresholds 0.25, no confidence floor."""

    box_threshold: float = 0.25
    text_threshold: float = 0.25
    max_grasps: int = 400
    min_confidence: float = -1.0
    min_points: int = 30
    min_depth_m: float = 0.05
    max_depth_m: float = 3.0
    # Lower-scored detections tried, in score order, when the scene finds no
    # support beneath the target (NoSupportError).
    support_fallbacks: int = 2
    scene_overlap: bool = True
    scene: SceneEstimationConfig = field(default_factory=lambda: SceneEstimationConfig(max_tilt_deg=35.0))

    def __post_init__(self):
        counts = (self.max_grasps, self.min_points)
        if any(type(value) is not int or value < 1 for value in counts) or type(self.support_fallbacks) is not int:
            raise ValueError("Positive perception counts required")
        if (
            not np.isfinite([self.min_depth_m, self.max_depth_m, self.box_threshold, self.text_threshold]).all()
            or not 0 < self.min_depth_m < self.max_depth_m
            or not 0 <= self.box_threshold <= 1
            or not 0 <= self.text_threshold <= 1
            or not np.isfinite(self.min_confidence)
            or self.support_fallbacks < 0
        ):
            raise ValueError("Invalid perception configuration")


def measured_spheres(model: RobotSphereModel, feedback) -> np.ndarray:
    """Torso-frame collision spheres of the measured pose."""
    spheres, _ = model.spheres(feedback, feedback.vector(LEFT_ARM)[None])
    return spheres[0].copy()


class X2Proposer:
    """``GraspProposer`` for the X2: GroundingDINO (+ SAM2), GraspGen-X and the collision scene."""

    def __init__(
        self,
        services: PerceptionServices,
        sphere_model: RobotSphereModel | None,
        *,
        settings: PerceptionSettings = PerceptionSettings(),
        preload: Callable[[Feedback, SceneEstimate], None] | None = None,
        remote_scene: Callable[[Observation, np.ndarray, float], object] | None = None,
    ):
        """``remote_scene(observation, mask, timeout_s)`` finishes the scene on the planning host.

        It raises NoSupportError when nothing supports the target, and what it
        returns is the proposals' scene. Without it the scene is built here,
        from an X2 head camera observation, with ``sphere_model`` and ``preload``.
        """
        if remote_scene is None and sphere_model is None:
            raise ValueError("a scene built here needs the robot's sphere model")
        self.services, self.sphere_model, self.settings, self.preload = services, sphere_model, settings, preload
        self.remote_scene = remote_scene

    def propose(self, observation: Observation, query: str, *, deadline: float, cancelled: Flag) -> Proposals:
        def checkpoint():
            if cancelled():
                raise GraspFailure("CANCELED", "goal cancelled")
            if time.monotonic() >= deadline:
                raise GraspFailure("SKILL_TIMEOUT", "goal deadline passed")

        calls = len(self.services.calls)
        report: dict = dict(prompt=detection_prompt(query), settings=asdict(self.settings))
        try:
            if self.remote_scene is None:
                frame, stamp_ns, scene = self._local_scene(observation, report)
            else:
                frame, stamp_ns = observed_frame(observation), observation.stamp_ns
                scene = _RemoteScene(self.remote_scene, observation, deadline)
            turn = ImageTurn(upright_turns(observation.base_from_camera), *frame.rgb.shape[:2])
            report["detection_turns"] = turn.turns
            target, estimate, timings = self._run_with_scene(
                frame, stamp_ns, report["prompt"], scene, turn, checkpoint, cancelled
            )
        except GraspFailure:
            raise
        except (RuntimeError, ValueError) as exc:
            raise GraspFailure(NO_EXECUTABLE_CANDIDATE, f"{type(exc).__name__}: {exc}") from exc
        finally:
            report["model_rpc"] = self.services.calls[calls:]
        detection, mask, points, poses, confidence = target
        host_report = getattr(estimate, "report", None)
        if self.remote_scene is not None and host_report:
            report["scene_host"] = dict(host_report)  # what the planning host measured
        report.update(
            perception_timings_s=timings,
            candidate_count=len(poses),
            detection=dict(
                label=detection.label,
                score=detection.score,
                box_xyxy=[float(value) for value in detection.box_xyxy],
            ),
        )
        return Proposals(mask, points, poses, confidence, report, scene=estimate)

    # -- the request -----------------------------------------------------------------

    def _detect(self, frame, stamp_ns, prompt, count, turn, cancelled):
        settings = self.settings
        detections = self.services.detect(
            turn.image(frame.rgb),
            frame.frame_id,
            stamp_ns,
            prompt,
            box_threshold=settings.box_threshold,
            text_threshold=settings.text_threshold,
            cancelled=cancelled,
        )
        return [turn.detection_back(detection) for detection in detections[:count]]

    def _segment(self, frame, stamp_ns, detection, turn, cancelled):
        """The detection's mask in the camera image: the detector's own, or the segmenter's on the turned image."""
        if detection.mask is not None:
            return [detection.mask]
        if not turn.turns:
            return self.services.masks(frame.rgb, frame.frame_id, stamp_ns, [detection], cancelled=cancelled)
        turned = replace(detection, box_xyxy=turn.box_turned(detection.box_xyxy))
        masks = self.services.masks(turn.image(frame.rgb), frame.frame_id, stamp_ns, [turned], cancelled=cancelled)
        return [turn.mask_back(np.asarray(mask, dtype=bool)) for mask in masks]

    def _cloud(self, frame, mask):
        points = mask_depth_to_points(
            mask,
            frame.depth,
            frame.intrinsics,
            depth_scale=frame.depth_scale,
            min_depth_m=self.settings.min_depth_m,
            max_depth_m=self.settings.max_depth_m,
        )
        if len(points) < self.settings.min_points:
            raise RuntimeError("Target has insufficient valid depth")
        return points

    def _clean(self, frame, mask):
        """Drop mask pixels outside the target's largest 3-D cluster (see largest_cluster_mask)."""
        cleaned, _ = largest_cluster_mask(
            mask,
            frame.depth,
            frame.intrinsics,
            frame.depth_scale,
            min_depth_m=self.settings.min_depth_m,
            max_depth_m=self.settings.max_depth_m,
        )
        return cleaned

    def _local_scene(self, observation: Observation, report: dict):
        admitted = observation.robot_state
        if not isinstance(admitted, AdmittedState):
            raise GraspFailure("INTERNAL_ERROR", "the X2 proposer needs an observation from the X2 head camera")
        registered = admitted.live.observation
        began = time.perf_counter()
        spheres = measured_spheres(self.sphere_model, registered.image_feedback)
        report["robot_spheres_s"] = time.perf_counter() - began
        scene = _LocalScene(registered, admitted.live.feedback, spheres, self.settings, self.preload, report)
        return registered.frame, registered.transform.stamp_ns, scene

    def _run_with_scene(self, frame, stamp_ns, prompt, scene, turn, checkpoint, cancelled):
        """Returns ((detection, mask, points, poses, confidence), scene, timings)."""
        settings = self.settings
        started = time.perf_counter()
        timings = {}

        def measured(name, operation):
            checkpoint()
            before = time.perf_counter()
            try:
                result = operation()
            finally:
                timings[name] = time.perf_counter() - before
            checkpoint()
            return result

        def infer(points):
            poses, confidence = measured(
                "grasp",
                lambda: self.services.generate_grasps(
                    points,
                    frame.frame_id,
                    stamp_ns,
                    max_grasps=settings.max_grasps,
                    min_confidence=settings.min_confidence,
                    cancelled=cancelled,
                ),
            )
            if not len(poses):
                raise RuntimeError("No grasp candidates returned")
            return poses, confidence

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="grasp-scene") if settings.scene_overlap else None
        try:
            if pool is not None:
                scene.start(pool, measured)
            detections = measured(
                "detect", lambda: self._detect(frame, stamp_ns, prompt, 1 + settings.support_fallbacks, turn, cancelled)
            )
            if not detections:
                raise RuntimeError("No target detected")
            rejected = []
            for detection in detections:
                if any(box_iou(detection.box_xyxy, other["box_xyxy"]) > 0.5 for other in rejected):
                    continue  # the same object, already found unsupported
                masks = measured(
                    "segment",
                    lambda detection=detection: self._segment(frame, stamp_ns, detection, turn, cancelled),
                )
                if len(masks) != 1:
                    raise RuntimeError("Expected one target mask")
                mask = measured("clean", lambda masks=masks: self._clean(frame, np.array(masks[0], dtype=bool)))
                points = measured("cloud", lambda mask=mask: self._cloud(frame, mask))
                try:
                    if pool is None:
                        poses, confidence = infer(points)
                        estimate = scene.whole(mask, measured)
                    else:
                        future = pool.submit(scene.finish, mask, measured)
                        poses, confidence = infer(points)
                        estimate = future.result()
                    break
                except NoSupportError as exc:
                    unsupported = exc
                    rejected.append(
                        dict(
                            score=float(detection.score),
                            error=str(exc),
                            box_xyxy=[float(value) for value in detection.box_xyxy],
                        )
                    )
                    timings["rejected_detections"] = rejected
            else:
                raise unsupported  # no detection tried has a support beneath it
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
        timings["total"] = time.perf_counter() - started
        timings["overlap_enabled"] = settings.scene_overlap
        return (detection, mask, points, poses, confidence), estimate, timings


def observed_frame(observation: Observation) -> RGBDFrame:
    """The registered RGB-D frame of a pipeline observation (depth in metres)."""
    k = observation.intrinsics
    return RGBDFrame(
        observation.rgb,
        observation.depth_m,
        CameraIntrinsics(float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])),
        frame_id=observation.frame_id,
    )


class _LocalScene:
    """The scene built in this process: the mask-independent stage alongside detection and
    segmentation, the rest on its own CUDA stream alongside GraspGen-X, then preloaded."""

    def __init__(self, registered, start: Feedback, spheres, settings: PerceptionSettings, preload, report: dict):
        frame, transform = registered.frame, registered.transform
        if frame.frame_id != transform.camera_frame:
            raise ValueError("Scene and perception must share the exact image timestamp/frame")
        # Private copies: the stages run on another thread while the request goes on.
        self.transform = replace(transform, base_from_camera=np.array(transform.base_from_camera, copy=True))
        self.frame = replace(frame, rgb=np.array(frame.rgb, copy=True), depth=np.array(frame.depth, copy=True))
        self.spheres = np.array(spheres, copy=True)
        self.stamp_ns, self.received_at = transform.stamp_ns, registered.received_at
        self.start_feedback, self.config, self.preload, self.report = start, settings.scene, preload, report
        self.overlap, self._prepared = settings.scene_overlap, None
        self._stream = None

    def _context(self) -> AbstractContextManager:
        if self._stream is None:
            return nullcontext()
        import torch

        return torch.cuda.stream(self._stream)

    def _synchronize(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()

    def start(self, pool, measured) -> None:
        from .scene_estimation import prepare_scene

        if self.overlap:
            try:
                import torch
            except ImportError:
                pass
            else:
                if torch.cuda.is_available():
                    self._stream = torch.cuda.Stream()
                    self._stream.wait_stream(torch.cuda.current_stream())

        def prepare():
            try:
                with self._context():
                    return measured(
                        "scene_prepare",
                        lambda: prepare_scene(
                            self.frame.depth,
                            self.frame.intrinsics,
                            self.transform,
                            stamp_ns=self.stamp_ns,
                            camera_frame=self.frame.frame_id,
                            received_at=self.received_at,
                            robot_spheres_torso=self.spheres,
                            depth_scale=self.frame.depth_scale,
                            config=self.config,
                        ),
                    )
            finally:
                self._synchronize()

        self._prepared = pool.submit(prepare)

    def finish(self, mask, measured) -> SceneEstimate:
        """Runs on the pool after the mask-independent stage it submitted first."""
        from .scene_estimation import complete_scene

        prepared = self._prepared.result()
        try:
            with self._context():
                estimate = measured("scene", lambda: complete_scene(prepared, mask, lazy_validation=True))
        finally:
            self._synchronize()
        return self._loaded(estimate, measured)

    def whole(self, mask, measured) -> SceneEstimate:
        from .scene_estimation import estimate_scene

        estimate = measured(
            "scene",
            lambda: estimate_scene(
                self.frame.depth,
                self.frame.intrinsics,
                self.transform,
                stamp_ns=self.stamp_ns,
                camera_frame=self.frame.frame_id,
                received_at=self.received_at,
                robot_spheres_torso=self.spheres,
                target_mask=mask,
                depth_scale=self.frame.depth_scale,
                config=self.config,
            ),
        )
        return self._loaded(estimate, measured)

    def _loaded(self, estimate: SceneEstimate, measured) -> SceneEstimate:
        # On the caller's default CUDA stream, not the scene stream. A failure
        # here is not the request's: plan prepares again itself.
        if self.preload is not None:
            try:
                measured("scene_load", lambda: self.preload(self.start_feedback, estimate))
            except GraspFailure:
                raise
            except Exception as exc:  # noqa: BLE001 - plan repeats the load and reports its own error
                self.report["scene_preload_error"] = f"{type(exc).__name__}: {exc}"
        return estimate


class _RemoteScene:
    """The scene the planning host builds from the capture it keeps; it prepared on capturing."""

    def __init__(self, complete, observation: Observation, deadline: float):
        self.complete, self.observation, self.deadline = complete, observation, deadline

    def start(self, pool, measured) -> None:
        pass

    def finish(self, mask, measured):
        return measured("scene", lambda: self.complete(self.observation, mask, self.deadline - time.monotonic()))

    whole = finish
