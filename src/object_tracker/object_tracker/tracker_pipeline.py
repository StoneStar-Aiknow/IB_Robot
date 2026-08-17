"""Single-target tracking pipeline: visual update, depth, filtering, motion, session."""

from dataclasses import dataclass, field

import numpy as np

from .geometry import back_project_pixel, robust_box_depth
from .kalman import ConstantVelocityFilter, FilterUpdate
from .motion import EgoCompensatedMotionClassifier, MotionEstimate
from .session import SessionState, SingleTargetSession
from .template_tracker import TemplateTracker


@dataclass(frozen=True)
class DepthParams:
    scale: float = 0.001
    min_m: float = 0.15
    max_m: float = 8.0
    min_valid_ratio: float = 0.2
    central_fraction: float = 0.6


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class TargetSnapshot:
    """Publishable state for one processed observation or prediction step."""

    stamp_s: float
    position_odom: tuple[float, float]
    position_variance_xy: tuple[float, float]
    velocity_odom: tuple[float, float]
    velocity_variance_xy: tuple[float, float]
    depth_m: float | None
    bbox: tuple[float, float, float, float] | None
    measured: bool
    confidence: float
    motion: MotionEstimate | None
    quality: dict[str, float] = field(default_factory=dict)
    reason: str = "measured"


class TrackerPipeline:
    """Coordinate template tracking, robust depth, Kalman filtering, and session state."""

    def __init__(
        self,
        *,
        session: SingleTargetSession,
        template: TemplateTracker,
        motion_classifier: EgoCompensatedMotionClassifier,
        innovation_gate: float = 9.21,
        max_prediction_s: float = 0.5,
        max_visual_failures: int = 5,
        search_radius_px: float = 60.0,
        search_radius_reacquire_px: float = 120.0,
        base_position_sigma_m: float = 0.02,
        depth_sigma_scale: float = 0.5,
        range_sigma_scale: float = 0.01,
        quality_sigma_scale: float = 0.02,
    ):
        self.session = session
        self.template = template
        self.motion = motion_classifier
        self.innovation_gate = float(innovation_gate)
        self.max_prediction_s = float(max_prediction_s)
        self.max_visual_failures = int(max_visual_failures)
        self.search_radius_px = float(search_radius_px)
        self.search_radius_reacquire_px = float(search_radius_reacquire_px)
        self.base_position_sigma_m = float(base_position_sigma_m)
        self.depth_sigma_scale = float(depth_sigma_scale)
        self.range_sigma_scale = float(range_sigma_scale)
        self.quality_sigma_scale = float(quality_sigma_scale)

        self.filter: ConstantVelocityFilter | None = None
        self._last_measured_stamp_s: float | None = None
        self._last_processed_stamp_s: float | None = None
        self._visual_failures = 0
        self._last_motion: MotionEstimate | None = None

    @property
    def current_state(self) -> SessionState | None:
        session = self.session.session
        return session.state if session else None

    def initialize_filter(self, position_odom: tuple[float, float]) -> None:
        """Seed the Kalman filter from the first accepted measurement."""
        self.filter = ConstantVelocityFilter(position_odom)
        self._last_measured_stamp_s = None
        self._last_processed_stamp_s = None
        self._visual_failures = 0

    def process_observation(
        self,
        *,
        stamp_s: float,
        gray: np.ndarray,
        depth_image: np.ndarray,
        intrinsics: Intrinsics,
        camera_to_odom: np.ndarray,
        depth_params: DepthParams | None = None,
        robot_linear_speed_mps: float = 0.0,
        robot_angular_speed_rps: float = 0.0,
    ) -> TargetSnapshot | None:
        """Run one synchronized RGB-D observation through the full pipeline."""
        if self.filter is None:
            return None
        state = self.current_state
        if state not in {SessionState.ACQUIRING, SessionState.TRACKING, SessionState.SEARCHING}:
            return None

        params = depth_params if depth_params is not None else DepthParams()
        searching = state == SessionState.SEARCHING
        radius = self.search_radius_reacquire_px if searching else self.search_radius_px
        visual = self.template.update(gray, search_radius_px=radius)
        measurement = None
        quality: dict[str, float] = {}
        if visual is not None:
            quality["match_score"] = visual.match_score
            quality["scale"] = visual.scale
            depth_info = robust_box_depth(
                depth_image,
                visual.bbox,
                depth_scale=params.scale,
                central_fraction=params.central_fraction,
                min_depth_m=params.min_m,
                max_depth_m=params.max_m,
                min_valid_ratio=params.min_valid_ratio,
            )
            if depth_info is not None:
                quality["depth_valid_ratio"] = depth_info.valid_ratio
                quality["depth_mad_m"] = depth_info.mad_m
                center_u, center_v = visual.center
                point_camera = back_project_pixel(
                    center_u, center_v, depth_info.depth_m, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
                )
                homogeneous = camera_to_odom @ np.asarray([*point_camera, 1.0])
                if homogeneous[3] != 0.0 and np.all(np.isfinite(homogeneous)):
                    position_odom = (float(homogeneous[0] / homogeneous[3]), float(homogeneous[1] / homogeneous[3]))
                    measurement = (position_odom, depth_info, visual)

        if measurement is None:
            self._visual_failures += 1
            return self._predict_only(stamp_s, reason=self._failure_reason(searching))

        position_odom, depth_info, visual = measurement
        sigma = max(
            self.base_position_sigma_m,
            self.depth_sigma_scale * depth_info.mad_m
            + self.range_sigma_scale * depth_info.depth_m
            + self.quality_sigma_scale * (1.0 - depth_info.valid_ratio),
        )
        covariance = np.diag([sigma * sigma, sigma * sigma])
        if self._last_measured_stamp_s is None:
            # The first confirmed measurement re-seeds the filter: the projected
            # map position is only a coarse acquisition prior.
            self.filter = ConstantVelocityFilter(position_odom)
            update = FilterUpdate(True, 0.0, "first_measurement")
        else:
            dt = self._advance_time(stamp_s)
            if dt > 0.0:
                self.filter.predict(dt)
            update = self.filter.update(position_odom, covariance, innovation_gate=self.innovation_gate)
        quality["innovation_distance"] = update.innovation_distance if update.innovation_distance is not None else -1.0
        if not update.accepted:
            self._visual_failures += 1
            return self._predict_only(stamp_s, reason="innovation_gate", quality=quality)

        self._visual_failures = 0
        self._last_measured_stamp_s = stamp_s
        state = self._transition_after_measurement()
        estimate = self._classify_motion(
            stamp_s,
            position_odom,
            covariance,
            robot_linear_speed_mps,
            robot_angular_speed_rps,
        )
        confidence = min(visual.match_score, 1.0) * 0.6 + depth_info.valid_ratio * 0.4
        return TargetSnapshot(
            stamp_s=stamp_s,
            position_odom=(float(self.filter.state[0]), float(self.filter.state[1])),
            position_variance_xy=(float(self.filter.covariance[0, 0]), float(self.filter.covariance[1, 1])),
            velocity_odom=(float(self.filter.state[2]), float(self.filter.state[3])),
            velocity_variance_xy=(float(self.filter.covariance[2, 2]), float(self.filter.covariance[3, 3])),
            depth_m=depth_info.depth_m,
            bbox=visual.bbox,
            measured=True,
            confidence=float(confidence),
            motion=estimate,
            quality=quality,
        )

    def predict_only(self, stamp_s: float) -> TargetSnapshot | None:
        """Advance the filter without visual evidence (missing depth or transform)."""
        return self._predict_only(stamp_s, reason="missing_depth_or_transform")

    def _predict_only(
        self, stamp_s: float, *, reason: str, quality: dict[str, float] | None = None
    ) -> TargetSnapshot | None:
        if self.filter is None:
            return None
        dt = self._advance_time(stamp_s)
        if dt > 0.0:
            self.filter.predict(dt)
        state = self.current_state
        if state == SessionState.ACQUIRING:
            if self._visual_failures > self.max_visual_failures:
                self.session.stop(self._session_id, "local confirmation failed")
                return None
        elif state == SessionState.TRACKING and self._visual_failures >= self.max_visual_failures:
            self.session.begin_search(self._session_id, reason)
            state = SessionState.SEARCHING
        elif state == SessionState.SEARCHING:
            last = self._last_measured_stamp_s if self._last_measured_stamp_s is not None else stamp_s
            if stamp_s - last > self.max_prediction_s:
                self.session.lose(self._session_id, "visual reacquisition exhausted")
                return None

        return TargetSnapshot(
            stamp_s=stamp_s,
            position_odom=(float(self.filter.state[0]), float(self.filter.state[1])),
            position_variance_xy=(float(self.filter.covariance[0, 0]), float(self.filter.covariance[1, 1])),
            velocity_odom=(float(self.filter.state[2]), float(self.filter.state[3])),
            velocity_variance_xy=(float(self.filter.covariance[2, 2]), float(self.filter.covariance[3, 3])),
            depth_m=None,
            bbox=self.template.bbox,
            measured=False,
            confidence=0.0,
            motion=self._last_motion,
            quality=quality or {},
            reason=reason,
        )

    def _transition_after_measurement(self) -> SessionState:
        state = self.current_state
        if state == SessionState.ACQUIRING:
            self.session.confirm(self._session_id)
        elif state == SessionState.SEARCHING:
            self.session.reacquire(self._session_id)
        return self.current_state or SessionState.TRACKING

    def _classify_motion(
        self,
        stamp_s: float,
        position_odom: tuple[float, float],
        covariance: np.ndarray,
        robot_linear_speed_mps: float,
        robot_angular_speed_rps: float,
    ) -> MotionEstimate:
        estimate = self.motion.update(
            stamp_s=stamp_s,
            position_odom=position_odom,
            position_covariance=covariance,
            robot_linear_speed_mps=robot_linear_speed_mps,
            robot_angular_speed_rps=robot_angular_speed_rps,
        )
        self._last_motion = estimate
        return estimate

    def _advance_time(self, stamp_s: float) -> float:
        previous = self._last_processed_stamp_s
        self._last_processed_stamp_s = stamp_s
        if previous is None or stamp_s <= previous:
            return 0.0
        return stamp_s - previous

    def _failure_reason(self, searching: bool) -> str:
        return "visual_reacquisition_miss" if searching else "visual_update_miss"

    @property
    def _session_id(self) -> str:
        session = self.session.session
        if session is None:
            raise RuntimeError("no tracking session exists")
        return session.session_id
