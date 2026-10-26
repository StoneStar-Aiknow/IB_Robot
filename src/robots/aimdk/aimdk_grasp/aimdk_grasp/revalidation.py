"""Compare a planned image with newer observations before authorizing a plan.

The first implementation requires a stationary camera and a stable target patch.
Camera/target changes request a new observation and inference; old candidates are
never relabelled with a new image timestamp. Only the target patch is
compared; environment geometry is not rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .joints import HEAD, LEFT_ARM, RIGHT_ARM


class ReobserveRequired(RuntimeError):
    planning_report: dict


@dataclass(frozen=True)
class RevalidationConfig:
    feedback_max_age_s: float = 0.1
    scene_max_age_s: float = 0.5
    start_tolerance_rad: float = 0.025
    depth_change_m: float = 0.01
    rgb_change: float = 30.0
    changed_fraction: float = 0.02  # of the target ROI, RGB
    # Depth on a transparent target flickers by more than depth_change_m in
    # 2.2-2.6 % of the ROI between still frames with unchanged RGB (6 of 21
    # live rechecks, 2026-10-09; passing ones 0.3-0.8 %). A moved target
    # changes its whole silhouette band, in RGB as well.
    depth_changed_fraction: float = 0.05

    def __post_init__(self):
        values = tuple(vars(self).values())
        if (
            not np.isfinite(values).all()
            or min(values) <= 0
            or max(self.changed_fraction, self.depth_changed_fraction) >= 1
        ):
            raise ValueError("Finite positive revalidation limits required")


def require_fresh(state, now: float, config: RevalidationConfig) -> None:
    """Reject stale, future-dated or unhealthy feedback and scenes of a :class:`LiveState`."""
    if (
        not state.feedback.healthy
        or not state.observation.image_feedback.healthy
        or not np.isfinite(now)
        or not 0 <= now - state.feedback.received_at <= config.feedback_max_age_s
        or not 0 <= now - state.observation.received_at <= config.scene_max_age_s
    ):
        raise ReobserveRequired(
            "Latest feedback or registered scene stale, future-dated or unhealthy: "
            f"feedback_age={now - state.feedback.received_at:.6f}s, "
            f"scene_age={now - state.observation.received_at:.6f}s, "
            f"healthy={state.feedback.healthy}/{state.observation.image_feedback.healthy}"
        )


def require_posture(reference, current, config: RevalidationConfig, *, include_left: bool = True) -> float:
    """The largest joint change since the planning snapshot; rejects one beyond the start tolerance."""
    names = LEFT_ARM + RIGHT_ARM + HEAD if include_left else RIGHT_ARM + HEAD
    error = max(
        float(np.max(np.abs(reference.vector(names) - current.vector(names)))),
        float(np.max(np.abs(reference.left_hand - current.left_hand))),
    )
    if error > config.start_tolerance_rad:
        raise ReobserveRequired("Robot moved beyond the planning snapshot; replan required")
    return error


def require_same_image_geometry(reference, current) -> None:
    """Pixel comparison needs the same camera, intrinsics and image size.

    Camera motion is the head posture check's to reject (require_posture);
    calibration identity is not compared here (nothing rechecks it after planning).
    """
    if (
        reference.frame.intrinsics != current.frame.intrinsics
        or reference.frame.depth.shape != current.frame.depth.shape
        or reference.transform.camera_frame != current.transform.camera_frame
    ):
        raise ReobserveRequired("Camera image geometry changed; reobserve target")


def require_stable_patch(reference, current, mask, config: RevalidationConfig) -> dict:
    """Reject a target region whose RGB or measured depth changed between two observations."""
    from scipy.ndimage import binary_dilation

    require_same_image_geometry(reference, current)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != reference.frame.depth.shape or not mask.any():
        raise ValueError("A target mask is required for observation revalidation")
    # The border catches silhouette motion into and out of the original mask.
    roi = binary_dilation(mask, iterations=3)
    a, b = reference.frame.depth[roi], current.frame.depth[roi]
    # Only depth measured in both frames counts: a transparent target's valid
    # pixels flicker between still frames (4-9 % of this ROI in a recorded
    # open-hand capture, 2026-09-29); a moved target changes RGB and valid depth.
    va, vb = np.isfinite(a) & (a > 0), np.isfinite(b) & (b > 0)
    changed = va & vb & (np.abs(a - b) > config.depth_change_m)
    rgb_changed = (
        np.max(
            np.abs(reference.frame.rgb[roi].astype(float) - current.frame.rgb[roi].astype(float)),
            axis=1,
        )
        > config.rgb_change
    )
    depth_fraction, rgb_fraction = float(changed.mean()), float(rgb_changed.mean())
    gap_s = (current.transform.stamp_ns - reference.transform.stamp_ns) * 1e-9
    if depth_fraction > config.depth_changed_fraction or rgb_fraction > config.changed_fraction:
        raise ReobserveRequired(
            f"Target/observed region changed (depth {depth_fraction:.3f} of the ROI, limit "
            f"{config.depth_changed_fraction}; rgb {rgb_fraction:.3f}, limit {config.changed_fraction}; "
            f"frames {gap_s:.2f} s apart); reobserve and regenerate candidates"
        )
    return {"depth_changed_fraction": depth_fraction, "rgb_changed_fraction": rgb_fraction, "frame_gap_s": gap_s}
