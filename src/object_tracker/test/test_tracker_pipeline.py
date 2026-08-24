import numpy as np
import pytest

from object_tracker.motion import EgoCompensatedMotionClassifier
from object_tracker.session import SessionState, SingleTargetSession
from object_tracker.template_tracker import TemplateTracker
from object_tracker.tracker_pipeline import Intrinsics, TrackerPipeline

FX = FY = 400.0
CX = 320.0
CY = 240.0
INTRINSICS = Intrinsics(fx=FX, fy=FY, cx=CX, cy=CY)
DEPTH_M = 2.0
DEPTH_MM = int(DEPTH_M / 0.001)


def _frame_with_target(shape, center, size=40):
    height, width = shape
    gray = np.full((height, width), 20, dtype=np.uint8)
    cx, cy = center
    gray[int(cy - size / 2) : int(cy + size / 2), int(cx - size / 2) : int(cx + size / 2)] = 220
    return gray


def _constant_depth(shape, value_mm=DEPTH_MM):
    return np.full(shape, value_mm, dtype=np.uint16)


def _identity_transform():
    return np.eye(4)


def _camera_at(x, y):
    """Camera-to-odom transform placing the camera origin at (x, y) in odom."""
    matrix = np.eye(4)
    matrix[0, 3] = x
    matrix[1, 3] = y
    return matrix


def _make_pipeline(expected_depth_m=None, **overrides):
    session = SingleTargetSession()
    session.start("obj-1", navigation_ready=True, map_ready=True)
    classifier_kwargs = {
        "window_samples": 3,
        "moving_confirmation_windows": 1,
        "stationary_confirmation_windows": 1,
    }
    pipeline = TrackerPipeline(
        session=session,
        template=TemplateTracker(),
        motion_classifier=EgoCompensatedMotionClassifier(**classifier_kwargs),
        max_visual_failures=3,
        max_prediction_s=0.5,
        **overrides,
    )
    frame = _frame_with_target((480, 640), (CX, CY))
    pipeline.template.initialize(frame, (CX - 30, CY - 30, CX + 30, CY + 30))
    pipeline.initialize_filter((1.5, -0.4), expected_depth_m=expected_depth_m)
    return pipeline


def _run_observation(
    pipeline, *, center=(CX, CY), stamp, transform=None, depth=None, shape=(480, 640), robot_speed=0.0
):
    gray = _frame_with_target(shape, center)
    depth_image = _constant_depth(shape) if depth is None else depth(shape)
    return pipeline.process_observation(
        stamp_s=stamp,
        gray=gray,
        depth_image=depth_image,
        intrinsics=INTRINSICS,
        camera_to_odom=transform if transform is not None else _identity_transform(),
        robot_linear_speed_mps=robot_speed,
    )


def test_first_measurement_confirms_session_and_reports_position():
    pipeline = _make_pipeline()
    snapshot = _run_observation(pipeline, stamp=1.0)

    assert snapshot is not None and snapshot.measured
    assert pipeline.current_state is SessionState.TRACKING
    expected_x = (CX - CX) * DEPTH_M / FX
    expected_y = (CY - CY) * DEPTH_M / FY
    assert snapshot.position_odom[0] == pytest.approx(expected_x, abs=1e-6)
    assert snapshot.position_odom[1] == pytest.approx(expected_y, abs=1e-6)
    assert snapshot.depth_m == pytest.approx(DEPTH_M)
    assert snapshot.confidence > 0.5


def test_camera_motion_translates_into_odom_position():
    pipeline = _make_pipeline()
    snapshot = _run_observation(pipeline, stamp=1.0, transform=_camera_at(0.7, 0.2))

    expected_x = 0.7
    expected_y = 0.2
    assert snapshot.position_odom[0] == pytest.approx(expected_x, abs=0.05)
    assert snapshot.position_odom[1] == pytest.approx(expected_y, abs=0.05)


def test_missing_depth_produces_prediction_only_snapshot():
    pipeline = _make_pipeline()

    def blank_depth(shape):
        return np.zeros(shape, dtype=np.uint16)

    snapshot = _run_observation(pipeline, stamp=1.0, depth=blank_depth)

    assert snapshot is not None
    assert not snapshot.measured
    assert snapshot.reason == "acquisition_miss"
    assert pipeline.current_state is SessionState.ACQUIRING


def test_visual_failure_streak_transitions_to_searching_then_lost():
    pipeline = _make_pipeline()
    assert _run_observation(pipeline, stamp=1.0).measured
    assert pipeline.current_state is SessionState.TRACKING

    blank = np.full((480, 640), 20, dtype=np.uint16)
    searching_snapshot = None
    for index in range(2, 8):
        searching_snapshot = pipeline.process_observation(
            stamp_s=float(index) * 0.1,
            gray=np.full((480, 640), 20, dtype=np.uint8),
            depth_image=blank,
            intrinsics=INTRINSICS,
            camera_to_odom=_identity_transform(),
        )
        if pipeline.current_state is SessionState.SEARCHING:
            break
    assert pipeline.current_state is SessionState.SEARCHING
    assert searching_snapshot is not None and not searching_snapshot.measured

    final = pipeline.process_observation(
        stamp_s=2.0,
        gray=np.full((480, 640), 20, dtype=np.uint8),
        depth_image=blank,
        intrinsics=INTRINSICS,
        camera_to_odom=_identity_transform(),
    )
    assert final is None
    assert pipeline.current_state is SessionState.LOST


def test_reacquisition_returns_session_to_tracking():
    pipeline = _make_pipeline()
    assert _run_observation(pipeline, stamp=1.0).measured

    blank_depth = np.full((480, 640), 0, dtype=np.uint16)
    blank_gray = np.full((480, 640), 20, dtype=np.uint8)
    for index in range(2, 6):
        pipeline.process_observation(
            stamp_s=float(index) * 0.1,
            gray=blank_gray,
            depth_image=blank_depth,
            intrinsics=INTRINSICS,
            camera_to_odom=_identity_transform(),
        )
    assert pipeline.current_state is SessionState.SEARCHING

    snapshot = _run_observation(pipeline, stamp=0.6)
    assert snapshot is not None and snapshot.measured
    assert pipeline.current_state is SessionState.TRACKING


def test_innovation_gate_rejects_large_object_jump():
    pipeline = _make_pipeline()
    assert _run_observation(pipeline, stamp=1.0).measured

    snapshot = _run_observation(pipeline, stamp=1.05, transform=_camera_at(3.0, 0.0))

    assert snapshot is not None
    assert not snapshot.measured
    assert snapshot.reason == "innovation_gate"


def test_moving_target_is_classified_after_displacement():
    pipeline = _make_pipeline()
    _run_observation(pipeline, stamp=1.0, transform=_camera_at(0.0, 0.0))
    _run_observation(pipeline, stamp=1.2, transform=_camera_at(0.0, 0.0))

    snapshot = _run_observation(pipeline, stamp=2.0, center=(CX + 150, CY), transform=_camera_at(0.0, 0.0))
    assert snapshot is not None
    assert snapshot.motion is not None


def test_predict_only_advances_filter_without_visual():
    pipeline = _make_pipeline()
    assert _run_observation(pipeline, stamp=1.0).measured

    snapshot = pipeline.predict_only(1.1)
    assert snapshot is not None
    assert not snapshot.measured
    assert snapshot.reason == "missing_depth_or_transform"
    assert snapshot.bbox is not None


def test_first_measurement_depth_mismatch_blocks_confirmation():
    """A template locked onto background must not confirm the session."""
    pipeline = _make_pipeline(expected_depth_m=1.0)

    def background_depth(shape):
        return np.full(shape, int(6.0 / 0.001), dtype=np.uint16)

    snapshot = _run_observation(pipeline, stamp=1.0, depth=background_depth)

    assert snapshot is not None
    assert not snapshot.measured
    assert snapshot.reason == "confirmation_depth_mismatch"
    assert pipeline.current_state is SessionState.ACQUIRING


def test_first_measurement_within_depth_tolerance_confirms():
    pipeline = _make_pipeline(expected_depth_m=1.0)

    def near_depth(shape):
        return np.full(shape, int(1.2 / 0.001), dtype=np.uint16)

    snapshot = _run_observation(pipeline, stamp=1.0, depth=near_depth)

    assert snapshot is not None and snapshot.measured
    assert pipeline.current_state is SessionState.TRACKING


def test_depth_jump_between_measurements_is_rejected():
    pipeline = _make_pipeline()
    assert _run_observation(pipeline, stamp=1.0).measured

    def far_depth(shape):
        return np.full(shape, int(5.0 / 0.001), dtype=np.uint16)

    snapshot = _run_observation(pipeline, stamp=1.05, depth=far_depth)

    assert snapshot is not None
    assert not snapshot.measured
    assert snapshot.reason == "depth_jump"
