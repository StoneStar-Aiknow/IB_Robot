import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped

from manipulation_execution.grasp_geometry import prepared_candidate_soft_score
from manipulation_execution.pick_executor_node import PickExecutorNode


class _RecordingBuffer:
    def __init__(self, transform: TransformStamped) -> None:
        self.transform = transform
        self.lookups = []

    def lookup_transform(self, target_frame, source_frame, lookup_time, *, timeout):
        self.lookups.append((target_frame, source_frame, lookup_time.nanoseconds, timeout.nanoseconds))
        return self.transform


class _ExecutorHarness:
    _stamp_to_ns = staticmethod(PickExecutorNode._stamp_to_ns)
    _transform_to_matrix = staticmethod(PickExecutorNode._transform_to_matrix)
    _lookup_base_transform = PickExecutorNode._lookup_base_transform
    _lookup_base_to_camera = PickExecutorNode._lookup_base_to_camera

    def __init__(self, transform: TransformStamped) -> None:
        self._base_frame = "base"
        self._rpc_timeout = 2.0
        self._tf_buffer = _RecordingBuffer(transform)


def test_camera_lookup_uses_candidate_capture_time():
    transform = TransformStamped()
    transform.header.stamp = TimeMsg(sec=17, nanosec=42)
    transform.transform.translation.x = 0.1
    transform.transform.translation.y = -0.2
    transform.transform.translation.z = 0.3
    transform.transform.rotation.w = 1.0
    executor = _ExecutorHarness(transform)
    capture_stamp = TimeMsg(sec=16, nanosec=123)

    matrix = executor._lookup_base_to_camera("camera_wrist_optical_frame", capture_stamp)

    assert executor._tf_buffer.lookups == [("base", "camera_wrist_optical_frame", 16_000_000_123, 2_000_000_000)]
    assert np.allclose(matrix[:3, 3], [0.1, -0.2, 0.3])


def test_prepared_soft_score_prioritizes_fixed_finger_envelope():
    config = {
        "fixed_finger_envelope_weight": 0.55,
        "contact_xy_weight": 0.25,
        "contact_z_weight": 0.15,
        "confidence_weight": 0.05,
        "contact_xy_scale_m": 0.03,
        "contact_z_scale_m": 0.02,
    }
    preferred = prepared_candidate_soft_score(
        config,
        fixed_finger_envelope=1.0,
        contact_residual_xy_m=0.009,
        contact_z_error_m=0.001,
        confidence=0.80,
    )
    front_contact = prepared_candidate_soft_score(
        config,
        fixed_finger_envelope=0.2,
        contact_residual_xy_m=0.002,
        contact_z_error_m=0.0005,
        confidence=0.85,
    )

    assert preferred > front_contact


def test_prepared_soft_score_balances_centroid_alignment():
    config = {
        "fixed_finger_envelope_weight": 0.55,
        "contact_xy_weight": 0.25,
        "contact_z_weight": 0.15,
        "confidence_weight": 0.05,
        "centroid_distance_weight": 0.50,
        "contact_xy_scale_m": 0.03,
        "contact_z_scale_m": 0.02,
        "centroid_distance_scale_m": 0.01,
    }
    off_center = prepared_candidate_soft_score(
        config,
        fixed_finger_envelope=0.992,
        contact_residual_xy_m=0.002,
        contact_z_error_m=0.0001,
        confidence=0.88,
        centroid_distance_m=0.0104,
    )
    centered = prepared_candidate_soft_score(
        config,
        fixed_finger_envelope=0.950,
        contact_residual_xy_m=0.006,
        contact_z_error_m=0.0015,
        confidence=0.83,
        centroid_distance_m=0.0053,
    )

    assert centered > off_center


def test_prepared_soft_score_prefers_robust_gap_headroom_over_near_score() -> None:
    config = {
        "confidence_weight": 1.0,
        "contact_xy_weight": 0.0,
        "contact_z_weight": 0.0,
        "fixed_finger_envelope_weight": 0.0,
        "robust_gap_headroom_weight": 0.35,
        "robust_gap_headroom_scale_m": 0.004,
    }
    higher_source_score = prepared_candidate_soft_score(
        config,
        fixed_finger_envelope=1.0,
        contact_residual_xy_m=0.0,
        contact_z_error_m=0.0,
        confidence=0.90,
        robust_gap_headroom_m=-0.0003,
    )
    safer_candidate = prepared_candidate_soft_score(
        config,
        fixed_finger_envelope=1.0,
        contact_residual_xy_m=0.0,
        contact_z_error_m=0.0,
        confidence=0.88,
        robust_gap_headroom_m=0.002,
    )

    assert safer_candidate > higher_source_score


def test_prepared_soft_score_keeps_legacy_result_without_headroom_weight() -> None:
    config = {"confidence_weight": 1.0, "fixed_finger_envelope_weight": 1.0}
    inputs = {
        "fixed_finger_envelope": None,
        "contact_residual_xy_m": 0.001,
        "contact_z_error_m": 0.001,
        "confidence": 0.8,
    }

    legacy = prepared_candidate_soft_score(config, **inputs)
    missing_headroom = prepared_candidate_soft_score(config, robust_gap_headroom_m=None, **inputs)

    assert missing_headroom == legacy
