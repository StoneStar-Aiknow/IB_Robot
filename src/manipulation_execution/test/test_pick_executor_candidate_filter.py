import pytest
from sensor_msgs.msg import JointState

from manipulation_execution.grasp_geometry import CandidatePlan
from manipulation_execution.pick_executor_node import IKPayload, PickExecutorNode, PickFlowError


class _ExecutorHarness:
    _validate_fk_fixed_finger_base_side = PickExecutorNode._validate_fk_fixed_finger_base_side

    def __init__(self) -> None:
        self._config = {
            "target_gripper": {
                "type": "asymmetric_single_moving_jaw",
                "fixed_finger_contact_ee": [-0.014, 0.0, 0.0],
                "fixed_finger_base_side": {
                    "enabled": True,
                    "reference_point_base": [0.0, 0.0, 0.0],
                    "min_alignment_cos": 0.0,
                    "min_fk_inward_offset_m": 0.003,
                },
            }
        }


def _plan() -> CandidatePlan:
    return CandidatePlan(
        approach=(0.194, 0.0, 0.1),
        grasp=(0.194, 0.0, 0.0),
        lift=(0.194, 0.0, 0.1),
        quaternion=(0.0, 0.0, 0.0, 1.0),
        approach_axis=(0.0, 0.0, -1.0),
        target_contact_ee=(0.0, 0.0, 0.0),
        target_contact_base=(0.2, 0.0, 0.0),
        target_width_m=0.02,
        width_reason="test",
        fixed_finger_target_gap_m=0.005,
        target_width_min_base=(0.19, 0.0, 0.0),
        target_width_max_base=(0.21, 0.0, 0.0),
        topdown_score=1.0,
    )


def test_final_fk_fixed_finger_base_side_accepts_inward_pose():
    executor = _ExecutorHarness()
    payload = IKPayload(
        joint_state=JointState(),
        ee_xyz=(0.194, 0.0, 0.0),
        ee_quaternion=(0.0, 0.0, 0.0, 1.0),
    )

    alignment = executor._validate_fk_fixed_finger_base_side(7, _plan(), payload)

    assert alignment is not None
    assert alignment.alignment_cos == 1.0
    assert alignment.inward_offset_m > 0.0


def test_final_fk_fixed_finger_base_side_rejects_mirrored_pose():
    executor = _ExecutorHarness()
    payload = IKPayload(
        joint_state=JointState(),
        ee_xyz=(0.194, 0.0, 0.0),
        ee_quaternion=(0.0, 0.0, 1.0, 0.0),
    )

    with pytest.raises(PickFlowError) as exc_info:
        executor._validate_fk_fixed_finger_base_side(11, _plan(), payload)

    assert exc_info.value.code == "FK_FIXED_FINGER_BASE_SIDE_REJECTED"
    assert exc_info.value.retryable is True
    assert "alignment=-1.000" in str(exc_info.value)


def test_final_fk_fixed_finger_base_side_rejects_edge_pose_with_too_little_inward_offset():
    executor = _ExecutorHarness()
    payload = IKPayload(
        joint_state=JointState(),
        ee_xyz=(0.2135, 0.0, 0.0),
        ee_quaternion=(0.0, 0.0, 0.0, 1.0),
    )

    with pytest.raises(PickFlowError) as exc_info:
        executor._validate_fk_fixed_finger_base_side(856, _plan(), payload)

    assert exc_info.value.code == "FK_FIXED_FINGER_BASE_SIDE_REJECTED"
    assert exc_info.value.retryable is True
    assert "inward_offset=0.0005m < 0.0030m" in str(exc_info.value)
