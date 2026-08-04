import math

import pytest
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from manipulation_execution.grasp_geometry import euler_xyz_matrix, quaternion_from_matrix
from manipulation_execution.pick_executor_models import PickFlowError
from manipulation_execution.pick_executor_node import IKPayload, PickExecutorNode


def _joint_state(joint5: float) -> JointState:
    state = JointState()
    state.name = ["1", "2", "3", "4", "5"]
    state.position = [0.0, 0.0, 0.0, 0.0, joint5]
    return state


class _OrientationHarness:
    _orientation_guard = PickExecutorNode._orientation_guard
    _orientation_limits = PickExecutorNode._orientation_limits
    _joint_position = staticmethod(PickExecutorNode._joint_position)
    _joint_state_with_joint5 = staticmethod(PickExecutorNode._joint_state_with_joint5)
    _pose_components = staticmethod(PickExecutorNode._pose_components)
    _solve_orientation_consistent_grasp_ik_fk = PickExecutorNode._solve_orientation_consistent_grasp_ik_fk

    def __init__(self) -> None:
        self._config = {
            "target_gripper": {
                "closing_axis_ee": [1.0, 0.0, 0.0],
                "ik_orientation_guard": {
                    "enabled": True,
                    "approach_axis_ee": [0.0, 0.0, 1.0],
                    "closing_axis_180_symmetric": True,
                    "max_approach_error_deg": 25.0,
                    "max_closing_error_deg": 20.0,
                },
            }
        }
        self.seeds: list[JointState | None] = []

    def _solve_grasp_ik_fk(
        self,
        _pose,
        _goal_handle,
        _deadline,
        seed,
        *,
        validate_orientation,
        ik_client=None,
        fk_client=None,
    ):
        del validate_orientation, ik_client, fk_client
        self.seeds.append(seed)
        if len(self.seeds) == 1:
            return IKPayload(
                joint_state=_joint_state(0.0),
                ee_xyz=(0.0, 0.0, 0.0),
                ee_quaternion=quaternion_from_matrix(euler_xyz_matrix((0.0, 0.0, -math.pi / 4.0))),
                approach_axis_error_deg=0.0,
                closing_axis_error_deg=45.0,
            )
        corrected_joint5 = self._joint_position(seed, "5")
        assert corrected_joint5 is not None
        return IKPayload(
            joint_state=_joint_state(corrected_joint5),
            ee_xyz=(0.0, 0.0, 0.0),
            ee_quaternion=(0.0, 0.0, 0.0, 1.0),
            approach_axis_error_deg=0.0,
            closing_axis_error_deg=0.0,
        )


def test_orientation_guard_reseeds_joint5_until_fk_axes_match():
    harness = _OrientationHarness()
    pose = Pose()
    pose.orientation.w = 1.0

    payload = harness._solve_orientation_consistent_grasp_ik_fk(pose, None, 0.0, _joint_state(0.0))

    assert len(harness.seeds) == 2
    assert harness.seeds[1] is not None
    corrected_joint5 = harness._joint_position(harness.seeds[1], "5")
    assert corrected_joint5 is not None
    assert math.isclose(corrected_joint5, math.pi / 4.0, abs_tol=1e-8)
    assert payload.closing_axis_error_deg == 0.0


class _BranchFilterHarness:
    _orientation_guard = PickExecutorNode._orientation_guard
    _joint5_branch_filter_config = PickExecutorNode._joint5_branch_filter_config
    _filter_joint5_branch_divergence = PickExecutorNode._filter_joint5_branch_divergence
    _joint_position = staticmethod(PickExecutorNode._joint_position)

    def __init__(self, *, enabled: bool = True, threshold: float | None = None) -> None:
        guard: dict = {
            "enabled": True,
            "approach_axis_ee": [0.0, 0.0, 1.0],
            "closing_axis_180_symmetric": True,
            "joint5_abs_max": 2.0,
            "joint5_branch_filter": enabled,
        }
        if threshold is not None:
            guard["joint5_branch_max_delta_rad"] = threshold
        self._config = {"target_gripper": {"ik_orientation_guard": guard}}


def test_branch_filter_rejects_cross_branch_candidate():
    harness = _BranchFilterHarness()
    seed = _joint_state(0.0)
    solution = _joint_state(math.pi)

    with pytest.raises(PickFlowError) as exc_info:
        harness._filter_joint5_branch_divergence(0, seed, solution)
    assert exc_info.value.code == "IK_JOINT5_BRANCH_DIVERGENCE"


def test_branch_filter_passes_same_branch_candidate():
    harness = _BranchFilterHarness()
    seed = _joint_state(0.4)
    solution = _joint_state(0.5)

    harness._filter_joint5_branch_divergence(0, seed, solution)


def test_branch_filter_disabled_does_not_raise():
    harness = _BranchFilterHarness(enabled=False)
    seed = _joint_state(0.0)
    solution = _joint_state(math.pi)

    harness._filter_joint5_branch_divergence(0, seed, solution)


def test_branch_filter_respects_custom_threshold():
    harness = _BranchFilterHarness(threshold=0.05)
    seed = _joint_state(0.0)
    solution = _joint_state(0.1)

    with pytest.raises(PickFlowError) as exc_info:
        harness._filter_joint5_branch_divergence(0, seed, solution)
    assert exc_info.value.code == "IK_JOINT5_BRANCH_DIVERGENCE"


def test_branch_filter_skips_when_seed_missing():
    harness = _BranchFilterHarness()
    solution = _joint_state(math.pi)

    harness._filter_joint5_branch_divergence(0, None, solution)


def test_branch_filter_skips_when_joint5_missing():
    harness = _BranchFilterHarness()
    seed = JointState()
    seed.name = ["1", "2", "3", "4"]
    seed.position = [0.0, 0.0, 0.0, 0.0]
    solution = _joint_state(math.pi)

    harness._filter_joint5_branch_divergence(0, seed, solution)


def test_branch_filter_rejects_negative_branch_jump():
    harness = _BranchFilterHarness()
    seed = _joint_state(0.5)
    solution = _joint_state(-1.2)

    with pytest.raises(PickFlowError) as exc_info:
        harness._filter_joint5_branch_divergence(0, seed, solution)
    assert exc_info.value.code == "IK_JOINT5_BRANCH_DIVERGENCE"


def test_branch_filter_rejects_invalid_threshold():
    harness = _BranchFilterHarness(threshold=-0.5)

    with pytest.raises(PickFlowError) as exc_info:
        harness._filter_joint5_branch_divergence(0, _joint_state(0.0), _joint_state(0.1))
    assert exc_info.value.code == "INVALID_GRASP_CONFIG"


class _TransitPoseHarness:
    _pose = staticmethod(PickExecutorNode._pose)
    _validate_candidate_transit_poses = PickExecutorNode._validate_candidate_transit_poses

    def __init__(self) -> None:
        self._config = {
            "ik": {"check_orientation": False},
            "target_gripper": {"ik_orientation_guard": {"enabled": True}},
        }
        self._workspace = {
            "x": [-1.0, 1.0],
            "y": [-1.0, 1.0],
            "z": [-1.0, 1.0],
        }
        self.labels: list[tuple[float, float, float]] = []

    def _orientation_guard(self):
        return self._config["target_gripper"]["ik_orientation_guard"]

    def _solve_orientation_consistent_grasp_ik_fk(self, pose, *_args, **_kwargs):
        xyz = (pose.position.x, pose.position.y, pose.position.z)
        self.labels.append(xyz)
        if xyz[2] > 0.15:
            raise PickFlowError("IK_ORIENTATION_REJECTED", "FK approach axis mismatch", retryable=True)
        return IKPayload(
            joint_state=_joint_state(0.1),
            ee_xyz=xyz,
            ee_quaternion=(0.0, 0.0, 0.0, 1.0),
            approach_axis_error_deg=0.0,
            closing_axis_error_deg=0.0,
        )

    def _validate_joint5_branch_continuity(self, _seed, _solution):
        return None


def test_transit_orientation_is_rejected_during_candidate_preparation():
    harness = _TransitPoseHarness()
    plan = type(
        "Plan",
        (),
        {
            "approach": (0.1, 0.0, 0.2),
            "lift": (0.1, 0.0, 0.1),
            "quaternion": (0.0, 0.0, 0.0, 1.0),
        },
    )()
    grasp_payload = IKPayload(
        joint_state=_joint_state(0.0),
        ee_xyz=(0.1, 0.0, 0.05),
        ee_quaternion=(0.0, 0.0, 0.0, 1.0),
        approach_axis_error_deg=0.0,
        closing_axis_error_deg=0.0,
    )

    with pytest.raises(PickFlowError) as exc_info:
        harness._validate_candidate_transit_poses(4, plan, grasp_payload, None, 0.0)

    assert exc_info.value.code == "IK_ORIENTATION_REJECTED"
    assert "candidate 4 approach" in str(exc_info.value)
    assert harness.labels == [(0.1, 0.0, 0.2)]
