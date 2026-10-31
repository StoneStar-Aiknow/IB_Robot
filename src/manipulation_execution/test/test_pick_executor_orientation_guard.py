import math

import pytest
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from manipulation_execution.grasp_geometry import euler_xyz_matrix, quaternion_from_matrix
from manipulation_execution.pick_executor_node import IKPayload, PickExecutorNode, PickFlowError


class _FakeWristGuard:
    """Scripted wrist-guard provider that records how the node calls it.

    These tests cover the generic node's use of a provider; the SO-101 joint-5
    semantics belong to so101_suite/test/test_wrist_guard.py.
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.closing_axis_correction = math.pi / 4.0
        self.within_limit = True
        self.branch_continuous = True

    def joint5_closing_axis_correction(
        self, target_quaternion, actual_quaternion, approach_axis_ee, closing_axis_ee, *, closing_axis_180_symmetric
    ):
        self.calls.append(
            ("closing_axis_correction", list(approach_axis_ee), list(closing_axis_ee), closing_axis_180_symmetric)
        )
        return self.closing_axis_correction

    def canonicalize_joint5(self, joint5, center=0.0):
        self.calls.append(("canonicalize", joint5, center))
        return joint5

    def joint5_within_abs_limit(self, joint5, limit, *, center, epsilon):
        self.calls.append(("within_abs_limit", joint5, limit, center, epsilon))
        return self.within_limit

    def joint5_branch_continuity_check(self, seed_joint5, solution_joint5, threshold):
        self.calls.append(("branch_continuity", seed_joint5, solution_joint5, threshold))
        return self.branch_continuous


def _joint_state(joint5: float) -> JointState:
    state = JointState()
    state.name = ["1", "2", "3", "4", "5"]
    state.position = [0.0, 0.0, 0.0, 0.0, joint5]
    return state


class _OrientationHarness:
    _load_json_object = staticmethod(PickExecutorNode._load_json_object)
    _load_home_joint_positions = classmethod(PickExecutorNode._load_home_joint_positions.__func__)
    _validate_home_joint_config = PickExecutorNode._validate_home_joint_config
    _validate_provider_requirements = PickExecutorNode._validate_provider_requirements
    _orientation_guard = PickExecutorNode._orientation_guard
    _orientation_limits = PickExecutorNode._orientation_limits
    _joint5_home_constraints = PickExecutorNode._joint5_home_constraints
    _joint_position = staticmethod(PickExecutorNode._joint_position)
    _joint_state_with_joint5 = staticmethod(PickExecutorNode._joint_state_with_joint5)
    _joint5_home_center = PickExecutorNode._joint5_home_center
    _validate_joint5 = PickExecutorNode._validate_joint5
    _validate_joint5_branch_continuity = PickExecutorNode._validate_joint5_branch_continuity
    _pose_components = staticmethod(PickExecutorNode._pose_components)
    _solve_orientation_consistent_grasp_ik_fk = PickExecutorNode._solve_orientation_consistent_grasp_ik_fk

    def __init__(self) -> None:
        self._home_joint_positions = {"5": 0.0}
        self._wrist_guard = _FakeWristGuard()
        self._grasp_geometry = None
        self._target_geometry = {}
        self._config = {
            "target_gripper": {
                "closing_axis_ee": [1.0, 0.0, 0.0],
                "ik_orientation_guard": {
                    "enabled": True,
                    "approach_axis_ee": [0.0, 0.0, 1.0],
                    "closing_axis_180_symmetric": True,
                    "max_approach_error_deg": 25.0,
                    "max_closing_error_deg": 27.0,
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
    assert harness._wrist_guard.calls == [
        ("closing_axis_correction", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], True),
        ("canonicalize", math.pi / 4.0, 0.0),
    ]


def test_home_joint_positions_are_loaded_from_launch_json():
    assert _OrientationHarness._load_home_joint_positions('{"1": 0, "5": 0.4}') == {"1": 0.0, "5": 0.4}


def test_home_joint_positions_reject_non_finite_values():
    with pytest.raises(ValueError, match="joint 5 must be finite"):
        _OrientationHarness._load_home_joint_positions('{"5": NaN}')


def test_orientation_guard_without_wrist_guard_provider_fails_fast():
    harness = _OrientationHarness()
    harness._wrist_guard = None
    harness._config["target_gripper"]["ik_orientation_guard"]["joint5_constraints_enabled"] = True

    with pytest.raises(ValueError, match="wrist_guard_provider is required"):
        harness._validate_provider_requirements()


def test_tabletop_filter_without_grasp_geometry_provider_fails_fast():
    harness = _OrientationHarness()
    harness._target_geometry = {"tabletop_filter": True}

    with pytest.raises(ValueError, match="grasp_geometry_provider is required"):
        harness._validate_provider_requirements()


def test_disabled_features_do_not_require_providers():
    harness = _OrientationHarness()
    harness._wrist_guard = None
    harness._grasp_geometry = None
    harness._config["target_gripper"]["ik_orientation_guard"]["enabled"] = False
    harness._target_geometry = {"tabletop_filter": False}

    harness._validate_provider_requirements()


def test_joint5_guard_requires_home_position_during_startup_validation():
    harness = _OrientationHarness()
    harness._home_joint_positions = {}
    harness._config["target_gripper"]["ik_orientation_guard"]["joint5_constraints_enabled"] = True

    with pytest.raises(ValueError, match=r'reset_positions\["5"\] is required'):
        harness._validate_home_joint_config()


def _joint5_guard_harness(*, enabled: bool = True, stage_continuity: bool = True) -> _OrientationHarness:
    harness = _OrientationHarness()
    harness._home_joint_positions = {"5": 0.4}
    harness._config["target_gripper"]["ik_orientation_guard"].update(
        {
            "joint5_constraints_enabled": enabled,
            "joint5_home_max_delta_rad": 2.0,
            "joint5_limit_epsilon_rad": 0.001,
            "joint5_stage_continuity": stage_continuity,
            "joint5_stage_max_delta_rad": 0.25,
        }
    )
    return harness


def test_joint5_runtime_guard_passes_home_center_limit_and_epsilon_to_the_provider():
    harness = _joint5_guard_harness()

    assert harness._validate_joint5(_joint_state(2.401)) == pytest.approx(2.401)
    assert harness._wrist_guard.calls == [("within_abs_limit", 2.401, 2.0, 0.4, 0.001)]


def test_joint5_runtime_guard_rejects_what_the_provider_rejects():
    harness = _joint5_guard_harness()
    harness._wrist_guard.within_limit = False

    with pytest.raises(PickFlowError) as exc_info:
        harness._validate_joint5(_joint_state(0.4 + 2.0011))

    assert exc_info.value.code == "IK_JOINT5_LIMIT"


def test_joint5_runtime_guard_can_be_disabled_from_config():
    harness = _joint5_guard_harness(enabled=False)

    assert harness._validate_joint5(_joint_state(10.0)) is None
    assert harness._wrist_guard.calls == []


def test_joint5_stage_continuity_uses_independent_configured_threshold():
    harness = _joint5_guard_harness()
    harness._wrist_guard.branch_continuous = False

    with pytest.raises(PickFlowError) as exc_info:
        harness._validate_joint5_branch_continuity(_joint_state(0.4), _joint_state(0.66))

    assert exc_info.value.code == "IK_JOINT5_BRANCH_CHANGED"
    assert harness._wrist_guard.calls == [("branch_continuity", 0.4, 0.66, 0.25)]


def test_joint5_stage_continuity_can_be_disabled_independently():
    harness = _joint5_guard_harness(stage_continuity=False)

    harness._validate_joint5_branch_continuity(_joint_state(0.4), _joint_state(2.4))
    assert harness._wrist_guard.calls == []
