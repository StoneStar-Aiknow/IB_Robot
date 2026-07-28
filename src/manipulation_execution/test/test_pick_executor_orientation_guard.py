import math

from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from manipulation_execution.grasp_geometry import euler_xyz_matrix, quaternion_from_matrix
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
