"""A TrajOpt rejection is rescued only if it reached the goal within limits; collision stays the fine check's."""

from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.joints import LEFT_ARM
from aimdk_grasp.planner import LeftArmPlanner

torch = pytest.importorskip("torch")


def planner(valid=True):
    value = object.__new__(LeftArmPlanner)
    value._fine_rescue = True
    bound = torch.tensor([[-1.0] * 7, [1.0] * 7])
    value.motion = SimpleNamespace(
        kinematics=SimpleNamespace(
            config=SimpleNamespace(
                kinematics_config=SimpleNamespace(
                    joint_limits=SimpleNamespace(velocity=bound, acceleration=bound * 10, jerk=bound * 100)
                )
            )
        ),
        trajopt_solver=SimpleNamespace(config=SimpleNamespace(interpolation_dt=0.02)),
    )
    value.locked_positions = {}
    value.valid_positions = lambda positions: np.full(len(positions), valid)
    value.violations = lambda positions: "world 1.0 mm at sample 2"
    return value


def result(*, success=False, position_error=0.001, velocity=0.5, steps=5):
    position = torch.linspace(0, 0.1, steps)[:, None].repeat(1, 7)[None]
    trajectory = SimpleNamespace(
        joint_names=list(LEFT_ARM),
        position=position,
        velocity=torch.full_like(position, velocity),
        acceleration=torch.zeros_like(position),
        jerk=torch.zeros_like(position),
    )
    return SimpleNamespace(
        success=torch.tensor([[success]]),
        interpolated_trajectory=trajectory,
        interpolated_last_tstep=torch.tensor([[steps - 1]]),
        position_error=torch.tensor([[position_error]]),
        rotation_error=torch.tensor([[0.01]]),
        position_tolerance=0.005,
        orientation_tolerance=0.05,
    )


def test_rejection_that_reached_the_goal_within_limits_is_rescued():
    p = planner()
    assert p._rescuable(result())
    assert not p._rescuable(result(success=True))  # nothing to rescue
    assert not p._rescuable(result(position_error=0.02))  # goal not reached
    assert not p._rescuable(result(velocity=1.5))  # velocity limit violated
    assert not p._rescuable(None)
    p._fine_rescue = False
    assert not p._rescuable(result())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("position_error", torch.tensor([[float("nan")]])),
        ("rotation_error", torch.tensor([[float("inf")]])),
        ("position_tolerance", float("nan")),
        ("orientation_tolerance", torch.tensor([float("inf")])),
    ],
)
def test_nonfinite_goal_errors_and_tolerances_are_not_rescuable(field, value):
    rejected = result()
    setattr(rejected, field, value)
    assert not planner()._rescuable(rejected)


def test_rescued_result_still_needs_the_fine_collision_check():
    rescued = result()
    rescued.aimdk_fine_rescue = True
    trajectory = planner(valid=True)._grasp_trajectory(rescued, "approach")
    assert len(trajectory.times) == 5
    with pytest.raises(RuntimeError, match="fine collision"):
        planner(valid=False)._grasp_trajectory(rescued, "approach")
    with pytest.raises(RuntimeError, match="rejected"):  # unrescued rejection
        planner(valid=True)._grasp_trajectory(result(), "approach")
