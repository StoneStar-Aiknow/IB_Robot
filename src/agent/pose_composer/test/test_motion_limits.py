"""Commissioning refuses excessive excursions, including detours and hand motion."""

from types import SimpleNamespace

import numpy as np
import pytest

from pose_composer import kin
from pose_composer.model import ARM_JOINTS, load
from pose_composer.motion_limits import check_displacement


def plan_for(*poses):
    return SimpleNamespace(traj=SimpleNamespace(positions=np.array([[q[n] for n in ARM_JOINTS] for q in poses])))


def test_a_small_excursion_is_allowed_but_a_large_detour_is_refused():
    start = load().rest
    small = dict(start, left_elbow_joint=start["left_elbow_joint"] + 0.01)
    far = dict(start, left_elbow_joint=start["left_elbow_joint"] + 0.2)
    assert not check_displacement(plan_for(start, small), start, 0.06, 0.025)
    assert "joint displacement" in check_displacement(plan_for(start, far, small), start, 0.06, 0.025)


def test_hand_displacement_is_independent_of_the_joint_limit():
    start = load().rest
    target = dict(start, left_elbow_joint=start["left_elbow_joint"] + 0.1)
    points = kin.fk(target).points
    distance = np.linalg.norm(points["left_hand"] - kin.fk(start).points["left_hand"])
    assert distance > 0.01
    assert "hand displacement" in check_displacement(plan_for(start, target), start, 0.2, 0.001)


def test_disabled_limits_preserve_existing_planning():
    start = load().rest
    far = dict(start, left_elbow_joint=start["left_elbow_joint"] + 0.2)
    assert not check_displacement(plan_for(start, far), start)


@pytest.mark.parametrize("limit", [-1, float("nan"), float("inf")])
def test_invalid_limits_are_refused(limit):
    start = load().rest
    with pytest.raises(ValueError):
        check_displacement(plan_for(start), start, limit)


def test_nonfinite_measurements_are_refused():
    start = dict(load().rest, left_elbow_joint=float("nan"))
    assert check_displacement(plan_for(start), load().rest, 0.06)
