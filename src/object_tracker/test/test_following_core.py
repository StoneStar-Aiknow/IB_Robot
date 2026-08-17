import math

import numpy as np
import pytest

from object_tracker.following_core import PathReplacementGate, should_replan, stand_off_goal


def test_stand_off_goal_faces_object_at_requested_distance():
    goal = stand_off_goal(np.array([2.0, 0.0]), np.array([0.0, 0.0]), 0.8)

    assert goal.position == pytest.approx([1.2, 0.0])
    assert goal.yaw == pytest.approx(0.0)


def test_replan_is_debounced_until_distance_or_heading_changes():
    previous = stand_off_goal(np.array([2.0, 0.0]), np.array([0.0, 0.0]), 0.8)
    current = stand_off_goal(np.array([2.1, 0.0]), np.array([0.0, 0.0]), 0.8)

    assert not should_replan(
        previous,
        current,
        displacement_m=0.2,
        heading_delta_rad=math.radians(15.0),
        elapsed_s=2.0,
        minimum_interval_s=1.0,
    )
    moved = stand_off_goal(np.array([2.3, 0.0]), np.array([0.0, 0.0]), 0.8)
    assert should_replan(
        previous,
        moved,
        displacement_m=0.2,
        heading_delta_rad=math.radians(15.0),
        elapsed_s=2.0,
        minimum_interval_s=1.0,
    )


def test_prediction_only_target_does_not_start_or_refresh_path():
    current = stand_off_goal(np.array([2.0, 0.0]), np.array([0.0, 0.0]), 0.8)
    assert not should_replan(
        None,
        current,
        displacement_m=0.2,
        heading_delta_rad=0.2,
        elapsed_s=2.0,
        minimum_interval_s=1.0,
        prediction_only=True,
    )


def test_path_replacement_keeps_only_latest_pending_path():
    gate = PathReplacementGate()
    gate.activate("path-1")
    assert gate.request_replacement("path-2")
    assert gate.request_replacement("path-3")
    assert gate.on_active_terminal() == "path-3"
    gate.activate("path-3")
    assert gate.active_path == "path-3"


def test_path_replacement_fails_closed():
    gate = PathReplacementGate()
    gate.activate("path-1")
    gate.fail_closed()

    with pytest.raises(RuntimeError):
        gate.request_replacement("path-2")
