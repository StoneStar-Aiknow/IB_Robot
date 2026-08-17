import numpy as np

from ibrobot_msgs.msg import TrackState
from object_tracker.motion import EgoCompensatedMotionClassifier, MotionState
from object_tracker.target_tracker_node import TargetTrackerNode


def _update(classifier, positions, *, robot_speed=0.0):
    estimates = []
    for index, position in enumerate(positions):
        estimates.append(
            classifier.update(
                stamp_s=float(index),
                position_odom=position,
                position_covariance=np.diag([0.001, 0.001]),
                robot_linear_speed_mps=robot_speed,
            )
        )
    return estimates


def test_robot_motion_does_not_mark_static_object_as_moving():
    classifier = EgoCompensatedMotionClassifier(
        window_samples=3,
        moving_confirmation_windows=1,
        stationary_confirmation_windows=1,
    )

    robot_positions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    camera_positions = [(1.0 - robot_x, 1.0) for robot_x in robot_positions]
    odom_positions = [
        (robot_x + camera_x, camera_y)
        for robot_x, (camera_x, camera_y) in zip(robot_positions, camera_positions, strict=True)
    ]
    estimates = _update(classifier, odom_positions, robot_speed=0.3)

    assert estimates[-1].state is MotionState.STATIONARY
    assert estimates[-1].ego_motion_active
    assert estimates[-1].speed_mps == 0.0


def test_static_robot_and_moving_object_reach_moving_state_after_confirmation():
    classifier = EgoCompensatedMotionClassifier(
        window_samples=3,
        min_displacement_m=0.05,
        min_speed_mps=0.05,
        moving_confirmation_windows=2,
    )

    estimates = _update(classifier, [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0)])

    assert estimates[2].state is MotionState.UNKNOWN
    assert estimates[3].state is MotionState.MOVING
    assert estimates[-1].state is MotionState.MOVING
    assert estimates[-1].speed_mps > 0.05


def test_depth_and_localization_uncertainty_fails_closed():
    classifier = EgoCompensatedMotionClassifier(window_samples=3, max_position_variance_m2=0.01)
    estimate = classifier.update(
        stamp_s=0.0,
        position_odom=(1.0, 1.0),
        position_covariance=np.diag([0.02, 0.001]),
        robot_angular_speed_rps=0.2,
    )

    assert estimate.state is MotionState.UNKNOWN
    assert estimate.ego_motion_active
    assert "covariance" in estimate.reason


def test_real_banana_observations_are_classified_as_independent_motion():
    positions = [
        (0.2740, -1.2283),
        (-0.0104, -1.2256),
        (-0.0863, -1.2152),
        (-0.0188, -1.1161),
        (0.0653, -0.9652),
        (0.0623, -0.9658),
        (0.2360, -0.9612),
        (0.2365, -0.9581),
        (0.2200, -0.9576),
        (-0.0760, -0.9541),
        (-0.0658, -0.9437),
        (0.1047, -1.2327),
        (0.1072, -1.2309),
        (0.1031, -1.2319),
    ]
    classifier = EgoCompensatedMotionClassifier(max_sample_gap_s=3.0)
    estimates = [
        classifier.update(
            stamp_s=index * 2.0,
            position_odom=position,
            position_covariance=np.diag([0.001, 0.001]),
        )
        for index, position in enumerate(positions)
    ]

    assert any(estimate.state is MotionState.MOVING for estimate in estimates)
    assert max(estimate.displacement_m for estimate in estimates) > 0.3


def test_unknown_motion_marks_track_state_non_actionable():
    classifier = EgoCompensatedMotionClassifier()
    estimate = classifier.update(
        stamp_s=0.0,
        position_odom=(1.0, 1.0),
        position_covariance=np.diag([0.001, 0.001]),
    )
    state = TrackState()
    state.actionable = True

    TargetTrackerNode.apply_motion_estimate(state, estimate)

    assert state.motion_state == TrackState.MOTION_UNKNOWN
    assert not state.actionable
    assert state.state_reason == "collecting motion window"
