import numpy as np
import pytest

from object_tracker.kalman import ConstantVelocityFilter


def test_filter_accepts_consistent_measurement_and_estimates_velocity():
    tracker = ConstantVelocityFilter((0.0, 0.0))
    measurement_covariance = np.diag([0.01, 0.01])

    tracker.predict(1.0)
    update = tracker.update((1.0, 0.0), measurement_covariance)

    assert update.accepted
    assert tracker.state[0] > 0.9
    assert tracker.state[2] > 0.5


def test_filter_rejects_large_innovation_without_state_jump():
    tracker = ConstantVelocityFilter((0.0, 0.0))
    tracker.predict(0.1)
    predicted = tracker.state.copy()

    update = tracker.update((20.0, 20.0), np.diag([0.01, 0.01]))

    assert not update.accepted
    assert update.reason == "innovation_gate"
    np.testing.assert_allclose(tracker.state, predicted)


def test_prediction_grows_position_covariance_and_rejects_invalid_dt():
    tracker = ConstantVelocityFilter((0.0, 0.0))
    initial_variance = tracker.position_variance

    tracker.predict(0.5)

    assert tracker.position_variance > initial_variance
    with pytest.raises(ValueError, match="dt"):
        tracker.predict(0.0)
