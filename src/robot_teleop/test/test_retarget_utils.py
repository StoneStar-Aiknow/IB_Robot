import math

import pytest

from robot_teleop.devices.retarget_utils import percentile, validate_joint_limits


def test_percentile_interpolates_shared_retarget_samples():
    assert percentile([0.0, 10.0, 20.0], 0.25) == pytest.approx(5.0)
    assert percentile([20.0, 0.0, 10.0], 0.5) == pytest.approx(10.0)


@pytest.mark.parametrize(("values", "quantile"), (([], 0.5), ([1.0], -0.1), ([1.0], 1.1), ([math.nan], 0.5)))
def test_percentile_rejects_invalid_inputs(values, quantile):
    with pytest.raises(ValueError, match="Percentile"):
        percentile(values, quantile)


def test_validate_joint_limits_normalizes_complete_channel_ranges():
    assert validate_joint_limits(
        ("thumb", "index"),
        {"thumb": {"min": 0, "max": 1}, "index": {"min": -1, "max": 2}},
    ) == {"thumb": (0.0, 1.0), "index": (-1.0, 2.0)}
