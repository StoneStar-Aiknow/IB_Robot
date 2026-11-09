"""Validate signed MC query values using the real generated vendor message."""

from unittest.mock import Mock

import pytest

pytest.importorskip("aimdk_msgs", reason="AimDK overlay not sourced")

from aimdk_msgs.msg import McLocomotionSpeedStatus  # noqa: E402
from aimdk_robot.mc_queries import _AsyncServiceCaller, speed_profile  # noqa: E402
from rclpy.task import Future  # noqa: E402


def test_speed_profiles_preserve_axis_direction_and_asymmetry():
    status = McLocomotionSpeedStatus()
    status.speed_mode.value = 4
    status.forward_bounds.min_value, status.forward_bounds.max_value = -0.2, 0.8
    status.lateral_bounds.min_value, status.lateral_bounds.max_value = -0.1, 0.15
    status.angular_bounds.min_value, status.angular_bounds.max_value = -0.3, 0.4
    answer = speed_profile(status)
    assert answer.mode == 4
    assert list(answer.forward) == pytest.approx([-0.2, 0.8])
    assert list(answer.lateral) == pytest.approx([-0.1, 0.15])
    assert list(answer.angular) == pytest.approx([-0.3, 0.4])


@pytest.mark.parametrize("minimum,maximum", [(float("nan"), 1.0), (-1.0, float("inf")), (0.5, -0.5)])
def test_speed_profiles_refuse_invalid_platform_limits(minimum, maximum):
    status = McLocomotionSpeedStatus()
    status.forward_bounds.min_value, status.forward_bounds.max_value = minimum, maximum
    with pytest.raises(ValueError, match="forward"):
        speed_profile(status)


def test_cancelled_query_cleans_up_its_vendor_future():
    future = Future()
    client = Mock()
    client.call_async.return_value = future
    caller = _AsyncServiceCaller(None, client, attempts=2, timeout_s=0.5)
    request = object()
    query = caller.call(request)
    assert query.send(None) is future
    query.close()
    assert future.cancelled()
    client.remove_pending_request.assert_called_once_with(future)


def test_query_transport_error_is_reported_and_cleaned_up():
    future = Future()
    client = Mock()
    client.call_async.return_value = future
    caller = _AsyncServiceCaller(None, client, attempts=2, timeout_s=0.5)
    query = caller.call(object())
    assert query.send(None) is future
    future.set_exception(RuntimeError("transport failed"))
    with pytest.raises(StopIteration) as completed:
        query.send(None)
    assert not completed.value.value.ok
    assert "transport failed" in completed.value.value.message
    client.remove_pending_request.assert_called_once_with(future)
