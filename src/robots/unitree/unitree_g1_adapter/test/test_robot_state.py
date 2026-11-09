"""Read-only service discovery; never create a DDS participant."""

from unittest.mock import Mock

import pytest
from fake_vendor import make_fake_vendor
from unitree_g1_adapter import vendor
from unitree_g1_adapter.clients import RobotStateClient


def test_service_list_preserves_status_and_timeout():
    fake = make_fake_vendor()
    client = RobotStateClient(vendor=fake)
    assert client.service_list().data == [{"name": "g1_arm_example", "status": 1}]
    fake["_state"]["clients"]["robot_state"].outcomes["ServiceList"] = (3104, None)
    result = client.service_list()
    assert not result.ok and result.timed_out and result.data is None


@pytest.mark.skipif(not vendor.vendor_available(), reason="official SDK unavailable")
def test_pinned_b2_service_list_protocol_uses_official_transport():
    client = object.__new__(vendor.RobotStateClient)
    client._Call = Mock(return_value=(0, '[{"name":"g1_arm_example","status":1}]'))
    assert client.ServiceList() == (0, [{"name": "g1_arm_example", "status": 1}])
    client._Call.assert_called_once_with(1003, "{}")
    client._Call.reset_mock()
    client._Call.return_value = (3104, "")
    assert client.ServiceList() == (3104, None)
    client._Call.assert_called_once()
