"""Facade lifecycle tests: singleton contract, resource tracking, teardown."""

from __future__ import annotations

import pytest
from fake_vendor import make_fake_vendor
from unitree_g1_adapter.errors import UnitreeSdkStateError
from unitree_g1_adapter.runtime import UnitreeSdk


def test_initialize_is_forwarded_once_and_conflicting_reinit_fails():
    surface = make_fake_vendor()
    sdk = UnitreeSdk(vendor=surface)

    sdk.initialize(0, "eno1")
    sdk.initialize(0, "eno1")  # same arguments: a no-op
    assert surface["_state"]["factory_calls"] == [(0, "eno1")]

    with pytest.raises(UnitreeSdkStateError, match="singleton"):
        sdk.initialize(1, "eno1")
    with pytest.raises(UnitreeSdkStateError, match="singleton"):
        sdk.initialize(0, None)


def test_resources_require_initialization():
    surface = make_fake_vendor()
    sdk = UnitreeSdk(vendor=surface)
    with pytest.raises(UnitreeSdkStateError, match="initialize"):
        sdk.state_feed("rt/lowstate", "low_state")
    with pytest.raises(UnitreeSdkStateError, match="initialize"):
        sdk.command_writer("rt/arm_sdk", "low_cmd")
    with pytest.raises(UnitreeSdkStateError, match="initialize"):
        sdk.loco(1.0)


def test_feeds_and_writers_are_tracked_and_closed_together():
    surface = make_fake_vendor()
    sdk = UnitreeSdk(vendor=surface)
    sdk.initialize(0, None)

    sdk.state_feed("rt/lowstate", "low_state")
    sdk.command_writer("rt/arm_sdk", "low_cmd")
    assert len(surface["_state"]["subscribers"]) == 1
    assert len(surface["_state"]["publishers"]) == 1

    sdk.close()
    assert surface["_state"]["subscribers"][0].closed == 1
    assert surface["_state"]["publishers"][0].closed == 1

    # Closed facade: no further resources, and double close stays safe.
    with pytest.raises(UnitreeSdkStateError, match="closed"):
        sdk.state_feed("rt/lowstate", "low_state")
    sdk.close()


def test_clients_are_cached_per_key_and_receive_the_timeout():
    surface = make_fake_vendor()
    sdk = UnitreeSdk(vendor=surface)
    sdk.initialize(0, None)

    loco = sdk.loco(2.5)
    assert sdk.loco(2.5) is loco
    assert loco._client.timeout == 2.5
    assert sdk.arm_action(1.5)._client.timeout == 1.5
    assert sdk.audio(1.0)._client.timeout == 1.0
    assert sdk.motion_switcher(1.0)._client.timeout == 1.0


def test_close_clears_initialization_so_reuse_needs_a_new_facade():
    surface = make_fake_vendor()
    sdk = UnitreeSdk(vendor=surface)
    sdk.initialize(0, None)
    sdk.close()
    assert sdk.initialized is False

    fresh = UnitreeSdk(vendor=surface)
    fresh.initialize(0, "eno1")
    assert surface["_state"]["factory_calls"] == [(0, None), (0, "eno1")]


def test_vendor_init_failure_is_wrapped_with_guidance():
    from unitree_g1_adapter.errors import UnitreeSdkInitError

    def broken_factory(domain_id: int, network_interface) -> None:
        raise Exception("channel factory init error.")

    surface = make_fake_vendor(channel_factory_initialize=broken_factory)
    sdk = UnitreeSdk(vendor=surface)

    with pytest.raises(UnitreeSdkInitError, match="ip link") as excinfo:
        sdk.initialize(0, "eno1")
    assert excinfo.value.domain_id == 0
    assert excinfo.value.network_interface == "eno1"
    # A failed initialization must not leave the facade half-initialized.
    assert sdk.initialized is False
