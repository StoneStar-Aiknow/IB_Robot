"""Import-guard and converter-contract tests for ``unitree_g1_adapter.vendor``.

The vendor SDK is not installed in this workspace's venv, so the guard path is
what runs here; the real-vendor converter smoke is exercised wherever the
pinned SDK is installed (see the ``pytest.mark`` guard below).
"""

from __future__ import annotations

import pytest
from unitree_g1_adapter import vendor
from unitree_g1_adapter.errors import UnitreeSdkUnavailableError


def test_vendor_availability_is_a_plain_boolean():
    assert isinstance(vendor.vendor_available(), bool)


def test_idl_type_rejects_unknown_keys_without_touching_the_vendor():
    with pytest.raises(KeyError, match="bogus"):
        vendor.idl_type("bogus")


@pytest.mark.skipif(not vendor.vendor_available(), reason="unitree_sdk2py is not installed")
def test_idl_type_resolves_the_four_hg_classes():
    for key in ("low_cmd", "low_state", "hand_cmd", "hand_state"):
        assert vendor.idl_type(key) is not None


def test_lowcmd_slot_count_is_enforced_only_with_the_vendor():
    """Without the SDK the converter fails with the guided error.

    The 35-slot contract itself is asserted by the real-vendor smoke below.
    """
    if vendor.vendor_available():
        pytest.skip("guard-path assertion applies only without the SDK")
    with pytest.raises(UnitreeSdkUnavailableError, match="setup.sh"):
        vendor.lowcmd_from_slots([])


@pytest.mark.skipif(not vendor.vendor_available(), reason="unitree_sdk2py is not installed")
def test_lowcmd_rejects_wrong_slot_counts():
    with pytest.raises(ValueError, match="fixed at 35"):
        vendor.lowcmd_from_slots([])
    with pytest.raises(ValueError, match="fixed at 35"):
        vendor.lowcmd_from_slots([{"q": 0.0}] * 36)


@pytest.mark.skipif(not vendor.vendor_available(), reason="unitree_sdk2py is not installed")
def test_lowcmd_builds_35_finite_slots():
    slots = [{"mode": 1, "q": 0.1 * index, "dq": 0.0, "tau": 0.0, "kp": 60.0, "kd": 1.5} for index in range(35)]
    message = vendor.lowcmd_from_slots(slots, mode_pr=1, mode_machine=5)
    assert len(message.motor_cmd) == 35
    assert message.motor_cmd[0].q == pytest.approx(0.0)
    assert message.motor_cmd[1].q == pytest.approx(0.1)
    assert message.mode_pr == 1
    assert message.mode_machine == 5
    assert message.crc == vendor.CRC().Crc(message)


@pytest.mark.skipif(not vendor.vendor_available(), reason="unitree_sdk2py is not installed")
def test_lowcmd_rejects_non_finite_targets():
    slots = [{"mode": 1, "q": float("nan"), "dq": 0.0, "tau": 0.0, "kp": 0.0, "kd": 0.0}] * 35
    with pytest.raises(ValueError, match="finite"):
        vendor.lowcmd_from_slots(slots)


@pytest.mark.skipif(not vendor.vendor_available(), reason="unitree_sdk2py is not installed")
def test_handcmd_builds_one_motor_per_slot_and_refuses_empty():
    with pytest.raises(ValueError, match="at least one"):
        vendor.handcmd_from_slots([])
    message = vendor.handcmd_from_slots([{"q": 0.5, "kp": 1.5, "kd": 0.1}] * 3)
    assert len(message.motor_cmd) == 3
    assert message.motor_cmd[0].q == pytest.approx(0.5)
