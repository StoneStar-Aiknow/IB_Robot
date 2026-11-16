"""The per-run preset commissioning gate: no profile, no auto mode.

The physical arm-preset path must be authorised for exactly one launch. These
tests exercise the same pure rule the launch entry and the runtime node call,
so a stale ``confirmed: true`` profile, a loopback NIC or an unset FSM cannot
turn into a silent default.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from unitree_g1_robot.preset_commissioning import (
    PRESET_HARDWARE_DISABLED,
    resolve_preset_commissioning,
)

PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "g1_29dof_arm_presets.yaml"


def _profile(**validation) -> dict:
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    if validation:
        profile["vendor"]["operator_validation"] = validation
    return profile


def _resolve(profile=None, **overrides):
    arguments = {
        "confirmed": True,
        "runtime_network_interface": "eth9",
        "expected_fsm_id": 802,
        "vendor_network_interface": "eth9",
    }
    arguments.update(overrides)
    return resolve_preset_commissioning(profile or _profile(), **arguments)


def test_per_run_confirmation_and_explicit_binding_are_accepted():
    assert _resolve() == ("eth9", 802)


def test_an_audited_profile_fsm_is_accepted_when_no_explicit_value_is_given():
    assert _resolve(_profile(expected_fsm_id=500), expected_fsm_id=None) == ("eth9", 500)


def test_a_persisted_confirmed_profile_does_not_replace_the_per_run_flag():
    with pytest.raises(RuntimeError, match="not a per-run confirmation"):
        _resolve(_profile(confirmed=True), confirmed=False)


def test_confirmation_defaults_to_false():
    with pytest.raises(RuntimeError, match=PRESET_HARDWARE_DISABLED):
        _resolve(confirmed=None)


@pytest.mark.parametrize("nic", ["", "lo", "  "])
def test_the_robot_facing_nic_is_required_and_never_loopback(nic):
    with pytest.raises(RuntimeError, match="robot-facing NIC"):
        _resolve(runtime_network_interface=nic, vendor_network_interface=nic)


def test_the_vendor_bound_nic_must_match_the_confirmed_nic():
    with pytest.raises(RuntimeError, match="vendor_network_interface='lo'"):
        _resolve(vendor_network_interface="lo")


def test_a_profile_nic_must_agree_with_the_per_run_nic():
    with pytest.raises(RuntimeError, match="conflicting network interface overrides"):
        _resolve(_profile(network_interface="eth0"))


def test_an_unset_fsm_is_refused_instead_of_selected_automatically():
    with pytest.raises(RuntimeError, match="never selects an FSM automatically"):
        _resolve(expected_fsm_id=None)


def test_explicit_and_audited_fsm_must_agree():
    with pytest.raises(RuntimeError, match="conflicting expected FSM ids"):
        _resolve(_profile(expected_fsm_id=500), expected_fsm_id=802)


@pytest.mark.parametrize("value", [True, -1, 0, "abc", "0"])
def test_an_invalid_fsm_is_refused(value):
    with pytest.raises(RuntimeError, match="FSM id"):
        _resolve(expected_fsm_id=value)
