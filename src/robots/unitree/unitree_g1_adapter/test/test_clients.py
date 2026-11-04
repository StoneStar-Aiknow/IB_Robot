"""RPC client wrapper tests: result mapping, timeout sentinel, lifecycle."""

from __future__ import annotations

import pytest
from fake_vendor import RPC_OK, RPC_SEND_ERR, RPC_TIMEOUT, make_fake_vendor
from unitree_g1_adapter.clients import (
    ArmActionClient,
    AudioClient,
    LocoClient,
    MotionSwitcherClient,
)


def test_tuple_outcomes_map_onto_ok_results():
    surface = make_fake_vendor()
    client = LocoClient(vendor=surface)
    assert client._client.init_calls == 1
    assert client._client.timeout == 1.0  # vendor default, made explicit

    result = client.get_fsm_id()
    assert (result.ok, result.code, result.timed_out) == (True, RPC_OK, False)
    assert result.data == {"data": 5}


def test_bare_code_outcomes_map_with_no_data():
    surface = make_fake_vendor()
    client = LocoClient(vendor=surface)
    result = client.set_velocity(0.2, 0.0, 0.1, 1.0)
    assert (result.ok, result.code, result.data, result.timed_out) == (True, RPC_OK, None, False)
    assert client._client.velocity_arg == (0.2, 0.0, 0.1, 1.0)


def test_timeout_sentinel_is_reported_as_timed_out_not_ok():
    surface = make_fake_vendor()
    client = LocoClient(vendor=surface)
    client._client.outcomes["SetVelocity"] = RPC_TIMEOUT
    result = client.set_velocity(0.0, 0.0, 0.0)
    assert (result.ok, result.code, result.timed_out) == (False, RPC_TIMEOUT, True)


def test_other_vendor_codes_are_plain_failures():
    surface = make_fake_vendor()
    client = LocoClient(vendor=surface)
    client._client.outcomes["SetVelocity"] = RPC_SEND_ERR
    result = client.set_velocity(0.0, 0.0, 0.0)
    assert (result.ok, result.code, result.timed_out) == (False, RPC_SEND_ERR, False)


def test_timeout_must_be_positive():
    with pytest.raises(ValueError, match="timeout_s"):
        LocoClient(vendor=make_fake_vendor(), timeout_s=0.0)


def test_closed_wrappers_refuse_calls():
    surface = make_fake_vendor()
    client = LocoClient(vendor=surface)
    client.close()
    with pytest.raises(RuntimeError, match="closed"):
        client.set_velocity(0.0, 0.0, 0.0)


def test_motion_and_interaction_wrappers_pass_arguments_through():
    surface = make_fake_vendor()
    loco = LocoClient(vendor=surface)
    assert loco.set_fsm_id(706).ok is True
    assert loco._client.set_fsm_id_arg == 706
    assert loco.switch_to_user_ctrl().ok is True
    assert loco.switch_to_internal_ctrl("mode-object").ok is True
    assert loco._client.internal_ctrl_arg == "mode-object"

    arm = ArmActionClient(vendor=surface)
    assert arm.execute_action(27).ok is True
    assert arm._client.action_arg == 27
    assert arm.get_action_list().data == {"data": []}

    audio = AudioClient(vendor=surface)
    assert audio.tts_maker("hello", 0).ok is True
    assert audio._client.tts_arg == ("hello", 0)
    assert audio.led_control(1, 2, 3).ok is True
    assert audio._client.led_arg == (1, 2, 3)

    switcher = MotionSwitcherClient(vendor=surface)
    checked = switcher.check_mode()
    assert checked.ok is True and checked.data == {"data": {"name": "ai_sport"}}
    assert switcher.select_mode("ai_sport").ok is True
    assert switcher._client.select_arg == "ai_sport"
    assert switcher.release_mode().ok is True


def test_short_tuple_outcomes_do_not_crash_the_mapping():
    surface = make_fake_vendor()
    client = LocoClient(vendor=surface)
    client._client.outcomes["GetFsmId"] = (RPC_OK,)
    result = client.get_fsm_id()
    assert result.ok is True and result.data is None


def test_per_call_timeout_bounds_only_that_call_and_restores_the_default():
    surface = make_fake_vendor()
    client = LocoClient(vendor=surface, timeout_s=5.0)
    observed: list[float] = []

    def record_timeout():
        observed.append(client._client.timeout)
        return (RPC_OK, {"data": 5})

    client._client.GetFsmId = record_timeout

    with client.call_timeout(0.25):
        assert client.get_fsm_id().ok is True
        assert client._client.timeout == 0.25

    assert observed == [0.25], "the vendor must see the short timeout during the call"
    assert client._client.timeout == 5.0, "the configured default comes back"
    assert client.timeout_s == 5.0


def test_per_call_timeout_restores_the_default_when_the_call_raises():
    surface = make_fake_vendor()
    client = LocoClient(vendor=surface, timeout_s=5.0)
    client._client.outcomes["GetFsmId"] = RuntimeError("vendor exploded")

    with pytest.raises(RuntimeError, match="vendor exploded"), client.call_timeout(0.25):
        client.get_fsm_id()

    assert client._client.timeout == 5.0


def test_absent_per_call_timeout_leaves_the_configured_one_alone():
    surface = make_fake_vendor()
    client = LocoClient(vendor=surface, timeout_s=5.0)
    with client.call_timeout(None):
        assert client._client.timeout == 5.0
    with pytest.raises(ValueError, match="positive"), client.call_timeout(0.0):
        pass
