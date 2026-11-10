"""Preset-only surface validation without a robot or DDS participant."""

import copy
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mock_process import vendor_test_domain
from unitree_g1_robot.runtime_node import UnitreeG1RuntimeNode

from robot_runtime.profile import load_profile

PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "g1_29dof_arm_presets.yaml"


@pytest.mark.parametrize("fsm,expected", [(802, True), (500, False), (None, False)])
def test_supervised_preset_rechecks_each_request_without_changing_fsm(fsm, expected):
    node = object.__new__(UnitreeG1RuntimeNode)
    node._preset_physical = True
    node._lock = threading.RLock()
    node._state = SimpleNamespace(stop_latched=False)
    # The ownership check the preflight runs is the profile's own declaration;
    # only the operator's expected FSM is per-session.
    node._vendor = {
        **load_profile(PROFILE)["vendor"],
        "operator_validation": {"expected_fsm_id": 802},
    }
    node._lowstate_stamp = time.monotonic()
    node._feedback_window_s = 0.5
    node.rpc_call = Mock(
        side_effect=[
            SimpleNamespace(
                ok=True,
                data=[
                    {"name": "ai_sport", "status": 0},
                    {"name": "g1_arm_example", "status": 0},
                ],
            ),
            SimpleNamespace(ok=True, data=fsm),
        ]
        * 2
    )
    assert node.admit_preset_attempt()[0] is expected
    assert node.admit_preset_attempt()[0] is expected
    assert [c.args[:2] for c in node.rpc_call.call_args_list] == [
        ("robot_state", "service_list"),
        ("loco", "get_fsm_id"),
    ] * 2


EXPECTED_ACTIONS = {
    "release_arm": (99, "release_arm"),
    "turn_back_wave": (1, "turn_back_wave"),
    "blow_kiss_with_both_hands": (11, "blow_kiss_with_both_hands"),
    "blow_kiss_with_left_hand": (12, "blow_kiss_with_left_hand"),
    "blow_kiss_with_right_hand": (13, "blow_kiss_with_right_hand"),
    "arms_horizontal": (15, "both_hands_up"),
    "clamp": (17, "clamp"),
    "high_five": (18, "high_five"),
    "hug": (19, "hug"),
    "make_heart_with_both_hands": (20, "make_heart_with_both_hands"),
    "make_heart_with_right_hand": (21, "make_heart_with_right_hand"),
    "refuse": (22, "refuse"),
    "right_hand_up": (23, "right_hand_up"),
    "ultraman_ray": (24, "ultraman_ray"),
    "wave_under_head": (25, "wave_under_head"),
    "wave_above_head": (26, "wave_above_head"),
    "shake_hand": (27, "shake_hand"),
    "box_left_hand_win": (28, "box_left_hand_win"),
    "box_right_hand_win": (29, "box_right_hand_win"),
    "box_both_hand_win": (30, "box_both_hand_win"),
    "right_hand_on_heart": (33, "right_hand_on_heart"),
    "both_hands_up_deviate_right": (34, "both_hands_up_deviate_right"),
    "forward_push": (36, "forward_push"),
}
MODE_MACHINE_LIMITED = {
    "make_heart_with_both_hands",
    "make_heart_with_right_hand",
    "box_left_hand_win",
    "box_right_hand_win",
    "box_both_hand_win",
    "forward_push",
}


def test_profile_exposes_23_operator_reported_presets_no_hand_or_base():
    profile = load_profile(PROFILE)
    assert profile["command_channels"] == []
    assert set(profile["modes"]) == {"initial", "idle"}
    assert profile["modes"]["idle"]["transitions"] == []
    assert profile["vendor"]["postures"] == {}
    actions = profile["vendor"]["arm_actions"]
    assert len(actions) == 23
    assert {name: (entry["action_id"], entry["action_name"]) for name, entry in actions.items()} == EXPECTED_ACTIONS
    assert set(actions) == set(profile["capabilities"]["motion.named"]["names"])
    assert profile["vendor"]["preset_commissioning"]["allowed_arm_actions"] == actions
    assert actions["arms_horizontal"]["action_name"] == "both_hands_up"
    assert "hand.multi_joint" not in profile["capabilities"]
    assert "runtime.stop" not in profile["capabilities"]
    assert len(profile["joints"]) == 29


def test_firmware_limits_are_only_the_operator_reported_ones():
    actions = load_profile(PROFILE)["vendor"]["arm_actions"]
    assert actions["turn_back_wave"].get("fsm_ids") == [500, 501]
    for name in MODE_MACHINE_LIMITED:
        assert actions[name].get("mode_machine") == [5, 6]
    constrained = {name for name, entry in actions.items() if "fsm_ids" in entry or "mode_machine" in entry}
    assert constrained == {"turn_back_wave"} | MODE_MACHINE_LIMITED
    allowed_fields = {"action_id", "action_name", "fsm_ids", "mode_machine"}
    assert all(set(entry) <= allowed_fields for entry in actions.values())


def _constraint_node(*, mode_machine: int = 5):
    node = object.__new__(UnitreeG1RuntimeNode)
    node._profile = load_profile(PROFILE)
    node._vendor = copy.deepcopy(node._profile["vendor"])
    node._vendor["mode_machine"] = mode_machine
    node._lock = threading.RLock()
    node._state = SimpleNamespace(stop_latched=False)
    node._preset_physical = False
    node._rpc_timeout_s = 5.0
    node._lowstate_stamp = time.monotonic()
    node._feedback_window_s = 0.5
    node._mode_machine_observed = mode_machine
    return node


def test_turn_back_wave_is_rejected_at_fsm_802_without_switching():
    node = _constraint_node()
    node._preset_physical = True
    node.rpc_call = Mock(return_value=SimpleNamespace(ok=True, data=802))
    admitted, detail = node.admit_preset_attempt("turn_back_wave")
    assert not admitted
    assert "requires FSM [500, 501]" in detail
    assert "current FSM is 802" in detail
    assert "will not switch FSM" in detail
    assert [call.args[:2] for call in node.rpc_call.call_args_list] == [("loco", "get_fsm_id")]


def test_mode_machine_limited_presets_admit_at_generation_5():
    node = _constraint_node(mode_machine=5)
    node.rpc_call = Mock(return_value=SimpleNamespace(ok=True, data=802))
    for name in MODE_MACHINE_LIMITED:
        assert node.preset_constraint_failure(name) == ""
    assert node.admit_preset_attempt("make_heart_with_both_hands") == (True, "mock transport")
    node.rpc_call.assert_not_called()


def test_mode_machine_constraint_reads_fresh_lowstate_not_the_profile():
    node = _constraint_node(mode_machine=5)
    node._mode_machine_observed = 7
    reason = node.preset_constraint_failure("make_heart_with_both_hands")
    assert "requires mode_machine [5, 6]" in reason
    assert "observed mode_machine is 7" in reason
    assert "preset not dispatched" in reason


def test_constrained_actions_fail_closed_on_stale_lowstate():
    node = _constraint_node(mode_machine=5)
    node._lowstate_stamp = time.monotonic() - 10.0
    admitted, detail = node.admit_preset_attempt("make_heart_with_both_hands")
    assert not admitted
    assert "lowstate feedback" in detail
    assert "stale" in detail
    assert "preset not dispatched" in detail


def test_mode_machine_limited_presets_reject_at_other_generations():
    node = _constraint_node(mode_machine=7)
    node.rpc_call = Mock(return_value=SimpleNamespace(ok=True, data=802))
    reason = node.preset_constraint_failure("make_heart_with_both_hands")
    assert "requires mode_machine [5, 6]" in reason
    assert "observed mode_machine is 7" in reason
    admitted, detail = node.admit_preset_attempt("make_heart_with_both_hands")
    assert not admitted and detail == reason
    node.rpc_call.assert_not_called()


@pytest.mark.parametrize(
    "reported,expected",
    [
        ([[{"id": 15, "name": "both_hands_up"}], []], {"arms_horizontal"}),
        ([[{"id": 23, "name": "both_hands_up"}], []], set()),
        ([[{"id": 15, "name": "wrong"}], []], set()),
        ([[], [{"id": 15, "name": "both_hands_up"}]], set()),
        ([], set()),
    ],
)
def test_actual_firmware_list_requires_matching_preset_id_and_name(reported, expected):
    node = object.__new__(UnitreeG1RuntimeNode)
    node._vendor = load_profile(PROFILE)["vendor"]
    node._lock = threading.RLock()
    node._state = Mock()
    node.get_logger = Mock(return_value=Mock())
    node.rpc_call = Mock(return_value=SimpleNamespace(ok=True, data=reported))
    node._confirm_vendor_actions()
    assert node._confirmed_actions == expected
    node.rpc_call.assert_called_once_with("arm", "get_action_list", None)


def test_preset_io_creates_no_hand_or_motion_writer(monkeypatch):
    from unitree_g1_robot import runtime_node

    node = object.__new__(UnitreeG1RuntimeNode)
    node._preset_only = True
    node._preset_physical = False
    node._vendor = load_profile(PROFILE)["vendor"]
    node._rpc_timeout_s = 5.0
    # The domain is inert here (the SDK is mocked); it still comes from the
    # per-process allocation so no test carries a deployment domain literal.
    domain_id = vendor_test_domain()
    node._vendor_domain_id = domain_id
    node._vendor_network_interface = "lo"
    params = {"vendor_domain_id": domain_id, "vendor_network_interface": "lo"}
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])
    sdk = Mock()
    # The injectable facade seam replaces the module-level constructor patch.
    node._sdk_factory = lambda: sdk
    monkeypatch.setattr(runtime_node.vg, "require_vendor", lambda: None)
    node._setup_vendor_io()
    sdk.command_writer.assert_not_called()
    sdk.loco.assert_not_called()
    sdk.motion_switcher.assert_not_called()
    sdk.audio.assert_not_called()
    assert len(sdk.state_feed.call_args_list) == 1
    assert sdk.state_feed.call_args.args[:2] == ("rt/lowstate", "low_state")
    assert set(node._rpc) == {"arm"}
