"""Preset-only surface validation without a robot or DDS participant."""

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


def test_profile_exposes_only_one_preset_no_hand_or_base():
    profile = load_profile(PROFILE)
    assert profile["command_channels"] == []
    assert set(profile["modes"]) == {"initial", "idle"}
    assert profile["modes"]["idle"]["transitions"] == []
    assert profile["vendor"]["postures"] == {}
    assert profile["vendor"]["arm_actions"] == {"arms_horizontal": {"action_id": 15, "action_name": "both_hands_up"}}
    assert "hand.multi_joint" not in profile["capabilities"]
    assert "runtime.stop" not in profile["capabilities"]
    assert len(profile["joints"]) == 29


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
    monkeypatch.setattr(runtime_node, "UnitreeSdk", Mock(return_value=sdk))
    monkeypatch.setattr(runtime_node.vg, "require_vendor", lambda: None)
    node._setup_vendor_io()
    sdk.command_writer.assert_not_called()
    sdk.loco.assert_not_called()
    sdk.motion_switcher.assert_not_called()
    sdk.audio.assert_not_called()
    assert len(sdk.state_feed.call_args_list) == 1
    assert sdk.state_feed.call_args.args[:2] == ("rt/lowstate", "low_state")
    assert set(node._rpc) == {"arm"}
