"""The static preset names are firmware facts and are matched exactly.

The saved response is reconstructed from the operator's 2026-11-09 run profile
(23 ``allowed_arm_actions``); the 2026-11-07 run still carried the earlier
single-action-15 profile. It is the only authority for the firmware name
strings: the in-repo mock echoes the profile's own table, so exercising the mock
alone cannot prove that table matches a robot. These tests pin both G1 profiles
to that report and prove the startup check keeps exact id+name matching: a
renamed or unreported action stays unadvertised instead of matching by id.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from unitree_g1_robot.runtime_node import UnitreeG1RuntimeNode

from robot_runtime.profile import load_profile

PACKAGE = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "g1_get_action_list_20261109.json"
PRESET_PROFILE = PACKAGE / "profiles" / "g1_29dof_arm_presets.yaml"
DEX3_PROFILE = PACKAGE / "profiles" / "g1_29dof_dex3.yaml"

# The dex3 profile's public skill names are an external surface; only the
# firmware ``action_name`` strings may move when the vendor report changes.
DEX3_PUBLIC_NAMES = {
    "release_arm",
    "blow_kiss",
    "hands_up",
    "clap",
    "high_five",
    "hug",
    "heart",
    "wave",
    "shake_hand",
}


def _saved_response() -> list:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["response"]


def _reported_pairs(response: list) -> set[tuple[int, str]]:
    presets = response[0] if response and isinstance(response[0], list) else response
    return {(int(item["id"]), str(item["name"])) for item in presets}


def _bare_node(profile_path: Path) -> UnitreeG1RuntimeNode:
    node = object.__new__(UnitreeG1RuntimeNode)
    node._vendor = load_profile(profile_path)["vendor"]
    node._lock = threading.RLock()
    node._confirmed_actions = set()
    node._state = SimpleNamespace(set_capability_params=Mock())
    node.get_logger = Mock(return_value=Mock())
    return node


def _confirm(node: UnitreeG1RuntimeNode, response: list) -> set[str]:
    node.rpc_call = Mock(return_value=SimpleNamespace(ok=True, data=response, timed_out=False))
    node._confirm_vendor_actions()
    return node.served_motion_names()


def test_saved_report_is_23_unique_id_name_pairs():
    response = _saved_response()
    assert len(response) == 2 and response[1] == []
    pairs = _reported_pairs(response)
    assert len(pairs) == 23
    assert len({action_id for action_id, _ in pairs}) == 23


def test_preset_profile_table_equals_the_saved_firmware_report():
    actions = load_profile(PRESET_PROFILE)["vendor"]["arm_actions"]
    assert {(entry["action_id"], entry["action_name"]) for entry in actions.values()} == _reported_pairs(
        _saved_response()
    )


def test_dex3_keeps_its_public_names_but_uses_the_saved_firmware_names():
    actions = load_profile(DEX3_PROFILE)["vendor"]["arm_actions"]
    assert set(actions) == DEX3_PUBLIC_NAMES
    firmware = dict(_reported_pairs(_saved_response()))
    for public_name, entry in actions.items():
        assert firmware[entry["action_id"]] == entry["action_name"], public_name


@pytest.mark.parametrize("profile_path", [PRESET_PROFILE, DEX3_PROFILE], ids=["preset", "dex3"])
def test_saved_report_confirms_exactly_the_profile_table(profile_path):
    vendor = load_profile(profile_path)["vendor"]
    served = _confirm(_bare_node(profile_path), _saved_response())
    assert served == set(vendor["arm_actions"]) | set(vendor.get("postures") or {})


def test_a_subset_report_shrinks_to_the_reported_actions():
    response = _saved_response()
    response[0] = [item for item in response[0] if item["id"] in {15, 99}]
    assert _confirm(_bare_node(PRESET_PROFILE), response) == {"arms_horizontal", "release_arm"}


def test_preset_clap_key_keeps_the_saved_firmware_action_name_for_id_17():
    actions = load_profile(PRESET_PROFILE)["vendor"]["arm_actions"]
    assert actions["clap"] == {"action_id": 17, "action_name": "clamp"}
    assert "clamp" not in actions
    # ID 15 is not part of this rename.
    assert actions["arms_horizontal"] == {"action_id": 15, "action_name": "both_hands_up"}


def test_preset_clap_key_does_not_match_a_renamed_firmware_token():
    response = _saved_response()
    response[0] = [item for item in response[0] if item["id"] != 17]
    response[0].append({"id": 17, "name": "clap"})
    served = _confirm(_bare_node(PRESET_PROFILE), response)
    assert "clap" not in served
    assert "arms_horizontal" in served


def test_a_renamed_or_unreported_action_stays_unadvertised():
    response = _saved_response()
    response[0] = [item for item in response[0] if item["id"] not in {17, 20}]
    # Same ID, different firmware name: the public key `clap` must not satisfy
    # exact matching against the firmware token `clamp`.
    response[0].append({"id": 17, "name": "clap"})
    served = _confirm(_bare_node(DEX3_PROFILE), response)
    assert "heart" not in served
    assert "clap" not in served
    assert "high_five" in served
