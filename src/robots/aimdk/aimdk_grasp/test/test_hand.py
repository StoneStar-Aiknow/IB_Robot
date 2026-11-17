"""Native left-hand positions, planning-model aliases and the distal coupling."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from aimdk_grasp.hand import HandMapping, hand_links, load_gripper_presets
from aimdk_grasp.joints import DIP_FOLLOWERS, HAND_ALIASES
from aimdk_grasp.planning_model import HAND_URDF, description_share
from aimdk_grasp.robot_model import planning_model

#: The release pose the motion-control unit leaves the hand in.
RELEASE = (-0.5, 1.0, -0.75, 0.0, 1.57, 1.57, 0.0, 1.57, 0.0, 1.57)
#: Open and close presets of the OmniHand gripper description, as aliases.
PRESETS = {
    "open": {
        "R_thumb_roll_joint": 0.129,
        "R_thumb_abad_joint": -1.54,
        "R_thumb_mcp_joint": 0.255,
        **{name: 0.0 for name in HAND_ALIASES[3:]},
    },
    "close": {
        "R_thumb_roll_joint": 0.43,
        "R_thumb_abad_joint": -1.4,
        "R_thumb_mcp_joint": 0.85,
        **{name: 1.48 if "pip" in name else 0.0 for name in HAND_ALIASES[3:]},
    },
}


@pytest.fixture(scope="module")
def mapping() -> HandMapping:
    return HandMapping(planning_model(), PRESETS)


def test_distal_followers_agree_with_the_official_coupling(mapping):
    # The measured polynomial and the official linear mimic differ mid-range by
    # up to ~0.3 rad; both start at zero, rise together and meet at full flexion.
    official = ET.parse(description_share() / HAND_URDF).getroot()
    for name, (master, coefficients) in DIP_FOLLOWERS.items():
        mimic = official.find(f"joint[@name='L_{name[2:]}']/mimic")
        assert mimic.get("joint") == "L_" + master[2:]
        bounds = planning_model().find(f"joint[@name='{master}']/limit")
        span = np.linspace(float(bounds.get("lower")), float(bounds.get("upper")), 50)
        follower = np.polynomial.polynomial.polyval(span, coefficients)
        assert follower[0] == pytest.approx(0.0) and np.all(np.diff(follower) > 0)
        assert follower[-1] == pytest.approx(float(mimic.get("multiplier")) * span[-1], abs=0.1)
    random = np.random.default_rng(18).uniform(mapping.lower, mapping.upper, (30, 10))
    for hand in [mapping.preset("open"), mapping.preset("close"), RELEASE, *random]:
        full = mapping.kinematic_alias(hand)
        assert len(full) == 15 and len(mapping.to_alias(hand)) == 10
    # Passive values are estimates with limits, never silently clamped.
    narrow = HandMapping(planning_model(), PRESETS)
    narrow.distal_limits["R_thumb_dip_joint"] = (0.0, 0.2)
    with pytest.raises(ValueError, match="Coupled hand joint"):
        narrow.kinematic_alias(RELEASE)


def test_aliases_round_trip_and_limits_are_enforced(mapping):
    native = np.asarray(RELEASE)
    np.testing.assert_allclose(mapping.to_native(mapping.to_alias(native)), native)
    with pytest.raises(ValueError, match="native URDF limits"):
        mapping.check(native + np.r_[0.0, 2.0, np.zeros(8)])
    with pytest.raises(KeyError, match="no hand preset"):
        HandMapping(planning_model()).preset("open")


def test_gripper_presets_must_describe_this_hand(tmp_path):
    gripper = tmp_path / "omnihand_left_x2"
    gripper.mkdir()
    links = list(hand_links(planning_model()))
    assert len(links) == 17 and "R_index_tip" not in links
    (gripper / "config.json").write_text(json.dumps({"links": links, **PRESETS}))
    assert load_gripper_presets(gripper, planning_model()) == PRESETS

    (gripper / "config.json").write_text(json.dumps({"links": links[:-1], **PRESETS}))
    with pytest.raises(ValueError, match="gripper links"):
        load_gripper_presets(gripper, planning_model())


def test_the_right_hand_is_mounted_natively_with_the_same_alias_presets():
    """One gripper description serves both hands: its alias presets are the right hand's native values."""
    from aimdk_grasp.joints import HAND_SIGNS
    from aimdk_grasp.robot_model import config_directory

    left = HandMapping(planning_model(), PRESETS)
    right = HandMapping(planning_model(config_directory("right")), PRESETS, side="right")
    assert right.side == "right" and left.side == "left"
    for name in ("open", "close"):
        np.testing.assert_allclose(right.preset(name), [PRESETS[name][joint] for joint in HAND_ALIASES])
        np.testing.assert_allclose(right.preset(name), left.preset(name) * HAND_SIGNS)
    # Mirror poses of the two hands give the same aliases and coupled distal joints.
    native = np.asarray(RELEASE) * HAND_SIGNS
    assert right.to_alias(native) == pytest.approx(left.to_alias(RELEASE))
    assert right.kinematic_alias(native) == pytest.approx(left.kinematic_alias(RELEASE))
    np.testing.assert_allclose(right.lower, np.sort(np.c_[left.lower, left.upper] * HAND_SIGNS[:, None])[:, 0])
    with pytest.raises(ValueError, match="Right hand exceeds"):
        right.check(native + np.r_[0.0, 4.0, np.zeros(8)])
    with pytest.raises(RuntimeError, match="Initial right hand is not open"):
        right.require_open(right.preset("close"))
    # A thumb abduction 0.088 rad short of the open preset counts as open; 0.12 does not.
    open_pose = right.preset("open")
    inward = 1.0 if open_pose[1] + 0.12 <= right.upper[1] else -1.0
    short = open_pose.copy()
    short[1] += 0.088 * inward
    assert right.require_open(short) == pytest.approx(0.088)
    short[1] = open_pose[1] + 0.12 * inward
    with pytest.raises(RuntimeError, match="Initial right hand is not open"):
        right.require_open(short)
    with pytest.raises(ValueError, match="side must be"):
        HandMapping(planning_model(), PRESETS, side="middle")
