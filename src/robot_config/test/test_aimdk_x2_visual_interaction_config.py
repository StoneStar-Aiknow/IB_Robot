"""The X2 visual interaction deployment watches a visitor and commands nothing.

Covers the gap the magic-mirror deployment left: there is no
test_aimdk_x2_magicmirror_config.py, so nothing checks that deployment's config
against the contract it relies on. This one checks the claim that matters most
here — the entry point is observation-only — at the config layer, where the
whole deployment is visible.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_config.loader import load_robot_config_dict, validate_robot_config_dict

ROBOTS = Path(__file__).resolve().parents[1] / "config" / "robots"
CONFIG = ROBOTS / "aimdk_x2_visual_interaction.yaml"
# src/robot_config/test/ -> src/
PROFILE = (
    Path(__file__).resolve().parents[2] / "robots" / "aimdk" / "aimdk_robot" / "profiles" / "x2_ultra_omnihand.yaml"
)


@pytest.fixture(scope="module")
def raw():
    """The file as written, before inheritance is resolved."""
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["robot"]


@pytest.fixture(scope="module")
def config():
    # The camera is a live logical binding, resolved at launch.
    return load_robot_config_dict(CONFIG, defer_interface_binding=True)


@pytest.fixture(scope="module")
def profile():
    return yaml.safe_load(PROFILE.read_text(encoding="utf-8"))


def test_the_configuration_is_valid(config):
    validate_robot_config_dict(config, deferred_interfaces=True)


def test_it_inherits_the_skills_deployment(raw):
    assert raw["base_config"] == "aimdk_x2_skills.yaml"
    assert raw["name"] == "aimdk_x2_visual_interaction"


class TestObservationOnly:
    def test_the_feature_s_own_mode_declares_no_runtime_mode(self, config):
        """A mode with one is a mode something could take over.

        Inherited modes keep theirs — that is what they are for — but nothing
        in this deployment holds one, which is why there is no release node and
        nothing to hand back.
        """
        assert "runtime_mode" not in config["control_modes"]["observation"]
        assert config["default_control_mode"] == "observation"
        assert config["skill_required_control_mode"] == "observation"

    @pytest.mark.parametrize("producer", ["inference", "executor"])
    def test_no_control_mode_launches_a_motion_producer(self, config, producer):
        """A missing enabled key counts as enabled, as execution.py reads it.

        teleop and base_navigation are inherited through the skills config and
        both drive the robot, so each has to be switched off explicitly.
        """
        for name, mode in config["control_modes"].items():
            section = mode.get(producer)
            if section is None:
                continue
            assert section.get("enabled", True) is False, f"{name}.{producer}"

    def test_the_inherited_driving_modes_are_switched_off(self, config):
        """Named so a later inheritance change cannot quietly re-enable one."""
        for name in ("teleop", "base_navigation"):
            assert config["control_modes"][name]["executor"]["enabled"] is False, name

    def test_the_launch_builder_accepts_this_config(self, config):
        """The gate that would refuse a motion producer must pass on the real file."""
        from visual_interaction.launch_builders.visual_interaction import (
            generate_visual_interaction_nodes,
        )

        # The description is bound at launch; supply the camera the config names.
        bound = dict(config)
        interface = config["visual_interaction"]["frame_source"]["interface"]
        bound["runtime_interface_description"] = {
            "robot": {
                "id": "x2",
                "type": "x2_ultra_omnihand",
                "runtime_name": "aimdk",
                "runtime_version": "1",
            },
            "interfaces": {
                interface: {
                    "kind": "topic",
                    "direction": "publish",
                    "message_type": "sensor_msgs/msg/CompressedImage",
                    "endpoint": "/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed",
                }
            },
        }

        assert len(generate_visual_interaction_nodes(bound)) == 1

    def test_named_motions_are_switched_off(self, config):
        """Inherited from the skills config; no skill here may move the robot."""
        assert config["embodied"]["runtime_named_motion"]["enabled"] is False

    def test_it_requires_no_motion_capability(self, config):
        """So launch never waits for an interface this entry point will not use."""
        required = config["capabilities"]["requires"]

        assert required == ["perception.camera"]
        for capability in required:
            assert not capability.startswith(("joint.", "hand.", "gripper.", "arm.", "base."))


class TestCamera:
    def test_it_names_the_camera_the_mirror_already_uses(self, config):
        from visual_interaction.session.endpoints import CAMERA_COMPRESSED

        assert config["visual_interaction"]["frame_source"]["interface"] == CAMERA_COMPRESSED

    def test_that_camera_is_declared_by_the_runtime_profile(self, config, profile):
        """Resolved by ID, so a name the profile does not publish fails at launch."""
        interface = config["visual_interaction"]["frame_source"]["interface"]
        declared = profile["interfaces"][interface]

        assert declared["kind"] == "topic"
        assert declared["direction"] == "publish"
        assert declared["message_type"] == "sensor_msgs/msg/CompressedImage"

    def test_the_withdrawn_centre_camera_is_not_named(self, config, profile):
        """The vendor withdrew it in v0.8.1; the board demo used it anyway."""
        interface = config["visual_interaction"]["frame_source"]["interface"]

        assert "rgb_head_front_center" not in interface
        assert not [
            spec.get("endpoint")
            for spec in profile["interfaces"].values()
            if "rgb_head_front_center" in str(spec.get("endpoint", ""))
        ]

    def test_the_profile_matches_the_declared_type(self, config, profile):
        assert config["type"] == profile["runtime"]["type"]
        assert config["runtime"]["profile"] == profile["runtime"]["type"]


class TestSubjectWhitelist:
    def test_it_lists_the_whole_contract_whitelist(self, config):
        from embodied_common.visual_interaction_contract import CATEGORIES

        assert tuple(config["visual_interaction"]["categories"]) == CATEGORIES

    def test_it_has_no_fallback_category(self, config):
        """A generic member would return the sensitivity judgement to the model."""
        categories = config["visual_interaction"]["categories"]

        for fallback in ("other", "unknown", "display_object", "object", "misc"):
            assert fallback not in categories

    def test_pendants_stay_out(self, config):
        for excluded in ("pendant", "necklace", "medal"):
            assert excluded not in config["visual_interaction"]["categories"]


class TestNoSpeech:
    def test_the_deployment_configures_no_tts(self, config):
        """The session publishes its text; saying it needs arbitration first."""
        assert "voice_tts" not in config.get("visual_interaction", {})
        assert config["visual_interaction"]["vl"]["implementation"] == "fake"

    def test_visual_games_are_not_enabled_here(self, config):
        """A different feature with a frozen announcement table; not this one."""
        games = config.get("embodied", {}).get("visual_games", {})

        assert all(not policy.get("enabled") for policy in games.values())


class TestTimings:
    def test_confirmation_needs_two_windows(self, config):
        assert config["visual_interaction"]["session"]["confirmation_count"] >= 2

    def test_a_session_outlasts_the_absence_timer(self, config):
        """Otherwise the duration limit fires first and the guard never does."""
        session = config["visual_interaction"]["session"]

        assert session["max_session_sec"] > session["absent_reset_sec"]

    def test_the_evidence_age_leaves_room_for_two_rounds(self, config):
        """A candidate must survive its own confirmation wait."""
        section = config["visual_interaction"]
        rounds = section["session"]["confirmation_count"] * section["round_period_sec"]

        assert section["session"]["max_evidence_age_sec"] > rounds

    def test_every_timing_is_positive(self, config):
        section = config["visual_interaction"]

        for key in ("round_period_sec", "window_sec", "buffer_sec", "stale_stream_sec"):
            assert section[key] > 0, key
        for key in ("max_evidence_age_sec", "min_gap_sec", "absent_reset_sec", "max_session_sec"):
            assert section["session"][key] > 0, key

    def test_the_buffer_holds_more_than_one_window(self, config):
        section = config["visual_interaction"]

        assert section["buffer_sec"] > section["window_sec"]
