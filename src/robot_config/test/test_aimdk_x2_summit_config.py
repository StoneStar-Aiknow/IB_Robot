"""The X2 exhibition deployment: recorded answers, gestures and the fruit gift, no mirror."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_config.loader import _load_robot_section, _validate_agent_voice_bridge

CONFIG = Path(__file__).resolve().parents[1] / "config" / "robots" / "aimdk_x2_summit.yaml"


@pytest.fixture(scope="module")
def config():
    return _load_robot_section(CONFIG)[1]


def test_the_mirror_and_its_models_stay_down(config):
    assert config["name"] == "aimdk_x2_summit"
    assert config["mirror_session"]["enabled"] is False
    assert config["motion_imitation_stream"]["enabled"] is False
    services = {service["id"] for service in config["perception_services"]["services"]}
    assert services == {"grasp_grounding", "grasp_segmentation", "grasp_generation"}


def test_fruit_grasp_can_select_either_installed_hand(config):
    grasp = config["grasp_execution"]
    assert grasp["backend_options"]["arm_side"] == "auto"
    assert set(grasp["backend_options"]["installed_hands"]) == {"left", "right"}
    assert {"joint.arm_stream", "joint.hand_stream"} <= set(grasp["command_interfaces"])


def test_the_voice_bridge_section_is_valid(config):
    assert _validate_agent_voice_bridge(config["embodied"]["agent"]["voice_bridge"]) == []


def test_the_allowlist_offers_no_mirror_or_free_pose(config):
    agent = config["embodied"]["agent"]
    assert config["embodied"]["skill_catalog_profile"] == "aimdk_x2_summit"
    assert not {"mirror_start", "mirror_stop", "free_pose"} & set(agent["test_allowlist"])


def test_direct_commands_only_reach_allowed_clip_skills(config):
    agent = config["embodied"]["agent"]
    commands = agent["planner"]["direct_commands"]
    assert set(commands.values()) == {
        "greet_visitor",
        "self_introduction",
        "introduce_platform",
        "introduce_agent",
        "invite_visitor",
    }
    assert set(commands.values()) <= set(agent["test_allowlist"])
    assert set(commands.values()) <= set(agent["voice_bridge"]["silent_skills"])


def test_only_summit_opts_in_to_social_wave_cancel_return(config):
    assert set(config["runtime"]["cancel_return_motions"].split(",")) == {
        "wave",
        "short_wave",
        "invite",
        "self_intro",
        "explain",
    }
    base = _load_robot_section(CONFIG.parent / "aimdk_x2_voice_mirror_grasp.yaml")[1]
    assert not base["runtime"].get("cancel_return_motions")


def test_the_summit_follows_only_the_waking_voice_and_vetoes_unaddressed_speech(config):
    """Voiceprint and intent gate are on for the exhibition, and a missing model fails loudly."""
    assert config["voice_asr"]["speaker_embedding"]["enabled"] is True
    bridge = config["embodied"]["agent"]["voice_bridge"]
    assert bridge["speaker_check_enabled"] is True
    assert bridge["addressee_trigger"] == "wake_phrase" and bridge["wake_phrases"]
    assert bridge["semantic_gate_mode"] == "enforce"
    assert bridge["semantic_gate_model_path"] == "$(env WORKSPACE)/models/addressee/addressee_zh_v1.json"
    assert bridge["semantic_gate_fail_open"] is False
    # The shared mirror/grasp base keeps both checks opt-in.
    base = _load_robot_section(CONFIG.parent / "aimdk_x2_voice_mirror_grasp.yaml")[1]
    assert base["embodied"]["agent"]["voice_bridge"]["speaker_check_enabled"] is False
    assert base["voice_asr"]["speaker_embedding"]["enabled"] is False
