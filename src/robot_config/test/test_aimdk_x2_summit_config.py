"""One X2 deployment for recorded answers, gestures, mirror and the fruit gift."""

from __future__ import annotations

from pathlib import Path

import pytest

from embodied_common.motion_imitation_contract import manipulation_pause_service
from robot_config.loader import _load_robot_section, _validate_agent_voice_bridge, validate_runtime_named_motion_config

CONFIG = Path(__file__).resolve().parents[1] / "config" / "robots" / "aimdk_x2_summit.yaml"


@pytest.fixture(scope="module")
def config():
    return _load_robot_section(CONFIG)[1]


def test_mirror_and_fruit_delivery_share_the_deployment(config):
    assert config["name"] == "aimdk_x2_summit"
    assert config["mirror_session"]["enabled"] is True
    assert config["mirror_session"]["publish_enabled"] is True
    assert config["motion_imitation_stream"]["enabled"] is True
    assert manipulation_pause_service(config) == "/motion_imitation_stream/set_paused"
    services = {service["id"] for service in config["perception_services"]["services"]}
    assert services == {
        "hri_yolox_person",
        "hri_pear_parameters",
        "grasp_grounding",
        "grasp_segmentation",
        "grasp_generation",
    }
    assert config["default_control_mode"] == config["skill_required_control_mode"] == "model_inference"
    assert set(config["skill_supported_control_modes"]) == {"model_inference", "moveit_planning", "named_motion"}
    # All arm producers are delegated to the Gateway or gated mirror session.
    for mode in config["skill_supported_control_modes"]:
        assert config["control_modes"][mode]["runtime_mode"] == "policy_stream"
        assert config["control_modes"][mode]["inference"]["enabled"] is False
        assert config["control_modes"][mode]["executor"]["enabled"] is False
    assert config["motion_imitation_stream"]["mock_playback"]["hands"]["enabled"] is True
    assert config["cbf_safety"]["hands_installed"] == [True, True]
    assert config["cbf_safety"]["enabled"] is False


def test_fruit_grasp_can_select_either_installed_hand(config):
    grasp = config["grasp_execution"]
    assert grasp["backend_options"]["arm_side"] == "auto"
    assert set(grasp["backend_options"]["installed_hands"]) == {"left", "right"}
    assert {"joint.arm_stream", "joint.hand_stream"} <= set(grasp["command_interfaces"])


def test_the_voice_bridge_section_is_valid(config):
    assert _validate_agent_voice_bridge(config["embodied"]["agent"]["voice_bridge"]) == []


def test_the_allowlist_offers_fixed_free_pose_and_mirror(config):
    agent = config["embodied"]["agent"]
    assert config["embodied"]["skill_catalog_profile"] == "aimdk_x2_summit"
    assert {"mirror_start", "mirror_stop"} <= set(agent["test_allowlist"])
    assert "mimic_not_ready" not in agent["test_allowlist"]
    assert "free_pose" in agent["test_allowlist"]
    assert config["pose_composer"]["mode"] == "fixed"
    assert config["pose_composer"]["memory"]["enabled"] is False
    assert set(config["pose_composer"]["display_completion"]["pose_ids"]) == {
        "arms_forward",
        "arms_up",
        "arms_out",
        "arms_v",
        "flex",
        "surrender",
        "right_forward",
        "left_forward",
    }


def test_direct_commands_route_mirror_without_the_unavailable_reply(config):
    agent = config["embodied"]["agent"]
    commands = agent["planner"]["direct_commands"]
    assert set(commands.values()) == {
        "greet_visitor",
        "self_introduction",
        "introduce_platform",
        "introduce_agent",
        "invite_visitor",
        "mirror_start",
        "mirror_stop",
    }
    assert set(commands.values()) <= set(agent["test_allowlist"])
    assert set(commands.values()) - {"mirror_start", "mirror_stop"} <= set(agent["voice_bridge"]["silent_skills"])
    assert commands["开始照镜子"] == commands["动作模仿"] == "mirror_start"
    assert commands["停止照镜子"] == "mirror_stop"


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


def test_display_policy_and_legacy_hand_preparation_are_bound_after_inheritance(config):
    assert config["pose_composer"]["display_completion"]["mode"] == "playback_only"
    preparation = config["embodied"]["runtime_named_motion"]["hand_preparation"]
    assert set(preparation["motions"]) == {"wave", "raise_hand", "blow_kiss", "clap", "handshake"}
    assert config["embodied"]["runtime_named_motion"]["enabled"] is True


def test_summit_hand_preparation_passes_the_startup_schema(config):
    assert validate_runtime_named_motion_config(config) == []


@pytest.mark.parametrize(
    "change", [{"factory": "bad"}, {"motions": []}, {"motions": ["wave", "wave"]}, {"extra": True}]
)
def test_summit_hand_preparation_rejects_invalid_deployment_bindings(config, change):
    import copy

    changed = copy.deepcopy(config)
    changed["embodied"]["runtime_named_motion"]["hand_preparation"].update(change)
    assert validate_runtime_named_motion_config(changed)
