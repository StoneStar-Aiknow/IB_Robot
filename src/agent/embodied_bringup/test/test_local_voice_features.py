"""Local config keeps pose capabilities and the production speech wiring together."""

import importlib.util
import json
from pathlib import Path

import yaml

from embodied_bringup.launch_builders.embodied import _agent_voice_bridge_node
from robot_config.loader import _load_robot_section

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("local_voice_config", ROOT / "tools/x2_local_voice/make_config.py")
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)


def params(node):
    result = {}
    for key, value in node._Node__parameters[0].items():
        key = "".join(getattr(item, "text", str(item)) for item in key)
        if isinstance(value, tuple):
            value = yaml.safe_load("".join(getattr(item, "text", str(item)) for item in value))
        result[key] = value
    return result


def test_local_catalog_and_config_keep_fixed_pose_skills(tmp_path):
    _, config = _load_robot_section(local.SUMMIT)
    local.deviate(config, models=tmp_path, state=tmp_path / "agent", keep_english=False)
    profile = yaml.safe_load(local.catalog(local.SUMMIT_CATALOG.read_text()))
    enabled = {entry["name"] for entry in profile["enabled_skills"]}
    assert {"free_pose", "return_to_task_pose", "pick_object", "place_in_container"} <= enabled
    assert not enabled.intersection({"handshake", "mirror_start", "mirror_stop"})
    assert enabled == set(config["embodied"]["agent"]["test_allowlist"])
    assert config["task_poses"]["file"]
    assert config["pose_composer"]["enabled"] and config["pose_composer"]["mode"] == "fixed"
    assert config["pose_composer"]["blocked_task_poses"] == []
    assert config["embodied"]["agent"]["planner"]["return_after_place"] == "grasp_ready"


def test_speech_enhancements_reach_the_bridge_from_robot_config():
    _, config = _load_robot_section(local.SUMMIT)
    # This test exercises speech parameter wiring without loading the separate pose artifact.
    config["pose_composer"] = {"enabled": False}
    agent = config["embodied"]["agent"]
    voice = agent["voice_bridge"]
    values = params(_agent_voice_bridge_node(config, agent, voice))
    assert json.loads(values["spoken_lines_json"]) == voice["spoken_lines"]
    assert json.loads(values["gift_messages_json"]) == voice["gift_messages"]
    assert json.loads(values["tts_pronunciations_json"]) == voice["tts_pronunciations"]
    assert values["pending_sentence_sec"] == 10.0
    assert values["gift_release_is_delivery"] is True
    assert json.loads(values["speech_clips_json"])
