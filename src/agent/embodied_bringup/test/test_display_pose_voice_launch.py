"""Enabled fixed pose aliases pass the intent gate without opening arbitrary descriptions."""

import hashlib
import json

from launch import LaunchContext
from launch_ros.utilities import evaluate_parameters

from embodied_bringup.launch_builders.embodied import _agent_voice_bridge_node


def parameters(node):
    result = evaluate_parameters(LaunchContext(), vars(node)["_Node__parameters"])[0]
    result["lexicon"] = json.loads(result["semantic_gate_lexicon_json"])
    return result


def test_enabled_fixed_pose_aliases_join_the_voice_lexicon(tmp_path):
    raw = json.dumps(
        {"poses": {"arms_out": {"user_message": "侧平举", "phrases": ["双手侧平举", "双臂侧平举"]}}}
    ).encode()
    path = tmp_path / "poses.json"
    path.write_bytes(raw)
    robot = {
        "pose_composer": {
            "mode": "fixed",
            "fixed_library": str(path),
            "fixed_library_sha256": hashlib.sha256(raw).hexdigest(),
        }
    }
    voice = {"semantic_gate_lexicon": ["你好", "双手侧平举"], "speaker_check_enabled": True}
    node = _agent_voice_bridge_node(robot, {"test_allowlist": ["free_pose"]}, voice)
    params = parameters(node)
    assert params["lexicon"] == ["你好", "双手侧平举", "双臂侧平举"]
    assert params["speaker_check_enabled"] is True
    node = _agent_voice_bridge_node(robot, {"test_allowlist": []}, voice)
    params = parameters(node)
    assert params["lexicon"] == voice["semantic_gate_lexicon"]
