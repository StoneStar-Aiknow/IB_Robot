"""Launch wiring for the resident motion-imitation frame stream.

The high-value assertion is the last one: the node's service parameters must
match the endpoints the perception model services actually advertise. A typo
there is otherwise a silent runtime hang with no error anywhere in the graph.

Migrated from ``robot_config/test/test_motion_imitation_stream_builder.py``
during the rebase onto the decentralised launch-builder registration: the
tests follow ``generate_motion_imitation_stream_nodes`` into its owning
package, and ``generate_perception_model_nodes`` now lives in
``inference_service``. robot_config assets load through its installed share
directory.
"""

import json
from pathlib import Path

import pytest
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext
from launch_ros.utilities import evaluate_parameters

from inference_service.launch_builders.perception_models import generate_perception_model_nodes
from manipulation_execution.launch_builders.motion_imitation_stream import (
    generate_motion_imitation_stream_nodes,
)
from robot_config.loader import load_robot_config_dict

CONFIG = Path(get_package_share_directory("robot_config")) / "config/robots/aimdk_x2_magicmirror.yaml"
BASE_CONFIG = Path(get_package_share_directory("robot_config")) / "config/robots/aimdk_x2.yaml"


def parameters(node):
    return evaluate_parameters(LaunchContext(), node._Node__parameters)[0]


def stream_config(**overrides):
    config = {
        "runtime": {"interface_description": {"interfaces": {}}},
        "motion_imitation_stream": {
            "enabled": True,
            "frame_source": {"topic": "/camera/head/image/compressed"},
        },
    }
    config["motion_imitation_stream"].update(overrides)
    return config


def test_enabled_block_launches_one_stream_node_with_defaults():
    nodes = generate_motion_imitation_stream_nodes(stream_config())

    assert len(nodes) == 1
    node = nodes[0]
    assert node._Node__package == "manipulation_execution"
    assert parameters(node) == {
        "rgb_compressed_topic": "/camera/head/image/compressed",
        "frame_topic": "/motion_imitation_stream/pear_frame",
        "frame_source_interface": "",
        "decoder": "mock_cpu",
        "decoder_params_json": "{}",
        "actor_state_topic": "/motion_imitation_stream/actor_state",
        "actor_stale_sec": 3.0,
        "status_topic": "/motion_imitation_stream/status",
        "yolox_detect_service": "/perception/hri/yolox_detect",
        "pear_parameters_service": "/perception/hri/pear_parameters",
        "person_confidence_threshold": 0.30,
        "yolox_refresh_interval_sec": 1.0,
        "rpc_timeout_sec": 2.0,
        "max_pear_hz": 0.0,
        "publish_fallback_frames": True,
        "status_period_sec": 5.0,
    }


def test_explicit_topic_does_not_require_a_runtime_description():
    config = {"motion_imitation_stream": {"enabled": True, "frame_source": {"topic": "/camera/half/compressed"}}}
    node = generate_motion_imitation_stream_nodes(config)[0]
    assert parameters(node)["rgb_compressed_topic"] == "/camera/half/compressed"


def test_absent_block_launches_nothing():
    assert generate_motion_imitation_stream_nodes({}) == []


def test_disabled_block_launches_nothing():
    assert generate_motion_imitation_stream_nodes(stream_config(enabled=False)) == []


def test_base_x2_config_does_not_launch_the_stream():
    # The stream belongs to the magicmirror entry; the shared X2 base must not
    # acquire a camera consumer by inheritance.
    assert generate_motion_imitation_stream_nodes(load_robot_config_dict(BASE_CONFIG)) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"enabled": "true"},
        {"frame_source": {"topic": "relative/topic"}},
        {"frame_source": {"topic": "/camera", "decoder_params": []}},
        {"frame_source": {"topic": "/camera", "decoder": ""}},
        {"rgb_compressed_topic": ""},
        {"rgb_compressed_topic": "relative/topic"},
        {"rgb_compressed_type": "sensor_msgs/msg/Image"},
        {"person_confidence_threshold": 1.5},
        {"person_confidence_threshold": float("nan")},
        {"person_confidence_threshold": True},
        {"yolox_refresh_interval_sec": 0.0},
        {"yolox_refresh_interval_sec": -1.0},
        {"rpc_timeout_sec": float("inf")},
        {"max_pear_hz": -1.0},
        {"publish_fallback_frames": "yes"},
        {"status_period_sec": 0.0},
        {"frame_topic": "/same", "status_topic": "/same"},
    ],
)
def test_malformed_block_raises(overrides):
    with pytest.raises(ValueError):
        generate_motion_imitation_stream_nodes(stream_config(**overrides))


def test_non_mapping_block_raises():
    with pytest.raises(ValueError):
        generate_motion_imitation_stream_nodes({"motion_imitation_stream": []})


def test_stream_services_match_the_launched_model_service_endpoints():
    config = load_robot_config_dict(CONFIG, defer_interface_binding=True)
    config["motion_imitation_stream"]["mock_playback"]["enabled"] = False
    source_config = config["motion_imitation_stream"]["frame_source"]
    assert source_config["topic"] == "/aima/hal/sensor/rgb_head_front_center/rgb_image_half/compressed"
    config["runtime"]["interface_description"] = {
        "interfaces": {
            "camera.stereo_head_front_left.color_compressed": {
                "kind": "topic",
                "direction": "publish",
                "message_type": "sensor_msgs/msg/CompressedImage",
                "endpoint": "/camera/stereo_left/image/compressed",
            },
        }
    }
    stream = parameters(generate_motion_imitation_stream_nodes(config)[0])
    endpoints = {parameters(node)["service_endpoint"] for node in generate_perception_model_nodes(config)}

    assert stream["yolox_detect_service"] in endpoints
    assert stream["pear_parameters_service"] in endpoints
    assert stream["frame_source_interface"] == ""
    assert stream["rgb_compressed_topic"] == "/aima/hal/sensor/rgb_head_front_center/rgb_image_half/compressed"
    assert stream["require_unique_publisher"] is True
    assert stream["decoder"] == "auto"
    assert json.loads(stream["decoder_params_json"]) == {"device_id": 0}
    assert config["mirror_session"]["publish_enabled"] is False


def test_half_resolution_topic_uses_the_existing_frame_source_contract():
    config = stream_config(
        frame_source={
            "topic": "/camera/stereo_left/half/compressed",
            "decoder": "dvpp",
            "decoder_params": {"device_id": 1},
        }
    )
    stream = parameters(generate_motion_imitation_stream_nodes(config)[0])
    assert stream["rgb_compressed_topic"] == "/camera/stereo_left/half/compressed"
    assert stream["frame_source_interface"] == ""
    assert stream["decoder"] == "dvpp"
    assert json.loads(stream["decoder_params_json"]) == {"device_id": 1}


def test_existing_half_resolution_topic_can_require_a_single_publisher():
    config = stream_config(
        frame_source={
            "topic": "/aima/hal/sensor/rgb_head_front_center/rgb_image_half/compressed",
            "require_unique_publisher": True,
        }
    )
    stream = parameters(generate_motion_imitation_stream_nodes(config)[0])
    assert stream["rgb_compressed_topic"] == "/aima/hal/sensor/rgb_head_front_center/rgb_image_half/compressed"
    assert stream["require_unique_publisher"] is True


def test_require_unique_publisher_must_be_boolean():
    config = stream_config(frame_source={"topic": "/camera/left/compressed", "require_unique_publisher": "yes"})
    with pytest.raises(ValueError, match="require_unique_publisher"):
        generate_motion_imitation_stream_nodes(config)
