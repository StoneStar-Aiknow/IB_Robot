"""Launch wiring for the resident motion-imitation frame stream.

The high-value assertion is the last one: the node's service parameters must
match the endpoints the perception model services actually advertise. A typo
there is otherwise a silent runtime hang with no error anywhere in the graph.

Migrated from ``robot_config/test/test_motion_imitation_stream_builder.py``
The end-to-end test loads the full ``aimdk_x2_magicmirror`` profile, so it needs the model
bundles that profile references (``yolox_x_640`` / ``pear_parameter_network`` / ``zipvoice``);
without them it skips (the root conftest's ``model_bundle`` marker), like the other suites that
drive real robot configs.

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
from robot_config.loader import load_robot_config_dict, resolve_robot_config_path

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


@pytest.mark.model_bundle("yolox_x_640")
@pytest.mark.model_bundle("pear_parameter_network")
@pytest.mark.model_bundle("zipvoice")
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


# ----- the standing-zone selector's files (dev plan D11-D15) -----
# D11 is test_enabled_block_launches_one_stream_node_with_defaults above: with
# no zone key the parameters are exactly the old ones.

_ZONE_YAML = """\
version: 1
image_size: [1000, 800]
crop_x: [100, 900]
floor_quad:
  - [300, 700]
  - [700, 700]
  - [650, 500]
  - [350, 500]
note: test
"""


# near_left 10 px from the bottom, under the 30 px default margin.
_NEAR_EDGE_TOO_LOW = _ZONE_YAML.replace("[300, 700]", "[300, 790]")


def _zone_files(tmp_path, zone=_ZONE_YAML, tuning="version: 1\nrelease_hold_s: 3.0\n"):
    zone_file = tmp_path / "actor_zone.yaml"
    zone_file.write_text(zone, encoding="utf-8")
    tuning_file = tmp_path / "actor_tuning.yaml"
    tuning_file.write_text(tuning, encoding="utf-8")
    return str(zone_file), str(tuning_file)


def test_zone_and_tuning_files_reach_the_node_as_absolute_paths(tmp_path):
    # D12
    zone_file, tuning_file = _zone_files(tmp_path)
    config = stream_config(actor_zone_file=zone_file, actor_tuning_file=tuning_file)
    stream = parameters(generate_motion_imitation_stream_nodes(config)[0])
    assert stream["actor_zone_file"] == zone_file
    assert stream["actor_tuning_file"] == tuning_file
    assert stream["actor_candidates_topic"] == "/motion_imitation_stream/actor_candidates"


def test_a_zone_without_a_tuning_file_uses_the_defaults(tmp_path):
    zone_file, _ = _zone_files(tmp_path)
    stream = parameters(generate_motion_imitation_stream_nodes(stream_config(actor_zone_file=zone_file))[0])
    assert stream["actor_zone_file"] == zone_file
    assert "actor_tuning_file" not in stream


@pytest.mark.parametrize(
    "case, match",
    [
        ("bad_zone", "actor_zone_file.*floor_quad"),
        ("missing_zone", "actor_zone_file.*cannot read"),
        ("relative_zone", "actor_zone_file must be an absolute path"),
        ("bad_tuning", "actor_tuning_file.*release_hold_s"),
        ("tuning_without_zone", "need actor_zone_file"),
        ("candidates_without_zone", "need actor_zone_file"),
        ("candidates_on_the_state_topic", "actor_candidates_topic must differ"),
    ],
)
def test_bad_zone_settings_stop_the_launch_with_the_field_named(tmp_path, case, match):
    # D13
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    good.mkdir()
    bad.mkdir()
    zone_file, tuning_file = _zone_files(good)
    overrides = {
        "bad_zone": lambda: {"actor_zone_file": _zone_files(bad, zone=_NEAR_EDGE_TOO_LOW)[0]},
        "missing_zone": lambda: {"actor_zone_file": str(tmp_path / "absent.yaml")},
        "relative_zone": lambda: {"actor_zone_file": "zone/actor_zone.yaml"},
        "bad_tuning": lambda: {
            "actor_zone_file": zone_file,
            "actor_tuning_file": _zone_files(bad, tuning="version: 1\nrelease_hold_s: 31.0\n")[1],
        },
        "tuning_without_zone": lambda: {"actor_tuning_file": tuning_file},
        "candidates_without_zone": lambda: {"actor_candidates_topic": "/x/candidates"},
        "candidates_on_the_state_topic": lambda: {
            "actor_zone_file": zone_file,
            "actor_candidates_topic": "/motion_imitation_stream/actor_state",
        },
    }[case]()
    with pytest.raises(ValueError, match=match):
        generate_motion_imitation_stream_nodes(stream_config(**overrides))


@pytest.mark.model_bundle("yolox_x_640", "pear_parameter_network", "zipvoice")
def test_zone_profile_is_dexhand_rec_plus_the_two_board_files():
    # D15: parsed only; the files live on the board, so the builder is not run.
    # The base is read beside the zone profile, not by name: before a build the
    # new profile exists only in the source tree while the old one is also
    # installed, and the two copies would differ in path.
    zone_path = resolve_robot_config_path(config_name="aimdk_x2_magicmirror_live_quiet_dexhand_rec_zone")
    # Live profiles bind their logical interfaces at launch, as in the test above.
    zone = load_robot_config_dict(zone_path, defer_interface_binding=True)
    base = load_robot_config_dict(
        zone_path.parent / "aimdk_x2_magicmirror_live_quiet_dexhand_rec.yaml", defer_interface_binding=True
    )
    stream = dict(zone["motion_imitation_stream"])
    assert stream.pop("actor_zone_file") == "/root/x2-actor-run/zone/actor_zone.yaml"
    assert stream.pop("actor_tuning_file") == "/root/x2-actor-run/zone/actor_tuning.yaml"
    assert stream == base["motion_imitation_stream"]
    assert "actor_zone_file" not in base["motion_imitation_stream"]
    provenance = ("motion_imitation_stream", "_config_path", "_config_sources")
    assert {k: v for k, v in zone.items() if k not in provenance} == {
        k: v for k, v in base.items() if k not in provenance
    }
    assert zone["_config_sources"] == [*base["_config_sources"], zone["_config_path"]]
