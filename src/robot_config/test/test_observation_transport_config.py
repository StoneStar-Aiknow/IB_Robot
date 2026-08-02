"""Tests for observation video transport configuration."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from robot_config.contract_utils import contract_fingerprint
from robot_config.generators.contract import (
    build_contract_from_robot_config_dict,
    generate_contract_from_robot_config,
    load_contract_with_robot_config,
)
from robot_config.loader import load_robot_config
from robot_config.observation_transport import (
    ObservationTransportSpec,
    effective_observation_transport,
    parse_observation_transport,
    validate_observation_transports,
    validate_robot_config_observation_transports,
)


def _rtp_observation(*, stream_id="top", port=5004, color_range="limited"):
    return {
        "key": "observation.images.top",
        "topic": "/camera/top/image_raw",
        "type": "sensor_msgs/msg/Image",
        "image": {"resize": [480, 640], "encoding": "rgb8"},
        "transport": {
            "mode": "rtp",
            "stream_id": stream_id,
            "endpoint": {"host": "127.0.0.1", "port": port},
            "h264": {"profile": "main", "bitrate_bps": 4_000_000, "gop_frames": 15},
            "media": {
                "width": 640,
                "height": 480,
                "frame_rate_hz": 30,
                "pixel_format": "nv12",
                "color_space": "bt709",
                "color_range": color_range,
            },
        },
    }


def _contract(*observations):
    return build_contract_from_robot_config_dict(
        {"name": "test", "contract": {"rate_hz": 20, "observations": list(observations), "actions": []}}
    )


def test_omitted_transport_defaults_to_dds_without_changing_declaration():
    contract = _contract(
        {
            "key": "observation.state",
            "topic": "/joint_states",
            "type": "sensor_msgs/msg/JointState",
        }
    )

    assert contract.observations[0].transport is None
    assert effective_observation_transport(contract.observations[0].transport).mode == "dds"


def test_rtp_transport_parses_and_validates():
    contract = _contract(_rtp_observation())

    transport = contract.observations[0].transport
    assert transport is not None
    assert transport.mode == "rtp"
    assert transport.media.width == 640
    assert transport.buffer.retention_ms == 1000
    assert validate_observation_transports(contract.observations) == []


def test_peripheral_values_complete_rtp_media():
    observation = _rtp_observation()
    observation["peripheral"] = "top"
    del observation["transport"]["media"]["width"]
    del observation["transport"]["media"]["height"]
    del observation["transport"]["media"]["frame_rate_hz"]
    contract = build_contract_from_robot_config_dict(
        {
            "name": "test",
            "peripherals": [
                {
                    "type": "camera",
                    "name": "top",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "pixel_format": "rgb8",
                }
            ],
            "contract": {"observations": [observation], "actions": []},
        }
    )

    assert contract.observations[0].transport.media.width == 640
    assert contract.observations[0].transport.media.height == 480
    assert contract.observations[0].transport.media.frame_rate_hz == 30


def test_transport_parser_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unsupported fields"):
        parse_observation_transport({"mode": "rtp", "protocol": "rtsp"})


def test_transport_validation_rejects_duplicate_streams_and_endpoints():
    observation = _contract(_rtp_observation()).observations[0]
    errors = validate_observation_transports([observation, observation])

    assert any("duplicates stream_id" in error for error in errors)
    assert any("duplicates RTP endpoint" in error for error in errors)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("security", "srtp", "security"),
        ("codec", "hevc", "codec"),
        ("encoder_backend", "unknown", "backend"),
    ],
)
def test_transport_validation_rejects_unsupported_claims(field, value, expected):
    observation = _rtp_observation()
    observation["transport"][field] = value

    with pytest.raises(ValueError, match=expected):
        _contract(observation)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("h264", "bitrate_bps", 0),
        ("h264", "gop_frames", 0),
        ("media", "width", 0),
        ("media", "frame_rate_hz", 0),
        ("buffer", "retention_ms", 0),
        ("readiness", "keyframe_timeout_ms", 0),
    ],
)
def test_transport_validation_rejects_non_positive_bounds(section, field, value):
    observation = _rtp_observation()
    observation["transport"].setdefault(
        "buffer",
        {
            "sender_queue_frames": 2,
            "receiver_queue_packets": 256,
            "decoded_frame_capacity": 32,
            "retention_ms": 1000,
        },
    )
    observation["transport"].setdefault(
        "readiness",
        {"keyframe_timeout_ms": 3000, "timestamp_mapping_max_age_ms": 1000, "max_inter_camera_skew_ms": 50},
    )
    observation["transport"][section][field] = value

    with pytest.raises(ValueError):
        _contract(observation)


def test_explicit_dds_rejects_rtp_fields():
    with pytest.raises(ValueError, match="dds.*RTP-specific"):
        parse_observation_transport({"mode": "dds", "stream_id": "camera"})


def test_rtp_requires_distributed_pipeline_when_mode_is_known():
    errors = validate_observation_transports(
        _contract(_rtp_observation()).observations,
        distributed_enabled=False,
    )

    assert any("distributed inference pipeline" in error for error in errors)


def test_raw_robot_config_rejects_rtp_without_distributed_pipeline():
    robot_config = {
        "contract": {"observations": [_rtp_observation()]},
        "control_modes": {"model_inference": {"inference": {"pipelines": {}}}},
    }

    errors = validate_robot_config_observation_transports(robot_config)

    assert any("distributed inference pipeline" in error for error in errors)


def test_transport_and_image_semantics_change_contract_fingerprint():
    base = _contract(_rtp_observation())
    changed_range = _contract(_rtp_observation(color_range="full"))
    changed_endpoint = _contract(_rtp_observation(port=5006))
    changed_observation = _rtp_observation()
    changed_observation["image"] = {"resize": [480, 640], "encoding": "bgr8"}
    changed_image = _contract(changed_observation)

    assert contract_fingerprint(base) != contract_fingerprint(changed_range)
    assert contract_fingerprint(base) != contract_fingerprint(changed_endpoint)
    assert contract_fingerprint(base) != contract_fingerprint(changed_image)


def test_omitted_and_explicit_default_dds_have_equal_fingerprints():
    observation = {
        "key": "observation.state",
        "topic": "/joint_states",
        "type": "sensor_msgs/msg/JointState",
    }
    omitted = _contract(observation)
    explicit = _contract({**observation, "transport": {"mode": "dds"}})

    assert contract_fingerprint(omitted) == contract_fingerprint(explicit)


def test_explicit_rtp_mode_is_never_rewritten_to_dds():
    transport = parse_observation_transport({"mode": "rtp"})
    assert transport == ObservationTransportSpec(mode="rtp")
    assert replace(transport, stream_id="top").mode == "rtp"


def test_generated_yaml_round_trip_preserves_rtp_transport():
    config = load_robot_config(
        # The development profile has no runtime dependency on a camera being present during parsing.
        "src/robot_config/config/robots/dev_rtp_single_camera.yaml"
    )

    generated = generate_contract_from_robot_config(config)
    round_trip = yaml.safe_load(generated)
    original_transport = round_trip["observations"][0]["transport"]
    reloaded = build_contract_from_robot_config_dict({"name": "round-trip", "contract": round_trip})

    assert reloaded.observations[0].transport == config.to_contract().observations[0].transport
    assert original_transport["mode"] == "rtp"


def test_standalone_contract_yaml_preserves_rtp_transport(tmp_path):
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        yaml.safe_dump({"name": "standalone", "observations": [_rtp_observation()], "actions": []}),
        encoding="utf-8",
    )

    contract = load_contract_with_robot_config(contract_path)

    assert contract.observations[0].transport == _contract(_rtp_observation()).observations[0].transport


def test_development_profile_typed_and_raw_contracts_match():
    path = Path("src/robot_config/config/robots/dev_rtp_multi_camera.yaml")
    typed = load_robot_config(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["robot"]

    typed_contract = typed.to_contract()
    raw_contract = build_contract_from_robot_config_dict(raw)

    assert typed_contract.observations == raw_contract.observations
