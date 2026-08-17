"""Tests for standalone semantic mapping SSOT parameter translation."""

import json
from pathlib import Path

import pytest
import yaml
from perception_bundle_fixture import configure_perception_bundles

from semantic_mapping.configuration import (
    load_semantic_mapping_robot_config,
    semantic_mapping_parameters,
    semantic_perception_nodes,
)

CONFIG_PATH = Path(__file__).parents[2] / "robot_config" / "config" / "robots" / "lekiwi_mapping.yaml"


def _write_enabled_config(tmp_path: Path) -> Path:
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    mapping = data["robot"]["semantic_mapping"]
    mapping["enabled"] = True
    mapping["slam"].update(
        {
            "geometry_map_hash": "geometry-hash",
            "localization_session_id": "session-id",
            "calibration_id": "calibration-hash",
            "urdf_hash": "urdf-hash",
        }
    )
    configure_perception_bundles(data["robot"], tmp_path / "bundles")
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_online_parameters_are_derived_from_top_level_ssot(tmp_path: Path) -> None:
    config = load_semantic_mapping_robot_config("unused", str(_write_enabled_config(tmp_path)))
    parameters = semantic_mapping_parameters(config)

    assert parameters["mapping_backend"] == "service"
    assert parameters["rgb_topic"] == "/camera/realsense/image_raw"
    assert parameters["cloud_map_topic"] == "/cloud_map"
    assert parameters["frame_queue_policy"] == "drop_oldest"
    assert parameters["max_masks_per_frame"] == 32
    assert parameters["max_masks_per_batch"] == 8
    assert parameters["max_object_distance_m"] == 2.5
    assert parameters["max_object_extent_m"] == 0.65
    assert parameters["association_max_size_ratio"] == 4.0
    assert parameters["label_switch_confidence_margin"] == 0.05
    assert parameters["label_recurrence_count_ratio"] == 3.0
    assert parameters["label_high_confidence_override_margin"] == 0.08
    assert json.loads(parameters["allowed_label_aliases_json"]) == {}
    assert "actionable_labels" not in parameters
    assert '"embedding_space_id":"siglip2-test-space:v1"' in parameters["siglip2_model_identity"]
    assert 0 < parameters["configuration_generation"] < 2**63
    assert parameters["target_service"] == "/semantic_mapping/resolve_target"
    assert parameters["track_state_updates_enabled"] is True
    assert parameters["track_state_topic"] == "/object_tracker/track_state"
    assert parameters["track_state_max_age_sec"] == 1.0
    assert parameters["track_state_max_covariance_m2"] == 0.25
    assert parameters["track_state_frame"] == "odom"
    assert parameters["track_state_confirmation_gap_sec"] == 1.0
    assert parameters["track_state_persist_interval_sec"] == 1.0
    assert parameters["gdino_confirmation_service"] == ("/semantic_perception/semantic_gdino_confirmation")


def test_offline_parameters_reuse_identity_filtering_and_services(tmp_path: Path) -> None:
    config = load_semantic_mapping_robot_config("unused", str(_write_enabled_config(tmp_path)))
    parameters = semantic_mapping_parameters(config, offline=True)

    assert parameters["global_frame"] == "map"
    assert parameters["sam_service"] == "/semantic_perception/semantic_sam2_masks"
    assert parameters["min_mask_valid_depth_ratio"] == 0.2
    assert parameters["artifact_output_dir"] == "~/.ros/ibrobot/semantic_artifacts"
    assert "mapping_backend" not in parameters


def test_configuration_generation_is_order_and_path_independent(tmp_path: Path) -> None:
    first = load_semantic_mapping_robot_config("unused", str(_write_enabled_config(tmp_path / "first")))
    second_path = _write_enabled_config(tmp_path / "second")
    data = yaml.safe_load(second_path.read_text(encoding="utf-8"))
    data["robot"]["perception_services"]["services"].reverse()
    second_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    second = load_semantic_mapping_robot_config("unused", str(second_path))

    assert (
        semantic_mapping_parameters(first)["configuration_generation"]
        == semantic_mapping_parameters(second)["configuration_generation"]
    )


def test_standalone_launch_rejects_disabled_mapping_config() -> None:
    with pytest.raises(ValueError, match="semantic_mapping.enabled must be true"):
        load_semantic_mapping_robot_config("unused", str(CONFIG_PATH))


def test_embedded_backend_does_not_require_service_bundles_or_identities(tmp_path: Path) -> None:
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    mapping = data["robot"]["semantic_mapping"]
    mapping["enabled"] = True
    mapping["perception"].update({"mapping_backend": "embedded", "allow_legacy_embedded": True})
    mapping["slam"].update(
        {
            "geometry_map_hash": "geometry-hash",
            "localization_session_id": "session-id",
            "calibration_id": "calibration-hash",
            "urdf_hash": "urdf-hash",
        }
    )
    data["robot"]["perception_services"] = {"services": []}
    path = tmp_path / "embedded.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    config = load_semantic_mapping_robot_config("unused", str(path))
    parameters = semantic_mapping_parameters(config)

    assert parameters["mapping_backend"] == "embedded"
    assert parameters["configuration_generation"] == 0
    assert "sam_model_identity" not in parameters
    assert semantic_perception_nodes(config) == []
