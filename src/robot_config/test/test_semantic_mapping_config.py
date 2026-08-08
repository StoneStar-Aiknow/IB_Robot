"""Tests for the standalone semantic mapping SSOT contract."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from robot_config.loader import load_robot_config, load_robot_config_dict, validate_config

CONFIG_PATH = Path(__file__).parents[1] / "config" / "robots" / "lekiwi_mapping.yaml"
FIXTURE_PATH = Path(__file__).parents[2] / "semantic_mapping" / "test" / "perception_bundle_fixture.py"
FIXTURE_SPEC = importlib.util.spec_from_file_location("semantic_perception_bundle_fixture", FIXTURE_PATH)
FIXTURE_MODULE = importlib.util.module_from_spec(FIXTURE_SPEC)
assert FIXTURE_SPEC.loader is not None
FIXTURE_SPEC.loader.exec_module(FIXTURE_MODULE)
configure_perception_bundles = FIXTURE_MODULE.configure_perception_bundles


def _enabled_config(tmp_path: Path) -> dict:
    config = load_robot_config_dict(CONFIG_PATH)
    config.pop("_config_path")
    semantic_mapping = config["semantic_mapping"]
    semantic_mapping["enabled"] = True
    semantic_mapping["slam"].update(
        {
            "geometry_map_hash": "geometry-sha256",
            "localization_session_id": "localization-session-1",
            "calibration_id": "d435-calibration-sha256",
            "urdf_hash": "urdf-sha256",
        }
    )
    configure_perception_bundles(config, tmp_path / "bundles")
    return config


def _write_config(tmp_path: Path, robot: dict) -> Path:
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump({"robot": robot}, sort_keys=False), encoding="utf-8")
    return path


def test_disabled_semantic_mapping_contract_is_preserved() -> None:
    config = load_robot_config(CONFIG_PATH)

    assert config.semantic_mapping.enabled is False
    assert config.semantic_mapping.camera["mounting"] == "fixed"
    assert config.semantic_mapping.perception["mapping_backend"] == "service"
    assert [service.instance_id for service in config.perception_services.services] == [
        "semantic_sam2_masks",
        "semantic_ram_plus_tags",
        "semantic_siglip2_image",
        "semantic_siglip2_text",
        "semantic_gdino_confirmation",
        "semantic_graspgen_grasps",
    ]
    assert config.perception_services.enabled_services == ()
    assert config.semantic_mapping.migration["grounded_sam2_node"] == "compatibility"


def test_enabled_semantic_mapping_contract_loads_and_validates(tmp_path: Path) -> None:
    config = load_robot_config(_write_config(tmp_path, _enabled_config(tmp_path)))

    assert config.semantic_mapping.enabled is True
    assert config.semantic_mapping.camera["peripheral"] == "realsense"
    assert validate_config(config) == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda config: config["semantic_mapping"]["camera"].update({"mounting": "eye_in_hand"}),
            "semantic_mapping.camera.mounting must be 'fixed'",
        ),
        (
            lambda config: config["semantic_mapping"]["camera"].update({"parent_frame": "wrist_link"}),
            "semantic_mapping.camera.parent_frame must match",
        ),
        (
            lambda config: config["semantic_mapping"]["perception"].update({"model_backend": "cuda"}),
            "model_backend is unsupported",
        ),
        (
            lambda config: config["semantic_mapping"]["queue"].update({"max_masks_per_batch": 9}),
            "max_masks_per_batch must be <= 8",
        ),
        (
            lambda config: config["semantic_mapping"]["perception"]["semantic_roles"].update(
                {"siglip2_image": "missing"}
            ),
            "references unknown perception service",
        ),
        (
            lambda config: config["semantic_mapping"]["slam"].update(
                {"geometry_map_hash": "REPLACE_WITH_ACTIVE_GEOMETRY_MAP_HASH"}
            ),
            "slam.geometry_map_hash must be an active identity",
        ),
        (
            lambda config: config["semantic_mapping"]["perception"].update(
                {"mapping_backend": "embedded", "allow_legacy_embedded": False}
            ),
            "allow_legacy_embedded must be true",
        ),
        (
            lambda config: config["semantic_mapping"].update(
                {"migration": {**config["semantic_mapping"]["migration"], "grounded_sam2_snapshot": "production"}}
            ),
            "migration.grounded_sam2_snapshot must be one of",
        ),
    ],
)
def test_semantic_mapping_contract_fails_closed(tmp_path: Path, mutate, expected: str) -> None:
    config = copy.deepcopy(_enabled_config(tmp_path))
    mutate(config)

    with pytest.raises(ValueError, match=expected):
        load_robot_config_dict(_write_config(tmp_path, config))


def test_embedded_mapping_requires_explicit_migration_state(tmp_path: Path) -> None:
    config = _enabled_config(tmp_path)
    config["semantic_mapping"]["perception"].update({"mapping_backend": "embedded", "allow_legacy_embedded": True})

    with pytest.raises(ValueError, match="embedded_mapping_backend must be 'migration_only'"):
        load_robot_config_dict(_write_config(tmp_path, config))


def test_embedded_mapping_validation_does_not_require_generic_services(tmp_path: Path) -> None:
    config = _enabled_config(tmp_path)
    config["semantic_mapping"]["perception"].update({"mapping_backend": "embedded", "allow_legacy_embedded": True})
    config["semantic_mapping"]["migration"]["embedded_mapping_backend"] = "migration_only"
    config["perception_services"] = {"services": []}

    loaded = load_robot_config_dict(_write_config(tmp_path, config))
    assert loaded["semantic_mapping"]["perception"]["mapping_backend"] == "embedded"


def test_semantic_roles_enforce_type_policy_identity_and_embedding_compatibility(tmp_path: Path) -> None:
    wrong_type = _enabled_config(tmp_path / "wrong_type")
    wrong_type["perception_services"]["services"][0]["service_type"] = "ibrobot_msgs/srv/RecognizeTags"
    with pytest.raises(ValueError, match="must reference service type ibrobot_msgs/srv/GenerateMasks"):
        load_robot_config_dict(_write_config(tmp_path / "wrong_type", wrong_type))

    wrong_policy = _enabled_config(tmp_path / "wrong_policy")
    wrong_policy["perception_services"]["services"][0]["required"] = False
    with pytest.raises(ValueError, match="must reference an enabled required service"):
        load_robot_config_dict(_write_config(tmp_path / "wrong_policy", wrong_policy))

    incompatible = _enabled_config(tmp_path / "incompatible")
    text = incompatible["perception_services"]["services"][3]
    manifest_path = Path(text["bundle_path"]) / "inference_manifest.json"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["model"]["semantic_identity"]["embedding"]["embedding_space_id"] = "other-space"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="compatible embedding metadata"):
        load_robot_config_dict(_write_config(tmp_path / "incompatible", incompatible))
