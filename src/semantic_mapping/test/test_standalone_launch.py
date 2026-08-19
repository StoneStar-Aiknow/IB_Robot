"""Package-level tests for standalone semantic mapping launch entries."""

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from perception_bundle_fixture import configure_perception_bundles

PACKAGE_ROOT = Path(__file__).parents[1]
ROBOT_CONFIG = PACKAGE_ROOT.parent / "robot_config" / "config" / "robots" / "lekiwi_realsense_mapping.yaml"


@dataclass
class _LaunchContext:
    launch_configurations: dict[str, str]


def _load_launch(name: str):
    path = PACKAGE_ROOT / "launch" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _enabled_config(tmp_path: Path) -> Path:
    data = yaml.safe_load(ROBOT_CONFIG.read_text(encoding="utf-8"))
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


def _node_parameters(node) -> dict:
    parameters = vars(node)["_Node__parameters"][0]
    normalized = {}
    for key, value in parameters.items():
        normalized_key = "".join(getattr(part, "text", str(part)) for part in key)
        if isinstance(value, tuple):
            normalized_value = "".join(getattr(part, "text", str(part)) for part in value).removesuffix("\n...\n")
        else:
            normalized_value = value
        normalized[normalized_key] = normalized_value
    return normalized


def test_online_launch_includes_all_enabled_referenced_generic_services(tmp_path: Path) -> None:
    module = _load_launch("semantic_mapping.launch.py")
    actions = module.launch_setup(
        _LaunchContext({"robot_config": "unused", "config_path": str(_enabled_config(tmp_path))})
    )

    assert len(actions) == 6
    assert {vars(node)["_Node__node_name"] for node in actions[:-1]} == {
        "model_service_semantic_sam2_masks",
        "model_service_semantic_ram_plus_tags",
        "model_service_semantic_siglip2_image",
        "model_service_semantic_siglip2_text",
        "model_service_semantic_gdino_confirmation",
    }
    node = actions[-1]
    assert vars(node)["_Node__package"] == "semantic_mapping"
    assert vars(node)["_Node__node_executable"] == "semantic_mapping_node"
    parameters = _node_parameters(node)
    assert parameters["mapping_backend"] == "service"
    assert parameters["geometry_map_hash"] == "geometry-hash"
    assert parameters["active_map_hash_topic"] == "/slam/active_geometry_map_hash"
    assert parameters["localization_ready_topic"] == "/slam/localization_ready"
    assert parameters["authoritative_map_odom_topic"] == "/slam/authoritative_map_odom_ready"
    assert parameters["reachability_ready_topic"] == "/navigation/reachability_ready"


def test_offline_launch_passes_explicit_bag_and_shared_contract(tmp_path: Path) -> None:
    module = _load_launch("offline_mapping.launch.py")
    actions = module.launch_setup(
        _LaunchContext(
            {
                "robot_config": "unused",
                "config_path": str(_enabled_config(tmp_path)),
                "bag_path": "/data/maps/run-1",
                "storage_id": "mcap",
                "start_frame": "120",
                "frame_sampling": "uniform",
            }
        )
    )

    assert len(actions) == 5
    assert {vars(node)["_Node__node_name"] for node in actions[:3]} == {
        "model_service_semantic_sam2_masks",
        "model_service_semantic_ram_plus_tags",
        "model_service_semantic_siglip2_image",
    }
    node = actions[-2]
    assert vars(node)["_Node__node_executable"] == "offline_mapping_node"
    assert actions[-1].__class__.__name__ == "RegisterEventHandler"
    parameters = _node_parameters(node)
    assert parameters["bag_path"] == "/data/maps/run-1"
    assert parameters["storage_id"] == "mcap"
    assert parameters["start_frame"] == 120
    assert parameters["frame_sampling"] == "uniform"
    assert '"logical_model_revision":"sam2@v1"' in parameters["sam_model_identity"]
    generic_parameters = _node_parameters(actions[0])
    assert generic_parameters["require_semantic_identity"] is True
    assert generic_parameters["configuration_generation"] == parameters["configuration_generation"]


def test_offline_launch_rejects_missing_bag_path() -> None:
    module = _load_launch("offline_mapping.launch.py")

    with pytest.raises(ValueError, match="bag_path is required"):
        module.launch_setup(_LaunchContext({"bag_path": ""}))


def test_embedded_online_launch_starts_no_generic_model_services(tmp_path: Path) -> None:
    path = _enabled_config(tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mapping = data["robot"]["semantic_mapping"]
    mapping["perception"].update({"mapping_backend": "embedded", "allow_legacy_embedded": True})
    data["robot"]["perception_services"] = {"services": []}
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    actions = _load_launch("semantic_mapping.launch.py").launch_setup(
        _LaunchContext({"robot_config": "unused", "config_path": str(path)})
    )

    assert len(actions) == 1
    assert _node_parameters(actions[0])["mapping_backend"] == "embedded"


def test_query_only_launch_starts_only_mapping_node_with_overrides(tmp_path: Path) -> None:
    module = _load_launch("semantic_mapping.launch.py")
    actions = module.launch_setup(
        _LaunchContext(
            {
                "robot_config": "unused",
                "config_path": str(_enabled_config(tmp_path)),
                "mode": "query_only",
                "database_path": "/data/semantic_map.sqlite3",
                "artifact_output_dir": "/data/artifacts",
            }
        )
    )

    assert len(actions) == 1
    parameters = _node_parameters(actions[0])
    assert parameters["database_path"] == "/data/semantic_map.sqlite3"
    assert parameters["artifact_output_dir"] == "/data/artifacts"
    assert parameters["geometry_map_hash"] == "geometry-hash"
