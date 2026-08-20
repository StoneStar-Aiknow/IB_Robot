"""Tests for the resolve_object_pose navigation skill manifest."""

from pathlib import Path

from skill_catalog.validator import validate_manifest

CATALOG_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = CATALOG_ROOT / "config" / "skills" / "resolve_object_pose" / "manifest.yaml"
IMPL_PATH = CATALOG_ROOT / "config" / "skills" / "resolve_object_pose" / "implementations" / "lekiwi_navigation_v2.yaml"


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_manifest_validates():
    manifest = _load_yaml(MANIFEST_PATH)
    errors = validate_manifest(
        manifest, package_name="resolve_object_pose", source_relative_path="resolve_object_pose/manifest.yaml"
    )
    assert errors == []


def test_manifest_has_query_only_contract():
    manifest = _load_yaml(MANIFEST_PATH)
    cap = manifest["capability"]
    assert cap["moves_robot"] is False
    assert cap["domain"] == "navigation"
    assert cap["required_control_mode"] == "base_navigation"
    params = cap["parameters"]["properties"]
    assert "target_name" in params
    assert "stand_off_distance_m" in params
    assert cap["parameters"]["required"] == ["target_name"]


def test_manifest_description_contract():
    manifest = _load_yaml(MANIFEST_PATH)
    desc = manifest["description"]
    assert desc["category"] == "navigation"
    assert desc["motion_scope"] == ["base"]
    assert desc["intensity"] == "subtle"
    assert desc["requires_motion_params"] is True
    assert "查物品位置" in desc.get("aliases_zh", [])


def test_implementation_is_delegated_executor():
    impl = _load_yaml(IMPL_PATH)
    assert impl["kind"] == "delegated_executor"
    assert impl["robot"] == "lekiwi_navigation_v2"
    assert impl["executor"] == "semantic_map_query"
    assert "target_name" in impl["required_args"]
