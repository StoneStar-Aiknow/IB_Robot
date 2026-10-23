from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from skill_catalog.digest import to_canonical_json
from skill_catalog.models import SkillCatalogError
from skill_catalog.release import materialize_release
from skill_catalog.source import DirectoryReleaseSkillSource, build_release_file_manifest

CATALOG_ROOT = Path(__file__).resolve().parents[1]


def _copy_catalog(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    shutil.copytree(CATALOG_ROOT / "config", source / "config")
    return source


def test_materialize_release_uses_digest_directory_and_atomic_current(tmp_path):
    source = _copy_catalog(tmp_path)
    destination = tmp_path / "installed"

    digest = materialize_release(source, destination)
    release = DirectoryReleaseSkillSource(destination).resolve_active_release()

    assert release.source_release_digest == digest
    assert release.root == (destination / "releases" / digest).resolve()
    assert DirectoryReleaseSkillSource(destination).compute_release_digest(release) == digest
    assert (destination / "release_index.json").is_file()


def test_materialize_rejects_changed_content_without_version_bump(tmp_path):
    source = _copy_catalog(tmp_path)
    destination = tmp_path / "installed"
    materialize_release(source, destination)
    manifest = source / "config" / "skills" / "wave_hello" / "manifest.yaml"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")

    with pytest.raises(SkillCatalogError, match="published skill version changed content") as exc_info:
        materialize_release(source, destination)

    assert exc_info.value.code == "SKILL_SEMVER_CONTENT_CHANGED"


def test_materialize_checks_semver_against_staged_copy(tmp_path, monkeypatch):
    source = _copy_catalog(tmp_path)
    destination = tmp_path / "installed"
    materialize_release(source, destination)
    original_copytree = shutil.copytree

    def copy_and_mutate(src, dst):
        copied = original_copytree(src, dst, symlinks=True)
        manifest = copied / "skills" / "wave_hello" / "manifest.yaml"
        manifest.write_text(manifest.read_text(encoding="utf-8") + "\n# staged mutation\n", encoding="utf-8")
        return copied

    monkeypatch.setattr("skill_catalog.release._stage_source_config", copy_and_mutate)

    with pytest.raises(SkillCatalogError) as exc_info:
        materialize_release(source, destination)

    assert exc_info.value.code == "SKILL_SEMVER_CONTENT_CHANGED"


def test_materialize_merges_fresh_history_under_lock(tmp_path):
    source = _copy_catalog(tmp_path)
    destination = tmp_path / "installed"
    materialize_release(source, destination)
    index = destination / "release_index.json"
    history = json.loads(index.read_text(encoding="utf-8"))
    history["external@1.0.0"] = "f" * 64
    index.write_text(to_canonical_json(history), encoding="utf-8")

    materialize_release(source, destination)

    assert json.loads(index.read_text(encoding="utf-8"))["external@1.0.0"] == "f" * 64


def test_production_current_cannot_escape_releases_directory(tmp_path):
    outside = tmp_path / ("a" * 64)
    outside.mkdir()
    (tmp_path / "current").symlink_to(outside)

    with pytest.raises(SkillCatalogError, match="releases/<source_release_digest>"):
        DirectoryReleaseSkillSource(tmp_path).resolve_active_release()


@pytest.mark.parametrize("name", [".hidden", "temporary~", "file.swp"])
def test_release_manifest_rejects_hidden_and_temporary_files(tmp_path, name):
    config = tmp_path / "config"
    config.mkdir()
    (config / name).write_text("forbidden", encoding="utf-8")

    with pytest.raises(SkillCatalogError, match="forbidden"):
        build_release_file_manifest(config)


def test_release_manifest_rejects_internal_symlink(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    target = config / "target.yaml"
    target.write_text("{}", encoding="utf-8")
    (config / "link.yaml").symlink_to(target)

    with pytest.raises(SkillCatalogError, match="forbidden"):
        build_release_file_manifest(config)


def test_canonical_json_rejects_non_string_mapping_keys():
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        to_canonical_json({1: "integer", "1": "string"})
