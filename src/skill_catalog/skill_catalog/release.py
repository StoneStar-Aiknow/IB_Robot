"""Materialize and atomically activate immutable catalog releases."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from skill_catalog.digest import compute_skill_package_digest, to_canonical_json
from skill_catalog.source import (
    SkillCatalogError,
    _compute_catalog_release_digest,
    build_package_file_manifest,
    build_release_file_manifest,
    load_yaml_mapping,
)


def _release_versions(config_root: Path) -> dict[str, str]:
    versions = {}
    skills_root = config_root / "skills"
    for package in sorted(skills_root.iterdir()):
        if not package.is_dir():
            continue
        manifest = load_yaml_mapping(package / "manifest.yaml")
        name = str(manifest.get("name", package.name))
        version = str(manifest.get("version", ""))
        digest = compute_skill_package_digest(build_package_file_manifest(package))
        versions[f"{name}@{version}"] = digest
    return versions


def materialize_release(source_root: Path, destination_root: Path) -> str:
    """Copy one catalog release, enforce SemVer history, and switch ``current``."""

    source_root = Path(source_root).resolve()
    destination_root = Path(destination_root).resolve()
    source_config = source_root / "config"
    if not source_config.is_dir() or source_config.is_symlink():
        raise SkillCatalogError("source config is missing", code="SKILL_RELEASE_NOT_IMMUTABLE")
    build_release_file_manifest(source_config)
    versions = _release_versions(source_config)
    index_path = destination_root / "release_index.json"
    history = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    conflicts = sorted(key for key, digest in versions.items() if key in history and history[key] != digest)
    if conflicts:
        raise SkillCatalogError(
            "published skill version changed content: " + ", ".join(conflicts),
            code="SKILL_SEMVER_CONTENT_CHANGED",
        )

    releases_root = destination_root / "releases"
    destination_root.mkdir(parents=True, exist_ok=True)
    releases_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".catalog-release-", dir=destination_root) as temporary:
        staged = Path(temporary) / "release"
        shutil.copytree(source_config, staged / "config", symlinks=False)
        digest = _compute_catalog_release_digest(staged)
        release_root = releases_root / digest
        if release_root.exists():
            if _compute_catalog_release_digest(release_root) != digest:
                raise SkillCatalogError("existing release is mutable", code="SKILL_RELEASE_NOT_IMMUTABLE")
        else:
            os.replace(staged, release_root)

    updated_history = {**history, **versions}
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination_root, delete=False) as handle:
        handle.write(to_canonical_json(updated_history))
        index_temporary = Path(handle.name)
    os.replace(index_temporary, index_path)
    pointer_temporary = destination_root / f".current-{os.getpid()}"
    pointer_temporary.unlink(missing_ok=True)
    pointer_temporary.symlink_to(Path("releases") / digest)
    os.replace(pointer_temporary, destination_root / "current")
    return digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("destination_root", type=Path)
    arguments = parser.parse_args(argv)
    print(materialize_release(arguments.source_root, arguments.destination_root))
    return 0
