"""Source abstraction for the skill catalog (section 9.6, 13).

All compiler, runtime and CLI paths must reuse the same source abstraction.
The default production path resolves through the ament package share; the
development path reads an explicit staging directory and re-validates the
release digest before and after a compile.
"""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import yaml

from skill_catalog.digest import compute_file_digest, compute_release_digest_from_manifest
from skill_catalog.models import SkillCatalogError

# Files that are never part of an immutable release (section 13).
_HIDDEN_PREFIXES = (".", "~", "#")
_EDITOR_SUFFIXES = ("~", ".swp", ".swo", ".swn", ".tmp", ".bak")


@dataclass(frozen=True)
class SkillReleaseLocation:
    """An immutable catalog release root."""

    root: Path
    source_release_digest: str = ""


@dataclass(frozen=True)
class SkillPackageLocation:
    """A discovered catalog entry package directory."""

    name: str
    package_dir: Path
    manifest_path: Path
    implementation_paths: Mapping[str, Path] = field(default_factory=dict)
    skill_md_path: Path | None = None
    source_relative_path: str = ""


def is_regular_catalog_file(path: Path, *, release_root: Path) -> bool:
    """Section 13: only regular files inside the release, no symlinks/hidden."""

    if path.is_symlink():
        return False
    if not path.is_file():
        return False
    try:
        path.relative_to(release_root)
    except ValueError:
        return False
    name = path.name
    if name.startswith(_HIDDEN_PREFIXES):
        return False
    return not any(name.endswith(suffix) for suffix in _EDITOR_SUFFIXES)


def iter_release_files(release_root: Path) -> list[Path]:
    """Return sorted files, rejecting forbidden release entries."""

    files: list[Path] = []
    for directory, dirs, names in os.walk(release_root, followlinks=False):
        dir_path = Path(directory)
        for name in dirs:
            candidate = dir_path / name
            if (
                candidate.is_symlink()
                or name.startswith(_HIDDEN_PREFIXES)
                or any(name.endswith(suffix) for suffix in _EDITOR_SUFFIXES)
            ):
                raise SkillCatalogError(
                    "release contains a forbidden directory entry",
                    code="SKILL_RELEASE_NOT_IMMUTABLE",
                    source_relative_path=str(candidate),
                )
        for name in names:
            candidate = dir_path / name
            if not is_regular_catalog_file(candidate, release_root=release_root):
                raise SkillCatalogError(
                    "release contains a forbidden file entry",
                    code="SKILL_RELEASE_NOT_IMMUTABLE",
                    source_relative_path=str(candidate),
                )
            files.append(candidate)
    files.sort()
    return files


def build_release_file_manifest(release_root: Path) -> list[dict[str, Any]]:
    """Build the ``files`` manifest (section 13) for a release root."""

    manifest: list[dict[str, Any]] = []
    for path in iter_release_files(release_root):
        relative = path.relative_to(release_root).as_posix()
        before = path.stat(follow_symlinks=False)
        digest = compute_file_digest(str(path))
        after = path.stat(follow_symlinks=False)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise SkillCatalogError(
                "release file changed while it was hashed",
                code="SKILL_SOURCE_CHANGED_DURING_COMPILE",
                source_relative_path=relative,
            )
        manifest.append(
            {
                "path": relative,
                "size": before.st_size,
                "sha256": digest,
            }
        )
    return manifest


def build_package_file_manifest(package_dir: Path) -> list[dict[str, Any]]:
    """Build the per-skill package file manifest (section 13)."""

    return build_release_file_manifest(package_dir)


def _compute_catalog_release_digest(release_root: Path) -> str:
    catalog_root = release_root / "config"
    if catalog_root.is_symlink() or not catalog_root.is_dir():
        raise SkillCatalogError(
            "release config directory is missing or is a symlink",
            code="SKILL_RELEASE_NOT_IMMUTABLE",
            source_relative_path=str(catalog_root),
        )
    return compute_release_digest_from_manifest(build_release_file_manifest(catalog_root))


class SkillSource(Protocol):
    """Source abstraction that every compiler/runtime/CLI must reuse."""

    def resolve_active_release(self) -> SkillReleaseLocation: ...

    def discover_packages(self, release: SkillReleaseLocation) -> Sequence[SkillPackageLocation]: ...

    def load_profile(self, release: SkillReleaseLocation, profile_name: str) -> Mapping[str, Any]: ...

    def compute_release_digest(self, release: SkillReleaseLocation) -> str: ...


# --------------------------------------------------------------------------- #
# YAML helpers                                                                #
# --------------------------------------------------------------------------- #


# Same safe constructor and resolver as ``yaml.safe_load``; the libyaml parser
# only makes the 70+ catalog files several times faster to read for every
# ``robot-skill`` invocation.  Fall back when PyYAML was built without it.
_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise SkillCatalogError(
            f"file does not exist: {path}",
            code="SKILL_PACKAGE_NOT_FOUND",
            source_relative_path=str(path),
        )
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = yaml.load(handle, Loader=_SAFE_LOADER)  # noqa: S506 - safe loader
    except yaml.YAMLError as exc:
        raise SkillCatalogError(
            "invalid YAML",
            code="SKILL_SCHEMA_INVALID",
        ) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise SkillCatalogError(
            f"YAML root must be a mapping: {path}",
            code="SKILL_SCHEMA_INVALID",
            source_relative_path=str(path),
        )
    return loaded


# --------------------------------------------------------------------------- #
# DirectoryReleaseSkillSource (production)                                    #
# --------------------------------------------------------------------------- #


class DirectoryReleaseSkillSource:
    """Load from an immutable release directory (section 9.6, 13).

    The release root must be ``<root>/releases/<digest>`` and ``<root>/current``
    is the only symlink allowed at the release level. Inside the release,
    symlinks are rejected.
    """

    def __init__(self, releases_root: Path) -> None:
        self._releases_root = Path(releases_root)

    def resolve_active_release(self) -> SkillReleaseLocation:
        current = self._releases_root / "current"
        if not current.is_symlink():
            raise SkillCatalogError(
                "production source requires an atomic 'current' symlink",
                code="SKILL_RELEASE_NOT_IMMUTABLE",
                source_relative_path=str(current),
            )
        target = Path(os.readlink(current))
        if not target.is_absolute():
            target = self._releases_root / target
        if not target.is_dir():
            raise SkillCatalogError(
                "current release target does not exist",
                code="SKILL_RELEASE_NOT_IMMUTABLE",
                source_relative_path=str(current),
            )
        resolved_target = target.resolve()
        releases = (self._releases_root / "releases").resolve()
        if resolved_target.parent != releases or not re.fullmatch(r"[0-9a-f]{64}", resolved_target.name):
            raise SkillCatalogError(
                "current must target releases/<source_release_digest>",
                code="SKILL_RELEASE_NOT_IMMUTABLE",
                source_relative_path=str(current),
            )
        return SkillReleaseLocation(root=resolved_target, source_release_digest=resolved_target.name)

    def discover_packages(self, release: SkillReleaseLocation) -> list[SkillPackageLocation]:
        skills_dir = release.root / "config" / "skills"
        if not skills_dir.is_dir():
            return []
        packages: list[SkillPackageLocation] = []
        for entry in sorted(skills_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith(_HIDDEN_PREFIXES):
                continue
            manifest_path = entry / "manifest.yaml"
            impl_paths: dict[str, Path] = {}
            if manifest_path.is_file():
                try:
                    manifest = load_yaml_mapping(manifest_path)
                except SkillCatalogError:
                    manifest = {}
                implementations = manifest.get("implementations", {})
                if isinstance(implementations, dict):
                    for impl_name, rel in implementations.items():
                        impl_paths[str(impl_name)] = entry / str(rel)
            skill_md = entry / "SKILL.md"
            packages.append(
                SkillPackageLocation(
                    name=entry.name,
                    package_dir=entry,
                    manifest_path=manifest_path,
                    implementation_paths=MappingProxyType(impl_paths),
                    skill_md_path=skill_md if skill_md.is_file() else None,
                    source_relative_path=str(entry.relative_to(release.root).as_posix()),
                )
            )
        return packages

    def load_profile(self, release: SkillReleaseLocation, profile_name: str) -> Mapping[str, Any]:
        profile_path = release.root / "config" / "profiles" / f"{profile_name}.yaml"
        return load_yaml_mapping(profile_path)

    def compute_release_digest(self, release: SkillReleaseLocation) -> str:
        if release.source_release_digest:
            self._verify_release_immutable(release)
        return _compute_catalog_release_digest(release.root)

    @staticmethod
    def _verify_release_immutable(release: SkillReleaseLocation) -> None:
        expected = release.source_release_digest
        actual = _compute_catalog_release_digest(release.root)
        if expected and actual != expected:
            raise SkillCatalogError(
                "release content changed after activation",
                code="SKILL_RELEASE_NOT_IMMUTABLE",
                source_relative_path=str(release.root),
            )


# --------------------------------------------------------------------------- #
# DevelopmentStagingSkillSource                                               #
# --------------------------------------------------------------------------- #


class DevelopmentStagingSkillSource:
    """Load from a development staging directory (section 9.6).

    Validates the release digest before and after a compile so that mid-compile
    edits are rejected.
    """

    def __init__(self, staging_root: Path) -> None:
        self._staging_root = Path(staging_root)

    def resolve_active_release(self) -> SkillReleaseLocation:
        if not self._staging_root.is_dir():
            raise SkillCatalogError(
                "staging source root does not exist",
                code="SKILL_RELEASE_NOT_IMMUTABLE",
                source_relative_path=str(self._staging_root),
            )
        digest = _compute_catalog_release_digest(self._staging_root)
        return SkillReleaseLocation(root=self._staging_root, source_release_digest=digest)

    def discover_packages(self, release: SkillReleaseLocation) -> list[SkillPackageLocation]:
        return DirectoryReleaseSkillSource(Path("/")).discover_packages(release)

    def load_profile(self, release: SkillReleaseLocation, profile_name: str) -> Mapping[str, Any]:
        return DirectoryReleaseSkillSource(Path("/")).load_profile(release, profile_name)

    def compute_release_digest(self, release: SkillReleaseLocation) -> str:
        return _compute_catalog_release_digest(release.root)

    def assert_unchanged(self, release: SkillReleaseLocation) -> None:
        current = _compute_catalog_release_digest(release.root)
        if current != release.source_release_digest:
            raise SkillCatalogError(
                "source tree changed during compile",
                code="SKILL_SOURCE_CHANGED_DURING_COMPILE",
                source_relative_path=str(release.root),
            )


def package_source_root() -> Path | None:
    """Return the source tree of this package when it runs from a development checkout.

    ``colcon build --symlink-install`` links the installed module back to the
    source tree, so the package root (``package.xml`` + ``config/``) is the
    parent of the resolved module directory. A regular install resolves into
    site-packages and returns ``None``.
    """

    package_root = Path(__file__).resolve().parent.parent
    if (package_root / "package.xml").is_file() and (package_root / "config").is_dir():
        return package_root
    return None


def resolve_development_source_root(configured_root: str = "", config_path: Path | None = None) -> Path | None:
    """Resolve ``skill_catalog_source_root`` for ``development`` mode.

    An absolute root is used as-is. An empty root selects the source tree of
    the installed ``skill_catalog`` package. A relative root is resolved against
    the repository that holds ``config_path``; when it does not exist there, it
    falls back to the package source tree with a ``FutureWarning``. ``None``
    means no source tree is available and the caller uses the installed catalog.
    """

    configured_root = str(configured_root or "").strip()
    if configured_root and Path(configured_root).is_absolute():
        return Path(configured_root)
    if configured_root:
        if config_path is not None and str(config_path):
            anchor = Path(config_path).resolve()
            repository_root = next((parent.parent for parent in anchor.parents if parent.name == "src"), None)
            if repository_root is not None:
                candidate = (repository_root / configured_root).resolve()
                if candidate.is_dir():
                    return candidate
        warnings.warn(
            f"skill_catalog_source_root {configured_root!r} does not exist; falling back to the skill_catalog "
            "package source tree. Drop the setting: development mode now defaults to that tree.",
            FutureWarning,
            stacklevel=2,
        )
    return package_source_root()


class AmentShareSkillSource:
    """Load the installed catalog from the ``skill_catalog`` ament share."""

    def __init__(self, package_name: str = "skill_catalog") -> None:
        self._package_name = package_name

    def resolve_active_release(self) -> SkillReleaseLocation:
        try:
            from ament_index_python.packages import get_package_share_directory

            root = Path(get_package_share_directory(self._package_name))
        except Exception as exc:
            raise SkillCatalogError(
                "installed skill catalog package is unavailable",
                code="SKILL_RELEASE_NOT_IMMUTABLE",
                source_relative_path=self._package_name,
            ) from exc
        if not (root / "config").is_dir():
            raise SkillCatalogError(
                "installed skill catalog config is missing",
                code="SKILL_RELEASE_NOT_IMMUTABLE",
                source_relative_path=str(root),
            )
        digest = _compute_catalog_release_digest(root)
        return SkillReleaseLocation(root=root, source_release_digest=digest)

    def discover_packages(self, release: SkillReleaseLocation) -> list[SkillPackageLocation]:
        return DirectoryReleaseSkillSource(Path("/")).discover_packages(release)

    def load_profile(self, release: SkillReleaseLocation, profile_name: str) -> Mapping[str, Any]:
        return DirectoryReleaseSkillSource(Path("/")).load_profile(release, profile_name)

    def compute_release_digest(self, release: SkillReleaseLocation) -> str:
        return _compute_catalog_release_digest(release.root)
