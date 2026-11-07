"""Resolve the RT-Infer vendor bundle payload shipped with this repository.

The launch owner installs the repository's vendor bundle to the canonical
prefix ``/opt/rt-infer/<version>``. The payload is the set of files the
vendor's own ``manifest.json`` declares (artifacts, payload_files,
metadata_files) plus the control scripts the bootstrap itself needs. This
module is the single source of truth for locating that payload, so the same
launch works from a source checkout and from a colcon-installed workspace:

* **source tree**: ``$WORKSPACE/third_party/vendor/rt-infer/<version>/``
* **installed share**: the ament share data installed by robot_config's
  ``setup.py`` (``share/robot_config/rt_infer/<version>/``)

The vendor's install command copies exactly the manifest-listed files; this
module only resolves directories and never copies anything itself, and it
deliberately refuses bundles that are incomplete (a manifest without its
payload would install a prefix whose doctor refuses to run).
"""

from __future__ import annotations

import json
from pathlib import Path

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

_BUNDLE_SUBDIR = ("third_party", "vendor", "rt-infer")


def _manifest_files(manifest_path: Path) -> list[str]:
    """Every file name the vendor manifest declares (artifacts + payload)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = ["manifest.json"]
    for artifact in manifest.get("artifacts", {}).values():
        names.append(artifact["path"])
    for entry in manifest.get("payload_files", []):
        names.append(entry["path"])
    names.extend(manifest.get("metadata_files", []))
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name:
            raise ValueError(f"Unsafe vendor payload member: {name!r}")
    return names


def _complete_bundle(candidate: Path) -> Path | None:
    """The candidate if it carries the manifest AND every declared file."""
    manifest = candidate / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        names = _manifest_files(manifest)
    except (OSError, ValueError, KeyError):
        return None
    if all((candidate / name).is_file() for name in names):
        return candidate
    return None


def _source_candidate(bundle_version: str) -> Path | None:
    import os

    workspace = os.environ.get("WORKSPACE", "")
    if not workspace:
        return None
    return Path(workspace, *_BUNDLE_SUBDIR, bundle_version)


def _installed_candidate(bundle_version: str) -> Path | None:
    try:
        share = Path(get_package_share_directory("robot_config"))
    except PackageNotFoundError:
        return None
    return share / "rt_infer" / bundle_version


def resolve_vendor_bundle(bundle_version: str, bundle_dir_override: str = "") -> Path:
    """Locate the vendor bundle payload for ``bundle_version``.

    Resolution order: an explicit ``bundle_dir`` override (tests), the source
    tree, then the installed share. The winner must be complete; an
    incomplete bundle is an error naming what it is, because installing a
    partial prefix would leave a doctor that refuses to run and no path to
    repair it.
    """
    candidates: list[Path] = []
    if bundle_dir_override:
        complete = _complete_bundle(Path(bundle_dir_override))
        if complete is None:
            raise RuntimeError(f"Explicit RT-Infer bundle is not complete: {bundle_dir_override}")
        return complete
    for candidate_factory in (_source_candidate, _installed_candidate):
        candidate = candidate_factory(bundle_version)
        if candidate is not None:
            candidates.append(candidate)
    for candidate in candidates:
        complete = _complete_bundle(candidate)
        if complete is not None:
            return complete
    searched = [str(candidate) for candidate in candidates]
    raise RuntimeError(
        f"RT-Infer bundle {bundle_version!r} was not found complete in any of: {searched}. "
        "A complete bundle carries manifest.json and every file the manifest declares. "
        "In a source checkout that is third_party/vendor/rt-infer/<version>; an installed "
        "workspace ships it under robot_config's share data."
    )


def bootstrap_control_script(bundle: Path) -> Path:
    """The bundle's own priorityctl, used to install the bundle itself.

    The installed prefix does not exist before the first install, so the
    bootstrap phase must run the script that ships *with the bundle* — after
    the install completes, every later control call uses the installed copy.
    """
    script = bundle / "scripts" / "priorityctl.py"
    if not script.is_file():
        raise RuntimeError(f"the vendor bundle has no scripts/priorityctl.py to bootstrap from: {bundle}")
    return script


def expected_toolkit_home(bundle_or_prefix: Path) -> str | None:
    """Return the manifest-pinned CANN home when its official runtime exists.

    CANN's ``latest`` directory on openEuler is a real compatibility directory,
    not a symlink to the versioned toolkit.  The vendor guard intentionally
    requires the versioned home recorded by ``official_runtime_path``.  Use
    that exact home for vendor control/client subprocesses while leaving the
    launch process environment unchanged.
    """
    manifest_path = bundle_or_prefix / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        runtime = Path(manifest["compatibility"]["official_runtime_path"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not runtime.is_file() or len(runtime.parents) < 3:
        return None
    return str(runtime.parents[2])


__all__ = ["bootstrap_control_script", "expected_toolkit_home", "resolve_vendor_bundle"]
