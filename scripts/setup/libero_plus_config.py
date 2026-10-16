"""Prepare workspace-owned configuration and assets for LIBERO-plus.

LIBERO-plus (https://github.com/sylvestf/LIBERO-plus) is a drop-in fork of
LIBERO kept as the pinned ``libs/libero_plus`` submodule. It shares the
top-level ``libero`` package name with the pip-installed ``hf-libero``
distribution, so it is never pip-installed: the benchmark adapter binds to it
per process through ``sys.path`` precedence and a dedicated
``LIBERO_CONFIG_PATH``.

This script mirrors ``libero_config.py`` for the plus provider:

1. resolving the pinned submodule checkout under ``${WORKSPACE}/libs/libero_plus``;
2. reusing a user-provided asset tree — setup never downloads the 6.4 GB
   ``assets.zip`` from the ``Sylvest/LIBERO-plus`` Hugging Face dataset. A
   ``assets.zip`` manually placed in the config directory is extracted and
   validated locally;
3. validating the asset tree against the pinned file count and SHA-256
   ( fail closed on any mismatch );
4. linking the provider's package-local ``assets`` path to the workspace-owned
   cache directory ( even while assets are still pending ); and
5. atomically writing ``${WORKSPACE}/venv/ibrobot_libero_plus/config.yaml``
   pointing at the submodule's ``bddl_files`` / ``init_files`` and the asset
   cache.

When assets are absent or empty, setup completes with ``assets_status=missing``:
it prints ``HINT:`` lines with the exact manual download command and leaves the
placeholder directory in place for the user to populate. Evaluation stays
unavailable until the tree validates.

Set ``IBROBOT_LIBERO_PLUS_ASSETS_DIR`` to reuse an explicitly selected local
asset tree; the same pinned hash validation applies.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PureWindowsPath

HF_ASSETS_REPO_ID = "Sylvest/LIBERO-plus"
# HF API history identifies this as the asset upload commit; its assets.zip
# LFS SHA-256 is 96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf
# (6,395,849,578 bytes), unchanged at dd2bd61b7d9a6fef1abc52d606e983b41886a149.
# Its 439,461 business files include 215 .msh meshes; asset_integrity.py pins the
# tree hash (path NUL raw-file-sha, sorted Path order), independently verified
# against the existing extracted tree, including its historical non-.msh hash.
# Update this revision, wrapper, suffixes, file count and tree hash together.
HF_ASSETS_REVISION = "b548dd25ee0401c46217ba7e3614a598e8979e48"
ASSETS_ZIP_NAME = "assets.zip"
ASSETS_ZIP_PREFIX = (
    "inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets/"
)

_integrity_spec = importlib.util.spec_from_file_location(
    "_ibrobot_libero_asset_integrity",
    Path(__file__).resolve().parents[2] / "src/benchmark/adapters/libero/benchmark_libero/asset_integrity.py",
)
_integrity = importlib.util.module_from_spec(_integrity_spec)
_integrity_spec.loader.exec_module(_integrity)


class LiberoPlusConfigError(RuntimeError):
    """Raised when the LIBERO-plus provider cannot be prepared."""


def _resolve_submodule_root() -> Path:
    override = os.environ.get("IBROBOT_LIBERO_PLUS_ROOT", "").strip()
    if override:
        root = Path(override).expanduser().resolve()
    else:
        workspace = os.environ.get("WORKSPACE", "").strip()
        root = Path(workspace).expanduser().resolve() / "libs" / "libero_plus" if workspace else None
        if root is None:
            raise LiberoPlusConfigError("WORKSPACE must be set or IBROBOT_LIBERO_PLUS_ROOT must point at the checkout")
    if not (root / "libero" / "libero" / "benchmark").is_dir():
        raise LiberoPlusConfigError(f"LIBERO-plus submodule checkout is missing: {root}")
    return root


def _validate_assets(assets_dir: Path) -> tuple[int, str]:
    try:
        return _integrity.validate_assets_tree(assets_dir, "libero-plus")
    except RuntimeError as exc:
        raise LiberoPlusConfigError(str(exc)) from exc


def _extract_assets_zip(zip_path: Path, target_dir: Path) -> tuple[int, str]:
    """Validate a fresh sibling tree before atomically publishing it, never merging."""
    if not _integrity.is_pending_assets_tree(target_dir):
        raise LiberoPlusConfigError(
            f"Refusing to merge assets.zip into populated or non-placeholder target {target_dir}; "
            "move it aside or repair it before retrying; no files were removed."
        )
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="libero-plus-extract-", dir=str(target_dir.parent)) as tmp:
        stage = Path(tmp) / "assets"
        stage.mkdir()
        with zipfile.ZipFile(zip_path) as archive:
            entries = archive.infolist()
            # Check original names before stripping any explicitly allowed wrapper.
            # ZipFile truncates NUL names and extraction must never follow links.
            for entry in entries:
                name = entry.orig_filename
                parts = name.removesuffix("/").split("/")
                kind = stat.S_IFMT(entry.external_attr >> 16)
                if (
                    not name
                    or "\0" in name
                    or "\\" in name
                    or any(PureWindowsPath(part).drive for part in parts)
                    or any(part in ("", ".", "..") for part in parts)
                    or kind not in (0, stat.S_IFREG, stat.S_IFDIR)
                    or (kind == stat.S_IFDIR and not entry.is_dir())
                    or (kind == stat.S_IFREG and entry.is_dir())
                ):
                    raise LiberoPlusConfigError(f"Unsafe ZIP member: {name!r}")
            wrapper = ()
            # The pinned upstream ZIP retains the uploader's full directory path.
            # Match an entire known wrapper for every member, never guess a prefix.
            for prefix in (ASSETS_ZIP_PREFIX, "assets/"):
                if entries and all(
                    entry.filename == prefix.removesuffix("/") or entry.filename.startswith(prefix) for entry in entries
                ):
                    wrapper = tuple(prefix.removesuffix("/").split("/"))
                    break
            seen = set()
            for entry in entries:
                parts = entry.filename.removesuffix("/").split("/")
                parts = parts[len(wrapper) :]
                if not parts:
                    if not entry.is_dir():
                        raise LiberoPlusConfigError("ZIP assets wrapper must be a directory")
                    continue
                destination = stage.joinpath(*parts)
                if destination in seen:
                    raise LiberoPlusConfigError(f"Duplicate ZIP member: {entry.filename!r}")
                seen.add(destination)
                if entry.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, destination.open("xb") as sink:
                    shutil.copyfileobj(source, sink)
        result = _validate_assets(stage)
        if not _integrity.is_pending_assets_tree(target_dir):
            raise LiberoPlusConfigError(f"Assets target changed during extraction; refusing to replace {target_dir}")
        os.replace(stage, target_dir)
    print(
        f"HINT: validated assets published at {target_dir}; ZIP preserved at {zip_path}. "
        "You may remove the ZIP manually to reclaim space.",
        flush=True,
    )
    return result


def ensure_managed_assets_exclude(config_dir: Path, provider_root: Path) -> None:
    """Keep only setup-owned assets symlinks out of the submodule dirty check."""
    package_assets = provider_root / "libero" / "libero" / "assets"
    if not package_assets.is_symlink():
        return
    managed_targets = {(config_dir / "assets").resolve()}
    source_value = os.environ.get("IBROBOT_LIBERO_PLUS_ASSETS_DIR", "").strip()
    if source_value:
        managed_targets.add(Path(source_value).expanduser().resolve())
    if package_assets.resolve() not in managed_targets:
        return

    exclude = Path(
        subprocess.run(
            ["git", "-C", str(provider_root), "rev-parse", "--git-path", "info/exclude"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.rstrip("\r\n")
    )
    if not exclude.is_absolute():
        exclude = provider_root / exclude
    exclude.parent.mkdir(parents=True, exist_ok=True)
    rule = "libero/libero/assets"
    existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    if rule not in existing.splitlines():
        with open(exclude, "a", encoding="utf-8") as stream:
            if existing and not existing.endswith("\n"):
                stream.write("\n")
            stream.write(f"{rule}\n")


def _print_manual_download_hint(config_dir: Path) -> None:
    """Print the manual asset download command (setup never downloads)."""
    print("WARN: LIBERO-plus assets are missing; setup does not download them.", flush=True)
    print("HINT: download assets.zip manually, then re-run setup to extract and validate it:", flush=True)
    print(
        f"HINT:   huggingface-cli download {HF_ASSETS_REPO_ID} --repo-type dataset "
        f"--revision {HF_ASSETS_REVISION} --include {ASSETS_ZIP_NAME} --local-dir {config_dir}",
        flush=True,
    )
    print(
        "HINT: (prepend HF_ENDPOINT=https://hf-mirror.com to use the mirror), or set "
        "IBROBOT_LIBERO_PLUS_ASSETS_DIR to an extracted asset tree;",
        flush=True,
    )
    print("HINT: evaluation stays unavailable until the assets validate.", flush=True)


def _prepare_assets(config_dir: Path, provider_root: Path) -> tuple[Path, int, str]:
    package_assets = provider_root / "libero" / "libero" / "assets"
    assets_dir = config_dir / "assets"
    source_value = os.environ.get("IBROBOT_LIBERO_PLUS_ASSETS_DIR")
    ensure_managed_assets_exclude(config_dir, provider_root)

    if package_assets.exists() or package_assets.is_symlink():
        try:
            count, tree_sha256 = _validate_assets(package_assets)
        except LiberoPlusConfigError as exc:
            if not (
                package_assets.is_symlink()
                and package_assets.resolve() == assets_dir.resolve()
                and _integrity.is_pending_assets_tree(assets_dir)
            ):
                raise LiberoPlusConfigError(
                    f"Invalid existing LIBERO-plus assets at {package_assets}: {exc}. "
                    "Move this path aside or repair it before retrying; no files were removed."
                ) from exc
        else:
            return package_assets.resolve(), count, tree_sha256

    assets_valid = False
    if assets_dir.exists() or assets_dir.is_symlink():
        try:
            count, tree_sha256 = _validate_assets(assets_dir)
        except LiberoPlusConfigError as exc:
            if not _integrity.is_pending_assets_tree(assets_dir):
                raise LiberoPlusConfigError(
                    f"Invalid existing LIBERO-plus assets at {assets_dir}: {exc}. "
                    "Move this path aside or repair it before retrying; no files were removed."
                ) from exc
        else:
            assets_valid = True

    if not assets_valid and source_value is not None:
        if not source_value.strip():
            raise LiberoPlusConfigError("IBROBOT_LIBERO_PLUS_ASSETS_DIR must name a non-empty assets directory")
        local_source = Path(source_value).expanduser().resolve()
        try:
            count, tree_sha256 = _validate_assets(local_source)
        except LiberoPlusConfigError as exc:
            raise LiberoPlusConfigError(f"Invalid IBROBOT_LIBERO_PLUS_ASSETS_DIR={local_source}: {exc}") from exc
        assets_dir.parent.mkdir(parents=True, exist_ok=True)
        if assets_dir.exists():
            assets_dir.rmdir()
        assets_dir.symlink_to(local_source, target_is_directory=True)
    elif not assets_valid:
        zip_path = config_dir / ASSETS_ZIP_NAME
        if zip_path.is_file():
            print(f"INFO: extracting {zip_path} ...", flush=True)
            count, tree_sha256 = _extract_assets_zip(zip_path, assets_dir)
        else:
            # Keep the placeholder linked so the provider cannot fall back to a
            # hidden user cache while the manual download is pending.
            assets_dir.mkdir(parents=True, exist_ok=True)
            count, tree_sha256 = 0, "pending"
            _print_manual_download_hint(config_dir)

    if package_assets.exists() or package_assets.is_symlink():
        if not (package_assets.is_symlink() and package_assets.resolve() == assets_dir.resolve()):
            raise LiberoPlusConfigError(f"refusing to replace unexpected LIBERO-plus assets path: {package_assets}")
    else:
        package_assets.symlink_to(assets_dir, target_is_directory=True)
    ensure_managed_assets_exclude(config_dir, provider_root)
    return assets_dir.resolve(), count, tree_sha256


def _generate_provider_shim(config_dir: Path, submodule_root: Path) -> Path:
    """Materialize the regular-package shim that shadows hf-libero per process.

    The submodule's top-level ``libero/`` directory has no ``__init__.py`` (a
    PEP 420 namespace package), so a plain ``sys.path`` entry can never shadow
    the regular ``libero`` package installed in site-packages: the import
    machinery always prefers a regular package over namespace portions. The
    shim is a workspace-owned regular ``libero`` package whose ``__path__`` is
    extended with the submodule's ``libero/`` directory, so ``import libero``
    resolves to the shim (winning the sys.path race) while ``libero.libero``
    and every nested module resolve inside the pinned submodule. The adapter
    prepends ``<config_dir>/provider`` to ``sys.path`` at runtime; nothing in
    the third-party checkout is modified.
    """
    provider_dir = config_dir / "provider"
    package_dir = provider_dir / "libero"
    package_dir.mkdir(parents=True, exist_ok=True)
    init_path = package_dir / "__init__.py"
    submodule_libero = (submodule_root / "libero").resolve()
    content = (
        '"""Workspace-owned LIBERO-plus provider shim.\n\n'
        "Generated by scripts/setup/libero_plus_config.py; do not edit. This\n"
        "regular package exists so the pinned libs/libero_plus submodule (whose\n"
        "top-level libero/ is a PEP 420 namespace package and therefore cannot\n"
        "shadow the installed hf-libero distribution through sys.path alone)\n"
        "is importable as ``libero.libero`` in processes that selected the\n"
        'libero-plus provider.\n"""\n\n'
        f"__path__.append({str(submodule_libero)!r})\n"
    )
    fd, tmp_path = tempfile.mkstemp(dir=package_dir, suffix=".tmp", prefix="init_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(tmp_path, init_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return provider_dir


def generate_config(workspace: str, config_dir: str | None = None) -> tuple[str, str, str, int, str]:
    """Prepare LIBERO-plus resources and atomically generate its config.yaml."""
    import yaml

    workspace_path = Path(workspace).resolve()
    config_override = config_dir or os.environ.get("IBROBOT_LIBERO_PLUS_CONFIG_DIR", "").strip()
    target_dir = (
        Path(config_override).expanduser().resolve()
        if config_override
        else workspace_path / "venv" / "ibrobot_libero_plus"
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    submodule_root = _resolve_submodule_root()
    provider_root = submodule_root / "libero" / "libero"
    assets_dir, asset_count, asset_tree_sha256 = _prepare_assets(target_dir, submodule_root)
    _generate_provider_shim(target_dir, submodule_root)

    paths = {
        "benchmark_root": str(provider_root),
        "bddl_files": str(provider_root / "bddl_files"),
        "init_states": str(provider_root / "init_files"),
        "assets": str(assets_dir),
        "datasets": str(workspace_path / "datasets" / "libero_plus"),
    }
    Path(paths["datasets"]).mkdir(parents=True, exist_ok=True)

    config_path = target_dir / "config.yaml"
    fd, tmp_path = tempfile.mkstemp(dir=target_dir, suffix=".tmp", prefix="config_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yaml.safe_dump(paths, stream, default_flow_style=False, sort_keys=True)
        os.replace(tmp_path, config_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return str(config_path), str(submodule_root), str(assets_dir), asset_count, asset_tree_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-submodule", type=Path, help="Repair only the managed assets symlink's Git exclude")
    args = parser.parse_args()
    workspace = os.path.abspath(os.environ.get("WORKSPACE", os.getcwd()))
    try:
        if args.prepare_submodule is not None:
            ensure_managed_assets_exclude(
                Path(os.environ.get("IBROBOT_LIBERO_PLUS_CONFIG_DIR", "").strip()).expanduser().resolve()
                if os.environ.get("IBROBOT_LIBERO_PLUS_CONFIG_DIR", "").strip()
                else Path(workspace) / "venv" / "ibrobot_libero_plus",
                args.prepare_submodule.resolve(),
            )
            return 0
        config_path, submodule_root, assets_dir, asset_count, asset_tree_sha256 = generate_config(workspace)
    except Exception as exc:
        print(f"ERROR: Failed to prepare LIBERO-plus: {exc}", file=sys.stderr)
        return 1
    print(f"LIBERO_CONFIG_PATH={Path(config_path).parent}")
    print(f"config_file={config_path}")
    print(f"libero_plus_root={submodule_root}")
    print(f"assets_status={'missing' if asset_tree_sha256 == 'pending' else 'ready'}")
    print(f"asset_business_file_count={asset_count}")
    print(f"asset_tree_sha256={asset_tree_sha256}")
    print(f"assets_dir={assets_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
