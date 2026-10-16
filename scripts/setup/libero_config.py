#!/usr/bin/env python3
"""Prepare workspace-owned configuration and assets for hf-libero.

The ``hf-libero`` wheel intentionally omits the approximately 409 MiB MuJoCo
asset tree. Its runtime resolves assets through ``get_assets_path()``, which
first checks ``<site-packages>/libero/libero/assets`` and otherwise downloads
to a user-home cache. Setup makes this deterministic and non-interactive by:

1. locating the installed ``hf-libero`` distribution without importing it;
2. reusing a user-provided asset tree — setup never downloads assets;
3. validating the expected 585 business asset files when a tree is present;
4. linking the provider's package-local ``assets`` path to the workspace-owned
   cache directory (even while assets are still pending) so the runtime can
   never fall back to its hidden user-home download; and
5. atomically writing ``${WORKSPACE}/venv/ibrobot_libero/config.yaml``.

When assets are absent or empty, setup completes with ``assets_status=missing``:
it prints ``HINT:`` lines with the exact pinned manual download command and
leaves the placeholder directory in place for the user to populate. Evaluation
stays unavailable until the tree validates.

Set ``IBROBOT_LIBERO_ASSETS_DIR`` to reuse an explicitly selected local asset
tree; the same pinned hash validation applies to local and downloaded assets.
This script never imports ``libero`` before the config exists and never probes
legacy workspaces automatically.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

from packaging.specifiers import SpecifierSet

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required but not installed", file=sys.stderr)
    sys.exit(1)

HF_LIBERO_SPEC = SpecifierSet(">=0.1.4,<0.2.0")
HF_ASSETS_REPO_ID = "lerobot/libero-assets"
HF_ASSETS_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
_integrity_spec = importlib.util.spec_from_file_location(
    "_ibrobot_libero_asset_integrity",
    Path(__file__).resolve().parents[2] / "src/benchmark/adapters/libero/benchmark_libero/asset_integrity.py",
)
_integrity = importlib.util.module_from_spec(_integrity_spec)
_integrity_spec.loader.exec_module(_integrity)


def _find_provider_root() -> tuple[Path, str]:
    """Return the installed hf-libero runtime root without importing it."""
    try:
        distribution = importlib.metadata.distribution("hf-libero")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("hf-libero is not installed; install local lerobot[libero] first") from exc
    version = distribution.version
    if version not in HF_LIBERO_SPEC:
        raise RuntimeError(f"hf-libero version {version!r} does not satisfy {HF_LIBERO_SPEC}")
    provider_root = Path(distribution.locate_file("libero/libero")).resolve()
    if not provider_root.is_dir():
        raise RuntimeError(f"hf-libero runtime root does not exist: {provider_root}")
    for dirname in ("bddl_files", "init_files"):
        if not (provider_root / dirname).is_dir():
            raise RuntimeError(f"hf-libero runtime is missing {dirname}: {provider_root / dirname}")
    return provider_root, version


def _validate_assets(assets_dir: Path) -> tuple[int, str]:
    return _integrity.validate_assets_tree(assets_dir, "hf-libero")


def _print_manual_download_hint(assets_dir: Path) -> None:
    """Print the pinned manual download command (setup never downloads)."""
    print("WARN: hf-libero assets are missing; setup does not download them.", flush=True)
    print("HINT: download the pinned asset tree manually, then re-run setup to validate:", flush=True)
    print(
        "HINT:   huggingface-cli download "
        f"{HF_ASSETS_REPO_ID} --repo-type dataset --revision {HF_ASSETS_REVISION} "
        f"--local-dir {assets_dir}",
        flush=True,
    )
    print(
        "HINT: (prepend HF_ENDPOINT=https://hf-mirror.com to use the mirror); "
        "evaluation stays unavailable until the assets validate.",
        flush=True,
    )


def _prepare_assets(config_dir: Path, provider_root: Path) -> tuple[Path, int, str]:
    """Reuse user-provided assets and make hf-libero's package lookup deterministic."""
    package_assets = provider_root / "assets"
    assets_dir = config_dir / "assets"
    source_value = os.environ.get("IBROBOT_LIBERO_ASSETS_DIR")
    if package_assets.exists() or package_assets.is_symlink():
        try:
            count, tree_sha256 = _validate_assets(package_assets)
        except RuntimeError as exc:
            if not (
                package_assets.is_symlink()
                and package_assets.resolve() == assets_dir.resolve()
                and _integrity.is_pending_assets_tree(assets_dir)
            ):
                raise RuntimeError(
                    f"Invalid existing LIBERO assets at {package_assets}: {exc}. "
                    "Move this path aside or repair it before retrying; no files were removed."
                ) from exc
        else:
            return package_assets.resolve(), count, tree_sha256

    assets_valid = False
    if assets_dir.exists() or assets_dir.is_symlink():
        try:
            count, tree_sha256 = _validate_assets(assets_dir)
        except RuntimeError as exc:
            if not _integrity.is_pending_assets_tree(assets_dir):
                raise RuntimeError(
                    f"Invalid existing LIBERO assets at {assets_dir}: {exc}. "
                    "Move this path aside or repair it before retrying; no files were removed."
                ) from exc
        else:
            assets_valid = True
    if not assets_valid:
        if source_value is not None:
            if not source_value.strip():
                raise RuntimeError("IBROBOT_LIBERO_ASSETS_DIR must name a non-empty assets directory")
            local_source = Path(source_value).expanduser().resolve()
            try:
                count, tree_sha256 = _validate_assets(local_source)
            except RuntimeError as exc:
                raise RuntimeError(f"Invalid IBROBOT_LIBERO_ASSETS_DIR={local_source}: {exc}") from exc
            assets_dir.parent.mkdir(parents=True, exist_ok=True)
            if assets_dir.exists():
                assets_dir.rmdir()
            assets_dir.symlink_to(local_source, target_is_directory=True)
        else:
            # Keep a placeholder directory and always link the package-local
            # path: an unlinked package would make get_assets_path() silently
            # download to a user-home cache at evaluation time.
            assets_dir.mkdir(parents=True, exist_ok=True)
            count, tree_sha256 = 0, "pending"
            _print_manual_download_hint(assets_dir)

    if package_assets.exists() or package_assets.is_symlink():
        if not (package_assets.is_symlink() and package_assets.resolve() == assets_dir.resolve()):
            raise RuntimeError(f"refusing to replace unexpected hf-libero assets path: {package_assets}")
    else:
        package_assets.symlink_to(assets_dir, target_is_directory=True)

    if package_assets.resolve() != assets_dir.resolve():
        raise RuntimeError(f"hf-libero assets link does not resolve to workspace cache: {package_assets}")
    return assets_dir.resolve(), count, tree_sha256


def _resolve_paths(workspace: str, provider_root: Path, assets_dir: Path) -> dict[str, str]:
    datasets_dir = Path(workspace) / "datasets" / "libero"
    return {
        "benchmark_root": str(provider_root),
        "bddl_files": str(provider_root / "bddl_files"),
        "init_states": str(provider_root / "init_files"),
        "assets": str(assets_dir),
        "datasets": str(datasets_dir),
    }


def generate_config(workspace: str, config_dir: str | None = None) -> tuple[str, str, str, int, str]:
    """Prepare hf-libero resources and atomically generate config.yaml."""
    workspace_path = Path(workspace).resolve()
    target_dir = Path(config_dir).resolve() if config_dir else workspace_path / "venv" / "ibrobot_libero"
    target_dir.mkdir(parents=True, exist_ok=True)

    provider_root, version = _find_provider_root()
    assets_dir, asset_count, asset_tree_sha256 = _prepare_assets(target_dir, provider_root)
    paths = _resolve_paths(str(workspace_path), provider_root, assets_dir)
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
    return str(config_path), version, str(assets_dir), asset_count, asset_tree_sha256


def main() -> int:
    workspace = os.path.abspath(os.environ.get("WORKSPACE", os.getcwd()))
    config_dir = os.path.join(workspace, "venv", "ibrobot_libero")
    try:
        config_path, version, assets_dir, asset_count, asset_tree_sha256 = generate_config(workspace, config_dir)
        provider_root, _ = _find_provider_root()
        paths = _resolve_paths(workspace, provider_root, Path(assets_dir))
    except Exception as exc:
        print(f"ERROR: Failed to prepare hf-libero: {exc}", file=sys.stderr)
        return 1

    print(f"LIBERO_CONFIG_PATH={config_dir}")
    print(f"config_file={config_path}")
    print(f"hf_libero_version={version}")
    print(f"assets_status={'missing' if asset_tree_sha256 == 'pending' else 'ready'}")
    print(f"asset_business_file_count={asset_count}")
    print(f"asset_tree_sha256={asset_tree_sha256}")
    for key, value in sorted(paths.items()):
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
