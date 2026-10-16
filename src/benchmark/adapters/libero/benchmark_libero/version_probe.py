"""Fail-closed hf-libero provider identity probe.

The Benchmark adapter consumes the provider installed by the local LeRobot
``libero`` extra. The legacy ``libs/libero`` submodule is no longer an import
provider once the migration is enabled; this probe prevents an accidental
mixed-provider environment.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from packaging.specifiers import SpecifierSet

# Setup loads this probe directly before colcon installs the adapter package.
if __package__:
    from . import asset_integrity
else:
    _integrity_spec = importlib.util.spec_from_file_location(
        "_ibrobot_libero_asset_integrity", Path(__file__).with_name("asset_integrity.py")
    )
    asset_integrity = importlib.util.module_from_spec(_integrity_spec)
    _integrity_spec.loader.exec_module(asset_integrity)


class ProviderProbeError(RuntimeError):
    """Raised when the hf-libero provider identity cannot be verified."""


EXPECTED_HF_LIBERO_SPEC = SpecifierSet(">=0.1.4,<0.2.0")
EXPECTED_ASSET_DIRS = asset_integrity.EXPECTED_ASSET_DIRS["hf-libero"]


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    """Snapshot of the verified hf-libero provider identity."""

    module_path: str
    distribution_name: str
    distribution_version: str
    provider_root: str
    assets_path: str
    apis_present: tuple[str, ...]


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _locations(module: object) -> tuple[Path, ...]:
    locations: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if isinstance(module_file, str) and module_file:
        locations.append(Path(module_file).resolve())
    for item in getattr(module, "__path__", ()):
        if isinstance(item, str) and item:
            locations.append(Path(item).resolve())
    return tuple(locations)


def _provider_distribution() -> tuple[importlib.metadata.Distribution, Path]:
    try:
        distribution = importlib.metadata.distribution("hf-libero")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ProviderProbeError("hf-libero distribution is not installed") from exc
    if distribution.version not in EXPECTED_HF_LIBERO_SPEC:
        raise ProviderProbeError(f"hf-libero version is {distribution.version!r}; expected {EXPECTED_HF_LIBERO_SPEC}")
    provider_root = Path(distribution.locate_file("libero/libero")).resolve()
    if not provider_root.is_dir():
        raise ProviderProbeError(f"hf-libero provider root does not exist: {provider_root}")
    return distribution, provider_root


def _verify_provider_import(
    distribution: importlib.metadata.Distribution, provider_root: Path
) -> tuple[str, tuple[str, ...], object, object, object]:
    package_root = Path(distribution.locate_file("libero")).resolve()
    try:
        top_module = importlib.import_module("libero")
        runtime_module = importlib.import_module("libero.libero")
        benchmark_module = importlib.import_module("libero.libero.benchmark")
        envs_module = importlib.import_module("libero.libero.envs")
    except Exception as exc:  # noqa: BLE001
        raise ProviderProbeError(f"failed to import hf-libero provider: {exc}") from exc

    all_locations = {
        name: _locations(module)
        for name, module in (
            ("libero", top_module),
            ("libero.libero", runtime_module),
            ("libero.libero.benchmark", benchmark_module),
            ("libero.libero.envs", envs_module),
        )
    }
    workspace = os.environ.get("WORKSPACE", "")
    legacy_provider = (Path(workspace).resolve() / "libs" / "libero") if workspace else None
    legacy_sys_path_entries = tuple(
        Path(entry).resolve()
        for entry in sys.path
        if entry and legacy_provider is not None and _is_within(Path(entry).resolve(), legacy_provider)
    )
    if legacy_sys_path_entries:
        raise ProviderProbeError(f"legacy libs/libero is exposed on sys.path: {legacy_sys_path_entries}")
    for name, locations in all_locations.items():
        if not locations:
            raise ProviderProbeError(f"{name} exposes no filesystem location")
        for location in locations:
            if not _is_within(location, package_root):
                raise ProviderProbeError(
                    f"{name} resolves outside hf-libero package root: {location}; expected under {package_root}"
                )
            if legacy_provider is not None and _is_within(location, legacy_provider):
                raise ProviderProbeError(f"{name} resolves from legacy libs/libero: {location}")

    try:
        importlib.metadata.distribution("libero")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise ProviderProbeError("legacy 'libero' distribution is installed alongside hf-libero")

    required = (
        (benchmark_module, "get_benchmark_dict"),
        (runtime_module, "get_libero_path"),
        (runtime_module, "get_assets_path"),
        (envs_module, "OffScreenRenderEnv"),
    )
    for module, attribute in required:
        if not hasattr(module, attribute):
            raise ProviderProbeError(f"{module.__name__} is missing required API {attribute}")
    if not isinstance(envs_module.OffScreenRenderEnv, type):
        raise ProviderProbeError("OffScreenRenderEnv is not a class")

    return (
        str(top_module.__file__ or next(iter(top_module.__path__))),
        ("get_benchmark_dict", "get_libero_path", "get_assets_path", "OffScreenRenderEnv"),
        runtime_module,
        benchmark_module,
        envs_module,
    )


def _verify_assets(assets_path: Path, provider: str, allow_missing_assets: bool) -> None:
    script = "libero_plus_config.py" if provider == "libero-plus" else "libero_config.py"
    hint = f"Setup never downloads assets; run scripts/setup/{script} for the pinned manual download command."
    pending_path = assets_path
    # hf-libero returns its package-local managed symlink, not the config path.
    config_value = os.environ.get("LIBERO_CONFIG_PATH", "")
    if assets_path.is_symlink() and config_value:
        managed_assets = Path(config_value).resolve() / "assets"
        if assets_path.resolve() == managed_assets and not managed_assets.is_symlink():
            pending_path = managed_assets
    if allow_missing_assets and asset_integrity.is_pending_assets_tree(pending_path):
        print(f"WARNING: {provider} assets are pending at {assets_path}; {hint}", flush=True)
        return
    try:
        asset_integrity.validate_assets_tree(assets_path, provider)
    except asset_integrity.AssetIntegrityError as exc:
        raise ProviderProbeError(f"Invalid {provider} assets at {assets_path}: {exc}. {hint}") from exc


def _verify_config_and_assets(runtime_module: object, provider_root: Path, allow_missing_assets: bool = False) -> Path:
    config_value = os.environ.get("LIBERO_CONFIG_PATH", "")
    if not config_value:
        raise ProviderProbeError("LIBERO_CONFIG_PATH is not set; workspace-owned hf-libero config is required")
    config_file = Path(config_value).resolve() / "config.yaml"
    if not config_file.is_file():
        raise ProviderProbeError(f"workspace-owned LIBERO config is missing: {config_file}")

    try:
        configured_root = Path(runtime_module.get_libero_path("benchmark_root")).resolve()
        assets_path = Path(runtime_module.get_assets_path()).absolute()
        bddl_path = Path(runtime_module.get_libero_path("bddl_files")).resolve()
        init_path = Path(runtime_module.get_libero_path("init_states")).resolve()
    except Exception as exc:  # noqa: BLE001
        raise ProviderProbeError(f"hf-libero resource path resolution failed: {exc}") from exc
    if configured_root != provider_root:
        raise ProviderProbeError(f"configured benchmark_root {configured_root} != provider root {provider_root}")
    if not _is_within(bddl_path, provider_root) or not bddl_path.is_dir():
        raise ProviderProbeError(f"invalid hf-libero bddl_files path: {bddl_path}")
    if not _is_within(init_path, provider_root) or not init_path.is_dir():
        raise ProviderProbeError(f"invalid hf-libero init_states path: {init_path}")
    _verify_assets(assets_path, "hf-libero", allow_missing_assets)
    return assets_path.resolve()


def probe_libero_provider(allow_missing_assets: bool = False) -> ProviderIdentity:
    """Verify and return the sole hf-libero provider identity.

    ``allow_missing_assets`` is a setup-time affordance: setup completes while
    the user-downloaded asset tree is still pending, so the setup verification
    gate probes the provider boundary without requiring assets. The
    evaluation path keeps the default fail-closed behavior.
    """
    distribution, provider_root = _provider_distribution()
    module_path, apis, runtime_module, _benchmark_module, _envs_module = _verify_provider_import(
        distribution, provider_root
    )
    assets_path = _verify_config_and_assets(runtime_module, provider_root, allow_missing_assets=allow_missing_assets)
    return ProviderIdentity(
        module_path=module_path,
        distribution_name="hf-libero",
        distribution_version=distribution.version,
        provider_root=str(provider_root),
        assets_path=str(assets_path),
        apis_present=apis,
    )


EXPECTED_LIBERO_PLUS_ASSET_DIRS = asset_integrity.EXPECTED_ASSET_DIRS["libero-plus"]
LIBERO_PLUS_SUBMODULE = "libs/libero_plus"


def resolve_libero_plus_root() -> Path:
    """Locate the pinned ``libs/libero_plus`` submodule root.

    ``IBROBOT_LIBERO_PLUS_ROOT`` overrides the walk-up for deployments that
    keep the checkout elsewhere; the default mirrors how the RoboTwin adapter
    resolves its provider root.
    """
    override = os.environ.get("IBROBOT_LIBERO_PLUS_ROOT", "").strip()
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / "libero" / "libero" / "benchmark").is_dir():
            raise ProviderProbeError(f"IBROBOT_LIBERO_PLUS_ROOT is not a LIBERO-plus checkout: {root}")
        return root
    current = Path(__file__).resolve().parent
    while current.parent != current:
        candidate = current / LIBERO_PLUS_SUBMODULE
        if (candidate / "libero" / "libero" / "benchmark").is_dir():
            return candidate
        current = current.parent
    raise ProviderProbeError(
        f"the pinned {LIBERO_PLUS_SUBMODULE} submodule was not found; run setup with --with-benchmark"
    )


def prepare_libero_plus_import() -> Path:
    """Bind this process to the LIBERO-plus provider before the first import.

    Both providers share the top-level ``libero`` package name, so exactly one
    may be imported per process. This must run before any ``libero`` import:
    it points ``LIBERO_CONFIG_PATH`` at the workspace-owned LIBERO-plus config
    and prepends the provider shim to ``sys.path``.

    The shim matters because the submodule's top-level ``libero/`` is a PEP 420
    namespace package: a bare sys.path entry can never shadow the regular
    ``libero`` distribution in site-packages. The workspace-owned shim under
    ``<config>/provider/libero`` is a regular package whose ``__path__`` also
    covers the submodule, so it wins the sys.path race and every nested module
    resolves inside the pinned checkout.
    """
    root = resolve_libero_plus_root()
    config_override = os.environ.get("IBROBOT_LIBERO_PLUS_CONFIG_DIR", "").strip()
    workspace = os.environ.get("WORKSPACE", "").strip()
    if config_override:
        config_dir = Path(config_override).expanduser().resolve()
    elif workspace:
        config_dir = Path(workspace).expanduser().resolve() / "venv" / "ibrobot_libero_plus"
    elif os.environ.get("IBROBOT_LIBERO_PLUS_ROOT", "").strip():
        raise ProviderProbeError(
            "IBROBOT_LIBERO_PLUS_ROOT requires WORKSPACE or IBROBOT_LIBERO_PLUS_CONFIG_DIR; "
            "the config directory cannot be inferred from an external checkout"
        )
    else:
        config_dir = root.parent.parent / "venv" / "ibrobot_libero_plus"
    config_file = config_dir / "config.yaml"
    if not config_file.is_file():
        raise ProviderProbeError(
            f"workspace-owned LIBERO-plus config is missing: {config_file}; run setup with --with-benchmark"
        )
    shim_package = config_dir / "provider" / "libero" / "__init__.py"
    if not shim_package.is_file():
        raise ProviderProbeError(
            f"LIBERO-plus provider shim is missing: {shim_package}; run setup with --with-benchmark"
        )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    if "libero" in sys.modules:
        existing = Path(sys.modules["libero"].__file__ or next(iter(sys.modules["libero"].__path__), "")).resolve()
        if not _is_within(existing, config_dir / "provider" / "libero"):
            raise ProviderProbeError(
                f"libero was already imported from {existing}; a single process cannot mix "
                "hf-libero and libero-plus providers"
            )
    provider_entry = str(config_dir / "provider")
    if provider_entry not in sys.path:
        sys.path.insert(0, provider_entry)
    return root


def _libero_plus_head(root: Path) -> str:
    if not (root / ".git").exists():
        raise ProviderProbeError(f"Cannot verify LIBERO-plus revision: {root} has no .git; use a versioned checkout")
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProviderProbeError(f"Cannot verify LIBERO-plus Git revision at {root}: {exc}") from exc


def probe_libero_plus_provider(allow_missing_assets: bool = False) -> ProviderIdentity:
    """Verify the LIBERO-plus submodule provider after :func:`prepare_libero_plus_import`."""
    root = resolve_libero_plus_root()
    provider_root = root / "libero" / "libero"
    try:
        top_module = importlib.import_module("libero")
        runtime_module = importlib.import_module("libero.libero")
        benchmark_module = importlib.import_module("libero.libero.benchmark")
        envs_module = importlib.import_module("libero.libero.envs")
    except Exception as exc:  # noqa: BLE001
        raise ProviderProbeError(f"failed to import libero-plus provider: {exc}") from exc

    top_file = Path(top_module.__file__ or next(iter(top_module.__path__), "")).resolve()
    runtime_file = Path(runtime_module.__file__ or next(iter(runtime_module.__path__), "")).resolve()
    if not _is_within(runtime_file, root / "libero"):
        raise ProviderProbeError(
            f"libero.libero resolves to {runtime_file} instead of the pinned submodule {root}; "
            "refuse to run LIBERO-plus evaluation against hf-libero code"
        )

    required = (
        (benchmark_module, "get_benchmark_dict"),
        (runtime_module, "get_libero_path"),
        (envs_module, "OffScreenRenderEnv"),
    )
    for module, attribute in required:
        if not hasattr(module, attribute):
            raise ProviderProbeError(f"{module.__name__} is missing required API {attribute}")

    config_value = os.environ.get("LIBERO_CONFIG_PATH", "")
    if not config_value:
        raise ProviderProbeError("LIBERO_CONFIG_PATH is not set; workspace-owned LIBERO-plus config is required")
    try:
        bddl_path = Path(runtime_module.get_libero_path("bddl_files")).resolve()
        init_path = Path(runtime_module.get_libero_path("init_states")).resolve()
        assets_path = Path(runtime_module.get_libero_path("assets")).absolute()
    except Exception as exc:  # noqa: BLE001
        raise ProviderProbeError(f"LIBERO-plus resource path resolution failed: {exc}") from exc
    if not _is_within(bddl_path, provider_root) or not bddl_path.is_dir():
        raise ProviderProbeError(f"invalid LIBERO-plus bddl_files path: {bddl_path}")
    if not _is_within(init_path, provider_root) or not init_path.is_dir():
        raise ProviderProbeError(f"invalid LIBERO-plus init_states path: {init_path}")
    _verify_assets(assets_path, "libero-plus", allow_missing_assets)

    return ProviderIdentity(
        module_path=str(top_file),
        distribution_name="libero-plus",
        distribution_version=_libero_plus_head(root),
        provider_root=str(provider_root),
        assets_path=str(assets_path.resolve()),
        apis_present=("get_benchmark_dict", "get_libero_path", "OffScreenRenderEnv"),
    )
