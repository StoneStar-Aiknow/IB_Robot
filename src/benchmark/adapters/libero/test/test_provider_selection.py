"""Provider selection tests for the LIBERO adapter (hf-libero vs libero-plus).

These tests exercise the pure path-resolution and fail-closed logic without
importing either provider: the workspace walk-up, the root override, the
per-process import guard, and the unknown-provider rejection.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark_libero.version_probe import (  # noqa: E402
    ProviderProbeError,
    _libero_plus_head,
    resolve_libero_plus_root,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT).strip()


def _init_repository(root: Path) -> str:
    root.mkdir(parents=True)
    _git(root, "init")
    _git(
        root,
        "-c",
        "user.name=Identity Test",
        "-c",
        "user.email=identity-test@example.invalid",
        "commit",
        "--allow-empty",
        "--no-gpg-sign",
        "-m",
        f"Provider identity fixture: {root.name}",
    )
    return _git(root, "rev-parse", "HEAD")


def test_libero_plus_head_git_directory(tmp_path: Path) -> None:
    root = tmp_path / "provider"
    expected = _init_repository(root)
    assert (root / ".git").is_dir()
    assert len(expected) == 40
    assert _libero_plus_head(root) == expected


def test_libero_plus_head_relative_submodule_gitfile(tmp_path: Path) -> None:
    source = tmp_path / "source"
    expected = _init_repository(source)
    workspace = tmp_path / "workspace"
    _init_repository(workspace)
    _git(workspace, "-c", "protocol.file.allow=always", "submodule", "add", str(source), "libs/libero_plus")
    root = workspace / "libs" / "libero_plus"
    gitfile = (root / ".git").read_text(encoding="utf-8")
    assert gitfile.startswith("gitdir: ")
    assert not Path(gitfile.removeprefix("gitdir: ").strip()).is_absolute()
    assert _libero_plus_head(root) == expected


def test_libero_plus_head_packed_refs(tmp_path: Path) -> None:
    root = tmp_path / "provider"
    expected = _init_repository(root)
    ref = _git(root, "symbolic-ref", "HEAD")
    _git(root, "pack-refs", "--all", "--prune")
    assert not (root / ".git" / ref).exists()
    assert expected in (root / ".git" / "packed-refs").read_text(encoding="utf-8")
    assert _libero_plus_head(root) == expected


def test_libero_plus_head_without_repository(tmp_path: Path) -> None:
    with pytest.raises(ProviderProbeError, match="has no .git"):
        _libero_plus_head(tmp_path)


def test_libero_plus_head_without_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "provider"
    _init_repository(root)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ProviderProbeError, match="Cannot verify"):
        _libero_plus_head(root)


def test_resolve_root_walks_up_to_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IBROBOT_LIBERO_PLUS_ROOT", raising=False)
    # The real checkout exists next to this package's parent directories.
    root = resolve_libero_plus_root()
    assert (root / "libero" / "libero" / "benchmark").is_dir()
    assert root.name == "libero_plus"


def test_unversioned_provider_does_not_record_parent_commit(tmp_path: Path) -> None:
    _init_repository(tmp_path / "workspace")
    root = tmp_path / "workspace" / "libs" / "libero_plus"
    root.mkdir(parents=True)
    with pytest.raises(ProviderProbeError, match="has no .git"):
        _libero_plus_head(root)


def test_resolve_root_override_requires_valid_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IBROBOT_LIBERO_PLUS_ROOT", str(tmp_path))
    with pytest.raises(ProviderProbeError, match="IBROBOT_LIBERO_PLUS_ROOT"):
        resolve_libero_plus_root()


def test_resolve_root_override_accepts_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = tmp_path / "libero_plus"
    (checkout / "libero" / "libero" / "benchmark").mkdir(parents=True)
    monkeypatch.setenv("IBROBOT_LIBERO_PLUS_ROOT", str(checkout))
    assert resolve_libero_plus_root() == checkout.resolve()


def test_prepare_fails_closed_without_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmark_libero.version_probe import prepare_libero_plus_import

    checkout = tmp_path / "libero_plus"
    (checkout / "libero" / "libero" / "benchmark").mkdir(parents=True)
    monkeypatch.setenv("IBROBOT_LIBERO_PLUS_ROOT", str(checkout))
    monkeypatch.setenv("IBROBOT_LIBERO_PLUS_CONFIG_DIR", str(tmp_path / "config"))
    with pytest.raises(ProviderProbeError, match="config is missing"):
        prepare_libero_plus_import()


def test_prepare_rejects_imported_hf_libero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from benchmark_libero.version_probe import prepare_libero_plus_import

    monkeypatch.delenv("IBROBOT_LIBERO_PLUS_ROOT", raising=False)
    monkeypatch.setenv("IBROBOT_LIBERO_PLUS_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yaml").write_text("benchmark_root: x\n", encoding="utf-8")
    shim = tmp_path / "provider" / "libero"
    shim.mkdir(parents=True)
    (shim / "__init__.py").write_text('"""shim"""\n', encoding="utf-8")

    class FakeModule:
        __file__ = "/site-packages/libero/__init__.py"
        __path__ = ["/site-packages/libero"]

    monkeypatch.setitem(sys.modules, "libero", FakeModule)
    with pytest.raises(ProviderProbeError, match="cannot mix"):
        prepare_libero_plus_import()


@pytest.mark.parametrize("workspace_set", [False, True])
def test_external_checkout_config_is_not_inferred_from_checkout(tmp_path, monkeypatch, workspace_set):
    from benchmark_libero.version_probe import prepare_libero_plus_import

    checkout = tmp_path / "external" / "provider"
    (checkout / "libero" / "libero" / "benchmark").mkdir(parents=True)
    monkeypatch.setenv("IBROBOT_LIBERO_PLUS_ROOT", str(checkout))
    monkeypatch.delenv("IBROBOT_LIBERO_PLUS_CONFIG_DIR", raising=False)
    monkeypatch.delenv("WORKSPACE", raising=False)
    if not workspace_set:
        with pytest.raises(ProviderProbeError, match="requires WORKSPACE or IBROBOT_LIBERO_PLUS_CONFIG_DIR"):
            prepare_libero_plus_import()
        return
    workspace = tmp_path / "workspace"
    config = workspace / "venv" / "ibrobot_libero_plus"
    shim = config / "provider" / "libero"
    shim.mkdir(parents=True)
    (config / "config.yaml").write_text("{}")
    (shim / "__init__.py").write_text("")
    monkeypatch.setenv("WORKSPACE", str(workspace))
    monkeypatch.delitem(sys.modules, "libero", raising=False)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    assert prepare_libero_plus_import() == checkout
    assert sys.path[0] == str(config / "provider")
    import os

    assert os.environ["LIBERO_CONFIG_PATH"] == str(config)


def test_unknown_provider_rejected() -> None:
    from benchmark_libero.adapter import LiberoAdapter
    from benchmark_runtime.models import BenchmarkEnvironmentConfig

    adapter = LiberoAdapter()
    options = {
        "libero_provider": "libero-pro",
        "camera_names": ["agentview_image", "robot0_eye_in_hand_image"],
        "image_size": [256, 256],
    }
    with pytest.raises(Exception, match="unknown libero_provider"):
        adapter.configure(BenchmarkEnvironmentConfig(benchmark_type="libero", adapter_name="libero", options=options))


def test_probe_tolerates_pending_assets_for_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import benchmark_libero.version_probe as vp

    provider_root = tmp_path / "libero" / "libero"
    bddl = provider_root / "bddl_files"
    init_states = provider_root / "init_files"
    assets = provider_root / "assets"
    for directory in (bddl, init_states, assets):
        directory.mkdir(parents=True)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("benchmark_root: x\n", encoding="utf-8")
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(config_dir))

    class FakeDistribution:
        version = "0.1.4"

    class FakeRuntime:
        @staticmethod
        def get_assets_path() -> str:
            return str(assets)

        @staticmethod
        def get_libero_path(key: str) -> str:
            return {
                "benchmark_root": str(provider_root),
                "bddl_files": str(bddl),
                "init_states": str(init_states),
            }[key]

    monkeypatch.setattr(vp, "_provider_distribution", lambda: (FakeDistribution(), provider_root))
    monkeypatch.setattr(
        vp, "_verify_provider_import", lambda d, r: ("fake", ("get_assets_path",), FakeRuntime, None, None)
    )

    # Default stays fail-closed on pending assets (evaluation path).
    with pytest.raises(ProviderProbeError, match="missing required directories"):
        vp.probe_libero_provider()

    # Setup-time probing tolerates the pending asset tree with a warning.
    identity = vp.probe_libero_provider(allow_missing_assets=True)
    assert identity.distribution_name == "hf-libero"
    assert identity.assets_path == str(assets)
    out = capsys.readouterr().out
    assert "assets are pending" in out
    assert "Setup never downloads assets" in out
