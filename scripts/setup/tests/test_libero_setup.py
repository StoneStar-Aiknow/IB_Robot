"""Setup regressions for the hf-libero config script without assets or downloads."""

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP_DIR = REPO_ROOT / "scripts" / "setup"


@pytest.fixture
def config_module(monkeypatch):
    spec = importlib.util.spec_from_file_location("libero_config", SETUP_DIR / "libero_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("IBROBOT_LIBERO_ASSETS_DIR", raising=False)
    return module


@pytest.fixture
def provider_root(tmp_path):
    root = tmp_path / "libero" / "libero"
    root.mkdir(parents=True)
    return root


def test_prepare_assets_pending_manual_download(config_module, provider_root, tmp_path, monkeypatch, capsys):
    download = Mock(side_effect=AssertionError("assets must never be downloaded by setup"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    config_dir = tmp_path / "config"
    assets = config_dir / "assets"

    # Two runs: the pending state must be idempotent and never download.
    for _ in range(2):
        assert config_module._prepare_assets(config_dir, provider_root) == (assets.resolve(), 0, "pending")
    package_assets = provider_root / "assets"
    assert assets.is_dir()
    assert package_assets.is_symlink()
    assert package_assets.resolve() == assets
    out = capsys.readouterr().out
    assert "WARN: hf-libero assets are missing; setup does not download them." in out
    assert f"HINT:   huggingface-cli download {config_module.HF_ASSETS_REPO_ID}" in out
    assert config_module.HF_ASSETS_REVISION in out
    download.assert_not_called()


@pytest.mark.parametrize("managed_link", [False, True])
def test_prepare_assets_rejects_and_keeps_partial_manual_download(config_module, provider_root, tmp_path, managed_link):
    config_dir = tmp_path / "config"
    assets = config_dir / "assets"
    (assets / "textures").mkdir(parents=True)
    (assets / "textures" / "partial.png").write_bytes(b"png")

    if managed_link:
        (provider_root / "assets").symlink_to(assets, target_is_directory=True)
    with pytest.raises(RuntimeError, match="Move this path aside or repair it"):
        config_module._prepare_assets(config_dir, provider_root)
    assert (assets / "textures" / "partial.png").read_bytes() == b"png"
    assert (provider_root / "assets").is_symlink() == managed_link


@pytest.mark.parametrize("dangling", [False, True])
def test_prepare_assets_preserves_stale_managed_symlink(config_module, provider_root, tmp_path, dangling):
    config_dir = tmp_path / "config"
    assets = config_dir / "assets"
    assets.parent.mkdir(parents=True)
    old_tree = tmp_path / "old-tree"
    if not dangling:
        old_tree.mkdir()
    assets.symlink_to(old_tree, target_is_directory=True)

    with pytest.raises(RuntimeError, match="no files were removed"):
        config_module._prepare_assets(config_dir, provider_root)
    assert assets.is_symlink()
    assert assets.resolve() == old_tree


def test_prepare_assets_env_var_invalid_fails(config_module, provider_root, tmp_path, monkeypatch):
    bogus = tmp_path / "bogus-assets"
    bogus.mkdir()
    monkeypatch.setenv("IBROBOT_LIBERO_ASSETS_DIR", str(bogus))
    with pytest.raises(RuntimeError, match="Invalid IBROBOT_LIBERO_ASSETS_DIR"):
        config_module._prepare_assets(tmp_path / "config", provider_root)


def test_generate_config_pending_writes_config(config_module, provider_root, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config_module, "_find_provider_root", lambda: (provider_root, "0.1.4"))
    workspace = tmp_path / "workspace"
    config_path, version, assets_dir, count, tree_sha256 = config_module.generate_config(str(workspace))

    assert version == "0.1.4"
    assert count == 0
    assert tree_sha256 == "pending"
    assert Path(config_path) == workspace / "venv" / "ibrobot_libero" / "config.yaml"
    assert (workspace / "datasets" / "libero").is_dir()
    assert (workspace / "venv" / "ibrobot_libero" / "assets").is_dir()
    data = config_module.yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    assert data["assets"] == assets_dir
    assert data["benchmark_root"] == str(provider_root)
    assert "HINT:   huggingface-cli download" in capsys.readouterr().out


def test_main_prints_assets_status(config_module, provider_root, tmp_path, monkeypatch, capsys):
    workspace = tmp_path / "workspace"
    monkeypatch.setenv("WORKSPACE", str(workspace))
    monkeypatch.setattr(config_module, "_find_provider_root", lambda: (provider_root, "0.1.4"))
    monkeypatch.setattr(sys, "argv", ["libero_config.py"])

    assert config_module.main() == 0
    out = capsys.readouterr().out
    assert "assets_status=missing" in out
    assert "asset_tree_sha256=pending" in out


@pytest.mark.parametrize("provider", ["hf-libero", "libero-plus"])
def test_shared_integrity_rehashes_payload(config_module, tmp_path, monkeypatch, provider):
    integrity = config_module._integrity
    root = tmp_path / ".cache" / "assets"
    for name in integrity.EXPECTED_ASSET_DIRS[provider]:
        (root / name).mkdir(parents=True)
    payload = root / "scenes" / "scene.xml"
    payload.write_bytes(b"original")
    digest = hashlib.sha256()
    digest.update(b"scenes/scene.xml\0")
    file_digest = hashlib.sha256(b"original")
    digest.update(file_digest.hexdigest().encode() + b"\n" if provider == "hf-libero" else file_digest.digest())
    expected = digest.hexdigest()
    monkeypatch.setitem(integrity.EXPECTED_ASSET_FILE_COUNT, provider, 1)
    monkeypatch.setitem(integrity.EXPECTED_ASSET_TREE_SHA256, provider, expected)
    (root / ".cache").mkdir()
    (root / ".cache" / "metadata").touch()
    (root / ".gitattributes").touch()
    assert integrity.validate_assets_tree(root, provider) == (1, expected)
    payload.write_bytes(b"modified")
    with pytest.raises(integrity.AssetIntegrityError, match="SHA-256"):
        integrity.validate_assets_tree(root, provider)


@pytest.mark.parametrize("provider", ["hf-libero", "libero-plus"])
def test_shared_integrity_missing_dirs_and_wrong_count(config_module, tmp_path, provider):
    integrity = config_module._integrity
    with pytest.raises(integrity.AssetIntegrityError, match="missing required directories"):
        integrity.validate_assets_tree(tmp_path, provider)
    for name in integrity.EXPECTED_ASSET_DIRS[provider]:
        (tmp_path / name).mkdir()
    with pytest.raises(integrity.AssetIntegrityError, match="contain 0 business files"):
        integrity.validate_assets_tree(tmp_path, provider)


def test_shared_integrity_loads_without_package_or_dependencies(config_module):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import importlib.util, sys; "
            "spec = importlib.util.spec_from_file_location('integrity', sys.argv[1]); "
            "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "assert 'benchmark_libero' not in sys.modules; "
            "assert 'libero' not in sys.modules; assert 'numpy' not in sys.modules",
            str(Path(config_module._integrity.__file__)),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_override_replaces_only_empty_placeholder(config_module, provider_root, tmp_path, monkeypatch):
    config = tmp_path / "config"
    assets = config / "assets"
    assets.mkdir(parents=True)
    (provider_root / "assets").symlink_to(assets, target_is_directory=True)
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setenv("IBROBOT_LIBERO_ASSETS_DIR", str(source))

    def validate(path):
        if path.resolve() == source:
            return 1, "sha256"
        raise RuntimeError("placeholder")

    monkeypatch.setattr(config_module, "_validate_assets", validate)
    assert config_module._prepare_assets(config, provider_root) == (source, 1, "sha256")
    assert assets.is_symlink()
