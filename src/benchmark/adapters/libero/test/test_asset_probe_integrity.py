"""Both provider probes enforce pinned payload integrity, not directory presence."""

import hashlib
from types import SimpleNamespace

import pytest

from benchmark_libero import asset_integrity, version_probe


@pytest.fixture(params=["hf-libero", "libero-plus"])
def provider_tree(request, tmp_path, monkeypatch):
    provider = request.param
    assets = tmp_path / "assets"
    for name in asset_integrity.EXPECTED_ASSET_DIRS[provider]:
        (assets / name).mkdir(parents=True)
    payload = b"verified scene"
    (assets / "scenes" / "scene.xml").write_bytes(payload)
    file_hash = hashlib.sha256(payload)
    digest_input = b"scenes/scene.xml\0" + (
        file_hash.digest() if provider == "libero-plus" else file_hash.hexdigest().encode() + b"\n"
    )
    monkeypatch.setitem(asset_integrity.EXPECTED_ASSET_FILE_COUNT, provider, 1)
    monkeypatch.setitem(asset_integrity.EXPECTED_ASSET_TREE_SHA256, provider, hashlib.sha256(digest_input).hexdigest())
    root = tmp_path / "provider"
    (root / "bddl_files").mkdir(parents=True)
    (root / "init_files").mkdir()
    config = tmp_path / "config"
    config.mkdir()
    (config / "config.yaml").write_text("{}")
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(config))
    paths = {
        "benchmark_root": root,
        "bddl_files": root / "bddl_files",
        "init_states": root / "init_files",
        "assets": assets,
    }
    runtime = SimpleNamespace(get_libero_path=lambda key: str(paths[key]), get_assets_path=lambda: str(assets))
    if provider == "hf-libero":
        monkeypatch.setattr(version_probe, "_provider_distribution", lambda: (SimpleNamespace(version="0.1.4"), root))
        monkeypatch.setattr(version_probe, "_verify_provider_import", lambda *args: ("module", (), runtime, None, None))
        probe = version_probe.probe_libero_provider
    else:
        # The plus package root is <checkout>/libero/libero.
        checkout = tmp_path / "checkout"
        package = checkout / "libero" / "libero"
        package.mkdir(parents=True)
        for key in ("bddl_files", "init_states"):
            paths[key] = package / key
            paths[key].mkdir()
        modules = {
            "libero": SimpleNamespace(__file__=str(checkout / "libero" / "__init__.py")),
            "libero.libero": SimpleNamespace(**vars(runtime), __file__=str(package / "__init__.py")),
            "libero.libero.benchmark": SimpleNamespace(get_benchmark_dict=lambda: {}),
            "libero.libero.envs": SimpleNamespace(OffScreenRenderEnv=type("Environment", (), {})),
        }
        monkeypatch.setattr(version_probe, "resolve_libero_plus_root", lambda: checkout)
        monkeypatch.setattr(version_probe, "_libero_plus_head", lambda root: "a" * 40)
        monkeypatch.setattr(version_probe.importlib, "import_module", lambda name: modules[name])
        probe = version_probe.probe_libero_plus_provider
    return assets, probe


def test_ready_tree_is_revalidated_after_mutation(provider_tree):
    assets, probe = provider_tree
    assert probe().assets_path == str(assets)
    (assets / "scenes" / "scene.xml").write_bytes(b"modified scene")
    for allow_missing in (False, True):
        with pytest.raises(version_probe.ProviderProbeError, match="tree SHA-256"):
            probe(allow_missing_assets=allow_missing)


def test_all_top_level_directories_do_not_prove_integrity(provider_tree):
    assets, probe = provider_tree
    (assets / "scenes" / "scene.xml").unlink()
    for allow_missing in (False, True):
        with pytest.raises(version_probe.ProviderProbeError, match="business files"):
            probe(allow_missing_assets=allow_missing)


def test_only_empty_placeholder_is_allowed_during_setup(provider_tree):
    assets, probe = provider_tree
    (assets / "scenes" / "scene.xml").unlink()
    for directory in assets.iterdir():
        directory.rmdir()
    with pytest.raises(version_probe.ProviderProbeError):
        probe()
    assert probe(allow_missing_assets=True).assets_path == str(assets)


@pytest.mark.parametrize("dangling", [False, True])
def test_user_selected_symlink_cannot_be_pending(provider_tree, dangling):
    assets, probe = provider_tree
    (assets / "scenes" / "scene.xml").unlink()
    for directory in assets.iterdir():
        directory.rmdir()
    assets.rmdir()
    target = assets.parent / "external"
    if not dangling:
        target.mkdir()
    assets.symlink_to(target, target_is_directory=True)
    with pytest.raises(version_probe.ProviderProbeError):
        probe(allow_missing_assets=True)


@pytest.mark.parametrize("provider", ["hf-libero", "libero-plus"])
def test_managed_package_symlink_can_point_to_empty_placeholder(tmp_path, monkeypatch, provider):
    config = tmp_path / "config"
    managed = config / "assets"
    managed.mkdir(parents=True)
    package_assets = tmp_path / "package-assets"
    package_assets.symlink_to(managed, target_is_directory=True)
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(config))
    version_probe._verify_assets(package_assets, provider, allow_missing_assets=True)
