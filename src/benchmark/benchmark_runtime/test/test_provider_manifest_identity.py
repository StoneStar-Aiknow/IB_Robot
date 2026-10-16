"""Provider snapshots are adapter-owned, not inferred by the evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_libero import adapter as libero_adapter
from benchmark_libero.version_probe import ProviderIdentity
from benchmark_runtime import evaluator_node
from benchmark_runtime.models import BenchmarkEnvironmentConfig


@pytest.mark.parametrize("provider", ["hf-libero", "libero-plus"])
def test_libero_identity_serializes_configure_probe_without_native_reset(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    version = "0.1.4" if provider == "hf-libero" else "a" * 40
    probed = ProviderIdentity(
        module_path="/actual/provider/libero/__init__.py",
        distribution_name=provider,
        distribution_version=version,
        provider_root="/actual/provider/libero/libero",
        assets_path="/actual/assets",
        apis_present=("get_benchmark_dict", "get_libero_path", "OffScreenRenderEnv"),
    )

    def probe():
        calls.append("probe")
        return probed

    monkeypatch.setattr(libero_adapter, "probe_libero_provider", probe)
    monkeypatch.setattr(libero_adapter, "probe_libero_plus_provider", probe)
    monkeypatch.setattr(libero_adapter, "prepare_libero_plus_import", lambda: calls.append("prepare"))
    monkeypatch.setenv("MUJOCO_GL", "egl")
    adapter = libero_adapter.LiberoAdapter()
    assert adapter._provider_identity is None
    assert adapter._libero_provider == "hf-libero"
    assert adapter.provider_identity() is None
    try:
        adapter.configure(BenchmarkEnvironmentConfig("libero", "libero", options={"libero_provider": provider}))
        expected = {
            "name": provider,
            "version": version,
            "module_path": probed.module_path,
            "provider_root": probed.provider_root,
            "assets_path": probed.assets_path,
            "apis_present": list(probed.apis_present),
        }
        assert adapter.provider_identity() == expected
        # Returned containers must not let a caller mutate the stored probe.
        adapter.provider_identity()["apis_present"].clear()
        assert adapter.provider_identity() == expected
        assert calls == (["prepare", "probe"] if provider == "libero-plus" else ["probe"])
        assert adapter._env is None
        assert adapter.active_task_identity is None
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "provider_identity",
    [
        None,
        {
            "name": "another-provider",
            "version": "actual-revision",
            "module_path": "/remote/module.py",
            "provider_root": "/remote/provider",
            "assets_path": "/remote/assets",
            "apis_present": ["reset", "step"],
            "extension": {"verified": True},
        },
    ],
)
def test_runtime_identity_preserves_provider_and_package_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider_identity: dict | None
) -> None:
    config = tmp_path / "robot.json"
    config.write_text(
        json.dumps(
            {
                "robot": {
                    "benchmark": {
                        "libero_provider": "libero-plus",
                        "identity_packages": ["hf-libero", "torch", "hf-libero"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("IBROBOT_LIBERO_PLUS_ROOT", "/not/the/executing/provider")
    packages = []

    def package_identity(name):
        packages.append(name)
        return {"version": f"{name}-version", "path": f"/packages/{name}"}

    monkeypatch.setattr(evaluator_node, "_package_identity", package_identity)
    monkeypatch.setattr(evaluator_node, "_git_identity", lambda root: {"commit": "workspace", "submodules": []})
    identity = evaluator_node._runtime_manifest_identity(str(config), provider_identity=provider_identity)
    assert identity["provider"] == provider_identity
    assert packages == ["torch", "torchvision", "hf-libero"]
    assert identity["packages"] == {
        name: {"version": f"{name}-version", "path": f"/packages/{name}"} for name in packages
    }
    assert "libero_provider" not in identity["packages"]
    assert evaluator_node._runtime_manifest_identity(str(config))["provider"] is None
