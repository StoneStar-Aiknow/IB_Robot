"""Tests for the RT-Infer vendor bundle payload resolution.

Review #A: the launch must find the bundle from a SOURCE checkout and from a
colcon-INSTALLED workspace, refuse an incomplete bundle, and bootstrap the
install from the bundle's own priorityctl (the prefix does not exist yet).
The ament share of this repository's own install tree provides the real
installed candidate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_config.rt_infer_bundle import bootstrap_control_script, expected_toolkit_home, resolve_vendor_bundle


def test_resolver_finds_the_source_tree_bundle(monkeypatch):
    """WORKSPACE point at this repository: the source tree candidate wins."""
    workspace = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("WORKSPACE", str(workspace))
    bundle = resolve_vendor_bundle("2026.11.05")
    assert (bundle / "manifest.json").is_file()
    assert (bundle / "bin" / "aicore_scheduler").is_file()
    assert (bundle / "scripts" / "priorityctl.py").is_file()


def test_resolver_falls_back_to_the_installed_share(monkeypatch, tmp_path):
    """Without WORKSPACE the ament share data of robot_config serves the payload."""
    monkeypatch.delenv("WORKSPACE", raising=False)
    monkeypatch.setattr(
        "robot_config.rt_infer_bundle.get_package_share_directory",
        lambda _name: str(Path(__file__).resolve().parents[3] / "install" / "robot_config" / "share" / "robot_config"),
    )
    bundle = resolve_vendor_bundle("2026.11.05")
    assert (bundle / "scripts" / "priorityctl.py").is_file(), "installed payload is complete"


def test_resolver_refuses_incomplete_bundles(monkeypatch, tmp_path):
    """A directory with a manifest but missing payload files is an error, not a partial install source."""
    monkeypatch.setenv("WORKSPACE", str(tmp_path))
    incomplete = tmp_path / "third_party" / "vendor" / "rt-infer" / "broken"
    incomplete.mkdir(parents=True)
    (incomplete / "manifest.json").write_text(
        json.dumps({"artifacts": {"scheduler": {"path": "bin/aicore_scheduler", "sha256": "x", "size_bytes": 1}}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="complete"):
        resolve_vendor_bundle("broken")


def test_bootstrap_script_comes_from_the_bundle_itself(monkeypatch, tmp_path):
    """The install command runs BEFORE the prefix exists, so the bootstrap is the bundle's own script."""
    workspace = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("WORKSPACE", str(workspace))
    bundle = resolve_vendor_bundle("2026.11.05")
    script = bootstrap_control_script(bundle)
    assert script == bundle / "scripts" / "priorityctl.py"
    assert script.is_file()


def test_bootstrap_refuses_a_bundle_without_scripts(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="bootstrap"):
        bootstrap_control_script(tmp_path)


def test_expected_toolkit_home_uses_manifest_versioned_runtime(tmp_path):
    toolkit = tmp_path / "ascend-toolkit" / "8.1.RC1"
    runtime = toolkit / "aarch64-linux/lib64/libruntime.so"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"runtime")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"compatibility": {"official_runtime_path": str(runtime)}}),
        encoding="utf-8",
    )

    assert expected_toolkit_home(bundle) == str(toolkit)
