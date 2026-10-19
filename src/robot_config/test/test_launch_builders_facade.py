"""Behavioral tests for the launch_builders compatibility lazy facade.

The facade must keep every legacy public symbol importable while
importing NO builder submodule on package import: importing the
registration framework (``robot_config.launch_registry.registry``) must
not pull benchmark (and through it tensormsg) or any other business
dependency. The no-submodule-import assertion runs in a fresh subprocess
so it cannot be polluted by this pytest process importing business
modules for other test files.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

import robot_config.launch_builders as launch_builders_pkg


def test_package_import_loads_no_builder_submodules():
    """A fresh interpreter must be able to import the package (and the
    framework registry/launch_options modules) without pulling any
    business builder submodule — benchmark (and through it tensormsg),
    control, perception, execution and friends must stay unloaded.

    The empty-submodule assertion applies to the bare package import;
    importing the framework registry/launch_options legitimately loads
    those two modules plus tracing (launch_options reads the default
    trace session name), so afterwards we assert the loaded set contains
    ONLY that allowed infrastructure, never a business builder.
    """
    code = (
        "import sys\n"
        "import robot_config.launch_builders\n"
        "bare = [n for n in sys.modules if n.startswith('robot_config.launch_builders.')]\n"
        "assert bare == [], f'bare package import loaded submodules: {bare}'\n"
        "import robot_config.launch_registry.registry\n"
        "import robot_config.launch_registry.launch_options\n"
        "ALLOWED = {\n"
        "    'robot_config.launch_registry.registry',\n"
        "    'robot_config.launch_registry.launch_options',\n"
        "    'robot_config.launch_builders.tracing',\n"
        "}\n"
        "loaded = {n for n in sys.modules if n.startswith('robot_config.launch_builders.')}\n"
        "assert loaded <= ALLOWED, f'unexpected builder imports: {sorted(loaded - ALLOWED)}'\n"
        "assert 'tensormsg' not in sys.modules, 'tensormsg must not load via the facade'\n"
        "print('facade-clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "facade-clean" in result.stdout


def test_legacy_attribute_resolves_from_source_submodule():
    """Facade attributes resolve to the real source-module callables."""
    hardware_mock = importlib.import_module("robot_config.launch_builders.hardware_mock")
    assert launch_builders_pkg.mock_mode_skips_subsystem is hardware_mock.mock_mode_skips_subsystem


def test_legacy_attribute_reresolves_on_every_access(monkeypatch):
    """The facade must not cache resolved callables, so patching the
    source module keeps working as a test seam."""
    hardware_mock = importlib.import_module("robot_config.launch_builders.hardware_mock")
    original = hardware_mock.mock_mode_skips_subsystem
    marker = lambda *args, **kwargs: "patched"  # noqa: E731
    monkeypatch.setattr(hardware_mock, "mock_mode_skips_subsystem", marker)
    assert launch_builders_pkg.mock_mode_skips_subsystem is marker
    monkeypatch.setattr(hardware_mock, "mock_mode_skips_subsystem", original)
    assert launch_builders_pkg.mock_mode_skips_subsystem is original


def test_from_import_submodule_fallback_keeps_working():
    """``from robot_config.launch_builders import <submodule>`` (the
    pattern used by embodied_bringup and package tests) resolves through
    the standard from-import fallback."""
    pkg = __import__("robot_config.launch_builders", fromlist=["tracing"])
    tracing = importlib.import_module("robot_config.launch_builders.tracing")
    assert pkg.tracing is tracing


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute 'generate_nonexistent'"):
        _ = launch_builders_pkg.generate_nonexistent


def test_dir_exposes_every_legacy_public_symbol():
    """``dir()`` and the exported table must cover the whole legacy
    ``__all__``; new registry entries are NOT added to the facade."""
    public = set(dir(launch_builders_pkg))
    assert set(launch_builders_pkg.__all__) <= public
    assert "generate_benchmark_nodes" in public
    # A registry-era symbol that must not leak into the legacy surface:
    assert "BuilderSpec" not in public
