"""Laziness tests for robot_teleop's PEP 562 package exports.

Openspec change ``migrate-business-launch-builders``, design decision 6:
the package ``__init__`` used to import the device, phone and safety-filter
modules eagerly (pulling numpy among others), but the launch-builder entry
point resolution imports ``robot_teleop`` on every launch description
generation. The PEP 562 lazy facade keeps ``__all__``, attribute access and
``from robot_teleop import X`` behavior while making the bare package
import light; the subprocess probing and the core-budget computation are
shared through ``robot_config.launch_builders.import_budget``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import robot_teleop

EXPECTED_ALL = [
    "BaseTeleopDevice",
    "device_factory",
    "DEVICE_MAP",
    "SafetyFilter",
    "LeaderArmDevice",
    "PhoneDevice",
    "PhoneConfig",
    "TeleoperationConfig",
    "TeleopDeviceConfig",
    "TeleopSafetyConfig",
    "load_teleoperation_config",
    "get_active_device_config",
]


def test_all_is_unchanged():
    assert robot_teleop.__all__ == EXPECTED_ALL


def test_dir_lists_every_export():
    assert sorted(dir(robot_teleop)) == sorted(EXPECTED_ALL)


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute 'no_such_export'"):
        _ = robot_teleop.no_such_export


def test_from_import_resolves_the_real_source_symbols():
    from robot_teleop import DEVICE_MAP, SafetyFilter, load_teleoperation_config
    from robot_teleop.base_teleop import BaseTeleopDevice
    from robot_teleop.config_loader import TeleopDeviceConfig
    from robot_teleop.safety_filter import SafetyFilter as RealSafetyFilter

    assert SafetyFilter is RealSafetyFilter
    assert BaseTeleopDevice.__module__ == "robot_teleop.base_teleop"
    assert TeleopDeviceConfig.__module__ == "robot_teleop.config_loader"
    assert callable(load_teleoperation_config)
    assert isinstance(DEVICE_MAP, dict)


def test_attribute_access_reresolves_on_every_access(monkeypatch):
    """No caching: patching the source module keeps working through the
    facade (the same seam contract as robot_config's launch_builders)."""
    from robot_teleop import safety_filter

    marker = object()
    monkeypatch.setattr(safety_filter, "SafetyFilter", marker)
    assert robot_teleop.SafetyFilter is marker


def test_bare_package_import_stays_light():
    """A fresh interpreter importing only robot_teleop must not pull the
    device/phone/numpy dependency chain (the launch-builder motivation)."""
    code = (
        "import sys\n"
        "import robot_teleop\n"
        "loaded = {n.split('.')[0] for n in sys.modules}\n"
        "forbidden = {'numpy', 'scipy', 'websockets'}\n"
        "assert not (loaded & forbidden), f'bare robot_teleop import pulled {sorted(loaded & forbidden)}'\n"
        "assert 'robot_teleop.devices' not in sys.modules\n"
        "assert 'robot_teleop.phone' not in sys.modules\n"
        "assert 'robot_teleop.safety_filter' not in sys.modules\n"
        "print('light')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "light" in result.stdout
