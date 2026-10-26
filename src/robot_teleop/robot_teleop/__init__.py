"""
robot_teleop - Minimal serial-to-controller bridge for zero-latency teleoperation

This package provides a unified teleoperation interface for IB-Robot,
supporting multiple teleoperation devices (leader arms, phones, gamepads, VR controllers)
through a device abstraction layer.
"""

__version__ = "0.1.0"

# PEP 562 lazy exports (openspec change ``migrate-business-launch-builders``,
# design decision 6): the device, phone and safety-filter modules pull heavy
# dependencies (numpy among them), and this package's launch-builder
# contribution is resolved through the ``ibrobot.launch_builders`` entry point
# on every launch description generation — importing ``robot_teleop`` itself
# must stay light. Each access resolves the attribute on the source module
# (never cached), so monkeypatching the source module keeps working as a test
# seam; ``from robot_teleop import X`` works through the same hook.
_LAZY_EXPORTS = {
    "BaseTeleopDevice": "robot_teleop.base_teleop",
    "TeleopDeviceConfig": "robot_teleop.config_loader",
    "TeleoperationConfig": "robot_teleop.config_loader",
    "TeleopSafetyConfig": "robot_teleop.config_loader",
    "get_active_device_config": "robot_teleop.config_loader",
    "load_teleoperation_config": "robot_teleop.config_loader",
    "DEVICE_MAP": "robot_teleop.device_factory",
    "device_factory": "robot_teleop.device_factory",
    "LeaderArmDevice": "robot_teleop.devices.leader_arm",
    "PhoneConfig": "robot_teleop.phone.config_phone",
    "PhoneDevice": "robot_teleop.phone.phone_device",
    "SafetyFilter": "robot_teleop.safety_filter",
}

__all__ = [
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


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
