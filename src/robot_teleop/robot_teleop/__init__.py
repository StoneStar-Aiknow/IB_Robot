"""
robot_teleop - Minimal serial-to-controller bridge for zero-latency teleoperation

This package provides a unified teleoperation interface for IB-Robot,
supporting multiple teleoperation devices (leader arms, phones, gamepads, VR controllers)
through a device abstraction layer.
"""

from .base_teleop import BaseTeleopDevice
from .config_loader import (
    TeleopDeviceConfig,
    TeleoperationConfig,
    TeleopSafetyConfig,
    get_active_device_config,
    load_teleoperation_config,
)
from .device_factory import DEVICE_MAP, device_factory
from .devices.leader_arm import LeaderArmDevice
from .phone.config_phone import PhoneConfig
from .phone.phone_device import PhoneDevice
from .safety_filter import SafetyFilter

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

__version__ = "0.1.0"
