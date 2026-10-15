"""RoboTwin benchmark adapter package.

Public API:

- :class:`RobotwinAdapter` — concrete ``BenchmarkAdapter`` implementation.
- :func:`create_plugin` — entry-point factory returning a ``BenchmarkPlugin``.
- :func:`validate_environment_config` — SSOT config validator.
- :func:`create_adapter` — zero-arg adapter factory.
- :func:`create_native_reporter` — zero-arg native reporter factory.
- :func:`convert_observation` — RoboTwin raw obs → policy contract dict.
- :func:`validate_robotwin_action_dict` — action dict validator.

Heavy Sapien/RoboTwin imports are deferred to the first ``reset()``;
discovery is exclusively via the ``ibrobot.benchmark_adapters`` entry-point group.
This package must NOT be imported by ``benchmark_runtime`` or ``robot_config``.
"""

from __future__ import annotations

from benchmark_robotwin.action_codec import validate_robotwin_action_dict
from benchmark_robotwin.adapter import RobotwinAdapter
from benchmark_robotwin.native_reporter import RobotwinNativeReporter, create_native_reporter
from benchmark_robotwin.observation_codec import (
    POLICY_IMAGE_HEAD_KEY,
    POLICY_IMAGE_LEFT_KEY,
    POLICY_IMAGE_RIGHT_KEY,
    POLICY_STATE_KEY,
    ObservationCodecError,
    convert_observation,
)
from benchmark_robotwin.plugin import (
    create_adapter,
    create_plugin,
    validate_environment_config,
)
from benchmark_robotwin.version_probe import (
    ProviderIdentity,
    ProviderProbeError,
    default_robotwin_root,
    probe_robotwin_provider,
)

__all__ = [
    "POLICY_IMAGE_HEAD_KEY",
    "POLICY_IMAGE_LEFT_KEY",
    "POLICY_IMAGE_RIGHT_KEY",
    "POLICY_STATE_KEY",
    "ObservationCodecError",
    "ProviderIdentity",
    "ProviderProbeError",
    "RobotwinAdapter",
    "RobotwinNativeReporter",
    "convert_observation",
    "create_adapter",
    "create_native_reporter",
    "create_plugin",
    "default_robotwin_root",
    "probe_robotwin_provider",
    "validate_environment_config",
    "validate_robotwin_action_dict",
]
