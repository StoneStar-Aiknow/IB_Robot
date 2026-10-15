"""RoboTwin benchmark adapter plugin.

Provides ``create_plugin() -> BenchmarkPlugin`` registered under the
``ibrobot.benchmark_adapters`` entry-point group with name ``robotwin``.

Discovery/load contract:

- ``create_plugin()`` only constructs the descriptor; it does NOT import
  Sapien, RoboTwin, or any heavy backend.
- Heavy imports are deferred to ``RobotwinAdapter.configure()``.
- ``validate_environment_config`` parses the SSOT ``benchmark`` mapping
  into a :class:`BenchmarkEnvironmentConfig`, preserving unknown keys
  through ``options``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmark_robotwin.adapter import RobotwinAdapter
from benchmark_robotwin.native_reporter import create_native_reporter
from benchmark_runtime.models import BenchmarkEnvironmentConfig
from benchmark_runtime.registry import BENCHMARK_PLUGIN_API_VERSION, BenchmarkPlugin

_PLUGIN_NAME = "robotwin"


def validate_environment_config(benchmark_mapping: Mapping[str, Any]) -> BenchmarkEnvironmentConfig:
    """Parse the SSOT ``benchmark`` mapping into a BenchmarkEnvironmentConfig.

    Raises :class:`ValueError` for any malformed configuration. The validator
    preserves unknown keys through ``BenchmarkEnvironmentConfig.options`` so
    the adapter can read its frozen task/camera/image_size/control_mode
    without the registry duplicating the schema.
    """
    if not isinstance(benchmark_mapping, Mapping):
        raise ValueError(f"benchmark mapping must be a Mapping, got {type(benchmark_mapping).__name__}")

    if "type" not in benchmark_mapping:
        raise ValueError("benchmark.type is required")
    if "adapter" not in benchmark_mapping:
        raise ValueError("benchmark.adapter is required")

    benchmark_type = str(benchmark_mapping["type"])
    adapter_name = str(benchmark_mapping["adapter"])
    if not benchmark_type or not adapter_name:
        raise ValueError("benchmark.type and benchmark.adapter must be non-empty strings")

    preserved: dict[str, Any] = {}
    for key, value in benchmark_mapping.items():
        if key in ("type", "adapter", "contract"):
            continue
        preserved[key] = value

    return BenchmarkEnvironmentConfig(
        benchmark_type=benchmark_type,
        adapter_name=adapter_name,
        options=preserved,
    )


def create_adapter() -> RobotwinAdapter:
    """Factory called by the environment node after plugin load + config validation."""
    return RobotwinAdapter()


def create_plugin() -> BenchmarkPlugin:
    """Return the RoboTwin ``BenchmarkPlugin`` descriptor.

    Registered as the entry point ``robotwin = benchmark_robotwin.plugin:create_plugin``
    in the ``ibrobot.benchmark_adapters`` group.
    """
    return BenchmarkPlugin(
        name=_PLUGIN_NAME,
        api_version=BENCHMARK_PLUGIN_API_VERSION,
        validate_environment_config=validate_environment_config,
        create_adapter=create_adapter,
        create_native_reporter=create_native_reporter,
    )
