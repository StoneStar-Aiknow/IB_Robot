"""Typed configuration for the RT-Infer Ascend software scheduler integration.

RT-Infer (``third_party/vendor/rt-infer``) provides software-controlled AI-core
scheduling on the Ascend 310P1. This module is the single source of truth for
how a robot opts into it and which priority/CPU each inference role runs at,
so the numbers the candidate10 validation measured live in configuration
instead of being scattered across launch code.

Scope guard: the whole feature is inert unless a robot YAML declares
``rt_infer.enable: true``. Only the oee_summit_2026 X2 live entries do that, so
CPU/CUDA/RKNN/mock deployments never load any of this.

Parsing is deliberately PURE: no filesystem access, no host probes, no
installed-manifest reads. Everything environment-dependent (bundle install,
compatibility checks, readiness) happens in the launch owner
(``launch_builders/rt_infer_scheduler.py``) at runtime on the robot, so the
same merged YAML parses on a workstation, in tests, and on the board.

Fail-closed rule: when ``enable: true`` but prerequisites are missing, the
launch owner raises a precise, actionable error rather than silently
continuing on the native path. The explicit diagnostic escape hatch is
``enable: false`` in the YAML (native), not an ignored error.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

_RT_INFER_PATH = "rt_infer"

# The validated candidate10 role profile. Priorities are RT-Infer's 0-7 scale
# (0 highest). The launch wraps each Ascend inference *process* in one
# priorityctl client, so a role maps to a process: PEAR gets its own
# model-service process (priority 0), YOLOX its own (priority 3), and the
# TTS and VL chains each run inside one process at priority 3.
_DEFAULT_ROLE_PRIORITIES = {
    "pear": 0,
    "yolox": 3,
    "tts": 3,
    "vl": 3,
}
_DEFAULT_ROLE_CPUS = {
    "pear": 13,
    "yolox": 10,
    "tts": 12,
    "vl": 11,
}
_ROLES = tuple(_DEFAULT_ROLE_PRIORITIES)
_PERIODIC_BUDGET_NS = 38_000_000
_PEAR_PERIOD_HZ = 18

_ALLOWED_FIELDS = frozenset(
    {
        "enable",
        "bundle_version",
        "bundle_dir",
        "scheduler_cpu",
        "role_priorities",
        "role_cpus",
        "readiness_timeout_sec",
        "verify_timeout_sec",
        "stop_timeout_sec",
        "client_poll_interval_sec",
        "pear_periodic",
    }
)
_PERIODIC_FIELDS = frozenset({"period_hz", "budget_ns", "control_path"})
_ROLE_KEYS = frozenset(_ROLES)


def rt_infer_role_for_manifest(validated):
    if validated is None:
        return None
    profile = validated.deployment.runtime_profile
    if profile is None or profile.backend != "ascend":
        return None
    return {"pear_parameter_network": "pear", "yolox_person": "yolox"}.get(validated.manifest.model.model_type)


def deployment_uses_ascend(deployment) -> bool:
    """Inspect validated runtime profiles, never deployment display names."""
    profiles = list((deployment.role_runtime_profiles or {}).values())
    if deployment.runtime_profile is not None:
        profiles.append(deployment.runtime_profile)
    return any(
        profile.backend == "ascend"
        or (profile.backend == "torch" and str(getattr(profile.backend_profile, "device", "")).split(":")[0] == "npu")
        for profile in profiles
    )


class RtInferConfigError(ValueError):
    """RT-Infer configuration is invalid for this deployment."""


@dataclass(frozen=True)
class PearPeriodicConfig:
    """The PEAR periodic admission mapping (candidate10's 18 Hz / 38 ms)."""

    period_hz: int = _PEAR_PERIOD_HZ
    budget_ns: int = _PERIODIC_BUDGET_NS
    control_path: str = "/run/ibrobot-rtinfer/periodic-control.bin"


@dataclass(frozen=True)
class RtInferConfig:
    """One robot's RT-Infer deployment profile.

    ``enable`` False means the native hardware scheduler stays in charge and
    nothing in the launch path references RT-Infer.

    The installation prefix is canonical and derived, never hand-picked:
    ``/opt/rt-infer/<bundle_version>``. The launch owner installs the
    repository's vendor bundle there through the vendor's own guarded,
    idempotent ``install`` command, so a normal setup+build+launch prepares
    the environment without any manual CLI step.
    """

    enable: bool = False
    bundle_version: str = ""
    bundle_dir: str = ""  # explicit override; default resolves from WORKSPACE
    scheduler_cpu: int = 15
    role_priorities: Mapping[str, int] = field(default_factory=lambda: dict(_DEFAULT_ROLE_PRIORITIES))
    role_cpus: Mapping[str, int] = field(default_factory=lambda: dict(_DEFAULT_ROLE_CPUS))
    readiness_timeout_sec: float = 20.0
    verify_timeout_sec: float = 60.0
    stop_timeout_sec: float = 30.0
    client_poll_interval_sec: float = 1.0
    pear_periodic: PearPeriodicConfig | None = None
    # Set in the launch snapshot, never in robot YAML.
    launch_token: str = ""

    @property
    def prefix(self) -> str:
        return f"/opt/rt-infer/{self.bundle_version}"

    def resolve_bundle_dir(self, workspace: str) -> str:
        """Where the vendor bundle for ``bundle_version`` lives in this tree."""
        root = self.bundle_dir or f"{workspace}/third_party/vendor/rt-infer/{self.bundle_version}"
        return root

    @property
    def control_tool(self) -> str:
        return f"{self.prefix}/scripts/priorityctl.py"

    def priority_for(self, role: str) -> int:
        self._require_role(role)
        return int(self.role_priorities[role])

    def cpu_for(self, role: str) -> int:
        self._require_role(role)
        return int(self.role_cpus[role])

    def _require_role(self, role: str) -> None:
        if role not in _ROLE_KEYS:
            raise RtInferConfigError(f"rt_infer role {role!r} is not one of the supported roles {sorted(_ROLE_KEYS)}")


def parse_rt_infer_config(robot_config: Mapping[str, Any]) -> RtInferConfig:
    """Parse and validate ``robot.rt_infer``; absent block means disabled.

    Validation runs even when disabled so a typo cannot hide behind
    ``enable: false`` and surprise the next enable. Pure: no filesystem or
    host access of any kind.
    """
    raw = robot_config.get(_RT_INFER_PATH)
    if raw is None:
        return RtInferConfig()
    if not isinstance(raw, Mapping):
        raise RtInferConfigError("robot.rt_infer must be a mapping")
    unknown = sorted(set(raw) - _ALLOWED_FIELDS)
    if unknown:
        raise RtInferConfigError(f"rt_infer contains unsupported fields: {unknown}")

    enable = raw.get("enable", False)
    if not isinstance(enable, bool):
        raise RtInferConfigError("rt_infer.enable must be a boolean")

    periodic_raw = raw.get("pear_periodic")
    periodic = None
    if periodic_raw is not None:
        if not isinstance(periodic_raw, Mapping):
            raise RtInferConfigError("rt_infer.pear_periodic must be a mapping")
        unknown_periodic = sorted(set(periodic_raw) - _PERIODIC_FIELDS)
        if unknown_periodic:
            raise RtInferConfigError(f"rt_infer.pear_periodic contains unsupported fields: {unknown_periodic}")
        periodic = PearPeriodicConfig(
            period_hz=_positive_int(periodic_raw.get("period_hz", _PEAR_PERIOD_HZ), "rt_infer.pear_periodic.period_hz"),
            budget_ns=_positive_int(
                periodic_raw.get("budget_ns", _PERIODIC_BUDGET_NS), "rt_infer.pear_periodic.budget_ns"
            ),
            control_path=_non_empty_string(
                periodic_raw.get("control_path", PearPeriodicConfig.control_path),
                "rt_infer.pear_periodic.control_path",
            ),
        )
        period_ns = (1_000_000_000 + periodic.period_hz - 1) // periodic.period_hz
        if periodic.budget_ns >= period_ns:
            raise RtInferConfigError("rt_infer.pear_periodic.budget_ns must be smaller than the period")
        control = PurePosixPath(periodic.control_path)
        if (
            not control.is_absolute()
            or ".." in control.parts
            or str(control) != periodic.control_path
            or control.parent != PurePosixPath("/run/ibrobot-rtinfer")
            or control.name != "periodic-control.bin"
        ):
            raise RtInferConfigError(
                "rt_infer.pear_periodic.control_path must be /run/ibrobot-rtinfer/periodic-control.bin"
            )

    priorities = _role_map(
        raw.get("role_priorities", {}), _ROLE_KEYS, _DEFAULT_ROLE_PRIORITIES, "rt_infer.role_priorities"
    )
    for role, value in priorities.items():
        if not 0 <= value <= 7:
            raise RtInferConfigError(f"rt_infer.role_priorities.{role} must be within RT-Infer's 0-7 scale")
    # Only inference roles are pinned; the scheduler CPU has its own field and
    # must not appear twice under different names.
    cpus = _role_map(raw.get("role_cpus", {}), _ROLE_KEYS, _DEFAULT_ROLE_CPUS, "rt_infer.role_cpus")

    bundle_version = raw.get("bundle_version", "")
    if not isinstance(bundle_version, str) or (
        bundle_version and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", bundle_version)
    ):
        raise RtInferConfigError("rt_infer.bundle_version must be a single version directory name")
    bundle_dir = raw.get("bundle_dir", "")
    if not isinstance(bundle_dir, str) or (
        bundle_dir and (not PurePosixPath(bundle_dir).is_absolute() or ".." in PurePosixPath(bundle_dir).parts)
    ):
        raise RtInferConfigError("rt_infer.bundle_dir must be an absolute directory path")

    config = RtInferConfig(
        enable=enable,
        bundle_version=bundle_version,
        bundle_dir=bundle_dir,
        scheduler_cpu=_require_cpu(raw.get("scheduler_cpu", 15), "rt_infer.scheduler_cpu"),
        role_priorities=priorities,
        role_cpus=cpus,
        readiness_timeout_sec=_positive_float(raw.get("readiness_timeout_sec", 20.0), "rt_infer.readiness_timeout_sec"),
        verify_timeout_sec=_positive_float(raw.get("verify_timeout_sec", 60.0), "rt_infer.verify_timeout_sec"),
        stop_timeout_sec=_positive_float(raw.get("stop_timeout_sec", 30.0), "rt_infer.stop_timeout_sec"),
        client_poll_interval_sec=_positive_float(
            raw.get("client_poll_interval_sec", 1.0), "rt_infer.client_poll_interval_sec"
        ),
        pear_periodic=periodic,
    )
    if enable and not config.bundle_version:
        raise RtInferConfigError(
            "rt_infer.enable=true requires rt_infer.bundle_version (the vendor bundle directory name under "
            "third_party/vendor/rt-infer/, e.g. '2026.11.05'); the launch owner installs it to the canonical "
            "prefix /opt/rt-infer/<bundle_version>"
        )
    return config


def _role_map(value: Any, allowed: frozenset[str], defaults: Mapping[str, int], path: str) -> dict[str, int]:
    """Merge a partial per-role override over the full default profile.

    The result always covers every role: a partial override replaces only the
    roles it names, and a missing role must never become a KeyError later.
    """
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise RtInferConfigError(f"{path} must be a mapping of role to integer")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise RtInferConfigError(f"{path} contains unknown roles: {unknown}; supported: {sorted(allowed)}")
    merged = {role: int(defaults[role]) for role in allowed}
    for role, entry in value.items():
        merged[role] = _require_cpu(entry, f"{path}.{role}")
    return merged


def _require_cpu(value: Any, path: str) -> int:
    if type(value) is not int or value < 0:
        raise RtInferConfigError(f"{path} must be a non-negative integer CPU id")
    return value


def _positive_int(value: Any, path: str) -> int:
    if type(value) is not int or value <= 0:
        raise RtInferConfigError(f"{path} must be a positive integer")
    return value


def _positive_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RtInferConfigError(f"{path} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise RtInferConfigError(f"{path} must be a positive finite number")
    return number


def _non_empty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RtInferConfigError(f"{path} must be a non-empty string")
    return value


__all__ = [
    "PearPeriodicConfig",
    "RtInferConfig",
    "RtInferConfigError",
    "parse_rt_infer_config",
]
