"""Pure-Python benchmark endpoint resolver (benchmark endpoint resolver).

Given the normalized robot SSOT ``benchmark`` mapping this module derives the
deterministic ROS namespace and ``reset`` / ``step`` / ``status`` names that
every benchmark instance (LIBERO today, Meta-World / future benchmarks later)
must use. The resolver is intentionally pure-Python:

- imports only the standard library;
- never imports ``rclpy``, ``launch`` or ROS messages;
- never imports ``benchmark_runtime`` or LIBERO / MuJoCo;
- does not read files, environment variables or the ROS graph;
- produces no network or process side-effects.

It does not create ROS clients, services, executors, registries, suite/task
parsers, episode/evaluator/report components or any production launch wiring.
Those are explicitly out of benchmark endpoint resolver scope and belong to benchmark step transport and later Work
Packages.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MAX_BENCHMARK_LANES = 20


class BenchmarkEndpointError(ValueError):
    """Raised when benchmark endpoint identity cannot be resolved."""


@dataclass(frozen=True, slots=True)
class BenchmarkEndpoints:
    """Endpoint identity for a single benchmark instance.

    The model only describes ROS namespace/service/topic identity. It never
    carries suite, task, checkpoint, episode, step, lane, reward or success
    information -- those belong to evaluator/RunPlanner layers in later Work
    Packages.
    """

    benchmark_type: str
    adapter_name: str
    instance_id: str
    namespace: str
    reset_service: str
    step_service: str
    lane_id: str = ""

    def scope_endpoint(self, endpoint: str | None) -> str:
        """Scope a transport/observation endpoint to this lane, preserving serial names."""
        if not endpoint or not self.lane_id:
            return endpoint or ""
        if endpoint == self.namespace or endpoint.startswith(f"{self.namespace}/"):
            return endpoint
        base = f"/benchmark/{self.instance_id}/"
        suffix = endpoint[len(base) :] if endpoint.startswith(base) else endpoint.lstrip("/")
        return f"{self.namespace}/{suffix}"


_INSTANCE_ID_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_FULL_PATH_OVERRIDE_KEYS = frozenset(("namespace", "reset_service", "step_service", "status_topic"))


def resolve_benchmark_cloud_deployment(evaluation: Mapping[str, Any]) -> str:
    """Resolve where the parallel shared cloud owner runs (SSOT switch).

    ``colocated`` (default) co-launches the shared cloud owner in the benchmark
    launch and pins every lane RTP endpoint to loopback. ``external`` omits the
    co-launched owner, keeps the profile-declared endpoint hosts, and exports
    the cloud configs so a ``benchmark_shared_cloud.launch.py`` on another host
    can own the single shared runtime. Effective only with parallel enabled.
    """
    if not isinstance(evaluation, Mapping):
        raise BenchmarkEndpointError("benchmark.evaluation must be a mapping")
    parallel = evaluation.get("parallel", {})
    if not isinstance(parallel, Mapping):
        raise BenchmarkEndpointError("benchmark.evaluation.parallel must be a mapping")
    deployment = parallel.get("cloud_deployment", "colocated")
    if deployment not in ("colocated", "external"):
        raise BenchmarkEndpointError("benchmark.evaluation.parallel.cloud_deployment must be 'colocated' or 'external'")
    return deployment


def resolve_benchmark_parallel(evaluation: Mapping[str, Any]) -> tuple[bool, int]:
    """Resolve the explicit shared-policy opt-in and active static lane count."""
    if not isinstance(evaluation, Mapping):
        raise BenchmarkEndpointError("benchmark.evaluation must be a mapping")
    enabled = evaluation.get("enabled", False)
    if not isinstance(enabled, bool):
        raise BenchmarkEndpointError("benchmark.evaluation.enabled must be a boolean")
    if "lane_count" in evaluation:
        raise BenchmarkEndpointError("use benchmark.evaluation.parallel.lane_count, not evaluation.lane_count")
    parallel = evaluation.get("parallel", {})
    if not isinstance(parallel, Mapping):
        raise BenchmarkEndpointError("benchmark.evaluation.parallel must be a mapping")
    parallel_enabled = parallel.get("enabled", False)
    if not isinstance(parallel_enabled, bool):
        raise BenchmarkEndpointError("benchmark.evaluation.parallel.enabled must be a boolean")
    count = parallel.get("lane_count", 1)
    if type(count) is not int or count < 1:
        raise BenchmarkEndpointError("benchmark.evaluation.parallel.lane_count must be a positive int (bool rejected)")
    if count > MAX_BENCHMARK_LANES:
        raise BenchmarkEndpointError(
            f"benchmark.evaluation.parallel.lane_count must be <= {MAX_BENCHMARK_LANES}; got {count}"
        )
    if parallel.get("scheduling", "static") != "static":
        raise BenchmarkEndpointError("benchmark.evaluation.parallel.scheduling must be static")
    if not isinstance(parallel.get("fail_fast", True), bool):
        raise BenchmarkEndpointError("benchmark.evaluation.parallel.fail_fast must be a boolean")
    if parallel_enabled:
        if not enabled:
            raise BenchmarkEndpointError("benchmark.evaluation.parallel.enabled requires evaluation.enabled=true")
        if parallel.get("fail_fast", True) is not True:
            raise BenchmarkEndpointError("parallel benchmark requires fail_fast=true")
        if count > 1 and evaluation.get("output", {}).get("write_canonical", True) is not True:
            raise BenchmarkEndpointError("parallel benchmark requires output.write_canonical=true")
    return parallel_enabled, count if parallel_enabled else 1


def resolve_benchmark_inference(evaluation: Mapping[str, Any]) -> dict[str, int]:
    """Validate shared admission settings, effective only with parallel.enabled=true.

    Serial evaluation does not consume these settings, but malformed settings
    are rejected in every mode. A single lane with parallel enabled still uses
    shared inference and consumes both limits. The concurrency level itself is
    model-neutral here; the shared owner enforces what it can actually host.
    """
    inference = evaluation.get("inference", {})
    if not isinstance(inference, Mapping):
        raise BenchmarkEndpointError("benchmark.evaluation.inference must be a mapping")
    limits = {}
    for key, default in (("max_in_flight", 1), ("queue_size", 64)):
        value = inference.get(key, default)
        if type(value) is not int or value < 1:
            raise BenchmarkEndpointError(f"benchmark.evaluation.inference.{key} must be a positive int (bool rejected)")
        limits[key] = value
    if limits["queue_size"] < limits["max_in_flight"]:
        raise BenchmarkEndpointError("benchmark.evaluation.inference.queue_size must be >= max_in_flight")
    return limits


def resolve_benchmark_run_timeout(evaluation: Mapping[str, Any]) -> float:
    """Resolve the multi-lane coordinator's whole-run wall-clock limit."""
    timeouts = evaluation.get("timeouts", {})
    if not isinstance(timeouts, Mapping):
        raise BenchmarkEndpointError("benchmark.evaluation.timeouts must be a mapping")
    value = timeouts.get("run_timeout_sec", 3600.0)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise BenchmarkEndpointError("benchmark.evaluation.timeouts.run_timeout_sec must be finite and positive")
    return float(value)


def _is_non_empty_exact_string(value: Any) -> bool:
    """Return True only for a non-empty string without surrounding whitespace.

    Whitespace-padded / empty values are rejected so the resolver does not
    silently ``.strip()`` caller input. Bool/list/int are rejected too.
    """
    if not isinstance(value, str):
        return False
    if not value:
        return False
    return value.strip() == value


def _require_benchmark_section(robot_config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "benchmark" not in robot_config:
        raise BenchmarkEndpointError(
            "robot config is missing the 'benchmark' section; "
            "benchmark endpoint identity requires benchmark.type and benchmark.adapter"
        )
    benchmark = robot_config["benchmark"]
    if not isinstance(benchmark, Mapping):
        raise BenchmarkEndpointError("benchmark section must be a mapping when present")
    return benchmark


def _require_type(benchmark: Mapping[str, Any]) -> str:
    if "type" not in benchmark:
        raise BenchmarkEndpointError("benchmark.type is required and must be an exact non-empty string")
    value = benchmark["type"]
    if not _is_non_empty_exact_string(value):
        raise BenchmarkEndpointError(
            "benchmark.type must be an exact non-empty string (no surrounding whitespace, no bool/number/list)"
        )
    return value


def _require_adapter(benchmark: Mapping[str, Any]) -> str:
    if "adapter" not in benchmark:
        raise BenchmarkEndpointError("benchmark.adapter is required and must be an exact non-empty string")
    value = benchmark["adapter"]
    if not _is_non_empty_exact_string(value):
        raise BenchmarkEndpointError(
            "benchmark.adapter must be an exact non-empty string (no surrounding whitespace, no bool/number/list)"
        )
    return value


def _resolve_instance_id(benchmark: Mapping[str, Any], benchmark_type: str) -> str:
    if "instance_id" not in benchmark:
        candidate = benchmark_type
    else:
        candidate = benchmark["instance_id"]
        if not _is_non_empty_exact_string(candidate):
            raise BenchmarkEndpointError(
                "benchmark.instance_id must be an exact non-empty string "
                "(no surrounding whitespace, no bool/number/list)"
            )
    if not _INSTANCE_ID_PATTERN.match(candidate):
        raise BenchmarkEndpointError(
            f"benchmark.instance_id token {candidate!r} is not a legal ROS "
            f"identifier (expected pattern {_INSTANCE_ID_PATTERN.pattern}); "
            "configure benchmark.instance_id explicitly when benchmark.type is "
            "not a legal token; resolver will not rewrite, lowercase or alias it"
        )
    return candidate


def _reject_full_path_overrides(benchmark: Mapping[str, Any]) -> None:
    overlaps = _FULL_PATH_OVERRIDE_KEYS.intersection(benchmark.keys())
    if overlaps:
        joined = ", ".join(sorted(overlaps))
        raise BenchmarkEndpointError(
            "benchmark section must not provide full ROS path keys; "
            f"found: {joined}. "
            "Endpoints are derived from benchmark.type / benchmark.instance_id."
        )


def resolve_benchmark_endpoints(
    robot_config: Mapping[str, Any],
    *,
    lane_id: str | None = None,
) -> BenchmarkEndpoints:
    """Resolve the deterministic benchmark ROS endpoint identity.

    Args:
        robot_config: The normalized robot SSOT mapping (the ``robot`` section
            after the loader has unpacked the outer ``{"robot": {...}}``
            wrapper). Must contain a top-level ``benchmark`` mapping with
            non-empty ``type`` and ``adapter`` strings.

    Returns:
        A frozen ``BenchmarkEndpoints`` with namespace, reset service and
        step service derived from ``instance_id``.

    Raises:
        BenchmarkEndpointError: For any malformed configuration. The resolver
            never returns ``None`` or partial results and never silently
            falls back.
    """
    if not isinstance(robot_config, Mapping):
        raise BenchmarkEndpointError("robot_config must be a mapping (the normalized robot SSOT)")

    benchmark = _require_benchmark_section(robot_config)
    _reject_full_path_overrides(benchmark)

    benchmark_type = _require_type(benchmark)
    adapter_name = _require_adapter(benchmark)
    instance_id = _resolve_instance_id(benchmark, benchmark_type)

    if lane_id is not None:
        if not isinstance(lane_id, str) or not re.fullmatch(r"lane_[0-9]{3,}", lane_id):
            raise BenchmarkEndpointError("lane_id must use the canonical lane_NNN format when provided")
        namespace = f"/benchmark/{instance_id}/{lane_id}"
    else:
        namespace = f"/benchmark/{instance_id}"
    return BenchmarkEndpoints(
        benchmark_type=benchmark_type,
        adapter_name=adapter_name,
        instance_id=instance_id,
        namespace=namespace,
        reset_service=f"{namespace}/reset",
        step_service=f"{namespace}/step",
        lane_id=lane_id or "",
    )
