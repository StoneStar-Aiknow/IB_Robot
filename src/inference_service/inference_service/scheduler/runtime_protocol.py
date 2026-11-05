"""Versioned, data-free control values for scheduler-owned runtime execution.

Credentials use the Linux boot clock domain. Hosts must run on the same machine;
ROS simulation time and wall-clock adjustments never extend an operation.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import UUID

from inference_service.unified_runtime.operations import Certainty

PROTOCOL_VERSION = 1
DEFAULT_RUNTIME_ENDPOINT = "/inference/runtime/v1"
HOST_HEARTBEAT_INTERVAL_NS = 100_000_000
DEFAULT_STATUS_STALE_TIMEOUT_NS = 2_000_000_000
DEFAULT_CONTROL_RPC_TIMEOUT_NS = 5_000_000_000
RUNTIME_WORK_TYPES = frozenset({"execute", "reset", "close", "drain"})
MAX_CONTROL_BYTES = 32768
MAX_INPUT_BUCKET_BYTES = 8192
EMPTY_MAPPING_BYTES = 128
# Two empty mapping snapshots for a Close control request; never business data.
CLOSE_INPUT_RESERVE_BYTES = 2 * EMPTY_MAPPING_BYTES
TERMINAL = frozenset({Certainty.NOT_STARTED.value, Certainty.COMPLETED.value})


class ControlError(RuntimeError):
    operation_started = True
    outcome_known = False

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class NotStartedControlError(ControlError):
    admission_rejected = True
    operation_started = False
    outcome_known = True
    state_mutated = False


class CompletedControlError(ControlError):
    outcome_known = True


def machine_clock_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def identifier(value: str, name: str, *, limit: int = 127) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_:./-]*", value) or len(value) > limit:
        raise ValueError(f"invalid {name}")


def uuid_value(value: str) -> None:
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError("expected canonical UUID")


@dataclass(frozen=True)
class RuntimeExecutionTarget:
    target_id: str
    endpoint: str
    hardware_resource_id: str
    physical_device: str
    deployment_fingerprint: str
    composition_fingerprint: str
    lifecycle_group: str
    configuration_generation: str
    interface: str
    model_type: str
    operation: str
    required: bool = True
    template_only: bool = False
    execution_kind: str = "executor"
    resource_mode: str = "serial"
    max_in_flight: int = 1
    max_pending_requests: int = 8
    max_retained_input_bytes: int = 67108864
    max_retained_result_bytes: int = 67108864
    max_records: int = 4096
    request_timeout_ms: int = 5000
    retry_window_ms: int = 30000
    default_priority: int = 0
    profile_path: str = ""
    profile_observation_inputs: tuple[str, ...] = ()
    predictive_admission: bool = False
    admission_safety_margin_ms: int = 5
    hardware_profile_fingerprint: str = ""
    runtime_version: str = ""

    def __post_init__(self):
        for name in ("required", "template_only", "predictive_admission"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"invalid {name}")
        for name in (
            "interface",
            "model_type",
            "operation",
            "runtime_version",
            "hardware_profile_fingerprint",
            "profile_path",
        ):
            value = getattr(self, name)
            limit = 4096 if name == "profile_path" else 256
            if (
                not isinstance(value, str)
                or len(value.encode()) > limit
                or (name in {"interface", "model_type", "operation"} and not value.strip())
            ):
                raise ValueError(f"invalid {name}")
        names = self.profile_observation_inputs
        if (
            not isinstance(names, list | tuple)
            or len(names) > 64
            or any(not isinstance(name, str) or not name or len(name.encode()) > 256 for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("invalid profile_observation_inputs")
        object.__setattr__(self, "profile_observation_inputs", tuple(sorted(names)))
        for name in ("target_id", "hardware_resource_id", "physical_device", "lifecycle_group"):
            identifier(getattr(self, name), name)
        # ROS service names reserve DDS suffix space; validate before publishing
        # dynamic membership, so client construction cannot poison the tick loop.
        if (
            not isinstance(self.endpoint, str)
            or len(self.endpoint) > 247
            or not re.fullmatch(r"/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*", self.endpoint)
        ):
            raise ValueError("invalid runtime endpoint")
        for name in ("deployment_fingerprint", "composition_fingerprint", "configuration_generation"):
            identifier(getattr(self, name), name)
        for name in (
            "max_in_flight",
            "max_pending_requests",
            "max_retained_input_bytes",
            "max_retained_result_bytes",
            "max_records",
            "request_timeout_ms",
            "retry_window_ms",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 2**31 - 1:
                raise ValueError(f"invalid {name}")
        if self.max_records < self.max_pending_requests or self.retry_window_ms > 300000:
            raise ValueError("invalid record capacity or retry window")
        if not self.accepts_priority(self.default_priority):
            raise ValueError("invalid priority")
        if type(self.admission_safety_margin_ms) is not int or self.admission_safety_margin_ms < 0:
            raise ValueError("invalid admission safety margin")
        if self.execution_kind not in {"executor", "stage"} or self.resource_mode not in {
            "serial",
            "ascend_concurrent",
        }:
            raise ValueError("unsupported execution kind/resource mode")
        if self.resource_mode == "serial" and self.execution_kind == "stage":
            raise ValueError("serial resources require complete executors")
        if self.predictive_admission and self.resource_mode != "serial":
            raise ValueError("prediction requires fully managed serial resources")
        if self.predictive_admission and (not self.hardware_profile_fingerprint or not self.runtime_version):
            raise ValueError("prediction requires hardware fingerprint and runtime version")
        # Leave room for status identity, sequence and control envelope. JSON
        # escaping can be larger than the individual UTF-8 field limits.
        if len(json.dumps(self.to_dict(), separators=(",", ":"), allow_nan=False).encode()) > MAX_CONTROL_BYTES // 2:
            raise ValueError("runtime target exceeds control descriptor budget")

    def accepts_priority(self, priority: int) -> bool:
        """Software queue priorities are unbounded; native streams define their range."""
        return type(priority) is int and priority >= 0 and (self.resource_mode != "ascend_concurrent" or priority <= 7)

    def to_dict(self):
        return {**asdict(self), "profile_observation_inputs": list(self.profile_observation_inputs)}


@dataclass(frozen=True)
class Submission:
    operation_id: str
    parent_id: str
    target_id: str
    boot_id: str
    generation: int
    request_ref: str
    digest: str
    clock_id: str
    accept_until_ns: int
    deadline_ns: int
    priority: int
    input_bytes: int
    input_bucket: str
    work: str = "execute"
    snapshot_valid_until_ns: int = 0

    def __post_init__(self):
        for value in (self.operation_id, self.boot_id, self.request_ref, self.clock_id):
            uuid_value(value)
        identifier(self.target_id, "target")
        if type(self.parent_id) is not str or not 1 <= len(self.parent_id.encode()) <= 256:
            raise ValueError("invalid parent ID")
        if not re.fullmatch(r"[a-f0-9]{64}", self.digest):
            raise ValueError("invalid input digest")
        if type(self.priority) is not int or self.priority < 0:
            raise ValueError("invalid priority")
        for name in ("generation", "accept_until_ns", "deadline_ns", "input_bytes", "snapshot_valid_until_ns"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 2**63 - 1:
                raise ValueError(f"invalid {name}")
        if not self.generation or not self.deadline_ns or not self.accept_until_ns:
            raise ValueError("missing generation/deadline/expiry")
        if not isinstance(self.input_bucket, str) or len(self.input_bucket.encode()) > MAX_INPUT_BUCKET_BYTES:
            raise ValueError("invalid input bucket")
        if self.work not in RUNTIME_WORK_TYPES:
            raise ValueError("invalid work")

    @property
    def latest_start_ns(self):
        return min(self.deadline_ns, self.accept_until_ns, self.snapshot_valid_until_ns or self.deadline_ns)

    def validate_window(self, now_ns: int, retry_window_ms: int):
        if now_ns >= self.accept_until_ns:
            raise ControlError("OPERATION_EXPIRED")
        if self.accept_until_ns - now_ns > retry_window_ms * 1000000:
            raise ControlError("INVALID_RETRY_WINDOW")

    def to_dict(self):
        return asdict(self)


def encode_control(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(payload.encode()) > MAX_CONTROL_BYTES:
        raise ControlError("CONTROL_TOO_LARGE")
    return payload


def decode_control(payload: str) -> dict:
    if len(payload.encode()) > MAX_CONTROL_BYTES:
        raise ControlError("CONTROL_TOO_LARGE")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ControlError("DUPLICATE_FIELD")
            result[key] = value
        return result

    result = json.loads(payload, object_pairs_hook=pairs)
    if not isinstance(result, dict):
        raise ControlError("INVALID_CONTROL")
    return result
