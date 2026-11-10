"""Transport-neutral runtime target value shared by configuration and control."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

MAX_CONTROL_BYTES = 32768
MAX_RETRY_WINDOW_MS = 300000
MAX_RUNTIME_GROUP_MEMBERS = 16
DEFAULT_RUNTIME_ENDPOINT = "/inference/runtime/v1"
DEFAULT_ADMISSION_SAFETY_MARGIN_MS = 5
RUNTIME_LIMIT_DEFAULTS = {
    "max_pending_requests": 8,
    "max_retained_input_bytes": 67108864,
    "max_retained_result_bytes": 67108864,
    "max_records": 4096,
    "retry_window_ms": 30000,
}


def validate_profile_observation_inputs(value):
    """Normalize the shared workload observation-name contract."""
    if (
        not isinstance(value, list | tuple)
        or len(value) > 64
        or any(not isinstance(name, str) or not name or len(name.encode()) > 256 for name in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("profile_observation_inputs must contain unique, nonempty input names (at most 64)")
    return tuple(sorted(value))


def validate_runtime_limits(value):
    """Validate retention limits using the same defaults as a target."""
    for name, number in value.items():
        if name not in RUNTIME_LIMIT_DEFAULTS or type(number) is not int or not 1 <= number <= 2**31 - 1:
            raise ValueError(f"{name} must be a positive integer runtime limit")
    limits = {**RUNTIME_LIMIT_DEFAULTS, **value}
    if limits["max_records"] < limits["max_pending_requests"] or limits["retry_window_ms"] > MAX_RETRY_WINDOW_MS:
        raise ValueError("invalid record capacity or retry window")


def validate_runtime_group_budget(targets):
    """Reserve a complete Begin/registration envelope, including every member ID.

    A group is atomic and is never split across control messages. Use maximum
    supported wire identities/counters so a valid group also fits heartbeats.
    """
    targets = [target.to_dict() if isinstance(target, RuntimeExecutionTarget) else target for target in targets]
    if not 1 <= len(targets) <= MAX_RUNTIME_GROUP_MEMBERS:
        raise ValueError(f"runtime group requires 1..{MAX_RUNTIME_GROUP_MEMBERS} members")
    identity = "f" * 36
    statuses = [
        {
            "target": target,
            "lifecycle_target_ids": [member["target_id"] for member in targets],
            "boot_id": identity,
            "clock_id": identity,
            "epoch": identity,
            "generation": 2**63 - 1,
            "sequence": 2**63 - 1,
            "status_ns": 2**63 - 1,
            "ready": False,
            "clean": False,
        }
        for target in targets
    ]
    envelope = {"verb": "begin_initialization", "statuses": statuses, "token": identity, "accept_until_ns": 2**63 - 1}
    if len(json.dumps(envelope, separators=(",", ":"), allow_nan=False).encode()) > MAX_CONTROL_BYTES:
        raise ValueError("runtime group exceeds control message budget")


def identifier(value: str, name: str, *, limit: int = 127) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_:./-]*", value) or len(value) > limit:
        raise ValueError(f"invalid {name}")


def validate_runtime_endpoint(value):
    """Validate an absolute ROS service name, reserving DDS suffix space."""
    if (
        not isinstance(value, str)
        or len(value) > 247
        or not re.fullmatch(r"/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*", value)
    ):
        raise ValueError("invalid runtime endpoint")


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
    max_pending_requests: int = RUNTIME_LIMIT_DEFAULTS["max_pending_requests"]
    max_retained_input_bytes: int = RUNTIME_LIMIT_DEFAULTS["max_retained_input_bytes"]
    max_retained_result_bytes: int = RUNTIME_LIMIT_DEFAULTS["max_retained_result_bytes"]
    max_records: int = RUNTIME_LIMIT_DEFAULTS["max_records"]
    request_timeout_ms: int = 5000
    retry_window_ms: int = RUNTIME_LIMIT_DEFAULTS["retry_window_ms"]
    default_priority: int = 0
    profile_path: str = ""
    profile_observation_inputs: tuple[str, ...] = ()
    predictive_admission: bool = False
    admission_safety_margin_ms: int = DEFAULT_ADMISSION_SAFETY_MARGIN_MS
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
        object.__setattr__(
            self, "profile_observation_inputs", validate_profile_observation_inputs(self.profile_observation_inputs)
        )
        for name in ("target_id", "hardware_resource_id", "physical_device", "lifecycle_group"):
            identifier(getattr(self, name), name)
        # ROS service names reserve DDS suffix space; validate before publishing
        # dynamic membership, so client construction cannot poison the tick loop.
        validate_runtime_endpoint(self.endpoint)
        for name in ("deployment_fingerprint", "composition_fingerprint", "configuration_generation"):
            identifier(getattr(self, name), name)
        for name in (
            "max_in_flight",
            "request_timeout_ms",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 2**31 - 1:
                raise ValueError(f"invalid {name}")
        validate_runtime_limits({name: getattr(self, name) for name in RUNTIME_LIMIT_DEFAULTS})
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
