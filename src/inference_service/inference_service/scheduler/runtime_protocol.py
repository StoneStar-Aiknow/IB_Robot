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

from inference_manifest.runtime_target import DEFAULT_RUNTIME_ENDPOINT as DEFAULT_RUNTIME_ENDPOINT
from inference_manifest.runtime_target import MAX_CONTROL_BYTES
from inference_manifest.runtime_target import RuntimeExecutionTarget as RuntimeExecutionTarget
from inference_manifest.runtime_target import identifier as identifier
from inference_service.unified_runtime.operations import Certainty

PROTOCOL_VERSION = 1
HOST_HEARTBEAT_INTERVAL_NS = 100_000_000
DEFAULT_STATUS_STALE_TIMEOUT_NS = 2_000_000_000
DEFAULT_CONTROL_RPC_TIMEOUT_NS = 5_000_000_000
RUNTIME_WORK_TYPES = frozenset({"execute", "reset", "close", "drain"})
# Local Handle dispatch metadata, shared by binding and profile selection.
RUNTIME_HANDLE_OPERATION = "runtime_handle_operation"
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


def uuid_value(value: str) -> None:
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError("expected canonical UUID")


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
