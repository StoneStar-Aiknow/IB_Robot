"""Host-local request ownership and exactly-once controlled handle invocation."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timedelta, timezone
from heapq import heappop, heappush
from types import MappingProxyType
from uuid import uuid4

from ibrobot_tracing import get_trace_emitter
from inference_service.unified_runtime import CancellationToken, Deadline, ExecutionContext, ModelRequest
from inference_service.unified_runtime.contracts import ModelResult, OutcomeEvidence, OutcomeState, RuntimeLatency
from inference_service.unified_runtime.errors import (
    ExecutionFailure,
    RecoveryAction,
    RecoveryRequirement,
    RecoveryScope,
)
from inference_service.unified_runtime.handle import _completed_control_failure

from .runtime_protocol import (
    CLOSE_INPUT_RESERVE_BYTES,
    EMPTY_MAPPING_BYTES,
    MAX_INPUT_BUCKET_BYTES,
    RUNTIME_HANDLE_OPERATION,
    TERMINAL,
    CompletedControlError,
    ControlError,
    NotStartedControlError,
    RuntimeExecutionTarget,
    Submission,
    encode_control,
    machine_clock_id,
    uuid_value,
)

trace = get_trace_emitter("ib_trace.runtime", component_id="runtime_controller")
_INPUT_CHECK_INTERVAL = 256
_MAX_INPUT_NESTING_DEPTH = 64
# Shared quota estimates, not interpreter object sizes or a process RSS limit.
_MAPPING_ENTRY_BYTES = 64
_EMPTY_SEQUENCE_BYTES = 64
_SEQUENCE_SLOT_BYTES = 8
_STRING_HEADER_BYTES = 64
_STRING_CHARACTER_BYTES = 4
# Per Host command receipts, including completed commands until coordinator ACK
# and the last accepted retry window have both been observed.
MAX_HOST_RECOVERY_RECEIPTS = 16


def _copy_failure(failure):
    """Replay structured evidence without retaining causes or caller frames."""
    copied = ExecutionFailure(
        code=failure.code,
        message=failure.message,
        recoverable=failure.recoverable,
        recovery=failure.recovery,
        evidence=failure.evidence,
        details=failure.details,
    )
    copied._runtime_failure_owner = getattr(failure, "_runtime_failure_owner", None)
    return copied


def _input_overhead(value):
    """Conservative snapshot bytes, including containers and per-element slots."""
    if isinstance(value, Mapping):
        return EMPTY_MAPPING_BYTES + _MAPPING_ENTRY_BYTES * len(value)
    if isinstance(value, list | tuple):
        return _EMPTY_SEQUENCE_BYTES + _SEQUENCE_SLOT_BYTES * len(value)
    if isinstance(value, frozenset):
        return sys.getsizeof(value)
    if type(value).__module__.startswith("numpy"):
        return 128 + int(value.nbytes)
    if type(value).__module__.startswith("torch"):
        if value.is_quantized:
            # Quantization parameters are separate, potentially shared storage.
            # Neither the byte estimate nor clone/digest supports that ownership.
            raise ControlError("OPAQUE_INPUT")
        return 128 + value.numel() * value.element_size()
    if isinstance(value, str):
        return _STRING_HEADER_BYTES + _STRING_CHARACTER_BYTES * len(value)
    if value is None or type(value) in (int, float, bool):
        return 64 + len(json.dumps(value, allow_nan=False).encode())
    raise ControlError("OPAQUE_INPUT")


def _opaque_numpy_dtype(dtype):
    """Tensor payloads exclude structured dtypes and hidden dtype ownership."""
    if dtype.hasobject or dtype.metadata or dtype.fields is not None:
        return True
    return dtype.subdtype is not None and _opaque_numpy_dtype(dtype.subdtype[0])


def freeze_inputs(request: ModelRequest, byte_limit: int, observation_inputs=(), *, require_profile=True, context=None):
    """Copy supported CPU inputs without invoking a controlled executor.

    Return (snapshot, digest, retained_bytes, workload_bucket): the owned request,
    exact input/metadata identity, charged snapshot size, and serialized profile
    descriptors. The bucket includes paths, containers, tensors and scalar content;
    it is empty when an optional profile exceeds its byte budget.
    """
    digest = hashlib.sha256()
    size = 0
    visited = 0
    bucket_entries = []
    bucket_bytes = 2
    observations = frozenset(observation_inputs)
    if observations - request.inputs.keys():
        raise ControlError("PROFILE_OBSERVATION_INPUT_MISSING")
    observed_tensors = set()

    def charge(amount):
        nonlocal size, visited
        if context is not None and visited % _INPUT_CHECK_INTERVAL == 0:
            context.check("input_preparation")
        visited += 1
        size += amount
        if size > byte_limit:
            raise ControlError("INPUT_CAPACITY")

    def describe(path, descriptor):
        nonlocal bucket_bytes, bucket_entries
        if bucket_entries is None:
            return
        if path[0] == "metadata" and len(path) > 1 and path[1] in request.identity_metadata_keys:
            return
        entry = [path, descriptor]
        bucket_bytes += len(json.dumps(entry, separators=(",", ":"), sort_keys=True).encode()) + bool(bucket_entries)
        if bucket_bytes > MAX_INPUT_BUCKET_BYTES:
            if require_profile:
                raise ControlError("INPUT_BUCKET_TOO_LARGE")
            # FIFO execution does not require a profile. An absent bucket is
            # deliberately not sampled, so it cannot alias a measured workload.
            bucket_entries = None
            return
        bucket_entries.append(entry)

    def tensor_descriptor(path, shape, dtype, data, observation):
        # Identity always covers the exact bytes, independently of profiling.
        descriptor = {"shape": shape, "dtype": dtype, "content_sha256": hashlib.sha256(data).hexdigest()}
        digest.update(json.dumps(descriptor, sort_keys=True).encode())
        if observation is not None:
            observed_tensors.add(observation)
            descriptor = {"shape": shape, "dtype": dtype}
        describe(path, descriptor)

    def freeze(value, path, observation=None):
        if len(path) > _MAX_INPUT_NESTING_DEPTH:
            raise ControlError("INPUT_NESTING_TOO_DEEP")
        charge(_input_overhead(value))
        if isinstance(value, Mapping):
            for key in value:
                if type(key) is not str:
                    raise ControlError("INVALID_INPUT_KEY")
                charge(_input_overhead(key))
            count = len(value)
            if path == ("metadata",):
                count -= len(request.identity_metadata_keys.intersection(value))
            describe(path, {"mapping": count})
            digest.update(b"map{")
            result = {}
            for key in sorted(value):
                digest.update(json.dumps(key).encode())
                selected = key if path == ("inputs",) and key in observations else observation
                result[key] = freeze(value[key], (*path, key), selected)
            digest.update(b"}")
            return MappingProxyType(result)
        if isinstance(value, list | tuple):
            describe(path, {"sequence": len(value)})
            digest.update(b"list[")
            result = tuple(freeze(item, (*path, i), observation) for i, item in enumerate(value))
            digest.update(b"]")
            return result
        if type(value).__module__.startswith("numpy"):
            import numpy as np

            array = np.asarray(value)
            if _opaque_numpy_dtype(array.dtype):
                raise ControlError("OPAQUE_INPUT")
            array = np.array(array, copy=True, order="C")
            array.flags.writeable = False
            data = memoryview(array.reshape(-1).view(np.uint8))
            tensor_descriptor(path, list(array.shape), str(array.dtype), data, observation)
            return array
        if type(value).__module__.startswith("torch"):
            if value.device.type != "cpu":
                raise ControlError("DEVICE_INPUT_OUTSIDE_EXECUTOR")
            # Preserve the caller's tensor type; all device movement belongs to
            # the controlled executor. No accelerator SDK is imported here.
            import torch

            tensor = value.detach().clone(memory_format=torch.contiguous_format)
            data = memoryview(tensor.reshape(-1).view(__import__("torch").uint8).numpy())
            tensor_descriptor(path, list(tensor.shape), str(tensor.dtype), data, observation)
            return tensor
        if value is None or type(value) in (str, int, float, bool):
            data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
            # Include scalar content, not just tensor shapes, in the profile
            # bucket: diffusion steps, prompts and options must not alias.
            if bucket_entries is not None:
                describe(path, {"scalar_sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
            digest.update(data)
            digest.update(b";")
            return value
        raise ControlError("OPAQUE_INPUT")

    # Nonempty identity selectors are retained alongside the mappings, even
    # when they name absent metadata. Empty selectors are fixed request metadata.
    if request.identity_metadata_keys:
        charge(_input_overhead(request.identity_metadata_keys))
        for key in request.identity_metadata_keys:
            if type(key) is not str:
                raise ControlError("INVALID_INPUT_KEY")
            charge(_input_overhead(key))
    inputs = freeze(request.inputs, ("inputs",))
    if observations != observed_tensors:
        raise ControlError("PROFILE_OBSERVATION_INPUT_NOT_TENSOR")
    metadata = freeze(request.metadata, ("metadata",))
    bucket = "" if bucket_entries is None else json.dumps(bucket_entries, sort_keys=True, separators=(",", ":"))
    return ModelRequest(inputs, metadata, request.identity_metadata_keys), digest.hexdigest(), size, bucket


def profile_inputs_for_request(request, configured, work="execute"):
    """Select observations for the actual operation; role identity stays in the bucket.

    A target may contain several manifest roles with disjoint inputs. Their
    required tensors are validated by the role contract, while this list only
    selects which present tensors use shape-based empirical profiles.
    """
    if work != "execute" or request.metadata.get(RUNTIME_HANDLE_OPERATION) in {
        "open_stream",
        "reset_stream",
        "close_stream",
    }:
        return ()
    if "role" in request.metadata:
        return tuple(name for name in configured if name in request.inputs)
    return configured


def input_bytes(request, byte_limit=None, *, context=None):
    """Size CPU snapshots without copying or scanning tensor contents."""
    total = 0
    visited = 0

    def size(value, depth=1):
        nonlocal total, visited
        if context is not None and visited % _INPUT_CHECK_INTERVAL == 0:
            context.check("input_preparation")
        visited += 1
        if depth > _MAX_INPUT_NESTING_DEPTH:
            raise ControlError("INPUT_NESTING_TOO_DEEP")
        total += _input_overhead(value)
        if byte_limit is not None and total > byte_limit:
            raise ControlError("INPUT_CAPACITY")
        if isinstance(value, Mapping):
            for key, item in value.items():
                if type(key) is not str:
                    raise ControlError("INVALID_INPUT_KEY")
                size(key, depth)
                size(item, depth + 1)
        elif isinstance(value, list | tuple | frozenset):
            for item in value:
                size(item, depth + 1)

    size(request.inputs)
    size(request.metadata)
    if request.identity_metadata_keys:
        if any(type(key) is not str for key in request.identity_metadata_keys):
            raise ControlError("INVALID_INPUT_KEY")
        size(request.identity_metadata_keys)
    return total


@dataclass
class LocalOperation:
    submission: Submission
    request: ModelRequest | None
    context: ExecutionContext | None
    state: str = "prepared"
    error: str = ""
    result: object = None
    failure: BaseException | None = None
    acknowledged: bool = False
    collection_scheduled: bool = False
    result_waiter: bool = False
    started_ns: int = 0
    finished_ns: int = 0
    close_complete: bool = False
    result_bytes: int = 0
    terminal_retry_after_ns: int = 0
    done: Future = field(default_factory=Future)
    receipt: Future = field(default_factory=Future)
    release_cancellation: Callable[[], None] | None = None

    def report(self):
        return {
            "submission": self.submission.to_dict(),
            "state": self.state,
            "error": self.error,
            "started_ns": self.started_ns,
            "finished_ns": self.finished_ns,
            **({"close_complete": self.close_complete} if self.submission.work == "close" else {}),
        }


@dataclass
class RecoveryReceipt:
    command: str
    accept_until_ns: int
    future: Future
    expected_generation: int | None
    stream_id: str | None
    retain_until_ns: int
    acknowledged: bool = False


class RuntimeExecutionController:
    """No device queue: validate, claim once, invoke the owned handle, retain evidence."""

    def __init__(self, target: RuntimeExecutionTarget, handle, *, now_ns=time.monotonic_ns):
        self.target = target
        self.handle = handle
        self.boot_id = str(uuid4())
        self.generation = 1
        self.epoch = ""
        self.clock_id = machine_clock_id()
        self.now_ns = now_ns
        self._lock = threading.RLock()
        self._preparation_done = threading.Condition(self._lock)
        self._records = {}
        self._terminal_pending = OrderedDict()
        self._pending = 0
        self._expirations = []
        self._preparing = {}
        self.on_terminal = None
        self._input_bytes = 0
        self._result_bytes = 0
        self._running = 0
        self._gate = "ready"
        self._sequence = 0
        self.startup_clean = True
        self.lifecycle_peers = [self]
        self._workers = ThreadPoolExecutor(max_workers=target.max_in_flight, thread_name_prefix="runtime-execute")
        self._recovery = ThreadPoolExecutor(max_workers=1, thread_name_prefix="runtime-recovery")
        self._recovery_records = {}
        # Receipt ACK/expiry is not device completion. Only a proven backend
        # drain clears uncertainty left by a recovery command.
        self._recovery_uncertain = False
        self._control_future = None
        self.policy = _ControllerPolicy(self)
        try:
            self._authority = handle.bind_execution_policy(self.policy)
        except BaseException:
            self._workers.shutdown(wait=True)
            self._recovery.shutdown(wait=True)
            raise

    @contextmanager
    def transaction(self):
        """Serialize a host transport transaction with lifecycle transitions."""
        with self._lock:
            yield

    @property
    def ready(self):
        with self._lock:
            return self._gate == "ready" and self.handle.local_ready

    @property
    def input_available(self):
        with self._lock:
            return self._gate == "maintenance" or self.ready

    def fence(self):
        with self._lock:
            if self._gate != "closing":
                self._gate = "recovery"

    def cancel_pending(self, parent_id=None, *, closing=False):
        with self._lock:
            if closing:
                self._gate = "closing"
            for _, context in self._preparing.values():
                if parent_id is None or context.request_id == parent_id:
                    context.cancellation_token.cancel("scheduler cancellation")
            pending = [
                record.submission
                for record in self._records.values()
                if record.state not in TERMINAL and (parent_id is None or record.submission.parent_id == parent_id)
            ]
            for submission in pending:
                self.cancel(submission, self.epoch)
            return pending

    def terminal_reports(self):
        with self._lock:
            return [record.report() for record in self._records.values() if record.state in TERMINAL]

    def require_transport_release(self):
        """Validate local cleanup evidence; Global retirement belongs to the adapter."""
        with self._lock:
            if any(record.state not in TERMINAL or not record.acknowledged for record in self._records.values()):
                raise ControlError("TERMINAL_EVIDENCE_PENDING")
            if not self.handle.closed or self._preparing or self._running:
                raise ControlError("RUNTIME_CLOSE_REQUIRED")

    def status(self):
        with self._lock:
            self._sequence += 1
            return {
                "target": self.target.to_dict(),
                "lifecycle_target_ids": sorted(peer.target.target_id for peer in self.lifecycle_peers),
                "boot_id": self.boot_id,
                "generation": self.generation,
                "epoch": self.epoch,
                "clock_id": self.clock_id,
                "sequence": self._sequence,
                "status_ns": self.now_ns(),
                "ready": self._gate == "ready" and self.handle.local_ready,
                "clean": (
                    self.startup_clean
                    and not self._pending
                    and not self._running
                    and not self._preparing
                    and not self._recovery_uncertain
                    and (self._control_future is None or self._control_future.done())
                ),
            }

    def accept_epoch(self, epoch):
        with self._lock:
            if self.epoch and epoch != self.epoch:
                self._gate = "recovery"
                raise ControlError("COORDINATED_RESTART_REQUIRED")
            self.epoch = epoch

    def prepare(
        self, request, context, *, priority=None, work="execute", snapshot_valid_until_ns=0, retain_for_caller=True
    ):
        # The caller may reuse its context across Hosts or fallback attempts.
        # Only parent cancellation propagates; local cleanup owns a child token,
        # including while the input snapshot is still being prepared.
        parent = context.cancellation_token
        child = CancellationToken()
        local_context = ExecutionContext(context.request_id, context.deadline, child)

        def cancel_child():
            child.cancel(parent.reason)

        def release_cancellation():
            parent.remove_callback(cancel_child)

        parent.add_callback(cancel_child)
        try:
            return self._prepare(
                request,
                local_context,
                priority=priority,
                work=work,
                snapshot_valid_until_ns=snapshot_valid_until_ns,
                retain_for_caller=retain_for_caller,
                release_cancellation=release_cancellation,
            )
        except BaseException:
            release_cancellation()
            raise

    def _prepare(
        self, request, context, *, priority, work, snapshot_valid_until_ns, retain_for_caller, release_cancellation
    ):
        context.cancellation_token.throw_if_cancelled("input_preparation")
        admitted_ns = self.now_ns()
        remaining = context.deadline.remaining_seconds()
        remaining = self.target.request_timeout_ms / 1000 if remaining is None else remaining
        if remaining <= 0:
            raise ControlError("DEADLINE_EXPIRED")
        byte_limit = self.target.max_retained_input_bytes
        if work == "close":
            if request.inputs or request.metadata or request.identity_metadata_keys:
                raise ControlError("CLOSE_PAYLOAD_UNSUPPORTED")
            # Only the empty control request may use this bounded reserve.
            # Ordinary snapshots still have exactly their configured quota.
            byte_limit += CLOSE_INPUT_RESERVE_BYTES
        extra_bytes = getattr(self.handle, "retained_request_bytes", lambda _: 0)(request)
        reserved = input_bytes(request, byte_limit - extra_bytes, context=context) + extra_bytes
        context.check("input_preparation")
        preparation = object()
        with self._lock:
            self.collect()
            if not self.epoch or self._gate != "ready":
                raise ControlError("RUNTIME_NOT_REGISTERED")
            pending = self._pending + len(self._preparing)
            if pending >= self.target.max_pending_requests + (work == "close") or len(self._records) + len(
                self._preparing
            ) >= self.target.max_records + (work == "close"):
                raise ControlError("REQUEST_CAPACITY")
            if reserved + self._input_bytes > byte_limit:
                raise ControlError("INPUT_CAPACITY")
            identity = (self.epoch, self.generation)
            self._preparing[preparation] = (reserved, context)
            self._input_bytes += reserved
        try:
            context.check("input_preparation")
            frozen, digest, size, bucket = freeze_inputs(
                request,
                reserved - extra_bytes,
                profile_inputs_for_request(request, self.target.profile_observation_inputs, work),
                require_profile=self.target.predictive_admission,
                context=context,
            )
            size += extra_bytes
            with self._lock:
                if identity != (self.epoch, self.generation) or self._gate != "ready":
                    raise ControlError("RUNTIME_FENCED")
                context.check("input_preparation")
                return self._commit_prepared(
                    frozen=frozen,
                    digest=digest,
                    size=size,
                    bucket=bucket,
                    context=context,
                    admitted_ns=admitted_ns,
                    remaining_seconds=remaining,
                    priority=priority,
                    work=work,
                    snapshot_valid_until_ns=snapshot_valid_until_ns,
                    retain_for_caller=retain_for_caller,
                    preparation=preparation,
                    release_cancellation=release_cancellation,
                )
        finally:
            with self._lock:
                reservation = self._preparing.pop(preparation, None)
                if reservation is not None:
                    self._input_bytes -= reservation[0]
                self._preparation_done.notify_all()

    def _commit_prepared(
        self,
        *,
        frozen,
        digest,
        size,
        bucket,
        context,
        admitted_ns,
        remaining_seconds,
        priority,
        work,
        snapshot_valid_until_ns,
        retain_for_caller,
        preparation,
        release_cancellation,
    ):
        now = self.now_ns()
        deadline_ns = admitted_ns + int(remaining_seconds * 1e9)
        if now >= deadline_ns:
            raise ControlError("DEADLINE_EXPIRED")
        submission = Submission(
            operation_id=str(uuid4()),
            parent_id=context.request_id,
            target_id=self.target.target_id,
            boot_id=self.boot_id,
            generation=self.generation,
            request_ref=str(uuid4()),
            digest=digest,
            clock_id=self.clock_id,
            accept_until_ns=now + self.target.retry_window_ms * 1000000,
            deadline_ns=deadline_ns,
            priority=self.target.default_priority if priority is None else priority,
            input_bytes=size,
            input_bucket=bucket,
            work=work,
            snapshot_valid_until_ns=snapshot_valid_until_ns,
        )
        # Execute is the longest request verb. Reserve the largest receipt too:
        # error is capped at 256 characters, each escaped as at most 12 bytes,
        # and boot-clock timestamps fit signed 64-bit integers. Check actual
        # envelopes so accepted software priorities remain deliverable at exit.
        record = LocalOperation(
            submission, frozen, context, result_waiter=retain_for_caller, release_cancellation=release_cancellation
        )
        encode_control({"verb": "execute", "epoch": self.epoch, "submission": submission.to_dict()})
        encode_control(
            {
                "verb": "terminal_report",
                "epoch": self.epoch,
                "report": {
                    **record.report(),
                    "state": "not_started",
                    "error": "\U0010ffff" * 256,
                    "started_ns": 2**63 - 1,
                    "finished_ns": 2**63 - 1,
                },
            }
        )
        self._records[submission.operation_id] = record
        self._pending += 1
        self._input_bytes += size - self._preparing.pop(preparation)[0]
        trace.event(
            "runtime_operation_prepared",
            request_id=submission.operation_id,
            parent_id=submission.parent_id,
            target_id=submission.target_id,
        )
        return submission

    def _validate(self, submission, epoch, *, window=True, allow_past_generation=False):
        if window:
            submission.validate_window(self.now_ns(), self.target.retry_window_ms)
        if epoch != self.epoch or not epoch:
            raise ControlError("EPOCH_MISMATCH")
        record = self._records.get(submission.operation_id)
        if (submission.target_id, submission.boot_id, submission.clock_id) != (
            self.target.target_id,
            self.boot_id,
            self.clock_id,
        ):
            raise ControlError("RUNTIME_IDENTITY_MISMATCH")
        if (
            record is None
            and submission.generation != self.generation
            and not (allow_past_generation and submission.generation < self.generation)
        ):
            raise ControlError("RUNTIME_GENERATION_MISMATCH")
        if record is not None and record.submission != submission:
            raise ControlError("OPERATION_CONFLICT")
        return record

    def execute(self, submission, epoch, *, preflight=None):
        with self._lock:
            record = self._validate(submission, epoch)
            if record is None:
                # Validation includes the retry window: a completed record
                # cannot have been collected while this reference is valid.
                raise NotStartedControlError("REQUEST_REF_EXPIRED")
            if record.state != "prepared":
                return record.report()
            if preflight is not None:
                try:
                    preflight()
                except ControlError as exc:
                    # Under the claim lock, only a still-prepared operation
                    # can supply this evidence. Replays retain their outcome.
                    raise NotStartedControlError(exc.code) from exc
            if not self.target.accepts_priority(submission.priority):
                self._finish(record, "not_started", error="UNSUPPORTED_PRIORITY")
                return record.report()
            if submission.generation != self.generation:
                self._finish(record, "not_started", error="RUNTIME_GENERATION_MISMATCH")
                return record.report()
            if self._gate != "ready" or self.now_ns() >= submission.latest_start_ns:
                self._finish(
                    record,
                    "not_started",
                    error="SNAPSHOT_OR_DEADLINE_EXPIRED" if self._gate == "ready" else "RUNTIME_FENCED",
                )
                return record.report()
            if self._running >= self.target.max_in_flight:
                self._finish(record, "not_started", error="INSTANCE_CAPACITY_MISMATCH")
                return record.report()
            try:
                self._workers.submit(self._run, record)
            except Exception:
                # submit can enqueue before thread creation fails. The worker
                # must acquire this same lock before it can claim the record.
                self._finish(record, "not_started", error="WORKER_SUBMIT_FAILED")
                return record.report()
            if submission.work != "execute":
                for peer in self.lifecycle_peers:
                    with peer._lock:
                        peer._gate = "closing" if submission.work == "close" else "maintenance"
            record.state = "running"
            record.started_ns = self.now_ns()
            self._running += 1
            return record.report()

    def _run(self, record):
        with self._lock:
            if record.state != "running":
                return
        state, error, result, failure = "completed", "", None, None
        try:
            if self.now_ns() >= record.submission.latest_start_ns:
                with self._lock:
                    if record.submission.work in {"reset", "drain"}:
                        for peer in self.lifecycle_peers:
                            with peer._lock:
                                # No lifecycle call started. Undo only our own
                                # maintenance gate, never Close or recovery.
                                if peer._gate == "maintenance":
                                    peer._gate = "ready"
                    self._running -= 1
                    self._finish(record, "not_started", error="SNAPSHOT_OR_DEADLINE_EXPIRED")
                return
            # Reuse Deadline's injected clock, anchored once to the boot clock.
            # Wall-clock adjustments after this point cannot change the budget.
            origin_ns, origin = self.now_ns(), datetime.now(timezone.utc)
            remaining = max(0, (record.submission.deadline_ns - origin_ns) / 1e9)
            context = ExecutionContext(
                record.submission.parent_id,
                Deadline(
                    origin + timedelta(seconds=remaining),
                    clock=lambda: origin + timedelta(microseconds=(self.now_ns() - origin_ns) // 1000),
                ),
                record.context.cancellation_token,
            )
            if record.submission.work == "execute":
                # Serial priorities order Global's queue; the native backend
                # keeps its ordinary execution channel, even without streams.
                backend_priority = 0 if self.target.resource_mode == "serial" else record.submission.priority
                request = ModelRequest(
                    record.request.inputs,
                    {**record.request.metadata, "priority": backend_priority},
                    record.request.identity_metadata_keys,
                )
                result = self.handle.execute(request, context, _dispatch_key=self._authority)
            elif record.submission.work == "reset":
                self.handle._reset_local(context, _dispatch_key=self._authority)
            elif record.submission.work == "close":
                self.handle._close_local(context, _dispatch_key=self._authority)
            elif record.submission.work == "drain":
                self._drain(context)
        except ExecutionFailure as exc:
            reset = record.submission.work == "reset" or (
                record.submission.work == "execute"
                and record.request.metadata.get(RUNTIME_HANDLE_OPERATION) == "reset_stream"
            )
            if reset and _completed_control_failure(exc):
                # The Handle finalized Reset before checking delivery. Keep
                # successful lifecycle evidence despite the departed waiter.
                if record.submission.work == "execute":
                    result = ModelResult({}, 0.0, exc.evidence)
            else:
                failure, error = exc, exc.code
                if exc.evidence.state is OutcomeState.NOT_STARTED:
                    state = "not_started"
                elif exc.evidence.outcome_known:
                    # STARTED describes how far execution got, not uncertainty.
                    # A synchronized backend can fail with a known final outcome.
                    state = "completed"
                else:
                    state = "unknown"
        except BaseException as exc:
            failure, error, state = exc, type(exc).__name__, "unknown"
        # Tracebacks retain executor frames and image/device buffers outside
        # the bounded request store. Keep structured failures, not live frames.
        pending = [failure] if failure is not None else []
        seen = set()
        while pending:
            exception = pending.pop()
            if id(exception) in seen:
                continue
            seen.add(id(exception))
            exception.__traceback__ = None
            pending.extend(
                value
                for value in (exception.__cause__, exception.__context__, getattr(exception, "cause", None))
                if isinstance(value, BaseException)
            )
        # Result traversal is proportional to the Python payload. Keep the
        # operation running (and shutdown ownership held), but free control RPCs.
        with self._lock:
            if state in TERMINAL and not record.result_waiter:
                result = failure = None
        result, failure, result_bytes, validation_error = self._prepare_result(result, failure)
        with self._lock:
            if record.submission.work == "reset" and state == "completed" and not error:
                for peer in self.lifecycle_peers:
                    with peer._lock:
                        peer.generation += 1
                        if peer._gate == "maintenance":
                            peer._gate = "ready"
            elif record.submission.work in {"reset", "drain"} and (
                state == "not_started" or (state == "completed" and not error)
            ):
                for peer in self.lifecycle_peers:
                    with peer._lock:
                        # Drain or proven NOT_STARTED preserves generation and only releases
                        # maintenance, never a concurrent Close or recovery.
                        if peer._gate == "maintenance":
                            peer._gate = "ready"
            elif record.submission.work != "execute" and (state != "completed" or error):
                for peer in self.lifecycle_peers:
                    with peer._lock:
                        if peer._gate != "closing":
                            peer._gate = "recovery"
            self._running -= 1
            self._finish(
                record,
                state,
                error=error,
                result=result,
                failure=failure,
                result_bytes=result_bytes,
                validation_error=validation_error,
            )

    def _prepare_result(self, result, failure):
        # Retain only structured replay data. Causes/args may own arbitrary
        # inputs or SDK objects even after all tracebacks have been cleared.
        error = ""
        try:
            if failure is not None:
                if isinstance(failure, ExecutionFailure):
                    failure = _copy_failure(failure)
                else:
                    failure = ExecutionFailure(
                        code=type(failure).__name__,
                        message=str(failure)[:1024] or type(failure).__name__,
                        evidence=OutcomeEvidence.started("backend", outcome_known=False),
                    )
            budget = self.target.max_retained_result_bytes
            result, result_bytes = retain_result(result, byte_limit=budget)
            if failure is not None:
                values, failure_bytes = retain_result(
                    (failure.code, failure.message, failure.details, failure.evidence, failure.recovery),
                    byte_limit=budget - result_bytes,
                )
                failure.code, failure.message, failure.details, failure.evidence, failure.recovery = values
                result_bytes += failure_bytes
        except ControlError as exc:
            result, failure, result_bytes, error = None, None, 0, exc.code
        except Exception:
            # Result inspection is host validation after execution returned.
            # It cannot erase the backend's known completion or strand done.
            result, failure, result_bytes, error = None, None, 0, "INVALID_RESULT"
        return result, failure, result_bytes, error

    def _finish(self, record, state, *, error="", result=None, failure=None, result_bytes=0, validation_error=""):
        if state in TERMINAL and not record.result_waiter:
            result, failure, result_bytes = None, None, 0
        else:
            error = validation_error or error
        if self._result_bytes - record.result_bytes + result_bytes > self.target.max_retained_result_bytes:
            result, failure, result_bytes, error = None, None, 0, "RESULT_CAPACITY"
        if state in TERMINAL and record.state not in TERMINAL:
            self._pending -= 1
        if state in TERMINAL:
            if record.release_cancellation is not None:
                record.release_cancellation()
                record.release_cancellation = None
            record.context = None
        self._result_bytes += result_bytes - record.result_bytes
        record.result_bytes = result_bytes
        record.state, record.error, record.result, record.failure = state, error[:256], result, failure
        # A recovery drain can settle a Close outcome without releasing its
        # resources. Snapshot actual release separately from terminal certainty.
        if record.submission.work == "close":
            record.close_complete = state == "completed" and self.handle.closed
        record.finished_ns = self.now_ns()
        if state in TERMINAL and record.request is not None:
            release = getattr(self.handle, "release_request", None)
            if callable(release):
                release(record.request)
            self._input_bytes -= record.submission.input_bytes
            record.request = None
        if state == "unknown" and self._gate != "closing":
            self._gate = "recovery"
        if not record.done.done():
            record.done.set_result(None)
        self._preparation_done.notify_all()
        if state in TERMINAL and not record.acknowledged:
            key = record.submission.operation_id
            if key not in self._terminal_pending:
                self._terminal_pending[key] = record
                # Fresh evidence must not wait behind a full replay backlog.
                # Retries continue rotating in the same bounded queue.
                self._terminal_pending.move_to_end(key, last=False)
        if state in TERMINAL and self.on_terminal is not None:
            self.on_terminal()

    def query(self, submission, epoch):
        with self._lock:
            record = self._validate(submission, epoch, window=False)
            if record is None:
                raise ControlError("REQUEST_REF_EXPIRED")
            return record.report()

    def cancel(self, submission, epoch):
        with self._lock:
            # Existing uncertain operations remain controllable after expiry.
            record = self._validate(submission, epoch, window=submission.operation_id not in self._records)
            if record is None:
                self.collect()
                if len(self._records) + len(self._preparing) >= self.target.max_records:
                    raise ControlError("REQUEST_CAPACITY")
                record = LocalOperation(submission, None, ExecutionContext(submission.operation_id))
                self._records[submission.operation_id] = record
                self._pending += 1
            if record.state == "prepared":
                record.context.cancellation_token.cancel("scheduler cancellation")
                self._finish(record, "not_started", error="CANCELLED")
            elif record.state not in TERMINAL:
                record.context.cancellation_token.cancel("scheduler cancellation")
            return record.report()

    def completion(self, submission):
        """Local outcome notification; terminal evidence still belongs to Global."""
        with self._lock:
            record = self._validate(submission, self.epoch, window=False)
            if record is None:
                raise ControlError("REQUEST_REF_EXPIRED")
            return record.done

    def result(self, submission):
        with self._lock:
            record = self._records.get(submission.operation_id)
            if record is None:
                raise ControlError("RESULT_EXPIRED")
            failure = record.failure
            if failure is None:
                if record.state not in TERMINAL:
                    raise ControlError("OUTCOME_UNKNOWN" if record.state == "unknown" else "RESULT_PENDING")
                if record.error:
                    error_type = NotStartedControlError if record.state == "not_started" else CompletedControlError
                    raise error_type(record.error)
                if record.result is None and submission.work == "execute":
                    raise ControlError("RESULT_EXPIRED")
                result = record.result
                record.result_waiter = False
                self._release_payload(record)
                return result
        # Keep a stable failure reference if ACK/abandon releases the record.
        # Copy diagnostics outside the control lock, retaining fresh exceptions
        # so caller traceback frames never attach to the stored evidence.
        raise _copy_failure(failure) from None

    def terminal_receipt(self, submission):
        with self._lock:
            return self._validate(submission, self.epoch, window=False).receipt

    def acknowledge(self, submission, epoch, *, outcome=None):
        with self._lock:
            # ACK is an idempotent cleanup operation, including after reset
            # advances the generation and the terminal record is collected.
            # Epoch, boot, target, clock and extant-record identity still apply.
            record = self._validate(submission, epoch, window=False, allow_past_generation=True)
            if record is not None and record.state in TERMINAL:
                record.acknowledged = True
                if not record.receipt.done():
                    record.receipt.set_result(outcome or record.report())
                self._terminal_pending.pop(submission.operation_id, None)
                self._schedule_collection(record)

    def pending_terminal_reports(self, *, limit=None, retry_interval_ns=0):
        """Rotate pending evidence, examining at most limit records, including backoff."""
        with self._lock:
            count = len(self._terminal_pending) if limit is None else min(limit, len(self._terminal_pending))
            reports = []
            now = self.now_ns()
            for _ in range(count):
                key = next(iter(self._terminal_pending))
                record = self._terminal_pending[key]
                self._terminal_pending.move_to_end(key)
                if retry_interval_ns and now < record.terminal_retry_after_ns:
                    continue
                record.terminal_retry_after_ns = now + retry_interval_ns
                reports.append(record.report())
            return reports

    def release_result_waiter(self, submission):
        """Release business ownership after consumption or waiter abandonment."""
        with self._lock:
            record = self._records.get(submission.operation_id)
            if record is not None:
                record.result_waiter = False
                if record.state in TERMINAL:
                    self._release_payload(record)
            self.collect()

    def _release_payload(self, record):
        record.result = record.failure = None
        self._result_bytes -= record.result_bytes
        record.result_bytes = 0
        self._schedule_collection(record)

    def _schedule_collection(self, record):
        if (
            record.state in TERMINAL
            and record.acknowledged
            and not record.result_waiter
            and not record.collection_scheduled
        ):
            heappush(self._expirations, (record.submission.accept_until_ns, record.submission.operation_id))
            record.collection_scheduled = True

    def retains_request(self, metadata_key, value):
        """Whether execution ownership still pins this request's side payload."""
        with self._lock:
            return any(
                record.request is not None and record.request.metadata.get(metadata_key) == value
                for record in self._records.values()
            )

    def collect(self):
        with self._lock:
            now = self.now_ns()
            self._recovery_records = {
                key: value
                for key, value in self._recovery_records.items()
                if not value.acknowledged or not value.future.done() or value.retain_until_ns > now
            }
            if self._control_future is not None and self._control_future.done():
                self._control_future = None
            while self._expirations and self._expirations[0][0] <= now:
                _, key = heappop(self._expirations)
                self._result_bytes -= self._records[key].result_bytes
                del self._records[key]

    def _drain(self, context):
        # A returned Python call is not proof of accelerator completion. A
        # backend must explicitly implement trustworthy drain for UNKNOWN.
        drain = getattr(self.handle, "drain_managed", None)
        if not callable(drain):
            raise ControlError("BACKEND_DRAIN_UNSUPPORTED")
        drain(context, _dispatch_key=self._authority)

    def shutdown(self, context=None):
        context = context or ExecutionContext("host-shutdown", Deadline.after(self.target.request_timeout_ms / 1000))
        with self._lock:
            self._gate = "closing"
            for _, preparing_context in self._preparing.values():
                preparing_context.cancellation_token.cancel("host shutdown")
            for record in tuple(self._records.values()):
                if record.state not in TERMINAL:
                    record.context.cancellation_token.cancel("host shutdown")
                    self.cancel(record.submission, self.epoch)
            while (
                self._preparing
                or self._running
                or (self._control_future is not None and not self._control_future.done())
            ):
                # A timeout leaves the closing gate, workers and ownership
                # intact. Cancellation alone is not proof of native cleanup.
                context.check("host_shutdown")
                remaining = context.deadline.remaining_seconds()
                self._preparation_done.wait(0.05 if remaining is None else max(0, min(remaining, 0.05)))
        # No native invocation remains. Do not add an unbounded thread join
        # after the caller's bounded wait for preparation and execution.
        self._workers.shutdown(wait=False)
        self._recovery.shutdown(wait=False)

    def join_lifecycle(self, leader, peers):
        """Bind stage peers before any transport or execution is exposed."""
        self._lock = leader._lock
        self._preparation_done = threading.Condition(self._lock)
        self.boot_id = leader.boot_id
        self.lifecycle_peers = list(peers)

    def close_snapshot(self, previous_submission):
        """Return close decisions and operation identities, never mutable records."""
        with self._lock:
            records = tuple(record for peer in self.lifecycle_peers for record in peer._records.values())
            previous = previous_submission is not None and previous_submission.operation_id in self._records
            return (
                any(record.state not in TERMINAL for record in records)
                or any(peer._preparing for peer in self.lifecycle_peers),
                previous
                or any(record.state == "unknown" for record in records)
                or any(peer._gate == "recovery" for peer in self.lifecycle_peers),
                frozenset(self._records),
            )

    def find_close_submission(self, previous_ids, fallback):
        with self._lock:
            return next(
                (
                    record.submission
                    for key, record in self._records.items()
                    if key not in previous_ids and record.submission.work == "close"
                ),
                fallback,
            )

    def recover(
        self,
        epoch,
        operation_id,
        command,
        accept_until_ns,
        *,
        expected_generation=None,
        resume_until_ns=None,
        stream_id=None,
        acknowledged=False,
    ):
        uuid_value(operation_id)
        if type(acknowledged) is not bool:
            raise ValueError("recovery acknowledgement must be boolean")
        with self._lock:
            if epoch != self.epoch:
                raise ControlError("EPOCH_MISMATCH")
            old = self._recovery_records.get(operation_id)
            if old is not None and (command, accept_until_ns, expected_generation, stream_id) != (
                old.command,
                old.accept_until_ns,
                old.expected_generation,
                old.stream_id,
            ):
                raise ControlError("OPERATION_CONFLICT")
            if acknowledged:
                if old is not None:
                    if not old.future.done():
                        raise ControlError("RECOVERY_EVIDENCE_PENDING")
                    old.acknowledged = True
                    succeeded = old.future.result() is None or (command == "close" and self.handle.closed)
                    if succeeded and command in {"activate", "close"}:
                        # Global has stored this cleanup/activation result. Earlier
                        # finished commands are now superseded by this barrier;
                        # their old waiters cannot authorize another mutation.
                        for receipt in self._recovery_records.values():
                            if receipt is old:
                                break
                            if receipt.future.done():
                                receipt.acknowledged = True
                    settle = getattr(self.handle, "settle_recovery", None)
                    if succeeded and callable(settle):
                        settle(command, stream_id)
                return {"state": "acknowledged"}
            expiry = accept_until_ns if resume_until_ns is None else resume_until_ns
            if (
                type(expiry) is not int
                or not self.now_ns() < expiry <= self.now_ns() + self.target.retry_window_ms * 1000000
            ):
                raise ControlError("OPERATION_EXPIRED")
            if command not in {"drain", "close", "reset", "activate", "reset_stream", "close_stream"}:
                raise ControlError("INVALID_RECOVERY_COMMAND")
            if (command in {"reset_stream", "close_stream"}) != (isinstance(stream_id, str) and bool(stream_id)):
                raise ControlError("INVALID_RECOVERY_STREAM")
            if command not in {"reset_stream", "close_stream"} and stream_id is not None:
                raise ControlError("INVALID_RECOVERY_STREAM")
            if command == "activate" and (type(expected_generation) is not int or expected_generation < 1):
                raise ControlError("INVALID_RECOVERY_GENERATION")
            if self._gate == "closing" and command not in {"drain", "close"}:
                raise ControlError("CLOSE_IS_IRREVERSIBLE")
            old = self._recovery_records.get(operation_id)
            if resume_until_ns is not None and old is None:
                # Never replay a lifecycle mutation after losing its receipt.
                raise ControlError("RECOVERY_RECEIPT_UNAVAILABLE")
            if old is not None:
                old.retain_until_ns = max(old.retain_until_ns, expiry)
                future = old.future
                if future.done():
                    try:
                        failure = future.result()
                        if isinstance(failure, ExecutionFailure):
                            raise _copy_failure(failure)
                        if failure is not None:
                            raise type(failure)(failure.code)
                    except Exception as exc:
                        if (
                            command == "close"
                            and self.handle.closed
                            and isinstance(exc, ExecutionFailure)
                            and exc.evidence.outcome_known
                        ):
                            # Cleanup diagnostics do not erase positive release
                            # evidence. Keep the failure on the Handle/receipt.
                            return {"state": "completed", "generation": self.generation, "error": exc.code}
                        if not (
                            (command == "activate" and getattr(exc, "code", None) == "RECOVERY_EVIDENCE_PENDING")
                            or (command == "drain" and resume_until_ns is not None)
                            or (
                                command in {"reset", "reset_stream"}
                                and isinstance(exc, ExecutionFailure)
                                and exc.evidence.state is OutcomeState.NOT_STARTED
                                and exc.evidence.outcome_known
                            )
                            or (
                                command in {"close", "close_stream"}
                                and resume_until_ns is not None
                                and getattr(exc, "details", {}).get("close_pending") is True
                            )
                        ):
                            raise
                        # Drain is a repeatable evidence barrier, including
                        # after its local execution deadline. Activation may
                        # retry only its unmet evidence precondition. Close
                        # may continue explicitly unfinished cleanup: the owner
                        # retains release progress and never reopens admission.
                        # Stream Close follows the same unfinished-cleanup
                        # contract. Reset can retry proven NOT_STARTED only.
                        # A completed mutation or an unknown reply
                        # without that evidence is not rerun.
                        del self._recovery_records[operation_id]
                    else:
                        return {
                            "state": "completed",
                            "generation": self.generation,
                            "ready": self._gate == "ready" and self.handle.local_ready,
                        }
                else:
                    return {"state": "running"}
            if self._control_future is not None and not self._control_future.done():
                raise ControlError("RECOVERY_BUSY")
            self.collect()
            # Retained mutations cannot consume the final drain and Close/
            # activation slots. Repeating an existing receipt reuses its slot.
            reserve = 0 if command in {"close", "activate"} else 1 if command == "drain" else 2
            if old is None and len(self._recovery_records) >= MAX_HOST_RECOVERY_RECEIPTS - reserve:
                raise NotStartedControlError("RECOVERY_CAPACITY")
            # Register before submission. ThreadPoolExecutor may enqueue work
            # and then fail to create its first worker; the callback must claim
            # this same receipt under the controller lock before doing anything.
            future = Future()
            self._control_future = future
            self._recovery_records[operation_id] = RecoveryReceipt(
                command, accept_until_ns, future, expected_generation, stream_id, expiry
            )
            try:
                self._recovery.submit(self._run_recovery, future, command, expected_generation, stream_id)
            except Exception as exc:
                future.cancel()
                # The controller lock prevents the worker claiming the Future
                # before cancellation. Its queued callback is now harmless;
                # Global may replace this proven-unstarted command's ID.
                del self._recovery_records[operation_id]
                self._control_future = None
                raise NotStartedControlError("RECOVERY_WORKER_SUBMIT_FAILED") from exc
            if command != "activate" and self._gate != "closing":
                self._gate = "closing" if command == "close" else "recovery"
            return {"state": "running"}

    def _run_recovery(self, receipt, command, expected_generation, stream_id):
        with self._lock:
            if not receipt.set_running_or_notify_cancel():
                return
        failure = None
        try:
            self._recover(command, expected_generation, stream_id)
        except ControlError as exc:
            kind = (
                NotStartedControlError
                if exc.operation_started is False
                else CompletedControlError
                if exc.outcome_known
                else ControlError
            )
            failure = kind(exc.code[:256])
        except BaseException as exc:
            # Use the ordinary bounded diagnostic projection; never retain
            # backend exception frames, causes, payloads or arbitrary details.
            if isinstance(exc, ExecutionFailure):
                failure = self.policy.failure_diagnostic(exc)
            else:
                failure = ExecutionFailure(
                    code=str(getattr(exc, "code", type(exc).__name__))[:128],
                    message=str(exc)[:256] or type(exc).__name__,
                    evidence=OutcomeEvidence.started("recovery", outcome_known=False),
                )
        with self._lock:
            # Activation checks local evidence only; it submits no device work.
            if failure is not None and command != "activate":
                evidence = failure.evidence if isinstance(failure, ExecutionFailure) else failure
                if not evidence.outcome_known:
                    self._recovery_uncertain = True
            # Store a value, not Future.set_exception: every public replay
            # creates a fresh exception and cannot grow a retained traceback.
            receipt.set_result(failure)
            self._preparation_done.notify_all()

    def _clean_optional_unavailable(self):
        """A rolled-back optional load owns no backend resources to recover."""
        return (
            not self.target.required
            and isinstance(self.handle, UnavailableRuntime)
            and self.startup_clean
            and not self._running
            and not self._preparing
            and all(r.state in TERMINAL for r in self._records.values())
        )

    def _recover(self, command, expected_generation=None, stream_id=None):
        context = ExecutionContext(str(uuid4()), Deadline.after(self.target.request_timeout_ms / 1000))
        if command == "activate":
            with self._lock:
                if (
                    self._gate == "closing"
                    or self._recovery_uncertain
                    or self._preparing
                    or (isinstance(self.handle, UnavailableRuntime) and not self._clean_optional_unavailable())
                    or any(r.state not in TERMINAL or not r.acknowledged for r in self._records.values())
                ):
                    raise ControlError("RECOVERY_EVIDENCE_PENDING")
                # A new recovery operation may retry an activation whose reply
                # was lost. Only the coordinated next generation (or its replay)
                # is valid; never absorb an unrelated generation jump.
                if type(expected_generation) is not int or expected_generation not in {
                    self.generation,
                    self.generation + 1,
                }:
                    raise ControlError("UNCOORDINATED_GENERATION")
                self.generation = expected_generation
                # Drain proves device completion, not model reset. Release the
                # resource barrier while preserving this owner's reset gate.
                self._gate = "ready" if self.handle.local_ready or self._clean_optional_unavailable() else "recovery"
            return
        with self._lock:
            for record in self._records.values():
                if record.state not in TERMINAL and record.submission.work != "close":
                    # An already-running Close is an irreversible cleanup
                    # barrier. Drain waits for its evidence without canceling
                    # the owner's public waiter or the cleanup it is doing.
                    record.context.cancellation_token.cancel("runtime recovery")
        while True:
            with self._lock:
                if not self._running and not self._preparing:
                    # A running call can lose completion evidence while drain
                    # waits. Decide only after all Python invocations return.
                    unknown = self._recovery_uncertain or any(r.state == "unknown" for r in self._records.values())
                    break
            context.check("drain")
            time.sleep(0.005)
        if unknown:
            self._drain(context)
        with self._lock:
            self._recovery_uncertain = False
            for record in self._records.values():
                if record.state not in TERMINAL:
                    self._finish(record, "completed" if record.state == "unknown" else "not_started", error="RECOVERED")
            if self._clean_optional_unavailable():
                return
        if command == "reset":
            self.handle._reset_local(context, _dispatch_key=self._authority)
        elif command == "close":
            self.handle._close_local(context, _dispatch_key=self._authority)
        elif command in {"reset_stream", "close_stream"}:
            self.handle.recover_stream(command, stream_id, context, _dispatch_key=self._authority)


def retain_result(value, *, byte_limit=None):
    """Build a charged result tree with explicit, inspectable storage owners.

    Keep ordinary owned NumPy/Torch storage and count shared allocations once.
    Producers transfer that storage: they must not mutate it after returning,
    including on later calls or Close. A producer reusing an SDK output buffer
    must copy/clone it before returning; allocation ownership is not evidence
    that no mutable alias exists. Result charging does not enforce that promise.
    External NumPy buffers and non-resizable CPU Tensor storage are copied.
    Non-resizable device Tensor storage is rejected; resizability alone does
    not prove external ownership. Domain objects must return plain containers.
    """
    memo, storage_seen, active = {}, set(), set()
    total = 0

    def charge(amount):
        nonlocal total
        total += amount
        if byte_limit is not None and total > byte_limit:
            raise ControlError("RESULT_CAPACITY")

    def visit(item, depth=1):
        if item is None:
            return None
        if depth > 64:
            raise ControlError("RESULT_NESTING_TOO_DEEP")
        identity = id(item)
        if identity in active:
            raise ControlError("OPAQUE_RESULT")
        if identity in memo:
            return memo[identity]
        active.add(identity)
        kind = type(item)
        if kind in (dict, MappingProxyType):
            charge(EMPTY_MAPPING_BYTES + _MAPPING_ENTRY_BYTES * len(item))
            result = {visit(key, depth + 1): visit(child, depth + 1) for key, child in item.items()}
            if kind is MappingProxyType:
                result = MappingProxyType(result)
        elif kind in (tuple, list):
            charge(max(sys.getsizeof(item), _EMPTY_SEQUENCE_BYTES + _SEQUENCE_SLOT_BYTES * len(item)))
            children = [visit(child, depth + 1) for child in item]
            result = tuple(children) if kind is tuple else children
        elif kind in (ModelResult, OutcomeEvidence, RuntimeLatency, RecoveryRequirement):
            # Reconstruct only our value contracts; never retain incidental
            # attributes or an arbitrary domain dataclass's cached SDK payload.
            charge(128)
            result = replace(
                item, **{member.name: visit(getattr(item, member.name), depth + 1) for member in fields(item)}
            )
        elif kind.__module__.startswith("numpy"):
            import numpy as np

            if (kind is not np.ndarray and not isinstance(item, np.generic)) or _opaque_numpy_dtype(item.dtype):
                raise ControlError("OPAQUE_RESULT")
            charge(128)
            root = item
            while type(getattr(root, "base", None)) is np.ndarray:
                root = root.base
                if _opaque_numpy_dtype(root.dtype):
                    raise ControlError("OPAQUE_RESULT")
            if getattr(root, "base", None) is not None:
                # ORT capsules, Torch exports and other external CPU buffers
                # expose only the view extent. Copy that extent before retaining.
                charge(int(item.nbytes))
                result = np.array(item, copy=True, order="C")
            else:
                key = ("numpy", id(root))
                if key not in storage_seen:
                    storage_seen.add(key)
                    charge(int(root.nbytes))
                result = item
        elif kind.__module__.startswith("torch"):
            import torch

            if kind is not torch.Tensor or item.layout != torch.strided or item.is_quantized:
                raise ControlError("OPAQUE_RESULT")
            base = item
            while base is not None:
                if base.grad_fn is not None or base.grad is not None:
                    raise ControlError("RESULT_RETAINS_AUTOGRAD")
                if vars(base):
                    raise ControlError("OPAQUE_RESULT")
                base = base._base
            charge(128)
            storage = item.untyped_storage()
            if not storage.resizable():
                if item.device.type != "cpu":
                    raise ControlError("OPAQUE_RESULT_STORAGE")
                charge(item.numel() * item.element_size())
                result = item.detach().clone(memory_format=torch.contiguous_format)
            else:
                key = ("torch_storage", str(storage.device), storage.data_ptr())
                if key not in storage_seen:
                    storage_seen.add(key)
                    charge(storage.nbytes())
                result = item
        elif kind in (str, bytes):
            charge(_STRING_HEADER_BYTES + _STRING_CHARACTER_BYTES * len(item) if kind is str else 64 + len(item))
            result = item
        elif kind in (bytearray, int, float, bool, datetime, OutcomeState, RecoveryScope, RecoveryAction):
            charge(max(64, sys.getsizeof(item)))
            result = item.copy() if kind is bytearray else item
        else:
            raise ControlError("OPAQUE_RESULT")
        active.remove(identity)
        memo[identity] = result
        return result

    result = visit(value)
    return result, total


class UnavailableRuntime:
    """Evidence-only placeholder for an optional host which could not load."""

    ready = local_ready = False
    closed = False

    def bind_execution_policy(self, policy):
        return object()

    def _close_local(self, context, *, _dispatch_key):
        self.closed = True


class _ControllerPolicy:
    """Dispatch bridge; a host without a transport cannot admit business work."""

    def __init__(self, controller):
        self.controller = controller
        self.delegate = None

    def failure_diagnostic(self, failure):
        """Keep one bounded control summary; only the Controller owns payloads."""
        # Even a small full diagnostic could outlive its charged result record.
        # Never duplicate arbitrary details in the Handle's last-failure slot.
        return ExecutionFailure(
            code=failure.code[:128].strip() or "execution_failed",
            message=failure.message[:256].strip() or "Execution failed",
            recoverable=failure.recoverable,
            recovery=failure.recovery,
            evidence=replace(failure.evidence, phase=failure.evidence.phase[:64].strip() or "backend", details={}),
            details={"close_pending": failure.close_pending, "diagnostic_summary": True},
        )

    @property
    def ready(self):
        return self.controller.ready if self.delegate is None else self.delegate.ready

    def execute(self, request, context):
        if self.delegate is None:
            raise NotStartedControlError("dispatch_token_required")
        return self.delegate.execute(request, context)

    def reset(self, context):
        if self.delegate is None:
            raise NotStartedControlError("dispatch_token_required")
        return self.delegate.reset(context)

    def close(self, context):
        if self.delegate is not None:
            return self.delegate.close(context)
        self.controller.shutdown(context)
        self.controller.handle._close_local(context, _dispatch_key=self.controller._authority)

    def cancel(self, request_id, *, deadline=None):
        if self.delegate is not None:
            return self.delegate.cancel(request_id, deadline=deadline)
        self.controller.cancel_pending(request_id)
