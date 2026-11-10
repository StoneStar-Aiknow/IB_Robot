"""Global runtime ledger, shared by policy, stages and typed perception hosts."""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from collections.abc import Mapping
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from heapq import heappop, heappush
from uuid import uuid4

from inference_manifest.runtime_target import (
    DEFAULT_ADMISSION_SAFETY_MARGIN_MS,
    MAX_RETRY_WINDOW_MS,
    MAX_RUNTIME_GROUP_MEMBERS,
    validate_runtime_group_budget,
)

from .runtime_profiles import DEFAULT_MIN_SAMPLES, RuntimeProfileRegistry, validate_profile_options
from .runtime_protocol import (
    CLOSE_INPUT_RESERVE_BYTES,
    DEFAULT_STATUS_STALE_TIMEOUT_NS,
    HOST_HEARTBEAT_INTERVAL_NS,
    MAX_CONTROL_BYTES,
    TERMINAL,
    ControlError,
    RuntimeExecutionTarget,
    Submission,
    encode_control,
    identifier,
    machine_clock_id,
    uuid_value,
)

# Cover the maximum supported host retry window, including delayed Abort.
INITIALIZATION_RECEIPT_RETENTION_NS = MAX_RETRY_WINDOW_MS * 1000000
_REGISTRATION_FIELDS = (
    "targets",
    "_hosts",
    "_profiles",
    "_retired",
    "_unregistered",
    "_retired_boots",
    "_pending_activations",
)


@dataclass
class RuntimeOperation:
    submission: Submission
    sequence: int
    state: str = "queued"
    error: str = ""
    reserved_ns: int = 0
    finished_ns: int = 0
    estimated_finish_ns: int = 0
    host_acknowledged: bool = False
    cancel_requested: bool = False
    host_terminal_state: str = ""
    close_complete: bool = False

    def report(self):
        return {"state": self.state, "error": self.error, "epoch": ""}


class RuntimeScheduler:
    """Bounded priority queue; priority zero uses FIFO/EDF, other levels FIFO.

    The ledger is process-local. After losing it, deployment must stop all old
    Hosts and establish device quiescence before restarting Global and Hosts.
    Late old-epoch detection cannot prove that a newly started device is idle.
    """

    def __init__(
        self,
        targets,
        *,
        policy="fifo",
        stale_ns=DEFAULT_STATUS_STALE_TIMEOUT_NS,
        now_ns=time.monotonic_ns,
        resources=None,
        max_registered_targets=256,
        profile_min_samples=DEFAULT_MIN_SAMPLES,
        profile_max_age_days=30,
    ):
        if policy not in {"fifo", "edf"}:
            raise ValueError("invalid global policy")
        # The Host consumes a reply on the next heartbeat and dates freshness
        # from its send time. A healthy lease spans two timer intervals.
        if type(stale_ns) is not int or not 2 * HOST_HEARTBEAT_INTERVAL_NS < stale_ns <= 2**63 - 1:
            raise ValueError("status stale timeout must be an integer greater than two Host heartbeat intervals")
        validate_profile_options(profile_min_samples, profile_max_age_days)
        self.targets = {target.target_id: target for target in targets}
        if len(self.targets) != len(targets):
            raise ValueError("duplicate runtime target")
        if len({target.endpoint for target in targets}) != len(targets):
            raise ValueError("duplicate runtime host endpoint")
        self.policy = policy
        self.epoch = str(uuid4())
        self.clock_id = machine_clock_id()
        self.now_ns = now_ns
        self.stale_ns = stale_ns
        self._profile_options = {"min_samples": profile_min_samples, "max_age_days": profile_max_age_days}
        self._lock = threading.RLock()
        self._hosts = {}
        self._records = {}
        self._record_counts = Counter()
        # Replay history stays in _records; only unsettled work drives ticks.
        self._unsettled = {}
        self._expirations = []
        self._expiration_ids = set()
        self._sequence = 0
        self._fenced = set()
        self._pending_close_retirements = {}
        self._initializations = {}
        self._initialization_receipts = {}
        self._pending_activations = {}
        self._configured_targets = frozenset(self.targets)
        self._retired = {}
        self._unregistered = {}
        self._retired_boots = {}
        if type(max_registered_targets) is not int or max_registered_targets < max(1, len(targets)):
            raise ValueError("invalid runtime target capacity")
        # Includes configured targets and retired entries awaiting collection.
        self.max_registered_targets = max_registered_targets
        self._profiles = {}
        # Optional resource attestations validate fixed-target profile coverage.
        if resources is not None and (
            not isinstance(resources, Mapping) or any(not isinstance(value, Mapping) for value in resources.values())
        ):
            raise ValueError("runtime resources must be a mapping")
        self.resources = None if resources is None else {key: dict(value) for key, value in resources.items()}
        self.measurements = deque(maxlen=1024)
        self._measurement_sequence = 0
        self._draining = False
        physical = {}
        modes = {}
        for key, value in (self.resources or {}).items():
            identifier(key, "hardware resource")
            identifier(value.get("physical_device"), "physical device")
            device = value["physical_device"]
            if device in physical:
                raise ValueError("physical device aliases multiple resource IDs")
            physical[device] = key
            mode = value.get("mode", "serial")
            if mode not in {"serial", "ascend_concurrent"} or (
                mode == "ascend_concurrent" and value.get("verified_concurrency") is not True
            ):
                raise ValueError("unverified runtime resource mode")
            if value.get("predictive_admission", False):
                if mode != "serial" or policy != "fifo":
                    raise ValueError("prediction requires serial FIFO resources")
                if any(
                    not isinstance(value.get(name), str) or not value[name].strip()
                    for name in ("hardware_profile_fingerprint",)
                ):
                    raise ValueError("prediction requires hardware fingerprint")
            modes[key] = (device, mode)
        for target in targets:
            if self.resources is None and (target.predictive_admission or target.resource_mode == "ascend_concurrent"):
                raise ControlError("RESOURCE_ATTESTATION_REQUIRED")
            if self.resources is not None:
                self._validate_resource(target)
            previous = physical.setdefault(target.physical_device, target.hardware_resource_id)
            if previous != target.hardware_resource_id:
                raise ValueError("physical device aliases multiple resource IDs")
            mode = modes.setdefault(target.hardware_resource_id, (target.physical_device, target.resource_mode))
            if mode != (target.physical_device, target.resource_mode):
                raise ValueError("resource maps to conflicting devices/modes")
            if target.predictive_admission:
                if policy != "fifo":
                    raise ValueError("EDF is best effort; profile admission requires FIFO")
                self._profiles[target.target_id] = RuntimeProfileRegistry(target, **self._profile_options)
        predictive_resources = {t.hardware_resource_id for t in targets if t.predictive_admission}
        if any(t.hardware_resource_id in predictive_resources and not t.predictive_admission for t in targets):
            raise ValueError("prediction must cover every participant on the resource")

    def begin_shutdown(self):
        """Reject new model work while preserving Close and terminal evidence."""
        with self._lock:
            self._draining = True

    def shutdown_pending(self):
        with self._lock:
            return sorted(
                {name for name, (status, _) in self._hosts.items() if self._unregistered.get(name) != status["boot_id"]}
                | {f"initialization:{resource}" for resource in self._initializations}
            )

    def read_measurements(self, after_sequence=0):
        """Read a stable cursor in this Global epoch and report buffer overrun."""
        with self._lock:
            if type(after_sequence) is not int or not 0 <= after_sequence <= self._measurement_sequence:
                raise ControlError("INVALID_MEASUREMENT_CURSOR")
            oldest = self.measurements[0]["sequence"] if self.measurements else self._measurement_sequence + 1
            page = {
                "samples": [],
                "next_sequence": after_sequence,
                "oldest_sequence": oldest,
                "latest_sequence": self._measurement_sequence,
                "dropped": max(0, oldest - after_sequence - 1),
            }
            sample_bytes = 0
            for sample in self.measurements:
                if sample["sequence"] <= after_sequence:
                    continue
                # Use the wire encoder for escaping/UTF-8 accounting, but do
                # not serialize the entire growing page for every sample.
                size = len(encode_control(sample).encode()) + bool(page["samples"])
                envelope = {**page, "samples": [], "next_sequence": sample["sequence"]}
                if len(encode_control(envelope).encode()) + sample_bytes + size > MAX_CONTROL_BYTES:
                    if not page["samples"]:
                        raise ControlError("CONTROL_TOO_LARGE")
                    break
                page["samples"].append(sample)
                page["next_sequence"] = sample["sequence"]
                sample_bytes += size
            return page

    def _validate_resource(self, target):
        resource = self.resources.get(target.hardware_resource_id)
        if resource is None:
            raise ControlError("UNKNOWN_RESOURCE")
        if (
            target.physical_device != resource["physical_device"]
            or target.resource_mode != resource.get("mode", "serial")
            or target.hardware_profile_fingerprint != resource.get("hardware_profile_fingerprint", "")
            or target.predictive_admission != resource.get("predictive_admission", False)
            or target.admission_safety_margin_ms
            != resource.get("admission_safety_margin_ms", DEFAULT_ADMISSION_SAFETY_MARGIN_MS)
        ):
            raise ControlError("RESOURCE_CONFIGURATION_MISMATCH")
        if target.predictive_admission and resource.get("managed_coverage") is not True:
            # Only the central resource owner can attest that independently
            # started hosts account for all work using this physical device.
            raise ControlError("PREDICTIVE_COVERAGE_REQUIRED")

    @contextmanager
    def transaction(self):
        """Serialize compound control operations with membership publication."""
        with self._lock:
            yield

    def has_operation(self, operation_id, *, pending_only=False):
        with self._lock:
            record = self._records.get(operation_id)
            return record is not None and (not pending_only or record.state not in TERMINAL)

    def host_boot(self, target_id):
        with self._lock:
            observed = self._hosts.get(target_id)
            return observed[0]["boot_id"] if observed else None

    def targets_ready_reason(self, target_ids, boot_id=None):
        """Public serving view; callers never interpret mutable host evidence."""
        with self._lock:
            if not target_ids:
                return "runtime_missing"
            for target_id in target_ids:
                target = self.targets.get(target_id)
                observed = self._hosts.get(target_id)
                if target is None or observed is None or target_id in self._retired:
                    return "runtime_unregistered"
                status, status_ns = observed
                if self.now_ns() - status_ns > self.stale_ns:
                    return "runtime_stale"
                if status["epoch"] != self.epoch or not status["ready"]:
                    return "runtime_not_ready"
                if boot_id is not None and status["boot_id"] != boot_id:
                    return "runtime_boot_mismatch"
                if not self.resource_ready(target.hardware_resource_id):
                    return "runtime_resource_not_ready"
            return ""

    def has_active_target(self, target_ids):
        with self._lock:
            return any(target_id in self._hosts and not self._inactive(target_id) for target_id in target_ids)

    def unavailable_resources(self):
        with self._lock:
            resources = set(self.resources or {}) | {target.hardware_resource_id for target in self.targets.values()}
            return tuple(resource for resource in resources if not self.resource_ready(resource))

    @property
    def ready(self):
        with self._lock:
            return not self._draining and not self.unavailable_resources()

    def _validate_status(self, status):
        uuid_value(status["boot_id"])
        uuid_value(status["clock_id"])
        if not isinstance(status["epoch"], str):
            raise ControlError("INVALID_HOST_STATUS")
        if status["epoch"]:
            uuid_value(status["epoch"])
        for key in ("generation", "sequence"):
            if type(status[key]) is not int or status[key] < 1:
                raise ControlError("INVALID_HOST_STATUS")
        if type(status["ready"]) is not bool or type(status["clean"]) is not bool:
            raise ControlError("INVALID_HOST_STATUS")
        sent = status.get("status_ns")
        if type(sent) is not int or not 0 <= self.now_ns() - sent <= self.stale_ns:
            raise ControlError("STALE_STATUS")

    def _prepare_profiles(self, statuses):
        """Read bounded profile files before entering a registration transaction."""
        pending = []
        with self._lock:
            for status in statuses:
                target = RuntimeExecutionTarget(**status["target"])
                if self.resources is not None:
                    self._validate_resource(target)
                previous = self._hosts.get(target.target_id)
                if (
                    self.resources is not None
                    and target.predictive_admission
                    and (
                        target.target_id not in self._profiles
                        or (previous is not None and previous[0]["boot_id"] != status["boot_id"])
                    )
                ):
                    pending.append(target)
        profiles = {}
        for target in pending:
            try:
                profiles[target.target_id] = RuntimeProfileRegistry(target, **self._profile_options)
            except (OSError, TypeError, ValueError) as exc:
                raise ControlError("INVALID_RUNTIME_PROFILE") from exc
        return profiles

    def register(self, status, *, _prepared_profiles=None):
        profiles = self._prepare_profiles((status,)) if _prepared_profiles is None else _prepared_profiles
        with self._lock:
            if self.resources is not None and status["target"].get("execution_kind") == "stage":
                raise ControlError("GROUP_REGISTRATION_REQUIRED")
            candidate = self._registration_candidate((status,))
            try:
                result = candidate._register_one(status, profiles=profiles)
            except Exception:
                # Invalid restart evidence still fences the physical resource,
                # but cannot publish a partial membership or collect its queue.
                self._fenced.update(candidate._fenced)
                raise
            self._publish_registration(candidate)
            return result

    def resource_ready(self, resource, *, closing_target=None, admitted=False):
        if (
            resource in self._fenced
            or resource in self._initializations
            or (self._draining and closing_target is None and not admitted)
        ):
            return False
        for target in self.targets.values():
            if target.hardware_resource_id != resource:
                continue
            if target.template_only or target.target_id in self._retired:
                continue
            observed = self._hosts.get(target.target_id)
            required_at_startup = target.required and target.target_id in self._configured_targets
            optional_dynamic = (
                not required_at_startup
                and self.resources is not None
                and (target.interface == "tensor_model" or target.target_id not in self._configured_targets)
            )
            if observed is None:
                if optional_dynamic:
                    continue
                return False
            if optional_dynamic and self._inactive(target.target_id):
                continue
            status, status_ns = observed
            if status["epoch"] != self.epoch or self.now_ns() - status_ns > self.stale_ns:
                return False
            if (
                target.target_id != closing_target
                and not status["ready"]
                and (required_at_startup or not status["clean"])
            ):
                return False
        return True

    def submit(self, submission):
        return self._submit(submission)

    def _submit(self, submission, *, cancelled=False):
        with self._lock:
            target = self.targets.get(submission.target_id)
            if target is None:
                raise ControlError("UNKNOWN_TARGET")
            if not target.accepts_priority(submission.priority):
                raise ControlError("UNSUPPORTED_PRIORITY")
            submission.validate_window(self.now_ns(), target.retry_window_ms)
            record = self._records.get(submission.operation_id)
            if record is not None:
                self._match(record, submission)
                return self._report(record)
            if self._draining and submission.work == "execute":
                raise ControlError("GLOBAL_SHUTTING_DOWN")
            observed = self._hosts.get(submission.target_id)
            if observed is None:
                raise ControlError("RUNTIME_NOT_REGISTERED")
            if submission.target_id in self._retired or self.now_ns() - observed[1] > self.stale_ns:
                raise ControlError("RUNTIME_NOT_REGISTERED")
            status = observed[0]
            if (submission.boot_id, submission.generation, submission.clock_id) != (
                status["boot_id"],
                status["generation"],
                self.clock_id,
            ):
                raise ControlError("RUNTIME_IDENTITY_MISMATCH")
            if target.hardware_resource_id in self._initializations:
                raise ControlError("RESOURCE_INITIALIZING")
            closing = target.target_id if submission.work == "close" else None
            if (not status["ready"] and closing is None) or not self.resource_ready(
                target.hardware_resource_id, closing_target=closing
            ):
                raise ControlError("RESOURCE_NOT_READY")
            if any(
                r.submission.work != "execute"
                and r.state not in TERMINAL
                and self.targets[r.submission.target_id].lifecycle_group == target.lifecycle_group
                for r in self._unsettled.values()
            ):
                raise ControlError("LIFECYCLE_BARRIER")
            self.collect()
            if self._record_counts[submission.target_id] >= target.max_records + (submission.work == "close") or sum(
                r.submission.target_id == submission.target_id and r.state not in TERMINAL
                for r in self._unsettled.values()
            ) >= target.max_pending_requests + (submission.work == "close"):
                raise ControlError("REQUEST_CAPACITY")
            input_limit = target.max_retained_input_bytes + (
                CLOSE_INPUT_RESERVE_BYTES if submission.work == "close" else 0
            )
            if submission.input_bytes > input_limit:
                raise ControlError("INPUT_CAPACITY")
            estimated_finish_ns = 0
            if not cancelled and target.predictive_admission and submission.work != "close":
                # Every participant needs cost evidence: even low-priority work
                # blocks priority zero once dispatched on a serial resource.
                estimated_duration_ns = self._profiles[target.target_id].estimate(submission)
                queue_tail_ns = self.now_ns()
                previous_records = (
                    sorted(
                        (
                            r
                            for r in self._unsettled.values()
                            if r.state not in TERMINAL
                            and self.targets[r.submission.target_id].hardware_resource_id == target.hardware_resource_id
                            and (r.state != "queued" or r.submission.priority <= submission.priority)
                            and not (
                                submission.work != "execute"
                                and r.state == "queued"
                                and self.targets[r.submission.target_id].lifecycle_group == target.lifecycle_group
                            )
                        ),
                        key=lambda r: (r.state == "queued", r.sequence),
                    )
                    if submission.priority == 0
                    else ()
                )
                for previous in previous_records:
                    if not previous.estimated_finish_ns or (
                        previous.state != "queued" and previous.estimated_finish_ns <= self.now_ns()
                    ):
                        raise ControlError("PREDICTION_UNAVAILABLE")
                    if previous.state == "queued":
                        queue_tail_ns += self._profiles[previous.submission.target_id].estimate(previous.submission)
                    else:
                        queue_tail_ns = max(queue_tail_ns, previous.estimated_finish_ns)
                estimated_finish_ns = queue_tail_ns + estimated_duration_ns
                if (
                    submission.priority == 0
                    and estimated_finish_ns + target.admission_safety_margin_ms * 1000000 > submission.deadline_ns
                ):
                    raise ControlError("NO_FEASIBLE_DEADLINE")
            self._sequence += 1
            record = RuntimeOperation(
                submission,
                self._sequence,
                state="not_started" if cancelled else "queued",
                error="CANCELLED" if cancelled else "",
                cancel_requested=cancelled,
                estimated_finish_ns=estimated_finish_ns,
            )
            self._records[submission.operation_id] = record
            self._record_counts[submission.target_id] += 1
            self._unsettled[submission.operation_id] = record
            if not cancelled and submission.work != "execute":
                for previous in self._unsettled.values():
                    if (
                        previous is not record
                        and previous.state == "queued"
                        and self.targets[previous.submission.target_id].lifecycle_group == target.lifecycle_group
                    ):
                        previous.state, previous.error = "not_started", "LIFECYCLE_BARRIER"
            return self._report(record)

    def dispatchable(self):
        """Atomically record ownership BEFORE returning a command to the transport."""
        with self._lock:
            now = self.now_ns()
            for record in self._unsettled.values():
                if record.state == "queued" and now >= record.submission.latest_start_ns:
                    record.state, record.error = "not_started", "SNAPSHOT_OR_DEADLINE_EXPIRED"
                elif record.state in {"dispatching", "running"}:
                    observed = self._hosts.get(record.submission.target_id)
                    if observed is None or now - observed[1] > self.stale_ns:
                        record.state = "unknown"
                        self._fenced.add(self.targets[record.submission.target_id].hardware_resource_id)
            waiting = [r for r in self._unsettled.values() if r.state == "queued"]
            waiting.sort(
                key=lambda r: (
                    r.submission.priority,
                    r.submission.deadline_ns if self.policy == "edf" and r.submission.priority == 0 else r.sequence,
                    r.sequence,
                )
            )
            commands = []
            active = [r for r in self._unsettled.values() if r.state in {"dispatching", "running", "unknown"}]
            target_counts = Counter(r.submission.target_id for r in active)
            resource_counts = Counter(self.targets[r.submission.target_id].hardware_resource_id for r in active)
            maintenance = {
                self.targets[r.submission.target_id].hardware_resource_id
                for r in active
                if r.submission.work != "execute"
            }
            # Reserve draining resources before priority ordering can refill
            # them. This includes barriers whose own Target is at capacity.
            # Recompute from live queued records so cancellation/expiry releases
            # the reservation without creating another persistent fence.
            waiting_maintenance = {
                self.targets[r.submission.target_id].hardware_resource_id
                for r in waiting
                if r.submission.work != "execute"
            }
            for record in waiting:
                submission = record.submission
                target = self.targets[submission.target_id]
                if submission.work == "close" and target.hardware_resource_id in self._fenced:
                    # Close may have been admitted before a peer became UNKNOWN.
                    # This queued command was never sent: release it to recovery
                    # while the caller still has budget, preserving peer ownership.
                    record.state, record.error = "not_started", "RESOURCE_NOT_READY"
                    continue
                if submission.work == "execute" and target.hardware_resource_id in waiting_maintenance:
                    continue
                if target_counts[submission.target_id] >= target.max_in_flight:
                    continue
                same_resource = resource_counts[target.hardware_resource_id]
                if submission.work != "execute" and same_resource:
                    continue
                if target.hardware_resource_id in maintenance:
                    continue
                if target.resource_mode == "serial" and same_resource:
                    continue
                if not self.resource_ready(
                    target.hardware_resource_id,
                    closing_target=target.target_id if submission.work == "close" else None,
                    admitted=True,
                ):
                    continue
                if target.predictive_admission and submission.work != "close":
                    try:
                        finish = now + self._profiles[submission.target_id].estimate(submission)
                    except ControlError as exc:
                        # A profile may expire while this request is queued.
                        # Retain NOT_STARTED evidence and keep dispatching peers.
                        record.state, record.error = "not_started", exc.code
                        continue
                    if (
                        submission.priority == 0
                        and finish + target.admission_safety_margin_ms * 1000000 > submission.deadline_ns
                    ):
                        record.state, record.error = "not_started", "NO_FEASIBLE_DEADLINE"
                        continue
                    record.estimated_finish_ns = finish
                record.state, record.reserved_ns = "dispatching", now
                target_counts[submission.target_id] += 1
                resource_counts[target.hardware_resource_id] += 1
                if submission.work != "execute":
                    maintenance.add(target.hardware_resource_id)
                commands.append((target, submission))
            return commands

    def report(self, report):
        with self._lock:
            submission = Submission(**report["submission"])
            record = self._records.get(submission.operation_id)
            if record is None:
                raise ControlError("OPERATION_EXPIRED")
            self._match(record, submission)
            state = report["state"]
            if state not in {"prepared", "running", "unknown", *TERMINAL}:
                raise ControlError("INVALID_EXECUTION_STATE")
            if state == "prepared":
                return self._report(record)  # A query cannot revoke a sent Execute.
            if record.state in TERMINAL:
                if state == record.state:
                    record.host_terminal_state = state
                elif state in TERMINAL or not record.host_terminal_state:
                    record.state = "unknown"
                    record.host_acknowledged = False
                    self._unsettled[submission.operation_id] = record
                    self._fenced.add(self.targets[submission.target_id].hardware_resource_id)
                # Execute/Query/Cancel replies may arrive out of order. Only
                # host-confirmed terminal evidence supersedes older observations;
                # a scheduler-local NOT_STARTED cannot disprove host execution.
                return self._report(record)
            if record.state == "queued" and state != "not_started":
                raise ControlError("UNDISPATCHED_EXECUTION")
            if record.state == "unknown" and state == "running":
                return self._report(record)
            record.state, record.error = state, report.get("error", "")[:256]
            record.close_complete = state == "completed" and report.get("close_complete") is True
            if state in TERMINAL and not record.host_terminal_state:
                record.host_terminal_state = state
                record.finished_ns = self.now_ns()
                if state == "completed" and not record.error and record.reserved_ns:
                    target = self.targets[submission.target_id]
                    if (
                        self.resources is not None
                        and self.resources[target.hardware_resource_id].get("managed_coverage") is True
                        and target.resource_mode == "serial"
                        and submission.input_bucket
                    ):
                        self._measurement_sequence += 1
                        self.measurements.append(
                            {
                                "operation_id": submission.operation_id,
                                "target_id": target.target_id,
                                "identity": RuntimeProfileRegistry.identity(target),
                                "work": submission.work,
                                "input_bucket": submission.input_bucket,
                                "sequence": self._measurement_sequence,
                                "occupancy_ns": record.finished_ns - record.reserved_ns,
                                "measured_at": time.time(),
                            }
                        )
                if submission.work == "reset" and state == "completed" and not record.error:
                    group = self.targets[submission.target_id].lifecycle_group
                    for target_id, (status, observed) in tuple(self._hosts.items()):
                        if self.targets[target_id].lifecycle_group == group:
                            self._hosts[target_id] = ({**status, "generation": status["generation"] + 1}, observed)
            resource = self.targets[submission.target_id].hardware_resource_id
            if state == "unknown":
                self._fenced.add(resource)
            return self._report(record)

    def cancel(self, submission):
        with self._lock:
            record = self._records.get(submission.operation_id)
            if record is None:
                return self._submit(submission, cancelled=True)
            self._match(record, submission)
            record.cancel_requested = True
            if record.state == "queued":
                record.state, record.error = "not_started", "CANCELLED"
            return self._report(record)

    def query(self, submission):
        with self._lock:
            record = self._records.get(submission.operation_id)
            if record is None:
                raise ControlError("OPERATION_EXPIRED")
            self._match(record, submission)
            return self._report(record)

    def pending_reports(self):
        with self._lock:
            commands = []
            for record in self._unsettled.values():
                if record.state == "queued" or (record.state in TERMINAL and record.host_acknowledged):
                    continue
                if record.state in TERMINAL and record.host_terminal_state == record.state:
                    # Retry ACK independently: the business caller may have
                    # consumed the result and the host may have collected it.
                    verb = "ack"
                elif (
                    record.state in TERMINAL
                    or record.cancel_requested
                    or self.now_ns() >= record.submission.deadline_ns
                ):
                    verb = "cancel"
                else:
                    verb = "query"
                commands.append((self.targets[record.submission.target_id], record.submission, verb))
            return commands

    def acknowledge_host(self, submission):
        with self._lock:
            record = self._records.get(submission.operation_id)
            if record is not None:
                self._match(record, submission)
                if record.state in TERMINAL and record.host_terminal_state == record.state:
                    if submission.operation_id not in self._expiration_ids:
                        # Each record enters the expiry index once, when Global
                        # has the host's terminal ACK. UNKNOWN is never indexed.
                        heappush(
                            self._expirations,
                            (max(submission.accept_until_ns, submission.deadline_ns), submission.operation_id),
                        )
                        self._expiration_ids.add(submission.operation_id)
                    first_ack = not record.host_acknowledged
                    record.host_acknowledged = True
                    self._unsettled.pop(submission.operation_id, None)
                    # Diagnostics require explicit Host release evidence;
                    # a known error alone cannot authorize retirement.
                    if (
                        first_ack
                        and submission.work == "close"
                        and record.state == "completed"
                        and (not record.error or record.close_complete)
                    ):
                        self._pending_close_retirements[submission.operation_id] = submission
                    self._retire_completed_closes()

    def mark_transport_unknown(self, submission):
        with self._lock:
            record = self._records.get(submission.operation_id)
            # Terminal, acknowledged records may expire while an older Query
            # or Cancel is still in transport. Its failure cannot revoke the
            # evidence that allowed collection, or recreate the operation.
            if record is None:
                return
            self._match(record, submission)
            if record.state not in TERMINAL:
                record.state = "unknown"
                self._fenced.add(self.targets[submission.target_id].hardware_resource_id)

    def reject_unsent(self, submission, error):
        """Execute was unsent or explicitly rejected before start; release its reservation."""
        with self._lock:
            record = self._records[submission.operation_id]
            self._match(record, submission)
            if record.state != "dispatching":
                raise ControlError("INVALID_UNSENT_STATE")
            record.state, record.error = "not_started", error
            # The normal Cancel/ACK exchange must still retire the prepared
            # host snapshot. Local rejection is not a forged host receipt.

    def collect(self):
        with self._lock:
            self._collect_initialization_receipts()
            self._collect_retired_targets()
            now = self.now_ns()
            # ACK confirms execution, not consumption: index expiry through
            # both the retry window and business deadline, without scanning history.
            while self._expirations and self._expirations[0][0] <= now:
                _, key = heappop(self._expirations)
                self._expiration_ids.remove(key)
                record = self._records[key]
                # Conflicting late evidence can revoke a previous terminal ACK.
                # Retain it until reconciliation provides a new confirmed terminal.
                if record.state not in TERMINAL or not record.host_acknowledged:
                    continue
                target_id = self._records[key].submission.target_id
                self._record_counts[target_id] -= 1
                if not self._record_counts[target_id]:
                    del self._record_counts[target_id]
                del self._records[key]

    def _report(self, record):
        return {**record.report(), "epoch": self.epoch}

    @staticmethod
    def _match(record, submission):
        if record.submission != submission:
            raise ControlError("OPERATION_CONFLICT")

    def _collect_initialization_receipts(self):
        now = self.now_ns()
        self._initialization_receipts = {
            key: receipt
            for key, receipt in self._initialization_receipts.items()
            if receipt["state"] in {"initializing", "unknown"}
            or (receipt["state"] == "initialized" and not receipt.get("acknowledged", False))
            or receipt["expires_ns"] > now
        }

    def _initialization_reply(self, resource, receipt):
        return {"epoch": self.epoch, "state": receipt["state"], "resource": resource}

    def discover_target(self, resource, deployment, composition, configured_target_id=None, configured_target_ids=()):
        """Select exact catalogue target IDs, e.g. perception/depth_front, not service IDs."""
        if not isinstance(configured_target_ids, list | tuple) or any(
            not isinstance(item, str) or not item for item in configured_target_ids
        ):
            raise ControlError("INVALID_CONFIGURED_TARGET_IDS")
        with self._lock:
            matches = [
                target
                for name, target in self.targets.items()
                if name in self._configured_targets
                and target.hardware_resource_id == resource
                and target.deployment_fingerprint == deployment
                and target.composition_fingerprint == composition
                and (configured_target_id is None or name == configured_target_id)
                and (not configured_target_ids or name in configured_target_ids)
            ]
            if len(matches) > 1:
                # Identical templates can be shared; different per-instance
                # policies require an explicit configuration identity.
                policies = [
                    {k: v for k, v in t.to_dict().items() if k not in {"target_id", "endpoint", "lifecycle_group"}}
                    for t in matches
                ]
                if any(policy != policies[0] for policy in policies[1:]):
                    raise ControlError("RUNTIME_TARGET_AMBIGUOUS")
            if not matches:
                if configured_target_id is not None or configured_target_ids:
                    raise ControlError("RUNTIME_TARGET_NOT_FOUND")
                return None
            target = matches[0]
            if target.predictive_admission:
                # Recheck age and identity at startup, before any SDK constructor.
                self._profiles[target.target_id].check_freshness()
            return target.to_dict()

    def _check_initialization_admission(self, statuses, token, accept_until_ns):
        """Cheap replay and occupancy checks; caller holds the resource lock."""
        targets = [RuntimeExecutionTarget(**status["target"]) for status in statuses]
        validate_runtime_group_budget(targets)
        resources = {target.hardware_resource_id for target in targets}
        if len(resources) != 1:
            raise ControlError("INVALID_INITIALIZATION_GROUP")
        resource = resources.pop()
        identity = (token, tuple((s["target"]["target_id"], s["boot_id"]) for s in statuses))
        self._collect_initialization_receipts()
        previous = self._initialization_receipts.get((resource, token))
        if (
            previous is None
            and accept_until_ns is not None
            and (
                type(accept_until_ns) is not int
                or accept_until_ns > self.now_ns() + INITIALIZATION_RECEIPT_RETENTION_NS
            )
        ):
            raise ControlError("INITIALIZATION_ADMISSION_TIMEOUT")
        if previous is not None:
            if previous["state"] == "aborted":
                raise ControlError("INITIALIZATION_CANCELED")
            if previous["identity"] != identity or previous["descriptors"] != tuple(
                (target, status["clock_id"]) for target, status in zip(targets, statuses, strict=True)
            ):
                raise ControlError("INITIALIZATION_IDENTITY_MISMATCH")
            if previous["state"] != "initializing":
                raise ControlError("INITIALIZATION_ALREADY_FINISHED")
            if accept_until_ns is None:
                accept_until_ns = previous["accept_until_ns"]
            elif accept_until_ns != previous["accept_until_ns"]:
                raise ControlError("INITIALIZATION_IDENTITY_MISMATCH")
        now = self.now_ns()
        if accept_until_ns is None:
            accept_until_ns = now + min(t.retry_window_ms for t in targets) * 1000000
        if type(accept_until_ns) is not int or accept_until_ns <= now:
            raise ControlError("INITIALIZATION_ADMISSION_TIMEOUT")
        active = self._initializations.get(resource)
        if active is not None:
            if active == identity:
                return resource, identity, accept_until_ns, True
            raise ControlError("RESOURCE_INITIALIZING")
        if self._draining:
            raise ControlError("GLOBAL_SHUTTING_DOWN")
        if len(self._initialization_receipts) >= self.max_registered_targets * 2:
            raise ControlError("INITIALIZATION_CAPACITY")
        if resource in self._fenced:
            raise ControlError("RESOURCE_NOT_READY")
        for record in self._unsettled.values():
            if self.targets[record.submission.target_id].hardware_resource_id == resource and record.state in {
                "queued",
                "dispatching",
                "running",
                "unknown",
            }:
                raise ControlError("RESOURCE_BUSY")
        for target_id, (status, observed_ns) in self._hosts.items():
            if (
                self.targets[target_id].hardware_resource_id == resource
                and not status["clean"]
                and self.now_ns() - observed_ns > self.stale_ns
            ):
                self._fenced.add(resource)
                raise ControlError("RESOURCE_NOT_READY")
        return resource, identity, accept_until_ns, False

    def begin_initialization(self, statuses, token, *, accept_until_ns=None):
        """Reserve a device before construction can enter a backend SDK.

        Reuse registration validation and the resource ledger, but do not wait
        for every configured host to be ready: they may also be starting.
        Callers must pause same-device submissions and drain accepted work;
        rejected initialization attempts do not acquire a queue position.
        """
        uuid_value(token)
        with self._lock:
            resource, identity, accept_until_ns, replay = self._check_initialization_admission(
                statuses, token, accept_until_ns
            )
            if replay:
                return {"epoch": self.epoch, "state": "initializing", "resource": resource}
        profiles = self._prepare_profiles(statuses)
        with self._lock:
            resource, identity, accept_until_ns, replay = self._check_initialization_admission(
                statuses, token, accept_until_ns
            )
            if replay:
                return {"epoch": self.epoch, "state": "initializing", "resource": resource}
            now = self.now_ns()
            if len(statuses) == 1:
                self.register(statuses[0], _prepared_profiles=profiles)
            else:
                self.register_group(statuses, _prepared_profiles=profiles)
            self._initializations[resource] = identity
            self._initialization_receipts[(resource, token)] = {
                "identity": identity,
                "descriptors": tuple(
                    (RuntimeExecutionTarget(**status["target"]), status["clock_id"]) for status in statuses
                ),
                "accept_until_ns": accept_until_ns,
                "state": "initializing",
                "retention_ns": max(INITIALIZATION_RECEIPT_RETENTION_NS, accept_until_ns - now),
                "expires_ns": 0,
            }
            return {"epoch": self.epoch, "state": "initializing", "resource": resource}

    def check_initialization(self, resource, token):
        """Verify original cleanup authority without extending or reacquiring it."""
        with self._lock:
            active = self._initializations.get(resource)
            receipt = self._initialization_receipts.get((resource, token))
            if active is None or active[0] != token or receipt is None or receipt["state"] != "initializing":
                raise ControlError("INITIALIZATION_IDENTITY_MISMATCH")
            return self._initialization_reply(resource, receipt)

    def finish_initialization(self, resource, token, *, clean, retire=False, acknowledged=False):
        with self._lock:
            if (
                type(clean) is not bool
                or type(retire) is not bool
                or type(acknowledged) is not bool
                or (retire and not clean)
                or (acknowledged and not clean)
            ):
                raise ControlError("INVALID_INITIALIZATION_EVIDENCE")
            receipt = self._initialization_receipts.get((resource, token))
            if acknowledged:
                if receipt is not None:
                    if (
                        receipt["state"] != "initialized"
                        or receipt["clean"] is not clean
                        or receipt["retire"] is not retire
                    ):
                        raise ControlError("INVALID_INITIALIZATION_EVIDENCE")
                    receipt["acknowledged"] = True
                # A lost ACK reply remains replayable after collection. This
                # branch cannot finish or retire any active initialization.
                return {"epoch": self.epoch, "state": "acknowledged", "resource": resource}
            if receipt is not None and receipt["state"] != "initializing":
                if receipt.get("clean") is not clean or receipt.get("retire", False) is not retire:
                    raise ControlError("INVALID_INITIALIZATION_EVIDENCE")
                return self._initialization_reply(resource, receipt)
            active = self._initializations.get(resource)
            if active is None or active[0] != token:
                raise ControlError("INITIALIZATION_IDENTITY_MISMATCH")
            candidate = None
            if retire:
                candidate = self._registration_candidate()
                # Failed dynamic creation has no host to heartbeat or unregister.
                # Retire exactly the incarnation owned by this initialization,
                # using the same settlement checks as ordinary host Close.
                for target_id, boot_id in active[1]:
                    status = self._hosts[target_id][0]
                    if status["boot_id"] != boot_id:
                        raise ControlError("INITIALIZATION_IDENTITY_MISMATCH")
                    candidate.unregister(
                        {
                            **status,
                            "epoch": self.epoch,
                            "sequence": status["sequence"] + 1,
                            "status_ns": self.now_ns(),
                            "ready": False,
                            "clean": True,
                        }
                    )
            receipt.update(
                state="initialized" if clean else "unknown",
                clean=clean,
                retire=retire,
                expires_ns=max(receipt["accept_until_ns"], self.now_ns() + receipt["retention_ns"]),
            )
            if not clean:
                self._fenced.add(resource)
                # Keep the barrier: recovery of other hosts cannot prove that
                # this unbound initializer has stopped using the device.
                return self._initialization_reply(resource, receipt)
            del self._initializations[resource]
            if candidate is not None:
                self._publish_registration(candidate)
            return self._initialization_reply(resource, receipt)

    def abort_initialization(self, resource, token, *, accept_until_ns):
        """Record the caller's proof that construction never started.

        A tombstone also rejects a delayed Begin arriving after this Abort.
        Uncertain SDK work can only finish with explicit cleanup evidence.
        """
        uuid_value(token)
        if type(accept_until_ns) is not int or accept_until_ns <= 0:
            raise ControlError("INVALID_INITIALIZATION_EVIDENCE")
        with self._lock:
            self._collect_initialization_receipts()
            key = (resource, token)
            receipt = self._initialization_receipts.get(key)
            if receipt is not None:
                if receipt["accept_until_ns"] != accept_until_ns:
                    raise ControlError("INITIALIZATION_IDENTITY_MISMATCH")
                if receipt["state"] == "unknown":
                    raise ControlError("INITIALIZATION_EVIDENCE_PENDING")
                if receipt["state"] in {"initialized", "aborted"}:
                    return self._initialization_reply(resource, receipt)
                # No SDK work started. Reuse the boot-bound retirement barrier
                # before publishing the Abort tombstone, including fixed hosts.
                self.finish_initialization(resource, token, clean=True, retire=True)
            else:
                if resource not in (self.resources or {}) and not any(
                    target.hardware_resource_id == resource for target in self.targets.values()
                ):
                    raise ControlError("UNKNOWN_RESOURCE")
                if accept_until_ns > self.now_ns() + INITIALIZATION_RECEIPT_RETENTION_NS:
                    raise ControlError("INITIALIZATION_ADMISSION_TIMEOUT")
                if len(self._initialization_receipts) >= self.max_registered_targets * 2:
                    raise ControlError("INITIALIZATION_CAPACITY")
                receipt = {}
                self._initialization_receipts[key] = receipt
            receipt.update(
                state="aborted",
                accept_until_ns=accept_until_ns,
                expires_ns=max(accept_until_ns, self.now_ns() + INITIALIZATION_RECEIPT_RETENTION_NS),
            )
            return self._initialization_reply(resource, receipt)

    def host_retired(self, target_id, boot_id):
        """Only acknowledged retirement of this incarnation removes a peer."""
        with self._lock:
            return target_id in self._retired and self.host_boot(target_id) == boot_id

    def recovery_targets(self, resource):
        with self._lock:
            return tuple(
                target
                for target in self.targets.values()
                if target.hardware_resource_id == resource
                and not target.template_only
                and target.target_id not in self._retired
                and not (
                    self.resources is not None
                    and (target.interface == "tensor_model" or target.target_id not in self._configured_targets)
                    and (target.target_id not in self._hosts or self._inactive(target.target_id))
                )
            )

    def fence_resource(self, resource):
        with self._lock:
            self._fenced.add(resource)

    def begin_recovery(self, resource, targets):
        """Validate ownership before publishing the recovery barrier."""
        with self._lock:
            if resource in self._initializations:
                raise ControlError("INITIALIZATION_EVIDENCE_PENDING")
            identities = {target.target_id: self.host_boot(target.target_id) for target in targets}
            if any(boot is None for boot in identities.values()):
                raise ControlError("RECOVERY_HOST_UNREACHABLE")
            self._fenced.add(resource)
            for record in self._unsettled.values():
                if self.targets[record.submission.target_id].hardware_resource_id == resource:
                    if record.submission.work == "close" and record.state in {"dispatching", "running"}:
                        # Let an admitted irreversible Close finish. Canceling
                        # it would also cancel its public waiter's context.
                        continue
                    if record.state == "queued":
                        record.state, record.error = "not_started", "RECOVERY_BARRIER"
                    record.cancel_requested = True
            return identities

    def resource_settled(self, resource):
        with self._lock:
            return not any(
                self.targets[record.submission.target_id].hardware_resource_id == resource
                for record in self._unsettled.values()
            )

    def next_activation_generation(self, target_id, boot_id):
        with self._lock:
            if self.host_boot(target_id) != boot_id:
                raise ControlError("RECOVERY_IDENTITY_MISMATCH")
            return self._pending_activations.setdefault(
                (target_id, boot_id), self._hosts[target_id][0]["generation"] + 1
            )

    def complete_activation(self, target_id, boot_id, generation, expected, *, ready=True):
        with self._lock:
            if self.host_boot(target_id) != boot_id:
                raise ControlError("RECOVERY_IDENTITY_MISMATCH")
            key = (target_id, boot_id)
            if generation != expected or self._pending_activations.get(key) != expected:
                raise ControlError("UNCOORDINATED_GENERATION")
            status, observed = self._hosts[target_id]
            self._hosts[target_id] = ({**status, "generation": generation, "ready": ready}, observed)
            self._pending_activations.pop(key, None)

    def retire_recovered(self, targets, identities):
        """Publish successful Close only after every owner crossed the ACK barrier."""
        with self._lock:
            if any(
                self.host_boot(t.target_id) != identities[t.target_id] or not self._settled(t.target_id)
                for t in targets
            ):
                raise ControlError("RECOVERY_CLOSE_EVIDENCE_MISMATCH")
            now = self.now_ns()
            for target in targets:
                status, _ = self._hosts[target.target_id]
                self._hosts[target.target_id] = ({**status, "ready": False, "clean": True}, now)
                expiry = now + max(target.retry_window_ms * 1000000, self.stale_ns + 1)
                self._retired[target.target_id] = expiry
                self._retired_boots[(target.target_id, status["boot_id"])] = expiry
                self._pending_activations.pop((target.target_id, status["boot_id"]), None)

    def finish_recovery(self, resource):
        with self._lock:
            if resource in self._initializations:
                raise ControlError("INITIALIZATION_EVIDENCE_PENDING")
            if not self.resource_settled(resource):
                raise ControlError("RECOVERY_EVIDENCE_PENDING")
            self._fenced.discard(resource)

    def _settled(self, target_id):
        return not any(record.submission.target_id == target_id for record in self._unsettled.values())

    def _inactive(self, target_id):
        """Only clean, settled hosts can stop participating without recovery."""
        if any(target_id in {name for name, _ in members} for _, members in self._initializations.values()):
            return False
        observed = self._hosts.get(target_id)
        return target_id in self._retired or (
            observed is not None
            # A recovery can own device work outside the ordinary ledger.
            # While fenced, stale clean heartbeats cannot remove participants
            # or authorize replacing their boot. Only retirement proves exit.
            and self.targets[target_id].hardware_resource_id not in self._fenced
            and self.now_ns() - observed[1] > self.stale_ns
            and observed[0]["clean"]
            and self._settled(target_id)
        )

    def _registration_candidate(self, statuses=()):
        """Stage membership changes; unchanged heartbeats only copy host status.

        _register_one still validates freshness, epoch and retirement. With the
        same descriptor, boot and generation it cannot mutate the directory or
        ledger. Group heartbeats publish their status changes atomically too.
        """
        heartbeat = bool(statuses) and all(
            (previous := self._hosts.get(status["target"]["target_id"])) is not None
            and status["target"] == previous[0]["target"]
            and all(status[name] == previous[0][name] for name in ("boot_id", "generation"))
            and status["target"]["target_id"] not in self._retired
            and status["target"]["target_id"] not in self._unregistered
            and (not status["target"].get("predictive_admission") or status["target"]["target_id"] in self._profiles)
            for status in statuses
        )
        candidate = copy(self)
        for name in ("_hosts",) if heartbeat else _REGISTRATION_FIELDS:
            setattr(candidate, name, getattr(self, name).copy())
        candidate._fenced = self._fenced.copy()
        if not heartbeat and self.resources is not None:
            # Membership retirement belongs to this atomic directory update.
            # Request expiry stays in collect(); registration never mutates or
            # copies other devices' operation history.
            candidate._collect_retired_targets()
        return candidate

    def _publish_registration(self, candidate):
        for name in _REGISTRATION_FIELDS:
            setattr(self, name, getattr(candidate, name))
        self._fenced.update(candidate._fenced)

    def register_group(self, statuses, *, _prepared_profiles=None):
        """Atomically publish one complete policy stage lifecycle incarnation."""
        if not isinstance(statuses, list | tuple) or not 2 <= len(statuses) <= MAX_RUNTIME_GROUP_MEMBERS:
            raise ControlError("INVALID_RUNTIME_GROUP")
        validate_runtime_group_budget([status["target"] for status in statuses])
        profiles = self._prepare_profiles(statuses) if _prepared_profiles is None else _prepared_profiles
        with self._lock:
            if not isinstance(statuses, list | tuple) or not 2 <= len(statuses) <= MAX_RUNTIME_GROUP_MEMBERS:
                raise ControlError("INVALID_RUNTIME_GROUP")
            targets = [RuntimeExecutionTarget(**status["target"]) for status in statuses]
            target_ids = {target.target_id for target in targets}
            if len(target_ids) != len(targets) or len({target.endpoint for target in targets}) != len(targets):
                raise ControlError("INVALID_RUNTIME_GROUP")
            first, first_status = targets[0], statuses[0]
            identity = (
                "lifecycle_group",
                "deployment_fingerprint",
                "configuration_generation",
                "interface",
                "model_type",
                "operation",
                "hardware_resource_id",
                "physical_device",
                "resource_mode",
            )
            for target, status in zip(targets, statuses, strict=True):
                members = status.get("lifecycle_target_ids")
                if (
                    target.execution_kind != "stage"
                    or target.interface != "policy"
                    or any(getattr(target, name) != getattr(first, name) for name in identity)
                    or any(
                        status[name] != first_status[name] for name in ("boot_id", "generation", "epoch", "clock_id")
                    )
                    or not isinstance(members, list)
                    or len(members) != len(target_ids)
                    or any(not isinstance(member, str) for member in members)
                    or set(members) != target_ids
                ):
                    raise ControlError("INVALID_RUNTIME_GROUP")
            existing = {
                target.target_id for target in self.targets.values() if target.lifecycle_group == first.lifecycle_group
            }
            if existing and existing != target_ids:
                raise ControlError("INCOMPLETE_RUNTIME_GROUP")
            candidate = self._registration_candidate(statuses)
            try:
                for status in statuses:
                    candidate._register_one(status, group_ids=target_ids, profiles=profiles)
            except Exception:
                self._fenced.update(candidate._fenced)
                raise
            self._publish_registration(candidate)
            return {
                "epoch": self.epoch,
                "state": "registered",
                "resource_ready": self.resource_ready(first.hardware_resource_id),
            }

    def _register_one(self, status, *, group_ids=None, profiles):
        with self._lock:
            self._validate_status(status)
            target_id = status["target"]["target_id"]
            if status["target"].get("template_only", False):
                raise ControlError("RUNTIME_TEMPLATE_NOT_EXECUTABLE")
            if self._draining and (
                target_id not in self._hosts or status["boot_id"] != self._hosts[target_id][0]["boot_id"]
            ):
                raise ControlError("GLOBAL_SHUTTING_DOWN")
            for _, members in self._initializations.values():
                if any(name == target_id and boot != status["boot_id"] for name, boot in members):
                    raise ControlError("RESOURCE_INITIALIZING")
            target = self.targets.get(target_id)
            added = target is None and self.resources is not None
            if added:
                target = RuntimeExecutionTarget(**status["target"])
                if target.interface not in {"tensor_model", "policy"}:
                    raise ControlError("DYNAMIC_MODEL_REQUIRED")
                if target.execution_kind == "stage" and group_ids is None:
                    raise ControlError("GROUP_REGISTRATION_REQUIRED")
                self._validate_resource(target)
                if target.hardware_resource_id in self._fenced:
                    raise ControlError("RESOURCE_NOT_READY")
                if len(self.targets) >= self.max_registered_targets:
                    raise ControlError("TARGET_CAPACITY")
                if any(
                    t.endpoint == target.endpoint
                    or (
                        t.lifecycle_group == target.lifecycle_group
                        and (group_ids is None or t.target_id not in group_ids)
                    )
                    for t in self.targets.values()
                ):
                    raise ControlError("TARGET_ENDPOINT_OR_LIFECYCLE_CONFLICT")
            if target is None or status["target"] != target.to_dict():
                raise ControlError("TARGET_CONFIGURATION_MISMATCH")
            if status["clock_id"] != self.clock_id:
                raise ControlError("CROSS_HOST_CLOCK_UNSUPPORTED")
            previous = self._hosts.get(target_id)
            if (target_id, status["boot_id"]) in self._retired_boots or (
                target_id in self._retired and previous is not None and previous[0]["boot_id"] == status["boot_id"]
            ):
                raise ControlError("HOST_RETIRED")
            if status["epoch"] not in ("", self.epoch):
                self._fenced.add(target.hardware_resource_id)
                raise ControlError("COORDINATED_RESTART_REQUIRED")
            if previous is not None:
                old, _ = previous
                if old["boot_id"] != status["boot_id"]:
                    if self.resources is None or not self._inactive(target_id) or not status["clean"]:
                        self._fenced.add(target.hardware_resource_id)
                        raise ControlError("HOST_RESTART_REQUIRES_COORDINATED_RESTART")
                    if len(self._retired_boots) >= self.max_registered_targets * 2:
                        raise ControlError("TARGET_CAPACITY")
                    self._retired_boots[(target_id, old["boot_id"])] = self.now_ns() + max(
                        target.retry_window_ms * 1000000, self.stale_ns + 1
                    )
                    self._pending_activations.pop((target_id, old["boot_id"]), None)
                elif status["sequence"] <= old["sequence"]:
                    raise ControlError("STALE_STATUS")
                # Generations only increase within a boot. Completion evidence
                # may advance our generation before an older heartbeat arrives;
                # discard it without refreshing readiness or fencing the new one.
                if old["boot_id"] == status["boot_id"] and status["generation"] < old["generation"]:
                    raise ControlError("STALE_STATUS")
                if old["boot_id"] == status["boot_id"] and status["generation"] != old["generation"]:
                    pending_reset = any(
                        r.submission.work == "reset"
                        and r.state in {"dispatching", "running", "unknown"}
                        and self.targets[r.submission.target_id].lifecycle_group == target.lifecycle_group
                        for r in self._unsettled.values()
                    )
                    if pending_reset and status["generation"] == old["generation"] + 1:
                        raise ControlError("RESET_COMPLETION_EVIDENCE_PENDING")
                    if status["generation"] == self._pending_activations.get((target_id, status["boot_id"])):
                        raise ControlError("RECOVERY_COMPLETION_EVIDENCE_PENDING")
                    self._fenced.add(target.hardware_resource_id)
                    raise ControlError("UNCOORDINATED_GENERATION")
            elif not status["clean"]:
                self._fenced.add(target.hardware_resource_id)
                raise ControlError("STARTUP_DRAIN_REQUIRED")
            if (
                self.resources is not None
                and target.predictive_admission
                and (
                    added
                    or target_id not in self._profiles
                    or (previous is not None and previous[0]["boot_id"] != status["boot_id"])
                )
            ):
                if self.policy != "fifo":
                    raise ControlError("PREDICTIVE_FIFO_REQUIRED")
                if target_id not in profiles:
                    raise ControlError("INVALID_RUNTIME_PROFILE")
                self._profiles[target_id] = profiles[target_id]
            if added:
                self.targets[target_id] = target
            self._retired.pop(target_id, None)
            self._unregistered.pop(target_id, None)
            self._hosts[target_id] = (dict(status), status["status_ns"])
            return {
                "epoch": self.epoch,
                "state": "registered",
                "resource_ready": self.resource_ready(target.hardware_resource_id),
            }

    def unregister(self, status):
        """Retire only a drained incarnation; retain replay evidence until expiry."""
        with self._lock:
            self._validate_status(status)
            target_id = status["target"]["target_id"]
            target = self.targets.get(target_id)
            observed = self._hosts.get(target_id)
            if target is None and self.resources is not None:
                # Dynamic collection proves that no work or membership remains.
                # A retry after a lost unregister reply must not register again
                # just to prove absence; it cannot mutate another incarnation.
                self._validate_resource(RuntimeExecutionTarget(**status["target"]))
                if status["epoch"] != self.epoch or status["clock_id"] != self.clock_id:
                    raise ControlError("RUNTIME_IDENTITY_MISMATCH")
                if status["ready"] or not status["clean"]:
                    raise ControlError("HOST_NOT_DRAINED")
                return {"state": "unregistered", "epoch": self.epoch}
            if target is None or observed is None or status["target"] != target.to_dict():
                raise ControlError("TARGET_CONFIGURATION_MISMATCH")
            old = observed[0]
            if (
                old["boot_id"] != status["boot_id"]
                and (target_id, status["boot_id"]) in self._retired_boots
                and status["epoch"] == self.epoch
                and status["clock_id"] == self.clock_id
            ):
                # Replay the retired boot's result without touching its successor.
                # Retirement already proves its work settled; current work and
                # generation belong exclusively to the replacement incarnation.
                if status["ready"] or not status["clean"]:
                    raise ControlError("HOST_NOT_DRAINED")
                return {"state": "unregistered", "epoch": self.epoch}
            if status["epoch"] != self.epoch or any(
                status[key] != old[key] for key in ("boot_id", "generation", "clock_id")
            ):
                raise ControlError("RUNTIME_IDENTITY_MISMATCH")
            if status["sequence"] < old["sequence"]:
                raise ControlError("STALE_STATUS")
            if status["ready"] or not status["clean"] or not self._settled(target_id):
                raise ControlError("HOST_NOT_DRAINED")
            if target.hardware_resource_id in self._fenced and not self.host_retired(target_id, status["boot_id"]):
                raise ControlError("RESOURCE_NOT_READY")
            self._hosts[target_id] = (dict(status), status["status_ns"])
            expiry = self.now_ns() + max(target.retry_window_ms * 1000000, self.stale_ns + 1)
            self._retired[target_id] = expiry
            self._retired_boots[(target_id, status["boot_id"])] = expiry
            # Retirement releases scheduling participation. Unregister also
            # proves the owner reached its final transport barrier; Global
            # must keep serving until this response can be delivered.
            self._unregistered[target_id] = status["boot_id"]
            self._pending_activations.pop((target_id, status["boot_id"]), None)
            return {"state": "unregistered", "epoch": self.epoch}

    def _retire_completed_closes(self):
        # Ordinary Close is the same cleanup proof as recovery Close. Publish
        # retirement at the Global ACK barrier, before the owner can destroy
        # its control transport or a concurrent recovery tries to activate it.
        for operation_id, submission in tuple(self._pending_close_retirements.items()):
            if self.host_retired(submission.target_id, submission.boot_id) or (
                self.host_boot(submission.target_id) != submission.boot_id
            ):
                self._pending_close_retirements.pop(operation_id)
                continue
            group = self.targets[submission.target_id].lifecycle_group
            targets = [target for target in self.targets.values() if target.lifecycle_group == group]
            if all(
                self.host_boot(target.target_id) == submission.boot_id and self._settled(target.target_id)
                for target in targets
            ):
                self.retire_recovered(targets, {target.target_id: submission.boot_id for target in targets})
                self._pending_close_retirements.pop(operation_id)

    def terminal_report(self, report):
        """Acknowledge retained host evidence, including uncertain admission.

        An absent, unexpired operation can still arrive in a delayed Submit.
        The host must keep its terminal tombstone until that window closes.
        This also permits bounded cleanup when Submit never reached Global.
        """
        with self._lock:
            submission = Submission(**report["submission"])
            if report.get("state") not in TERMINAL:
                raise ControlError("TERMINAL_EVIDENCE_REQUIRED")
            record = self._records.get(submission.operation_id)
            if record is not None:
                self.report(report)
                if record.state not in TERMINAL or record.host_terminal_state != record.state:
                    raise ControlError("TERMINAL_EVIDENCE_CONFLICT")
                self.acknowledge_host(submission)
                # Preserve Global's admission/expiry reason without making a
                # completed business waiter perform another Query round trip.
                return {"state": "acknowledged", "outcome": self._report(record)}
            else:
                status = self._hosts.get(submission.target_id)
                if status is None or (
                    submission.boot_id,
                    submission.clock_id,
                ) != (status[0]["boot_id"], self.clock_id):
                    raise ControlError("RUNTIME_IDENTITY_MISMATCH")
                if submission.generation > status[0]["generation"]:
                    raise ControlError("RUNTIME_GENERATION_MISMATCH")
                if self.now_ns() < submission.accept_until_ns:
                    return {"state": "pending"}
            return {"state": "acknowledged"}

    def _collect_retired_targets(self):
        with self._lock:
            if self.resources is not None:
                now = self.now_ns()
                self._retired_boots = {key: expiry for key, expiry in self._retired_boots.items() if expiry > now}
                removable = set()
                for target_id, target in self.targets.items():
                    if target_id in self._configured_targets or target.hardware_resource_id in self._fenced:
                        continue
                    observed = self._hosts.get(target_id)
                    if observed is None:
                        continue
                    if self._draining or (
                        target_id in self._retired and self._unregistered.get(target_id) != observed[0]["boot_id"]
                    ):
                        # Recovery Close proves local release, not that its
                        # caller observed the final unregister response.
                        continue
                    expiry = self._retired.get(target_id, observed[1] + target.retry_window_ms * 1000000)
                    if now < expiry or self._record_counts.get(target_id, 0) or not self._inactive(target_id):
                        continue
                    removable.add(target_id)
                for target_id in tuple(removable):
                    target = self.targets[target_id]
                    if target.execution_kind == "stage":
                        peers = {
                            peer.target_id
                            for peer in self.targets.values()
                            if peer.lifecycle_group == target.lifecycle_group
                        }
                        if not peers <= removable:
                            removable.difference_update(peers)
                for target_id in removable:
                    del self.targets[target_id]
                    self._pending_activations.pop((target_id, self.host_boot(target_id)), None)
                    self._hosts.pop(target_id, None)
                    self._profiles.pop(target_id, None)
                    self._retired.pop(target_id, None)
                    self._unregistered.pop(target_id, None)
