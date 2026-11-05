"""Global runtime ledger, shared by policy, stages and typed perception hosts."""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from uuid import uuid4

from .runtime_profiles import DEFAULT_MIN_SAMPLES, RuntimeProfileRegistry, validate_profile_options
from .runtime_protocol import (
    CLOSE_INPUT_RESERVE_BYTES,
    DEFAULT_STATUS_STALE_TIMEOUT_NS,
    HOST_HEARTBEAT_INTERVAL_NS,
    TERMINAL,
    ControlError,
    Submission,
    encode_control,
    identifier,
    machine_clock_id,
    uuid_value,
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

    def report(self):
        return {"state": self.state, "error": self.error, "epoch": ""}


class RuntimeScheduler:
    """Bounded priority queue; priority zero uses FIFO/EDF, other levels FIFO."""

    def __init__(
        self,
        targets,
        *,
        policy="fifo",
        stale_ns=DEFAULT_STATUS_STALE_TIMEOUT_NS,
        now_ns=time.monotonic_ns,
        resources=None,
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
        self._next_collect_ns = None
        self._sequence = 0
        self._fenced = set()
        self._closed_hosts = set()
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
            if target.template_only:
                raise ValueError("fixed runtime targets cannot be templates")
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
            return sorted(name for name in self._hosts if name not in self._closed_hosts)

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
            for sample in self.measurements:
                if sample["sequence"] <= after_sequence:
                    continue
                candidate = {**page, "samples": [*page["samples"], sample], "next_sequence": sample["sequence"]}
                try:
                    encode_control(candidate)
                except ControlError:
                    if not page["samples"]:
                        raise
                    break
                page = candidate
                if len(page["samples"]) == 2:
                    break
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
            or target.admission_safety_margin_ms != resource.get("admission_safety_margin_ms", 5)
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
                if target is None or observed is None:
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
            return any(target_id in self._hosts for target_id in target_ids)

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

    def register(self, status):
        """Register or heartbeat an exact configured target; membership never grows."""
        with self._lock:
            self._validate_status(status)
            target_id = status["target"]["target_id"]
            target = self.targets.get(target_id)
            if target is None or status["target"] != target.to_dict():
                raise ControlError("TARGET_CONFIGURATION_MISMATCH")
            if status["clock_id"] != self.clock_id:
                raise ControlError("CROSS_HOST_CLOCK_UNSUPPORTED")
            if status["epoch"] not in ("", self.epoch):
                self._fenced.add(target.hardware_resource_id)
                raise ControlError("COORDINATED_RESTART_REQUIRED")
            previous = self._hosts.get(target_id)
            if self._draining and (previous is None or previous[0]["boot_id"] != status["boot_id"]):
                raise ControlError("GLOBAL_SHUTTING_DOWN")
            if previous is not None:
                old, _ = previous
                if old["boot_id"] != status["boot_id"]:
                    self._fenced.add(target.hardware_resource_id)
                    raise ControlError("HOST_RESTART_REQUIRES_COORDINATED_RESTART")
                if status["sequence"] <= old["sequence"] or status["generation"] < old["generation"]:
                    raise ControlError("STALE_STATUS")
                if status["generation"] != old["generation"]:
                    pending_reset = any(
                        r.submission.work == "reset"
                        and r.state in {"dispatching", "running", "unknown"}
                        and self.targets[r.submission.target_id].lifecycle_group == target.lifecycle_group
                        for r in self._unsettled.values()
                    )
                    if pending_reset and status["generation"] == old["generation"] + 1:
                        raise ControlError("RESET_COMPLETION_EVIDENCE_PENDING")
                    self._fenced.add(target.hardware_resource_id)
                    raise ControlError("UNCOORDINATED_GENERATION")
            elif not status["clean"]:
                self._fenced.add(target.hardware_resource_id)
                raise ControlError("STARTUP_DRAIN_REQUIRED")
            self._hosts[target_id] = (dict(status), status["status_ns"])
            return {
                "epoch": self.epoch,
                "state": "registered",
                "resource_ready": self.resource_ready(target.hardware_resource_id),
            }

    def resource_ready(self, resource, *, closing_target=None, admitted=False):
        if resource in self._fenced or (self._draining and closing_target is None and not admitted):
            return False
        for target in self.targets.values():
            if target.hardware_resource_id != resource:
                continue
            if target.target_id in self._closed_hosts:
                if closing_target is not None:
                    continue
                return False
            observed = self._hosts.get(target.target_id)
            if observed is None:
                return False
            status, status_ns = observed
            if status["epoch"] != self.epoch or self.now_ns() - status_ns > self.stale_ns:
                return False
            if target.target_id != closing_target and not status["ready"] and (target.required or not status["clean"]):
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
            if self.resources is not None and (self.now_ns() - observed[1] > self.stale_ns):
                raise ControlError("RUNTIME_NOT_REGISTERED")
            status = observed[0]
            if (submission.boot_id, submission.generation, submission.clock_id) != (
                status["boot_id"],
                status["generation"],
                self.clock_id,
            ):
                raise ControlError("RUNTIME_IDENTITY_MISMATCH")
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
                    record.host_acknowledged = True
                    self._unsettled.pop(submission.operation_id, None)
                    expiry = max(submission.accept_until_ns, submission.deadline_ns)
                    self._next_collect_ns = (
                        expiry if self._next_collect_ns is None else min(expiry, self._next_collect_ns)
                    )
                    if submission.work == "close" and record.state == "completed" and not record.error:
                        self._closed_hosts.add(submission.target_id)

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
            now = self.now_ns()
            if self._next_collect_ns is None or now < self._next_collect_ns:
                return
            self._next_collect_ns = None
            expired = []
            for key, record in self._records.items():
                if record.state not in TERMINAL or not record.host_acknowledged:
                    continue
                # ACK confirms execution, not consumption: keep query/replay
                # through both the retry window and the business deadline.
                expiry = max(record.submission.accept_until_ns, record.submission.deadline_ns)
                if now >= expiry:
                    expired.append(key)
                else:
                    self._next_collect_ns = (
                        expiry if self._next_collect_ns is None else min(expiry, self._next_collect_ns)
                    )
            for key in expired:
                self._record_counts[self._records[key].submission.target_id] -= 1
                del self._records[key]

    def _report(self, record):
        return {**record.report(), "epoch": self.epoch}

    @staticmethod
    def _match(record, submission):
        if record.submission != submission:
            raise ControlError("OPERATION_CONFLICT")
