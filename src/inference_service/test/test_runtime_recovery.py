"""Controlled runtime scheduling regression tests (no accelerator required)."""

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from inference_service.scheduler.runtime_controller import RuntimeExecutionController
from inference_service.scheduler.runtime_protocol import (
    ControlError,
)
from inference_service.scheduler.runtime_scheduler import RuntimeScheduler
from inference_service.unified_runtime import Deadline, ExecutionContext, ModelRequest, ModelRuntimeHandle

from .runtime_scheduler_helpers import Executor, connect, controller, prepare, target

pytestmark = pytest.mark.usefixtures("simulated_runtime_construction")


@pytest.mark.parametrize("command", ["reset", "close", "drain"])
@pytest.mark.parametrize("renew_before_completion", [False, True])
def test_recovery_receipt_survives_collection_until_coordinator_settles(monkeypatch, command, renew_before_completion):
    clock = [time.monotonic_ns()]
    host, _ = controller(target(), now_ns=lambda: clock[0])
    host.accept_epoch("epoch")
    entered, release = threading.Event(), threading.Event()
    calls = []

    def perform(*args):
        calls.append(args)
        entered.set()
        assert release.wait(2)

    monkeypatch.setattr(host, "_recover", perform)
    operation = str(uuid4())
    expiry = clock[0] + 1_000_000
    try:
        host.recover("epoch", operation, command, expiry)
        assert entered.wait(1)
        clock[0] = expiry + 1
        renewed = clock[0] + 1_000_000
        if renew_before_completion:
            assert host.recover("epoch", operation, command, expiry, resume_until_ns=renewed)["state"] == "running"
        release.set()
        host._control_future.result(timeout=1)
        host.collect()
        assert operation in host._recovery_records
        assert host.recover("epoch", operation, command, expiry, resume_until_ns=renewed)["state"] == "completed"
        assert len(calls) == 1
        host.recover("epoch", operation, command, expiry, acknowledged=True)
        host.collect()
        assert operation in host._recovery_records  # Retain the renewed retry window.
        clock[0] = renewed + 1
        host.collect()
        assert operation not in host._recovery_records
        assert host.recover("epoch", operation, command, expiry, acknowledged=True)["state"] == "acknowledged"
        with pytest.raises(ControlError, match="RECOVERY_RECEIPT_UNAVAILABLE"):
            host.recover("epoch", operation, command, expiry, resume_until_ns=clock[0] + 1_000_000)
        assert len(calls) == 1
    finally:
        release.set()
        host.shutdown()


def test_confirmed_activation_reclaims_superseded_failed_host_commands(monkeypatch):
    clock = [time.monotonic_ns()]
    host, _ = controller(target(), now_ns=lambda: clock[0])
    host.accept_epoch("epoch")
    failed, activation, newer = str(uuid4()), str(uuid4()), str(uuid4())
    expiry = clock[0] + 1_000_000

    def recover(command, *_):
        if command == "drain":
            raise RuntimeError("device completion unknown")

    monkeypatch.setattr(host, "_recover", recover)
    try:
        for operation, command, generation in (
            (failed, "drain", None),
            (activation, "activate", 2),
            (newer, "drain", None),
        ):
            host.recover("epoch", operation, command, expiry, expected_generation=generation)
            host._control_future.result(timeout=1)
        host.recover("epoch", activation, "activate", expiry, expected_generation=2, acknowledged=True)
        clock[0] = expiry + 1
        host.collect()
        # A delayed ACK only settles preceding receipts, never a newer failure.
        assert set(host._recovery_records) == {newer}
    finally:
        host.shutdown()


@pytest.mark.parametrize("final_command", ["activate", "close"])
def test_full_mutation_receipts_leave_capacity_for_evidence_and_cleanup(monkeypatch, final_command):
    from inference_service.scheduler.runtime_controller import MAX_HOST_RECOVERY_RECEIPTS
    from inference_service.scheduler.runtime_protocol import NotStartedControlError

    host, executor = controller(target())
    host.accept_epoch("epoch")
    calls = []

    def fail(*_args, **_kwargs):
        calls.append("reset_stream")
        raise RuntimeError("native completion unknown")

    monkeypatch.setattr(host.handle, "recover_stream", fail, raising=False)
    monkeypatch.setattr(executor, "drain_uncertain_operations", lambda: calls.append("drain"), raising=False)
    expiry = host.now_ns() + 10_000_000_000
    try:
        for index in range(MAX_HOST_RECOVERY_RECEIPTS - 2):
            operation = str(uuid4())
            host.recover("epoch", operation, "reset_stream", expiry, stream_id=str(index))
            host._control_future.result(timeout=1)
            with pytest.raises(Exception, match="native completion unknown"):
                host.recover("epoch", operation, "reset_stream", expiry, stream_id=str(index))
        with pytest.raises(NotStartedControlError, match="RECOVERY_CAPACITY"):
            host.recover("epoch", str(uuid4()), "reset_stream", expiry, stream_id="full")
        for command in ("drain", final_command):
            operation = str(uuid4())
            kwargs = {"expected_generation": 2} if command == "activate" else {}
            host.recover("epoch", operation, command, expiry, **kwargs)
            host._control_future.result(timeout=1)
            assert host.recover("epoch", operation, command, expiry, **kwargs)["state"] == "completed"
            host.recover("epoch", operation, command, expiry, acknowledged=True, **kwargs)
        assert len(host._recovery_records) == MAX_HOST_RECOVERY_RECEIPTS
        assert all(receipt.acknowledged for receipt in host._recovery_records.values())
        assert calls.count("reset_stream") == MAX_HOST_RECOVERY_RECEIPTS - 2
        assert host.handle.closed if final_command == "close" else host.generation == 2
    finally:
        host.shutdown()


def test_shutdown_rejects_new_work_but_preserves_close_and_evidence():
    host, _ = controller(target())
    core = RuntimeScheduler([host.target])
    connect(core, host)
    try:
        submitted = prepare(host)
        core.submit(submitted)
        core.begin_shutdown()
        assert core.shutdown_pending() == [host.target.target_id]
        assert core.submit(submitted)["state"] == "queued"
        with pytest.raises(ControlError, match="GLOBAL_SHUTTING_DOWN"):
            core.submit(prepare(host))
        # Heartbeats and cleanup work are still accepted.
        core.register(host.status())
        core.cancel(submitted)
        maintenance = host.prepare(ModelRequest(), ExecutionContext("close", Deadline.after(3)), work="close")
        assert core.submit(maintenance)["state"] == "queued"
    finally:
        host.shutdown()


@pytest.mark.parametrize("lost_phase", ["activate", "drain"])
def test_expired_recovery_retries_activation_without_generation_fork(lost_phase):
    from inference_service.scheduler.runtime_protocol import decode_control, encode_control
    from inference_service.scheduler.runtime_recovery import ResourceRecovery

    clock = [time.monotonic_ns()]
    t = target()
    core = RuntimeScheduler([t], now_ns=lambda: clock[0])
    host, _ = controller(t, now_ns=lambda: clock[0])
    drop = [True]
    sent = []

    def call_async(request):
        message = decode_control(request.payload)
        sent.append(message)
        kwargs = {
            "expected_generation": message.get("expected_generation"),
            "resume_until_ns": message.get("resume_until_ns"),
            "acknowledged": message.get("acknowledged", False),
        }
        args = (message["epoch"], message["operation_id"], message["command"], message["accept_until_ns"])
        host.recover(*args, **kwargs)
        host._control_future.result(timeout=1)
        reply = host.recover(*args, **kwargs)
        future = Future()
        if not (drop[0] and message["command"] == lost_phase):
            future.set_result(SimpleNamespace(payload=encode_control(reply)))
        return future

    recovery = ResourceRecovery(
        SimpleNamespace(
            core=core,
            control_rpc_timeout_ns=5_000_000_000,
            _clients={
                t.target_id: SimpleNamespace(
                    service_is_ready=lambda: True, call_async=call_async, remove_pending_request=lambda _: None
                )
            },
        )
    )
    try:
        connect(core, host)
        for attempt in range(2):
            operation_id = str(uuid4())
            recovery.request(
                {
                    "resource": t.hardware_resource_id,
                    "operation_id": operation_id,
                    "command": "drain",
                    "accept_until_ns": clock[0] + 1_000_000_000,
                }
            )
            for _ in range(10):
                recovery.tick()
            if attempt == 0:
                clock[0] += 1_000_000_001
                recovery.tick()
                assert recovery.records[operation_id]["error"] == "RECOVERY_EXPIRED"
                expired_request = dict(recovery.records[operation_id]["request"])
                assert not core.resource_ready(t.hardware_resource_id)
                drop[0] = False
            else:
                assert recovery.records[operation_id]["state"] == "completed"
        superseded = [record for record in recovery.records.values() if record["state"] == "superseded"]
        # Periodic collection reclaims expired superseded non-Close receipts;
        # they no longer need a new request to trigger eviction.
        assert not superseded
        with pytest.raises(ControlError, match="RECOVERY_RECEIPT_UNAVAILABLE"):
            recovery.request({**expired_request, "resume_until_ns": clock[0] + 1_000_000_000})
        recovery.request(
            {
                "resource": t.hardware_resource_id,
                "operation_id": str(uuid4()),
                "command": "drain",
                "accept_until_ns": clock[0] + 1_000_000_000,
            }
        )
        assert all(record["state"] != "superseded" for record in recovery.records.values())
        for _ in range(10):
            recovery.tick()
        core.register(host.status())
        assert host.generation == core._hosts[t.target_id][0]["generation"] == 3
        assert core.resource_ready(t.hardware_resource_id)
        assert {m["expected_generation"] for m in sent if m["command"] == "activate"} == {2, 3}
    finally:
        host.shutdown()


@pytest.mark.parametrize("ack_arrives", [False, True])
def test_activation_waits_for_host_terminal_ack_within_original_window(ack_arrives):
    from inference_service.scheduler.runtime_protocol import decode_control, encode_control
    from inference_service.scheduler.runtime_recovery import ResourceRecovery

    clock = [time.monotonic_ns()]
    t = target()
    core = RuntimeScheduler([t], now_ns=lambda: clock[0])
    host, _ = controller(t, now_ns=lambda: clock[0])
    sent = []

    def call_async(request):
        message = decode_control(request.payload)
        sent.append(message)
        if message.get("acknowledged"):
            future = Future()
            future.set_result(SimpleNamespace(payload=encode_control({"state": "acknowledged"})))
            return future
        try:
            host._recover(message["command"], message.get("expected_generation"))
            reply = {"state": "completed", "generation": host.generation}
        except ControlError as exc:
            reply = {"control_error": exc.code}
        future = Future()
        future.set_result(SimpleNamespace(payload=encode_control(reply)))
        return future

    recovery = ResourceRecovery(
        SimpleNamespace(
            core=core, _clients={t.target_id: SimpleNamespace(service_is_ready=lambda: True, call_async=call_async)}
        )
    )
    try:
        connect(core, host)
        sub = prepare(host)
        core.submit(sub)
        host.cancel(sub, core.epoch)
        assert core.terminal_report(host.terminal_reports()[0])["state"] == "acknowledged"
        assert core.resource_settled(t.hardware_resource_id)
        operation = str(uuid4())
        expiry = clock[0] + 1_000_000_000
        recovery.request(
            {
                "resource": t.hardware_resource_id,
                "operation_id": operation,
                "command": "drain",
                "accept_until_ns": expiry,
            }
        )
        for _ in range(10):
            recovery.tick()
        assert recovery.records[operation]["state"] == "running"
        assert not core.resource_ready(t.hardware_resource_id)
        if ack_arrives:
            host.acknowledge(sub, core.epoch)
            for _ in range(5):
                recovery.tick()
            assert recovery.records[operation]["state"] == "completed"
            core.register(host.status())
            assert core.resource_ready(t.hardware_resource_id)
            assert host.generation == 2
        else:
            clock[0] = expiry
            recovery.tick()
            assert recovery.records[operation]["state"] == "unknown"
            assert recovery.records[operation]["error"] == "RECOVERY_EXPIRED"
            assert not core.resource_ready(t.hardware_resource_id)
        activations = [
            message for message in sent if message["command"] == "activate" and not message.get("acknowledged")
        ]
        assert len(activations) > 1
        assert all(message == activations[0] for message in activations)
    finally:
        host.shutdown()


def test_activation_rejects_generation_conflicts_and_jumps():
    host, _ = controller(target())
    host.accept_epoch("test")
    operation_id = str(uuid4())
    expiry = host.now_ns() + 1_000_000_000
    try:
        host.recover("test", operation_id, "activate", expiry, expected_generation=2)
        host._control_future.result(timeout=1)
        with pytest.raises(ControlError, match="OPERATION_CONFLICT"):
            host.recover("test", operation_id, "activate", expiry, expected_generation=3)
        for generation in (1, 4, None):
            with pytest.raises(ControlError, match="UNCOORDINATED_GENERATION"):
                host._recover("activate", generation)
        assert host.generation == 2
    finally:
        host.shutdown()


def test_host_reconciliation_keeps_cancel_immediate_and_deduplicates_inflight(monkeypatch):
    from inference_service.scheduler import runtime_ros

    clock = [0.0]
    state = ["running"]
    sent = []
    endpoint = object.__new__(runtime_ros.GlobalRuntimeEndpoint)
    endpoint._pending = {}
    endpoint._control_schedule = {}
    endpoint.timer = object()
    endpoint.control_rpc_timeout_ns = 1_000_000_000
    endpoint._sync_clients = lambda: None
    endpoint.core = SimpleNamespace(epoch="test", query=lambda _: {"state": state[0]})
    sub = SimpleNamespace(operation_id="request", to_dict=lambda: {})
    t = target()

    def call(request):
        sent.append(request["verb"])
        return Future()

    endpoint._clients = {t.target_id: SimpleNamespace(service_is_ready=lambda: True, call_async=call)}
    monkeypatch.setattr(runtime_ros, "_request", lambda value: value)
    monkeypatch.setattr(runtime_ros.time, "monotonic", lambda: clock[0])
    for index in range(100):
        clock[0] = index / 100
        # A fast server has replied before the next tick.
        endpoint._pending.clear()
        endpoint._send(t, sub, "query")
    assert sent.count("query") == 3
    clock[0] += 0.25
    endpoint._pending.clear()
    state[0] = "unknown"
    endpoint._send(t, sub, "query")
    count = len(sent)
    endpoint._send(t, sub, "query")
    assert len(sent) == count  # In-flight dedup remains in force.
    endpoint._send(t, sub, "cancel")
    assert sent[-1] == "cancel"


@pytest.mark.parametrize("heartbeat_first", [False, True])
def test_reset_completion_and_heartbeat_order_preserves_resource(heartbeat_first):
    t = target()
    core = RuntimeScheduler([t])
    host, _ = controller(t)
    try:
        connect(core, host)
        delayed = host.status()
        sub = prepare(host, work="reset")
        core.submit(sub)
        core.dispatchable()
        host.execute(sub, core.epoch)
        host._records[sub.operation_id].done.result(timeout=1)
        if heartbeat_first:
            with pytest.raises(ControlError, match="RESET_COMPLETION_EVIDENCE_PENDING"):
                core.register(host.status())
        core.report(host.query(sub, core.epoch))
        observed = core._hosts[t.target_id]
        with pytest.raises(ControlError, match="STALE_STATUS"):
            core.register(delayed)
        assert core._hosts[t.target_id] == observed
        core.register(host.status())
        assert core.resource_ready(t.hardware_resource_id)
        assert core.submit(prepare(host))["state"] == "queued"
    finally:
        host.shutdown()


@pytest.mark.parametrize("heartbeat_first,bad_response", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("command", ["drain", "reset"])
def test_recovery_completion_and_heartbeat_order_preserves_resource(command, heartbeat_first, bad_response):
    from inference_service.scheduler.runtime_protocol import decode_control, encode_control
    from inference_service.scheduler.runtime_recovery import ResourceRecovery

    t = target()
    core = RuntimeScheduler([t])
    host, _ = controller(t)
    delayed = []

    def call_async(request):
        message = decode_control(request.payload)
        if message.get("acknowledged"):
            future = Future()
            future.set_result(SimpleNamespace(payload=encode_control({"state": "acknowledged"})))
            return future
        if message["command"] == "activate":
            delayed.append(host.status())
        host._recover(message["command"], message.get("expected_generation"))
        if message["command"] == "activate" and heartbeat_first:
            with pytest.raises(ControlError, match="RECOVERY_COMPLETION_EVIDENCE_PENDING"):
                core.register(host.status())
            assert not core.resource_ready(t.hardware_resource_id)
        future = Future()
        generation = host.generation + (bad_response and message["command"] == "activate")
        future.set_result(SimpleNamespace(payload=encode_control({"state": "completed", "generation": generation})))
        return future

    recovery = ResourceRecovery(
        SimpleNamespace(
            core=core,
            _clients={t.target_id: SimpleNamespace(service_is_ready=lambda: True, call_async=call_async)},
        )
    )
    try:
        connect(core, host)
        operation_id = str(uuid4())
        recovery.request(
            {
                "resource": t.hardware_resource_id,
                "operation_id": operation_id,
                "command": command,
                "accept_until_ns": core.now_ns() + 1000000000,
            }
        )
        for _ in range(10):
            recovery.tick()
        if bad_response:
            assert recovery.records[operation_id]["state"] == "unknown"
            assert recovery.records[operation_id]["error"] == "UNCOORDINATED_GENERATION"
            assert not core.resource_ready(t.hardware_resource_id)
            return
        assert recovery.records[operation_id]["state"] == "completed"
        assert not core._pending_activations
        observed = core._hosts[t.target_id]
        with pytest.raises(ControlError, match="STALE_STATUS"):
            core.register(delayed[0])
        assert core._hosts[t.target_id] == observed
        core.register(host.status())
        assert core.resource_ready(t.hardware_resource_id)
        assert core.submit(prepare(host))["state"] == "queued"
    finally:
        host.shutdown()


@pytest.mark.parametrize("field", ["generation", "boot_id", "epoch"])
def test_pending_activation_does_not_waive_identity_fences(field):
    t = target()
    core = RuntimeScheduler([t])
    host, _ = controller(t)
    try:
        connect(core, host)
        core.next_activation_generation(t.target_id, host.boot_id)
        status = host.status()
        status[field] = host.generation + 2 if field == "generation" else str(uuid4())
        with pytest.raises(ControlError):
            core.register(status)
        assert not core.resource_ready(t.hardware_resource_id)
    finally:
        host.shutdown()


@pytest.mark.parametrize("verb", ["execute", "query", "cancel", "ack"])
@pytest.mark.parametrize("reply", ["terminal", "running", "error", "timeout"])
def test_transport_after_terminal_collection_is_harmless(verb, reply):
    from inference_service.scheduler.runtime_protocol import encode_control
    from inference_service.scheduler.runtime_ros import GlobalRuntimeEndpoint

    clock = [time.monotonic_ns()]
    t = target(retry_window_ms=100)
    core = RuntimeScheduler([t], now_ns=lambda: clock[0])
    host, _ = controller(t, now_ns=lambda: clock[0])
    try:
        connect(core, host)
        sub = prepare(host)
        core.submit(sub)
        core.dispatchable()
        report = {"submission": sub.to_dict(), "state": "completed"}
        core.report(report)
        core.acknowledge_host(sub)
        clock[0] = max(sub.accept_until_ns, sub.deadline_ns) + 1
        core.collect()
        assert sub.operation_id not in core._records

        future = Future()
        if reply == "error":
            future.set_exception(RuntimeError("transport failed"))
        elif reply != "timeout":
            future.set_result(
                SimpleNamespace(
                    payload=encode_control({**report, "state": "running" if reply == "running" else "completed"})
                )
            )
        removed = []
        endpoint = object.__new__(GlobalRuntimeEndpoint)
        endpoint._control_schedule = {}
        endpoint.timer = object()
        endpoint.control_rpc_timeout_ns = 1_000_000_000
        endpoint._sync_clients = lambda: None
        endpoint.core = core
        endpoint.recovery = SimpleNamespace(tick=lambda: None)
        endpoint._clients = {t.target_id: SimpleNamespace(remove_pending_request=removed.append)}
        endpoint._pending = {
            (sub.operation_id, verb): (future, time.monotonic_ns() - endpoint.control_rpc_timeout_ns - 1, t, sub)
        }
        endpoint._tick()
        assert endpoint._pending == {}
        assert removed == ([future] if reply == "timeout" else [])
        assert sub.operation_id not in core._records
        connect(core, host)
        assert core.resource_ready(t.hardware_resource_id)
        # Collection and stale transport must not prevent fresh dispatch.
        next_sub = prepare(host)
        core.submit(next_sub)
        assert core.dispatchable() == [(t, next_sub)]
    finally:
        host.shutdown()


@pytest.mark.parametrize("work", ["execute", "reset"])
@pytest.mark.parametrize("lost_reply", ["timeout", "error", "malformed"])
def test_terminal_ack_retries_after_host_collection_and_unblocks_recovery(work, lost_reply):
    from inference_service.scheduler.runtime_protocol import decode_control, encode_control
    from inference_service.scheduler.runtime_recovery import ResourceRecovery
    from inference_service.scheduler.runtime_ros import GlobalRuntimeEndpoint, SchedulerRuntimeAdapter

    clock = [time.monotonic_ns()]
    t = target(retry_window_ms=100)
    core = RuntimeScheduler([t], now_ns=lambda: clock[0])
    host, _ = controller(t, now_ns=lambda: clock[0])
    adapter = object.__new__(SchedulerRuntimeAdapter)
    adapter._registration_error = ""
    adapter.controller = host
    adapter.target = t
    sent, removed = [], []

    def call_async(request):
        message = decode_control(request.payload)
        sent.append(message["verb"])
        future = Future()
        try:
            reply = adapter._handle(message)
            if len(sent) == 1:
                if lost_reply == "error":
                    future.set_exception(RuntimeError("lost ACK reply"))
                elif lost_reply == "malformed":
                    future.set_result(SimpleNamespace(payload=encode_control({"state": "running"})))
                return future
            future.set_result(SimpleNamespace(payload=encode_control(reply)))
        except Exception as exc:
            future.set_exception(exc)
        return future

    endpoint = object.__new__(GlobalRuntimeEndpoint)
    endpoint._control_schedule = {}
    endpoint.timer = object()
    endpoint.control_rpc_timeout_ns = 1_000_000_000
    endpoint._sync_clients = lambda: None
    endpoint.core = core
    endpoint._pending = {}
    endpoint._clients = {
        t.target_id: SimpleNamespace(
            service_is_ready=lambda: True, call_async=call_async, remove_pending_request=removed.append
        )
    }
    endpoint.recovery = ResourceRecovery(endpoint)
    try:
        connect(core, host)
        sub = prepare(host, work=work)
        core.submit(sub)
        core.dispatchable()
        host.execute(sub, core.epoch)
        host._records[sub.operation_id].done.result(timeout=1)
        core.report(host.query(sub, core.epoch))
        host.result(sub)
        host.acknowledge(sub, core.epoch)  # Business caller finishes independently of Global.
        endpoint._tick()
        assert sent == ["ack"]
        assert not core._records[sub.operation_id].host_acknowledged

        clock[0] = max(sub.accept_until_ns, sub.deadline_ns) + 1
        for retained in tuple(host._records.values()):
            if retained.state in {"completed", "not_started"}:
                host.release_result_waiter(retained.submission)
        host.collect()
        assert sub.operation_id not in host._records
        assert host.generation == sub.generation + (work == "reset")
        if lost_reply == "timeout":
            future, _, target_, submission = endpoint._pending[(sub.operation_id, "ack")]
            endpoint._pending[(sub.operation_id, "ack")] = (
                future,
                time.monotonic_ns() - endpoint.control_rpc_timeout_ns - 1,
                target_,
                submission,
            )
        time.sleep(0.26)
        endpoint._tick()
        assert sent == ["ack", "ack"]
        assert len(removed) == (lost_reply == "timeout")

        recovery_id = str(uuid4())
        endpoint.recovery.request(
            {
                "resource": t.hardware_resource_id,
                "operation_id": recovery_id,
                "command": "drain",
                "accept_until_ns": clock[0] + t.retry_window_ms * 1000000,
            }
        )
        until = time.monotonic() + 2
        while endpoint.recovery.records[recovery_id]["state"] == "running" and time.monotonic() < until:
            endpoint._tick()
            time.sleep(0.001)
        assert sub.operation_id not in core._records
        assert endpoint.recovery.records[recovery_id]["state"] == "completed"
        assert core.resource_ready(t.hardware_resource_id)
    finally:
        host.shutdown()


@pytest.mark.parametrize("field", ["epoch", "boot_id", "clock_id", "target_id", "generation"])
def test_collected_ack_keeps_identity_fences(field):
    t = target()
    core = RuntimeScheduler([t])
    host, _ = controller(t)
    try:
        connect(core, host)
        sub = prepare(host, work="reset")
        host.execute(sub, core.epoch)
        host._records[sub.operation_id].done.result(timeout=1)
        host.acknowledge(sub, core.epoch)
        host.now_ns = lambda: max(sub.accept_until_ns, sub.deadline_ns) + 1
        for retained in tuple(host._records.values()):
            if retained.state in {"completed", "not_started"}:
                host.release_result_waiter(retained.submission)
        host.collect()
        assert sub.operation_id not in host._records
        assert host.generation > sub.generation
        if field == "epoch":
            changed, epoch, error = sub, str(uuid4()), "EPOCH_MISMATCH"
        elif field == "generation":
            changed, epoch, error = (
                replace(sub, generation=host.generation + 1),
                core.epoch,
                "RUNTIME_GENERATION_MISMATCH",
            )
        else:
            changed, epoch, error = replace(sub, **{field: str(uuid4())}), core.epoch, "RUNTIME_IDENTITY_MISMATCH"
        with pytest.raises(ControlError, match=error):
            host.acknowledge(changed, epoch)
        # Cleanup retries alone may cross generations; execution still cannot.
        host.acknowledge(sub, core.epoch)
        host.now_ns = lambda: sub.accept_until_ns - 1
        with pytest.raises(ControlError, match="RUNTIME_GENERATION_MISMATCH"):
            host.execute(sub, core.epoch)
    finally:
        host.shutdown()


def test_rejected_submissions_release_local_capacity_after_retry_window():
    from inference_service.scheduler.runtime_ros import SchedulerRuntimeAdapter, _RemoteControlError

    clock = [1_000_000_000]
    t = target(max_records=2, max_pending_requests=1)
    core = RuntimeScheduler([t, target("missing")], now_ns=lambda: clock[0])
    host, executor = controller(t, now_ns=lambda: clock[0])
    adapter = SchedulerRuntimeAdapter.__new__(SchedulerRuntimeAdapter)
    adapter.controller, adapter.target = host, t

    def call(verb, submission, timeout, **kwargs):
        try:
            return getattr(core, verb)(submission)
        except ControlError as exc:
            raise _RemoteControlError(exc.code) from exc

    adapter._call = call
    try:
        connect(core, host)
        # Test the adapter rejection cleanup without needing a DDS failure.
        # Its waiter clock is monotonic; local expiry uses the injected clock.
        import time

        clock[0] = time.monotonic_ns()
        connect(core, host)
        for _ in range(t.max_records):
            from inference_service.scheduler.runtime_protocol import NotStartedControlError

            with pytest.raises(NotStartedControlError):
                adapter.execute(ModelRequest(), ExecutionContext(str(uuid4()), Deadline.after(0.05)))
        assert not core._records
        assert all(r.state == "not_started" and r.acknowledged for r in host._records.values())
        for retained in tuple(host._records.values()):
            if retained.state in {"completed", "not_started"}:
                host.release_result_waiter(retained.submission)
        host.collect()
        assert len(host._records) == t.max_records  # Late Execute still sees tombstones.
        clock[0] += t.retry_window_ms * 1_000_000 + 1
        for record in tuple(host._records.values()):
            assert core.terminal_report(record.report())["state"] == "acknowledged"
            host.acknowledge(record.submission, core.epoch)
        for retained in tuple(host._records.values()):
            if retained.state in {"completed", "not_started"}:
                host.release_result_waiter(retained.submission)
        host.collect()
        assert not host._records
        assert prepare(host)
        assert not executor.calls
    finally:
        host.shutdown()


@pytest.mark.parametrize("stop", ["deadline", "cancel", "lost_reply"])
def test_startup_readiness_wait_is_bounded_and_preserves_uncertain_submit(stop):
    from concurrent.futures import TimeoutError as TransportTimeout

    from inference_service.scheduler.runtime_protocol import NotStartedControlError
    from inference_service.scheduler.runtime_ros import SchedulerRuntimeAdapter, _RemoteControlError

    t = target()
    core = RuntimeScheduler([t, target("starting_peer")])
    host, executor = controller(t)
    connect(core, host)
    adapter = SchedulerRuntimeAdapter.__new__(SchedulerRuntimeAdapter)
    adapter.controller, adapter.target = host, t
    context = ExecutionContext("startup", Deadline.after(0.2 if stop == "deadline" else 2))
    submissions, cancellations = [], []

    def call(verb, submission, timeout, **kwargs):
        if verb == "cancel":
            cancellations.append(submission)
            return core.cancel(submission)
        assert verb == "submit"
        submissions.append(submission)
        if stop == "cancel":
            context.cancellation.cancel("stop startup")
        if stop == "lost_reply" and len(submissions) == 2:
            raise TransportTimeout("Submit reply lost")
        try:
            return core.submit(submission)
        except ControlError as exc:
            raise _RemoteControlError(exc.code) from exc

    adapter._call = call
    try:
        with pytest.raises(NotStartedControlError):
            adapter.execute(ModelRequest(), context)
        assert submissions
        assert len({s.operation_id for s in submissions}) == 1
        assert len({s.deadline_ns for s in submissions}) == 1
        assert not executor.calls
        if stop == "lost_reply":
            assert len(submissions) == 3
            assert len(cancellations) == 1
            assert all(not r.acknowledged for r in host._records.values())
        else:
            assert not cancellations and not core._records
            assert all(r.acknowledged for r in host._records.values())
    finally:
        host.shutdown()


@pytest.mark.parametrize("admission", ["absent", "queued", "dispatching"])
def test_abandoned_submit_retains_terminal_evidence_until_global_receipt(admission):
    from inference_service.scheduler.runtime_protocol import decode_control, encode_control
    from inference_service.scheduler.runtime_recovery import ResourceRecovery
    from inference_service.scheduler.runtime_ros import SchedulerRuntimeAdapter

    clock = [time.monotonic_ns()]
    t = target(retry_window_ms=100)
    core = RuntimeScheduler([t], now_ns=lambda: clock[0])
    host, executor = controller(t, now_ns=lambda: clock[0])
    adapter = object.__new__(SchedulerRuntimeAdapter)
    adapter._registration_error = ""
    adapter.controller, adapter.target = host, t
    adapter._terminal_receipts = {}
    context = ExecutionContext("lost-submit", Deadline.after(5))

    def call(verb, submission, timeout, **kwargs):
        if verb == "submit":
            if admission != "absent":
                core.submit(submission)
                if admission == "dispatching":
                    assert core.dispatchable() == [(t, submission)]
            context.cancellation_token.cancel("business abandoned after lost Submit reply")
        raise TimeoutError("lost reply")

    def terminal_receipt(request):
        message = decode_control(request.payload)
        assert message["verb"] == "terminal_report"
        reply = core.terminal_report(message["report"])
        future = Future()
        future.set_result(SimpleNamespace(payload=encode_control(reply)))
        return future

    def recovery_call(request):
        message = decode_control(request.payload)
        args = (message["epoch"], message["operation_id"], message["command"], message["accept_until_ns"])
        kwargs = {
            "expected_generation": message.get("expected_generation"),
            "resume_until_ns": message.get("resume_until_ns"),
            "acknowledged": message.get("acknowledged", False),
        }
        host.recover(*args, **kwargs)
        host._control_future.result(timeout=1)
        future = Future()
        future.set_result(SimpleNamespace(payload=encode_control(host.recover(*args, **kwargs))))
        return future

    adapter._call = call
    adapter._client = SimpleNamespace(service_is_ready=lambda: True, call_async=terminal_receipt)
    try:
        connect(core, host)
        with pytest.raises(ControlError):
            adapter.execute(ModelRequest(), context)
        record = next(iter(host._records.values()))
        submission = record.submission
        assert record.state == "not_started" and not record.acknowledged
        clock[0] = submission.accept_until_ns + 1
        for retained in tuple(host._records.values()):
            if retained.state in {"completed", "not_started"}:
                host.release_result_waiter(retained.submission)
        host.collect()
        assert submission.operation_id in host._records
        if admission == "dispatching":
            core.mark_transport_unknown(submission)
        adapter._flush_terminal_evidence()
        adapter._flush_terminal_evidence()
        for retained in tuple(host._records.values()):
            if retained.state in {"completed", "not_started"}:
                host.release_result_waiter(retained.submission)
        host.collect()
        assert not host._records and not executor.calls
        if admission != "absent":
            assert core.query(submission)["state"] == "not_started"
            assert core._records[submission.operation_id].host_acknowledged
        recovery = ResourceRecovery(
            SimpleNamespace(
                core=core,
                _clients={t.target_id: SimpleNamespace(service_is_ready=lambda: True, call_async=recovery_call)},
            )
        )
        operation_id = str(uuid4())
        recovery.request(
            {
                "resource": t.hardware_resource_id,
                "operation_id": operation_id,
                "command": "drain",
                "accept_until_ns": clock[0] + 100_000_000,
            }
        )
        for _ in range(10):
            recovery.tick()
        assert recovery.records[operation_id]["state"] == "completed"
        assert core.resource_ready(t.hardware_resource_id)
    finally:
        host.shutdown()


def test_unsubmitted_terminal_receipt_waits_for_delayed_submit_window():
    clock = [time.monotonic_ns()]
    t = target(retry_window_ms=100)
    core = RuntimeScheduler([t], now_ns=lambda: clock[0])
    host, executor = controller(t, now_ns=lambda: clock[0])
    try:
        connect(core, host)
        submission = prepare(host)
        report = host.cancel(submission, core.epoch)
        assert core.terminal_report(report) == {"state": "pending"}
        assert not core._records and not host._records[submission.operation_id].acknowledged
        # A delayed Submit remains possible, but the retained host cancellation
        # proves that its subsequent Execute cannot call the backend.
        core.submit(submission)
        core.dispatchable()
        assert host.execute(submission, core.epoch)["state"] == "not_started"
        receipt = core.terminal_report(report)
        assert receipt["state"] == "acknowledged"
        assert receipt["outcome"]["state"] == "not_started"
        assert receipt["outcome"]["error"] == "CANCELLED"
        host.acknowledge(submission, core.epoch)
        clock[0] = submission.accept_until_ns + 1
        for retained in tuple(host._records.values()):
            if retained.state in {"completed", "not_started"}:
                host.release_result_waiter(retained.submission)
        host.collect()
        assert not host._records and not executor.calls
    finally:
        host.shutdown()


@pytest.mark.parametrize("field", ["boot_id", "clock_id", "target_id", "generation", "state", "epoch"])
def test_expired_terminal_receipt_keeps_identity_and_certainty_fences(field):
    from inference_service.scheduler.runtime_ros import GlobalRuntimeEndpoint

    clock = [time.monotonic_ns()]
    t = target(retry_window_ms=100)
    core = RuntimeScheduler([t], now_ns=lambda: clock[0])
    host, _ = controller(t, now_ns=lambda: clock[0])
    endpoint = object.__new__(GlobalRuntimeEndpoint)
    endpoint.timer = object()
    endpoint.core = core
    try:
        connect(core, host)
        submission = prepare(host)
        report = host.cancel(submission, core.epoch)
        clock[0] = submission.accept_until_ns + 1
        message = {"verb": "terminal_report", "epoch": core.epoch, "report": report}
        if field in {"boot_id", "clock_id"}:
            report["submission"][field] = str(uuid4())
        elif field == "target_id":
            report["submission"][field] = "perception/other"
        elif field == "generation":
            report["submission"][field] += 1
        elif field == "state":
            report[field] = "unknown"
        else:
            message[field] = str(uuid4())
        with pytest.raises(ControlError):
            endpoint._handle(message)
        assert not host._records[submission.operation_id].acknowledged
    finally:
        host.shutdown()


def test_cancel_before_submit_reclaims_without_global_contact():
    from inference_service.scheduler.runtime_ros import SchedulerRuntimeAdapter

    clock = [time.monotonic_ns()]
    host, executor = controller(target(retry_window_ms=100), now_ns=lambda: clock[0])
    host.accept_epoch("test")
    adapter = object.__new__(SchedulerRuntimeAdapter)
    adapter._registration_error = ""
    adapter.controller, adapter.target = host, host.target
    adapter._call = lambda *_: pytest.fail("a cancelled caller must not submit or cancel at Global")
    context = ExecutionContext("never-submitted", Deadline.after(1))
    context.cancellation_token.cancel()
    try:
        with pytest.raises(ControlError):
            adapter.execute(ModelRequest(), context)
        for retained in tuple(host._records.values()):
            if retained.state in {"completed", "not_started"}:
                host.release_result_waiter(retained.submission)
        host.collect()
        assert not host._records and not executor.calls and host._input_bytes == 0
    finally:
        host.shutdown()


@pytest.mark.parametrize("failure", ["late_rejection", "malformed_reply"])
def test_submit_reply_failures_cannot_prove_an_admitted_operation_absent(failure):
    from concurrent.futures import TimeoutError as TransportTimeoutError

    from inference_service.scheduler.runtime_ros import SchedulerRuntimeAdapter, _reply

    host, _ = controller(target())
    core = RuntimeScheduler([host.target])
    adapter = object.__new__(SchedulerRuntimeAdapter)
    adapter._registration_error = ""
    adapter.controller, adapter.target = host, host.target
    attempts = []

    def call(verb, submission, timeout, **kwargs):
        if verb == "cancel":
            raise TimeoutError("lost Cancel reply")
        attempts.append(submission)
        if len(attempts) == 1:
            core.submit(submission)
            if failure == "late_rejection":
                raise TransportTimeoutError("lost Submit reply")
        payload = (
            '{"control_error":"OPERATION_EXPIRED"}'
            if failure == "late_rejection"
            else '{"state":"queued","state":"queued"}'
        )
        return _reply(SimpleNamespace(payload=payload))

    adapter._call = call
    try:
        connect(core, host)
        with pytest.raises(ControlError):
            adapter.execute(ModelRequest(), ExecutionContext("ambiguous", Deadline.after(1)))
        record = host._records[attempts[0].operation_id]
        assert record.state == "not_started" and not record.acknowledged
        assert len(attempts) == (2 if failure == "late_rejection" else 1)
        assert core.terminal_report(record.report())["state"] == "acknowledged"
    finally:
        host.shutdown()


def test_unavailable_global_before_send_reclaims_local_capacity():
    from inference_service.scheduler.runtime_ros import SchedulerRuntimeAdapter

    host, _ = controller(target())
    host.accept_epoch("test")
    adapter = object.__new__(SchedulerRuntimeAdapter)
    adapter._registration_error = ""
    adapter.controller, adapter.target = host, host.target
    adapter._client = SimpleNamespace(service_is_ready=lambda: False)
    try:
        with pytest.raises(ControlError, match="GLOBAL_UNAVAILABLE"):
            adapter.execute(ModelRequest(), ExecutionContext("unsent", Deadline.after(1)))
        record = next(iter(host._records.values()))
        assert record.acknowledged and record.state == "not_started"
        host.now_ns = lambda: max(record.submission.accept_until_ns, record.submission.deadline_ns) + 1
        for retained in tuple(host._records.values()):
            if retained.state in {"completed", "not_started"}:
                host.release_result_waiter(retained.submission)
        host.collect()
        assert not host._records
    finally:
        host.shutdown()


def test_terminal_receipt_transport_has_one_inflight_request_per_operation_and_removes_timeout(monkeypatch):
    from inference_service.scheduler import runtime_ros
    from inference_service.scheduler.runtime_ros import SchedulerRuntimeAdapter

    host, _ = controller(target())
    clock = [time.monotonic()]
    host.now_ns = lambda: int(clock[0] * 1e9)
    monkeypatch.setattr(runtime_ros, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    host.accept_epoch("test")
    adapter = object.__new__(SchedulerRuntimeAdapter)
    adapter._registration_error = ""
    adapter.controller = host
    adapter._terminal_receipts = {}
    sent, removed = [], []

    def call_async(request):
        future = Future()
        sent.append(future)
        return future

    adapter._client = SimpleNamespace(
        service_is_ready=lambda: True, call_async=call_async, remove_pending_request=removed.append
    )
    try:
        submission = prepare(host)
        host.cancel(submission, host.epoch)
        adapter._flush_terminal_evidence()
        for _ in range(5):
            adapter._flush_terminal_evidence()
        assert len(sent) == 1
        future = next(iter(adapter._terminal_receipts))
        clock[0] += runtime_ros.TERMINAL_RECEIPT_TIMEOUT_SECONDS + 0.01
        adapter._flush_terminal_evidence()
        assert removed == [future] and future not in adapter._terminal_receipts
        assert not host._records[submission.operation_id].acknowledged
        adapter._flush_terminal_evidence()
        assert len(sent) == 2
    finally:
        host.shutdown()


def test_orphan_terminal_evidence_eventually_unblocks_same_recovery():
    from inference_service.scheduler.runtime_protocol import decode_control, encode_control
    from inference_service.scheduler.runtime_recovery import ResourceRecovery

    clock = [time.monotonic_ns()]
    t = target(retry_window_ms=100)
    core = RuntimeScheduler([t], now_ns=lambda: clock[0])
    host, _ = controller(t, now_ns=lambda: clock[0])

    def call_async(request):
        message = decode_control(request.payload)
        args = (message["epoch"], message["operation_id"], message["command"], message["accept_until_ns"])
        kwargs = {
            "expected_generation": message.get("expected_generation"),
            "resume_until_ns": message.get("resume_until_ns"),
            "acknowledged": message.get("acknowledged", False),
        }
        reply = host.recover(*args, **kwargs)
        try:
            host._control_future.result(timeout=1)
        except ControlError as exc:
            assert exc.code == "RECOVERY_EVIDENCE_PENDING"
        future = Future()
        future.set_result(SimpleNamespace(payload=encode_control(reply)))
        return future

    recovery = ResourceRecovery(
        SimpleNamespace(
            core=core, _clients={t.target_id: SimpleNamespace(service_is_ready=lambda: True, call_async=call_async)}
        )
    )
    try:
        connect(core, host)
        submission = prepare(host)
        report = host.cancel(submission, core.epoch)
        assert core.terminal_report(report)["state"] == "pending"
        clock[0] += 50_000_000
        operation_id = str(uuid4())
        recovery.request(
            {
                "resource": "gpu",
                "operation_id": operation_id,
                "command": "drain",
                "accept_until_ns": clock[0] + 100_000_000,
            }
        )
        for _ in range(20):
            recovery.tick()
        record = recovery.records[operation_id]
        assert (record["state"], record["phase"]) == ("running", "activate")
        assert host.generation == 1
        assert len(host._recovery_records) <= 2
        activate_id = record["commands"][t.target_id]["id"]
        clock[0] = submission.accept_until_ns + 1
        assert core.terminal_report(report)["state"] == "acknowledged"
        host.acknowledge(submission, core.epoch)
        for retained in tuple(host._records.values()):
            if retained.state in {"completed", "not_started"}:
                host.release_result_waiter(retained.submission)
        host.collect()
        for _ in range(10):
            recovery.tick()
        assert record["state"] == "completed"
        assert record["commands"][t.target_id]["id"] == activate_id
        assert core._hosts[t.target_id][0]["generation"] == host.generation == 2
        assert core.resource_ready("gpu")
    finally:
        host.shutdown()


def test_reset_terminal_receipt_replay_advances_generation_once_even_after_collection():
    clock = [time.monotonic_ns()]
    t = target(retry_window_ms=100)
    core = RuntimeScheduler([t], now_ns=lambda: clock[0])
    host, _ = controller(t, now_ns=lambda: clock[0])
    try:
        connect(core, host)
        submission = prepare(host, work="reset")
        core.submit(submission)
        core.dispatchable()
        host.execute(submission, core.epoch)
        host._records[submission.operation_id].done.result(timeout=1)
        report = host.query(submission, core.epoch)
        assert host.generation == 2
        for _ in range(2):  # First receipt reply was lost.
            assert core.terminal_report(report)["state"] == "acknowledged"
            assert core._hosts[t.target_id][0]["generation"] == 2
        clock[0] = max(submission.accept_until_ns, submission.deadline_ns) + 1
        core.collect()
        assert not core._records
        assert core.terminal_report(report)["state"] == "acknowledged"
        assert not core._records and core._hosts[t.target_id][0]["generation"] == 2
        host.acknowledge(submission, core.epoch)
        for retained in tuple(host._records.values()):
            if retained.state in {"completed", "not_started"}:
                host.release_result_waiter(retained.submission)
        host.collect()
        assert not host._records
    finally:
        host.shutdown()


def test_pending_orphan_receipt_does_not_starve_admitted_terminal_evidence():
    from inference_service.scheduler.runtime_protocol import decode_control, encode_control
    from inference_service.scheduler.runtime_ros import SchedulerRuntimeAdapter

    host, _ = controller(target())
    core = RuntimeScheduler([host.target])
    adapter = object.__new__(SchedulerRuntimeAdapter)
    adapter._registration_error = ""
    adapter.controller = host
    adapter._terminal_receipts = {}
    sent = []

    def call_async(request):
        report = decode_control(request.payload)["report"]
        sent.append(report["submission"]["operation_id"])
        future = Future()
        future.set_result(SimpleNamespace(payload=encode_control(core.terminal_report(report))))
        return future

    adapter._client = SimpleNamespace(service_is_ready=lambda: True, call_async=call_async)
    try:
        connect(core, host)
        orphan, admitted = prepare(host), prepare(host)
        core.submit(admitted)
        host.cancel(orphan, core.epoch)
        host.cancel(admitted, core.epoch)
        for _ in range(3):
            adapter._flush_terminal_evidence()
        assert set(sent[:2]) == {orphan.operation_id, admitted.operation_id}
        assert not host._records[orphan.operation_id].acknowledged
        assert host._records[admitted.operation_id].acknowledged
        assert core._records[admitted.operation_id].host_acknowledged
    finally:
        host.shutdown()


@pytest.mark.parametrize("failure", ["expiry", "peer_error"])
def test_recovery_terminal_cleanup_removes_all_pending_ros_requests(failure):
    from inference_service.scheduler.runtime_recovery import ResourceRecovery

    clock = [time.monotonic_ns()]
    targets = [target("a", retry_window_ms=100), target("b", retry_window_ms=100)]
    core = RuntimeScheduler(targets, now_ns=lambda: clock[0])
    for t in targets:
        core._hosts[t.target_id] = ({"boot_id": str(uuid4())}, clock[0])

    class Client:
        def __init__(self):
            self.pending = []
            self.removed = []

        def service_is_ready(self):
            return True

        def call_async(self, request):
            future = Future()
            self.pending.append(future)
            future.add_done_callback(self.remove_pending_request)
            return future

        def remove_pending_request(self, future):
            if future in self.pending:
                self.pending.remove(future)
                self.removed.append(future)

    clients = {t.target_id: Client() for t in targets}
    recovery = ResourceRecovery(SimpleNamespace(core=core, _clients=clients))
    for _ in range(5):
        operation_id = str(uuid4())
        expiry = clock[0] + 100_000_000
        recovery.request(
            {"resource": "gpu", "operation_id": operation_id, "command": "drain", "accept_until_ns": expiry}
        )
        recovery.tick()
        assert all(len(client.pending) == 1 for client in clients.values())
        if failure == "peer_error":
            clients[targets[0].target_id].pending[0].set_exception(ControlError("FAILED_RECOVERY"))
        else:
            clock[0] = expiry + 1
        recovery.tick()
        assert recovery.records[operation_id]["state"] == "unknown"
        assert all(not client.pending for client in clients.values())
        assert all(command["future"] is None for command in recovery.records[operation_id]["commands"].values())
        clock[0] = expiry + 1
    # Transport is reclaimed, but uncertain phase receipts must survive their
    # windows so a caller can resume without repeating Reset/Close. The global
    # recovery capacity bounds retained evidence instead of expiring it.
    assert len(recovery.records) == 5
    assert all(len(client.removed) == 5 for client in clients.values())


def test_recovery_rechecks_unknown_after_running_call_returns():
    from inference_service.backends.errors import BackendInferenceError
    from inference_service.unified_runtime import CancellationToken

    host, executor = controller(target())
    core = RuntimeScheduler([host.target])
    cancelled = threading.Event()
    drained = []

    class ObservedCancellation(CancellationToken):
        def cancel(self, reason="cancelled"):
            super().cancel(reason)
            cancelled.set()

    try:
        connect(core, host)
        submission = host.prepare(ModelRequest(), ExecutionContext("parent", Deadline.after(2), ObservedCancellation()))
        executor.release.clear()
        executor.error = BackendInferenceError("device completion lost", operation_started=True, outcome_known=False)
        host.handle.drain_managed = lambda context, **kwargs: drained.append(True)
        host.execute(submission, core.epoch)
        assert executor.entered.wait(1)
        with ThreadPoolExecutor(max_workers=1) as worker:
            recovery = worker.submit(host._recover, "drain")
            until = time.monotonic() + 1
            while not host._records[submission.operation_id].context.cancellation_token.cancelled:
                assert time.monotonic() < until
                time.sleep(0.001)
            executor.release.set()
            recovery.result(timeout=2)
        assert drained == [True]
        assert host.query(submission, core.epoch)["state"] == "completed"
    finally:
        executor.release.set()
        host.shutdown()


def test_observed_old_epoch_requires_coordinated_restart():
    # This detects an already-reporting old Host; it is not a startup barrier.
    # Deployment must stop old Hosts before starting a new in-memory Global.
    t = target()
    core = RuntimeScheduler([t])
    host, _ = controller(t)
    try:
        connect(core, host)
        replacement = RuntimeScheduler([t])
        with pytest.raises(ControlError, match="COORDINATED_RESTART_REQUIRED"):
            replacement.register(host.status())
        status = host.status()
        status["boot_id"] = str(uuid4())
        with pytest.raises(ControlError, match="HOST_RESTART_REQUIRES_COORDINATED_RESTART"):
            core.register(status)
    finally:
        host.shutdown()


@pytest.mark.parametrize("later_gate", [None, "closing", "recovery"])
def test_successful_drain_restores_only_maintenance_and_preserves_generation(later_gate):
    executor = Executor()
    handle = ModelRuntimeHandle(executor=executor, resettable=True)
    handle.load()
    host = RuntimeExecutionController(target(), handle)
    peer, _ = controller(replace(target("peer"), lifecycle_group=host.target.lifecycle_group))
    host.join_lifecycle(host, [host, peer])
    peer.join_lifecycle(host, [host, peer])
    core = RuntimeScheduler([host.target, peer.target])
    connect(core, host)
    connect(core, peer)

    def drain():
        assert host._gate == peer._gate == "maintenance"
        if later_gate is not None:
            host._gate = peer._gate = later_gate

    executor.drain_uncertain_operations = drain
    try:
        submission = prepare(host, work="drain")
        core.submit(submission)
        assert core.dispatchable() == [(host.target, submission)]
        host.execute(submission, core.epoch)
        host._records[submission.operation_id].done.result(timeout=1)
        report = host.query(submission, core.epoch)
        assert report["state"] == "completed" and not report["error"]
        core.report(report)
        core.register(host.status())
        core.register(peer.status())
        assert host.generation == peer.generation == 1
        assert host._gate == peer._gate == (later_gate or "ready")
        assert not core.resource_ready(host.target.hardware_resource_id)
        assert host.handle.reset_required
        if later_gate is None:
            host.handle._reset_local(ExecutionContext("after-drain"), _dispatch_key=host._authority)
            core.register(host.status())
            next_request = prepare(peer)
            core.submit(next_request)
            assert core.dispatchable() == [(peer.target, next_request)]
            peer.execute(next_request, core.epoch)
            peer._records[next_request.operation_id].done.result(timeout=1)
            assert peer.result(next_request).outputs["answer"][0] == 1
        else:
            with pytest.raises(ControlError, match="RUNTIME_NOT_REGISTERED"):
                prepare(host)
            with pytest.raises(ControlError, match="RUNTIME_NOT_REGISTERED"):
                prepare(peer)
    finally:
        host.shutdown()
        peer.shutdown()


@pytest.mark.parametrize("work", ["reset", "drain"])
@pytest.mark.parametrize("later_gate", [None, "closing", "recovery"])
def test_expired_lifecycle_claim_releases_only_maintenance(work, later_gate):
    host, _ = controller(target())
    peer, _ = controller(target("peer"))
    host.join_lifecycle(host, [host, peer])
    peer.join_lifecycle(host, [host, peer])
    core = RuntimeScheduler([host.target, peer.target])
    connect(core, host)
    connect(core, peer)
    workers, queued = host._workers, []
    host._workers = SimpleNamespace(submit=lambda fn, record: queued.append((fn, record)))
    try:
        submission = prepare(host, work=work)
        host.execute(submission, core.epoch)
        assert host._gate == peer._gate == "maintenance"
        host.now_ns = lambda: submission.deadline_ns + 1
        if later_gate:
            peer._gate = later_gate
        fn, record = queued.pop()
        fn(record)
        assert record.state == "not_started"
        assert host.status()["ready"]
        assert peer._gate == (later_gate or "ready")
        assert host.generation == peer.generation == 1
        assert host._running == 0
        if later_gate is None:
            assert prepare(host).generation == 1
            assert prepare(peer).generation == 1
    finally:
        host._workers = workers
        host.shutdown()
        peer.shutdown()


@pytest.mark.parametrize("work", ["execute", "reset"])
@pytest.mark.parametrize("coverage", [False, True])
def test_terminal_host_evidence_survives_late_replies_once(work, coverage):
    t = target()
    core = RuntimeScheduler([t], resources={"gpu": {"physical_device": "local/gpu", "managed_coverage": coverage}})
    host, executor = controller(t)
    try:
        connect(core, host)
        sub = prepare(host, work=work)
        core.submit(sub)
        core.dispatchable()
        executor.release.clear()
        # Keep an actual Execute reply while the host completes the operation.
        running = host.execute(sub, core.epoch)
        executor.release.set()
        host._records[sub.operation_id].done.result(timeout=1)
        terminal = host.query(sub, core.epoch)
        assert terminal["state"] == "completed"
        core.report(terminal)
        generation = core._hosts[t.target_id][0]["generation"]
        assert len(core.measurements) == int(coverage)
        for late in (running, {**terminal, "state": "prepared"}, {**terminal, "state": "unknown"}, terminal):
            assert core.report(late)["state"] == "completed"
        assert core.resource_ready(t.hardware_resource_id)
        assert len(core.measurements) == int(coverage)
        assert core._hosts[t.target_id][0]["generation"] == generation
    finally:
        executor.release.set()
        host.shutdown()


@pytest.mark.parametrize("command", ["drain", "reset", "close"])
def test_clean_optional_unavailable_recovery_owns_no_backend(command):
    from inference_service.scheduler.runtime_controller import UnavailableRuntime

    host = RuntimeExecutionController(target(required=False), UnavailableRuntime())
    try:
        host._gate = "closing" if command == "close" else "recovery"
        host._recover(command)
        if command == "close":
            with pytest.raises(ControlError, match="RECOVERY_EVIDENCE_PENDING"):
                host._recover("activate", 2)
        else:
            host._recover("activate", 2)
            assert host.generation == 2
            assert host.status()["clean"]
            assert not host.status()["ready"]
    finally:
        host.shutdown()


@pytest.mark.parametrize("case", ["required", "dirty", "closing", "unacknowledged"])
def test_optional_recovery_never_waives_missing_evidence(case):
    from inference_service.scheduler.runtime_controller import UnavailableRuntime

    host = RuntimeExecutionController(target(required=case == "required"), UnavailableRuntime())
    try:
        host.startup_clean = case != "dirty"
        host._gate = "closing" if case == "closing" else "recovery"
        if case == "unacknowledged":
            host._records["unacknowledged"] = SimpleNamespace(
                state="not_started", acknowledged=False, context=ExecutionContext.create("unacknowledged")
            )
        with pytest.raises(ControlError, match="RECOVERY_EVIDENCE_PENDING"):
            host._recover("activate", 2)
    finally:
        host.shutdown()


@pytest.mark.parametrize("stages", [1, 2])
def test_adapter_reset_refreshes_maintenance_heartbeat_before_next_request(stages, monkeypatch):
    from inference_service.scheduler import runtime_ros
    from inference_service.scheduler.runtime_ros import SchedulerRuntimeAdapter

    from .runtime_scheduler_helpers import FakeNode

    monkeypatch.setattr(runtime_ros, "_service_type", lambda: SimpleNamespace(Request=SimpleNamespace))
    targets = [replace(target(f"stage{i}", resource="cpu"), lifecycle_group="policy") for i in range(stages)]
    core = RuntimeScheduler(targets)
    handle = ModelRuntimeHandle(executor=Executor())
    handle.load()
    adapter = SchedulerRuntimeAdapter(FakeNode({}), targets[0], handle)
    hosts = [adapter.controller, *(controller(t)[0] for t in targets[1:])]
    entered, release = threading.Event(), threading.Event()

    def reset(_context, **kwargs):
        entered.set()
        assert release.wait(2)

    hosts[0].handle._reset_local = reset
    for host in hosts:
        host.join_lifecycle(hosts[0], hosts)
        connect(core, host)

    def call(verb, submission, _timeout, **kwargs):
        if verb == "submit":
            core.submit(submission)
            assert core.dispatchable()
            hosts[0].execute(submission, core.epoch)
            assert entered.wait(1)
            for host in hosts:
                core.register(host.status())
            assert not core.resource_ready("cpu")
            release.set()
            hosts[0]._records[submission.operation_id].done.result(timeout=1)
            core.report(hosts[0].query(submission, core.epoch))
            assert not core.resource_ready("cpu")
        return core.query(submission)

    registrations = []

    def control(message, _timeout, **kwargs):
        registrations.append(message["status"])
        return core.register(message["status"])

    adapter._call = call
    adapter._call_control = control
    try:
        adapter.reset(ExecutionContext("reset-with-heartbeat", Deadline.after(1)))
        assert {s["target"]["target_id"] for s in registrations} == {t.target_id for t in targets}
        assert all(s["ready"] and s["generation"] == 2 for s in registrations)
        assert core.resource_ready("cpu")
        for host in hosts:
            assert core.submit(prepare(host))["state"] == "queued"
    finally:
        release.set()
        for host in hosts:
            host.shutdown()


def test_reset_refresh_cannot_invent_readiness_and_obeys_deadline():
    from inference_service.scheduler.runtime_ros import SchedulerRuntimeAdapter

    t = target()
    core = RuntimeScheduler([t])
    host, _ = controller(t)
    connect(core, host)
    host.handle.mark_reset_required()
    adapter = object.__new__(SchedulerRuntimeAdapter)
    adapter._registration_error = ""
    adapter.controller, adapter.target = host, t
    adapter._call_control = lambda message, _timeout, **kwargs: core.register(message["status"])
    try:
        with pytest.raises(Exception, match="deadline|REGISTRATION_READINESS_TIMEOUT"):
            adapter._refresh_after_reset(ExecutionContext("not-ready", Deadline.after(0.03)))
        assert not core.resource_ready("gpu")
        assert not host.handle.ready
    finally:
        host.shutdown()


def test_observation_declaration_fails_closed_and_does_not_affect_control_work():
    host, _ = controller(target(profile_observation_inputs=("image",)))
    host.accept_epoch("epoch")
    try:
        with pytest.raises(ControlError, match="PROFILE_OBSERVATION_INPUT_MISSING"):
            host.prepare(ModelRequest({"other": np.zeros(3)}), ExecutionContext("missing", Deadline.after(1)))
        with pytest.raises(ControlError, match="PROFILE_OBSERVATION_INPUT_NOT_TENSOR"):
            host.prepare(ModelRequest({"image": 3}), ExecutionContext("scalar", Deadline.after(1)))
        reset = host.prepare(ModelRequest(), ExecutionContext("reset", Deadline.after(1)), work="reset")
        assert reset.work == "reset"
    finally:
        host.shutdown()


def test_terminal_notification_only_reads_pending_evidence():
    class History(dict):
        def values(self):
            pytest.fail("notification scanned replay history")

    host, _ = controller(target())
    host.accept_epoch("epoch")
    try:
        submission = prepare(host)
        host.cancel(submission, "epoch")
        assert host.pending_terminal_reports(limit=1)[0]["submission"] == submission.to_dict()
        host.acknowledge(submission, "epoch")
        history = host._records
        host._records = History(history)
        for _ in range(100):
            assert not host.pending_terminal_reports(limit=1)
        host._records = history
    finally:
        host.shutdown()


def test_resource_settlement_ignores_replay_history_but_waits_for_terminal_ack():
    class History(dict):
        def values(self):
            pytest.fail("recovery scanned settled replay history")

    host, _ = controller(target())
    core = RuntimeScheduler([host.target])
    connect(core, host)
    try:
        submission = prepare(host)
        core.submit(submission)
        core.report(host.cancel(submission, core.epoch))
        core._records = History(core._records)
        assert not core.resource_settled(host.target.hardware_resource_id)
        assert core.resource_settled("another-device")
        core.acknowledge_host(submission)
        assert core.resource_settled(host.target.hardware_resource_id)
    finally:
        host.shutdown()


def test_terminal_rotation_skips_backoff_and_preserves_fairness():
    clock = [time.monotonic_ns()]
    host, _ = controller(target(), now_ns=lambda: clock[0])
    host.accept_epoch("epoch")
    try:
        submissions = [prepare(host) for _ in range(3)]
        for submission in submissions:
            host.cancel(submission, "epoch")
        reports = [host.pending_terminal_reports(limit=1, retry_interval_ns=100)[0] for _ in range(3)]
        assert {report["submission"]["operation_id"] for report in reports} == {
            item.operation_id for item in submissions
        }
        assert not host.pending_terminal_reports(limit=1, retry_interval_ns=100)
        host.acknowledge(submissions[1], "epoch")
        clock[0] += 100
        # A bounded backoff scan may rotate part of the queue, but every
        # unacknowledged operation must still be visited once per full round.
        assert {report["submission"]["operation_id"] for report in host.pending_terminal_reports()} == {
            submissions[0].operation_id,
            submissions[2].operation_id,
        }
    finally:
        host.shutdown()


def test_terminal_rotation_bounds_scanning_during_backoff():
    from collections import OrderedDict

    class PendingEvidence(OrderedDict):
        scans = 0

        def move_to_end(self, key, last=True):
            self.scans += 1
            super().move_to_end(key, last=last)

    host, _ = controller(target(max_records=1024))
    host.accept_epoch("epoch")
    try:
        for _ in range(513):
            host.cancel(prepare(host), "epoch")
        host.pending_terminal_reports(retry_interval_ns=10**12)
        host._terminal_pending = PendingEvidence(host._terminal_pending)
        assert not host.pending_terminal_reports(limit=16, retry_interval_ns=10**12)
        assert host._terminal_pending.scans <= 16
        newest = prepare(host)
        host.cancel(newest, "epoch")
        reports = host.pending_terminal_reports(limit=16, retry_interval_ns=10**12)
        assert [report["submission"] for report in reports] == [newest.to_dict()]
    finally:
        host.shutdown()


@pytest.mark.parametrize("not_started", [True, False])
def test_control_transport_preserves_reset_non_start_evidence(not_started):
    from inference_service.scheduler.runtime_protocol import decode_control, encode_control
    from inference_service.scheduler.runtime_ros import _serve
    from inference_service.unified_runtime.contracts import OutcomeEvidence
    from inference_service.unified_runtime.errors import ExecutionFailure

    def fail(_):
        raise ExecutionFailure(
            code="deadline_exceeded",
            message="reset deadline expired",
            evidence=OutcomeEvidence.not_started("control_lock") if not_started else OutcomeEvidence.started("reset"),
        )

    reply = _serve(fail)(SimpleNamespace(payload=encode_control({"verb": "recover"})), SimpleNamespace())
    payload = decode_control(reply.payload)
    assert payload["control_error"] == "deadline_exceeded"
    assert (payload.get("not_started") is True) is not_started


@pytest.mark.parametrize("command", ["reset", "reset_stream"])
@pytest.mark.parametrize("not_started", [True, False])
def test_reset_replay_only_retries_proven_non_start(monkeypatch, command, not_started):
    from inference_service.unified_runtime.contracts import OutcomeEvidence
    from inference_service.unified_runtime.errors import ExecutionFailure

    host, _ = controller(target())
    host.accept_epoch("epoch")
    calls = []

    def recover(*_):
        calls.append(command)
        if len(calls) == 1:
            raise ExecutionFailure(
                code="deadline_exceeded",
                message="reset deadline expired",
                evidence=OutcomeEvidence.not_started("control_lock")
                if not_started
                else OutcomeEvidence.started("reset"),
            )

    monkeypatch.setattr(host, "_recover", recover)
    operation, expiry = str(uuid4()), host.now_ns() + 1_000_000_000
    kwargs = {"stream_id": "stream"} if command == "reset_stream" else {}
    try:
        host.recover("epoch", operation, command, expiry, **kwargs)
        host._control_future.result(timeout=1)
        if not_started:
            host.recover("epoch", operation, command, expiry, **kwargs)
            host._control_future.result(timeout=1)
            assert host.recover("epoch", operation, command, expiry, **kwargs)["state"] == "completed"
            assert len(calls) == 2
        else:
            with pytest.raises(ExecutionFailure):
                host.recover("epoch", operation, command, expiry, **kwargs)
            assert len(calls) == 1
    finally:
        host.shutdown()


def test_activation_preserves_reset_requirement_without_holding_resource_recovery():
    host, _ = controller(target())
    try:
        host.handle.mark_reset_required()
        host._gate = "recovery"
        host._recover("activate", 2)
        assert host.generation == 2 and not host.handle.local_ready
        assert not host.status()["ready"] and host._gate == "recovery"
        with pytest.raises(ControlError):
            prepare(host)
    finally:
        host.shutdown()


def test_recovery_failures_replay_without_retaining_backend_frames(monkeypatch):
    import gc
    import weakref

    clock = [time.monotonic_ns()]
    host, _ = controller(target(), now_ns=lambda: clock[0])
    references = []

    def fail(*_):
        temporary = np.zeros(1_000_000)
        references.append(weakref.ref(temporary))
        try:
            raise ValueError("backend cause")
        except ValueError as cause:
            error = RuntimeError("drain failed")
            error.buffer = temporary
            raise error from cause

    monkeypatch.setattr(host, "_recover", fail)
    host.accept_epoch("test")
    operation = str(uuid4())
    expiry = clock[0] + 1_000_000_000
    try:
        host.recover("test", operation, "drain", expiry)
        snapshot = host._control_future.result(timeout=1)
        errors = []
        for _ in range(20):
            with pytest.raises(Exception, match="drain failed") as caught:
                host.recover("test", operation, "drain", expiry)
            errors.append(caught.value)
        assert len({id(error) for error in errors}) == 20
        assert snapshot.__traceback__ is snapshot.__cause__ is snapshot.__context__ is None
        assert all(error.__cause__ is None and error.__context__ is None for error in errors)
        gc.collect()
        assert references[0]() is None
        clock[0] = expiry + 1
        host.collect()
        assert operation in host._recovery_records and host._control_future is None
        host.recover("test", operation, "drain", expiry, acknowledged=True)
        host.collect()
        assert not host._recovery_records
    finally:
        host.shutdown()


def test_recovery_submit_failure_fences_enqueued_work_and_can_retry(monkeypatch):
    from inference_service.scheduler.runtime_protocol import NotStartedControlError

    host, _ = controller(target())
    host.accept_epoch("test")
    original_submit = host._recovery.submit
    queued, calls = [], []

    def submit(function, *args):
        if not queued:
            queued.append((function, args))
            raise RuntimeError("cannot start new thread")
        return original_submit(function, *args)

    monkeypatch.setattr(host._recovery, "submit", submit)
    monkeypatch.setattr(host, "_recover", lambda *args: calls.append(args[0]))
    operation, expiry = str(uuid4()), host.now_ns() + 1_000_000_000
    try:
        with pytest.raises(NotStartedControlError, match="RECOVERY_WORKER_SUBMIT_FAILED"):
            host.recover("test", operation, "drain", expiry)
        assert host._gate == "ready" and operation not in host._recovery_records
        assert queued[0][1][0].cancelled()
        host.recover("test", operation, "drain", expiry)
        host._control_future.result(timeout=1)
        function, args = queued[0]
        function(*args)  # The thread pool eventually dequeues the stranded work.
        assert calls == ["drain"]
        assert host.recover("test", operation, "drain", expiry)["state"] == "completed"
    finally:
        host.shutdown()


def test_unknown_recovery_capacity_isolated_between_devices():
    from inference_service.scheduler.runtime_recovery import MAX_RESOURCE_RECOVERY_RECEIPTS, ResourceRecovery

    clock = [time.monotonic_ns()]
    targets = [target("a", "gpu"), target("b", "cpu")]
    core = RuntimeScheduler(targets, now_ns=lambda: clock[0])
    hosts = [controller(t, now_ns=lambda: clock[0])[0] for t in targets]
    from inference_service.scheduler.runtime_protocol import decode_control, encode_control

    host = hosts[1]

    def call_async(request):
        message = decode_control(request.payload)
        args = (message["epoch"], message["operation_id"], message["command"], message["accept_until_ns"])
        kwargs = {
            "expected_generation": message.get("expected_generation"),
            "resume_until_ns": message.get("resume_until_ns"),
            "acknowledged": message.get("acknowledged", False),
        }
        host.recover(*args, **kwargs)
        host._control_future.result(timeout=1)
        response = host.recover(*args, **kwargs)
        future = Future()
        future.set_result(SimpleNamespace(payload=encode_control(response)))
        return future

    recovery = ResourceRecovery(
        SimpleNamespace(
            core=core,
            _clients={targets[1].target_id: SimpleNamespace(service_is_ready=lambda: True, call_async=call_async)},
        )
    )
    try:
        for participant in hosts:
            connect(core, participant)
        for _ in range(MAX_RESOURCE_RECOVERY_RECEIPTS - 1):
            recovery.request(
                {
                    "resource": "gpu",
                    "operation_id": str(uuid4()),
                    "command": "drain",
                    "accept_until_ns": clock[0] + 1_000_000,
                }
            )
            clock[0] += 1_000_001
            recovery.tick()
        assert len(recovery.records) == MAX_RESOURCE_RECOVERY_RECEIPTS - 1
        assert all(record["state"] == "unknown" for record in recovery.records.values())
        with pytest.raises(ControlError, match="RECOVERY_BUSY"):
            recovery.request(
                {
                    "resource": "gpu",
                    "operation_id": str(uuid4()),
                    "command": "drain",
                    "accept_until_ns": clock[0] + 1_000_000,
                }
            )
        operation = str(uuid4())
        assert (
            recovery.request(
                {
                    "resource": "cpu",
                    "operation_id": operation,
                    "command": "drain",
                    "accept_until_ns": clock[0] + 1_000_000,
                }
            )["state"]
            == "running"
        )
        assert len(recovery.records) == MAX_RESOURCE_RECOVERY_RECEIPTS
        for _ in range(12):
            recovery.tick()
        assert recovery.records[operation]["state"] == "completed"
        core.register(host.status())
        assert core.resource_ready("cpu") and not core.resource_ready("gpu")
    finally:
        for host in hosts:
            host.shutdown()
