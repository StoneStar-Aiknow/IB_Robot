"""Transport faults against real controller/queue instances and an in-process bus."""

import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from inference_service.scheduler.runtime_protocol import (
    ControlError,
    NotStartedControlError,
    decode_control,
    encode_control,
)
from inference_service.scheduler.runtime_ros import GlobalRuntimeEndpoint, SchedulerRuntimeAdapter, _serve
from inference_service.unified_runtime import (
    Deadline,
    ExecutionContext,
    ExecutionFailure,
    ModelRequest,
    ModelRuntimeHandle,
)
from inference_service.unified_runtime.errors import CancellationRequested, DeadlineExceeded

from .runtime_scheduler_helpers import Executor, FakeNode, connect, prepare, target


def test_ros_endpoint_enforces_production_sample_threshold():
    with pytest.raises(ValueError, match="profile_min_samples"):
        GlobalRuntimeEndpoint(FakeNode({}), [], profile_min_samples=20)


@pytest.mark.parametrize("failure_at", ["client", "service", "timer"])
def test_adapter_entity_failure_leaves_handle_unbound_and_retryable(monkeypatch, failure_at):
    node = FakeNode({})
    handle = ModelRuntimeHandle(executor=Executor())
    handle.load()
    original = getattr(node, "create_" + failure_at)
    destroyed = []
    for kind in ("client", "service", "timer"):
        destroy = getattr(node, "destroy_" + kind)

        def remove(entity, kind=kind, destroy=destroy):
            destroyed.append(kind)
            destroy(entity)

        monkeypatch.setattr(node, "destroy_" + kind, remove)

    def fail(*args, **kwargs):
        raise RuntimeError("ROS entity failed")

    monkeypatch.setattr(node, "create_" + failure_at, fail)
    adapter = None
    try:
        with pytest.raises(RuntimeError, match="ROS entity failed"):
            SchedulerRuntimeAdapter(node, target(resource="cpu"), handle)
        assert not handle.execution_policy_bound
        assert handle.execute(ModelRequest(), ExecutionContext("direct-retry")).outputs["answer"].tolist() == [1]
        assert not node.bus
        assert destroyed == {"client": [], "service": ["client"], "timer": ["service", "client"]}[failure_at]
        monkeypatch.setattr(node, "create_" + failure_at, original)
        adapter = SchedulerRuntimeAdapter(node, target(resource="cpu"), handle)
        assert handle.execution_policy_bound
    finally:
        if adapter is not None:
            adapter.controller.shutdown()
            handle._close_local(ExecutionContext("cleanup"), _dispatch_key=adapter.controller._authority)
            adapter.destroy_transport()
        else:
            handle.close()


def test_shutdown_heartbeat_revokes_adapter_readiness_and_allows_close(transport):
    endpoint, adapter, _ = transport
    adapter._heartbeat()
    adapter._heartbeat()
    assert adapter.ready
    endpoint.core.begin_shutdown()
    assert endpoint.core.unavailable_resources() == ("cpu",)
    adapter._heartbeat()
    adapter._heartbeat()
    assert not adapter.ready and not adapter.controller.handle.ready
    sub = adapter.controller.prepare(ModelRequest(), ExecutionContext("close"), work="close")
    endpoint.core.submit(sub)
    assert endpoint.core.dispatchable() == [(adapter.target, sub)]


def test_transport_release_requires_local_evidence_global_retirement_and_quiescent_node(transport):
    endpoint, adapter, _ = transport
    host = adapter.controller
    adapter._heartbeat()
    adapter._heartbeat()
    sub = prepare(host)
    host.cancel(sub, host.epoch)
    with pytest.raises(ControlError, match="TERMINAL_EVIDENCE_PENDING"):
        adapter.destroy_transport()
    host.acknowledge(sub, host.epoch)
    with pytest.raises(ControlError, match="RUNTIME_CLOSE_REQUIRED"):
        adapter.destroy_transport()
    host.shutdown()
    host.handle._close_local(ExecutionContext("local-cleanup"), _dispatch_key=host._authority)
    with pytest.raises(ControlError, match="RUNTIME_RETIREMENT_REQUIRED"):
        adapter.destroy_transport()
    assert not getattr(adapter, "_transport_destroyed", False)
    # Local ACK alone has not settled Global's membership. The normal Close
    # path owns that barrier; then ROS wait-set quiescence remains mandatory.
    adapter.node.context = object()
    adapter.node.executor = object()
    adapter.close()
    assert not endpoint.core.shutdown_pending()
    with pytest.raises(RuntimeError, match="quiescent"):
        adapter.destroy_transport()
    adapter.node.executor = None
    adapter.destroy_transport()
    assert adapter._transport_destroyed


@pytest.mark.parametrize("verb", ["cancel", "ack"])
def test_control_retries_are_bounded_despite_immediate_replies(transport, monkeypatch, verb):
    import time

    from inference_service.scheduler import runtime_ros

    endpoint, adapter, _ = transport
    host, core = adapter.controller, endpoint.core
    clock = [time.monotonic()]
    monkeypatch.setattr(
        runtime_ros, "time", SimpleNamespace(monotonic=lambda: clock[0], monotonic_ns=lambda: int(clock[0] * 1e9))
    )
    sub = prepare(host)
    core.submit(sub)
    core.dispatchable()
    if verb == "cancel":
        core.cancel(sub)
    else:
        core.report({"submission": sub.to_dict(), "state": "completed"})
    calls = []
    finish = [False]

    def reply(request):
        sent = decode_control(request.payload)
        calls.append(sent["verb"])
        if not finish[0]:
            response = (
                {"submission": sub.to_dict(), "state": "running"}
                if verb == "cancel"
                else {"control_error": "RETRY_ACK"}
            )
        elif sent["verb"] == "cancel":
            response = {"submission": sub.to_dict(), "state": "completed"}
        else:
            response = {"state": "acknowledged"}
        future = Future()
        future.set_result(SimpleNamespace(payload=encode_control(response)))
        return future

    monkeypatch.setattr(endpoint._clients[host.target.target_id], "call_async", reply)
    endpoint._tick()
    assert calls == [verb]  # First cleanup request is immediate.
    for _ in range(3):
        for _ in range(100):
            endpoint._tick()  # Model response-driven guard wakes at one clock instant.
        assert not core._records[sub.operation_id].host_acknowledged
        clock[0] += 0.25
        endpoint._tick()
    assert calls == [verb] * 4
    finish[0] = True
    clock[0] += 0.25
    for _ in range(4):
        endpoint._tick()
    assert core.query(sub)["state"] == "completed"
    assert core._records[sub.operation_id].host_acknowledged


@pytest.fixture
def transport():
    bus = {}
    t = target(resource="cpu")
    model = Executor()
    handle = ModelRuntimeHandle(executor=model, resettable=True)
    handle.load()
    endpoint = GlobalRuntimeEndpoint(FakeNode(bus), [t])
    adapter = SchedulerRuntimeAdapter(FakeNode(bus), t, handle)
    connect(endpoint.core, adapter.controller)
    try:
        yield endpoint, adapter, model
    finally:
        adapter.controller.shutdown()
        handle._close_local(ExecutionContext("test-cleanup"), _dispatch_key=adapter.controller._authority)


def test_first_execute_without_discovered_service_is_not_started(transport, monkeypatch):
    import time

    from inference_service.scheduler import runtime_ros

    clock = [time.monotonic()]
    monkeypatch.setattr(
        runtime_ros, "time", SimpleNamespace(monotonic=lambda: clock[0], monotonic_ns=lambda: int(clock[0] * 1e9))
    )
    endpoint, adapter, model = transport
    host, core = adapter.controller, endpoint.core
    sub = prepare(host)
    core.submit(sub)
    client = endpoint._clients[host.target.target_id]
    client.available = False
    endpoint._tick()
    assert core.query(sub)["state"] == "not_started"
    assert not model.calls and not core._fenced
    assert core.pending_reports() == [(host.target, sub, "cancel")]
    client.available = True
    clock[0] += 0.25  # Service recovery does not bypass the Cancel retry interval.
    for _ in range(3):
        endpoint._tick()
    assert host.query(sub, core.epoch)["state"] == "not_started"
    assert host._records[sub.operation_id].acknowledged
    assert core._records[sub.operation_id].host_acknowledged
    next_sub = prepare(host)
    core.submit(next_sub)
    assert core.dispatchable() == [(host.target, next_sub)]


@pytest.mark.parametrize("reply_kind", ["rejection", "terminal"])
def test_global_rejection_does_not_cancel_callers_context(transport, monkeypatch, reply_kind):
    from inference_service.scheduler.runtime_ros import _RemoteControlError

    _, adapter, model = transport
    context = ExecutionContext("retryable-business", Deadline.after(2))

    def reject(*args, **kwargs):
        if reply_kind == "rejection":
            raise _RemoteControlError("NO_FEASIBLE_DEADLINE")
        return {"state": "not_started", "error": "NO_FEASIBLE_DEADLINE"}

    monkeypatch.setattr(adapter, "_call", reject)
    with pytest.raises(NotStartedControlError, match="NO_FEASIBLE_DEADLINE"):
        adapter.execute(ModelRequest(), context)
    context.check("fallback")
    assert not model.calls


@pytest.mark.parametrize("rejection", ["closing", "backend", "missing_snapshot"])
def test_execute_preflight_and_missing_snapshot_preserve_not_started(transport, monkeypatch, rejection):
    from inference_service.scheduler.runtime_protocol import ControlError

    endpoint, adapter, model = transport
    host, core = adapter.controller, endpoint.core
    sub = prepare(host)
    if rejection == "missing_snapshot":
        sub = replace(sub, operation_id=str(uuid4()), request_ref=str(uuid4()))
    core.submit(sub)
    if rejection == "closing":
        monkeypatch.setattr(adapter, "_closing", True)
        expected = "RUNTIME_CLOSING"
    elif rejection == "backend":
        expected = "BACKEND_PRIORITY_UNAVAILABLE"

        def fail(_):
            raise ControlError(expected)

        monkeypatch.setattr(adapter, "_validate_backend", fail)
    else:
        expected = "REQUEST_REF_EXPIRED"
        # A valid reference that this Host does not own has never been run.
    endpoint._tick()  # Execute request reaches the real Host handler.
    endpoint._tick()  # Its error envelope reaches Global.
    report = core.query(sub)
    assert report["state"] == "not_started" and report["error"] == expected
    assert not core._fenced and not model.calls
    monkeypatch.undo()
    for _ in range(4):
        endpoint._tick()
    next_sub = prepare(host)
    core.submit(next_sub)
    assert core.dispatchable() == [(host.target, next_sub)]


@pytest.mark.parametrize("state", ["running", "completed"])
def test_replayed_execute_preserves_evidence_when_host_is_closing(transport, monkeypatch, state):
    endpoint, adapter, model = transport
    host, core = adapter.controller, endpoint.core
    sub = prepare(host)
    model.release.clear()
    try:
        host.execute(sub, core.epoch)
        assert model.entered.wait(1)
        if state == "completed":
            model.release.set()
            host.completion(sub).result(timeout=1)
        monkeypatch.setattr(adapter, "_closing", True)
        report = adapter._handle({"verb": "execute", "epoch": core.epoch, "submission": sub.to_dict()})
        assert report["state"] == state and len(model.calls) == 1
    finally:
        model.release.set()
        host.completion(sub).result(timeout=1)


@pytest.mark.parametrize("timeout_ns", [0, -1, True, 0.5, 2**63])
def test_control_rpc_timeout_rejects_invalid_budgets(timeout_ns):
    with pytest.raises(ValueError, match="control RPC timeout"):
        GlobalRuntimeEndpoint(FakeNode({}), [], control_rpc_timeout_ns=timeout_ns)


@pytest.mark.parametrize("rejection", ["capacity", "terminal"])
def test_global_rejection_preserves_callers_context_for_retry(transport, monkeypatch, rejection):
    endpoint, adapter, model = transport
    host, core = adapter.controller, endpoint.core
    context = ExecutionContext("reusable-business-context", Deadline.after(2))
    if rejection == "capacity":
        core.targets[host.target.target_id] = replace(host.target, max_pending_requests=1)
        core.submit(prepare(host))
        expected = "REQUEST_CAPACITY"
    else:
        monkeypatch.setattr(core, "submit", core.cancel)
        expected = "CANCELLED"
    with pytest.raises(NotStartedControlError, match=expected):
        adapter.execute(ModelRequest(), context)
    context.check("retry-after-rejection")
    assert not model.calls
    retry = host.prepare(ModelRequest(), context)
    assert host.query(retry, core.epoch)["state"] == "prepared"


@pytest.mark.parametrize("failure", ["closing", "backend"])
def test_explicit_execute_rejection_does_not_fence_resource(transport, monkeypatch, failure):
    endpoint, adapter, model = transport
    host, core = adapter.controller, endpoint.core
    sub = prepare(host)
    core.submit(sub)
    if failure == "closing":
        monkeypatch.setattr(adapter, "_closing", True)
        expected = "RUNTIME_CLOSING"
    else:

        def reject(_handle):
            raise ControlError("INSTANCE_CAPACITY_MISMATCH")

        monkeypatch.setattr(adapter, "_validate_backend", reject)
        expected = "INSTANCE_CAPACITY_MISMATCH"
    endpoint._tick()  # Execute is rejected through the real service envelope.
    endpoint._tick()
    assert core.query(sub) == {"state": "not_started", "error": expected, "epoch": core.epoch}
    assert not core._fenced and not model.calls
    for _ in range(3):
        endpoint._tick()
    assert host._records[sub.operation_id].acknowledged
    assert core._records[sub.operation_id].host_acknowledged
    next_sub = prepare(host)
    core.submit(next_sub)
    assert core.dispatchable() == [(host.target, next_sub)]


@pytest.mark.parametrize("evidence", [None, False, "true", 1])
def test_execute_error_without_strict_not_started_evidence_stays_unknown(transport, monkeypatch, evidence):
    endpoint, adapter, _ = transport
    host, core = adapter.controller, endpoint.core
    sub = prepare(host)
    core.submit(sub)

    def reply(_request):
        payload = {"control_error": "EXECUTION_UNCERTAIN"}
        if evidence is not None:
            payload["not_started"] = evidence
        future = Future()
        future.set_result(SimpleNamespace(payload=encode_control(payload)))
        return future

    monkeypatch.setattr(endpoint._clients[host.target.target_id], "call_async", reply)
    endpoint._tick()
    endpoint._tick()
    assert core.query(sub)["state"] == "unknown"
    assert not core.resource_ready(host.target.hardware_resource_id)


@pytest.mark.parametrize("verb", ["execute", "query", "cancel"])
def test_control_rpc_timeout_is_independent_of_healthy_status_lease(transport, monkeypatch, verb):
    import time

    from inference_service.scheduler import runtime_ros

    endpoint, adapter, _ = transport
    host, core = adapter.controller, endpoint.core
    now = [time.monotonic_ns()]
    monkeypatch.setattr(
        runtime_ros, "time", SimpleNamespace(monotonic=lambda: now[0] / 1e9, monotonic_ns=lambda: now[0])
    )
    monkeypatch.setattr(core, "now_ns", lambda: now[0])
    monkeypatch.setattr(host, "now_ns", lambda: now[0])
    core.stale_ns = 250_000_000
    endpoint.control_rpc_timeout_ns = 1_000_000_000
    sub = host.prepare(ModelRequest(), ExecutionContext("slow-control", Deadline.after(5)))
    core.submit(sub)
    core.dispatchable()
    pending = Future()
    monkeypatch.setattr(endpoint._clients[host.target.target_id], "call_async", lambda _: pending)
    endpoint._send(host.target, sub, verb)
    now[0] += 250_000_000  # Query's initial reconciliation delay.
    endpoint._send(host.target, sub, verb)
    assert (sub.operation_id, verb) in endpoint._pending
    now[0] += 300_000_000  # Beyond the status lease, within the RPC budget.
    core.register(host.status())
    endpoint._tick()
    assert core.query(sub)["state"] == "dispatching" and not core._fenced
    now[0] += 1_000_000_001
    core.register(host.status())
    endpoint._tick()
    assert core.query(sub)["state"] == "unknown"
    assert not core.resource_ready(host.target.hardware_resource_id)


def test_call_async_failure_stays_unknown_and_is_diagnosable(transport, monkeypatch, caplog):
    endpoint, adapter, model = transport
    host, core = adapter.controller, endpoint.core
    sub = prepare(host)
    core.submit(sub)
    assert core.dispatchable() == [(host.target, sub)]

    def fail(request):
        raise RuntimeError("middleware send failed")

    monkeypatch.setattr(endpoint._clients[host.target.target_id], "call_async", fail)
    endpoint._send(host.target, sub, "execute")
    assert core.query(sub)["state"] == "unknown"
    assert not core.resource_ready(host.target.hardware_resource_id)
    assert not model.calls
    assert "middleware send failed" in caplog.text and "execute" in caplog.text


def test_heartbeat_send_failure_reports_error_and_recovers(transport, monkeypatch, caplog):
    _, adapter, _ = transport
    call = adapter._client.call_async

    def fail(request):
        raise RuntimeError("heartbeat send failed")

    monkeypatch.setattr(adapter._client, "call_async", fail)
    adapter._heartbeat()
    assert adapter.registration_error == "REGISTRATION_SEND_FAILED"
    assert "heartbeat send failed" in caplog.text
    monkeypatch.setattr(adapter._client, "call_async", call)
    adapter._heartbeat()
    adapter._heartbeat()
    assert adapter.registration_error == "" and adapter.ready


@pytest.mark.parametrize("boot_id", ["", "not-a-uuid", "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"])
def test_invalid_boot_id_does_not_bind_existing_handle(boot_id):
    model = Executor()
    handle = ModelRuntimeHandle(executor=model)
    handle.load()
    adapter = None
    try:
        with pytest.raises(ValueError):
            SchedulerRuntimeAdapter(FakeNode({}), target(resource="cpu"), handle, boot_id=boot_id)
        result = handle.execute(ModelRequest(), ExecutionContext("after-invalid-boot"))
        assert result.outputs["answer"].tolist() == [1]
        adapter = SchedulerRuntimeAdapter(FakeNode({}), target(resource="cpu"), handle)
        assert adapter.controller.handle is handle
    finally:
        if adapter is not None:
            adapter.controller.shutdown()
            handle._close_local(ExecutionContext("cleanup"), _dispatch_key=adapter.controller._authority)
        elif handle._execution_policy is None:
            handle.close()


def test_reset_registration_clears_prior_failure_and_updates_lease(transport, monkeypatch):
    endpoint, adapter, model = transport
    host, core = adapter.controller, endpoint.core
    monkeypatch.setattr(model, "reset", lambda *_: None, raising=False)

    def call(verb, sub, timeout, **kwargs):
        if verb == "submit":
            core.submit(sub)
            assert core.dispatchable() == [(host.target, sub)]
            host.execute(sub, core.epoch)
            host.completion(sub).result(timeout=1)
            core.report(host.query(sub, core.epoch))
        return core.query(sub)

    monkeypatch.setattr(adapter, "_call", call)
    core.stale_ns = 4_000_000_000
    adapter._set_registration_error("REGISTRATION_TIMEOUT")
    host.handle.reset(ExecutionContext("reset", Deadline.after(2)))
    assert adapter.registration_error == ""
    assert adapter._stale_seconds == 4
    assert adapter.ready and host.generation == 2


def test_internal_service_failure_preserves_operation_and_stack_in_log(caplog):
    def fail(message):
        raise KeyError("internal ledger invariant")

    response = _serve(fail)(SimpleNamespace(payload=encode_control({"verb": "query"})), SimpleNamespace())
    assert decode_control(response.payload) == {"control_error": "INVALID_CONTROL"}
    assert "query" in caplog.text and "KeyError" in caplog.text and "internal ledger invariant" in caplog.text


@pytest.mark.parametrize("interrupt", ["deadline", "cancel"])
def test_close_lock_wait_obeys_callers_budget(transport, interrupt):
    _, adapter, _ = transport
    context = ExecutionContext("second-close", Deadline.after(0.05 if interrupt == "deadline" else 2))
    entered = threading.Event()

    def close():
        entered.set()
        adapter.close(context)

    adapter._close_lock.acquire()
    with ThreadPoolExecutor(1) as workers:
        future = workers.submit(close)
        try:
            assert entered.wait(1)
            if interrupt == "cancel":
                context.cancellation_token.cancel("stop waiting")
            expected = DeadlineExceeded if interrupt == "deadline" else CancellationRequested
            with pytest.raises(expected):
                future.result(timeout=0.5)
        finally:
            adapter._close_lock.release()


@pytest.mark.parametrize("error", ["deadline", "unavailable"])
def test_local_completion_wins_over_later_global_control_failure(transport, monkeypatch, error):
    endpoint, adapter, _ = transport
    host, core = adapter.controller, endpoint.core

    def call(verb, sub, timeout, **kwargs):
        if verb == "submit":
            core.submit(sub)
            assert core.dispatchable() == [(host.target, sub)]
            host.execute(sub, core.epoch)
            host.completion(sub).result(timeout=1)
            # Intentionally withhold the terminal report from Global.
            return {"state": "running"}
        if error == "deadline":
            raise DeadlineExceeded(phase="scheduler_wait")
        raise NotStartedControlError("GLOBAL_UNAVAILABLE")

    monkeypatch.setattr(adapter, "_call", call)
    original_send = adapter._client.call_async

    def send(request):
        payload = decode_control(request.payload)
        if payload["verb"] == "terminal_report":
            return call("terminal_report", None, 0)
        return original_send(request)

    monkeypatch.setattr(adapter._client, "call_async", send)
    with pytest.raises(ExecutionFailure) as failed:
        host.handle.execute(ModelRequest(), ExecutionContext("completed-before-control-error", Deadline.after(2)))
    assert failed.value.evidence.outcome_known and failed.value.evidence.state.value != "not_started"
    assert host.handle.local_ready
    sub = next(iter(core._records.values())).submission
    assert host.query(sub, core.epoch)["state"] == "completed"
    assert core.query(sub)["state"] == "dispatching"
    assert not host._records[sub.operation_id].acknowledged
    # Global still owns the reconciliation/ACK, even though business wait ended.
    core.report(host.query(sub, core.epoch))
    assert core.query(sub)["state"] == "completed"


@pytest.mark.parametrize("digit_limit", [4300, 0], ids=["encoding_limit", "wire_budget"])
def test_unsent_integer_encoding_failure_does_not_consume_record_capacity(transport, monkeypatch, digit_limit):
    endpoint, adapter, model = transport
    host = adapter.controller
    host.target = replace(host.target, max_pending_requests=1, max_records=1)
    monkeypatch.setattr(adapter._client, "call_async", lambda _: pytest.fail("unencodable request was sent"))
    original_limit = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(digit_limit)
        with pytest.raises(NotStartedControlError) as failure:
            adapter.execute(ModelRequest(), ExecutionContext("huge-priority", Deadline.after(1)), priority=10**40000)
        assert failure.value.code == ("INVALID_REQUEST" if digit_limit else "CONTROL_TOO_LARGE")
    finally:
        sys.set_int_max_str_digits(original_limit)
    assert not model.calls and not endpoint.core._records
    assert not host._records and host._input_bytes == host._result_bytes == 0
    prepare(host, priority=10**100)  # Software priority is not restricted to an int32/native range.


@pytest.mark.parametrize("boundary", ["report", "transport_envelope"])
def test_submission_budget_reserves_terminal_receipt_before_admission(transport, monkeypatch, boundary):
    import json

    from inference_service.scheduler.runtime_protocol import MAX_CONTROL_BYTES, ControlError

    endpoint, adapter, model = transport
    host = adapter.controller
    # Fixed clock widths make the boundary independent of machine uptime.
    now = 1_000_000_000_000_000
    monkeypatch.setattr(host, "now_ns", lambda: now)
    host.target = replace(host.target, max_pending_requests=1, max_records=1)
    seed = host.prepare(ModelRequest(), ExecutionContext("boundary", Deadline.after(1)), priority=0)
    host.cancel(seed, host.epoch)
    host.acknowledge(seed, host.epoch)
    host.release_result_waiter(seed)
    monkeypatch.setattr(host, "now_ns", lambda: seed.accept_until_ns + 1)
    host.collect()
    assert not host._records

    original_limit = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(0)
        base = {"verb": "submit", "epoch": host.epoch, "submission": seed.to_dict()}
        priority = 10 ** (MAX_CONTROL_BYTES - 18 - len(encode_control(base).encode()))
        worst = {
            "submission": seed.to_dict(),
            "state": "not_started",
            "error": "\U0010ffff" * 256,
            "started_ns": 2**63 - 1,
            "finished_ns": 2**63 - 1,
        }
        if boundary == "transport_envelope":
            # The bare report fits with eight bytes left; its transport does not.
            priority = 10 ** (MAX_CONTROL_BYTES - 8 - len(encode_control(worst).encode()))
        candidate = replace(seed, priority=priority)
        submit = {**base, "submission": candidate.to_dict()}
        completed = {
            "submission": candidate.to_dict(),
            "state": "completed",
            "error": "",
            "started_ns": now,
            "finished_ns": now + 1,
        }
        assert len(encode_control(submit).encode()) <= MAX_CONTROL_BYTES
        if boundary == "report":
            oversized = completed
        else:
            worst["submission"] = candidate.to_dict()
            assert len(encode_control(worst).encode()) <= MAX_CONTROL_BYTES
            oversized = {"verb": "terminal_report", "epoch": host.epoch, "report": worst}
        assert len(json.dumps(oversized, sort_keys=True, separators=(",", ":")).encode()) > MAX_CONTROL_BYTES
        with pytest.raises(ControlError, match="CONTROL_TOO_LARGE"):
            host.prepare(ModelRequest(), ExecutionContext("boundary", Deadline.after(1)), priority=priority)
        assert not host._records and host._input_bytes == host._result_bytes == 0
        assert not model.calls and not endpoint.core._records

        # A smaller descriptor still supports large software priorities and
        # the largest diagnostic envelope, including JSON Unicode escaping.
        sub = host.prepare(ModelRequest(), ExecutionContext("boundary", Deadline.after(1)), priority=10**28000)
        for verb in ("submit", "execute", "query", "cancel", "ack"):
            encode_control({"verb": verb, "epoch": host.epoch, "submission": sub.to_dict()})
        encode_control(
            {
                "verb": "terminal_report",
                "epoch": host.epoch,
                "report": {**worst, "submission": sub.to_dict()},
            }
        )
    finally:
        sys.set_int_max_str_digits(original_limit)


def test_public_cancel_reaches_executor_with_business_request_id(transport, monkeypatch):
    endpoint, adapter, model = transport
    host, core = adapter.controller, endpoint.core
    model.release.clear()
    cancelled = []

    def cancel(request_id, **kwargs):
        cancelled.append(request_id)
        model.release.set()

    monkeypatch.setattr(model, "cancel", cancel, raising=False)

    def call(verb, sub, timeout, **kwargs):
        if verb == "submit":
            core.submit(sub)
            core.dispatchable()
            host.execute(sub, core.epoch)
            return {"state": "running"}
        return {"state": host.query(sub, core.epoch)["state"]}

    monkeypatch.setattr(adapter, "_call", call)
    try:
        with ThreadPoolExecutor(1) as workers:
            future = workers.submit(
                host.handle.execute, ModelRequest(), ExecutionContext("business", Deadline.after(2))
            )
            assert model.entered.wait(1)
            host.handle.cancel("business")
            with pytest.raises(ExecutionFailure):
                future.result(timeout=2)
        assert cancelled == ["business"]
    finally:
        model.release.set()


def test_public_cancel_stops_prepared_resource_not_ready_retries(monkeypatch):
    bus = {}
    first, peer = target("first", resource="cpu"), target("peer", resource="cpu")
    model = Executor()
    handle = ModelRuntimeHandle(executor=model)
    handle.load()
    endpoint = GlobalRuntimeEndpoint(FakeNode(bus), [first, peer])
    adapter = SchedulerRuntimeAdapter(FakeNode(bus), first, handle)
    connect(endpoint.core, adapter.controller)  # Required peer has not registered.
    rejected = threading.Event()
    submits = []
    actual = adapter._client.call_async

    def counted(request):
        message = decode_control(request.payload)
        reply = actual(request)
        if message["verb"] == "submit":
            assert decode_control(reply.result().payload)["control_error"] == "RESOURCE_NOT_READY"
            submits.append(message)
            if len(submits) == 2:
                rejected.set()
        return reply

    monkeypatch.setattr(adapter._client, "call_async", counted)
    context = ExecutionContext("prepared-business", Deadline.after(2))
    try:
        with ThreadPoolExecutor(1) as workers:
            pending = workers.submit(handle.execute, ModelRequest(), context)
            try:
                assert rejected.wait(1)
                handle.cancel(context.request_id)
                with pytest.raises(ExecutionFailure) as failure:
                    pending.result(timeout=0.3)
                assert failure.value.evidence.state.value == "not_started"
                context.check("reuse-after-local-cancel")
                assert len(submits) <= 3 and not model.calls
            finally:
                context.cancellation_token.cancel("test cleanup")
    finally:
        adapter.controller.shutdown()
        handle._close_local(ExecutionContext("cleanup"), _dispatch_key=adapter.controller._authority)


def test_execute_reply_cannot_release_another_operations_resource(monkeypatch):
    bus = {}
    targets = [target("a", resource="cpu"), target("b", resource="mps"), target("c", resource="mps")]
    endpoint = GlobalRuntimeEndpoint(FakeNode(bus), targets)
    adapters, models = [], []
    try:
        for t in targets:
            model = Executor()
            model.release.clear()
            handle = ModelRuntimeHandle(executor=model)
            handle.load()
            adapter = SchedulerRuntimeAdapter(FakeNode(bus), t, handle)
            models.append(model)
            adapters.append(adapter)
            connect(endpoint.core, adapter.controller)
        submissions = [prepare(adapter.controller) for adapter in adapters]
        for sub in submissions:
            endpoint.core.submit(sub)

        def mismatched_reply(_):
            future = Future()
            future.set_result(
                SimpleNamespace(
                    payload=encode_control(
                        {
                            "submission": submissions[1].to_dict(),
                            "state": "completed",
                        }
                    )
                )
            )
            return future

        monkeypatch.setattr(endpoint._clients[targets[0].target_id], "call_async", mismatched_reply)
        endpoint._tick()
        assert models[1].entered.wait(1)
        endpoint._tick()
        assert adapters[1].controller.query(submissions[1], endpoint.core.epoch)["state"] == "running"
        assert endpoint.core.query(submissions[1])["state"] == "running"
        assert endpoint.core.query(submissions[2])["state"] == "queued"
        assert endpoint.core.query(submissions[0])["state"] == "unknown"
        assert not models[2].calls
    finally:
        for model in models:
            model.release.set()
        for adapter in adapters:
            adapter.controller.shutdown()
            adapter.controller.handle._close_local(
                ExecutionContext("cleanup"),
                _dispatch_key=adapter.controller._authority,
            )


@pytest.mark.parametrize("stale_ns", [200_000_001, 250_000_000])
def test_healthy_heartbeat_stays_ready_between_reply_consumption_ticks(monkeypatch, stale_ns):
    from inference_service.scheduler import runtime_ros
    from inference_service.scheduler.runtime_protocol import HOST_HEARTBEAT_INTERVAL_NS

    now = [1_000_000_000]
    monkeypatch.setattr(runtime_ros, "time", SimpleNamespace(monotonic=lambda: now[0] / 1e9))
    bus = {}
    t = target(resource="cpu")
    handle = ModelRuntimeHandle(executor=Executor())
    handle.load()
    endpoint = GlobalRuntimeEndpoint(FakeNode(bus), [t], stale_ns=stale_ns)
    adapter = SchedulerRuntimeAdapter(FakeNode(bus), t, handle)
    monkeypatch.setattr(endpoint.core, "now_ns", lambda: now[0])
    monkeypatch.setattr(adapter.controller, "now_ns", lambda: now[0])
    try:
        for tick in range(8):
            adapter._heartbeat()
            # Two replies establish the epoch and Global's resource readiness.
            for offset in (50_000_000, 49_999_999, 1):
                now[0] += offset
                if tick >= 2:
                    assert endpoint.core.resource_ready("cpu")
                    assert adapter.ready and handle.ready
        # Heartbeat completion does not grant a fresh receive-time lease.
        now[0] += stale_ns + HOST_HEARTBEAT_INTERVAL_NS
        assert not adapter.ready and not handle.ready
    finally:
        adapter.controller.shutdown()
        handle._close_local(ExecutionContext("cleanup"), _dispatch_key=adapter.controller._authority)
        adapter.close()


@pytest.mark.parametrize("registration", ["heartbeat", "reset"])
def test_delayed_registration_reply_does_not_extend_host_readiness(transport, monkeypatch, registration):
    import time

    from inference_service.scheduler import runtime_ros

    endpoint, adapter, model = transport
    now = [time.monotonic_ns()]
    monkeypatch.setattr(endpoint.core, "now_ns", lambda: now[0])
    monkeypatch.setattr(adapter.controller, "now_ns", lambda: now[0])
    monkeypatch.setattr(
        runtime_ros,
        "time",
        SimpleNamespace(
            monotonic=lambda: now[0] / 1e9,
            monotonic_ns=lambda: now[0],
        ),
    )
    if registration == "heartbeat":
        adapter._heartbeat()  # Global accepts now; Host consumes the reply later.
        now[0] += 1_900_000_000
        monkeypatch.setattr(adapter._client, "service_is_ready", lambda: False)
        adapter._heartbeat()
    else:
        monkeypatch.setattr(adapter, "execute", lambda *_args, **_kwargs: None)
        call = adapter._call_control

        def delayed(*args, **kwargs):
            reply = call(*args, **kwargs)
            now[0] += 1_900_000_000
            return reply

        monkeypatch.setattr(adapter, "_call_control", delayed)
        adapter.reset(ExecutionContext("reset-registration"))
    assert adapter.ready and endpoint.core.resource_ready("cpu")
    now[0] += 200_000_000
    assert not endpoint.core.resource_ready("cpu")
    assert not adapter.ready and not adapter.controller.handle.ready
    assert not adapter.stream_input_available
    assert not model.calls


def test_public_managed_failure_is_counted_once(transport, monkeypatch):
    from inference_service.backends.errors import BackendInferenceError

    endpoint, adapter, model = transport
    host, core = adapter.controller, endpoint.core
    model.error = BackendInferenceError("known failure", operation_started=True, outcome_known=True)

    def call(verb, sub, timeout, **kwargs):
        if verb == "submit":
            core.submit(sub)
            core.dispatchable()
            host.execute(sub, core.epoch)
            host.completion(sub).result(timeout=1)
            core.report(host.query(sub, core.epoch))
        return core.query(sub)

    monkeypatch.setattr(adapter, "_call", call)
    with pytest.raises(ExecutionFailure):
        host.handle.execute(ModelRequest(), ExecutionContext("known-failure", Deadline.after(2)))
    assert host.handle.health.failure_count == 1


@pytest.mark.parametrize("failure", ["deadline", "cancel", "unavailable"])
def test_completed_reset_preserves_evidence_when_registration_fails(transport, monkeypatch, failure):
    endpoint, adapter, model = transport
    host, core = adapter.controller, endpoint.core
    resets = []
    monkeypatch.setattr(model, "reset", lambda *_: resets.append(None), raising=False)

    def call(verb, sub, timeout, **kwargs):
        if verb == "submit":
            core.submit(sub)
            assert core.dispatchable() == [(host.target, sub)]
            host.execute(sub, core.epoch)
            host.completion(sub).result(timeout=1)
            core.report(host.query(sub, core.epoch))
        return core.query(sub)

    def fail_registration(*args, **kwargs):
        if failure == "deadline":
            raise DeadlineExceeded(phase="reset_registration")
        if failure == "cancel":
            kwargs["context"].cancellation_token.cancel("cancel registration")
            kwargs["context"].check("reset_registration")
        raise NotStartedControlError("GLOBAL_UNAVAILABLE")

    monkeypatch.setattr(adapter, "_call", call)
    monkeypatch.setattr(adapter, "_call_control", fail_registration)
    with pytest.raises(ExecutionFailure) as error:
        host.handle.reset(ExecutionContext("reset", Deadline.after(1)))
    assert error.value.evidence.outcome_known
    assert error.value.evidence.state.value != "not_started"
    assert resets == [None] and host.generation == 2
    assert host.handle.local_ready


@pytest.mark.parametrize("interruption", ["deadline", "cancel"])
def test_control_wait_and_cancel_cleanup_respect_business_budget(transport, monkeypatch, interruption):
    _, adapter, model = transport
    entered = threading.Event()
    futures, cancel_budgets = [], []
    context = ExecutionContext("bounded", Deadline.after(0.03 if interruption == "deadline" else 2))
    actual_call = adapter._call

    def stalled(request):
        future = Future()
        futures.append(future)
        entered.set()
        return future

    def call(verb, sub, timeout, **kwargs):
        if verb == "cancel":
            cancel_budgets.append(timeout)
        return actual_call(verb, sub, timeout, **kwargs)

    monkeypatch.setattr(adapter._client, "call_async", stalled)
    monkeypatch.setattr(adapter, "_call", call)
    with ThreadPoolExecutor(1) as workers:
        pending = workers.submit(adapter.execute, ModelRequest(), context)
        assert entered.wait(1)
        if interruption == "cancel":
            context.cancellation_token.cancel("stop waiting")
        with pytest.raises(NotStartedControlError):
            pending.result(timeout=0.15)
    assert cancel_budgets == [0] and not model.calls


@pytest.mark.parametrize("failure_at", ["client", "service", "guard_condition", "timer"])
def test_global_endpoint_construction_rolls_back_node_entities(monkeypatch, failure_at):
    class TrackingNode(FakeNode):
        def __init__(self):
            super().__init__({})
            self.live = []

    node = TrackingNode()
    for kind in ("client", "service", "guard_condition", "timer"):
        actual = getattr(node, "create_" + kind)
        actual_destroy = getattr(node, "destroy_" + kind)

        def create(*args, kind=kind, actual=actual, **kwargs):
            if kind == failure_at and (kind != "client" or node.live):
                raise RuntimeError("ROS construction failed")
            entity = actual(*args, **kwargs)
            node.live.append(entity)
            return entity

        def destroy(entity, actual_destroy=actual_destroy):
            node.live.remove(entity)
            actual_destroy(entity)

        monkeypatch.setattr(node, "create_" + kind, create)
        monkeypatch.setattr(node, "destroy_" + kind, destroy)
    with pytest.raises(RuntimeError, match="ROS construction failed"):
        GlobalRuntimeEndpoint(node, [target("first", resource="cpu"), target("second", resource="cpu")])
    assert not node.live and not node.bus


@pytest.mark.parametrize("assembly_capacity,session_capacity", [(1, 2), (2, 1)])
def test_adapter_rejects_capacity_above_handle_or_native_instance(assembly_capacity, session_capacity):
    from inference_service.scheduler.runtime_protocol import ControlError

    adapter = SchedulerRuntimeAdapter.__new__(SchedulerRuntimeAdapter)
    adapter.target = target(resource_mode="ascend_concurrent", max_in_flight=2)
    handle = SimpleNamespace(
        local_ready=True,
        assembly=SimpleNamespace(
            max_active_executions=assembly_capacity,
            session=SimpleNamespace(capabilities=SimpleNamespace(max_in_flight_per_instance=session_capacity)),
        ),
    )
    with pytest.raises(ControlError, match="INSTANCE_CAPACITY_MISMATCH"):
        adapter._validate_backend(handle)
