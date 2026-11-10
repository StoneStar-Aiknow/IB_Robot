"""Dynamic membership transport retains execution evidence during host exit."""

import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from inference_service.backends import BackendAdmissionEvidence, BackendCapabilities, BackendPriorityMapping
from inference_service.scheduler import runtime_ros
from inference_service.scheduler.runtime_binding import bind_runtime
from inference_service.scheduler.runtime_controller import RuntimeExecutionController
from inference_service.scheduler.runtime_protocol import (
    ControlError,
    NotStartedControlError,
    decode_control,
    encode_control,
)
from inference_service.scheduler.runtime_recovery import ResourceRecovery
from inference_service.scheduler.runtime_ros import (
    GlobalRuntimeEndpoint,
    SchedulerRuntimeAdapter,
    bind_registration_group,
)
from inference_service.unified_runtime import (
    Deadline,
    ExecutionContext,
    ExecutionFailure,
    ModelRequest,
    ModelRuntimeHandle,
)

from .runtime_scheduler_helpers import LocalNode, StreamFixture
from .runtime_scheduler_helpers import transport_target as target

pytestmark = pytest.mark.usefixtures("simulated_runtime_construction")


@contextmanager
def _runtime_pump(*callbacks):
    """Run this test's control callbacks; surface failures before teardown."""
    stopped, errors = threading.Event(), []

    def run():
        try:
            while not stopped.wait(0.002):
                for callback in callbacks:
                    callback()
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join(2)
        assert not worker.is_alive(), "runtime control pump did not stop"
        if errors:
            raise errors[0]


def _dispose_fake_transport(adapter):
    """End an in-memory fixture; no real SDK or ROS executor is involved."""
    if adapter._closing:
        return
    adapter.controller.shutdown()
    adapter.controller.handle._close_local(
        ExecutionContext("fixture-cleanup"),
        _dispatch_key=adapter.controller._authority,
    )
    adapter._unregister_clean_host()
    adapter._closing = True
    adapter.destroy_transport()


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setattr(runtime_ros, "_service_type", lambda: SimpleNamespace(Request=SimpleNamespace))
    services = {}
    node = LocalNode(services)
    endpoint = GlobalRuntimeEndpoint(node, [], resources={"cpu": {"physical_device": "local/cpu", "mode": "serial"}})
    return services, node, endpoint


def host(services, selected):
    handle = ModelRuntimeHandle(executor=SimpleNamespace(execute=lambda *_: {"answer": 1}))
    handle.load()
    adapter = SchedulerRuntimeAdapter(LocalNode(services), selected, handle)
    # Complete the registration handshake using the same heartbeat callback
    # and serialized control envelopes as the ROS transport.
    adapter._heartbeat()
    adapter._heartbeat()
    adapter._heartbeat()
    return adapter, handle


@pytest.mark.parametrize("business_error", [False, True])
@pytest.mark.parametrize("error", [RuntimeError("handler failed"), NotStartedControlError("NOT_SENT")])
def test_local_transport_preserves_service_errors_and_transport_exceptions(transport, business_error, error):
    services, node, _ = transport

    def fail(_message):
        raise error

    if business_error:
        node.create_service(None, "/failure", runtime_ros._serve(fail))
    else:
        services["/failure"] = fail
    client = node.create_client(None, "/failure")
    future = client.call_async(runtime_ros._request({"verb": "test"}))
    if business_error:
        reply = decode_control(future.result().payload)
        assert reply == (
            {"control_error": "NOT_SENT", "not_started": True}
            if isinstance(error, NotStartedControlError)
            else {"control_error": "INVALID_CONTROL"}
        )
    else:
        assert future.exception() is error


@pytest.mark.parametrize("drain", [False, True])
def test_unreleased_close_stays_pending_after_outcome_reconciliation(transport, drain):
    from inference_service.backends.errors import BackendLifecycleError

    services, _, endpoint = transport
    calls = []

    def close():
        calls.append("close")
        if len(calls) == 1:
            raise BackendLifecycleError("owner still holds resources", close_pending=True)

    handle = ModelRuntimeHandle(
        SimpleNamespace(execute=lambda *_: {}, close=close, drain_uncertain_operations=lambda: None)
    )
    handle.load()
    adapter = SchedulerRuntimeAdapter(
        LocalNode(services), replace(target("unreleased_close"), retry_window_ms=3000), handle
    )
    for _ in range(3):
        adapter._heartbeat()
    try:
        controller = adapter.controller
        submission = controller.prepare(ModelRequest(), ExecutionContext("close", Deadline.after(1)), work="close")
        endpoint.core.submit(submission)
        assert endpoint.core.dispatchable() == [(adapter.target, submission)]
        controller.execute(submission, endpoint.core.epoch)
        controller.completion(submission).result(timeout=1)
        report = controller.query(submission, endpoint.core.epoch)
        assert report["state"] == "unknown" and report["error"] == "close_drain_failed"
        endpoint.core.report(report)
        if drain:
            operation = str(uuid4())
            expiry = time.monotonic_ns() + 1_000_000_000
            controller.recover(endpoint.core.epoch, operation, "drain", expiry)
            controller._control_future.result(timeout=1)
            assert controller.recover(endpoint.core.epoch, operation, "drain", expiry)["state"] == "completed"
            report = controller.query(submission, endpoint.core.epoch)
            assert report["state"] == "completed" and report["error"] == "RECOVERED"
            endpoint.core.report(report)
        assert not handle.closed and not report["close_complete"]
        endpoint.core.acknowledge_host(submission)
        assert not endpoint.core.host_retired(adapter.target.target_id, controller.boot_id)
        assert "cpu" in endpoint.core._fenced
        assert endpoint.core.shutdown_pending() == [adapter.target.target_id]
    finally:
        _dispose_fake_transport(adapter)


def test_global_recovery_survives_more_submit_failures_than_host_capacity(transport, monkeypatch):
    from inference_service.scheduler.runtime_controller import MAX_HOST_RECOVERY_RECEIPTS

    services, _, endpoint = transport
    adapter, handle = host(services, replace(target(), retry_window_ms=3000))
    original = adapter.controller._recovery.submit
    queued = []
    failures = MAX_HOST_RECOVERY_RECEIPTS + 2

    def submit(fn, *args):
        if len(queued) < failures:
            queued.append((fn, args))
            raise RuntimeError("cannot start new thread")
        return original(fn, *args)

    monkeypatch.setattr(adapter.controller._recovery, "submit", submit)
    operation = str(uuid4())
    try:
        endpoint.recovery.request(
            {
                "resource": "cpu",
                "target_id": adapter.target.target_id,
                "boot_id": adapter.controller.boot_id,
                "operation_id": operation,
                "command": "close",
                "accept_until_ns": time.monotonic_ns() + 2_000_000_000,
            }
        )
        until = time.monotonic() + 2
        while endpoint.recovery.records[operation]["state"] == "running" and time.monotonic() < until:
            endpoint._tick()
            time.sleep(0.002)
        assert len(queued) == failures
        assert endpoint.recovery.records[operation]["state"] == "completed"
        assert handle.closed and not endpoint.core._fenced
        for fn, args in queued:
            assert args[0].cancelled()
            fn(*args)  # Late dequeues cannot repeat the completed Close.
        assert len(adapter.controller._recovery_records) < MAX_HOST_RECOVERY_RECEIPTS
    finally:
        _dispose_fake_transport(adapter)


def test_failed_stream_recovery_receipts_cannot_exhaust_public_close(transport, monkeypatch):
    from inference_service.scheduler.runtime_controller import MAX_HOST_RECOVERY_RECEIPTS
    from inference_service.scheduler.runtime_recovery import MAX_RESOURCE_RECOVERY_RECEIPTS

    services, _, endpoint = transport
    clock = [time.monotonic_ns()]
    selected = replace(target("failure_capacity"), retry_window_ms=1000)
    calls = []

    def reset_stream(*_, **_kwargs):
        calls.append("reset")
        raise RuntimeError("stream reset completion unknown")

    def drain(*_, **_kwargs):
        calls.append("drain")

    handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}))
    handle.load()
    binding = bind_runtime(LocalNode(services), handle, selected)
    adapter, host = binding.adapter, binding.controller
    # Exercise Global/Host wire handlers and the public Close path. A fake
    # clock advances replay windows without waiting minutes in this test.
    endpoint.core.now_ns = host.now_ns = lambda: clock[0]
    monkeypatch.setattr(host.handle, "recover_stream", reset_stream, raising=False)
    monkeypatch.setattr(host.handle, "drain_managed", drain)
    for _ in range(3):
        adapter._heartbeat()

    def pump(operation):
        for _ in range(80):
            endpoint._tick()
            future = host._control_future
            if future is not None:
                future.result(timeout=1)
            if endpoint.recovery.records[operation]["state"] != "running":
                return
        pytest.fail("recovery did not converge")

    try:
        # More failed mutations than the Host's full capacity, and enough to
        # fill Global's ordinary quota. UNKNOWN must remain quarantined.
        for index in range(MAX_RESOURCE_RECOVERY_RECEIPTS - 1):
            operation = str(uuid4())
            endpoint.recovery.request(
                {
                    "resource": "cpu",
                    "target_id": selected.target_id,
                    "boot_id": host.boot_id,
                    "operation_id": operation,
                    "command": "reset_stream",
                    "stream_id": f"stream-{index}",
                    "accept_until_ns": clock[0] + 1_000_000_000,
                }
            )
            pump(operation)
            assert endpoint.recovery.records[operation]["state"] == "unknown"
            assert not endpoint.core.resource_ready("cpu")
            clock[0] += 1_000_000_001
            endpoint.core.register(host.status())
            host.collect()
            assert len(host._recovery_records) < MAX_HOST_RECOVERY_RECEIPTS
        assert calls.count("reset") == MAX_RESOURCE_RECOVERY_RECEIPTS - 1
        # Return to a real monotonic clock for the public adapter's budget.
        endpoint.core.now_ns = host.now_ns = time.monotonic_ns
        call_control = adapter._call_control

        def progress(*args, **kwargs):
            result = call_control(*args, **kwargs)
            endpoint._tick()
            return result

        monkeypatch.setattr(adapter, "_call_control", progress)
        handle.close(deadline=Deadline.after(3))
        assert handle.closed and calls.count("drain") >= MAX_RESOURCE_RECOVERY_RECEIPTS - 1
        assert not endpoint.core.shutdown_pending()
    finally:
        _dispose_fake_transport(adapter)


@pytest.mark.parametrize("boundary", ["fresh", "stale", "restart"])
def test_peer_close_waits_for_unknown_recovery_device_work(transport, boundary):
    services, node, endpoint = transport
    device_done = threading.Event()
    calls = []

    def reset():
        calls.append("reset")
        raise RuntimeError("RESET_COMPLETION_UNKNOWN")

    def drain():
        calls.append("drain")
        if not device_done.is_set():
            raise RuntimeError("device still busy")

    handle = ModelRuntimeHandle(
        SimpleNamespace(execute=lambda *_: {}, reset=reset, drain_uncertain_operations=drain), resettable=True
    )
    handle.load()
    first = SchedulerRuntimeAdapter(node, replace(target("first"), retry_window_ms=3000), handle)
    peer_handle = ModelRuntimeHandle(
        SimpleNamespace(
            execute=lambda *_: {},
            close=lambda: (
                calls.append("close") if device_done.is_set() else pytest.fail("Close overlapped device work")
            ),
        )
    )
    peer_handle.load()
    second = SchedulerRuntimeAdapter(node, replace(target("second"), retry_window_ms=3000), peer_handle)
    for _ in range(3):
        first._heartbeat()
        second._heartbeat()

    def run(adapter, command):
        message = {
            "resource": "cpu",
            "target_id": adapter.target.target_id,
            "boot_id": adapter.controller.boot_id,
            "operation_id": str(uuid4()),
            "command": command,
            "accept_until_ns": endpoint.core.now_ns() + 2_000_000_000,
        }
        endpoint.recovery.request(message)
        return message

    def settle(message):
        until = time.monotonic() + 2
        record = endpoint.recovery.records[message["operation_id"]]
        while record["state"] == "running" and time.monotonic() < until:
            endpoint._tick()
            time.sleep(0.002)
        return record

    try:
        reset_request = run(first, "reset")
        assert settle(reset_request)["state"] == "unknown"
        assert not first.controller.status()["clean"]
        if boundary != "fresh":
            # No heartbeat since registration: Global still has positive old
            # evidence, even though Reset now owns uncertain device work.
            assert endpoint.core._hosts[first.target.target_id][0]["clean"]
            clock = time.monotonic_ns() + endpoint.core.stale_ns + 1
            endpoint.core.now_ns = first.controller.now_ns = second.controller.now_ns = lambda: clock
            endpoint.core.register(second.controller.status())
            if boundary == "restart":
                replacement = {**first.controller.status(), "boot_id": str(uuid4()), "clean": True, "ready": True}
                with pytest.raises(ControlError, match="HOST_RESTART_REQUIRES_COORDINATED_RESTART"):
                    endpoint.core.register(replacement)
                assert endpoint.core.host_boot(first.target.target_id) == first.controller.boot_id
        with pytest.raises(ControlError, match="RECOVERY_EVIDENCE_PENDING"):
            first.controller._recover("activate", 2)
        close_request = run(second, "close")
        assert settle(close_request)["state"] == "unknown"
        assert "drain" in calls and "close" not in calls
        assert "cpu" in endpoint.core._fenced
        device_done.set()
        if boundary != "fresh":
            clock += 1
        endpoint.recovery.request({**close_request, "resume_until_ns": endpoint.core.now_ns() + 2_000_000_000})
        assert settle(close_request)["state"] == "completed"
        assert calls.count("reset") == calls.count("close") == 1
        assert calls.count("drain") >= 2 and not endpoint.core._fenced
        assert endpoint.recovery.records[reset_request["operation_id"]]["state"] == "superseded"
    finally:
        device_done.set()
        _dispose_fake_transport(first)
        _dispose_fake_transport(second)


def test_global_reset_retries_control_lock_timeout_without_repeating_reset(transport):
    services, node, endpoint = transport
    resets = []
    handle = ModelRuntimeHandle(
        SimpleNamespace(execute=lambda *_: {}, reset=lambda: resets.append("reset")), resettable=True
    )
    handle.load()
    adapter = SchedulerRuntimeAdapter(node, replace(target(), request_timeout_ms=20, retry_window_ms=3000), handle)
    for _ in range(3):
        adapter._heartbeat()
    operation = str(uuid4())
    handle._control_lock.acquire()
    try:
        endpoint.recovery.request(
            {
                "resource": "cpu",
                "target_id": adapter.target.target_id,
                "boot_id": adapter.controller.boot_id,
                "operation_id": operation,
                "command": "reset",
                "accept_until_ns": time.monotonic_ns() + 2_000_000_000,
            }
        )
        until = time.monotonic() + 2
        failed = None
        while failed is None and time.monotonic() < until:
            endpoint._tick()
            for receipt in tuple(adapter.controller._recovery_records.values()):
                if receipt.command == "reset":
                    failed = receipt.future.result(timeout=1)
            time.sleep(0.002)
        assert isinstance(failed, ExecutionFailure) and failed.evidence.outcome_known
        assert not resets
        identity = endpoint.recovery.records[operation]["commands"][adapter.target.target_id]["id"]
    finally:
        handle._control_lock.release()
    try:
        record = endpoint.recovery.records[operation]
        until = time.monotonic() + 2
        while record["state"] == "running" and time.monotonic() < until:
            endpoint._tick()
            time.sleep(0.002)
        assert record["state"] == "completed" and resets == ["reset"]
        assert adapter.controller._recovery_records[identity].future.result() is None
        assert not endpoint.core._fenced
    finally:
        _dispose_fake_transport(adapter)


@pytest.mark.parametrize("command", ["reset", "close"])
def test_recovery_discovery_after_unsent_window_can_resume(transport, command):
    services, node, endpoint = transport
    calls = []
    handle = ModelRuntimeHandle(
        SimpleNamespace(
            execute=lambda *_: {}, reset=lambda: calls.append("reset"), close=lambda: calls.append("close")
        ),
        resettable=True,
    )
    handle.load()
    binding = bind_runtime(node, handle, replace(target(), retry_window_ms=300))
    for _ in range(3):
        binding.adapter._heartbeat()
    service = services.pop(binding.target.endpoint)

    def pump_once():
        endpoint._tick()
        binding.adapter._heartbeat()

    try:
        with _runtime_pump(pump_once):
            try:
                with pytest.raises(ControlError, match="RECOVERY_EXPIRED|OPERATION_EXPIRED"):
                    binding.adapter.recover(ExecutionContext("missing-service", Deadline.after(1)), command=command)
                assert not calls
                services[binding.target.endpoint] = service
                binding.adapter.recover(ExecutionContext("discovered-service", Deadline.after(2)), command=command)
                assert calls == [command]
                assert endpoint.core.resource_ready("cpu")
            finally:
                services[binding.target.endpoint] = service
    finally:
        _dispose_fake_transport(binding.adapter)


@pytest.mark.parametrize("first_attempt", ["unavailable", "expired", "ambiguous"])
def test_unsent_public_recovery_drops_only_never_attempted_identity(transport, monkeypatch, first_attempt):
    from inference_service.unified_runtime.errors import DeadlineExceeded

    services, node, endpoint = transport
    calls = []
    handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}, close=lambda: calls.append("close")))
    handle.load()
    binding = bind_runtime(node, handle, replace(target(), retry_window_ms=100))
    for _ in range(3):
        binding.adapter._heartbeat()
    original = binding.adapter._call_control
    attempted = []

    def uncertain(payload, *args, **kwargs):
        attempted.append(payload["operation_id"])
        raise runtime_ros.TimeoutError("send outcome unknown")

    try:
        if first_attempt == "expired":
            with pytest.raises(DeadlineExceeded):
                binding.adapter.recover(ExecutionContext("expired", Deadline.after(0)), command="close")
        elif first_attempt == "ambiguous":
            monkeypatch.setattr(binding.adapter, "_call_control", uncertain)
            with pytest.raises((ControlError, DeadlineExceeded)):
                binding.adapter.recover(ExecutionContext("uncertain", Deadline.after(0.02)), command="close")
            monkeypatch.setattr(binding.adapter, "_call_control", original)
        global_service = services.pop(endpoint.endpoint)
        try:
            with pytest.raises(ControlError, match="GLOBAL_UNAVAILABLE"):
                binding.adapter.recover(ExecutionContext("unavailable", Deadline.after(1)), command="close")
        finally:
            services[endpoint.endpoint] = global_service
        if first_attempt == "ambiguous":
            assert binding.adapter._recovery_requests["close"]["operation_id"] == attempted[0]
        else:
            assert not binding.adapter.has_recovery("close")
        time.sleep(0.11)

        def pump_once():
            endpoint._tick()
            binding.adapter._heartbeat()

        with _runtime_pump(pump_once):
            handle.close(deadline=Deadline.after(2))
            assert handle.closed and calls == ["close"]
    finally:
        _dispose_fake_transport(binding.adapter)


@pytest.mark.parametrize("phase", ["drain", "close", "activate"])
def test_close_recovers_after_host_command_never_arrived(transport, monkeypatch, phase):
    services, node, endpoint = transport
    calls = []
    handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}, close=lambda: calls.append("close")))
    handle.load()
    binding = bind_runtime(node, handle, replace(target(), retry_window_ms=150))
    # A peer is needed for the activation phase after the first owner closes.
    peer, _ = host(services, replace(target("peer"), retry_window_ms=150))
    for _ in range(3):
        binding.adapter._heartbeat()
        peer._heartbeat()
    endpoint._tick()
    dropped, sent = [True], []
    for client in tuple(endpoint._clients.values()):
        send = client.call_async

        def call_async(request, send=send):
            message = decode_control(request.payload)
            if message.get("verb") == "recover" and message.get("command") == phase:
                sent.append(message["operation_id"])
                if dropped[0]:
                    return Future()  # Middleware accepted the send, but Host never saw it.
            return send(request)

        monkeypatch.setattr(client, "call_async", call_async)

    def pump_once():
        endpoint._tick()
        binding.adapter._heartbeat()
        peer._heartbeat()

    try:
        with _runtime_pump(pump_once):
            with pytest.raises(ControlError, match="RECOVERY_EXPIRED|OPERATION_EXPIRED"):
                binding.adapter.recover(ExecutionContext("lost-command", Deadline.after(1)), command="close")
            assert sent and not endpoint.core.resource_ready("cpu")
            dropped[0] = False
            handle.close(deadline=Deadline.after(2))
            assert handle.closed and calls == ["close"]
            assert len(set(sent)) >= 2
            assert endpoint.core.resource_ready("cpu")
            assert peer.execute(ModelRequest(), ExecutionContext("survivor", Deadline.after(1))).outputs == {
                "answer": 1
            }
    finally:
        _dispose_fake_transport(binding.adapter)
        _dispose_fake_transport(peer)


@pytest.mark.parametrize("elapsed_ns,expired", [(1_500_000_000, False), (3_000_000_001, True)])
def test_recovery_transport_uses_configured_control_timeout(transport, monkeypatch, elapsed_ns, expired):
    from inference_service.scheduler import runtime_recovery

    services, _, endpoint = transport
    adapter, _ = host(services, replace(target(), retry_window_ms=10000))
    endpoint._tick()
    endpoint.control_rpc_timeout_ns = 3_000_000_000
    clock = [endpoint.core.now_ns()]
    endpoint.core.now_ns = lambda: clock[0]
    monkeypatch.setattr(runtime_recovery, "time", SimpleNamespace(monotonic_ns=lambda: clock[0]))
    client = endpoint._clients[adapter.target.target_id]
    future = Future()
    monkeypatch.setattr(client, "call_async", lambda _: future)
    operation_id = str(uuid4())
    try:
        endpoint.recovery.request(
            {
                "resource": "cpu",
                "operation_id": operation_id,
                "command": "drain",
                "accept_until_ns": clock[0] + 9_000_000_000,
            }
        )
        endpoint.recovery.tick()
        clock[0] += elapsed_ns
        endpoint.recovery.tick()
        assert (future in client.removed) is expired
        assert endpoint.recovery.records[operation_id]["state"] == "running"
        # The transaction's acceptance window still bounds the larger RPC budget.
        clock[0] += 10_000_000_000
        endpoint.recovery.tick()
        assert endpoint.recovery.records[operation_id]["state"] == "unknown"
        assert future in client.removed
    finally:
        _dispose_fake_transport(adapter)


def test_pending_terminal_reply_preserves_independent_delivery_and_retry_throttle(transport, monkeypatch):
    services, _, endpoint = transport
    adapter, _ = host(services, target())
    controller = adapter.controller
    clock = [controller.now_ns()]
    controller.now_ns = lambda: clock[0]
    wake = threading.Event()
    monkeypatch.setattr(adapter._terminal_guard, "trigger", wake.set)
    sent = []

    def send(request):
        message = decode_control(request.payload)
        future = Future()
        sent.append((message, future))
        return future

    monkeypatch.setattr(adapter._client, "call_async", send)
    try:
        orphan = controller.prepare(ModelRequest(), ExecutionContext("orphan", Deadline.after(1)))
        controller.cancel(orphan, controller.epoch)
        adapter._terminal_tick()
        assert len(sent) == 1
        admitted = controller.prepare(ModelRequest(), ExecutionContext("new", Deadline.after(1)))
        endpoint.core.submit(admitted)
        endpoint.core.dispatchable()
        controller.execute(admitted, controller.epoch)
        controller.completion(admitted).result(timeout=1)
        adapter._terminal_tick()  # New evidence sends while the orphan still awaits a reply.
        assert len(sent) == 2
        assert sent[1][0]["report"]["submission"]["operation_id"] == admitted.operation_id
        wake.clear()
        reply = endpoint.core.terminal_report(sent[0][0]["report"])
        assert reply["state"] == "pending"
        sent[0][1].set_result(SimpleNamespace(payload=encode_control(reply)))
        assert wake.is_set()
        adapter._terminal_tick()  # Driven only by the callback's guard.
        assert len(sent) == 2
        reply = endpoint.core.terminal_report(sent[1][0]["report"])
        sent[1][1].set_result(SimpleNamespace(payload=encode_control(reply)))
        assert controller.terminal_receipt(admitted).done()
        for _ in range(3):
            adapter._terminal_tick()
        assert len(sent) == 2  # The orphan is throttled, even after an ACK wake.
        clock[0] += runtime_ros.HOST_HEARTBEAT_INTERVAL_NS
        adapter._terminal_tick()
        assert len(sent) == 3
        assert sent[2][0]["report"]["submission"]["operation_id"] == orphan.operation_id
    finally:
        monkeypatch.undo()
        _dispose_fake_transport(adapter)


@pytest.mark.parametrize("interruption", ["cancel", "deadline"])
@pytest.mark.parametrize("fenced", [False, True])
def test_public_reset_waiter_cannot_change_running_reset(transport, interruption, fenced):
    from concurrent.futures import ThreadPoolExecutor

    from inference_service.unified_runtime import LifecycleState, OutcomeState

    services, node, endpoint = transport
    entered, release = threading.Event(), threading.Event()
    resets = []

    def reset():
        resets.append(True)
        entered.set()
        assert release.wait(3)

    handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}, reset=reset), resettable=True)
    handle.load()
    binding = bind_runtime(node, handle, replace(target(), retry_window_ms=5000))
    for _ in range(3):
        binding.adapter._heartbeat()
    if fenced:
        endpoint.core.fence_resource("cpu")
    try:
        with _runtime_pump(endpoint._tick, binding.adapter._heartbeat), ThreadPoolExecutor(1) as pool:
            first = pool.submit(handle.reset, ExecutionContext("first", Deadline.after(3)))
            try:
                assert entered.wait(1)
                context = ExecutionContext("second", Deadline.after(0.03 if interruption == "deadline" else 1))
                if interruption == "cancel":
                    context.token.cancel("only the second waiter leaves")
                with pytest.raises(ExecutionFailure) as failure:
                    handle.reset(context)
                assert failure.value.evidence.state is OutcomeState.NOT_STARTED
                assert handle.state is LifecycleState.RESETTING
            finally:
                release.set()
            first.result(timeout=2)
            assert handle.local_ready and resets == [True]
            assert handle.execute(ModelRequest(), ExecutionContext("after-reset", Deadline.after(1))).outputs == {}
    finally:
        release.set()
        _dispose_fake_transport(binding.adapter)


@pytest.mark.parametrize("interruption", ["refresh_timeout", "refresh_cancel", "ack_lost"])
def test_public_reset_replays_completion_after_delivery_failure(transport, monkeypatch, interruption):
    from inference_service.unified_runtime import OutcomeState
    from inference_service.unified_runtime.errors import CancellationRequested

    services, node, endpoint = transport
    resets = []
    handle = ModelRuntimeHandle(
        SimpleNamespace(execute=lambda *_: {"answer": 1}, reset=lambda: resets.append(True)), resettable=True
    )
    handle.load()
    binding = bind_runtime(node, handle, replace(target(), retry_window_ms=3000))
    adapter = binding.adapter
    for _ in range(3):
        adapter._heartbeat()
    endpoint.core.fence_resource("cpu")
    context = ExecutionContext("reset-with-lost-delivery", Deadline.after(1))
    try:
        with _runtime_pump(endpoint._tick, adapter._heartbeat):
            with monkeypatch.context() as patch:
                if interruption == "ack_lost":
                    call = adapter._call_control

                    def lose_ack(message, *args, **kwargs):
                        result = call(message, *args, **kwargs)
                        if message.get("verb") == "recover" and message.get("acknowledged"):
                            context.token.cancel("ACK reply lost after Reset completed")
                            raise runtime_ros.TimeoutError("ACK reply lost")
                        return result

                    patch.setattr(adapter, "_call_control", lose_ack)
                else:

                    def fail_refresh(_):
                        if interruption == "refresh_cancel":
                            raise CancellationRequested(phase="runtime_registration")
                        raise TimeoutError("registration refresh timed out")

                    patch.setattr(adapter, "_refresh_after_reset", fail_refresh)
                with pytest.raises(ExecutionFailure) as failure:
                    handle.reset(context)
            assert failure.value.evidence.state is OutcomeState.STARTED
            assert failure.value.evidence.outcome_known
            assert handle.local_ready and not handle.reset_required
            operation = adapter._recovery_requests["reset"]["operation_id"]
            assert endpoint.recovery.records[operation]["state"] == "completed"
            handle.reset(ExecutionContext("retry-original-reset", Deadline.after(2)))
            assert resets == [True] and handle.ready
            assert not adapter.has_recovery("reset")
            assert handle.execute(ModelRequest(), ExecutionContext("after-reset", Deadline.after(1))).outputs == {
                "answer": 1
            }
    finally:
        _dispose_fake_transport(adapter)


@pytest.mark.parametrize("operation", ["reset", "reset_stream"])
@pytest.mark.parametrize("interruption", ["refresh_timeout", "refresh_cancel", "ack_lost", "late_completion"])
def test_scheduled_reset_retry_preserves_state_after_delivery_failure(transport, monkeypatch, operation, interruption):
    from inference_service.unified_runtime import OutcomeState, StreamHandle
    from inference_service.unified_runtime.errors import CancellationRequested

    _, node, endpoint = transport
    resets, state = [], []
    model = StreamFixture()
    model.open_stream = lambda context: StreamHandle(str(uuid4()))

    def reset(*_):
        resets.append(True)
        state.clear()
        if interruption == "late_completion" and len(resets) == 1:
            time.sleep(0.35)

    def execute(*_):
        state.append("new state")
        return {"count": len(state)}

    model.reset = model.reset_stream = reset
    model.execute = model.step = execute
    handle = ModelRuntimeHandle(
        executor=model,
        streaming_runtime=model,
        stateful=True,
        resettable=True,
        state_scope="stream",
    )
    handle.load()
    binding = bind_runtime(node, handle, replace(target(), retry_window_ms=100))
    adapter = binding.adapter
    try:
        with _runtime_pump(endpoint._tick, adapter._heartbeat):
            adapter.wait_for_registration(ExecutionContext("ready", Deadline.after(1)))
            stream = handle.open_stream(ExecutionContext("open", Deadline.after(1)))

            def public_reset(context):
                return handle.reset(context) if operation == "reset" else handle.reset_stream(stream, context)

            assert endpoint.core.resource_ready("cpu") and not endpoint.core._fenced
            with monkeypatch.context() as patch:
                if interruption in {"refresh_timeout", "refresh_cancel"}:

                    def fail_refresh(_):
                        if interruption == "refresh_cancel":
                            raise CancellationRequested(phase="runtime_registration")
                        raise TimeoutError("registration refresh timed out")

                    patch.setattr(adapter, "_refresh_after_reset", fail_refresh)
                elif interruption == "late_completion":
                    call = adapter._call

                    def deliver_after_completion(verb, submission, *args, **kwargs):
                        result = call(verb, submission, *args, **kwargs)
                        if verb == "submit":
                            adapter.controller.completion(submission).result(timeout=1)
                        return result

                    patch.setattr(adapter, "_call", deliver_after_completion)
                else:
                    # Lose both terminal-report confirmation and the independent
                    # Global ACK at the host, after the backend has completed.
                    patch.setattr(adapter.controller, "acknowledge", lambda *_args, **_kwargs: None)
                with pytest.raises(ExecutionFailure) as failure:
                    public_reset(ExecutionContext("first-reset", Deadline.after(0.3)))
            assert failure.value.evidence.state is not OutcomeState.NOT_STARTED
            assert failure.value.evidence.outcome_known
            assert resets == [True]
            # Expire ordinary replay history before retrying the public call.
            time.sleep(0.15)
            context = ExecutionContext("new-work", Deadline.after(1))
            result = handle.step(stream, ModelRequest(), context)
            assert result.outputs == {"count": 1}
            public_reset(ExecutionContext("retry-original-reset", Deadline.after(1)))
            assert resets == [True] and state == ["new state"]
            assert not endpoint.recovery.records
            # A successfully delivered reset releases its identity for new intent.
            public_reset(ExecutionContext("distinct-reset", Deadline.after(1)))
            assert resets == [True, True] and state == []
    finally:
        _dispose_fake_transport(adapter)


@pytest.mark.parametrize("operation", ["reset", "reset_stream"])
def test_close_releases_abandoned_scheduled_reset_waiter(transport, monkeypatch, operation):
    from inference_service.unified_runtime import StreamHandle

    _, node, endpoint = transport
    resets = []
    model = StreamFixture()
    model.open_stream = lambda context: StreamHandle(str(uuid4()))
    model.reset = model.reset_stream = lambda *_: resets.append(True)
    model.close = lambda: None
    handle = ModelRuntimeHandle(
        executor=model, streaming_runtime=model, stateful=True, resettable=True, state_scope="stream"
    )
    handle.load()
    binding = bind_runtime(node, handle, replace(target(), retry_window_ms=100, max_records=3, max_pending_requests=1))
    adapter = binding.adapter
    try:
        with _runtime_pump(endpoint._tick, adapter._heartbeat):
            adapter.wait_for_registration(ExecutionContext("ready", Deadline.after(1)))
            # Repeated stream closure must not exhaust the bounded operation store.
            for index in range(1 if operation == "reset" else 5):
                stream = handle.open_stream(ExecutionContext("open", Deadline.after(1)))
                with monkeypatch.context() as patch:
                    patch.setattr(adapter.controller, "acknowledge", lambda *_args, **_kwargs: None)
                    with pytest.raises(ExecutionFailure):
                        context = ExecutionContext("abandoned-reset", Deadline.after(0.3))
                        if operation == "reset":
                            handle.reset(context)
                        else:
                            handle.reset_stream(stream, context)
                context = ExecutionContext("close", Deadline.after(1))
                if operation == "reset":
                    handle.close(context)
                    assert handle.closed
                else:
                    handle.close_stream(stream, context)
                    assert stream.closed
                assert resets == [True] * (index + 1)
                assert not adapter._reset_requests
                # Global also retains history through the business deadline.
                time.sleep(1.05)
    finally:
        _dispose_fake_transport(adapter)


def test_recovery_lock_wait_observes_cancellation(transport):
    from inference_service.unified_runtime.errors import CancellationRequested

    services, _, _ = transport
    adapter, _ = host(services, target())
    context = ExecutionContext("waiting-recovery", Deadline.after(60))
    done = threading.Event()
    errors = []

    def recover():
        try:
            adapter.recover(context)
        except Exception as exc:
            errors.append(exc)
        finally:
            done.set()

    adapter._recovery_lock.acquire()
    worker = threading.Thread(target=recover)
    worker.start()
    try:
        context.cancellation_token.cancel()
        assert done.wait(0.5), "canceled recovery continued waiting on a busy lock"
        assert len(errors) == 1 and isinstance(errors[0], CancellationRequested)
    finally:
        adapter._recovery_lock.release()
        worker.join(2)
        _dispose_fake_transport(adapter)


@pytest.mark.parametrize("replace_boot", [False, True])
def test_lost_unregister_reply_does_not_resurrect_closed_dynamic_host(transport, monkeypatch, replace_boot):
    services, node, endpoint = transport
    handles, bindings = [], []
    closes = []
    for name in ("retiring", "survivor"):
        handle = ModelRuntimeHandle(
            SimpleNamespace(execute=lambda *_: {"answer": 1}, close=lambda: closes.append(True))
        )
        handle.load()
        binding = bind_runtime(node, handle, target(name))
        handles.append(handle)
        bindings.append(binding)
        for _ in range(3):
            binding.adapter._heartbeat()
    original = bindings[0].adapter._call_control

    def lose_reply(payload, *args, **kwargs):
        result = original(payload, *args, **kwargs)
        if payload["verb"] == "unregister":
            raise TimeoutError("lost unregister reply")
        return result

    def pump_once():
        endpoint._tick()
        for binding in bindings:
            if not binding._closed:
                binding.adapter._heartbeat()

    try:
        with _runtime_pump(pump_once):
            monkeypatch.setattr(bindings[0].adapter, "_call_control", lose_reply)
            with pytest.raises(ExecutionFailure):
                handles[0].close(deadline=Deadline.after(0.3))
            assert handles[0].closed and closes == [True]
            if replace_boot:
                replacement, _ = host(services, bindings[0].target)
                assert endpoint.core.host_boot(replacement.target.target_id) == replacement.controller.boot_id
            if not replace_boot:
                time.sleep(2.2)  # Cross the retirement floor while the control loop is alive.
            if replace_boot:
                replacement.refresh_registration(ExecutionContext("replacement-ready", Deadline.after(1)))
            bindings[1].adapter.refresh_registration(ExecutionContext("peer-ready", Deadline.after(1)))
            assert endpoint.core.resource_ready("cpu")
            assert handles[1].execute(ModelRequest(), ExecutionContext("peer-execute", Deadline.after(1))).outputs == {
                "answer": 1
            }
            monkeypatch.setattr(bindings[0].adapter, "_call_control", original)
            handles[0].close(deadline=Deadline.after(2))
            assert bindings[0]._closed and closes == [True]
    finally:
        for binding in bindings:
            _dispose_fake_transport(binding.adapter)
        if replace_boot and "replacement" in locals():
            _dispose_fake_transport(replacement)


def test_public_binding_bounds_diagnostic_after_waiter_abandons_late_failure(transport):
    from inference_service.scheduler.runtime_controller import retain_result
    from inference_service.unified_runtime import OutcomeEvidence

    services, node, endpoint = transport
    entered, release = threading.Event(), threading.Event()

    def execute(*_):
        entered.set()
        assert release.wait(2)
        raise ExecutionFailure(
            "late_failure",
            "known backend failure",
            evidence=OutcomeEvidence.completed(),
            details={"payload": "x" * 1024 * 1024},
        )

    handle = ModelRuntimeHandle(SimpleNamespace(execute=execute))
    handle.load()
    binding = bind_runtime(node, handle, replace(target(), max_retained_result_bytes=4096))
    for _ in range(3):
        binding.adapter._heartbeat()
    original = binding.adapter._call

    def dispatch(verb, submission, timeout, **kwargs):
        result = original(verb, submission, timeout, **kwargs)
        if verb == "submit":
            endpoint._tick()
            assert entered.wait(1)
            raise TimeoutError("lost submit response")
        return result

    binding.adapter._call = dispatch
    try:
        with pytest.raises(ExecutionFailure):
            handle.execute(ModelRequest(), ExecutionContext("abandon", Deadline.after(0.05)))
        release.set()
        record = next(iter(binding.controller._records.values()))
        record.done.result(timeout=1)
        assert binding.controller._result_bytes == 0
        assert handle.last_failure.code == "late_failure"
        assert handle.last_failure.details["diagnostic_summary"]
        assert retain_result(handle.last_failure.details)[1] < 4096
        assert "payload" not in handle.last_failure.details
    finally:
        release.set()
        _dispose_fake_transport(binding.adapter)


@pytest.mark.parametrize("background_first", [False, True])
def test_normal_completion_reuses_one_terminal_receipt(transport, monkeypatch, background_first):
    _, _, endpoint = transport
    services = transport[0]
    adapter, handle = host(services, target())
    sent = []
    original_send = adapter._client.call_async
    original_call = adapter._call

    def send(request):
        sent.append(decode_control(request.payload)["verb"])
        return original_send(request)

    def call(verb, submission, timeout, **kwargs):
        response = original_call(verb, submission, timeout, **kwargs)
        if verb == "submit":
            # Dispatch without Global Query/ACK, then choose which waiter
            # observes the terminal before the business loop continues.
            endpoint._tick()
            adapter.controller.completion(submission).result(timeout=1)
            if background_first:
                adapter._terminal_tick()
        return response

    monkeypatch.setattr(adapter._client, "call_async", send)
    monkeypatch.setattr(adapter, "_call", call)
    try:
        result = handle.execute(ModelRequest(), ExecutionContext("one-receipt", Deadline.after(1)))
        assert result.outputs == {"answer": 1}
        assert sent == ["submit", "terminal_report"]
    finally:
        _dispose_fake_transport(adapter)


def test_lost_old_terminal_reply_does_not_delay_new_short_request(transport, monkeypatch):
    services, _, endpoint = transport
    adapter, handle = host(services, target())
    original_send, original_call = adapter._client.call_async, adapter._call
    old = adapter.controller.prepare(ModelRequest(), ExecutionContext("old", Deadline.after(5)))
    endpoint.core.submit(old)
    adapter.controller.cancel(old, adapter.controller.epoch)
    lost, reports = Future(), []

    def send(request):
        message = decode_control(request.payload)
        if message["verb"] == "terminal_report":
            operation = message["report"]["submission"]["operation_id"]
            reports.append(operation)
            response = original_send(request)
            return lost if operation == old.operation_id else response
        return original_send(request)

    def dispatch(verb, submission, timeout, **kwargs):
        result = original_call(verb, submission, timeout, **kwargs)
        if verb == "submit":
            endpoint._tick()
            adapter.controller.completion(submission).result(timeout=1)
        return result

    monkeypatch.setattr(adapter._client, "call_async", send)
    monkeypatch.setattr(adapter, "_call", dispatch)
    try:
        adapter._terminal_tick()
        assert not lost.done()
        result = handle.execute(ModelRequest(), ExecutionContext("new", Deadline.after(0.15)))
        assert result.outputs == {"answer": 1}
        assert len(reports) == 2 and reports[0] == old.operation_id and reports[1] != old.operation_id
        assert not lost.done()
    finally:
        monkeypatch.undo()
        _dispose_fake_transport(adapter)


@pytest.mark.parametrize("destination", ["host", "global"])
@pytest.mark.parametrize(
    "invalid", [{"operation_id": ""}, {"operation_id": "not-a-uuid"}, {"acknowledged": "false"}, {"acknowledged": 1}]
)
def test_recovery_wire_rejects_invalid_identity_and_ack_before_mutation(transport, destination, invalid):
    services, _, endpoint = transport
    adapter, _ = host(services, replace(target(), retry_window_ms=3000))
    message = {
        "verb": "recover",
        "epoch": endpoint.core.epoch,
        "operation_id": str(uuid4()),
        "resource": "cpu",
        "target_id": adapter.target.target_id,
        "boot_id": adapter.controller.boot_id,
        "lifecycle_group": adapter.target.lifecycle_group,
        "command": "reset",
        "accept_until_ns": time.monotonic_ns() + 1_000_000_000,
    }
    address = adapter.target.endpoint if destination == "host" else adapter._client.endpoint
    try:
        reply = services[address]({**message, **invalid})
        assert reply.get("control_error") == "INVALID_CONTROL"
        assert adapter.controller.ready and adapter.controller.generation == 1
        assert not adapter.controller._recovery_records and not endpoint.recovery.records
        assert endpoint.core.resource_ready("cpu")
    finally:
        _dispose_fake_transport(adapter)


@pytest.mark.parametrize("destination", ["host", "global"])
def test_non_boolean_ack_cannot_settle_a_completed_recovery(transport, monkeypatch, destination):
    services, _, endpoint = transport
    adapter, _ = host(services, replace(target(), retry_window_ms=3000))
    operation = str(uuid4())
    expiry = time.monotonic_ns() + 1_000_000_000
    message = {
        "verb": "recover",
        "epoch": endpoint.core.epoch,
        "operation_id": operation,
        "resource": "cpu",
        "command": "drain",
        "accept_until_ns": expiry,
        "boot_id": adapter.controller.boot_id,
        "lifecycle_group": adapter.target.lifecycle_group,
    }
    try:
        if destination == "host":
            adapter.controller.recover(adapter.controller.epoch, operation, "drain", expiry)
            receipt = adapter.controller._recovery_records[operation]
            receipt.future.result(timeout=1)
            address = adapter.target.endpoint
        else:
            endpoint.recovery.request({k: v for k, v in message.items() if k not in {"verb", "epoch"}})
            receipt = endpoint.recovery.records[operation]
            receipt["state"] = "completed"
            address = adapter._client.endpoint
        assert services[address]({**message, "acknowledged": "false"}).get("control_error") == "INVALID_CONTROL"
        if destination == "host":
            monkeypatch.setattr(adapter.controller, "now_ns", lambda: expiry + 1)
            adapter.controller.collect()
            assert not receipt.acknowledged and operation in adapter.controller._recovery_records
        else:
            monkeypatch.setattr(endpoint.core, "now_ns", lambda: expiry + 1)
            endpoint.recovery.collect()
            assert not receipt.get("acknowledged") and operation in endpoint.recovery.records
    finally:
        monkeypatch.undo()
        _dispose_fake_transport(adapter)


def test_adapter_close_retry_settles_original_completed_recovery(transport, monkeypatch):
    services, _, endpoint = transport
    adapter, handle = host(services, replace(target(), retry_window_ms=3000))
    original = adapter._call_control
    first_context = ExecutionContext("first-close", Deadline.after(1))
    interrupted = False
    adapter.controller.fence()  # Ordinary Close must take the recovery path.

    def progress(message, timeout):
        nonlocal interrupted
        reply = original(message, timeout)
        endpoint._tick()
        if message.get("verb") == "recover" and reply.get("state") == "completed" and not interrupted:
            interrupted = True
            time.sleep(max(0, first_context.deadline.remaining_seconds()) + 0.001)
            raise TimeoutError("completion response lost before settlement ACK")
        return reply

    monkeypatch.setattr(adapter, "_call_control", progress)
    try:
        with pytest.raises(TimeoutError, match="deadline|settlement ACK"):
            adapter.close(first_context)
        assert handle.closed and adapter.has_recovery("close")
        operation = adapter._recovery_requests["close"]["operation_id"]
        record = endpoint.recovery.records[operation]
        assert not record.get("acknowledged")
        adapter.close(ExecutionContext("retry-close", Deadline.after(2)))
        assert not adapter.has_recovery("close") and record["acknowledged"]
        assert len(endpoint.recovery.records) == 1
        monkeypatch.setattr(endpoint.core, "now_ns", lambda: record["accept_until_ns"] + 1)
        endpoint.recovery.collect()
        assert operation not in endpoint.recovery.records
    finally:
        monkeypatch.undo()
        _dispose_fake_transport(adapter)


@pytest.mark.parametrize("command", ["reset", "close"])
def test_global_completed_recovery_retained_until_caller_acknowledges(transport, command):
    services, _, endpoint = transport
    adapter, _ = host(services, target())
    clock = [endpoint.core.now_ns()]
    endpoint.core.now_ns = lambda: clock[0]
    request = {
        "resource": "cpu",
        "operation_id": str(uuid4()),
        "command": command,
        "target_id": adapter.target.target_id,
        "boot_id": adapter.controller.boot_id,
        "accept_until_ns": clock[0] + 1_000_000,
    }
    try:
        endpoint.recovery.request(request)
        record = endpoint.recovery.records[request["operation_id"]]
        record.update(state="completed", phase="activate")
        clock[0] += 2_000_000
        other = {**request, "operation_id": str(uuid4()), "accept_until_ns": clock[0] + 1_000_000}
        endpoint.recovery.request(other)  # Triggers historical collection.
        resumed = {**request, "resume_until_ns": clock[0] + 1_000_000}
        assert endpoint.recovery.request(resumed)["state"] == "completed"
        assert endpoint.recovery.request({**resumed, "acknowledged": True})["state"] == "acknowledged"
        endpoint.recovery.records[other["operation_id"]].update(state="completed", acknowledged=True)
        clock[0] += 2_000_000
        endpoint.recovery.tick()  # No new request is needed to collect settled receipts.
        assert request["operation_id"] not in endpoint.recovery.records
        assert endpoint.recovery.request({**resumed, "acknowledged": True})["state"] == "acknowledged"
    finally:
        _dispose_fake_transport(adapter)


@pytest.mark.parametrize(
    "state,acknowledged,command",
    [
        ("unknown", False, "reset"),
        ("completed", False, "reset"),
        ("superseded", False, "close"),
        ("completed", True, "close"),
    ],
)
def test_recovery_tick_retains_unsettled_evidence(transport, state, acknowledged, command):
    services, _, endpoint = transport
    adapter, _ = host(services, target())
    clock = [endpoint.core.now_ns()]
    endpoint.core.now_ns = lambda: clock[0]
    request = {
        "resource": "cpu",
        "operation_id": str(uuid4()),
        "command": command,
        "accept_until_ns": clock[0] + 1_000_000,
    }
    try:
        endpoint.recovery.request(request)
        endpoint.recovery.records[request["operation_id"]].update(state=state, acknowledged=acknowledged)
        clock[0] += 2_000_000
        endpoint.recovery.tick()
        assert (request["operation_id"] in endpoint.recovery.records) is (not acknowledged)
    finally:
        _dispose_fake_transport(adapter)


@pytest.mark.parametrize("stream_id", ["s" * 32400, "流" * 342])
def test_oversized_stream_recovery_does_not_fence_or_cancel_device(transport, stream_id):
    services, _, endpoint = transport
    adapter, _ = host(services, target())
    core = endpoint.core
    try:
        submission = adapter.controller.prepare(ModelRequest(), ExecutionContext("queued", Deadline.after(5)))
        core.submit(submission)
        request = {
            "resource": "cpu",
            "operation_id": str(uuid4()),
            "command": "reset_stream",
            "target_id": adapter.target.target_id,
            "boot_id": adapter.controller.boot_id,
            "accept_until_ns": core.now_ns() + 1_000_000_000,
            "stream_id": stream_id,
        }
        encode_control(request)  # The incoming wire request itself fits.
        with pytest.raises(ControlError, match="INVALID_RECOVERY_STREAM"):
            endpoint.recovery.request(request)
        assert core.resource_ready("cpu") and not core._fenced
        assert not endpoint.recovery.records
        assert core.query(submission)["state"] == "queued"
        core.cancel(submission)
    finally:
        _dispose_fake_transport(adapter)


def test_stream_recovery_envelopes_fit_with_maximum_supported_identity(transport):
    services, _, endpoint = transport
    adapter, _ = host(services, replace(target(), lifecycle_group="g" * 127, retry_window_ms=3000))
    try:
        request = {
            "resource": "cpu",
            "operation_id": str(uuid4()),
            "command": "close_stream",
            "target_id": adapter.target.target_id,
            "boot_id": adapter.controller.boot_id,
            "accept_until_ns": endpoint.core.now_ns() + 1_000_000_000,
            "stream_id": "\x00" * 1024,
        }
        endpoint.recovery.request(request)
        record = endpoint.recovery.records[request["operation_id"]]
        record.update(phase="close_stream", accept_until_ns=request["accept_until_ns"] + 1_000_000)
        command = {"id": str(uuid4()), "accept_until_ns": request["accept_until_ns"], "result": {}}
        payload = endpoint.recovery._command_payload(record, adapter.target, command)
        assert decode_control(encode_control(payload))["stream_id"] == request["stream_id"]
        assert payload["acknowledged"] and payload["resume_until_ns"] > payload["accept_until_ns"]
        endpoint.recovery.records.clear()
    finally:
        _dispose_fake_transport(adapter)


@pytest.mark.parametrize("path", ["callback", "poll", "retirement"])
def test_terminal_ack_paths_preserve_global_outcome(transport, monkeypatch, path):
    from concurrent.futures import Future

    services, _, endpoint = transport
    adapter, _ = host(services, target())
    try:
        submission = adapter.controller.prepare(ModelRequest(), ExecutionContext("ack", Deadline.after(5)))
        adapter.controller.cancel(submission, adapter.controller.epoch)
        outcome = {"state": "not_started", "error": "GLOBAL_ADMISSION_REJECTED"}
        reply = {"state": "acknowledged", "outcome": outcome}
        future = Future()
        future.set_result(SimpleNamespace(payload=encode_control(reply)))
        if path == "retirement":

            def call(payload, timeout):
                if payload["verb"] == "unregister":
                    raise ControlError("HOST_NOT_DRAINED")
                return reply

            monkeypatch.setattr(adapter, "_call_control", call)
            adapter._unregister_clean_host()
        else:
            adapter._terminal_receipts[future] = (
                submission,
                time.monotonic(),
                adapter.controller.terminal_receipt(submission),
            )
            if path == "callback":
                adapter._terminal_receipt_done(future)
            else:
                adapter._flush_terminal_evidence()
        assert adapter.controller.terminal_receipt(submission).result(timeout=1) == outcome
    finally:
        monkeypatch.undo()
        _dispose_fake_transport(adapter)


@pytest.mark.parametrize("recovery_close", [False, True])
def test_close_settles_completed_reset_after_caller_abandons_ack(transport, monkeypatch, recovery_close):
    from inference_service.scheduler import runtime_recovery

    services, node, endpoint = transport
    monkeypatch.setattr(runtime_recovery, "MAX_RESOURCE_RECOVERY_RECEIPTS", 3)
    adapters = []
    reset_ids = []

    def pump_once():
        endpoint._tick()
        for adapter in tuple(adapters):
            adapter._heartbeat()

    try:
        with _runtime_pump(pump_once):
            # More incarnations than the device receipt limit, on the same device.
            for index in range(5):
                calls = []
                handle = ModelRuntimeHandle(
                    SimpleNamespace(
                        execute=lambda *_: {},
                        reset=lambda context, *, calls=calls: calls.append("reset"),
                        close=lambda *, calls=calls: calls.append("close"),
                    ),
                    resettable=True,
                )
                handle.load()
                binding = bind_runtime(node, handle, replace(target(f"owner_{index}"), retry_window_ms=300))
                adapter = binding.adapter
                adapters.append(adapter)
                adapter.wait_for_registration(ExecutionContext("registered", Deadline.after(1)))
                refresh = adapter._refresh_after_reset

                def abandon(_):
                    raise TimeoutError("caller timed out before registration refresh and ACK")

                monkeypatch.setattr(adapter, "_refresh_after_reset", abandon)
                with pytest.raises(runtime_ros.CompletedControlError, match="COMPLETED_DELIVERY_FAILED"):
                    adapter.recover(ExecutionContext("reset", Deadline.after(2)))
                message = dict(adapter._recovery_requests["reset"])
                message.pop("acknowledged", None)
                reset_ids.append(message["operation_id"])
                record = endpoint.recovery.records[message["operation_id"]]
                assert record["state"] == "completed" and not record.get("acknowledged")
                monkeypatch.setattr(adapter, "_refresh_after_reset", refresh)
                if recovery_close:
                    adapter.recover(ExecutionContext("close", Deadline.after(2)), command="close")
                handle.close(deadline=Deadline.after(2))
                assert calls == ["reset", "close"]
                assert record["acknowledged"]
                # Settlement does not erase the replay result inside its window.
                assert endpoint.recovery.request(message)["state"] == "completed"
                while time.monotonic_ns() <= record["accept_until_ns"]:
                    time.sleep(0.01)
            # Every earlier owner's completed receipt was collected on subsequent admission.
            assert set(reset_ids[:-1]).isdisjoint(endpoint.recovery.records)
    finally:
        for adapter in adapters:
            _dispose_fake_transport(adapter)


@pytest.mark.parametrize("operation", ["reset", "reset_stream", "close_stream"])
def test_control_recovers_when_input_bytes_are_full(transport, operation):
    import numpy as np

    from inference_service.scheduler.runtime_controller import input_bytes
    from inference_service.unified_runtime import StreamHandle

    _, node, endpoint = transport
    calls = []
    model = StreamFixture()
    model.open_stream = lambda context: StreamHandle(str(uuid4()))
    model.reset = lambda: calls.append("reset")
    model.drain_uncertain_operations = lambda: None
    model.reset_stream = lambda *_: calls.append("reset_stream")
    model.close_stream = lambda *_: calls.append("close_stream")
    handle = ModelRuntimeHandle(
        executor=model, streaming_runtime=model, stateful=True, resettable=True, state_scope="stream"
    )
    handle.load()
    binding = bind_runtime(node, handle, replace(target(), max_retained_input_bytes=4096, retry_window_ms=1000))
    adapter = binding.adapter
    try:
        with _runtime_pump(endpoint._tick, adapter._heartbeat):
            adapter.wait_for_registration(ExecutionContext("ready", Deadline.after(1)))
            stream = None if operation == "reset" else handle.open_stream(ExecutionContext("open", Deadline.after(1)))
        overhead = input_bytes(ModelRequest({"x": np.zeros(0, dtype=np.uint8)}))
        queued = adapter.controller.prepare(
            ModelRequest({"x": np.zeros(4096 - 8 - overhead, dtype=np.uint8)}),
            ExecutionContext("large-input", Deadline.after(5)),
        )
        endpoint.core.submit(queued)
        assert adapter.controller._input_bytes == 4088
        assert len(adapter.controller._records) < adapter.target.max_records
        with _runtime_pump(lambda: endpoint._tick() if endpoint.recovery.records else None, adapter._heartbeat):
            context = ExecutionContext(operation, Deadline.after(2))
            if operation == "reset":
                handle.reset(context)
            else:
                getattr(handle, operation)(stream, context)
            assert calls == [operation]
            assert endpoint.core.query(queued)["state"] == "not_started"
            assert not endpoint.core._fenced
    finally:
        _dispose_fake_transport(adapter)


@pytest.mark.parametrize("recovery_close", [False, True])
def test_stream_close_settles_abandoned_completed_reset_receipt(transport, monkeypatch, recovery_close):
    from inference_service.scheduler import runtime_recovery
    from inference_service.unified_runtime import OutcomeState, StreamHandle

    _, node, endpoint = transport
    monkeypatch.setattr(runtime_recovery, "MAX_RESOURCE_RECOVERY_RECEIPTS", 3)
    calls = []
    model = StreamFixture()
    model.open_stream = lambda context: StreamHandle(str(uuid4()))
    model.reset_stream = lambda stream, context: calls.append(stream.stream_id)
    model.drain_uncertain_operations = lambda: None
    handle = ModelRuntimeHandle(
        executor=model,
        streaming_runtime=model,
        stateful=True,
        resettable=True,
        state_scope="stream",
        state_bank_mode="per_stream",
        max_open_streams=2,
    )
    handle.load()
    binding = bind_runtime(node, handle, replace(target(), retry_window_ms=1000))
    adapter = binding.adapter
    try:
        with _runtime_pump(endpoint._tick, adapter._heartbeat):
            adapter.wait_for_registration(ExecutionContext("ready", Deadline.after(1)))
            survivor = handle.open_stream(ExecutionContext("survivor", Deadline.after(1)))
            for index in range(5):
                stream = handle.open_stream(ExecutionContext(f"open-{index}", Deadline.after(1)))
                with monkeypatch.context() as patch:

                    def abandon(_):
                        raise TimeoutError("caller left before registration and ACK")

                    patch.setattr(adapter, "_refresh_after_reset", abandon)
                    with pytest.raises(runtime_ros.CompletedControlError):
                        adapter.recover(
                            ExecutionContext("reset", Deadline.after(2)),
                            command="reset_stream",
                            stream_id=stream.stream_id,
                        )
                    with pytest.raises(ExecutionFailure) as close_failure:
                        handle.close_stream(stream, ExecutionContext("close-before-ack", Deadline.after(1)))
                    assert close_failure.value.evidence.state is OutcomeState.NOT_STARTED
                    assert not stream.closed
                message = dict(adapter._recovery_requests[("reset_stream", stream.stream_id)])
                message.pop("acknowledged")
                record = endpoint.recovery.records[message["operation_id"]]
                assert record["state"] == "completed" and not record.get("acknowledged")
                if recovery_close:
                    endpoint.core.fence_resource("cpu")
                handle.close_stream(stream, ExecutionContext("close-stream", Deadline.after(2)))
                assert stream.closed and record["acknowledged"]
                assert not adapter.has_recovery(("reset_stream", stream.stream_id))
                # A competing waiter may enter after the receipt was removed.
                # Settlement must not turn that absence into another Reset.
                adapter.recover(
                    ExecutionContext("already-settled", Deadline.after(1)),
                    command="reset_stream",
                    stream_id=stream.stream_id,
                    completed_only=True,
                )
                assert endpoint.recovery.request(message)["state"] == "completed"
                assert (
                    handle.step(survivor, ModelRequest(), ExecutionContext("survives", Deadline.after(1))).outputs == {}
                )
                while time.monotonic_ns() <= record["accept_until_ns"]:
                    time.sleep(0.01)
            assert len(calls) == 5 and not handle.closed
    finally:
        _dispose_fake_transport(adapter)


@pytest.mark.parametrize("profile_failure", ["expired", "missing_close_bucket"])
def test_existing_stream_can_close_without_predictive_cost(transport, tmp_path, profile_failure):
    import json

    from inference_service.scheduler.runtime_controller import freeze_inputs
    from inference_service.scheduler.runtime_profiles import RuntimeProfileRegistry
    from inference_service.scheduler.runtime_protocol import RUNTIME_HANDLE_OPERATION
    from inference_service.unified_runtime import StreamHandle

    from .runtime_scheduler_helpers import attested_resources

    _, node, _ = transport
    path = tmp_path / "stream-profile.json"
    selected = replace(
        target(),
        predictive_admission=True,
        profile_path=str(path),
        retry_window_ms=1000,
        hardware_profile_fingerprint="cpu-v1",
        runtime_version="test-v1",
    )
    bucket = freeze_inputs(ModelRequest(metadata={RUNTIME_HANDLE_OPERATION: "open_stream"}), 4096)[3]
    measured = time.time()
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "scope": "runtime_execution",
                "identity": RuntimeProfileRegistry.identity(selected),
                "measured_at": measured,
                "buckets": [{"work": "execute", "input_bucket": bucket, "samples": 10000, "upper_bound_ns": 1000}],
            }
        )
    )
    endpoint = GlobalRuntimeEndpoint(node, [selected], resources=attested_resources(selected))
    model = StreamFixture()
    model.runtime_version = "test-v1"
    model.drain_uncertain_operations = lambda: None
    model.open_stream = lambda context: StreamHandle(str(uuid4()))
    closed = []
    model.close_stream = lambda stream, context: closed.append(stream.stream_id)
    handle = ModelRuntimeHandle(
        executor=model, streaming_runtime=model, session=model, stateful=True, state_scope="stream"
    )
    handle.load()
    binding = bind_runtime(node, handle, selected)
    try:
        with _runtime_pump(endpoint._tick, binding.adapter._heartbeat):
            binding.adapter.wait_for_registration(ExecutionContext("ready", Deadline.after(1)))
            stream = handle.open_stream(ExecutionContext("open", Deadline.after(1)))
            if profile_failure == "expired":
                endpoint.core._profiles[selected.target_id]._now = lambda: measured + 31 * 86400
            handle.close_stream(stream, ExecutionContext("close", Deadline.after(2)))
            handle.close_stream(stream, ExecutionContext("replay", Deadline.after(1)))
            assert stream.closed and handle.open_stream_count == 0
            assert len(closed) == 1 and not handle.closed
    finally:
        _dispose_fake_transport(binding.adapter)


def test_runtime_reset_collects_streams_closed_without_individual_receipts(transport):
    from inference_service.unified_runtime import StreamHandle

    _, node, endpoint = transport
    model = StreamFixture()
    model.open_stream = lambda context: StreamHandle(str(uuid4()))
    handle = ModelRuntimeHandle(
        executor=model, streaming_runtime=model, stateful=True, resettable=True, state_scope="stream"
    )
    handle.load()
    binding = bind_runtime(node, handle, replace(target(), retry_window_ms=1000))
    try:
        with _runtime_pump(endpoint._tick, binding.adapter._heartbeat):
            binding.adapter.wait_for_registration(ExecutionContext("ready", Deadline.after(1)))
            for index in range(130):
                stream = handle.open_stream(ExecutionContext(f"open-{index}", Deadline.after(1)))
                with pytest.raises(ExecutionFailure):
                    handle.close_stream(stream, ExecutionContext("expired-close", Deadline.after(0)))
                handle.reset(ExecutionContext("reset", Deadline.after(1)))
                assert stream.closed and handle.open_stream_count == 0
            assert len(handle._streams) == len(handle._closed_stream_ids) == 128
    finally:
        _dispose_fake_transport(binding.adapter)


def test_rejected_recovery_during_peer_initialization_uses_new_id_after_window(transport):
    services, _, endpoint = transport
    adapter, _ = host(services, replace(target(), retry_window_ms=300))
    peer = replace(target("initializing"), required=False)
    boot, token = str(uuid4()), str(uuid4())
    status = {
        **adapter.controller.status(),
        "target": peer.to_dict(),
        "boot_id": boot,
        "lifecycle_target_ids": [peer.target_id],
        "ready": False,
        "clean": True,
    }
    endpoint.core.begin_initialization([status], token, accept_until_ns=time.monotonic_ns() + 100_000_000)
    try:
        with pytest.raises(ControlError, match="INITIALIZATION_EVIDENCE_PENDING"):
            adapter.recover(ExecutionContext("rejected", Deadline.after(1)), command="close")
        assert not adapter._recovery_requests and not endpoint.recovery.records
        time.sleep(0.31)
        endpoint.core.finish_initialization("cpu", token, clean=True, retire=True)
        # Drive exactly the existing Global/Host control path for this retry.
        original = adapter._call_control

        def progress(message, timeout, **kwargs):
            endpoint._tick()
            return original(message, timeout, **kwargs)

        adapter._call_control = progress
        adapter.recover(ExecutionContext("retry", Deadline.after(1)), command="close")
        assert adapter.controller.handle.closed
        assert len(endpoint.recovery.records) == 1
    finally:
        _dispose_fake_transport(adapter)


def test_abandoned_closed_streams_enter_bounded_history_on_background_receipt(transport):
    from inference_service.unified_runtime import StreamHandle

    _, node, endpoint = transport
    model = StreamFixture()
    model.open_stream = lambda context: StreamHandle(str(uuid4()))
    handle = ModelRuntimeHandle(executor=model, streaming_runtime=model, stateful=True, state_scope="stream")
    handle.load()
    binding = bind_runtime(node, handle, replace(target(), retry_window_ms=1000))
    adapter = binding.adapter
    for _ in range(3):
        adapter._heartbeat()
    original = adapter._call
    flush = adapter._terminal_tick
    blocked = [False]
    adapter._terminal_tick = lambda: None if blocked[0] else flush()

    def dispatch(verb, submission, timeout, **kwargs):
        response = original(verb, submission, timeout, **kwargs)
        if verb == "submit":
            endpoint._tick()
            adapter.controller.completion(submission).result(timeout=1)
            if submission.parent_id.startswith("close-"):
                blocked[0] = True
                kwargs["context"].token.cancel("caller leaves before receipt")
        return response

    adapter._call = dispatch
    try:
        for i in range(130):
            for _ in range(3):
                adapter._heartbeat()
            stream = handle.open_stream(ExecutionContext(f"open-{i}", Deadline.after(1)))
            with pytest.raises(ExecutionFailure):
                handle.close_stream(stream, ExecutionContext(f"close-{i}", Deadline.after(1)))
            assert handle._streams[stream.stream_id].state.value == "closed"
            blocked[0] = False
            adapter._terminal_tick()
        assert len(handle._streams) == len(handle._closed_stream_ids) == 128
        assert not adapter.controller.pending_terminal_reports()
    finally:
        _dispose_fake_transport(adapter)


@pytest.mark.parametrize("kind,command", [("tensor", "close"), ("lerobot", "reset"), ("lerobot", "close")])
def test_real_torch_session_unknown_can_drain_through_global_recovery(transport, kind, command):
    from .test_torch_session_completion import session_probe

    services, node, endpoint = transport
    events, failure = [], ["sync"]
    session = session_probe(kind, True, events, failure)
    # The probe is preloaded. This facade forwards all lifecycle operations to
    # the real Session and exposes its real drain implementation to the Handle.
    model = SimpleNamespace(
        execute=session.execute,
        close=session.close,
        drain_uncertain_operations=session.drain_uncertain_operations,
    )
    if kind == "lerobot":
        session._capabilities = replace(session.capabilities, resettable=True)
        session._policy.reset = lambda: events.append("reset")
        model.reset = session.reset
    handle = ModelRuntimeHandle(model, resettable=kind == "lerobot")
    handle.load()
    binding = bind_runtime(node, handle, replace(target(), retry_window_ms=1000))
    for _ in range(3):
        binding.adapter._heartbeat()

    def pump_once():
        endpoint._tick()
        binding.adapter._heartbeat()

    try:
        with _runtime_pump(pump_once):
            try:
                with pytest.raises(ExecutionFailure):
                    handle.execute(ModelRequest(), ExecutionContext("unknown", Deadline.after(1)))
                assert any(r.state == "unknown" for r in binding.controller._records.values())
                failure[0] = None
                binding.adapter.recover(ExecutionContext("recover", Deadline.after(2)), command=command)
                assert not endpoint.core._fenced
                assert events.count("sync") >= 2
                if command == "close":
                    assert handle.closed
                else:
                    assert handle.local_ready
            finally:
                failure[0] = None
    finally:
        _dispose_fake_transport(binding.adapter)


def test_registration_rejection_is_reported_without_log_spam_and_clears_on_success(transport):
    services, node, endpoint = transport
    selected = replace(target(), hardware_profile_fingerprint="host-profile")
    adapter, handle = host(services, selected)
    try:
        assert not adapter.ready
        assert adapter.registration_error == "RESOURCE_CONFIGURATION_MISMATCH"
        assert not endpoint.core.targets
        for _ in range(5):
            adapter._heartbeat()
        assert len(adapter.node.warnings) == 1
        assert selected.target_id in adapter.node.warnings[0]
        assert adapter.registration_error in adapter.node.warnings[0]
        # Restart the authority with corrected configuration; the host retries.
        corrected = GlobalRuntimeEndpoint(
            node,
            [],
            resources={"cpu": {"physical_device": "local/cpu", "hardware_profile_fingerprint": "host-profile"}},
        )
        for _ in range(3):
            adapter._heartbeat()
        assert adapter.ready and corrected.core.resource_ready("cpu")
        assert adapter.registration_error == ""
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


@pytest.mark.parametrize("failure", ["timeout", "transport"])
def test_registration_transport_failure_remains_visible_until_a_successful_retry(transport, monkeypatch, failure):
    services, _, endpoint = transport
    handle = ModelRuntimeHandle(executor=SimpleNamespace(execute=lambda *_: {}))
    handle.load()
    adapter = SchedulerRuntimeAdapter(LocalNode(services), target(), handle)
    send = adapter._client.call_async
    pending = Future()
    if failure == "transport":
        pending.set_exception(ConnectionError("lost control reply"))
    adapter._client.call_async = lambda _: pending
    try:
        adapter._heartbeat()
        if failure == "timeout":
            with monkeypatch.context() as clock:
                clock.setattr(runtime_ros.time, "monotonic", lambda: adapter._registration_started + 3)
                adapter._heartbeat()
            assert pending in adapter._client.removed
            assert adapter.registration_error == "REGISTRATION_TIMEOUT"
        else:
            adapter._heartbeat()
            assert adapter.registration_error == "REGISTRATION_FAILED"
        assert not adapter.ready
        adapter._client.call_async = send
        for _ in range(4):
            adapter._heartbeat()
        assert adapter.ready and endpoint.core.resource_ready("cpu")
        assert adapter.registration_error == ""
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


@pytest.mark.parametrize("grouped", [False, True])
def test_registration_refresh_waits_for_reset_completion_evidence(monkeypatch, grouped):
    monkeypatch.setattr(runtime_ros, "_service_type", lambda: SimpleNamespace(Request=SimpleNamespace))
    services = {}
    endpoint = GlobalRuntimeEndpoint(
        LocalNode(services),
        [],
        resources={"cpu": {"physical_device": "local/cpu", "mode": "ascend_concurrent", "verified_concurrency": True}},
    )
    adapters, handles = [], []
    try:
        for stage in ("visual", "action") if grouped else ("visual",):
            selected = replace(
                target(stage),
                target_id=f"policy/late/{stage}",
                interface="policy",
                lifecycle_group="policy/late",
                execution_kind="stage" if grouped else "executor",
                resource_mode="ascend_concurrent",
                model_type="pi05",
                operation="policy",
            )
            handle = ModelRuntimeHandle(executor=SimpleNamespace(execute=lambda *_: {}))
            handle.load()
            # Exercise transport grouping with a backend capability double.
            handle.assembly.session = SimpleNamespace(
                capabilities=BackendCapabilities(
                    priority_mapping=BackendPriorityMapping(tuple(range(8))),
                    supports_isolated_stage_execution=True,
                    supports_multiple_instances=True,
                    admission_evidence=BackendAdmissionEvidence(
                        sdk_initialization=True,
                        multi_instance_execution=True,
                        failure_isolation=True,
                        independent_close=True,
                    ),
                )
            )
            handles.append(handle)
            adapters.append(SchedulerRuntimeAdapter(LocalNode(services), selected, handle, registration_enabled=False))
        adapters[0]._heartbeat()
        assert not endpoint.core.targets
        if grouped:
            bind_registration_group(adapters)
        else:
            adapters[0]._registration_enabled = True
        for _ in range(3):
            for adapter in adapters:
                adapter._heartbeat()
        assert endpoint.core.resource_ready("cpu")
        assert len(endpoint.core.targets) == len(adapters)
        assert len({a.controller.boot_id for a in adapters}) == 1
        assert all(a.ready for a in adapters)
        assert all(a.registration_error == "" for a in adapters)
        # A reset initiated through the follower must refresh the whole group,
        # not use the single-target registration forbidden for dynamic stages.
        adapter = adapters[-1]
        adapter._refresh_after_reset(ExecutionContext("reset", Deadline.after(1)))
        controller = adapter.controller
        submission = controller.prepare(ModelRequest(), ExecutionContext("reset-race", Deadline.after(2)), work="reset")
        endpoint.core.submit(submission)
        assert endpoint.core.dispatchable() == [(adapter.target, submission)]
        controller.execute(submission, endpoint.core.epoch)
        controller.completion(submission).result(timeout=1)
        assert controller.query(submission, endpoint.core.epoch)["state"] == "completed"
        rejected = []
        send = adapter._call_control

        def deliver_reset_after_registration(payload, timeout, **kwargs):
            try:
                return send(payload, timeout, **kwargs)
            except ControlError as exc:
                rejected.append(exc.code)
                # The Host advanced its generation before Global received the
                # Reset completion. Deliver that real evidence after rejection.
                endpoint.core.terminal_report(controller.query(submission, endpoint.core.epoch))
                controller.acknowledge(submission, endpoint.core.epoch)
                raise

        monkeypatch.setattr(adapter, "_call_control", deliver_reset_after_registration)
        adapter.refresh_registration(ExecutionContext("refresh-race", Deadline.after(1)), require_ready=True)
        assert rejected == ["RESET_COMPLETION_EVIDENCE_PENDING"]
        assert endpoint.core.resource_ready("cpu")
        assert not endpoint.core._fenced
        assert all(
            endpoint.core._hosts[a.target.target_id][0]["generation"] == a.controller.generation for a in adapters
        )
        monkeypatch.setattr(adapter, "_call_control", send)
        statuses = [a.controller.status() for a in adapters]
        before = dict(endpoint.core._hosts)
        statuses[-1]["target"]["endpoint"] = "/conflicting/replacement"
        with pytest.raises(ControlError):
            endpoint._handle(
                {"verb": "register_group", "statuses": statuses}
                if grouped
                else {"verb": "register", "status": statuses[0]}
            )
        assert endpoint.core._hosts == before
    finally:
        for adapter in adapters:
            _dispose_fake_transport(adapter)
        for handle in handles:
            handle.close()


def test_product_descriptor_rejection_does_not_publish_runtime_target(transport):
    services, node, endpoint = transport

    selected = replace(target(), interface="policy", target_id="policy/late")
    handle = ModelRuntimeHandle(executor=SimpleNamespace(execute=lambda *_: {}))
    handle.load()
    adapter = SchedulerRuntimeAdapter(LocalNode(services), selected, handle)
    try:
        with pytest.raises(ControlError, match="PRODUCT_REGISTRATION_REQUIRES_POLICY_SERVICE"):
            endpoint._handle({"verb": "register", "status": adapter.controller.status(), "pipeline": {}})
        assert not endpoint.core.targets and not node.created
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


def test_invalid_ros_endpoint_cannot_publish_membership_or_break_tick(transport):
    _services, node, endpoint = transport
    invalid = target().to_dict()
    invalid["endpoint"] = "/" + "x" * 247
    with pytest.raises(ValueError, match="invalid runtime endpoint"):
        endpoint._handle({"verb": "register", "status": {"target": invalid}})
    assert not endpoint.core.targets and not node.created
    endpoint._tick()


def test_late_host_registers_once_and_clean_exit_retires_client(transport):
    services, node, endpoint = transport
    assert not endpoint.core.targets and not node.created
    adapter, handle = host(services, target())
    try:
        assert endpoint.core.resource_ready("cpu")
        assert len(node.created) == 1
        adapter._heartbeat()
        assert len(node.created) == 1

        submission = adapter.controller.prepare(ModelRequest(), ExecutionContext("request", Deadline.after(1)))
        endpoint.core.submit(submission)
        endpoint._tick()
        adapter.controller._records[submission.operation_id].done.result(timeout=1)
        # Close flushes terminal evidence even if the coordinator has not yet
        # processed Execute's reply or sent its next Query.
        _dispose_fake_transport(adapter)
        record = endpoint.core._records[submission.operation_id]
        assert record.state == "completed" and record.host_acknowledged
        assert target().target_id in endpoint.core._retired
        assert not node.destroyed
        collect_after = (
            max(submission.deadline_ns, submission.accept_until_ns, endpoint.core._retired[adapter.target.target_id])
            + 1
        )
        endpoint.core.now_ns = lambda: collect_after
        endpoint._tick()
        endpoint._tick()  # Collect the directory after replay records expire.
        assert target().target_id not in endpoint.core.targets
        assert target().target_id not in endpoint._clients
        assert len(node.destroyed) == 1
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


def test_unapproved_resource_does_not_create_transport_client(transport):
    services, node, endpoint = transport
    selected = replace(target(), hardware_resource_id="other", physical_device="local/mps")
    handle = ModelRuntimeHandle(executor=SimpleNamespace(execute=lambda *_: {}))
    handle.load()
    adapter = SchedulerRuntimeAdapter(LocalNode(services), selected, handle)
    try:
        with pytest.raises(ControlError, match="UNKNOWN_RESOURCE"):
            endpoint._handle({"verb": "register", "status": adapter.controller.status()})
        assert not node.created and not endpoint._clients and not endpoint.core.targets
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


@pytest.mark.parametrize("recovery", [False, True])
def test_global_admission_survives_pending_client_collection(monkeypatch, recovery):
    monkeypatch.setattr(runtime_ros, "_service_type", lambda: SimpleNamespace(Request=SimpleNamespace))
    services = {}
    node = LocalNode(services)
    node.executor = object()
    endpoint = GlobalRuntimeEndpoint(
        node, [], resources={"cpu": {"physical_device": "local/cpu"}}, max_registered_targets=1
    )
    for index in range(2):
        client = node.create_client(None, f"/retired/{index}")
        client._executor_event = False
        endpoint._retire_client(client)
    adapter, handle = host(services, replace(target(), retry_window_ms=1000))
    try:
        assert endpoint.core.resource_ready("cpu") and not endpoint._clients
        if recovery:
            operation_id = str(uuid4())
            endpoint.recovery.request(
                {
                    "resource": "cpu",
                    "operation_id": operation_id,
                    "command": "drain",
                    "accept_until_ns": endpoint.core.now_ns() + 1_000_000_000,
                }
            )
            endpoint._tick()
            assert endpoint.recovery.records[operation_id]["state"] == "running"
        else:
            submission = adapter.controller.prepare(
                ModelRequest(), ExecutionContext("pending-client", Deadline.after(1))
            )
            endpoint.core.submit(submission)
            endpoint._tick()
            assert endpoint.core.query(submission)["state"] == "not_started"

        endpoint.collect_retired_clients(SimpleNamespace(_cb_iter=None))
        expires = time.monotonic() + 1
        while time.monotonic() < expires:
            endpoint._tick()
            adapter._heartbeat()
            if recovery:
                if endpoint.recovery.records[operation_id]["state"] == "completed":
                    break
            elif endpoint.core.resource_settled("cpu"):
                break
            time.sleep(0.002)
        assert adapter.target.target_id in endpoint._clients
        assert endpoint.core.resource_settled("cpu")
        if recovery:
            assert endpoint.recovery.records[operation_id]["state"] == "completed"
        assert not endpoint.core._fenced
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


def test_retired_client_survives_pending_transport(transport):
    services, node, endpoint = transport
    adapter, handle = host(services, target())
    try:
        _dispose_fake_transport(adapter)
        submission = SimpleNamespace(target_id=target().target_id)
        pending = Future()
        endpoint._pending[("old", "query")] = (pending, time.monotonic_ns(), target(), submission)
        collect_after = endpoint.core._retired[adapter.target.target_id] + 1
        endpoint.core.now_ns = lambda: collect_after
        endpoint.core.collect()
        assert target().target_id not in endpoint.core.targets
        endpoint._sync_clients()
        assert not node.destroyed
        endpoint._pending.clear()
        endpoint._sync_clients()
        assert len(node.destroyed) == 1
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


def test_reused_collected_target_rebinds_endpoint_and_discards_old_transport(transport):
    services, node, endpoint = transport
    adapter, handle = host(services, target())
    replacement = None
    try:
        _dispose_fake_transport(adapter)
        old_client = endpoint._clients[target().target_id]
        pending = Future()
        submission = SimpleNamespace(operation_id="expired", target_id=target().target_id)
        endpoint._pending[("expired", "query")] = (pending, time.monotonic_ns(), target(), submission)
        old_now = endpoint.core.now_ns
        collect_after = endpoint.core._retired[adapter.target.target_id] + 1
        endpoint.core.now_ns = lambda: collect_after
        endpoint.core.collect()
        endpoint.core.now_ns = old_now
        assert target().target_id not in endpoint.core.targets
        replacement, replacement_handle = host(services, replace(target(), endpoint="/runtime/replacement"))
        assert endpoint._clients[target().target_id].endpoint == "/runtime/replacement"
        assert old_client.removed == [pending]
        assert old_client in node.destroyed
        assert not endpoint._pending
    finally:
        if replacement is not None:
            _dispose_fake_transport(replacement)
            replacement_handle.close()
        _dispose_fake_transport(adapter)
        handle.close()


def test_unknown_host_cannot_unregister_or_release_shared_resource(transport):
    services, node, endpoint = transport
    adapter, handle = host(services, target())
    try:
        submission = adapter.controller.prepare(ModelRequest(), ExecutionContext("unknown", Deadline.after(1)))
        endpoint.core.submit(submission)
        endpoint.core.dispatchable()
        endpoint.core.mark_transport_unknown(submission)
        # Simulate backend uncertainty retained at both peers. Ordinary close
        # must not convert it into terminal evidence or unregister membership.
        adapter.controller._records[submission.operation_id].state = "unknown"
        _dispose_fake_transport(adapter)
        assert submission.target_id not in endpoint.core._retired
        assert endpoint.core.query(submission)["state"] == "unknown"
        assert not endpoint.core.resource_ready("cpu")
        assert submission.target_id in endpoint._clients and not node.destroyed
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


def test_close_sends_unregister_even_when_executor_cannot_deliver_receipts(transport):
    services, _node, endpoint = transport
    adapter, handle = host(services, target())
    try:
        submission = adapter.controller.prepare(ModelRequest(), ExecutionContext("exit", Deadline.after(1)))
        endpoint.core.submit(submission)
        endpoint.core.dispatchable()
        adapter.controller.execute(submission, adapter.controller.epoch)
        adapter.controller._records[submission.operation_id].done.result(timeout=1)
        delivered = []
        call_async = adapter._client.call_async

        def without_reply(request):
            delivered.append(decode_control(request.payload)["verb"])
            call_async(request)  # Global receives the request; host cannot spin its reply.
            return Future()

        adapter._client.call_async = without_reply
        started = time.monotonic()
        _dispose_fake_transport(adapter)
        assert time.monotonic() - started < 0.6
        assert delivered == ["unregister", "terminal_report", "unregister"]
        assert endpoint.core._records[submission.operation_id].host_acknowledged
        assert adapter.target.target_id in endpoint.core._retired
        assert endpoint.core.resource_ready("cpu")
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


def test_dynamic_recovery_excludes_never_started_model_hosts(transport):
    services, _node, endpoint = transport
    adapter, handle = host(services, target())
    try:
        never_started = target("never_started")
        endpoint.core.targets[never_started.target_id] = never_started
        recovery = ResourceRecovery(endpoint)
        operation_id = str(uuid4())
        recovery.request(
            {
                "resource": "cpu",
                "operation_id": operation_id,
                "command": "drain",
                "accept_until_ns": endpoint.core.now_ns() + 50_000_000,
            }
        )
        assert recovery.records[operation_id]["targets"] == (adapter.target,)
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


def test_completed_recovery_receipt_survives_membership_retirement(transport):
    services, _node, endpoint = transport
    adapter, handle = host(services, target())
    try:
        recovery = ResourceRecovery(endpoint)
        request = {
            "resource": "cpu",
            "operation_id": str(uuid4()),
            "command": "drain",
            "accept_until_ns": endpoint.core.now_ns() + 50_000_000,
        }
        recovery.request(request)
        recovery.records[request["operation_id"]].update(state="completed", phase="activate")
        endpoint.core._fenced.clear()
        _dispose_fake_transport(adapter)
        assert endpoint.core._inactive(adapter.target.target_id)
        assert recovery.request(request) == {"state": "completed", "phase": "activate", "error": ""}
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


@pytest.mark.parametrize("lost_close_reply", [None, "completed", "acknowledged"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_unknown_close_drains_once_retires_owner_and_restores_other_host(transport, lost_close_reply, dynamic):
    services, node, endpoint = transport
    selected = [replace(target(name), retry_window_ms=3000) for name in ("closing", "survivor")]
    if not dynamic:
        endpoint = GlobalRuntimeEndpoint(node, selected)
    drained = threading.Event()
    permit_drain = threading.Event()

    class Model:
        closes = 0
        drains = 0

        def execute(self, request, context):
            if request.inputs.get("unknown"):

                class Unknown(RuntimeError):
                    operation_started = True
                    outcome_known = False

                raise Unknown("missing completion evidence")
            return {"answer": 7}

        def drain_uncertain_operations(self):
            self.drains += 1
            drained.set()
            assert permit_drain.wait(2)

        def close(self):
            self.closes += 1

    model, peer_model = Model(), Model()
    handles = [ModelRuntimeHandle(executor=item, resettable=True) for item in (model, peer_model)]
    bindings = []
    for host_target, handle in zip(selected, handles, strict=True):
        handle.load()
        binding = bind_runtime(LocalNode(services), handle, host_target)
        bindings.append(binding)
        for _ in range(3):
            binding.adapter._heartbeat()

    def pump_once():
        endpoint._tick()
        for binding in bindings:
            if not binding._closed:
                binding.adapter._heartbeat()

    original = bindings[0].adapter._call_control
    with _runtime_pump(pump_once):
        try:
            with pytest.raises(ExecutionFailure):
                handles[0].execute(ModelRequest({"unknown": True}), ExecutionContext("uncertain", Deadline.after(1)))
            assert any(record.state == "unknown" for record in bindings[0].controller._records.values())
            # First caller stops waiting while drain remains active. Its immutable
            # recovery operation must remain available to the next Close caller.
            with pytest.raises(ExecutionFailure):
                handles[0].close(ExecutionContext("short-close", Deadline.after(0.03)))
            assert drained.wait(1)
            assert not bindings[0]._closed and bindings[0].target.endpoint in services
            recovery = dict(bindings[0].adapter._recovery_requests["close"])
            if lost_close_reply:

                def lose_completed(payload, timeout):
                    response = original(payload, timeout)
                    if payload.get("verb") == "recover" and response.get("state") == lost_close_reply:
                        raise TimeoutError("lost final close receipt")
                    return response

                bindings[0].adapter._call_control = lose_completed
            permit_drain.set()
            if lost_close_reply:
                with pytest.raises(ExecutionFailure):
                    handles[0].close(ExecutionContext("lost-close", Deadline.after(0.1)))
                assert not bindings[0]._closed
                bindings[0].adapter._call_control = original
            handles[0].close(ExecutionContext("retry-close", Deadline.after(2)))
            assert bindings[0]._closed and handles[0].closed
            assert model.closes == 1 and model.drains == 1
            assert "close" not in bindings[0].adapter._recovery_requests
            assert endpoint.recovery.records[recovery["operation_id"]]["acknowledged"]
            assert len(endpoint.recovery.records) == 1
            expiry = time.monotonic() + 1
            while not endpoint.core.resource_ready("cpu"):
                assert time.monotonic() < expiry
                time.sleep(0.002)
            result = handles[1].execute(ModelRequest(), ExecutionContext("peer-still-active", Deadline.after(1)))
            assert result.outputs == {"answer": 7}
        finally:
            permit_drain.set()
            bindings[0].adapter._call_control = original
            for binding, handle in zip(bindings, handles, strict=True):
                if not binding._closed:
                    handle.close(ExecutionContext("cleanup", Deadline.after(2)))


@pytest.mark.parametrize("executor_stopped", [False, True])
@pytest.mark.parametrize("registered", [False, True])
def test_close_admission_timeout_waits_for_other_lifecycle_on_shared_device(transport, executor_stopped, registered):
    from concurrent.futures import ThreadPoolExecutor

    services, _node, endpoint = transport
    (entered, release) = (threading.Event(), threading.Event())
    closes = []

    def execute(request, context):
        entered.set()
        assert release.wait(5)
        return {"answer": 1}

    handles = [
        ModelRuntimeHandle(executor=SimpleNamespace(execute=execute)),
        ModelRuntimeHandle(
            executor=SimpleNamespace(execute=lambda *_: {}, close=lambda: closes.append(release.is_set()))
        ),
    ]
    bindings = []
    for name, handle in zip(("busy", "closing"), handles, strict=True):
        handle.load()
        binding = bind_runtime(LocalNode(services), handle, replace(target(name), retry_window_ms=3000))
        bindings.append(binding)
        if name != "closing" or registered:
            for _ in range(3):
                binding.adapter._heartbeat()
    if executor_stopped:
        bindings[1].adapter.node.executor = SimpleNamespace(is_spinning=False)

    def pump_once():
        endpoint._tick()
        for binding in bindings:
            if not binding._closed and (binding is bindings[0] or registered or binding.controller.epoch):
                binding.adapter._heartbeat()

    with _runtime_pump(pump_once):
        try:
            with ThreadPoolExecutor(1) as pool:
                running = pool.submit(handles[0].execute, ModelRequest(), ExecutionContext("busy", Deadline.after(5)))
                try:
                    assert entered.wait(1)
                    assert bool(bindings[1].controller.epoch) is registered
                    with pytest.raises(ExecutionFailure):
                        handles[1].close(ExecutionContext("short-close", Deadline.after(0.03)))
                    assert not closes
                    assert not handles[1].closed and not bindings[1]._closed
                    assert not handles[1].ready
                    assert bindings[1].target.endpoint in services
                finally:
                    release.set()
                running.result(timeout=2)
            handles[1].close(ExecutionContext("resume-close", Deadline.after(2)))
            assert closes == [True]
            assert handles[1].closed and bindings[1]._closed
        finally:
            release.set()
            for binding, handle in zip(bindings, handles, strict=True):
                if not binding._closed:
                    handle.close(ExecutionContext("cleanup", Deadline.after(2)))


def test_unregistered_close_retains_loaded_model_when_global_is_unavailable(transport):
    services, _node, _endpoint = transport
    closes = []
    handle = ModelRuntimeHandle(executor=SimpleNamespace(execute=lambda *_: {}, close=lambda: closes.append(True)))
    handle.load()
    binding = bind_runtime(LocalNode(services), handle, target("unregistered_close"))
    services.pop("/inference/runtime/v1")
    try:
        with pytest.raises(ExecutionFailure):
            handle.close(ExecutionContext("close-before-heartbeat", Deadline.after(0.1)))
        assert not closes
        assert not handle.closed and not binding._closed
        assert binding.target.endpoint in services
    finally:
        _dispose_fake_transport(binding.adapter)
        binding.local._close_local(ExecutionContext("fixture-cleanup"), _dispatch_key=binding.controller._authority)


@pytest.mark.parametrize("lost_retirement_reply", [False, True])
def test_healthy_static_close_retires_after_ack_and_peer_remains_usable(transport, lost_retirement_reply):
    services, node, _endpoint = transport
    targets = [replace(target(name), retry_window_ms=3000) for name in ("static_close", "static_peer")]
    endpoint = GlobalRuntimeEndpoint(node, targets)
    closes = []
    handles = [
        ModelRuntimeHandle(
            executor=SimpleNamespace(execute=lambda *_: {"answer": 11}, close=lambda: closes.append(True))
        )
        for _ in targets
    ]
    bindings = []
    for handle, selected in zip(handles, targets, strict=True):
        handle.load()
        binding = bind_runtime(LocalNode(services), handle, selected)
        bindings.append(binding)
        for _ in range(3):
            binding.adapter._heartbeat()

    def pump_once():
        endpoint._tick()
        for binding in bindings:
            if not binding._closed:
                binding.adapter._heartbeat()

    with _runtime_pump(pump_once):
        try:
            if lost_retirement_reply:
                adapter = bindings[0].adapter
                original = adapter._call_control

                def lose_retirement(payload, timeout, **kwargs):
                    reply = original(payload, timeout, **kwargs)
                    if payload["verb"] == "unregister":
                        raise TimeoutError("lost retirement receipt")
                    return reply

                adapter._call_control = lose_retirement
                try:
                    with pytest.raises(ExecutionFailure):
                        handles[0].close(ExecutionContext("lost-retirement", Deadline.after(0.1)))
                    assert handles[0].closed and not bindings[0]._closed
                    assert targets[0].endpoint in services
                finally:
                    adapter._call_control = original
            handles[0].close(ExecutionContext("static-close", Deadline.after(1)))
            assert closes == [True]
            assert targets[0].target_id in endpoint.core._retired
            assert endpoint.core.resource_ready("cpu")
            assert all(record.host_acknowledged for record in endpoint.core._records.values())
            assert not endpoint.recovery.records
            result = handles[1].execute(ModelRequest(), ExecutionContext("still-open", Deadline.after(1)))
            assert result.outputs == {"answer": 11}
        finally:
            for handle in handles:
                handle.close(ExecutionContext("cleanup", Deadline.after(1)))


@pytest.mark.parametrize("first_in_recovery", [False, True])
def test_concurrent_close_recovery_observes_peer_retirement(transport, first_in_recovery):
    from concurrent.futures import ThreadPoolExecutor

    services, node, _endpoint = transport
    targets = [replace(target(name), retry_window_ms=3000) for name in ("first", "second", "survivor")]
    endpoint = GlobalRuntimeEndpoint(node, targets)
    (entered, release) = (threading.Event(), threading.Event())
    closes = []

    def close_first():
        entered.set()
        assert release.wait(5)
        closes.append("first")

    def execute_first(request, context):
        if request.inputs.get("unknown"):

            class Unknown(RuntimeError):
                operation_started = True
                outcome_known = False

            raise Unknown("lost device completion")
        return {}

    handles = [
        ModelRuntimeHandle(
            executor=SimpleNamespace(execute=execute_first, close=close_first, drain_uncertain_operations=lambda: None)
        ),
        ModelRuntimeHandle(executor=SimpleNamespace(execute=lambda *_: {}, close=lambda: closes.append("second"))),
        ModelRuntimeHandle(executor=SimpleNamespace(execute=lambda *_: {"answer": 11})),
    ]
    bindings = []
    for handle, selected in zip(handles, targets, strict=True):
        handle.load()
        binding = bind_runtime(LocalNode(services), handle, selected)
        bindings.append(binding)
        for _ in range(3):
            binding.adapter._heartbeat()

    drain_sent = threading.Event()
    lost_drain = Future()
    first_client = endpoint._clients[targets[0].target_id]
    send = first_client.call_async

    def lose_peer_drain_reply(request):
        response = send(request)
        message = decode_control(request.payload)
        if message.get("verb") == "recover" and message.get("command") == "drain":
            drain_sent.set()
            return lost_drain
        return response

    if not first_in_recovery:
        first_client.call_async = lose_peer_drain_reply

    def pump_once():
        endpoint._tick()
        for binding in bindings:
            if not binding._closed:
                binding.adapter._heartbeat()

    try:
        with _runtime_pump(pump_once):
            try:
                if first_in_recovery:
                    with pytest.raises(ExecutionFailure):
                        handles[0].execute(
                            ModelRequest({"unknown": True}), ExecutionContext("unknown", Deadline.after(1))
                        )
                with ThreadPoolExecutor(2) as pool:
                    first_context = ExecutionContext("first-close", Deadline.after(3))
                    first = pool.submit(handles[0].close, first_context)
                    assert entered.wait(1)
                    # Publish the first owner's irreversible Close before admitting
                    # the second. Recovery must capture that still-running peer.
                    bindings[0].adapter._heartbeat()
                    bindings[0].adapter._heartbeat()
                    if first_in_recovery:
                        adapter = bindings[1].adapter
                        original = adapter._call_control
                        lost_busy = []

                        def lose_first_busy(payload, timeout):
                            try:
                                return original(payload, timeout)
                            except ControlError as exc:
                                if exc.code == "RECOVERY_BUSY" and not lost_busy:
                                    lost_busy.append(True)
                                    raise runtime_ros.TimeoutError("lost rejection") from exc
                                raise

                        adapter._call_control = lose_first_busy
                        try:
                            with pytest.raises(ExecutionFailure) as failure:
                                handles[1].close(ExecutionContext("busy-recovery", Deadline.after(0.05)))
                        finally:
                            adapter._call_control = original
                        assert lost_busy
                        assert not bindings[1].adapter.has_recovery("close"), (
                            str(failure.value),
                            adapter._recovery_requests,
                        )
                        assert not handles[1].closed
                        assert targets[1].endpoint in services
                    second = pool.submit(handles[1].close, ExecutionContext("second-close", Deadline.after(3)))
                    expires = time.monotonic() + 1
                    while not endpoint.recovery.records and time.monotonic() < expires:
                        time.sleep(0.002)
                    assert endpoint.recovery.records
                    recovery = next(iter(endpoint.recovery.records.values()))
                    assert targets[0] in recovery["targets"]
                    if not first_in_recovery:
                        assert drain_sent.wait(1)
                    release.set()
                    first_error = first.exception(timeout=4)
                    assert first_error is None, (str(first_error), first_context.cancellation_token.reason)
                    second.result(timeout=4)
                assert closes == ["first", "second"]
                assert all(binding._closed for binding in bindings[:2])
                assert all(selected.target_id in endpoint.core._retired for selected in targets[:2])
                assert endpoint.core.resource_ready("cpu")
                assert recovery["state"] == "completed"
                if not first_in_recovery:
                    assert lost_drain in first_client.removed
                result = handles[2].execute(ModelRequest(), ExecutionContext("surviving-peer", Deadline.after(1)))
                assert result.outputs == {"answer": 11}
            finally:
                release.set()
                # Failed regressions must still release local executor threads without
                # requiring the broken Global recovery to finish.
    finally:
        for binding in bindings:
            _dispose_fake_transport(binding.adapter)
            binding.local._close_local(ExecutionContext("fixture-cleanup"), _dispatch_key=binding.controller._authority)


def test_close_replays_terminal_receipt_when_host_ack_overtakes_global_ack(transport):
    services, node, _endpoint = transport
    selected = replace(target("ack_race"), retry_window_ms=3000)
    endpoint = GlobalRuntimeEndpoint(node, [selected])
    adapter, handle = host(services, selected)
    try:
        submission = adapter.controller.prepare(ModelRequest(), ExecutionContext("completed", Deadline.after(1)))
        endpoint.core.submit(submission)
        endpoint.core.dispatchable()
        local = adapter.controller._records[submission.operation_id]
        with adapter.controller.transaction():
            adapter.controller._finish(local, "completed", result={})
        endpoint.core.report(local.report())
        # Global's Ack request reached the host, but its reply is still in
        # transport. Host acknowledgment does not prove Global owns the ACK.
        adapter.controller.acknowledge(submission, adapter.controller.epoch)
        assert local.acknowledged
        assert not endpoint.core._records[submission.operation_id].host_acknowledged
        _dispose_fake_transport(adapter)
        assert selected.target_id in endpoint.core._retired
        assert endpoint.core._records[submission.operation_id].host_acknowledged
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


@pytest.mark.parametrize("unknown_peer", [False, True])
def test_coordinator_loss_never_authorizes_local_device_close(transport, unknown_peer):
    services, _node, _endpoint = transport
    handle = ModelRuntimeHandle(executor=SimpleNamespace(execute=lambda *_: {}))
    handle.load()
    binding = bind_runtime(LocalNode(services), handle, target("offline_close"))
    for _ in range(3):
        binding.adapter._heartbeat()
    controller = binding.controller
    peer_handle = None
    uncertain = controller
    if unknown_peer:
        peer_handle = ModelRuntimeHandle(executor=SimpleNamespace(execute=lambda *_: {}))
        peer_handle.load()
        uncertain = RuntimeExecutionController(target("action_peer"), peer_handle)
        uncertain.join_lifecycle(controller, [controller, uncertain])
        uncertain.accept_epoch(controller.epoch)
        controller.lifecycle_peers = uncertain.lifecycle_peers = [controller, uncertain]
    operation = uncertain.prepare(ModelRequest(), ExecutionContext("pending", Deadline.after(1)))
    uncertain._records[operation.operation_id].state = "unknown"
    services.pop("/inference/runtime/v1")
    try:
        with pytest.raises(ExecutionFailure):
            handle.close(ExecutionContext("unsafe-close", Deadline.after(0.02)))
        assert not handle.closed and not binding._closed
        assert binding.target.endpoint in services
        with uncertain.transaction():
            uncertain._finish(uncertain._records[operation.operation_id], "completed")
        # Local completion cannot prove that other lifecycle groups stopped.
        with pytest.raises(ExecutionFailure):
            handle.close(ExecutionContext("local-only-evidence", Deadline.after(0.02)))
        assert not handle.closed and not binding._closed
    finally:
        _dispose_fake_transport(binding.adapter)
        if peer_handle is not None:
            uncertain.shutdown()
            peer_handle.close()
        binding.local._close_local(ExecutionContext("fixture-cleanup"), _dispatch_key=binding.controller._authority)


@pytest.mark.parametrize("lost_first_reply", [False, True])
def test_recovery_first_admission_expiry_does_not_poison_next_reset(transport, monkeypatch, lost_first_reply):
    _, node, endpoint = transport
    calls = []
    handle = ModelRuntimeHandle(
        SimpleNamespace(execute=lambda *_: {}, reset=lambda *_: calls.append("reset")), resettable=True
    )
    handle.load()
    binding = bind_runtime(node, handle, replace(target(), retry_window_ms=100))
    adapter = binding.adapter
    for _ in range(3):
        adapter._heartbeat()
    original_call = adapter._call_control
    identities = []

    def delayed_call(message, timeout, **kwargs):
        if message.get("verb") == "recover":
            identities.append(message["operation_id"])
            if len(identities) == 1:
                if lost_first_reply:
                    # The first result is unknown. Even an expired response to
                    # a subsequent send must not authorize a different Reset.
                    raise runtime_ros.TimeoutError("reply lost")
                time.sleep(0.12)
            elif lost_first_reply:
                from inference_service.scheduler.runtime_ros import _RemoteControlError

                raise _RemoteControlError("OPERATION_EXPIRED")
        return original_call(message, timeout, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(adapter, "_call_control", delayed_call)
            with pytest.raises(ControlError, match="OPERATION_EXPIRED"):
                adapter.recover(ExecutionContext("first", Deadline.after(1)))
        assert not calls and not endpoint.recovery.records
        if lost_first_reply:
            assert adapter.has_recovery("reset")
            assert adapter._recovery_requests["reset"]["operation_id"] == identities[0]
        else:
            assert not adapter.has_recovery("reset")
            with _runtime_pump(endpoint._tick, adapter._heartbeat):
                adapter.recover(ExecutionContext("retry", Deadline.after(2)))
            assert calls == ["reset"]
            assert not adapter.has_recovery("reset")
            assert next(iter(endpoint.recovery.records)) != identities[0]
    finally:
        _dispose_fake_transport(adapter)
        handle._close_local(_dispatch_key=binding.local.owner_authority)


@pytest.mark.parametrize("command", ["reset", "close"])
@pytest.mark.parametrize("late_expired_reply", [False, True])
def test_recovery_resumes_expired_window_without_repeating_lifecycle_call(transport, command, late_expired_reply):
    services, node, endpoint = transport
    (entered, release) = (threading.Event(), threading.Event())

    class Model:
        def __init__(self):
            self.calls = []

        def execute(self, request, context):
            return {"answer": 9}

        def reset(self, context):
            self.calls.append("reset")
            entered.set()
            assert release.wait(2)

        def close(self):
            self.calls.append("close")
            entered.set()
            assert release.wait(2)

    model = Model()
    handle = ModelRuntimeHandle(executor=model, resettable=True)
    handle.load()
    binding = bind_runtime(node, handle, replace(target(), retry_window_ms=200))
    for _ in range(3):
        binding.adapter._heartbeat()

    def pump_once():
        endpoint._tick()
        binding.adapter._heartbeat()

    try:
        with _runtime_pump(pump_once):
            try:
                with pytest.raises(ControlError, match="RECOVERY_EXPIRED|OPERATION_EXPIRED"):
                    binding.adapter.recover(ExecutionContext("first", Deadline.after(1)), command=command)
                assert entered.is_set() and model.calls == [command]
                assert not endpoint.core.resource_ready("cpu")
                if late_expired_reply:
                    # The caller can expire before Global's next tick discards the old
                    # command future. A response from that window must not poison its
                    # renewal, even if DDS delivers it after renewal was requested.
                    with endpoint.core._lock:
                        record = next(iter(endpoint.recovery.records.values()))
                        old_reply = Future()
                        old_reply.set_result(
                            SimpleNamespace(payload=encode_control({"control_error": "OPERATION_EXPIRED"}))
                        )
                        for pending in record["commands"].values():
                            pending["future"] = old_reply
                release.set()
                binding.adapter.recover(ExecutionContext("retry", Deadline.after(1)), command=command)
                assert model.calls == [command]
                assert len(endpoint.recovery.records) == 1
                assert endpoint.core.resource_ready("cpu")
                if command == "reset":
                    assert handle.execute(ModelRequest(), ExecutionContext("next", Deadline.after(1))).outputs == {
                        "answer": 9
                    }
                else:
                    assert handle.closed
            finally:
                release.set()
    finally:
        _dispose_fake_transport(binding.adapter)
        # Adapter transport teardown does not own the model lifecycle.
        handle._close_local(_dispatch_key=binding.local.owner_authority)


def test_recovery_renewal_without_receipt_never_repeats_a_mutation(transport):
    services, _node, endpoint = transport
    adapter, handle = host(services, target())
    request = {
        "resource": "cpu",
        "operation_id": str(uuid4()),
        "command": "reset",
        "accept_until_ns": time.monotonic_ns() - 1,
        "resume_until_ns": time.monotonic_ns() + 50_000_000,
    }
    try:
        with pytest.raises(ControlError, match="RECOVERY_RECEIPT_UNAVAILABLE"):
            endpoint.recovery.request(request)
        with pytest.raises(ControlError, match="RECOVERY_RECEIPT_UNAVAILABLE"):
            adapter.controller.recover(
                adapter.controller.epoch,
                request["operation_id"],
                "reset",
                request["accept_until_ns"],
                resume_until_ns=request["resume_until_ns"],
            )
        assert not endpoint.recovery.records
        assert handle.ready
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


def test_recovery_can_resume_drain_after_its_execution_deadline(transport, monkeypatch):
    from inference_service.unified_runtime.errors import DeadlineExceeded

    services, _node, _endpoint = transport
    adapter, handle = host(services, target())
    controller = adapter.controller
    calls = []

    def drain(command, expected_generation, stream_id=None):
        assert command == "drain"
        calls.append(command)
        if len(calls) == 1:
            raise DeadlineExceeded(phase="drain")

    monkeypatch.setattr(controller, "_recover", drain)
    operation_id = str(uuid4())
    expiry = time.monotonic_ns() + 50_000_000
    try:
        controller.recover(controller.epoch, operation_id, "drain", expiry)
        controller._control_future.result(timeout=1)
        with pytest.raises(ExecutionFailure, match="[Dd]eadline"):
            controller.recover(controller.epoch, operation_id, "drain", expiry)
        controller.recover(
            controller.epoch,
            operation_id,
            "drain",
            expiry,
            resume_until_ns=time.monotonic_ns() + 100_000_000,
        )
        controller._control_future.result(timeout=1)
        assert calls == ["drain", "drain"]
        assert controller.status()["ready"] is False
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


def test_registration_uses_global_freshness_limit_and_expires_when_global_disappears(transport, monkeypatch):
    services, node, endpoint = transport
    endpoint.core.stale_ns = 7_000_000_000
    adapter, handle = host(services, target())
    try:
        registered_at = adapter._last_registered
        monkeypatch.setattr(runtime_ros.time, "monotonic", lambda: registered_at + 3)
        assert adapter.ready
        monkeypatch.setattr(runtime_ros.time, "monotonic", lambda: registered_at + 8)
        assert not adapter.ready
    finally:
        monkeypatch.undo()
        _dispose_fake_transport(adapter)
        handle.close()


def test_late_host_cannot_claim_global_control_endpoint(transport):
    services, node, endpoint = transport
    selected = replace(target(), endpoint=endpoint.endpoint)
    report = {
        "verb": "register",
        "status": {"target": selected.to_dict()},
    }
    with pytest.raises(ControlError, match="TARGET_ENDPOINT_CONFLICT"):
        endpoint._handle(report)
    assert not endpoint.core.targets


def test_host_readiness_tracks_missing_required_peer_and_resource_fence():
    services = {}
    first = replace(target("first"), required=True)
    second = replace(target("second"), required=True)
    endpoint = GlobalRuntimeEndpoint(
        LocalNode(services),
        [first, second],
        resources={"cpu": {"physical_device": "local/cpu", "mode": "serial"}},
    )
    adapter, handle = host(services, first)
    peer_adapter = peer_handle = None
    try:
        # Successful registration of one locally ready host is insufficient.
        assert adapter.controller.ready
        assert not adapter.ready
        peer_adapter, peer_handle = host(services, second)
        adapter._heartbeat()
        adapter._heartbeat()
        assert adapter.ready
        # The host keeps advertising local health, so a shared fence can clear.
        peer_adapter.controller.fence()
        peer_adapter._heartbeat()
        peer_adapter._heartbeat()
        adapter._heartbeat()
        adapter._heartbeat()
        assert adapter.controller.ready
        assert not endpoint.core.resource_ready("cpu")
        assert not adapter.ready
    finally:
        if peer_adapter is not None:
            _dispose_fake_transport(peer_adapter)
            peer_handle.close()
        _dispose_fake_transport(adapter)
        handle.close()


@pytest.mark.parametrize("retryable", [True, False])
def test_recovery_close_resumes_only_explicit_unfinished_cleanup(transport, monkeypatch, retryable):
    services, _, _ = transport
    adapter, handle = host(services, target())
    controller = adapter.controller
    calls = []

    def cleanup(command, expected_generation, stream_id=None):
        calls.append(command)
        if len(calls) == 1:
            raise ExecutionFailure("cleanup_failed", "release failed", details={"close_pending": retryable})

    monkeypatch.setattr(controller, "_recover", cleanup)
    operation = str(uuid4())
    expiry = time.monotonic_ns() + 100_000_000
    try:
        controller.recover(controller.epoch, operation, "close", expiry)
        controller._control_future.result(timeout=1)
        with pytest.raises(ExecutionFailure):
            controller.recover(controller.epoch, operation, "close", expiry)
        if retryable:
            controller.recover(
                controller.epoch, operation, "close", expiry, resume_until_ns=time.monotonic_ns() + 90_000_000
            )
            controller._control_future.result(timeout=1)
            assert calls == ["close", "close"]
        else:
            with pytest.raises(ExecutionFailure):
                controller.recover(
                    controller.epoch, operation, "close", expiry, resume_until_ns=time.monotonic_ns() + 90_000_000
                )
            assert calls == ["close"]
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


def test_adapter_constructor_releases_client_if_service_creation_fails(transport, monkeypatch):
    services, _, _ = transport
    node = LocalNode(services)
    handle = ModelRuntimeHandle(executor=SimpleNamespace(execute=lambda *_: {}))
    handle.load()

    def fail(*args, **kwargs):
        raise RuntimeError("service creation failed")

    monkeypatch.setattr(node, "create_service", fail)
    try:
        with pytest.raises(RuntimeError, match="service creation failed"):
            bind_runtime(node, handle, target())
        assert node.created == node.destroyed
        assert target().endpoint not in services
    finally:
        handle._close_local(_dispatch_key=handle._policy_authority)


@pytest.mark.parametrize("failure", ["lookup", "ready", "send"])
def test_recovery_transport_failure_keeps_resource_fenced_and_global_running(transport, monkeypatch, failure):
    services, _node, endpoint = transport
    adapter, handle = host(services, target("recovery_transport"))
    operation = str(uuid4())
    try:
        endpoint.recovery.request(
            {
                "resource": "cpu",
                "operation_id": operation,
                "command": "drain",
                "accept_until_ns": endpoint.core.now_ns() + 50_000_000,
            }
        )

        def fail(*_):
            raise RuntimeError("injected transport failure")

        if failure == "lookup":
            endpoint._clients.pop(adapter.target.target_id)
        else:
            monkeypatch.setattr(
                endpoint._clients[adapter.target.target_id],
                "service_is_ready" if failure == "ready" else "call_async",
                fail,
            )
        endpoint._tick()
        record = endpoint.recovery.records[operation]
        assert "cpu" in endpoint.core._fenced
        if failure == "lookup":
            # No client means no command was sent. Wait for the normal sync
            # boundary to rebuild it; do not turn non-start into UNKNOWN.
            assert record["state"] == "running" and not record["commands"]
            assert adapter.target.target_id in endpoint._clients
        else:
            assert record["state"] == "unknown"
        endpoint._tick()
        assert endpoint.core.shutdown_pending() == [adapter.target.target_id]
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


@pytest.mark.parametrize("failure", ["ready", "send"])
def test_terminal_receipt_send_error_can_retry_on_the_shared_control_thread(transport, monkeypatch, failure):
    services, _node, endpoint = transport
    adapter, handle = host(services, target("receipt_transport"))
    try:
        submission = adapter.controller.prepare(ModelRequest(), ExecutionContext("terminal", Deadline.after(1)))
        endpoint.core.submit(submission)
        record = adapter.controller._records[submission.operation_id]
        with adapter.controller.transaction():
            adapter.controller._finish(record, "not_started")
        method = "service_is_ready" if failure == "ready" else "call_async"
        original = getattr(adapter._client, method)
        monkeypatch.setattr(adapter._client, method, lambda *_: (_ for _ in ()).throw(RuntimeError("send failed")))
        adapter._flush_terminal_evidence()
        assert adapter.controller.pending_terminal_reports()
        monkeypatch.setattr(adapter._client, method, original)
        adapter._flush_terminal_evidence()
        adapter._flush_terminal_evidence()
        assert not adapter.controller.pending_terminal_reports()
        assert endpoint.core._records[submission.operation_id].host_acknowledged
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


def test_clean_unregister_skips_history_replay(transport, monkeypatch):
    services, _node, endpoint = transport
    adapter, handle = host(services, target("clean_exit"))
    try:
        handle._close_local(_dispatch_key=adapter.controller._authority)
        monkeypatch.setattr(
            adapter.controller, "terminal_reports", lambda: pytest.fail("settled exit enumerated history")
        )
        assert adapter._unregister_clean_host()
        assert not endpoint.core.shutdown_pending()
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


def test_bounded_retirement_replay_advances_across_attempts(transport, monkeypatch):
    services, _node, _endpoint = transport
    adapter, handle = host(services, target("replay_progress"))
    submissions = []
    try:
        for index in range(8):
            submission = adapter.controller.prepare(ModelRequest(), ExecutionContext(str(index), Deadline.after(2)))
            submissions.append(submission)
            with adapter.controller.transaction():
                adapter.controller._finish(adapter.controller._records[submission.operation_id], "not_started")
        seen = []
        original = adapter._call_control

        def slow_receipt(payload, timeout):
            if payload["verb"] == "unregister":
                raise ControlError("HOST_NOT_DRAINED")
            seen.append(payload["report"]["submission"]["operation_id"])
            time.sleep(timeout)
            raise TimeoutError("lost response")

        monkeypatch.setattr(adapter, "_call_control", slow_receipt)
        for _ in range(8):
            assert not adapter._unregister_clean_host(timeout=0.12)
        assert set(seen) == {submission.operation_id for submission in submissions}
        monkeypatch.setattr(adapter, "_call_control", original)
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


def test_dynamic_registration_reserves_global_readiness_endpoint(transport):
    _services, _node, endpoint = transport
    invalid = replace(target(), endpoint=endpoint.endpoint + "/ready")
    with pytest.raises(ControlError, match="TARGET_ENDPOINT_CONFLICT"):
        endpoint._handle({"verb": "register", "status": {"target": invalid.to_dict()}})
    assert not endpoint.core.targets


def test_recovery_close_accepts_release_evidence_with_cleanup_diagnostics(transport):
    from inference_service.backends.errors import BackendLifecycleError

    services, _node, endpoint = transport
    calls = []

    def close():
        calls.append("close")
        raise BackendLifecycleError("released with a diagnostic")

    handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}, close=close))
    handle.load()
    binding = bind_runtime(LocalNode(services), handle, target("close_diagnostic"))
    for _ in range(3):
        binding.adapter._heartbeat()
    endpoint.core.fence_resource("cpu")

    def pump_once():
        endpoint._tick()

    try:
        with _runtime_pump(pump_once):
            handle.close(deadline=Deadline.after(2))
            assert handle.closed and binding._closed and calls == ["close"]
            assert not endpoint.core._fenced and not endpoint.core.shutdown_pending()
            assert handle.last_failure.code == "close_failed"
    finally:
        _dispose_fake_transport(binding.adapter)


@pytest.mark.parametrize("peer_recovery", [False, True])
def test_ordinary_close_diagnostic_retires_before_peer_close(transport, peer_recovery):
    from inference_service.backends.errors import BackendLifecycleError

    services, node, _ = transport
    selected = [replace(target(name), retry_window_ms=3000) for name in ("diagnostic", "peer")]
    endpoint = GlobalRuntimeEndpoint(node, selected)
    calls = []

    def close_first():
        calls.append("first")
        raise BackendLifecycleError("released with a diagnostic")

    handles = [
        ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}, close=close_first)),
        ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}, close=lambda: calls.append("second"))),
    ]
    bindings = []
    for handle, descriptor in zip(handles, selected, strict=True):
        handle.load()
        binding = bind_runtime(LocalNode(services), handle, descriptor)
        bindings.append(binding)
        for _ in range(3):
            binding.adapter._heartbeat()

    def pump_once():
        endpoint._tick()
        for binding in bindings:
            if not binding._closed:
                binding.adapter._heartbeat()

    try:
        with _runtime_pump(pump_once):
            with pytest.raises(ExecutionFailure, match="close_failed"):
                handles[0].close(deadline=Deadline.after(2))
            assert handles[0].closed and handles[0].last_failure.code == "close_failed"
            if peer_recovery:
                endpoint.core.fence_resource("cpu")
            handles[1].close(deadline=Deadline.after(2))
            # The first caller still receives its diagnostic. Retrying Close
            # observes retirement without repeating irreversible SDK cleanup.
            handles[0].close(deadline=Deadline.after(2))
            handles[1].close(deadline=Deadline.after(2))
            assert calls == ["first", "second"]
            assert all(handle.closed for handle in handles)
            assert all(binding._closed for binding in bindings)
            assert not endpoint.core.shutdown_pending()
            assert not endpoint.core._fenced
    finally:
        for binding in bindings:
            _dispose_fake_transport(binding.adapter)


@pytest.mark.parametrize("static_required", [False, True])
@pytest.mark.parametrize("late_fault_report", [False, True])
def test_drained_peer_health_only_blocks_configured_required_members(
    transport, monkeypatch, static_required, late_fault_report
):
    from inference_service.backends.errors import BackendInferenceError

    services, node, endpoint = transport
    calls = []

    class Model:
        def execute(self, request, context):
            if request.inputs.get("unknown"):
                raise BackendInferenceError("device completion unknown", operation_started=True, outcome_known=False)
            return {"answer": 7}

        def drain_uncertain_operations(self):
            calls.append("drain")

        def reset(self):
            calls.append("reset")

    selected = [replace(target(name), retry_window_ms=3000) for name in ("failed", "closing", "healthy")]
    if static_required:
        endpoint = GlobalRuntimeEndpoint(
            node, selected, resources={"cpu": {"physical_device": "local/cpu", "mode": "serial"}}
        )
    handles, bindings = [], []
    for descriptor in selected:
        handle = ModelRuntimeHandle(Model(), resettable=True)
        handle.load()
        binding = bind_runtime(LocalNode(services), handle, descriptor)
        handles.append(handle)
        bindings.append(binding)
        for _ in range(3):
            binding.adapter._heartbeat()

    def heartbeat():
        for binding in bindings:
            if not binding._closed:
                binding.adapter._heartbeat()

    close_queued = threading.Event()
    close_submissions = []
    submit = endpoint.core.submit

    def submit_with_close_barrier(submission):
        result = submit(submission)
        if submission.work == "close":
            close_submissions.append(submission)
            close_queued.set()
        return result

    def tick():
        endpoint._tick()
        if late_fault_report and not close_queued.is_set():
            with endpoint.core.transaction():
                dispatched = any(
                    record.submission.target_id == selected[0].target_id and record.state in {"dispatching", "running"}
                    for record in endpoint.core._records.values()
                )
            if dispatched:
                # Host completion can reach its caller before Global receives
                # UNKNOWN or a dirty heartbeat. Admit the peer Close first.
                assert close_queued.wait(1), "peer Close was not admitted"

    monkeypatch.setattr(endpoint.core, "submit", submit_with_close_barrier)
    try:
        with _runtime_pump(tick, heartbeat):
            with pytest.raises(ExecutionFailure):
                handles[0].execute(ModelRequest({"unknown": True}), ExecutionContext("fault", Deadline.after(1)))
            handles[1].close(deadline=Deadline.after(2))
            assert handles[1].closed
            if late_fault_report:
                report = endpoint.core.query(close_submissions[0])
                assert report["state"] == "not_started" and report["error"] == "RESOURCE_NOT_READY"
            assert handles[0].reset_required and not handles[0].local_ready
            assert bindings[0].controller.status()["clean"] and not endpoint.core._fenced
            assert calls == ["drain"]
            if static_required:
                assert not endpoint.core.resource_ready("cpu")
                with pytest.raises(ExecutionFailure):
                    handles[2].execute(ModelRequest(), ExecutionContext("blocked", Deadline.after(0.05)))
            else:
                assert endpoint.core.resource_ready("cpu")
                assert handles[2].execute(ModelRequest(), ExecutionContext("healthy", Deadline.after(1))).outputs == {
                    "answer": 7
                }
            # Completion of this model's Reset does not depend on whether a
            # different lifecycle group has regained business readiness.
            handles[2].reset(ExecutionContext("healthy-reset", Deadline.after(2)))
            assert handles[2].local_ready and handles[0].reset_required
            assert calls == ["drain", "reset"]
            handles[0].reset(ExecutionContext("faulted-peer-reset", Deadline.after(2)))
            assert endpoint.core.resource_ready("cpu")
    finally:
        for binding in bindings:
            _dispose_fake_transport(binding.adapter)


@pytest.mark.parametrize("scenario", ["peer_unknown", "capacity", "superseded_close"])
def test_public_recovery_continues_after_resource_and_receipt_boundaries(transport, monkeypatch, scenario):
    services, _, endpoint = transport
    events = []

    class Model:
        def execute(self, request, context):
            if request.inputs.get("unknown"):

                class Unknown(RuntimeError):
                    operation_started = True
                    outcome_known = False

                raise Unknown("missing device evidence")
            return {"answer": 7}

        def drain_uncertain_operations(self):
            events.append("drain")

        def reset(self):
            events.append("reset")

        def close(self):
            events.append("close")

    selected = [
        replace(target(f"model_{i}"), retry_window_ms=1000, max_records=1, max_pending_requests=1) for i in range(2)
    ]
    handles = [ModelRuntimeHandle(Model(), resettable=True) for _ in selected]
    bindings = []
    for handle, descriptor in zip(handles, selected, strict=True):
        handle.load()
        binding = bind_runtime(LocalNode(services), handle, descriptor)
        bindings.append(binding)
        for _ in range(3):
            binding.adapter._heartbeat()

    def pump_once():
        endpoint._tick()
        for binding in bindings:
            if not binding._closed:
                binding.adapter._heartbeat()

    try:
        with _runtime_pump(pump_once):
            if scenario == "peer_unknown":
                with pytest.raises(ExecutionFailure):
                    handles[1].execute(
                        ModelRequest({"unknown": True}), ExecutionContext("uncertain", Deadline.after(1))
                    )
                handles[0].close(deadline=Deadline.after(2))
                assert handles[0].closed and not handles[1].local_ready
                # The peer owns its reset. Closing another model must not reset it.
                assert "reset" not in events and not endpoint.core._fenced
                handles[1].reset(ExecutionContext("peer-reset", Deadline.after(2)))
                assert handles[1].local_ready
            elif scenario == "capacity":
                handles[0].execute(ModelRequest(), ExecutionContext("retain-ordinary-history", Deadline.after(1)))
                for _ in range(5):
                    handles[0].reset(ExecutionContext("reset", Deadline.after(1)))
                with pytest.raises(ExecutionFailure):
                    handles[0].reset(ExecutionContext("capacity", Deadline.after(0.05)))
                time.sleep(1.05)
                handles[0].reset(ExecutionContext("retry-capacity", Deadline.after(2)))
                assert handles[0].local_ready and not endpoint.core._fenced
            else:
                original = bindings[1].controller._recover
                fail_activation = [True]

                def recover(command, *args):
                    if command == "activate" and fail_activation[0]:
                        raise ControlError("RECOVERY_EVIDENCE_PENDING")
                    return original(command, *args)

                monkeypatch.setattr(bindings[1].controller, "_recover", recover)
                endpoint.core.fence_resource("cpu")
                with pytest.raises(ExecutionFailure):
                    handles[0].close(deadline=Deadline.after(0.15))
                assert handles[0].closed and not bindings[0]._closed
                old_id = bindings[0].adapter._recovery_requests["close"]["operation_id"]
                time.sleep(2.1)  # Cross both retry expiry and the two-second retirement floor.
                fail_activation[0] = False
                handles[1].reset(ExecutionContext("superseding-reset", Deadline.after(2)))
                assert endpoint.recovery.records[old_id]["state"] == "superseded"
                endpoint._tick()
                assert selected[0].target_id in endpoint.core.targets
                endpoint.core.begin_shutdown()
                handles[0].close(deadline=Deadline.after(2))
                assert bindings[0]._closed and events.count("close") == 1
            for handle in handles:
                handle.close(deadline=Deadline.after(3))
            assert not endpoint.core.shutdown_pending()
    finally:
        for binding in bindings:
            _dispose_fake_transport(binding.adapter)


def test_dynamic_client_allocation_failure_isolated_and_retried(transport, monkeypatch, caplog):
    services, node, endpoint = transport
    healthy, healthy_handle = host(services, target("healthy"))
    create = node.create_client
    blocked = [True]

    def create_client(kind, name, **kwargs):
        if name == "/runtime/unreachable" and blocked[0]:
            raise RuntimeError("middleware allocation failed")
        return create(kind, name, **kwargs)

    monkeypatch.setattr(node, "create_client", create_client)
    failed, failed_handle = host(services, target("unreachable"))
    try:
        for _ in range(3):
            endpoint._tick()
        assert failed.target.target_id not in endpoint._clients
        assert sum("middleware allocation failed" in r.message for r in caplog.records) == 1
        submission = healthy.controller.prepare(ModelRequest(), ExecutionContext("healthy", Deadline.after(1)))
        endpoint.core.submit(submission)
        endpoint._tick()
        healthy.controller.completion(submission).result(timeout=1)
        assert healthy.controller.result(submission).outputs == {"answer": 1}
        blocked[0] = False
        endpoint._tick()
        assert failed.target.target_id in endpoint._clients
        assert not endpoint._client_errors
    finally:
        _dispose_fake_transport(healthy)
        _dispose_fake_transport(failed)
        healthy_handle.close()
        failed_handle.close()


@pytest.mark.parametrize("collect_receipt", [False, True])
def test_public_reset_releases_superseded_receipt_after_peer_recovery(transport, monkeypatch, collect_receipt):
    services, _, endpoint = transport
    resets = [0, 0]

    def reset(index):
        resets[index] += 1

    handles = [
        ModelRuntimeHandle(
            SimpleNamespace(
                execute=lambda *_: {}, reset=lambda _context, i=i: reset(i), drain_uncertain_operations=lambda: None
            ),
            resettable=True,
        )
        for i in range(2)
    ]
    bindings = []
    for i, handle in enumerate(handles):
        handle.load()
        binding = bind_runtime(LocalNode(services), handle, replace(target(f"reset_{i}"), retry_window_ms=1000))
        bindings.append(binding)
        for _ in range(3):
            binding.adapter._heartbeat()
    original = bindings[1].controller._recover
    fail_activation = [True]

    def recover(command, *args):
        if command == "activate" and fail_activation[0]:
            raise ControlError("RECOVERY_EVIDENCE_PENDING")
        return original(command, *args)

    monkeypatch.setattr(bindings[1].controller, "_recover", recover)

    def pump():
        endpoint._tick()
        for binding in bindings:
            binding.adapter._heartbeat()

    try:
        with _runtime_pump(pump):
            endpoint.core.fence_resource("cpu")
            with pytest.raises(ExecutionFailure):
                handles[0].reset(ExecutionContext("initial-reset", Deadline.after(0.2)))
            old_id = bindings[0].adapter._recovery_requests["reset"]["operation_id"]
            assert resets == [1, 0]
        time.sleep(1.05)
        if not collect_receipt:
            monkeypatch.setattr(endpoint.recovery, "collect", lambda: None)
        fail_activation[0] = False
        with _runtime_pump(pump):
            handles[1].reset(ExecutionContext("peer-reset", Deadline.after(2)))
            endpoint.recovery.collect()
            if collect_receipt:
                assert old_id not in endpoint.recovery.records
            else:
                assert endpoint.recovery.records[old_id]["state"] == "superseded"
            expected = "RECOVERY_RECEIPT_UNAVAILABLE" if collect_receipt else "RECOVERY_SUPERSEDED"
            with pytest.raises(ExecutionFailure, match=expected):
                handles[0].reset(ExecutionContext("settle-old-reset", Deadline.after(2)))
            assert resets == [1, 1], "settling an unknown receipt must not repeat Reset"
            assert not bindings[0].adapter.has_recovery("reset")
            handles[0].reset(ExecutionContext("new-reset", Deadline.after(2)))
            assert resets == [2, 1] and handles[0].local_ready
    finally:
        for binding in bindings:
            _dispose_fake_transport(binding.adapter)


@pytest.mark.parametrize("delayed", [False, True])
def test_terminal_backlog_has_bounded_batches_and_inflight_requests(transport, monkeypatch, delayed):
    services, _, _ = transport
    adapter, handle = host(services, replace(target("backlog"), max_records=1024))
    sent = []
    futures = []

    def call(request):
        message = decode_control(request.payload)
        assert message["verb"] == "terminal_report"
        sent.append(message["report"]["submission"]["operation_id"])
        future = Future()
        if not delayed:
            future.set_result(SimpleNamespace(payload=encode_control({"state": "pending"})))
        futures.append(future)
        return future

    monkeypatch.setattr(adapter._client, "call_async", call)
    try:
        for _ in range(513):
            submission = adapter.controller.prepare(ModelRequest(), ExecutionContext(str(uuid4()), Deadline.after(2)))
            adapter.controller.cancel(submission, adapter.controller.epoch)
        for _ in range(5):
            before = len(sent)
            adapter._terminal_tick()
            assert len(sent) - before <= 32
            assert len(adapter._terminal_receipts) <= 64
        if delayed:
            assert len(sent) == 64
            for future in futures:
                future.set_result(SimpleNamespace(payload=encode_control({"state": "pending"})))
        newest = adapter.controller.prepare(ModelRequest(), ExecutionContext("newest", Deadline.after(2)))
        adapter.controller.cancel(newest, adapter.controller.epoch)
        before = len(sent)
        adapter._terminal_tick()
        assert newest.operation_id in sent[before:]
        for _ in range(20):
            adapter._terminal_tick()
            for future in tuple(futures):
                if not future.done():
                    future.set_result(SimpleNamespace(payload=encode_control({"state": "pending"})))
        assert newest.operation_id in sent
    finally:
        monkeypatch.undo()
        _dispose_fake_transport(adapter)
        handle.close()


def test_unchanged_heartbeat_does_not_rescan_client_directory(transport, monkeypatch):
    services, _, endpoint = transport
    adapter, handle = host(services, target("heartbeat"))
    try:
        with monkeypatch.context() as patch:
            patch.setattr(endpoint, "_sync_clients", lambda: pytest.fail("heartbeat rescanned client directory"))
            for _ in range(5):
                adapter._heartbeat()
            assert adapter.controller.epoch == endpoint.core.epoch
            assert not adapter._registration_error
        submission = adapter.controller.prepare(ModelRequest(), ExecutionContext("after-heartbeats", Deadline.after(1)))
        endpoint.core.submit(submission)
        endpoint._tick()
        adapter.controller.completion(submission).result(timeout=1)
        assert adapter.controller.result(submission).outputs == {"answer": 1}
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


def test_retirement_errors_logged_only_when_the_reason_changes(transport, monkeypatch):
    services, _, _ = transport
    adapter, handle = host(services, target())
    reason = [ControlError("RUNTIME_IDENTITY_MISMATCH")]

    def reject(*args, **kwargs):
        raise reason[0]

    try:
        with monkeypatch.context() as patch:
            patch.setattr(adapter, "_call_control", reject)
            for _ in range(3):
                assert not adapter._unregister_clean_host()
            assert len(adapter.node.warnings) == 1
            assert "RUNTIME_IDENTITY_MISMATCH" in adapter.node.warnings[0]
            reason[0] = RuntimeError("middleware unavailable")
            assert not adapter._unregister_clean_host()
            assert len(adapter.node.warnings) == 2
            assert "middleware unavailable" in adapter.node.warnings[-1]
        assert not adapter._closing
    finally:
        _dispose_fake_transport(adapter)
        handle.close()


def test_public_binder_preserves_optional_unavailable_recovery(transport):
    from inference_service.scheduler.runtime_controller import UnavailableRuntime

    services, _, endpoint = transport
    unavailable = UnavailableRuntime()
    adapter = bind_runtime(LocalNode(services), unavailable, replace(target(), required=False))
    try:
        for _ in range(3):
            adapter._heartbeat()
        assert adapter.controller.handle is unavailable
        assert adapter.controller._clean_optional_unavailable()
        assert endpoint.core.resource_ready("cpu")
        adapter.controller._recover("drain")
        adapter.controller._recover("activate", 2)
        assert adapter.controller.generation == 2
    finally:
        # Restore the expected generation for fixture retirement.
        adapter.controller.generation = 1
        _dispose_fake_transport(adapter)
