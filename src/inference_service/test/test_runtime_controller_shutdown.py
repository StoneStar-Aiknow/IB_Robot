"""Controller shutdown must remain irreversible across lifecycle completion."""

import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from inference_service.scheduler.runtime_controller import RuntimeExecutionController
from inference_service.scheduler.runtime_protocol import ControlError, RuntimeExecutionTarget
from inference_service.unified_runtime import Deadline, ExecutionContext, ModelRequest, ModelRuntimeHandle


class _ResettableExecutor:
    def __init__(self):
        self.resets = 0

    def execute(self, _request, _context):
        return {}

    def reset(self, _context):
        self.resets += 1


def test_successful_reset_completion_cannot_reopen_controller_during_shutdown(monkeypatch):
    executor = _ResettableExecutor()
    handle = ModelRuntimeHandle(executor=executor, resettable=True)
    handle.load()
    target = RuntimeExecutionTarget(
        target_id="perception/resettable",
        endpoint="/runtime/resettable",
        hardware_resource_id="cpu",
        physical_device="local/cpu",
        deployment_fingerprint="model",
        composition_fingerprint="plugin",
        lifecycle_group="resettable",
        configuration_generation="config",
        interface="tensor_model",
        model_type="test",
        operation="infer",
    )
    controller = RuntimeExecutionController(target, handle)
    controller.accept_epoch(str(uuid4()))
    reset_returned = threading.Event()
    release_completion = threading.Event()
    actual_reset = handle._reset_local

    def hold_successful_reset(context, **kwargs):
        actual_reset(context, **kwargs)
        reset_returned.set()
        assert release_completion.wait(2)

    monkeypatch.setattr(handle, "_reset_local", hold_successful_reset)
    workers = ThreadPoolExecutor(max_workers=1)
    try:
        submission = controller.prepare(ModelRequest(), ExecutionContext("reset", Deadline.after(2)), work="reset")
        controller.execute(submission, controller.epoch)
        assert reset_returned.wait(1), "real handle reset must finish before shutdown begins"
        assert executor.resets == 1 and handle.local_ready
        assert not handle.ready  # The controller's maintenance barrier is still held.
        record = controller._records[submission.operation_id]
        shutdown = workers.submit(controller.shutdown)
        assert record.context.cancellation_token.wait(1), "shutdown must close admission before reset completion"
        assert controller._gate == "closing"
        assert not shutdown.done(), "shutdown must wait for the held lifecycle worker"
        release_completion.set()
        shutdown.result(timeout=2)
        report = controller.query(submission, controller.epoch)
        assert report["state"] == "completed" and not report["error"]
        assert controller.generation == 2
        assert controller._gate == "closing"
        assert not controller.status()["ready"]
        with pytest.raises(ControlError, match="RUNTIME_NOT_REGISTERED"):
            controller.prepare(ModelRequest(), ExecutionContext("after-close", Deadline.after(1)))
    finally:
        release_completion.set()
        workers.shutdown(wait=True)
        controller.shutdown()
        handle.close()


def test_shutdown_does_not_cancel_completed_callers_shared_context():
    from .runtime_scheduler_helpers import controller, target

    context = ExecutionContext("shared-cleanup", Deadline.after(2))
    hosts = [controller(target(name))[0] for name in ("first", "second")]
    try:
        for host in hosts:
            host.accept_epoch("test")
            sub = host.prepare(ModelRequest(), context, work="close")
            host.execute(sub, "test")
            host.completion(sub).result(timeout=1)
            assert host.handle.closed
            host.shutdown(context)
            context.check("next-host")
    finally:
        for host in hosts:
            host.shutdown()


@pytest.mark.parametrize("action", ["cancel", "cancel_pending", "shutdown"])
def test_local_cancellation_is_isolated_from_parent_and_sibling(action):
    from .runtime_scheduler_helpers import controller, target

    context = ExecutionContext("shared-business", Deadline.after(2))
    hosts = [controller(target(name))[0] for name in ("first", "second")]
    try:
        for host in hosts:
            host.accept_epoch("test")
        submissions = [host.prepare(ModelRequest(), context) for host in hosts]
        sibling_token = hosts[1]._records[submissions[1].operation_id].context.cancellation_token
        if action == "cancel":
            hosts[0].cancel(submissions[0], "test")
        else:
            getattr(hosts[0], action)()
        context.check("next-host")
        assert not sibling_token.cancelled
        context.cancellation_token.cancel("caller stopped")
        assert sibling_token.cancelled and sibling_token.reason == "caller stopped"
    finally:
        for host in hosts:
            host.shutdown()
            host.handle.close()


@pytest.mark.parametrize("stop", ["cancel", "cancel_pending", "shutdown"])
def test_stopping_prepared_host_preserves_shared_caller_and_other_host(stop):
    from .runtime_scheduler_helpers import controller, target

    context = ExecutionContext("shared-call", Deadline.after(2))
    hosts = [controller(target(name))[0] for name in ("first", "second")]
    try:
        for host in hosts:
            host.accept_epoch("test")
        submissions = [host.prepare(ModelRequest(), context) for host in hosts]
        peer_token = hosts[1]._records[submissions[1].operation_id].context.cancellation_token
        if stop == "cancel":
            hosts[0].cancel(submissions[0], "test")
        else:
            getattr(hosts[0], stop)()
        assert hosts[0].query(submissions[0], "test")["state"] == "not_started"
        context.check("continue-on-peer")
        assert not peer_token.cancelled
        context.cancellation_token.cancel("caller cancelled all stages")
        assert peer_token.wait(0) and peer_token.reason == "caller cancelled all stages"
        hosts[1].execute(submissions[1], "test")
        hosts[1].completion(submissions[1]).result(timeout=1)
        assert hosts[1].query(submissions[1], "test")["state"] == "not_started"
    finally:
        for host in hosts:
            host.shutdown()


def test_shutdown_deadline_retains_running_owner_until_worker_returns():
    from inference_service.unified_runtime.errors import DeadlineExceeded

    from .runtime_scheduler_helpers import controller, prepare, target

    host, executor = controller(target())
    executor.release.clear()
    host.accept_epoch("test")
    try:
        sub = prepare(host)
        host.execute(sub, "test")
        assert executor.entered.wait(1)
        with pytest.raises(DeadlineExceeded):
            host.shutdown(ExecutionContext("bounded-shutdown", Deadline.after(0.02)))
        assert not host.completion(sub).done()
        assert host._input_bytes > 0 and host._gate == "closing"
        executor.release.set()
        host.completion(sub).result(timeout=1)
        host.shutdown()
        assert host.query(sub, "test")["state"] == "completed"
    finally:
        executor.release.set()
        host.shutdown()


def test_child_cancellation_links_are_released_after_rejection_and_completion():
    from .runtime_scheduler_helpers import controller, target

    host, _ = controller(target())
    host.accept_epoch("test")
    context = ExecutionContext("reusable", Deadline.after(2))
    try:
        for _ in range(3):
            with pytest.raises(ControlError, match="CLOSE_PAYLOAD_UNSUPPORTED"):
                host.prepare(ModelRequest({"unexpected": 1}), context, work="close")
            assert not context.cancellation_token._callbacks
            sub = host.prepare(ModelRequest(), context)
            host.cancel(sub, "test")
            context.check("retry")
            assert not context.cancellation_token._callbacks
        sub = host.prepare(ModelRequest(), context)
        host.execute(sub, "test")
        host.completion(sub).result(timeout=1)
        assert not context.cancellation_token._callbacks
    finally:
        host.shutdown()
        host.handle.close()


def test_shutdown_waits_for_input_preparation_with_its_own_context(monkeypatch):
    from inference_service.scheduler import runtime_controller
    from inference_service.unified_runtime.errors import CancellationRequested

    from .runtime_scheduler_helpers import controller, target

    host, _ = controller(target())
    host.accept_epoch("test")
    entered, release = threading.Event(), threading.Event()
    actual_freeze = runtime_controller.freeze_inputs
    request_context = ExecutionContext("preparing", Deadline.after(2))
    cleanup_context = ExecutionContext("shutdown", Deadline.after(2))
    preparing_tokens = []

    def freeze(*args, **kwargs):
        preparing_tokens.append(kwargs["context"].cancellation_token)
        entered.set()
        assert release.wait(2)
        return actual_freeze(*args, **kwargs)

    monkeypatch.setattr(runtime_controller, "freeze_inputs", freeze)
    try:
        with ThreadPoolExecutor(2) as workers:
            preparing = workers.submit(host.prepare, ModelRequest(), request_context)
            assert entered.wait(1)
            shutdown = workers.submit(host.shutdown, cleanup_context)
            try:
                assert preparing_tokens[0].wait(1)
                request_context.check("caller-still-active")
                assert not shutdown.done()
                cleanup_context.check("shutdown-still-waiting")
            finally:
                release.set()
            with pytest.raises(CancellationRequested):
                preparing.result(timeout=1)
            shutdown.result(timeout=1)
        assert host._input_bytes == 0 and not host._records
        with pytest.raises(RuntimeError, match="shutdown"):
            host._workers.submit(lambda: None)
    finally:
        release.set()
        host.shutdown()
        host.handle.close()


def test_shutdown_settles_prepared_requests_but_keeps_terminal_evidence():
    from .runtime_scheduler_helpers import controller, prepare, target

    host, model = controller(target())
    host.accept_epoch("test")
    sub = prepare(host)
    done = host.completion(sub)
    host.shutdown()
    assert done.done() and not model.calls
    assert host.query(sub, "test")["state"] == "not_started"
    assert host._input_bytes == 0
    assert not host._records[sub.operation_id].acknowledged
    host.release_result_waiter(sub)
    host.acknowledge(sub, "test")
    host.now_ns = lambda: sub.accept_until_ns
    host.collect()
    assert not host._records
    host.handle.close()


def test_cancelled_reset_before_native_entry_preserves_healthy_admission():
    from .runtime_scheduler_helpers import target

    model = _ResettableExecutor()
    handle = ModelRuntimeHandle(executor=model, resettable=True)
    handle.load()
    host = RuntimeExecutionController(target(), handle)
    host.accept_epoch("test")
    context = ExecutionContext("cancelled-reset", Deadline.after(1))
    try:
        sub = host.prepare(ModelRequest(), context, work="reset")
        context.cancellation_token.cancel("cancel before native reset")
        host.execute(sub, "test")
        host.completion(sub).result(timeout=1)
        assert host.query(sub, "test")["state"] == "not_started"
        assert model.resets == 0 and host.generation == 1
        assert handle.local_ready and host.ready
    finally:
        host.shutdown()
        handle.close()
