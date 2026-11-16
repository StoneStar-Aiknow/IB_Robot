"""Startup and cleanup share the stop signal's absolute budget."""

import signal
import time
from contextlib import nullcontext

import pytest

from inference_service.host_lifecycle import HostRequests, host_stop_signals
from inference_service.unified_runtime import ExecutionContext, ModelRuntimeHandle
from inference_service.unified_runtime.construction import RuntimeConstruction
from inference_service.unified_runtime.lifecycle import lifecycle_budget, lifecycle_remaining, wait_for_startup


def test_duplicate_context_is_rejected_without_losing_original_request():
    requests = HostRequests()
    context = ExecutionContext("shared")
    with requests.request(context):
        with pytest.raises(ValueError, match="already active"), requests.request(context):
            pytest.fail("duplicate scope entered")
        assert not requests.drained
        requests.stop()
        assert context.cancellation.cancelled
        assert not requests.drained
    assert requests.drained


def test_signal_during_model_load_rolls_back_without_publishing(monkeypatch):
    monkeypatch.setattr("inference_service.host_lifecycle.HOST_SHUTDOWN_TIMEOUT_SEC", 0.5)
    events = []

    class Model:
        def execute(self, request, context):
            return {}

        def close(self):
            events.append("closed")
            assert 0 < lifecycle_remaining(50) < 0.48

    with host_stop_signals() as stop:
        construction = RuntimeConstruction(
            "online", initialization=nullcontext(), bind=lambda h: events.append("bound")
        )
        with pytest.raises(RuntimeError, match="stopped during initialization"), construction:
            handle = ModelRuntimeHandle(executor=Model())
            handle.load()
            signal.raise_signal(signal.SIGTERM)
            deadline = stop.deadline
            time.sleep(0.04)  # SDK returns after the stop request.
            signal.raise_signal(signal.SIGINT)
            assert stop.deadline is deadline
        assert events == ["closed"]
        assert handle.closed and construction.rollback_clean
        with lifecycle_budget(50):
            assert lifecycle_remaining(50) < 0.48


def test_discovery_stops_before_starting_another_model(monkeypatch):
    monkeypatch.setattr("inference_service.host_lifecycle.HOST_SHUTDOWN_TIMEOUT_SEC", 0.01)
    with host_stop_signals():

        def wait(timeout):
            signal.raise_signal(signal.SIGTERM)
            return False

        with pytest.raises(RuntimeError, match="stopped during initialization"):
            wait_for_startup(wait, 60)
        time.sleep(0.02)
        with lifecycle_budget(50):
            assert lifecycle_remaining(50) == 0


def test_dependency_construction_failure_shuts_down_ros_and_allows_retry():
    import rclpy

    from inference_service.host_lifecycle import run_model_node

    def fail():
        raise ValueError("dependency configuration rejected")

    for _ in range(2):
        with pytest.raises(ValueError, match="dependency configuration rejected"):
            run_model_node(lambda *_: pytest.fail("node created after dependency failure"), fail)
        assert not rclpy.ok()


def test_signal_reaches_loading_worker_and_its_rollback_deadline(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from inference_service.runtime_creation import runtime_creation
    from inference_service.unified_runtime import RuntimeProviders
    from inference_service.unified_runtime.lifecycle import lifecycle_deadline

    monkeypatch.setattr("inference_service.host_lifecycle.HOST_SHUTDOWN_TIMEOUT_SEC", 1.0)
    entered, release = threading.Event(), threading.Event()
    observed = []

    class Model:
        def load(self):
            entered.set()
            assert release.wait(3)

        def execute(self, *_):
            return {}

        def close(self):
            observed.append(lifecycle_deadline())

    providers = RuntimeProviders.create(object(), object())

    def load():
        with runtime_creation(providers=providers, timeout=30):
            handle = ModelRuntimeHandle(Model())
            handle.load()
        pytest.fail("startup completed after the stop signal")

    try:
        with host_stop_signals() as stop, ThreadPoolExecutor(1) as workers:
            pending = workers.submit(load)
            assert entered.wait(2)
            signal.raise_signal(signal.SIGTERM)
            release.set()
            with pytest.raises(RuntimeError, match="stopped during initialization"):
                pending.result(timeout=2)
            assert observed == [stop.deadline]
    finally:
        release.set()
        providers.close()
