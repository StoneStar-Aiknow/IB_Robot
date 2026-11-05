import logging
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from inference_service.scheduler.runtime_controller import RuntimeExecutionController
from inference_service.scheduler.runtime_protocol import (
    RuntimeExecutionTarget,
)
from inference_service.unified_runtime import Deadline, ExecutionContext, ModelRequest, ModelRuntimeHandle


def target(name="sam", resource="gpu", **kwargs):
    return RuntimeExecutionTarget(
        target_id=f"perception/{name}",
        endpoint=f"/runtime/{name}",
        hardware_resource_id=resource,
        physical_device=f"local/{resource}",
        deployment_fingerprint="deployment",
        composition_fingerprint="composition",
        lifecycle_group=name,
        configuration_generation="configuration",
        interface="tensor_model",
        model_type=name,
        operation="infer",
        **kwargs,
    )


def attested_resources(*targets):
    """Explicit resource-owner evidence for synthetic prediction/concurrency tests."""
    return {
        t.hardware_resource_id: {
            "physical_device": t.physical_device,
            "mode": t.resource_mode,
            "verified_concurrency": True,
            "managed_coverage": True,
            "predictive_admission": t.predictive_admission,
            "hardware_profile_fingerprint": t.hardware_profile_fingerprint,
            "admission_safety_margin_ms": t.admission_safety_margin_ms,
        }
        for t in targets
    }


class Executor:
    def __init__(self):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.error = None

    def execute(self, request, context):
        self.calls.append(request)
        self.entered.set()
        assert self.release.wait(2)
        if self.error:
            raise self.error
        return {"answer": np.array([1])}


def controller(t, **kwargs):
    executor = Executor()
    handle = ModelRuntimeHandle(executor=executor)
    handle.load()
    return RuntimeExecutionController(t, handle, **kwargs), executor


def connect(core, host):
    host.accept_epoch(core.register(host.status())["epoch"])
    core.register(host.status())


def prepare(host, **kwargs):
    return host.prepare(
        ModelRequest({"image": np.zeros((2, 3), dtype=np.uint8)}),
        ExecutionContext(str(uuid4()), Deadline.after(1)),
        **kwargs,
    )


class FakeClient:
    def __init__(self, bus, endpoint):
        self.bus, self.endpoint = bus, endpoint
        self.available = True

    def service_is_ready(self):
        return self.available and self.endpoint in self.bus

    def call_async(self, request):
        future = Future()
        future.set_result(self.bus[self.endpoint](request, SimpleNamespace()))
        return future

    def remove_pending_request(self, future):
        pass


class FakeNode:
    def __init__(self, bus):
        self.bus = bus

    def create_client(self, kind, endpoint, **kwargs):
        return FakeClient(self.bus, endpoint)

    def create_service(self, kind, endpoint, callback, **kwargs):
        self.bus[endpoint] = callback
        return endpoint

    def create_timer(self, interval, callback, **kwargs):
        return SimpleNamespace(cancel=lambda: None)

    def create_guard_condition(self, callback, **kwargs):
        return SimpleNamespace(trigger=lambda: None)

    def get_logger(self):
        return logging.getLogger(__name__)

    def destroy_client(self, client):
        pass

    def destroy_service(self, endpoint):
        self.bus.pop(endpoint, None)

    def destroy_timer(self, timer):
        pass

    def destroy_guard_condition(self, guard):
        pass
