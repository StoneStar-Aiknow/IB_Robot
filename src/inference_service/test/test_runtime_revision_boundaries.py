"""Regression coverage for public construction, recovery and transport boundaries."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from inference_service.runtime_creation import runtime_creation
from inference_service.scheduler.runtime_environment import RuntimeEnvironment
from inference_service.scheduler.runtime_initialization import InitializationSettlement, settle_initializations
from inference_service.unified_runtime import ExecutionContext, ModelRequest, ModelRuntimeHandle
from inference_service.unified_runtime.construction import RuntimeConstruction
from inference_service.unified_runtime.factory import ModelRuntimeFactory, RuntimeFactoryError
from inference_service.unified_runtime.lifecycle import lifecycle_budget, lifecycle_stop_scope
from inference_service.unified_runtime.loading import LoadRollback
from inference_service.unified_runtime.registry import RuntimeDependencyError


@pytest.mark.parametrize("registry,providers", [(None, None), (object(), None), (object(), object())])
def test_factory_preconditions_prove_no_resource_acquired(registry, providers):
    construction = RuntimeConstruction("online", initialization=nullcontext(), bind=lambda _: None)
    with pytest.raises(RuntimeFactoryError), construction:
        ModelRuntimeFactory.create(object(), registry, providers)
    assert construction.failed and construction.rollback_clean
    construction.retry_rollback()


@pytest.mark.parametrize("caught", [False, True])
def test_pending_factory_cleanup_cannot_publish_or_release_initialization(caught):
    published, settled = [], []
    blocked = [True]

    def release():
        if blocked[0]:
            raise RuntimeError("native cleanup pending")

    construction = RuntimeConstruction(
        "online",
        initialization=nullcontext(),
        bind=lambda _: pytest.fail("published pending cleanup"),
        publish=published.append,
    )
    with pytest.raises(RuntimeDependencyError if caught else RuntimeError), construction:
        handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}))
        handle.load()
        construction.begin_acquisition()
        rollback = LoadRollback()
        rollback.defer(release)
        assert rollback.rollback()
        construction.cleanup_owners.append(rollback)
        if not caught:
            raise RuntimeError("second assembly failed")
    settled.append(construction.rollback_clean)
    assert handle.closed and settled == [False] and not published
    with pytest.raises(RuntimeError, match="native cleanup pending"):
        construction.retry_rollback()
    blocked[0] = False
    construction.retry_rollback()
    assert construction.rollback_clean and not construction.cleanup_owners


def test_settlement_disposal_and_concurrent_removal_are_retryable():
    node = SimpleNamespace(_runtime_initialization_receipts=[])
    replays, disposals = [], []

    def dispose():
        disposals.append(True)
        if len(disposals) == 1:
            raise RuntimeError("transport busy")

    receipt = InitializationSettlement(node, {}, lambda *_: replays.append(True), dispose, 1)
    node._runtime_initialization_receipts.append(receipt)
    with pytest.raises(RuntimeError, match="transport busy"):
        receipt.settle()
    assert not receipt.done and node._runtime_initialization_receipts == [receipt]
    with ThreadPoolExecutor(2) as workers:
        calls = [workers.submit(receipt.settle), workers.submit(settle_initializations, node)]
        for call in calls:
            call.result(timeout=1)
    assert receipt.done and not node._runtime_initialization_receipts
    assert replays == [True] and len(disposals) == 2


def configured_node(**values):
    def declare(name, default, *args):
        return SimpleNamespace(value=values.setdefault(name, default))

    return SimpleNamespace(
        has_parameter=lambda name: name in values,
        get_parameter=lambda name: SimpleNamespace(value=values[name]),
        declare_parameter=declare,
    )


def test_local_handle_freezes_process_scheduler_before_a_control_node_exists():
    environment = RuntimeEnvironment()
    providers = SimpleNamespace(runtime_environment=environment)
    with runtime_creation(providers=providers):
        handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}))
        handle.load()
    try:
        assert environment._node is None
        with pytest.raises(ValueError, match="cannot change runtime scheduling"):
            environment.configure_from_node(configured_node(runtime_scheduler_enabled=True))
        assert not environment.scheduler_enabled
        assert handle.execute(ModelRequest(), ExecutionContext("direct")).outputs == {}
    finally:
        environment.close()


@pytest.mark.parametrize(
    "conflict",
    [
        {"runtime_scheduler_enabled": False},
        {"runtime_scheduler_endpoint": "/other/runtime"},
        {"runtime_initialization_timeout": 12.0},
        {"configured_target_ids": ["other"]},
    ],
)
def test_first_host_configuration_cannot_be_replaced_before_freeze(monkeypatch, conflict):
    environment = RuntimeEnvironment()
    configured, competing = threading.Event(), threading.Event()
    original = environment.configure_from_node
    values = {"runtime_scheduler_enabled": True, "runtime_scheduler_endpoint": "/first/runtime"}
    first = configured_node(**values)
    second = configured_node(**{**values, **conflict})

    def configure(node):
        original(node)
        if node is first:
            configured.set()
            assert competing.wait(1)

    monkeypatch.setattr(environment, "configure_from_node", configure)

    def create_first():
        with environment.initialization(first):
            return environment.scheduler_enabled, environment.endpoint, environment.configured_target_ids

    def create_second():
        assert configured.wait(1)
        competing.set()
        with environment.initialization(second):
            pytest.fail("conflicting host replaced admitted configuration")

    try:
        with ThreadPoolExecutor(2) as workers:
            a = workers.submit(create_first)
            b = workers.submit(create_second)
            assert a.result(timeout=2) == (True, "/first/runtime", ())
            with pytest.raises(ValueError, match="after runtime creation starts"):
                b.result(timeout=2)
    finally:
        environment.close()


@pytest.mark.parametrize(
    "invalid",
    [
        {"runtime_initialization_timeout": -1},
        {"configured_target_ids": [""]},
        {"runtime_scheduler_endpoint": "relative"},
        {"runtime_scheduler_endpoint": "/bad-name"},
        {"runtime_scheduler_endpoint": "/a//b"},
        {"runtime_scheduler_endpoint": "/" + "a" * 247},
        {"runtime_scheduler_endpoint": "/" + "a" * 241},
    ],
)
def test_failed_configuration_does_not_change_process_policy(invalid):
    environment = RuntimeEnvironment()
    before = (
        environment.scheduler_enabled,
        environment.endpoint,
        environment.configured_target_ids,
        environment.initialization_timeout,
    )
    with pytest.raises(ValueError):
        environment.configure_from_node(configured_node(runtime_scheduler_enabled=True, **invalid))
    assert before == (
        environment.scheduler_enabled,
        environment.endpoint,
        environment.configured_target_ids,
        environment.initialization_timeout,
    )
    environment.configure_from_node(
        configured_node(runtime_scheduler_enabled=True, runtime_scheduler_endpoint="/corrected/runtime")
    )
    assert environment.scheduler_enabled and environment.endpoint == "/corrected/runtime"
    environment.close()


@pytest.mark.parametrize("endpoint", ["/bad-name", "/a//b", "/" + "a" * 241])
def test_global_rejects_invalid_endpoint_before_creating_ros_entities(endpoint):
    from inference_service.scheduler.runtime_ros import GlobalRuntimeEndpoint

    # No node methods can be called before validation finishes.
    with pytest.raises(ValueError, match="endpoint"):
        GlobalRuntimeEndpoint(object(), [], endpoint=endpoint)


@pytest.mark.parametrize("stop", [False, True])
def test_publication_shares_startup_deadline_and_stop(stop):
    environment = RuntimeEnvironment()
    stopping = threading.Event()
    observed = []

    def wait(context, **kwargs):
        observed.append(context.deadline.remaining_seconds())
        if stop:
            stopping.set()
        else:
            time.sleep(0.04)

    adapter = SimpleNamespace(node=None, wait_for_registration=wait, refresh_registration=lambda *_args, **_kw: None)
    with lifecycle_budget(0.03), lifecycle_stop_scope(stopping), pytest.raises(Exception, match="deadline|stopped"):
        environment.publish([SimpleNamespace(adapter=adapter)], 60)
    assert observed and observed[0] <= 0.03


def test_failed_transport_retirement_does_not_stop_other_control_callbacks():
    environment = RuntimeEnvironment()
    destroyed, spins = [], []

    def destroy(*, _quiescent=False):
        assert _quiescent
        destroyed.append(True)
        if len(destroyed) == 1:
            raise RuntimeError("ROS destroy failed once")

    def spin_once(**kwargs):
        spins.append(True)
        if len(spins) == 3:
            environment._stopping.set()

    environment._node = SimpleNamespace(executor=None)
    environment._executor = SimpleNamespace(
        _cb_iter=None,
        spin_once=spin_once,
        remove_node=lambda _: None,
        add_node=lambda _: None,
        shutdown=lambda: None,
    )
    environment._retired.put(SimpleNamespace(destroy_transport=destroy))
    environment._spin_control()
    assert len(destroyed) == 2 and len(spins) == 3
    assert environment._retired.unfinished_tasks == 0


def test_detached_control_cleanup_timeout_can_finish_before_next_close():
    from rclpy.context import Context
    from rclpy.node import Node

    from inference_service.scheduler.runtime_binding import closing_control_transport

    context = Context()
    context.init()
    node = Node("detached_cleanup_retry", context=context)
    entered, release, responded = threading.Event(), threading.Event(), threading.Event()

    def slow_callback():
        entered.set()
        assert release.wait(3)

    guard = node.create_guard_condition(slow_callback)
    try:
        with (
            pytest.raises(TimeoutError, match="still running"),
            lifecycle_budget(0.1),
            closing_control_transport(node),
        ):
            guard.trigger()
            assert entered.wait(1)
        old_executor = node.executor
        assert old_executor is not None
        # The retry must wait for this callback, not borrow a stopped pump.
        release.set()
        node.create_guard_condition(responded.set).trigger()
        with lifecycle_budget(2), closing_control_transport(node):
            assert node.executor is not old_executor
            assert responded.wait(1)
        assert node.executor is None
    finally:
        release.set()
        with lifecycle_budget(3), closing_control_transport(node):
            pass
        node.destroy_node()
        context.try_shutdown()


@pytest.mark.parametrize("phase", ["construct", "start"])
def test_detached_control_start_failure_releases_executor_before_retry(monkeypatch, phase):
    from rclpy.context import Context
    from rclpy.node import Node

    from inference_service.scheduler.runtime_binding import closing_control_transport

    context = Context()
    context.init()
    node = Node("detached_start_retry", context=context)
    responded = threading.Event()
    node.create_guard_condition(responded.set).trigger()
    original_thread = threading.Thread

    def fail(*args, **kwargs):
        raise RuntimeError("control thread unavailable")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                threading if phase == "construct" else original_thread,
                "Thread" if phase == "construct" else "start",
                fail,
            )
            with (
                pytest.raises(RuntimeError, match="control thread unavailable") as failure,
                closing_control_transport(node),
            ):
                pytest.fail("startup failure reached the Close body")
        # Keep the traceback alive: collection must not be responsible for detaching.
        assert failure.value.__traceback__ is not None
        assert node.executor is None
        with lifecycle_budget(2), closing_control_transport(node):
            assert responded.wait(1)
        assert node.executor is None
    finally:
        if node.executor is not None:
            node.executor.shutdown(timeout_sec=1)
            node.executor = None
        node.destroy_node()
        context.try_shutdown()


def test_first_parameter_declaration_is_shared_by_explicit_and_default_timeouts():
    from rclpy.exceptions import ParameterAlreadyDeclaredException

    environment = RuntimeEnvironment()
    first_checked, second_started = threading.Event(), threading.Event()
    parameters = {}

    def has_parameter(name):
        present = name in parameters
        if name == "runtime_scheduler_enabled" and not first_checked.is_set():
            first_checked.set()
            assert second_started.wait(1)
            # Give the default-timeout entrant a chance to contend for declaration.
            time.sleep(0.05)
        return present

    def declare(name, default, *args):
        if name in parameters:
            raise ParameterAlreadyDeclaredException(name)
        parameters[name] = default
        return SimpleNamespace(value=default)

    node = SimpleNamespace(
        has_parameter=has_parameter,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        declare_parameter=declare,
    )

    def create(timeout):
        if timeout is None:
            assert first_checked.wait(1)
            second_started.set()
        with runtime_creation(providers=SimpleNamespace(runtime_environment=environment), node=node, timeout=timeout):
            handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {"local": True}))
            handle.load()
        return handle

    try:
        with ThreadPoolExecutor(2) as workers:
            explicit = workers.submit(create, 2)
            default = workers.submit(create, None)
            handles = [explicit.result(timeout=3), default.result(timeout=3)]
        assert all(
            handle.execute(ModelRequest(), ExecutionContext("local")).outputs == {"local": True} for handle in handles
        )
    finally:
        environment.close()


def test_stream_step_forwards_the_same_priority_as_execute():
    from inference_service.scheduler.runtime_binding import RuntimeBinding

    priorities = []
    adapter = SimpleNamespace(execute=lambda request, context, **kwargs: priorities.append(kwargs["priority"]))
    binding = RuntimeBinding(adapter, None)
    request = ModelRequest(metadata={"priority": 7})
    binding.execute(request, ExecutionContext("execute"))
    binding.step("stream", request, ExecutionContext("step"))
    assert priorities == [7, 7]


def test_adapter_close_retries_worker_shutdown_before_destroying_transport(monkeypatch):
    from inference_service.scheduler import runtime_ros
    from inference_service.scheduler.runtime_ros import SchedulerRuntimeAdapter

    from .runtime_scheduler_helpers import LocalNode
    from .runtime_scheduler_helpers import transport_target as target

    monkeypatch.setattr(runtime_ros, "_service_type", lambda: SimpleNamespace(Request=SimpleNamespace))
    with runtime_creation(purpose="simulation"):
        handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}))
        handle.load()
    node = LocalNode({})
    node.context, node.executor = object(), object()
    adapter = SchedulerRuntimeAdapter(node, target(), handle, registration_enabled=False)
    handle._close_local(ExecutionContext("cleanup"), _dispatch_key=adapter.controller._authority)
    shutdown = adapter.controller.shutdown
    attempts = []

    def flaky_shutdown(context=None):
        attempts.append(True)
        if len(attempts) == 1:
            raise TimeoutError("worker is still running")
        shutdown(context)

    monkeypatch.setattr(adapter.controller, "shutdown", flaky_shutdown)
    with pytest.raises(TimeoutError):
        adapter.close()
    assert not node.destroyed
    adapter.close()
    assert len(attempts) == 2
    node.executor = None
    adapter.destroy_transport()
    assert len(node.destroyed) == 1


def test_global_client_retirement_waits_for_spin_boundary_and_worker(monkeypatch):
    from inference_service.scheduler import runtime_ros
    from inference_service.scheduler.runtime_ros import GlobalRuntimeEndpoint

    from .runtime_scheduler_helpers import LocalNode

    monkeypatch.setattr(runtime_ros, "_service_type", lambda: SimpleNamespace(Request=SimpleNamespace))
    node = LocalNode({})
    node.executor = object()
    endpoint = GlobalRuntimeEndpoint(node, [], resources={"cpu": {"physical_device": "local/cpu"}})
    client = SimpleNamespace(_executor_event=True)
    endpoint._retire_client(client)
    assert not node.destroyed
    events = []
    executor = SimpleNamespace(_cb_iter=SimpleNamespace(close=lambda: events.append("wait-set-released")))
    endpoint.collect_retired_clients(executor)
    assert not node.destroyed and events == ["wait-set-released"]
    client._executor_event = False
    endpoint.collect_retired_clients(executor)
    assert node.destroyed == [client] and not endpoint._retired_clients


def test_quiescent_close_retries_partial_transport_disposal(monkeypatch):
    from inference_service.scheduler import runtime_ros
    from inference_service.scheduler.runtime_ros import SchedulerRuntimeAdapter

    from .runtime_scheduler_helpers import LocalNode
    from .runtime_scheduler_helpers import transport_target as target

    monkeypatch.setattr(runtime_ros, "_service_type", lambda: SimpleNamespace(Request=SimpleNamespace))
    with runtime_creation(purpose="simulation"):
        handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}))
        handle.load()
    node = LocalNode({})
    adapter = SchedulerRuntimeAdapter(node, target(), handle, registration_enabled=False)
    handle._close_local(ExecutionContext("cleanup"), _dispatch_key=adapter.controller._authority)
    attempts = []
    original = node.destroy_client

    def destroy(client):
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("client disposal failed")
        original(client)

    monkeypatch.setattr(node, "destroy_client", destroy)
    with pytest.raises(RuntimeError, match="client disposal failed"):
        adapter.close()
    adapter.close()
    assert len(attempts) == 2 and adapter._transport_destroyed
