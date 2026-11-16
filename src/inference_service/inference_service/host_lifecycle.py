"""Process stop signals must leave ROS available for coordinated host Close."""

import logging
import signal
import threading
import time
from contextlib import contextmanager

from inference_service.ros_executor import release_cached_wait_set
from inference_service.unified_runtime.lifecycle import (
    LifecycleStop,
    lifecycle_budget,
    lifecycle_remaining,
    lifecycle_stop_scope,
)

# Leave time for final receipts before Global's 60-second transport drain.
HOST_SHUTDOWN_TIMEOUT_SEC = 50.0


class HostRequests:
    """Track requests until exit; each context may have only one active scope."""

    def __init__(self):
        self._lock = threading.Lock()
        self._contexts = {}
        self._stopping = False

    @contextmanager
    def request(self, context):
        with self._lock:
            if id(context) in self._contexts:
                raise ValueError("request context is already active")
            if self._stopping:
                context.cancellation.cancel("host stopping")
            else:
                self._contexts[id(context)] = context
        try:
            context.check("host_admission")
            yield context
        finally:
            with self._lock:
                self._contexts.pop(id(context), None)

    def stop(self):
        with self._lock:
            self._stopping = True
            contexts = tuple(self._contexts.values())
        for context in contexts:
            context.cancellation.cancel("host stopping")

    @property
    def drained(self):
        with self._lock:
            return not self._contexts


def _drain_requests(executor, requests, timeout):
    deadline = time.monotonic() + lifecycle_remaining(timeout)
    requests.stop()
    while not requests.drained:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("model host request drain is pending")
        executor.spin_once(timeout_sec=min(0.05, remaining))


@contextmanager
def host_stop_signals():
    """Install before construction and retain handlers throughout cleanup."""
    stopping = LifecycleStop(HOST_SHUTDOWN_TIMEOUT_SEC)
    previous = {sig: signal.signal(sig, lambda *_: stopping.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with lifecycle_stop_scope(stopping):
            yield stopping
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def close_model_nodes(nodes, providers, *, timeout=HOST_SHUTDOWN_TIMEOUT_SEC):
    """Retry unfinished node cleanup with the model control transport alive.

    Nodes are retained before initialization, so constructor exceptions cannot
    orphan workers or models. A deadline failure deliberately leaves providers
    and ROS alive; it is not evidence that device cleanup completed.
    """
    with lifecycle_budget(timeout):
        _close_model_nodes(nodes, providers, timeout)


def _close_model_nodes(nodes, providers, timeout):
    deadline = time.monotonic() + lifecycle_remaining(timeout)
    pending = list(nodes)
    while True:
        error = None
        for node in tuple(pending):
            if time.monotonic() >= deadline:
                raise TimeoutError("model host cleanup budget expired; control transport retained")
            try:
                if node.destroy_node() is False:
                    raise RuntimeError("node cleanup is pending")
                pending.remove(node)
            except Exception as exc:
                error = exc
        if not pending and time.monotonic() < deadline:
            try:
                providers.close()
                return
            except Exception as exc:
                error = exc
        if time.monotonic() >= deadline:
            raise TimeoutError("model host cleanup is pending; control transport retained") from error
        time.sleep(0.05)


def close_node_runtimes(node):
    """Close explicit-host constructions, including a failed constructor's owners."""
    from inference_service.scheduler.runtime_initialization import settle_initializations

    errors = []
    constructions = getattr(node, "_runtime_constructions", [])
    for construction in tuple(constructions):
        try:
            for handle in tuple(construction.handles):
                handle.close()
            if construction in constructions:
                constructions.remove(construction)
        except Exception as exc:
            errors.append(exc)
    try:
        settle_initializations(node)
    except Exception as exc:
        errors.append(exc)
    collect_node_runtime_transport(node)
    if errors:
        raise errors[0]


def collect_node_runtime_transport(node, executor=None):
    """Reclaim explicit-host transport between spin_once calls on the spin owner.

    run_model_node supplies this boundary. A custom host spin loop must call it
    too; callbacks may retire owners but must never destroy live wait-set data.
    """
    retired = getattr(node, "_runtime_retired_adapters", [])
    if not retired:
        return
    if executor is None:
        if getattr(node, "executor", None) is not None:
            return
    else:
        release_cached_wait_set(executor)
    for adapter in tuple(retired):
        entities = [getattr(adapter, name, None) for name in ("_timer", "_terminal_guard", "_service", "_client")]
        entity = next((item for item in entities if item is not None), None)
        group = entity.callback_group if executor is not None and entity is not None else None
        # All adapter entities share one mutually exclusive group. Hold it
        # through destruction so a queued worker cannot start taking ROS data.
        if group is not None and not group.beginning_execution(entity):
            continue
        try:
            if any(getattr(item, "_executor_event", False) for item in entities):
                continue
            adapter.destroy_transport(_quiescent=executor is not None)
        except Exception:
            logging.getLogger(__name__).exception("Explicit runtime transport retirement will retry")
        else:
            retired.remove(adapter)
        finally:
            if group is not None:
                group.ending_execution(entity)


def run_model_node(
    create_node,
    build_dependencies,
    *,
    args=None,
    executor_factory=None,
    callback_error=None,
    drain_callbacks=False,
    request_tracker=None,
):
    """Run an in-process model consumer using the common stop/close protocol.

    create_node receives dependencies and a callback to retain the node as soon
    as its ROS base and cleanup fields exist, before fallible initialization.
    Native consumers can drain callbacks first; serving hosts instead cancel
    and drain their requests in destroy_node with control transport still live.
    """
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.signals import SignalHandlerOptions

    with host_stop_signals() as stopping:
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
        nodes = []
        dependencies = None
        executor = None
        try:
            dependencies = build_dependencies()
            node = create_node(dependencies, nodes.append)
            if node not in nodes:
                nodes.append(node)
            if executor_factory is not None:
                executor = executor_factory(node)
            else:
                # The default business callback group serializes sync model
                # callers; a second worker serves independent control callback
                # groups, including mutually exclusive control RPC callbacks.
                # Models can be created after the first spin, including from
                # a business callback, so construction history cannot select it.
                executor = MultiThreadedExecutor(num_threads=2)
            executor.add_node(node)
            while not stopping.is_set() and rclpy.ok():
                try:
                    executor.spin_once(timeout_sec=0.1)
                    collect_node_runtime_transport(node, executor)
                except Exception as exc:
                    if callback_error is None:
                        raise
                    callback_error(node, exc)
        except KeyboardInterrupt:
            pass
        finally:
            with lifecycle_budget(HOST_SHUTDOWN_TIMEOUT_SEC):
                if executor is not None:
                    if request_tracker is not None:
                        for node in nodes:
                            _drain_requests(executor, request_tracker(node), HOST_SHUTDOWN_TIMEOUT_SEC)
                    for node in nodes:
                        executor.remove_node(node)
                    if drain_callbacks and not executor.shutdown(
                        timeout_sec=lifecycle_remaining(HOST_SHUTDOWN_TIMEOUT_SEC)
                    ):
                        raise TimeoutError("serving callbacks are still running; runtime owners retained")
                if dependencies is not None:
                    close_model_nodes(nodes, dependencies.providers)
                if executor is not None and not executor.shutdown(
                    timeout_sec=lifecycle_remaining(HOST_SHUTDOWN_TIMEOUT_SEC)
                ):
                    raise TimeoutError("serving executor shutdown is pending")
                rclpy.try_shutdown()
