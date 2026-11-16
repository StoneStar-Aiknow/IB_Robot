"""Resource-wide initialization barrier for common hosts, before SDK loading."""

import math
import threading
import time
from contextlib import contextmanager
from uuid import uuid4

from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from inference_manifest.runtime_target import MAX_RETRY_WINDOW_MS, validate_runtime_group_budget
from inference_service.unified_runtime.lifecycle import check_startup, lifecycle_remaining, wait_for_startup

from .runtime_protocol import ControlError, NotStartedControlError, RuntimeExecutionTarget, machine_clock_id
from .runtime_ros import _RemoteControlError, _reply, _request, _service_type


def validate_initialization_targets(targets, validated_manifest, runtime_options, *, adapter="model_runtime"):
    """Validate actual placement before any constructor can enter the SDK."""
    from inference_manifest.runtime_identity import (
        physical_device,
        runtime_composition_fingerprint,
        validate_runtime_device_visibility,
    )

    device = physical_device(
        validated_manifest, runtime_options, torch_device_index=runtime_options.get("torch_device_index")
    )
    # The public single-model creation path uses this adapter identity. Policy
    # stages and business adapters must validate their own composition when
    # their dedicated binding paths are introduced.
    composition = runtime_composition_fingerprint(adapter, runtime_options)
    for target in targets:
        if target["deployment_fingerprint"] != validated_manifest.fingerprint or target["physical_device"] != device:
            raise ValueError("managed runtime deployment or physical device mismatch")
        if target["composition_fingerprint"] != composition:
            raise ValueError(f"managed runtime composition mismatch (expected {adapter} and actual load options)")
        validate_runtime_device_visibility(device, target["target_id"])


def initialization_rollback_proven(error):
    """Only explicit rollback evidence permits releasing a failed load."""
    if isinstance(error, NotStartedControlError):
        return True
    code = getattr(error, "code", "")
    details = getattr(error, "details", {})
    return (code == "runtime_load_failed" and details.get("cleanup_complete") is True) or (
        code == "startup_failed" and details.get("cleanup_errors") == ()
    )


class InitializationSettlement:
    """Retain the exact receipt and its transport across a bounded waiter timeout."""

    def __init__(self, node, payload, replay, dispose, timeout, before=None):
        self.node, self.payload = node, payload
        self.replay, self.dispose, self.timeout = replay, dispose, timeout
        self.done = False
        self._replayed = False
        self.before = before
        self._lock = threading.Lock()

    def settle(self):
        from inference_service.unified_runtime.errors import DeadlineExceeded
        from inference_service.unified_runtime.lifecycle import lifecycle_deadline, lifecycle_lock, lifecycle_remaining

        try:
            with lifecycle_lock(self._lock):
                if not self.done:
                    lifecycle_deadline().check("initialization_settlement")
                    if self.before is not None:
                        self.before()
                        self.before = None
                    if not self._replayed:
                        self.replay(self.payload, time.monotonic() + lifecycle_remaining(self.timeout))
                        if (
                            self.payload.get("verb") == "finish_initialization"
                            and self.payload["clean"]
                            and not self.payload.get("acknowledged", False)
                        ):
                            # Reuse the existing receipt ACK convention. Persist
                            # the ACK payload before sending so a lost reply can
                            # be retried even after Global collects the receipt.
                            self.payload["acknowledged"] = True
                            self.replay(self.payload, time.monotonic() + lifecycle_remaining(self.timeout))
                        self._replayed = True
                    self.dispose()
                    self.done = True
                with _receipts_lock:
                    receipts = getattr(self.node, "_runtime_initialization_receipts", [])
                    if self in receipts:
                        receipts.remove(self)
        except (TimeoutError, DeadlineExceeded) as exc:
            raise ControlError("INITIALIZATION_CONTROL_TIMEOUT") from exc


_receipts_lock = threading.Lock()


def settle_initializations(node, *, token=None):
    """Retry retained initialization evidence before Close or provider shutdown."""
    with _receipts_lock:
        receipts = tuple(getattr(node, "_runtime_initialization_receipts", ()))
    errors = []
    for receipt in receipts:
        if token is None or receipt.payload["token"] == token:
            try:
                receipt.settle()
            except Exception as exc:
                errors.append(exc)
    if errors:
        raise errors[0]


@contextmanager
def runtime_initialization(
    node, targets, endpoint, *, boot_id, clean=None, retire=None, cleanup=None, timeout=60.0, token=None
):
    """Acquire before construction; replay receipts and retain uncertain SDK work.

    The short-lived control node can spin while the owning node is still in
    its constructor. It never executes business callbacks or starts Global.
    """
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("runtime initialization timeout must be finite and positive")
    targets = [RuntimeExecutionTarget(**target).to_dict() for target in targets]
    validate_runtime_group_budget(targets)
    control = executor = None
    token = token or str(uuid4())
    admitted = None
    settlement = None

    def retain(payload, before=None):
        receipt = InitializationSettlement(node, payload, replay, dispose, timeout, before)
        with _receipts_lock:
            if not hasattr(node, "_runtime_initialization_receipts"):
                node._runtime_initialization_receipts = []
            node._runtime_initialization_receipts.append(receipt)
        return receipt

    def dispose():
        nonlocal executor, control
        if executor is not None:
            if executor.shutdown() is False:
                raise ControlError("INITIALIZATION_CONTROL_SHUTDOWN_PENDING")
            executor = None
        if control is not None:
            control.destroy_node()
            control = None

    def call(payload, remaining):
        future = client.call_async(_request(payload))
        executor.spin_until_future_complete(future, timeout_sec=max(0.0, remaining))
        if not future.done():
            client.remove_pending_request(future)
            raise ControlError("INITIALIZATION_CONTROL_TIMEOUT")
        return _reply(future.result())

    def replay(payload, deadline):
        while True:
            remaining = lifecycle_remaining(max(0.0, deadline - time.monotonic()))
            if remaining <= 0:
                raise ControlError("INITIALIZATION_CONTROL_TIMEOUT")
            try:
                return call(payload, min(0.25, remaining))
            except _RemoteControlError:
                raise
            except ControlError:
                # A timeout does not prove that Global rejected the operation.
                # Reuse its identity instead of abandoning the resource owner.
                continue

    try:
        control = Node(f"runtime_initializer_{uuid4().hex}", context=node.context, use_global_arguments=False)
        executor = SingleThreadedExecutor(context=node.context)
        executor.add_node(control)
        client = control.create_client(_service_type(), endpoint)
        deadline = time.monotonic() + timeout
        # Bound first admission independently of the caller's total load budget.
        # Reuse this exact window for Begin retries and compensating Abort.
        accept_until_ns = time.monotonic_ns() + int(min(timeout, MAX_RETRY_WINDOW_MS / 1000) * 1000000000)
        if not wait_for_startup(lambda seconds: client.wait_for_service(timeout_sec=seconds), timeout):
            raise NotStartedControlError("GLOBAL_UNAVAILABLE")
        attempted = False
        try:
            while admitted is None:
                check_startup()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise NotStartedControlError("INITIALIZATION_ADMISSION_TIMEOUT")
                statuses = [
                    {
                        "target": target,
                        "boot_id": boot_id,
                        "clock_id": machine_clock_id(),
                        "epoch": "",
                        "generation": 1,
                        "sequence": 1,
                        "ready": False,
                        "clean": True,
                        "status_ns": time.monotonic_ns(),
                        "lifecycle_target_ids": [t["target_id"] for t in targets],
                    }
                    for target in targets
                ]
                try:
                    attempted = True
                    admitted = call(
                        {
                            "verb": "begin_initialization",
                            "statuses": statuses,
                            "token": token,
                            "accept_until_ns": accept_until_ns,
                        },
                        min(0.25, remaining),
                    )
                except ControlError as exc:
                    if exc.code not in {"RESOURCE_BUSY", "RESOURCE_INITIALIZING", "INITIALIZATION_CONTROL_TIMEOUT"}:
                        raise NotStartedControlError(exc.code) from exc
                    time.sleep(min(0.02, remaining))
            if time.monotonic() >= deadline:
                raise NotStartedControlError("INITIALIZATION_ADMISSION_TIMEOUT")
            check_startup()
        except BaseException:
            if attempted:
                # No constructor has run. Explicitly release any accepted Begin
                # and reject delayed copies, even if its epoch reply was lost.
                settlement = retain(
                    {
                        "verb": "abort_initialization",
                        "resource": targets[0]["hardware_resource_id"],
                        "token": token,
                        "clock_id": machine_clock_id(),
                        "accept_until_ns": accept_until_ns,
                    },
                )
                settlement.settle()
            raise
        proven = False
        try:
            yield
            proven = clean() if clean is not None else True
        except BaseException as exc:
            proven = clean() if clean is not None else initialization_rollback_proven(exc)
            raise
        finally:
            payload = {
                "verb": "finish_initialization",
                "epoch": admitted["epoch"],
                "resource": admitted["resource"],
                "token": token,
                "clean": proven,
                "retire": bool(proven and retire is not None and retire()),
            }
            if not proven and cleanup is not None:
                # Keep the original barrier until cleanup can actually finish.
                # Do not publish an immutable terminal UNKNOWN receipt and then
                # attempt to rewrite it as clean on a later retry.
                def retry_cleanup():
                    replay({**payload, "verb": "check_initialization"}, time.monotonic() + timeout)
                    cleanup()

                payload.update(clean=True, retire=bool(retire is not None and retire()))
                settlement = retain(payload, retry_cleanup)
            else:
                settlement = retain(payload)
                settlement.settle()

    finally:
        if settlement is None:
            dispose()
