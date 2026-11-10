"""Initialization is exclusive with execution and cannot erase unknown work."""

import threading
import time
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from inference_service.scheduler.runtime_controller import RuntimeExecutionController
from inference_service.scheduler.runtime_initialization import runtime_initialization
from inference_service.scheduler.runtime_protocol import ControlError, machine_clock_id
from inference_service.scheduler.runtime_ros import GlobalRuntimeEndpoint
from inference_service.scheduler.runtime_scheduler import RuntimeScheduler
from inference_service.unified_runtime import ExecutionContext, ModelRequest, ModelRuntimeHandle

from .runtime_scheduler_helpers import initialization_target as target


def status(t, boot=None):
    return {
        "target": t.to_dict(),
        "boot_id": boot or str(uuid4()),
        "clock_id": machine_clock_id(),
        "epoch": "",
        "generation": 1,
        "sequence": 1,
        "ready": False,
        "clean": True,
        "status_ns": time.monotonic_ns(),
    }


def test_initialization_is_exclusive_and_uncertain_cleanup_cannot_be_recovered_away():
    core = RuntimeScheduler([], resources={"cpu": {"physical_device": "local/cpu"}})
    first, second = target(), target("other")
    token = str(uuid4())
    first_status = status(first)
    core.begin_initialization([first_status], token)
    assert not core.resource_ready("cpu")
    # Retransmission cannot acquire twice or release another caller's barrier.
    core.begin_initialization([first_status], token)
    with pytest.raises(ControlError, match="RESOURCE_INITIALIZING"):
        core.begin_initialization([status(second)], str(uuid4()))
    with pytest.raises(ControlError, match="IDENTITY_MISMATCH"):
        core.finish_initialization("cpu", str(uuid4()), clean=True)
    core.finish_initialization("cpu", token, clean=False)
    with pytest.raises(ControlError, match="INITIALIZATION_EVIDENCE_PENDING"):
        core.begin_recovery("cpu", ())
    with pytest.raises(ControlError, match="INITIALIZATION_EVIDENCE_PENDING"):
        core.finish_recovery("cpu")
    assert not core.resource_ready("cpu")


def test_rejected_target_never_reserves_or_loads():
    core = RuntimeScheduler([], resources={"cpu": {"physical_device": "local/cpu"}})
    invalid = status(target())
    invalid["target"]["physical_device"] = "local/cuda:0"
    with pytest.raises(ControlError, match="RESOURCE_CONFIGURATION_MISMATCH"):
        core.begin_initialization([invalid], str(uuid4()))
    assert core._initializations == {}
    assert core.targets == {}


@pytest.mark.parametrize("field", ["physical_device", "deployment_fingerprint", "composition_fingerprint", "clock_id"])
def test_begin_replay_preserves_descriptor_and_clock(field):
    core = RuntimeScheduler([], resources={"cpu": {"physical_device": "local/cpu"}})
    original = status(target())
    token = str(uuid4())
    core.begin_initialization([original], token)
    replay = deepcopy(original)
    replay["sequence"] += 1
    replay["status_ns"] = time.monotonic_ns()
    assert core.begin_initialization([replay], token)["state"] == "initializing"
    if field == "clock_id":
        replay[field] = str(uuid4())
    else:
        replay["target"][field] = "local/cuda:0" if field == "physical_device" else "changed"
    with pytest.raises(ControlError, match="INITIALIZATION_IDENTITY_MISMATCH"):
        core.begin_initialization([replay], token)
    assert core.check_initialization("cpu", token)["state"] == "initializing"
    assert core.begin_initialization([original], token)["state"] == "initializing"
    core.finish_initialization("cpu", token, clean=True)


def test_finished_initialization_replays_without_releasing_next_owner():
    core = RuntimeScheduler([], resources={"cpu": {"physical_device": "local/cpu"}})
    token, next_token = str(uuid4()), str(uuid4())
    first = status(target())
    core.begin_initialization([first], token)
    receipt = core.finish_initialization("cpu", token, clean=True)
    core.begin_initialization([status(target("next"))], next_token)
    assert core.finish_initialization("cpu", token, clean=True) == receipt
    assert not core.resource_ready("cpu")
    with pytest.raises(ControlError, match="ALREADY_FINISHED"):
        core.begin_initialization([first], token)
    core.finish_initialization("cpu", next_token, clean=True)
    assert core._initializations == {}


def test_rejected_recovery_does_not_block_missing_policy_initialization():
    from dataclasses import replace

    missing = replace(target("missing_policy"), interface="policy", operation="policy")
    core = RuntimeScheduler([missing], resources={"cpu": {"physical_device": "local/cpu"}})
    live, token = status(target("loaded")), str(uuid4())
    core.begin_initialization([live], token)
    core.finish_initialization("cpu", token, clean=True)
    with pytest.raises(ControlError, match="RECOVERY_HOST_UNREACHABLE"):
        core.begin_recovery("cpu", core.recovery_targets("cpu"))
    next_token = str(uuid4())
    assert core.begin_initialization([status(missing)], next_token)["state"] == "initializing"
    core.finish_initialization("cpu", next_token, clean=True)


def test_finish_receipt_survives_collection_before_caller_consumption():
    clock = [time.monotonic_ns()]
    core = RuntimeScheduler([], resources={"cpu": {"physical_device": "local/cpu"}}, now_ns=lambda: clock[0])
    token = str(uuid4())
    initial = status(target())
    initial["status_ns"] = clock[0]
    core.begin_initialization([initial], token)
    receipt = core.finish_initialization("cpu", token, clean=True)
    clock[0] += 301_000_000_000
    core.collect()
    assert core.finish_initialization("cpu", token, clean=True) == receipt
    core.finish_initialization("cpu", token, clean=True, acknowledged=True)
    core.collect()
    assert ("cpu", token) not in core._initialization_receipts
    # A lost ACK can be replayed, but an expired Finish cannot create authority.
    assert core.finish_initialization("cpu", token, clean=True, acknowledged=True)["state"] == "acknowledged"
    with pytest.raises(ControlError, match="INITIALIZATION_IDENTITY_MISMATCH"):
        core.finish_initialization("cpu", token, clean=True)


@pytest.mark.parametrize("begin_arrived", [False, True])
@pytest.mark.parametrize("configured", [False, True])
def test_abort_rejects_delayed_initialization_even_after_receipt_collection(begin_arrived, configured):
    now = time.monotonic_ns()
    core = RuntimeScheduler(
        [target()] if configured else [], resources={"cpu": {"physical_device": "local/cpu"}}, now_ns=lambda: now
    )
    first, token, expires = status(target()), str(uuid4()), now + 1000000000
    first["status_ns"] = now
    if begin_arrived:
        core.begin_initialization([first], token, accept_until_ns=expires)
    receipt = core.abort_initialization("cpu", token, accept_until_ns=expires)
    assert core.abort_initialization("cpu", token, accept_until_ns=expires) == receipt
    with pytest.raises(ControlError, match="CANCELED"):
        core.begin_initialization([first], token, accept_until_ns=expires)
    assert core._initializations == {}
    assert core.shutdown_pending() == []
    now += 301000000000
    core.collect()
    assert core.shutdown_pending() == []
    with pytest.raises(ControlError, match="ADMISSION_TIMEOUT"):
        core.begin_initialization([first], token, accept_until_ns=expires)


def test_abort_cannot_release_uncertain_model_loading():
    core = RuntimeScheduler([], resources={"cpu": {"physical_device": "local/cpu"}})
    token, expires = str(uuid4()), time.monotonic_ns() + 1000000000
    core.begin_initialization([status(target())], token, accept_until_ns=expires)
    receipt = core.finish_initialization("cpu", token, clean=False)
    assert core.finish_initialization("cpu", token, clean=False) == receipt
    with pytest.raises(ControlError, match="EVIDENCE_PENDING"):
        core.abort_initialization("cpu", token, accept_until_ns=expires)
    with pytest.raises(ControlError, match="INVALID_INITIALIZATION_EVIDENCE"):
        core.finish_initialization("cpu", token, clean=True)
    assert not core.resource_ready("cpu")


@pytest.mark.parametrize("loss", ["begin", "finish", "all_begin"])
def test_initializer_recovers_lost_replies_without_loading_twice_or_leaking_barrier(monkeypatch, loss):
    from inference_service.scheduler import runtime_initialization as initialization_module

    # Drop a receipt only after the real ROS service has applied the operation.
    original_reply = initialization_module._reply
    receipts = []

    def reply(response):
        result = original_reply(response)
        state = result.get("state")
        first = state not in receipts
        receipts.append(state)
        if (state == "initializing" and (loss == "all_begin" or (loss == "begin" and first))) or (
            state == "initialized" and loss == "finish" and first
        ):
            raise ControlError("INITIALIZATION_CONTROL_TIMEOUT")
        return result

    monkeypatch.setattr(initialization_module, "_reply", reply)
    rclpy.init()
    global_node, host = Node("receipt_global"), Node("receipt_host")
    endpoint = GlobalRuntimeEndpoint(global_node, [], resources={"cpu": {"physical_device": "local/cpu"}})
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(global_node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    loaded = []

    def initialize():
        with runtime_initialization(host, [target().to_dict()], endpoint.endpoint, boot_id=str(uuid4()), timeout=1):
            loaded.append(True)

    try:
        if loss == "all_begin":
            with pytest.raises(ControlError, match="ADMISSION_TIMEOUT"):
                initialize()
            assert loaded == [] and "aborted" in receipts
        else:
            initialize()
            assert loaded == [True]
            assert receipts.count("initializing" if loss == "begin" else "initialized") == 2
        assert endpoint.core._initializations == {}
    finally:
        executor.shutdown(timeout_sec=3)
        spin.join(3)
        assert not spin.is_alive()
        host.destroy_node()
        global_node.destroy_node()
        rclpy.shutdown()


def test_initialization_waits_for_execution_and_pins_membership_past_status_expiry(simulated_runtime_construction):
    t = target("running")
    handle = ModelRuntimeHandle(executor=SimpleNamespace(execute=lambda *args: {}))
    handle.load()
    host = RuntimeExecutionController(t, handle)
    core = RuntimeScheduler([t], resources={"cpu": {"physical_device": "local/cpu"}})
    try:
        host.accept_epoch(core.register(host.status())["epoch"])
        core.register(host.status())
        submission = host.prepare(ModelRequest(), ExecutionContext.create("inference", timeout=5))
        core.submit(submission)
        assert core.dispatchable()
        with pytest.raises(ControlError, match="RESOURCE_BUSY"):
            core.begin_initialization([status(target("new"))], str(uuid4()))
        core.report({"submission": submission.to_dict(), "state": "completed"})
        token = str(uuid4())
        core.begin_initialization([status(target("new"))], token)
        core.now_ns = lambda: time.monotonic_ns() + 100_000_000_000
        core.collect()
        assert "perception/new" in core.targets
        assert not core._inactive("perception/new")
        assert not core.dispatchable()
        core.finish_initialization("cpu", token, clean=True)
    finally:
        host.shutdown()
        handle.close()


def test_online_initialization_requires_pausing_and_draining_same_device_traffic(simulated_runtime_construction):
    from .runtime_scheduler_helpers import connect, controller, prepare

    active = target("serving")
    core = RuntimeScheduler([active], resources={"cpu": {"physical_device": "local/cpu"}})
    host, _ = controller(active)
    candidate, token = status(target("new_model")), str(uuid4())
    try:
        connect(core, host)
        current = prepare(host)
        core.submit(current)
        for _ in range(3):
            assert core.dispatchable()
            following = prepare(host)
            core.submit(following)
            completed = host.execute(current, core.epoch)
            if completed["state"] not in {"completed", "not_started"}:
                host.completion(current).result(timeout=1)
            core.terminal_report(host.query(current, core.epoch))
            host.acknowledge(current, core.epoch)
            with pytest.raises(ControlError, match="RESOURCE_BUSY"):
                core.begin_initialization([candidate], token)
            assert core.shutdown_pending()  # Accepted inference remains owned.
            assert not core._initializations
            current = following
        # The caller now stops replenishing the queue and drains accepted work.
        assert core.dispatchable()
        host.execute(current, core.epoch)
        host.completion(current).result(timeout=1)
        core.terminal_report(host.query(current, core.epoch))
        host.acknowledge(current, core.epoch)
        assert core.begin_initialization([candidate], token)["state"] == "initializing"
        core.finish_initialization("cpu", token, clean=True, retire=True)
    finally:
        host.shutdown()
        host.handle.close()


@pytest.mark.parametrize("load_timeout", [2, 600])
def test_real_ros_initializers_serialize_and_do_not_load_when_global_is_fenced(load_timeout):
    rclpy.init()
    global_node, host = Node("load_global"), Node("load_host")
    endpoint = GlobalRuntimeEndpoint(global_node, [], resources={"cpu": {"physical_device": "local/cpu"}})
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(global_node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    entered, release = threading.Event(), threading.Event()
    failures = []

    def load_first():
        try:
            with runtime_initialization(host, [target().to_dict()], endpoint.endpoint, boot_id=str(uuid4()), timeout=2):
                entered.set()
                assert release.wait(3)
        except BaseException as exc:
            failures.append(exc)

    loading = threading.Thread(target=load_first)
    loading.start()
    try:
        assert entered.wait(2)
        with (
            pytest.raises(ControlError, match="INITIALIZATION_ADMISSION_TIMEOUT"),
            runtime_initialization(
                host, [target("second").to_dict()], endpoint.endpoint, boot_id=str(uuid4()), timeout=0.1
            ),
        ):
            pytest.fail("second SDK load must not start")
        release.set()
        loading.join(3)
        assert not loading.is_alive() and failures == []
        with runtime_initialization(
            host, [target("second").to_dict()], endpoint.endpoint, boot_id=str(uuid4()), timeout=load_timeout
        ):
            assert not endpoint.core.resource_ready("cpu")
            receipt = next(r for r in endpoint.core._initialization_receipts.values() if r["state"] == "initializing")
            assert 0 < receipt["accept_until_ns"] - time.monotonic_ns() <= min(load_timeout, 300) * 1000000000
        # CPU-side validation failures prove no SDK work began, even for a
        # required host that propagates the startup error.
        with (
            pytest.raises(ValueError, match="invalid manifest"),
            runtime_initialization(
                host,
                [target("invalid").to_dict()],
                endpoint.endpoint,
                boot_id=str(uuid4()),
                clean=lambda: True,
                timeout=2,
            ),
        ):
            raise ValueError("invalid manifest")
        assert endpoint.core._initializations == {}
        endpoint.core.fence_resource("cpu")
        with (
            pytest.raises(ControlError, match="RESOURCE_NOT_READY"),
            runtime_initialization(
                host, [target("third").to_dict()], endpoint.endpoint, boot_id=str(uuid4()), timeout=2
            ),
        ):
            pytest.fail("quarantined resource must not load")
    finally:
        release.set()
        loading.join(3)
        executor.shutdown(timeout_sec=3)
        spin.join(3)
        assert not spin.is_alive()
        host.destroy_node()
        global_node.destroy_node()
        rclpy.shutdown()


def test_clean_failed_initializer_retires_members_and_replays_without_changing_identity():
    core = RuntimeScheduler([], resources={"cpu": {"physical_device": "local/cpu"}})
    participant = target()
    initial = status(participant)
    token = str(uuid4())
    core.begin_initialization([initial], token)
    with pytest.raises(ControlError, match="INVALID_INITIALIZATION_EVIDENCE"):
        core.finish_initialization("cpu", token, clean=False, retire=True)
    assert "cpu" in core._initializations
    receipt = core.finish_initialization("cpu", token, clean=True, retire=True)
    assert core.host_retired(participant.target_id, initial["boot_id"])
    assert not core._initializations
    assert core.finish_initialization("cpu", token, clean=True, retire=True) == receipt
    with pytest.raises(ControlError, match="INVALID_INITIALIZATION_EVIDENCE"):
        core.finish_initialization("cpu", token, clean=True, retire=False)


@pytest.mark.parametrize("verb", ["begin", "abort"])
def test_initialization_rejects_far_future_window_without_consuming_capacity(verb):
    core = RuntimeScheduler([], resources={"cpu": {"physical_device": "local/cpu"}})
    with pytest.raises(ControlError, match="INITIALIZATION_ADMISSION_TIMEOUT"):
        if verb == "begin":
            core.begin_initialization([status(target())], str(uuid4()), accept_until_ns=2**63 - 1)
        else:
            core.abort_initialization("cpu", str(uuid4()), accept_until_ns=2**63 - 1)
    assert not core._initialization_receipts
    core.begin_initialization([status(target())], str(uuid4()))


def test_abort_rejects_unknown_resource_and_retains_late_legal_replay():
    clock = [time.monotonic_ns()]
    core = RuntimeScheduler([], resources={"cpu": {"physical_device": "local/cpu"}}, now_ns=lambda: clock[0])
    expiry = clock[0] + 1_000_000
    with pytest.raises(ControlError, match="UNKNOWN_RESOURCE"):
        core.abort_initialization("missing", str(uuid4()), accept_until_ns=expiry)
    assert not core._initialization_receipts
    token = str(uuid4())
    first = core.abort_initialization("cpu", token, accept_until_ns=expiry)
    clock[0] = expiry + 1
    assert core.abort_initialization("cpu", token, accept_until_ns=expiry) == first


@pytest.mark.parametrize("busy", ["initializing", "queued", "running"])
def test_busy_initialization_does_not_prepare_profiles(monkeypatch, busy):
    from dataclasses import replace

    from .runtime_scheduler_helpers import connect, controller, prepare

    first = target("active")
    core = RuntimeScheduler([first], resources={"cpu": {"physical_device": "local/cpu"}})
    host, _ = controller(first)
    candidate = replace(
        target("candidate"),
        predictive_admission=True,
        profile_path="must-not-read.json",
        runtime_version="v1",
        hardware_profile_fingerprint="hw",
    )
    try:
        if busy == "initializing":
            initial = status(first)
            token = str(uuid4())
            core.begin_initialization([initial], token)
        else:
            connect(core, host)
            submitted = prepare(host)
            core.submit(submitted)
            if busy == "running":
                assert core.dispatchable()
        monkeypatch.setattr(core, "_prepare_profiles", lambda *_: pytest.fail("busy retry loaded profiles"))
        for _ in range(5):
            with pytest.raises(ControlError, match="RESOURCE_INITIALIZING|RESOURCE_BUSY"):
                core.begin_initialization([status(candidate)], str(uuid4()))
        if busy == "initializing":
            assert core.begin_initialization([initial], token)["state"] == "initializing"
    finally:
        host.shutdown()


@pytest.mark.parametrize("group", [False, True])
def test_initialization_and_retirement_do_not_copy_operation_history(group):
    from dataclasses import replace

    copies = []

    class Ledger(dict):
        def copy(self):
            copies.append(len(self))
            return Ledger(self)

    mode = "ascend_concurrent" if group else "serial"
    core = RuntimeScheduler(
        [], resources={"cpu": {"physical_device": "local/cpu", "mode": mode, "verified_concurrency": True}}
    )
    ledger = Ledger(core._records)
    core._records = ledger
    targets = [target()]
    if group:
        targets = [
            replace(
                target(name), interface="policy", execution_kind="stage", lifecycle_group="model", resource_mode=mode
            )
            for name in ("preprocess", "execute")
        ]
    boot = str(uuid4())
    statuses = [status(t, boot) for t in targets]
    if group:
        for item in statuses:
            item["lifecycle_target_ids"] = [t.target_id for t in targets]
    token = str(uuid4())
    core.begin_initialization(statuses, token)
    assert not copies
    assert set(core.targets) == {t.target_id for t in targets}
    assert core.check_initialization("cpu", token)["state"] == "initializing"
    core.finish_initialization("cpu", token, clean=True, retire=True)
    assert not core._initializations
    assert not copies
