import time
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from inference_service.scheduler.runtime_controller import RuntimeExecutionController
from inference_service.scheduler.runtime_protocol import (
    ControlError,
    Submission,
    machine_clock_id,
)
from inference_service.scheduler.runtime_scheduler import RuntimeScheduler
from inference_service.unified_runtime import Deadline, ExecutionContext, ModelRequest

from .runtime_scheduler_helpers import attested_resources, connect, controller, prepare, target


@pytest.mark.parametrize("mode", ["serial", "ascend_concurrent"])
@pytest.mark.parametrize("policy", ["fifo", "edf"])
@pytest.mark.parametrize("work", ["reset", "drain", "close"])
@pytest.mark.parametrize("same_target", [False, True])
def test_waiting_maintenance_reserves_resource_until_active_work_finishes(mode, policy, work, same_target):
    targets = [
        target("maintenance", resource_mode=mode),
        target("busy", resource_mode=mode, max_in_flight=2 if mode == "ascend_concurrent" else 1),
        target("independent", resource="cpu"),
    ]
    core = RuntimeScheduler(targets, policy=policy, resources=attested_resources(*targets))
    hosts = [controller(t)[0] for t in targets]
    try:
        for host in hosts:
            connect(core, host)
        active_index = 0 if same_target else 1
        active = prepare(hosts[active_index])
        core.submit(active)
        assert core.dispatchable() == [(targets[active_index], active)]
        # Even higher-priority arrivals must let an admitted barrier drain.
        maintenance = hosts[0].prepare(
            ModelRequest(), ExecutionContext("maintenance", Deadline.after(2)), work=work, priority=3
        )
        core.submit(maintenance)
        queued = prepare(hosts[1], priority=0)
        core.submit(queued)
        independent = prepare(hosts[2])
        core.submit(independent)
        assert core.dispatchable() == [(targets[2], independent)]
        for _ in range(5):
            assert core.dispatchable() == []
            assert core.query(maintenance)["state"] == "queued"
        core.report(hosts[active_index].cancel(active, core.epoch))
        assert core.dispatchable() == [(targets[0], maintenance)]
        assert core.query(queued)["state"] == "queued"
        core.report(hosts[0].cancel(maintenance, core.epoch))
        assert core.dispatchable() == [(targets[1], queued)]
    finally:
        for host in hosts:
            host.shutdown()
            host.handle.close()


@pytest.mark.parametrize("mode", ["serial", "ascend_concurrent"])
@pytest.mark.parametrize("work", ["reset", "drain", "close"])
@pytest.mark.parametrize("priority", [0, 2])
def test_waiting_maintenance_drains_resource_before_new_execute(mode, work, priority):
    targets = [
        target("maintenance", resource_mode=mode),
        target("busy", resource_mode=mode, max_in_flight=2),
        target("unrelated", resource="cpu"),
    ]
    core = RuntimeScheduler(targets, resources=attested_resources(*targets))
    hosts = [controller(t)[0] for t in targets]
    try:
        for host in hosts:
            connect(core, host)
        active = prepare(hosts[1])
        core.submit(active)
        assert core.dispatchable() == [(targets[1], active)]
        maintenance = hosts[0].prepare(
            ModelRequest(), ExecutionContext("maintenance", Deadline.after(2)), work=work, priority=priority
        )
        core.submit(maintenance)
        queued = prepare(hosts[1])
        core.submit(queued)
        unrelated = prepare(hosts[2])
        core.submit(unrelated)
        assert core.dispatchable() == [(targets[2], unrelated)]
        for _ in range(3):
            assert not core.dispatchable()
            assert core.query(queued)["state"] == "queued"
        core.report({"submission": active.to_dict(), "state": "completed"})
        assert core.dispatchable() == [(targets[0], maintenance)]
        assert core.query(queued)["state"] == "queued"
        core.report({"submission": maintenance.to_dict(), "state": "completed"})
        assert core.dispatchable() == [(targets[1], queued)]
    finally:
        for host in hosts:
            host.shutdown()


@pytest.mark.parametrize("release", ["cancel", "expire"])
def test_withdrawn_maintenance_releases_resource_reservation(release):
    targets = [target(name, resource_mode="ascend_concurrent") for name in ("maintenance", "busy", "new")]
    core = RuntimeScheduler(targets, resources=attested_resources(*targets))
    hosts = [controller(t)[0] for t in targets]
    try:
        for host in hosts:
            connect(core, host)
        active = prepare(hosts[1])
        core.submit(active)
        assert core.dispatchable() == [(targets[1], active)]
        maintenance = prepare(hosts[0], work="reset", snapshot_valid_until_ns=core.now_ns() + 100_000_000)
        core.submit(maintenance)
        queued = prepare(hosts[2])
        core.submit(queued)
        assert not core.dispatchable()
        if release == "cancel":
            core.cancel(maintenance)
        else:
            core.now_ns = lambda: maintenance.snapshot_valid_until_ns
        assert core.dispatchable() == [(targets[2], queued)]
        assert core.query(maintenance)["state"] == "not_started"
    finally:
        for host in hosts:
            host.shutdown()
            host.handle.close()


@pytest.mark.parametrize("blocker", ["capacity", "serial", "maintenance", "control_waiting"])
def test_busy_requests_skip_readiness_scan_but_other_resources_dispatch(monkeypatch, blocker):
    mode = "serial" if blocker == "serial" else "ascend_concurrent"
    targets = [
        target("first", resource_mode=mode),
        target("second", resource_mode=mode),
        target("independent", resource="cpu"),
    ]
    core = RuntimeScheduler(targets, resources=attested_resources(*targets))
    hosts = [controller(t)[0] for t in targets]
    try:
        for host in hosts:
            connect(core, host)
        active = prepare(hosts[0], work="reset" if blocker == "maintenance" else "execute")
        core.submit(active)
        assert core.dispatchable() == [(targets[0], active)]
        waiting_host = hosts[0] if blocker == "capacity" else hosts[1]
        waiting = prepare(waiting_host, work="reset" if blocker == "control_waiting" else "execute")
        core.submit(waiting)
        independent = prepare(hosts[2])
        core.submit(independent)
        checked = []
        ready = core.resource_ready

        def resource_ready(resource, **kwargs):
            checked.append(resource)
            return ready(resource, **kwargs)

        monkeypatch.setattr(core, "resource_ready", resource_ready)
        assert core.dispatchable() == [(targets[2], independent)]
        assert checked == ["cpu"]
        assert core.query(waiting)["state"] == "queued"
    finally:
        for host in hosts:
            host.shutdown()


@pytest.mark.parametrize("mode,predictive", [("ascend_concurrent", False), ("serial", True)])
def test_unattested_resource_modes_are_rejected_before_profile_loading(mode, predictive):
    t = target(
        resource_mode=mode,
        predictive_admission=predictive,
        hardware_profile_fingerprint="hardware",
        runtime_version="test",
    )
    with pytest.raises(ControlError, match="RESOURCE_ATTESTATION_REQUIRED"):
        RuntimeScheduler([t])
    with pytest.raises(ControlError, match="UNKNOWN_RESOURCE"):
        RuntimeScheduler([t], resources={})
    resources = attested_resources(t)
    if predictive:
        resources[t.hardware_resource_id]["managed_coverage"] = False
        with pytest.raises(ControlError, match="PREDICTIVE_COVERAGE_REQUIRED"):
            RuntimeScheduler([t], resources=resources)
    else:
        resources[t.hardware_resource_id]["verified_concurrency"] = False
        with pytest.raises(ValueError, match="unverified"):
            RuntimeScheduler([t], resources=resources)


@pytest.mark.parametrize("ordinary_state", ["queued", "completed"])
def test_close_has_a_bounded_slot_even_when_ordinary_capacity_is_full(ordinary_state):
    t = target(max_records=1, max_pending_requests=1)
    core = RuntimeScheduler([t])
    host, _ = controller(t)
    connect(core, host)
    try:
        sub = prepare(host)
        core.submit(sub)
        if ordinary_state == "completed":
            core.dispatchable()
            host.execute(sub, core.epoch)
            host.completion(sub).result(timeout=1)
            core.report(host.query(sub, core.epoch))
            host.result(sub)
            host.acknowledge(sub, core.epoch)
            core.acknowledge_host(sub)
        with pytest.raises(ControlError, match="REQUEST_CAPACITY"):
            prepare(host)
        close = host.prepare(ModelRequest(), ExecutionContext("close-full-ledger", Deadline.after(2)), work="close")
        core.submit(close)
        assert len(host._records) == len(core._records) == 2
        with pytest.raises(ControlError, match="REQUEST_CAPACITY"):
            host.prepare(ModelRequest(), ExecutionContext("extra-close", Deadline.after(2)), work="close")
        # Settle the lifecycle barrier's rejected queued request before Close.
        if ordinary_state == "queued":
            core.report(host.cancel(sub, core.epoch))
        assert core.dispatchable() == [(t, close)]
        host.execute(close, core.epoch)
        host.completion(close).result(timeout=1)
        assert host.handle.closed and host.query(close, core.epoch)["state"] == "completed"
    finally:
        host.shutdown()


@pytest.mark.parametrize("quota", [128, 1024])
def test_close_reserves_only_empty_control_bytes(quota):
    from inference_service.scheduler.runtime_controller import input_bytes

    t = target(max_retained_input_bytes=quota)
    host, _ = controller(t)
    core = RuntimeScheduler([t])
    connect(core, host)
    try:
        ordinary = None
        if quota == 1024:
            request = ModelRequest({"x": np.zeros(508, dtype=np.uint8)})
            assert input_bytes(request) == quota
            ordinary = host.prepare(request, ExecutionContext("full", Deadline.after(2)))
            core.submit(ordinary)
        with pytest.raises(ControlError, match="INPUT_CAPACITY"):
            host.prepare(ModelRequest(), ExecutionContext("ordinary", Deadline.after(2)))
        with pytest.raises(ControlError, match="CLOSE_PAYLOAD_UNSUPPORTED"):
            host.prepare(ModelRequest({"x": 1}), ExecutionContext("payload-close"), work="close")
        close = host.prepare(ModelRequest(), ExecutionContext("close", Deadline.after(2)), work="close")
        assert host._input_bytes <= quota + input_bytes(ModelRequest())
        with pytest.raises(ControlError, match="INPUT_CAPACITY"):
            host.prepare(ModelRequest(), ExecutionContext("duplicate-close"), work="close")
        core.submit(close)
        if ordinary is not None:
            core.report(host.cancel(ordinary, core.epoch))
        assert core.dispatchable() == [(t, close)]
        host.execute(close, core.epoch)
        host.completion(close).result(timeout=1)
        assert host.handle.closed and host._input_bytes == 0
    finally:
        host.shutdown()
        host.handle.close()


@pytest.mark.parametrize("field", ["required", "predictive_admission", "template_only"])
@pytest.mark.parametrize("value", ["false", 0, None])
def test_target_booleans_are_strict(field, value):
    with pytest.raises(ValueError, match=field):
        target(**{field: value})


@pytest.mark.parametrize("field", ["interface", "model_type", "operation", "runtime_version", "profile_path"])
def test_target_descriptor_rejects_unbounded_or_nonstring_fields(field):
    t = target()
    for value in (None, 123, "x" * 40000):
        with pytest.raises(ValueError, match=field):
            replace(t, **{field: value})


def test_target_json_escaping_cannot_overflow_heartbeat():
    from inference_service.scheduler.runtime_protocol import encode_control

    with pytest.raises(ValueError, match="descriptor budget"):
        target(profile_observation_inputs=tuple(f"{i}" + "\x01" * 250 for i in range(64)))
    host, _ = controller(target(profile_observation_inputs=("a." * 120,)))
    try:
        assert len(encode_control({"verb": "register", "status": host.status()}).encode()) < 32768
    finally:
        host.shutdown()


def test_fixed_catalogue_rejects_ambiguous_host_service():
    first, second = target("first"), target("second")
    with pytest.raises(ValueError, match="duplicate runtime host endpoint"):
        RuntimeScheduler([first, replace(second, endpoint=first.endpoint)])


@pytest.mark.parametrize("stale", [0, -1, True, 1.5, 50_000_000, 100_000_000, 150_000_000, 200_000_000])
def test_scheduler_rejects_invalid_status_timeout(stale):
    with pytest.raises(ValueError, match="stale timeout"):
        RuntimeScheduler([], stale_ns=stale)


@pytest.mark.parametrize("epoch", [None, False, 0, [], {}, "not-a-uuid", "A" * 32])
def test_invalid_epoch_cannot_fence_a_healthy_resource(epoch):
    t = target()
    host, _ = controller(t)
    core = RuntimeScheduler([t])
    try:
        connect(core, host)
        status = host.status()
        with pytest.raises((ValueError, ControlError), match="INVALID_HOST_STATUS|canonical UUID|UUID string"):
            core.register({**status, "epoch": epoch})
        # The rejected heartbeat must not consume the sequence or isolate the device.
        core.register(status)
        sub = prepare(host)
        core.submit(sub)
        assert core.dispatchable() == [(t, sub)]
    finally:
        host.shutdown()


@pytest.mark.parametrize("work", ["reset", "drain", "close"])
def test_cancel_before_submit_does_not_apply_lifecycle_barrier(work):
    t = target()
    host, _ = controller(t)
    core = RuntimeScheduler([t])
    try:
        connect(core, host)
        ordinary = prepare(host)
        core.submit(ordinary)
        maintenance = host.prepare(ModelRequest(), ExecutionContext("maintenance", Deadline.after(2)), work=work)
        cancelled = core.cancel(maintenance)
        assert cancelled["state"] == "not_started" and cancelled["error"] == "CANCELLED"
        assert core.query(ordinary)["state"] == "queued"
        assert core.cancel(maintenance) == core.submit(maintenance) == cancelled
        with pytest.raises(ControlError, match="OPERATION_CONFLICT"):
            core.cancel(replace(maintenance, digest="f" * 64))
        assert core.dispatchable() == [(t, ordinary)]
    finally:
        host.shutdown()


@pytest.mark.parametrize("with_resources", [False, True])
def test_heartbeat_freshness_uses_send_time_with_or_without_resource_descriptors(with_resources):
    now = [time.monotonic_ns()]
    t = target()
    core = RuntimeScheduler(
        [t], now_ns=lambda: now[0], resources={"gpu": {"physical_device": "local/gpu"}} if with_resources else None
    )
    host, _ = controller(t, now_ns=lambda: now[0])
    try:
        connect(core, host)
        delayed = host.status()
        now[0] += core.stale_ns - 1
        core.register(delayed)
        now[0] += 2
        assert not core.resource_ready("gpu")
        delayed["sequence"] += 1
        with pytest.raises(ControlError, match="STALE_STATUS"):
            core.register(delayed)
        assert not core.resource_ready("gpu")
    finally:
        host.shutdown()


@pytest.mark.parametrize("predictive", [False, True])
def test_ticks_skip_settled_history_but_preserve_replay_expiry_and_conflicting_evidence(tmp_path, predictive):
    import json

    from inference_service.scheduler.runtime_controller import freeze_inputs
    from inference_service.scheduler.runtime_profiles import RuntimeProfileRegistry
    from inference_service.scheduler.runtime_ros import GlobalRuntimeEndpoint

    from .runtime_scheduler_helpers import FakeNode

    class CountedRecords(dict):
        visits = 0

        def values(self):
            for value in super().values():
                self.visits += 1
                yield value

        def items(self):
            for item in super().items():
                self.visits += 1
                yield item

    profile = tmp_path / "profile.json"
    t = target(
        max_pending_requests=512,
        predictive_admission=predictive,
        profile_path=str(profile),
        hardware_profile_fingerprint="hardware",
        runtime_version="mock",
    )
    if predictive:
        _, _, _, bucket = freeze_inputs(ModelRequest({"image": np.zeros((2, 3), dtype=np.uint8)}), 1024)
        profile.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "scope": "runtime_execution",
                    "identity": RuntimeProfileRegistry.identity(t),
                    "measured_at": time.time(),
                    "buckets": [
                        {"work": "execute", "input_bucket": bucket, "samples": 10000, "upper_bound_ns": 1000000}
                    ],
                }
            )
        )
    endpoint = GlobalRuntimeEndpoint(FakeNode({}), [t], resources=attested_resources(t))
    core = endpoint.core
    now = [time.monotonic_ns()]
    core.now_ns = lambda: now[0]
    host, _ = controller(t, now_ns=lambda: now[0])
    try:
        connect(core, host)
        template = prepare(host)
        history = []
        for _ in range(256):
            sub = replace(template, operation_id=str(uuid4()), request_ref=str(uuid4()))
            core.submit(sub)
            assert core.dispatchable() == [(t, sub)]
            core.report({"submission": sub.to_dict(), "state": "completed"})
            core.acknowledge_host(sub)
            history.append(sub)
        records = CountedRecords(core._records)
        core._records = records
        for _ in range(100):
            endpoint._tick()
        assert records.visits == 0
        assert core.query(history[0])["state"] == "completed"
        assert core.submit(history[0])["state"] == "completed"
        assert core.pending_reports() == []
        fresh = replace(template, operation_id=str(uuid4()), request_ref=str(uuid4()))
        core.submit(fresh)
        assert core.dispatchable() == [(t, fresh)]
        core.report({"submission": fresh.to_dict(), "state": "completed"})
        core.acknowledge_host(fresh)
        assert records.visits == 0, "new admission scanned settled replay history"
        # Conflicting late terminal evidence must return to reconciliation.
        core.report({"submission": history[-1].to_dict(), "state": "not_started"})
        assert core.query(history[-1])["state"] == "unknown"
        assert core.pending_reports() == [(t, history[-1], "query")]
        now[0] = max(template.accept_until_ns, template.deadline_ns)
        core.collect()
        assert not core.has_operation(history[0].operation_id)
        assert core.has_operation(history[-1].operation_id)
        assert core.pending_reports() == [(t, history[-1], "cancel")]
        core.report({"submission": history[-1].to_dict(), "state": "completed"})
        core.acknowledge_host(history[-1])
        core.collect()
        assert not core.has_operation(history[-1].operation_id)
    finally:
        host.shutdown()


@pytest.mark.parametrize("has_target", [False, True])
def test_shutdown_readiness_is_false_while_already_admitted_work_can_finish(has_target):
    from types import SimpleNamespace

    from std_srvs.srv import Trigger

    from inference_service.global_runtime_scheduler_node import GlobalRuntimeSchedulerNode

    t = target()
    core = RuntimeScheduler([t] if has_target else [])
    host, _ = controller(t)
    try:
        if has_target:
            connect(core, host)
            sub = prepare(host)
            core.submit(sub)
        assert core.ready
        core.begin_shutdown()
        node = SimpleNamespace(_runtime_endpoint=SimpleNamespace(core=core))
        response = GlobalRuntimeSchedulerNode._readiness_callback(node, Trigger.Request(), Trigger.Response())
        assert not response.success and not core.ready
        if has_target:
            assert not core.resource_ready(t.hardware_resource_id)
            assert core.dispatchable() == [(t, sub)]
            with pytest.raises(ControlError, match="GLOBAL_SHUTTING_DOWN"):
                core.submit(prepare(host))
    finally:
        host.shutdown()


def test_dispatch_only_scans_unsettled_work_with_long_queue():
    t = target(max_pending_requests=512)
    core = RuntimeScheduler([t])
    host, _ = controller(t)
    try:
        connect(core, host)
        template = prepare(host)
        core.submit(template)
        assert core.dispatchable() == [(t, template)]
        for _ in range(256):
            core.submit(replace(template, operation_id=str(uuid4()), request_ref=str(uuid4())))
        assert core.dispatchable() == []
        core.report({"submission": template.to_dict(), "state": "completed"})
        assert len(core.dispatchable()) == 1
    finally:
        host.shutdown()


def test_generic_workload_identity_is_declared_by_the_request_producer():
    from inference_service.scheduler.runtime_controller import freeze_inputs

    def request(identity, steps):
        return ModelRequest({}, {"tenant_run": {"id": identity}, "steps": steps}, frozenset({"tenant_run"}))

    frozen, digest, _, bucket = freeze_inputs(request("one", 3), 4096)
    assert frozen.metadata["tenant_run"]["id"] == "one"
    other = freeze_inputs(request("two", 3), 4096)
    assert other[1] != digest and other[3] == bucket
    assert freeze_inputs(request("one", 4), 4096)[3] != bucket
    # A business-looking name has no special meaning to the scheduler.
    assert (
        freeze_inputs(ModelRequest({}, {"visual_trigger_epoch": 1}), 4096)[3]
        != freeze_inputs(ModelRequest({}, {"visual_trigger_epoch": 2}), 4096)[3]
    )


def test_delayed_backend_readiness_cannot_bypass_priority_validation():
    from inference_service.scheduler.runtime_ros import SchedulerRuntimeAdapter

    adapter = object.__new__(SchedulerRuntimeAdapter)
    adapter.target = target(resource_mode="ascend_concurrent")
    handle = SimpleNamespace(local_ready=False, assembly=SimpleNamespace(session=SimpleNamespace(capabilities=None)))
    adapter._validate_backend(handle)
    handle.local_ready = True
    with pytest.raises(ControlError, match="BACKEND_PRIORITY_UNAVAILABLE"):
        adapter._validate_backend(handle)


@pytest.mark.parametrize(
    "missing", ["supports_isolated_stage_execution", "supports_multiple_instances", "failure_isolation"]
)
def test_concurrent_resource_requires_backend_isolation_evidence(missing):
    from inference_service.scheduler.runtime_ros import SchedulerRuntimeAdapter

    capabilities = SimpleNamespace(
        priority_mapping=object(),
        supports_isolated_stage_execution=True,
        supports_multiple_instances=True,
        admission_evidence=SimpleNamespace(
            sdk_initialization=True, multi_instance_execution=True, failure_isolation=True, independent_close=True
        ),
    )
    adapter = object.__new__(SchedulerRuntimeAdapter)
    adapter.target = target(resource_mode="ascend_concurrent")
    handle = SimpleNamespace(
        local_ready=True, assembly=SimpleNamespace(session=SimpleNamespace(capabilities=capabilities))
    )
    adapter._validate_backend(handle)
    setattr(capabilities.admission_evidence if missing == "failure_isolation" else capabilities, missing, False)
    with pytest.raises(ControlError, match="BACKEND_CONCURRENCY_UNAVAILABLE"):
        adapter._validate_backend(handle)


@pytest.mark.parametrize("policy", ["fifo", "edf"])
@pytest.mark.parametrize("mode", ["serial", "ascend_concurrent"])
def test_mixed_priorities_compete_for_one_target_slot(policy, mode):
    t = target(resource_mode=mode)
    core = RuntimeScheduler([t], policy=policy, resources=attested_resources(t))
    host, _ = controller(t)
    try:
        connect(core, host)
        submissions = [prepare(host, priority=p) for p in (2, 1, 0)]
        for sub in submissions:
            core.submit(sub)
        assert [sub.priority for _, sub in core.dispatchable()] == [0]
        assert core.dispatchable() == []
        for priority in (1, 2):
            active = next(r for r in core._records.values() if r.state == "dispatching")
            core.report({"submission": active.submission.to_dict(), "state": "not_started"})
            assert [sub.priority for _, sub in core.dispatchable()] == [priority]
    finally:
        host.shutdown()


@pytest.mark.parametrize("policy", ["fifo", "edf"])
@pytest.mark.parametrize("priorities", [(3, 2), (2**80, 2**64)])
def test_serial_priorities_order_across_hosts_without_preempting_or_requiring_native_streams(policy, priorities):
    targets = [
        target(name, default_priority=priority)
        for name, priority in (("active", priorities[0]), ("low", priorities[1]), ("high", 0))
    ]
    core = RuntimeScheduler(targets, policy=policy)
    pairs = [controller(t) for t in targets]
    try:
        for host, _ in pairs:
            connect(core, host)
        submissions = [prepare(host) for host, _ in pairs]
        core.submit(submissions[0])
        assert core.dispatchable() == [(targets[0], submissions[0])]
        for sub in submissions[1:]:
            core.submit(sub)
        for index in (0, 2, 1):
            host, executor = pairs[index]
            sub = submissions[index]
            executor.release.clear()
            host.execute(sub, core.epoch)
            assert executor.entered.wait(1)
            assert core.dispatchable() == []
            assert executor.calls[0].metadata["priority"] == 0
            assert host.query(sub, core.epoch)["submission"]["priority"] == sub.priority
            executor.release.set()
            host._records[sub.operation_id].done.result(timeout=1)
            core.report(host.query(sub, core.epoch))
            expected = {0: 2, 2: 1}.get(index)
            assert core.dispatchable() == ([] if expected is None else [(targets[expected], submissions[expected])])
    finally:
        for host, executor in pairs:
            executor.release.set()
            host.shutdown()


@pytest.mark.parametrize("priority", [0, 2])
def test_ascend_concurrent_same_priority_overlaps_across_instances_but_respects_instance_capacity(priority):
    targets = [target(name, resource_mode="ascend_concurrent") for name in ("first", "second")]
    core = RuntimeScheduler(targets, resources=attested_resources(*targets))
    pairs = [controller(t) for t in targets]
    try:
        for host, executor in pairs:
            connect(core, host)
            executor.release.clear()
        submissions = [prepare(host, priority=priority) for host, _ in pairs]
        extra = prepare(pairs[0][0], priority=priority)
        for sub in (*submissions, extra):
            core.submit(sub)
        assert core.dispatchable() == list(zip(targets, submissions, strict=True))
        for (host, executor), sub in zip(pairs, submissions, strict=True):
            host.execute(sub, core.epoch)
            assert executor.entered.wait(1)
            assert executor.calls[0].metadata["priority"] == priority
        assert core.dispatchable() == []
        host, executor = pairs[0]
        executor.release.set()
        host._records[submissions[0].operation_id].done.result(timeout=1)
        core.report(host.query(submissions[0], core.epoch))
        assert core.dispatchable() == [(targets[0], extra)]
        assert pairs[1][0].query(submissions[1], core.epoch)["state"] == "running"
    finally:
        for host, executor in pairs:
            executor.release.set()
            host.shutdown()


def test_ascend_priority_range_is_enforced_by_global_and_host_before_execution():
    with pytest.raises(ValueError, match="priority"):
        target(resource_mode="ascend_concurrent", default_priority=8)
    t = target(resource_mode="ascend_concurrent")
    core = RuntimeScheduler([t], resources=attested_resources(t))
    host, executor = controller(t)
    try:
        connect(core, host)
        sub = prepare(host, priority=8)
        with pytest.raises(ControlError, match="UNSUPPORTED_PRIORITY"):
            core.submit(sub)
        assert core.dispatchable() == []
        rejected = host.execute(sub, core.epoch)
        assert rejected["state"] == "not_started"
        assert rejected["error"] == "UNSUPPORTED_PRIORITY"
        assert executor.calls == []
    finally:
        host.shutdown()


@pytest.mark.parametrize("priority", [-1, True, 1.5, "8"])
def test_software_priorities_still_require_nonnegative_integers(priority):
    with pytest.raises(ValueError, match="priority"):
        target(default_priority=priority)
    host, _ = controller(target())
    try:
        host.accept_epoch("test")
        with pytest.raises(ValueError, match="priority"):
            prepare(host, priority=priority)
    finally:
        host.shutdown()


def test_prepare_traces_business_and_operation_identity(monkeypatch):
    from inference_service.scheduler import runtime_controller

    events = []
    monkeypatch.setattr(
        runtime_controller, "trace", SimpleNamespace(event=lambda name, **fields: events.append((name, fields)))
    )
    host, _ = controller(target())
    host.accept_epoch("test")
    try:
        sub = host.prepare(ModelRequest(), ExecutionContext.create("business-request"))
        assert events == [
            (
                "runtime_operation_prepared",
                {"request_id": sub.operation_id, "parent_id": "business-request", "target_id": sub.target_id},
            )
        ]
        assert host._records[sub.operation_id].context.request_id == sub.parent_id
    finally:
        host.shutdown()


def test_shared_device_serial_and_distinct_devices_overlap():
    targets = [target("policy"), target("sam"), target("ram", "other")]
    core = RuntimeScheduler(targets)
    hosts = [controller(t)[0] for t in targets]
    try:
        for host in hosts:
            connect(core, host)
        submissions = [prepare(host) for host in hosts]
        for item in submissions:
            core.submit(item)
        commands = core.dispatchable()
        assert [t.target_id for t, _ in commands] == [targets[0].target_id, targets[2].target_id]
        first = submissions[0]
        core.report({"submission": first.to_dict(), "state": "completed"})
        assert core.dispatchable()[0][1] == submissions[1]
    finally:
        for host in hosts:
            host.shutdown()


def test_duplicate_execute_cancel_before_claim_and_frozen_inputs():
    t = target()
    core = RuntimeScheduler([t])
    host, executor = controller(t)
    try:
        connect(core, host)
        image = np.ones((3, 3), dtype=np.uint8)
        sub = host.prepare(ModelRequest({"image": image}), ExecutionContext("parent", Deadline.after(1)))
        image[:] = 0
        core.submit(sub)
        core.submit(sub)
        assert len(core.dispatchable()) == 1
        executor.release.clear()
        host.execute(sub, core.epoch)
        assert executor.entered.wait(1)
        assert host.execute(sub, core.epoch)["state"] == "running"
        assert np.all(executor.calls[0].inputs["image"] == 1)
        executor.release.set()
        host._records[sub.operation_id].done.result(timeout=1)
        assert len(executor.calls) == 1
        core.report(host.query(sub, core.epoch))
        assert host.result(sub).outputs["answer"][0] == 1
        with pytest.raises(ControlError, match="RESULT_EXPIRED"):
            host.result(sub)
        assert host.execute(sub, core.epoch)["state"] == "completed"
        cancelled = prepare(host)
        host.cancel(cancelled, core.epoch)
        assert host.execute(cancelled, core.epoch)["state"] == "not_started"
        assert len(executor.calls) == 1
    finally:
        executor.release.set()
        host.shutdown()


def test_unknown_survives_expiry_and_blocks_resource_but_not_other_device():
    clock = [1000000000]

    def now():
        return clock[0]

    targets = [target("sam"), target("policy"), target("other", "cpu")]
    core = RuntimeScheduler(targets, now_ns=now)
    hosts = [controller(t, now_ns=now)[0] for t in targets]
    try:
        for host in hosts:
            connect(core, host)
        first = prepare(hosts[0])
        core.submit(first)
        core.dispatchable()
        core.mark_transport_unknown(first)
        clock[0] = first.accept_until_ns + 1
        core.collect()
        assert core.query(first)["state"] == "unknown"
        assert first.operation_id in core._records
        for host in hosts:
            connect(core, host)
        with pytest.raises(ControlError, match="RESOURCE_NOT_READY"):
            core.submit(prepare(hosts[1]))
        other = prepare(hosts[2])
        core.submit(other)
        assert core.dispatchable()[0][1] == other
    finally:
        for host in hosts:
            host.shutdown()


def test_fifo_edf_do_not_preempt_running_work():
    for policy, expected in (("fifo", "second"), ("edf", "third")):
        targets = [target(name) for name in ("first", "second", "third")]
        core = RuntimeScheduler(targets, policy=policy)
        hosts = [controller(t)[0] for t in targets]
        try:
            for host in hosts:
                connect(core, host)
            submissions = [prepare(host) for host in hosts]
            submissions[2] = replace(submissions[2], deadline_ns=submissions[1].deadline_ns - 1000)
            core.submit(submissions[0])
            core.dispatchable()
            core.submit(submissions[1])
            core.submit(submissions[2])
            assert core.dispatchable() == []
            core.report({"submission": submissions[0].to_dict(), "state": "completed"})
            assert core.dispatchable()[0][0].target_id == f"perception/{expected}"
        finally:
            for host in hosts:
                host.shutdown()


@pytest.mark.parametrize("expiry", ["retry", "snapshot"])
def test_edf_uses_business_deadline_and_separately_expires_credentials(expiry):
    clock = [time.monotonic_ns()]
    targets = [target("later", retry_window_ms=100 if expiry == "retry" else 30000), target("earlier")]
    core = RuntimeScheduler(targets, policy="edf", now_ns=lambda: clock[0])
    hosts = [controller(t, now_ns=lambda: clock[0])[0] for t in targets]
    try:
        for host in hosts:
            connect(core, host)
        later = replace(prepare(hosts[0]), deadline_ns=clock[0] + 5_000_000_000)
        if expiry == "snapshot":
            later = replace(later, snapshot_valid_until_ns=clock[0] + 100_000_000)
        earlier = prepare(hosts[1])
        core.submit(later)
        core.submit(earlier)
        assert later.latest_start_ns < earlier.latest_start_ns
        assert core.dispatchable() == [(targets[1], earlier)]
        core.report({"submission": earlier.to_dict(), "state": "completed"})
        clock[0] = later.latest_start_ns
        assert core.dispatchable() == []
        assert core.query(later)["state"] == "not_started"
    finally:
        for host in hosts:
            host.shutdown()


@pytest.mark.parametrize("outcome", ["completed", "failure", "abandoned"])
def test_business_waiter_retains_result_after_retry_expiry(outcome):
    from inference_service.backends.errors import BackendInferenceError
    from inference_service.scheduler.runtime_ros import SchedulerRuntimeAdapter
    from inference_service.unified_runtime.errors import ExecutionFailure

    clock = [time.monotonic_ns()]
    t = target(retry_window_ms=20)
    core = RuntimeScheduler([t], now_ns=lambda: clock[0])
    host, executor = controller(t, now_ns=lambda: clock[0])
    adapter = SchedulerRuntimeAdapter.__new__(SchedulerRuntimeAdapter)
    adapter.controller, adapter.target = host, t
    submissions = []
    if outcome == "failure":
        executor.error = BackendInferenceError("known failure", operation_started=True, outcome_known=True)

    def call(verb, sub, timeout, **kwargs):
        if verb == "submit":
            submissions.append(sub)
            core.submit(sub)
            assert core.dispatchable() == [(t, sub)]
            host.execute(sub, core.epoch)
            host._records[sub.operation_id].done.result(timeout=1)
            core.report(host.query(sub, core.epoch))
            # Execution/transport outlives credentials while the caller's
            # deadline is still valid; ACK and GC precede business Query.
            clock[0] = sub.accept_until_ns + 1
            acknowledge(sub, core.epoch)
            core.acknowledge_host(sub)
            host.collect()
            core.collect()
            assert sub.operation_id in host._records
            assert core.query(sub)["state"] == "completed"
            if outcome == "abandoned":
                raise RuntimeError("business waiter abandoned")
        return getattr(core, "query" if verb == "submit" else verb)(sub)

    adapter._call = call
    # Also collect between the adapter's own ACK and result access, even past
    # the deadline: local ownership must survive until the waiter unwinds.
    acknowledge = host.acknowledge

    def acknowledge_and_collect(sub, epoch):
        acknowledge(sub, epoch)
        if clock[0] >= sub.accept_until_ns:
            clock[0] = sub.deadline_ns + 1
            host.collect()

    try:
        connect(core, host)
        host.acknowledge = acknowledge_and_collect
        if outcome == "failure":
            with pytest.raises(ExecutionFailure, match="known failure"):
                adapter.execute(ModelRequest(), ExecutionContext("business", Deadline.after(1)))
        elif outcome == "abandoned":
            with pytest.raises(ControlError, match="COMPLETED_DELIVERY_FAILED"):
                adapter.execute(ModelRequest(), ExecutionContext("business", Deadline.after(1)))
        else:
            assert (
                adapter.execute(ModelRequest(), ExecutionContext("business", Deadline.after(1))).outputs["answer"][0]
                == 1
            )
        assert len(executor.calls) == 1
        assert not host._records
        assert host._input_bytes == host._result_bytes == 0
        clock[0] = submissions[0].deadline_ns + 1
        core.collect()
        assert not core._records
    finally:
        host.shutdown()


def test_conflicting_retry_and_expired_tombstone_never_reexecute():
    clock = [1000000000]
    t = target()
    core = RuntimeScheduler([t], now_ns=lambda: clock[0])
    host, executor = controller(t, now_ns=lambda: clock[0])
    try:
        connect(core, host)
        sub = prepare(host)
        core.submit(sub)
        with pytest.raises(ControlError, match="OPERATION_CONFLICT"):
            core.submit(replace(sub, deadline_ns=sub.deadline_ns + 1))
        host.cancel(sub, core.epoch)
        host.acknowledge(sub, core.epoch)
        host.release_result_waiter(sub)
        clock[0] = sub.accept_until_ns + 1
        host.collect()
        assert sub.operation_id not in host._records
        with pytest.raises(ControlError, match="OPERATION_EXPIRED"):
            host.execute(sub, core.epoch)
        with pytest.raises(ControlError, match="OPERATION_EXPIRED"):
            host.cancel(sub, core.epoch)
        assert not executor.calls
    finally:
        host.shutdown()


def test_managed_handle_rejects_direct_execution():
    host, _ = controller(target())
    try:
        with pytest.raises(Exception, match="dispatch_token_required"):
            host.handle.execute(ModelRequest(), ExecutionContext("bypass"))
    finally:
        host.shutdown()


def test_input_capacity_and_snapshot_dispatch_deadline():
    from inference_service.scheduler.runtime_controller import input_bytes

    budget = input_bytes(ModelRequest({"image": np.zeros((2, 3), np.uint8)}))
    host, executor = controller(target(max_retained_input_bytes=budget))
    core = RuntimeScheduler([host.target])
    try:
        connect(core, host)
        with pytest.raises(ControlError, match="INPUT_CAPACITY"):
            host.prepare(ModelRequest({"image": np.zeros(11, np.uint8)}), ExecutionContext("too-big"))
        sub = prepare(host, snapshot_valid_until_ns=1)
        assert host.execute(sub, core.epoch)["state"] == "not_started"
        assert not executor.calls
    finally:
        host.shutdown()


def test_submitted_identity_is_not_a_policy_session():
    submission = Submission(
        operation_id=str(uuid4()),
        parent_id="business",
        target_id="perception/sam",
        boot_id=str(uuid4()),
        generation=1,
        request_ref=str(uuid4()),
        digest="f" * 64,
        clock_id=machine_clock_id(),
        accept_until_ns=1,
        deadline_ns=1,
        priority=0,
        input_bytes=1,
        input_bucket="{}",
    )
    assert "session_id" not in submission.to_dict()


@pytest.mark.parametrize("cancel_before_reply", [False, True])
def test_global_snapshot_expiry_survives_local_cleanup(cancel_before_reply):
    from inference_service.scheduler.runtime_ros import SchedulerRuntimeAdapter

    host, executor = controller(target())
    core = RuntimeScheduler([host.target])
    connect(core, host)
    adapter = SchedulerRuntimeAdapter.__new__(SchedulerRuntimeAdapter)
    adapter.controller, adapter.target = host, host.target

    def call(verb, submission, timeout, **kwargs):
        result = getattr(core, verb)(submission)
        if verb == "submit":
            assert core.dispatchable() == []
            if cancel_before_reply:
                core.report(host.cancel(submission, core.epoch))
            result = core.query(submission)
        return result

    adapter._call = call
    try:
        with pytest.raises(ControlError) as failure:
            adapter.execute(ModelRequest(), ExecutionContext("parent", Deadline.after(2)), snapshot_valid_until_ns=1)
        assert failure.value.code == "SNAPSHOT_OR_DEADLINE_EXPIRED"
        assert not executor.calls
        assert host._input_bytes == 0
        assert all(not record.acknowledged for record in host._records.values())
        for record in host._records.values():
            assert core.report(record.report())["state"] == "not_started"
            host.acknowledge(record.submission, core.epoch)
            core.acknowledge_host(record.submission)
    finally:
        host.shutdown()


def test_retry_record_retention_with_simulated_50hz_clock():
    # Advance a simulated clock by 20 ms per operation; this is not a throughput test.
    now = time.monotonic_ns()
    host, executor = controller(target(), now_ns=lambda: now)
    core = RuntimeScheduler([host.target], now_ns=lambda: now)
    connect(core, host)
    first = None
    try:
        # Cross the complete 30-second retention window, not just the old
        # 256-record limit. Both host and Global must collect acknowledged work.
        for index in range(1600):
            connect(core, host)
            submission = prepare(host)
            first = first or submission
            core.submit(submission)
            assert core.dispatchable()[0][1] == submission
            host.execute(submission, core.epoch)
            host._records[submission.operation_id].done.result(timeout=2)
            core.report(host.query(submission, core.epoch))
            host.result(submission)
            host.acknowledge(submission, core.epoch)
            core.acknowledge_host(submission)
            if index == 256:
                assert first.operation_id in host._records
                assert host.execute(first, core.epoch)["state"] == "completed"
                assert len(executor.calls) == index + 1
            now += 20000000
        assert first.operation_id not in host._records
        assert first.operation_id not in core._records
        assert len(executor.calls) == 1600
        assert len(host._records) <= 1501
        assert len(core._records) <= 1501
    finally:
        host.shutdown()


def test_optional_clean_failure_does_not_block_required_target():
    from inference_service.scheduler.runtime_controller import UnavailableRuntime

    required, optional = target("required"), target("optional", required=False)
    core = RuntimeScheduler([required, optional])
    host, _ = controller(required)
    failed = RuntimeExecutionController(optional, UnavailableRuntime())
    try:
        connect(core, host)
        connect(core, failed)
        assert core.resource_ready("gpu")
        assert core.submit(prepare(host))["state"] == "queued"
    finally:
        host.shutdown()
        failed.shutdown()


@pytest.mark.parametrize("host_confirmed", [False, True])
def test_cancelled_operation_requires_host_terminal_evidence(host_confirmed):
    t = target()
    core = RuntimeScheduler([t])
    host, _ = controller(t)
    try:
        connect(core, host)
        sub = prepare(host)
        core.submit(sub)
        assert core.cancel(sub)["state"] == "not_started"
        core.acknowledge_host(sub)
        assert not core._records[sub.operation_id].host_acknowledged
        assert core.pending_reports() == [(t, sub, "cancel")]
        if host_confirmed:
            core.report(host.cancel(sub, core.epoch))
            assert core.pending_reports() == [(t, sub, "ack")]
        assert core.report({"submission": sub.to_dict(), "state": "prepared"})["state"] == "not_started"
        result = core.report({"submission": sub.to_dict(), "state": "running"})
        assert result["state"] == ("not_started" if host_confirmed else "unknown")
        assert core.resource_ready(t.hardware_resource_id) == host_confirmed
        # Conflicting terminal states are never treated as harmless reordering.
        result = core.report({"submission": sub.to_dict(), "state": "completed"})
        assert result["state"] == ("unknown" if host_confirmed else "completed")
        assert not core.resource_ready(t.hardware_resource_id)
    finally:
        host.shutdown()
