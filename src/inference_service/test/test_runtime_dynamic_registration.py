"""Model membership changes must preserve device ownership and replay evidence."""

import json
import time
from dataclasses import replace
from uuid import uuid4

import pytest

from inference_service.scheduler.runtime_profiles import RuntimeProfileRegistry
from inference_service.scheduler.runtime_protocol import ControlError, RuntimeExecutionTarget, Submission
from inference_service.scheduler.runtime_scheduler import RuntimeScheduler


def target(name="first", **options):
    return RuntimeExecutionTarget(
        target_id=f"perception/{name}",
        endpoint=f"/runtime/{name}",
        hardware_resource_id="cpu",
        physical_device="local/cpu",
        deployment_fingerprint="model",
        composition_fingerprint="plugin",
        lifecycle_group=f"perception/{name}",
        configuration_generation="config",
        interface="tensor_model",
        model_type="test",
        operation="infer",
        **options,
    )


def status(core, model, **values):
    return {
        "target": model.to_dict(),
        "boot_id": str(uuid4()),
        "generation": 1,
        "epoch": "",
        "clock_id": core.clock_id,
        "sequence": 1,
        "ready": True,
        "clean": True,
        "status_ns": core.now_ns(),
        **values,
    }


def connect(core, model, **values):
    report = status(core, model, **values)
    core.register(report)
    report.update(epoch=core.epoch, sequence=2)
    core.register(report)
    return report


def request(core, report):
    now = core.now_ns()
    return Submission(
        operation_id=str(uuid4()),
        parent_id="business",
        target_id=report["target"]["target_id"],
        boot_id=report["boot_id"],
        generation=report["generation"],
        request_ref=str(uuid4()),
        digest="0" * 64,
        clock_id=core.clock_id,
        accept_until_ns=now + 10_000_000_000,
        deadline_ns=now + 5_000_000_000,
        priority=0,
        input_bytes=0,
        input_bucket="{}",
    )


def scheduler(targets=(), **options):
    return RuntimeScheduler(targets, resources={"cpu": {"physical_device": "local/cpu"}}, **options)


def test_close_retirement_waits_for_terminal_acks_and_shutdown_waits_for_unregister():
    model = target()
    core = scheduler([model])
    observed = connect(core, model)
    previous = request(core, observed)
    core.submit(previous)
    core.dispatchable()
    core.report({"submission": previous.to_dict(), "state": "completed"})
    closing = replace(request(core, observed), work="close")
    core.submit(closing)
    core.dispatchable()
    core.report({"submission": closing.to_dict(), "state": "completed"})
    core.acknowledge_host(closing)
    assert not core.host_retired(model.target_id, observed["boot_id"])
    core.acknowledge_host(previous)
    assert core.host_retired(model.target_id, observed["boot_id"])
    assert not core.host_retired(model.target_id, str(uuid4()))
    assert core.shutdown_pending() == [model.target_id]
    core.fence_resource("cpu")
    core.unregister({**observed, "sequence": 3, "ready": False})
    assert not core.shutdown_pending()
    assert not core.resource_ready("cpu")  # Departure cannot release a recovery fence.


def test_replayed_close_ack_does_not_outlive_replaced_dynamic_target():
    clock = [time.monotonic_ns()]
    core = scheduler(now_ns=lambda: clock[0])
    model = target("reuse")
    old = connect(core, model)
    closing = replace(request(core, old), work="close")
    core.submit(closing)
    core.dispatchable()
    report = {"submission": closing.to_dict(), "state": "completed"}
    core.terminal_report(report)
    core.unregister({**old, "sequence": 3, "ready": False})
    new = connect(core, model)
    assert core.terminal_report(report)["state"] == "acknowledged"
    assert core.host_boot(model.target_id) == new["boot_id"]
    assert not core.host_retired(model.target_id, new["boot_id"])
    core.unregister({**new, "sequence": 3, "ready": False})
    clock[0] += 61_000_000_000
    core.collect()
    core.collect()  # Membership collection follows expiry of the last operation.
    assert model.target_id not in core.targets
    peer = connect(core, target("peer"))
    work = request(core, peer)
    core.submit(work)
    core.dispatchable()
    assert core.terminal_report({"submission": work.to_dict(), "state": "completed"})["state"] == "acknowledged"
    assert not core._pending_close_retirements


@pytest.mark.parametrize("unregistered", [False, True])
def test_retired_boot_unregister_replays_without_mutating_replacement(unregistered):
    core = scheduler()
    model = target("reuse")
    old = connect(core, model, generation=2)
    core.retire_recovered([model], {model.target_id: old["boot_id"]})
    departure = {**old, "sequence": 3, "ready": False}
    if unregistered:
        core.unregister(departure)
    new = connect(core, model)
    before = core._hosts[model.target_id]
    assert core.unregister(departure)["state"] == "unregistered"
    assert core._hosts[model.target_id] == before
    assert not core.host_retired(model.target_id, new["boot_id"])
    with pytest.raises(ControlError, match="RUNTIME_IDENTITY_MISMATCH"):
        core.unregister({**departure, "boot_id": str(uuid4())})
    for invalid in ({"epoch": str(uuid4())}, {"clock_id": str(uuid4())}, {"ready": True}, {"clean": False}):
        with pytest.raises(ControlError):
            core.unregister({**departure, **invalid})


@pytest.mark.parametrize("close_complete", [None, False, "true"])
@pytest.mark.parametrize(
    "state,error", [("unknown", ""), ("not_started", ""), ("completed", "CLOSE_FAILED"), ("completed", "close_failed")]
)
def test_unproven_close_never_retires_a_host(state, error, close_complete):
    model = target()
    core = scheduler([model])
    observed = connect(core, model)
    closing = replace(request(core, observed), work="close")
    core.submit(closing)
    core.dispatchable()
    report = {"submission": closing.to_dict(), "state": state, "error": error}
    if close_complete is not None:
        report["close_complete"] = close_complete
    core.report(report)
    core.acknowledge_host(closing)
    assert not core.host_retired(model.target_id, observed["boot_id"])
    assert core.shutdown_pending() == [model.target_id]


@pytest.mark.parametrize("retirement", ["close", "unregister", "stale", "collect"])
def test_new_boot_never_inherits_pending_activation(retirement):
    clock = [1_000_000_000]
    core = scheduler(now_ns=lambda: clock[0])
    model = target()
    old = connect(core, model, generation=2)
    key = model.target_id
    assert core.next_activation_generation(key, old["boot_id"]) == 3
    # A lost reply must reuse generation 3 for this boot.
    assert core.next_activation_generation(key, old["boot_id"]) == 3
    if retirement == "close":
        core.retire_recovered([model], {key: old["boot_id"]})
    elif retirement == "unregister":
        core.unregister({**old, "sequence": 3, "ready": False})
    else:
        clock[0] += max(core.stale_ns, model.retry_window_ms * 1_000_000) + 1
        if retirement == "collect":
            core.collect()
            assert key not in core.targets
            assert not core._pending_activations
    new = connect(core, model)
    assert core.next_activation_generation(key, new["boot_id"]) == 2
    with pytest.raises(ControlError, match="RECOVERY_IDENTITY_MISMATCH"):
        core.complete_activation(key, old["boot_id"], 3, 3)
    with pytest.raises(ControlError, match="RECOVERY_IDENTITY_MISMATCH"):
        core.next_activation_generation(key, old["boot_id"])
    assert core._hosts[key][0]["generation"] == 1
    core.complete_activation(key, new["boot_id"], 2, 2)
    assert core._hosts[key][0]["generation"] == 2
    assert not core._pending_activations


def test_failed_registration_collection_preserves_pending_activation():
    clock = [1_000_000_000]
    core = scheduler(now_ns=lambda: clock[0])
    model = target()
    old = connect(core, model)
    expected = core.next_activation_generation(model.target_id, old["boot_id"])
    clock[0] += max(core.stale_ns, model.retry_window_ms * 1_000_000) + 1
    invalid = replace(target("invalid"), hardware_resource_id="missing")
    with pytest.raises(ControlError):
        core.register(status(core, invalid))
    assert core.host_boot(model.target_id) == old["boot_id"]
    assert core.next_activation_generation(model.target_id, old["boot_id"]) == expected
    core.complete_activation(model.target_id, old["boot_id"], expected, expected)


def test_empty_public_scheduler_accepts_new_models_and_serializes_the_same_device():
    core = scheduler()
    first = request(core, connect(core, target("first")))
    second = request(core, connect(core, target("second")))
    core.submit(first)
    core.submit(second)
    assert [s for _, s in core.dispatchable()] == [first]
    assert core.dispatchable() == []
    core.report({"submission": first.to_dict(), "state": "completed"})
    assert [s for _, s in core.dispatchable()] == [second]


def test_declared_but_not_yet_started_model_does_not_block_registered_peer():
    later = target("later", required=False)
    core = scheduler([later])
    report = connect(core, target("first"))
    assert core.resource_ready("cpu")
    sub = request(core, report)
    core.submit(sub)
    assert core.dispatchable()
    connect(core, later)
    assert set(core._hosts) == {"perception/first", "perception/later"}


@pytest.mark.parametrize("name", ["perception/depth", "voice/tts"])
def test_required_static_host_must_register_and_keep_heartbeating(name):
    from types import SimpleNamespace

    from std_srvs.srv import Trigger

    from inference_service.global_runtime_scheduler_node import GlobalRuntimeSchedulerNode

    clock = [1_000_000_000]
    required = replace(target(), target_id=name, lifecycle_group=name)
    core = scheduler([required], now_ns=lambda: clock[0])
    node = SimpleNamespace(_runtime_endpoint=SimpleNamespace(core=core))

    def global_ready():
        return GlobalRuntimeSchedulerNode._readiness_callback(node, None, Trigger.Response()).success

    peer = connect(core, target("peer"))
    assert not core.resource_ready("cpu")
    assert not global_ready()
    with pytest.raises(ControlError, match="RESOURCE_NOT_READY"):
        core.submit(request(core, peer))
    connect(core, required)
    assert core.resource_ready("cpu")
    assert global_ready()
    clock[0] += core.stale_ns + 1
    connect(core, target("peer"))
    assert not core.resource_ready("cpu")
    assert not global_ready()


def test_static_policy_only_compatibility_still_requires_declared_members():
    model = target()
    core = RuntimeScheduler([model])
    assert not core.resource_ready("cpu")
    with pytest.raises(ControlError, match="TARGET_CONFIGURATION_MISMATCH"):
        core.register(status(core, target("extra")))


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"hardware_resource_id": "other"}, "UNKNOWN_RESOURCE"),
        ({"physical_device": "local/cuda:0"}, "RESOURCE_CONFIGURATION_MISMATCH"),
        ({"resource_mode": "ascend_concurrent"}, "RESOURCE_CONFIGURATION_MISMATCH"),
        ({"interface": "unknown"}, "DYNAMIC_MODEL_REQUIRED"),
    ],
)
def test_registration_checks_resource_contract_before_adding_target(changes, error):
    core = scheduler()
    with pytest.raises(ControlError, match=error):
        core.register(status(core, replace(target(), **changes)))
    assert not core.targets and not core._hosts


@pytest.mark.parametrize("field", ["endpoint", "lifecycle_group"])
def test_model_cannot_alias_an_existing_control_endpoint_or_lifecycle(field):
    core = scheduler()
    first = target()
    connect(core, first)
    with pytest.raises(ControlError, match="TARGET_ENDPOINT_OR_LIFECYCLE_CONFLICT"):
        core.register(status(core, replace(target("other"), **{field: getattr(first, field)})))


def test_clean_unregistration_allows_new_boot_but_rejects_late_old_heartbeat():
    core = scheduler()
    model = target()
    first = connect(core, model)
    closing = {**first, "ready": False, "sequence": 3}
    assert core.unregister(closing)["state"] == "unregistered"
    assert core.resource_ready("cpu")
    with pytest.raises(ControlError, match="HOST_RETIRED"):
        core.register({**first, "sequence": 4})
    second = connect(core, model)
    assert second["boot_id"] != first["boot_id"]
    with pytest.raises(ControlError, match="RUNTIME_IDENTITY_MISMATCH"):
        core.submit(request(core, first))


@pytest.mark.parametrize("state", ["queued", "dispatching", "unknown", "completed"])
def test_unregistration_cannot_erase_unsettled_operation(state):
    core = scheduler()
    report = connect(core, target())
    sub = request(core, report)
    core.submit(sub)
    if state != "queued":
        core.dispatchable()
    if state == "unknown":
        core.mark_transport_unknown(sub)
    if state == "completed":
        core.report({"submission": sub.to_dict(), "state": "completed"})
    with pytest.raises(ControlError, match="HOST_NOT_DRAINED"):
        core.unregister({**report, "sequence": 3, "ready": False})
    assert core._records[sub.operation_id].state == state


def test_terminal_receipt_allows_retirement_without_discarding_query_evidence():
    core = scheduler()
    report = connect(core, target())
    sub = request(core, report)
    core.submit(sub)
    core.dispatchable()
    core.terminal_report({"submission": sub.to_dict(), "state": "completed"})
    core.unregister({**report, "sequence": 3, "ready": False})
    assert core.query(sub)["state"] == "completed"
    assert core._records[sub.operation_id].host_acknowledged


@pytest.mark.parametrize("clean", [True, False])
def test_stale_host_can_only_leave_the_resource_when_last_evidence_is_clean(clean):
    clock = [1_000_000_000]
    core = scheduler(now_ns=lambda: clock[0])
    first = connect(core, target())
    core.register({**first, "sequence": 3, "clean": clean})
    clock[0] += core.stale_ns + 1
    connect(core, target("second"))
    assert core.resource_ready("cpu") is clean
    with pytest.raises(ControlError, match="RUNTIME_NOT_REGISTERED"):
        core.submit(request(core, first))


def test_unknown_work_stays_fenced_after_heartbeat_expiry_and_new_registration():
    clock = [1_000_000_000]
    core = scheduler(now_ns=lambda: clock[0])
    report = connect(core, target())
    sub = request(core, report)
    core.submit(sub)
    core.dispatchable()
    clock[0] += core.stale_ns + 1
    core.dispatchable()
    assert core.query(sub)["state"] == "unknown"
    with pytest.raises(ControlError, match="RESOURCE_NOT_READY"):
        connect(core, target("new"))
    core.collect()
    assert core.targets and not core.resource_ready("cpu")


def test_new_boot_while_old_host_is_active_requires_coordinated_recovery():
    core = scheduler()
    connect(core, target())
    with pytest.raises(ControlError, match="HOST_RESTART_REQUIRES_COORDINATED_RESTART"):
        connect(core, target())
    assert not core.resource_ready("cpu")


def test_clean_stale_incarnation_can_restart_without_blocking_other_models():
    clock = [1_000_000_000]
    core = scheduler(now_ns=lambda: clock[0])
    old = connect(core, target())
    clock[0] += core.stale_ns + 1
    new = connect(core, target())
    assert old["boot_id"] != new["boot_id"] and core.resource_ready("cpu")


@pytest.mark.parametrize("configured", [False, True])
def test_dynamic_directory_is_bounded_and_retired_models_are_collected(configured):
    clock = [1_000_000_000]
    core = scheduler(
        [target("configured")] if configured else [],
        now_ns=lambda: clock[0],
        max_registered_targets=2 if configured else 1,
    )
    report = connect(core, target())
    with pytest.raises(ControlError, match="TARGET_CAPACITY"):
        connect(core, target("next"))
    core.unregister({**report, "sequence": 3, "ready": False})
    clock[0] += 31_000_000_000
    core.collect()
    assert set(core.targets) == ({"perception/configured"} if configured else set())
    assert not core._hosts and not core._retired_boots
    connect(core, target("next"))


def test_bad_registration_cannot_reserve_an_endpoint():
    core = scheduler()
    with pytest.raises(ValueError):
        core.register(status(core, target(), boot_id="bad-uuid"))
    assert not core.targets and not core._hosts


def test_delayed_registration_cannot_resurrect_collected_incarnation():
    clock = [1_000_000_000]
    core = scheduler(now_ns=lambda: clock[0])
    report = connect(core, target())
    core.unregister({**report, "sequence": 3, "ready": False})
    clock[0] += 31_000_000_000
    core.collect()
    with pytest.raises(ControlError, match="STALE_STATUS"):
        core.register({**report, "sequence": 4})
    assert not core.targets


def test_short_retry_window_still_retains_boot_until_heartbeat_can_no_longer_arrive():
    clock = [1_000_000_000]
    core = scheduler(now_ns=lambda: clock[0])
    model = target(retry_window_ms=1)
    report = connect(core, model)
    core.unregister({**report, "sequence": 3, "ready": False})
    clock[0] += 2_000_000
    core.collect()
    with pytest.raises(ControlError, match="HOST_RETIRED"):
        core.register({**report, "sequence": 4})
    clock[0] = report["status_ns"] + core.stale_ns
    core.collect()
    with pytest.raises(ControlError, match="HOST_RETIRED"):
        core.register({**report, "sequence": 4})
    connect(core, model)
    assert core.resource_ready("cpu")


def test_predictive_membership_requires_central_resource_coverage():
    core = RuntimeScheduler(
        [],
        resources={
            "cpu": {
                "physical_device": "local/cpu",
                "predictive_admission": True,
                "hardware_profile_fingerprint": "hardware",
            }
        },
    )
    model = target(predictive_admission=True, hardware_profile_fingerprint="hardware", runtime_version="runtime")
    with pytest.raises(ControlError, match="PREDICTIVE_COVERAGE_REQUIRED"):
        core.register(status(core, model))
    assert not core.targets


@pytest.mark.parametrize("field", ["hardware_profile_fingerprint"])
def test_empty_predictive_resource_rejects_missing_profile_identity(field):
    resource = {
        "physical_device": "local/cpu",
        "predictive_admission": True,
        "managed_coverage": True,
        "hardware_profile_fingerprint": "hardware",
    }
    resource.pop(field)
    with pytest.raises(ValueError, match="prediction requires hardware fingerprint"):
        RuntimeScheduler([], resources={"cpu": resource})


def test_dynamic_model_cannot_use_a_different_configuration_while_evidence_is_retained():
    core = scheduler()
    report = connect(core, target())
    core.unregister({**report, "sequence": 3, "ready": False})
    with pytest.raises(ControlError, match="TARGET_CONFIGURATION_MISMATCH"):
        connect(core, replace(target(), deployment_fingerprint="replacement"))


def stages():
    base = replace(
        target(),
        interface="policy",
        execution_kind="stage",
        resource_mode="ascend_concurrent",
        lifecycle_group="policy/pi05",
        target_id="policy/pi05/visual",
        endpoint="/runtime/pi05/visual",
    )
    return [
        base,
        replace(
            base, target_id="policy/pi05/action", endpoint="/runtime/pi05/action", composition_fingerprint="action"
        ),
    ]


def stage_core(models=(), **options):
    return RuntimeScheduler(
        models,
        resources={"cpu": {"physical_device": "local/cpu", "mode": "ascend_concurrent", "verified_concurrency": True}},
        **options,
    )


def reports(core, models, **changes):
    boot = str(uuid4())
    members = sorted(model.target_id for model in models)
    return [status(core, model, boot_id=boot, lifecycle_target_ids=members, **changes) for model in models]


def register_stages(core, models):
    result = reports(core, models)
    core.register_group(result)
    result = [{**report, "epoch": core.epoch, "sequence": 2} for report in result]
    core.register_group(result)
    return result


def test_dynamic_policy_executor_joins_without_blocking_other_hosts_after_clean_retirement():
    core = RuntimeScheduler([], resources={"cpu": {"physical_device": "local/cpu"}})
    model = replace(target(), interface="policy")
    report = connect(core, model)
    assert core.resource_ready("cpu")
    core.unregister({**report, "sequence": 3, "ready": False})
    connect(core, target("other"))
    assert core.resource_ready("cpu")


def test_complete_stage_group_registers_and_heartbeats_atomically():
    models = stages()
    core = stage_core()
    group = register_stages(core, models)
    assert set(core.targets) == {model.target_id for model in models}
    assert core.resource_ready("cpu")
    for report in group:
        with pytest.raises(ControlError, match="GROUP_REGISTRATION_REQUIRED"):
            core.register({**report, "sequence": 3})
    core.register_group([{**report, "sequence": 3} for report in group])
    assert all(host[0]["sequence"] == 3 for host in core._hosts.values())


@pytest.mark.parametrize(
    "change", ["boot", "generation", "configuration", "resource", "endpoint", "incomplete", "dirty", "capacity"]
)
def test_invalid_group_never_publishes_partial_membership(change):
    models = stages()
    core = stage_core(max_registered_targets=1 if change == "capacity" else 256)
    group = reports(core, models)
    if change == "boot":
        group[1]["boot_id"] = str(uuid4())
    elif change == "generation":
        group[1]["generation"] = 2
    elif change == "configuration":
        group[1]["target"]["configuration_generation"] = "other"
    elif change == "resource":
        group[1]["target"]["physical_device"] = "local/other"
    elif change == "endpoint":
        group[1]["target"]["endpoint"] = models[0].endpoint
    elif change == "incomplete":
        group[1]["lifecycle_target_ids"] = [models[1].target_id]
    elif change == "dirty":
        group[1]["clean"] = False
    with pytest.raises(ControlError):
        core.register_group(group)
    assert not core.targets and not core._hosts and not core._profiles


def test_failed_group_heartbeat_preserves_both_prior_observations_and_active_work():
    core = stage_core()
    group = register_stages(core, stages())
    active = request(core, group[0])
    core.submit(active)
    core.dispatchable()
    before = core._hosts.copy()
    bad = [{**group[0], "sequence": 3}, {**group[1], "sequence": 1}]
    with pytest.raises(ControlError, match="STALE_STATUS"):
        core.register_group(bad)
    assert core._hosts == before
    assert core.query(active)["state"] == "dispatching"


def test_stage_group_cannot_replace_or_retire_one_peer_with_live_unknown_work():
    now = [1_000_000_000]
    core = stage_core(now_ns=lambda: now[0])
    group = register_stages(core, stages())
    active = request(core, group[0])
    core.submit(active)
    core.dispatchable()
    core.mark_transport_unknown(active)
    now[0] += core.stale_ns + 1
    with pytest.raises(ControlError):
        core.register_group(reports(core, stages()))
    core.collect()
    assert len(core.targets) == 2
    assert core.query(active)["state"] == "unknown"
    assert not core.resource_ready("cpu")


def test_group_is_collected_together_after_all_peer_retirement_windows():
    now = [1_000_000_000]
    core = stage_core(now_ns=lambda: now[0])
    group = register_stages(core, stages())
    core.unregister({**group[0], "ready": False, "sequence": 3})
    now[0] += 31_000_000_000
    # Keep the peer live: only one member's tombstone is otherwise collectible.
    core._hosts[group[1]["target"]["target_id"]] = ({**group[1], "status_ns": now[0]}, now[0])
    core.collect()
    assert len(core.targets) == 2


def predictive_core():
    return RuntimeScheduler(
        [],
        resources={
            "cpu": {
                "physical_device": "local/cpu",
                "predictive_admission": True,
                "managed_coverage": True,
                "hardware_profile_fingerprint": "hardware",
            }
        },
    )


def predictive_target(tmp_path, name):
    model = target(
        name,
        predictive_admission=True,
        hardware_profile_fingerprint="hardware",
        runtime_version="runtime",
        profile_path=str(tmp_path / f"{name}.json"),
    )
    document = {
        "schema_version": 2,
        "scope": "runtime_execution",
        "identity": RuntimeProfileRegistry.identity(model),
        "measured_at": time.time(),
        "buckets": [{"work": "execute", "input_bucket": "{}", "samples": 10000, "upper_bound_ns": 20_000_000}],
    }
    (tmp_path / f"{name}.json").write_text(json.dumps(document))
    return model


def test_dynamic_predictive_registration_loads_profile_and_accounts_for_existing_queue(tmp_path):
    core = predictive_core()
    first = request(core, connect(core, predictive_target(tmp_path, "first")))
    core.submit(first)
    second = request(core, connect(core, predictive_target(tmp_path, "second")))
    second = replace(second, deadline_ns=core.now_ns() + 30_000_000)
    with pytest.raises(ControlError, match="NO_FEASIBLE_DEADLINE"):
        core.submit(second)
    assert core.query(first)["state"] == "queued"
    assert second.operation_id not in core._records
    assert len(core._profiles) == 2


@pytest.mark.parametrize("failure", ["missing", "identity", "age", "bucket"])
def test_bad_dynamic_profile_cannot_change_membership_or_queued_work(tmp_path, failure):
    core = predictive_core()
    first = request(core, connect(core, predictive_target(tmp_path, "first")))
    core.submit(first)
    model = predictive_target(tmp_path, "second")
    path = tmp_path / "second.json"
    document = json.loads(path.read_text())
    if failure == "missing":
        path.unlink()
    elif failure == "identity":
        document["identity"]["runtime_version"] = "wrong"
    elif failure == "age":
        document["measured_at"] -= 31 * 86400
    else:
        document["buckets"][0]["input_bucket"] = []
    if failure != "missing":
        path.write_text(json.dumps(document))
    old_targets, old_hosts = core.targets.copy(), core._hosts.copy()
    with pytest.raises(ControlError, match="INVALID_RUNTIME_PROFILE"):
        core.register(status(core, model))
    assert core.targets == old_targets and core._hosts == old_hosts
    assert set(core._profiles) == {first.target_id}
    assert core.query(first)["state"] == "queued"


def test_predictive_new_incarnation_revalidates_profile_before_replacing_retired_host(tmp_path):
    core = predictive_core()
    model = predictive_target(tmp_path, "first")
    old = connect(core, model)
    core.unregister({**old, "sequence": 3, "ready": False})
    old_host = core._hosts[model.target_id]
    old_profile = core._profiles[model.target_id]
    old_tombstones = core._retired_boots.copy()
    (tmp_path / "first.json").write_text("{}")
    with pytest.raises(ControlError, match="INVALID_RUNTIME_PROFILE"):
        core.register(status(core, model))
    assert core._hosts[model.target_id] == old_host
    assert core._profiles[model.target_id] is old_profile
    assert core._retired_boots == old_tombstones
    assert model.target_id in core._retired


def test_group_cannot_omit_a_preconfigured_peer():
    models = stages()
    third = replace(models[1], target_id="policy/pi05/third", endpoint="/runtime/pi05/third")
    core = stage_core([*models, third])
    with pytest.raises(ControlError, match="INCOMPLETE_RUNTIME_GROUP"):
        core.register_group(reports(core, models))
    assert len(core.targets) == 3 and not core._hosts


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("restriction", ["samples", "age"])
def test_configured_profile_thresholds_reject_insufficient_evidence(tmp_path, dynamic, restriction):
    model = predictive_target(tmp_path, "configured")
    path = tmp_path / "configured.json"
    document = json.loads(path.read_text())
    document["measured_at"] -= 2 * 86400
    path.write_text(json.dumps(document))
    options = {"profile_min_samples": 10001} if restriction == "samples" else {"profile_max_age_days": 1}
    resources = {
        "cpu": {
            "physical_device": "local/cpu",
            "predictive_admission": True,
            "managed_coverage": True,
            "hardware_profile_fingerprint": "hardware",
        }
    }
    if dynamic:
        core = RuntimeScheduler([], resources=resources, **options)
        with pytest.raises((ValueError, ControlError), match="samples|stale|PROFILE"):
            connect(core, model)
        assert not core.targets
    else:
        with pytest.raises(ValueError, match="samples|stale"):
            RuntimeScheduler([model], resources=resources, **options)


def test_instance_scope_selects_distinct_policies_for_the_same_deployment():
    first = target("asr_vad", template_only=True, request_timeout_ms=1000)
    second = target("direction_vad", template_only=True, request_timeout_ms=2000)
    core = scheduler([first, second])
    with pytest.raises(ControlError, match="RUNTIME_TARGET_AMBIGUOUS"):
        core.discover_target("cpu", "model", "plugin")
    for expected in (first, second):
        actual = core.discover_target("cpu", "model", "plugin", configured_target_ids=[expected.target_id])
        assert actual["target_id"] == expected.target_id
        assert actual["request_timeout_ms"] == expected.request_timeout_ms
        assert core.discover_target("cpu", "model", "plugin", configured_target_id=expected.target_id) == actual
        with pytest.raises(ControlError, match="RUNTIME_TARGET_NOT_FOUND"):
            core.discover_target("cpu", "model", "plugin", configured_target_id=expected.target_id.split("/")[-1])


@pytest.mark.parametrize("group", [False, True])
def test_unchanged_heartbeats_do_not_copy_retained_operation_history(group):
    models = stages() if group else [target()]
    core = stage_core() if group else scheduler()
    boot = str(uuid4())
    reports = [
        status(core, model, boot_id=boot, lifecycle_target_ids=[t.target_id for t in models]) for model in models
    ]
    register = core.register_group if group else lambda reports: core.register(reports[0])
    register(reports)

    class RetainedHistory(dict):
        def copy(self):
            pytest.fail("unchanged heartbeat copied the global operation history")

    core._records = RetainedHistory(core._records)
    for sequence in range(2, 6):
        reports = [
            {**report, "epoch": core.epoch, "sequence": sequence, "ready": sequence % 2 == 0} for report in reports
        ]
        register(reports)
        assert all(core._hosts[t.target_id][0]["sequence"] == sequence for t in models)
    if group:
        reports = [{**report, "sequence": 6} for report in reports]
        reports[-1]["sequence"] = 5
        with pytest.raises(ControlError, match="STALE_STATUS"):
            register(reports)
        assert all(core._hosts[t.target_id][0]["sequence"] == 5 for t in models)


def test_new_registration_preserves_history_until_normal_collection():
    class History(dict):
        def copy(self):
            pytest.fail("new registration copied unrelated operation history")

    clock = [time.monotonic_ns()]
    core = scheduler(now_ns=lambda: clock[0])
    report = connect(core, target("history"))
    submission = request(core, report)
    core.submit(submission)
    core.report({"submission": submission.to_dict(), "state": "not_started"})
    core.acknowledge_host(submission)
    core._records = History(core._records)
    # Registration must not perform request expiry inside its candidate state.
    clock[0] = submission.accept_until_ns + 1
    newcomer = target("newcomer")
    core.register(status(core, newcomer))
    assert core.host_boot(newcomer.target_id) is not None
    assert core.has_operation(submission.operation_id)
    core.collect()
    assert not core.has_operation(submission.operation_id)


def test_retired_dynamic_target_does_not_retain_zero_history_counter():
    from inference_service.scheduler.runtime_scheduler import RuntimeOperation

    now = [time.monotonic_ns()]
    core = RuntimeScheduler([], resources={"cpu": {"physical_device": "local/cpu"}}, now_ns=lambda: now[0])
    for index in range(5):
        model = target(f"model_{index}")
        report = connect(core, model)
        submission = request(core, report)
        # A retained, acknowledged terminal operation is collectible only after
        # both its acceptance window and business deadline have passed.
        core._records[submission.operation_id] = RuntimeOperation(
            submission, index, state="completed", host_terminal_state="completed"
        )
        core._record_counts[model.target_id] += 1
        core.acknowledge_host(submission)
        core.unregister({**report, "ready": False, "sequence": 3})
        now[0] = max(submission.deadline_ns, submission.accept_until_ns, core._retired[model.target_id]) + 1
        core.collect()
        core.collect()
        assert not core.targets and not core._records and not core._record_counts


def test_expired_hosts_do_not_scan_settled_history():
    class History(dict):
        def values(self):
            pytest.fail("retirement scanned operation history")

    clock = [time.monotonic_ns()]
    core = scheduler(now_ns=lambda: clock[0])
    for index in range(8):
        connect(core, target(f"model_{index}"))
    core._records = History({str(index): None for index in range(4096)})
    clock[0] += core.stale_ns + 1
    for _ in range(100):
        core._collect_retired_targets()
    assert len(core.targets) == 8


@pytest.mark.parametrize("special", ["fifo", "zero"])
def test_dynamic_profile_rejects_special_files_without_blocking(tmp_path, special):
    import os

    core = predictive_core()
    model = predictive_target(tmp_path, "unsafe")
    path = tmp_path / "unsafe.json"
    path.unlink()
    if special == "fifo":
        os.mkfifo(path)
    else:
        path.symlink_to("/dev/zero")
    with pytest.raises(ControlError, match="INVALID_RUNTIME_PROFILE"):
        core.register(status(core, model))
    assert not core.targets


def test_slow_profile_does_not_hold_global_ledger_lock(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from inference_service.scheduler import runtime_scheduler

    core = predictive_core()
    model = predictive_target(tmp_path, "slow")
    entered, release = threading.Event(), threading.Event()
    registry = runtime_scheduler.RuntimeProfileRegistry

    def load(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return registry(*args, **kwargs)

    monkeypatch.setattr(runtime_scheduler, "RuntimeProfileRegistry", load)
    with ThreadPoolExecutor(1) as worker:
        future = worker.submit(core.register, status(core, model))
        try:
            assert entered.wait(1)
            assert core._lock.acquire(timeout=0.1)
            core._lock.release()
        finally:
            release.set()
        future.result(timeout=2)


@pytest.mark.parametrize("entry", ["register", "initialize", "bind"])
def test_group_budget_rejects_individually_valid_descriptors_before_publication(entry):
    from types import SimpleNamespace

    from inference_service.scheduler.runtime_initialization import runtime_initialization
    from inference_service.scheduler.runtime_ros import bind_registration_group

    base = stages()[0]
    models = [
        replace(
            base,
            target_id=f"policy/pi05/stage_{index}",
            endpoint=f"/runtime/pi05/stage_{index}",
            profile_observation_inputs=tuple(str(i) + "x" * 190 for i in range(64)),
        )
        for index in range(3)
    ]
    core = stage_core()
    with pytest.raises(ValueError, match="group exceeds control message budget"):
        if entry == "register":
            core.register_group(reports(core, models))
        elif entry == "initialize":
            with runtime_initialization(
                SimpleNamespace(), [t.to_dict() for t in models], "/global", boot_id=str(uuid4())
            ):
                pytest.fail("oversized group initialized")
        else:
            bind_registration_group([SimpleNamespace(target=t) for t in models])
    assert not core.targets and not core._initializations


@pytest.mark.parametrize("members, observation_count", [(2, 64), (16, 0)])
def test_supported_large_group_fits_real_initialization_and_heartbeat_envelopes(members, observation_count):
    from inference_manifest.runtime_target import validate_runtime_group_budget
    from inference_service.scheduler.runtime_protocol import encode_control

    base = stages()[0]
    models = [
        replace(
            base,
            target_id=f"policy/pi05/stage_{index}",
            endpoint=f"/runtime/pi05/stage_{index}",
            profile_observation_inputs=tuple(str(i) + "x" * 190 for i in range(observation_count)),
        )
        for index in range(members)
    ]
    validate_runtime_group_budget(models)
    core = stage_core()
    group = reports(core, models)
    encode_control(
        {"verb": "begin_initialization", "statuses": group, "token": str(uuid4()), "accept_until_ns": 2**63 - 1}
    )
    core.register_group(group)
    group = [{**s, "epoch": core.epoch, "sequence": 2**63 - 1} for s in group]
    encode_control({"verb": "register_group", "statuses": group})
    assert core.register_group(group)["state"] == "registered"
