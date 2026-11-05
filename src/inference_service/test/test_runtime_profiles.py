import time
from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest

from inference_service.scheduler.runtime_protocol import (
    ControlError,
)
from inference_service.scheduler.runtime_scheduler import RuntimeScheduler
from inference_service.unified_runtime import Deadline, ExecutionContext, ModelRequest

from .runtime_scheduler_helpers import attested_resources, connect, controller, prepare, target


def _valid_samples(t, count=20):
    from inference_service.scheduler.runtime_profiles import RuntimeProfileRegistry

    return [
        {
            "operation_id": str(i),
            "target_id": t.target_id,
            "identity": RuntimeProfileRegistry.identity(t),
            "work": "execute",
            "input_bucket": "input",
            "occupancy_ns": 100,
            "measured_at": time.time(),
        }
        for i in range(count)
    ]


def test_live_collection_crosses_retention_window_and_builds_admissible_profile(tmp_path):
    from inference_service.scheduler.runtime_profiles import (
        RuntimeProfileRegistry,
        collect_runtime_samples,
        read_runtime_samples,
        write_runtime_profile,
    )
    from inference_service.scheduler.runtime_ros import GlobalRuntimeEndpoint, _reply, _request

    from .runtime_scheduler_helpers import FakeNode

    t = target(resource="cpu", retry_window_ms=10, hardware_profile_fingerprint="cpu-v1", runtime_version="test")
    bus = {}
    endpoint = GlobalRuntimeEndpoint(FakeNode(bus), [t], resources=attested_resources(t))
    core = endpoint.core
    now = [time.monotonic_ns()]
    core.now_ns = lambda: now[0]
    host, model = controller(t, now_ns=lambda: now[0])
    # Let control records expire throughout this long collection.
    initial = [True]
    samples = tmp_path / "models" / "_work" / "sam" / "profiling" / "samples.jsonl"

    def call(payload):
        if payload["verb"] == "measurements":
            if initial[0]:
                initial[0] = False
            elif payload["after_sequence"] == core._measurement_sequence:
                for _ in range(32):
                    connect(core, host)
                    sub = host.prepare(ModelRequest({"x": 1}), ExecutionContext(str(uuid4()), Deadline.after(0.1)))
                    core.submit(sub)
                    assert core.dispatchable() == [(t, sub)]
                    host.execute(sub, core.epoch)
                    host.completion(sub).result(timeout=2)
                    now[0] += 1_000_000
                    core.report(host.query(sub, core.epoch))
                    host.result(sub)
                    host.acknowledge(sub, core.epoch)
                    core.acknowledge_host(sub)
                    host.collect()
                    core.collect()
        from types import SimpleNamespace

        return _reply(bus[endpoint.endpoint](_request(payload), SimpleNamespace()))

    try:
        connect(core, host)
        assert collect_runtime_samples(t, call, samples, count=10000) == 10000
        assert len(model.calls) == 10016
        assert len(core.measurements) == 1024
        collected = list(read_runtime_samples(samples))
        assert [s["sequence"] for s in collected] == list(range(1, 10001))
        profile = samples.with_suffix(".json")
        write_runtime_profile(t, iter(collected), profile)
        predicted = replace(t, predictive_admission=True, profile_path=str(profile))
        registry = RuntimeProfileRegistry(predicted)
        assert registry.estimates[("execute", collected[0]["input_bucket"])] == 1_200_000
        admission = RuntimeScheduler([predicted], resources=attested_resources(predicted))
        predicted_host, _ = controller(predicted)
        try:
            connect(admission, predicted_host)
            sub = predicted_host.prepare(ModelRequest({"x": 1}), ExecutionContext("admit", Deadline.after(1)))
            assert admission.submit(sub)["state"] == "queued"
            assert admission.dispatchable() == [(predicted, sub)]
        finally:
            predicted_host.shutdown()
    finally:
        host.shutdown()


@pytest.mark.parametrize("fault", ["overrun", "restart", "identity"])
def test_collection_stops_on_invalid_evidence_and_preserves_prefix(tmp_path, fault):
    from inference_service.scheduler.runtime_profiles import collect_runtime_samples, read_runtime_samples

    t = target()
    samples = _valid_samples(t)
    calls = []

    def call(payload):
        calls.append(payload)
        if payload["verb"] == "describe":
            return {"epoch": "original", "target_count": 1, "targets": [t.to_dict()]}
        assert payload["epoch"] == "original"
        if len(calls) == 2:
            # Startup discards old/warmup data, even if that old buffer overran.
            return {"latest_sequence": 5000, "dropped": 3976}
        if len(calls) == 3:
            assert payload["after_sequence"] == 5000
            return {"samples": [samples[0]], "next_sequence": 5001, "dropped": 0}
        assert payload["after_sequence"] == 5001
        if fault == "restart":
            raise ControlError("EPOCH_MISMATCH")
        if fault == "overrun":
            return {"samples": [], "dropped": 1}
        return {"samples": [{**samples[1], "identity": {}}], "next_sequence": 5002, "dropped": 0}

    output = tmp_path / "models" / "_work" / "sam" / "profiling" / "samples.jsonl"
    with pytest.raises((ValueError, ControlError), match="overrun|EPOCH_MISMATCH|identity mismatch"):
        collect_runtime_samples(t, call, output, count=2)
    assert list(read_runtime_samples(output)) == [samples[0]]


def test_collector_filters_targets_without_losing_global_cursor(tmp_path):
    from inference_service.scheduler.runtime_profiles import collect_runtime_samples, read_runtime_samples

    t = target()
    wanted = _valid_samples(t, count=1)[0]
    other = _valid_samples(target("other"), count=1)[0]
    reads = []

    def call(payload):
        if payload["verb"] == "describe":
            return {"epoch": "epoch", "target_count": 1, "targets": [t.to_dict()]}
        reads.append(payload["after_sequence"])
        if len(reads) == 1:
            return {"latest_sequence": 10}
        if len(reads) == 2:
            return {"samples": [other], "next_sequence": 11, "dropped": 0}
        return {"samples": [wanted], "next_sequence": 12, "dropped": 0}

    output = tmp_path / "models" / "_work" / "sam" / "profiling" / "samples.jsonl"
    assert collect_runtime_samples(t, call, output, count=1) == 1
    assert reads == [0, 10, 11]
    assert list(read_runtime_samples(output)) == [wanted]


@pytest.mark.parametrize("fault", ["missing", "changed", "restart", "none"])
def test_collector_validates_all_catalogue_pages_before_creating_output(tmp_path, fault):
    from inference_service.scheduler.runtime_profiles import collect_runtime_samples, read_runtime_samples

    t = target()
    other = target("other")
    calls = []
    sample = _valid_samples(t, 1)[0]

    def call(payload):
        calls.append(payload)
        if payload["verb"] == "describe":
            offset = payload.get("offset", 0)
            assert offset in (0, 1)
            entry = other if offset == 0 or fault == "missing" else t
            if offset == 1 and fault == "changed":
                entry = replace(t, composition_fingerprint="changed")
            return {
                "epoch": "new" if offset and fault == "restart" else "epoch",
                "target_count": 2,
                "targets": [entry.to_dict()],
            }
        assert fault == "none", "invalid catalogue started reading measurements"
        if len(calls) == 3:
            return {"latest_sequence": 0}
        return {"samples": [sample], "next_sequence": 1, "dropped": 0}

    output = tmp_path / "models" / "_work" / "sam" / "profiling" / "samples.jsonl"
    if fault == "none":
        assert collect_runtime_samples(t, call, output, count=1) == 1
        assert list(read_runtime_samples(output)) == [sample]
    else:
        with pytest.raises((ValueError, ControlError), match="target|EPOCH_MISMATCH"):
            collect_runtime_samples(t, call, output, count=1)
        assert not output.exists()
    assert calls[:2] == [{"verb": "describe"}, {"verb": "describe", "offset": 1}]


def test_profile_writer_limits_total_samples_and_preserves_previous_profile(tmp_path, monkeypatch):
    from inference_service.scheduler import runtime_profiles

    t = target()
    output = tmp_path / "models" / "_work" / "sam" / "profiling" / "runtime.json"
    samples = _valid_samples(t, 21)
    monkeypatch.setattr(runtime_profiles, "MAX_PROFILE_SAMPLES", 20, raising=False)
    runtime_profiles.write_runtime_profile(t, iter(samples[:20]), output, min_samples=20)
    original = output.read_bytes()
    with pytest.raises(ValueError, match="sample limit"):
        runtime_profiles.write_runtime_profile(t, iter(samples), output, min_samples=20)
    assert output.read_bytes() == original


def test_profile_writer_bounds_retained_operation_identifiers(tmp_path):
    from inference_service.scheduler.runtime_profiles import write_runtime_profile

    t = target()
    samples = _valid_samples(t)
    samples[-1]["operation_id"] = "x" * 1024 * 1024
    output = tmp_path / "models" / "_work" / "sam" / "profiling" / "runtime.json"
    with pytest.raises(ValueError, match="operation"):
        write_runtime_profile(t, samples, output, min_samples=20)
    assert not output.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("work", "invalid"),
        ("measured_at", float("nan")),
        ("measured_at", float("inf")),
        ("measured_at", -float("inf")),
        ("measured_at", True),
        ("measured_at", "123"),
        ("measured_at", 1e308),
        ("measured_at", time.time() + 86400),
    ],
)
def test_writer_rejects_each_invalid_sample_before_replacing_profile(tmp_path, field, value):
    from inference_service.scheduler.runtime_profiles import RuntimeProfileRegistry, write_runtime_profile

    t = target()
    path = tmp_path / "models" / "_work" / "samples" / "profiling" / "runtime.json"
    samples = _valid_samples(t)
    write_runtime_profile(t, samples, path, min_samples=20)
    original = path.read_bytes()
    samples[-1][field] = value
    with pytest.raises(ValueError, match="invalid"):
        write_runtime_profile(t, samples, path, min_samples=20)
    assert path.read_bytes() == original
    assert RuntimeProfileRegistry(replace(t, profile_path=str(path)), min_samples=20).estimates


@pytest.mark.parametrize("measured", [1e308, time.time() + 86400])
def test_writer_rejects_entire_future_sample_set(tmp_path, measured):
    from inference_service.scheduler.runtime_profiles import RuntimeProfileRegistry, write_runtime_profile

    t = target()
    path = tmp_path / "models" / "_work" / "future" / "profiling" / "runtime.json"
    samples = _valid_samples(t)
    write_runtime_profile(t, samples, path, min_samples=20)
    original = path.read_bytes()
    for sample in samples:
        sample["measured_at"] = measured
    with pytest.raises(ValueError, match="invalid measurement time"):
        write_runtime_profile(t, samples, path, min_samples=20)
    assert path.read_bytes() == original
    assert RuntimeProfileRegistry(replace(t, profile_path=str(path)), min_samples=20).estimates


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_profile_publication_failure_preserves_readable_file(tmp_path, monkeypatch, failure):
    from inference_service.scheduler import runtime_profiles

    t = target()
    path = tmp_path / "models" / "_work" / "atomic" / "profiling" / "runtime.json"
    samples = _valid_samples(t)
    runtime_profiles.write_runtime_profile(t, samples, path, min_samples=20)
    original = path.read_bytes()

    def fail(*args):
        assert path.read_bytes() == original
        raise OSError("disk failure")

    if failure == "write":
        actual_open = runtime_profiles.tempfile.NamedTemporaryFile

        def partial_file(**kwargs):
            stream = actual_open(**kwargs)
            write = stream.write

            def partial_write(data):
                write(data[:10])
                fail()

            stream.write = partial_write
            return stream

        monkeypatch.setattr(runtime_profiles.tempfile, "NamedTemporaryFile", partial_file)
    else:
        monkeypatch.setattr(runtime_profiles.os, "replace", fail)
    with pytest.raises(OSError, match="disk failure"):
        runtime_profiles.write_runtime_profile(t, samples, path, min_samples=20)
    assert list(path.parent.iterdir()) == [path]
    assert path.read_bytes() == original
    assert runtime_profiles.RuntimeProfileRegistry(replace(t, profile_path=str(path)), min_samples=20).estimates


def test_measurement_page_fits_wire_budget_with_long_identity_and_bucket():
    from inference_service.scheduler.runtime_profiles import RuntimeProfileRegistry
    from inference_service.scheduler.runtime_protocol import encode_control

    names = tuple(f"observation_{i:02d}_" + "x" * 185 for i in range(64))
    t = target(profile_observation_inputs=names)
    core = RuntimeScheduler([t])
    for sequence in range(1, 4):
        core.measurements.append(
            {
                "sequence": sequence,
                "operation_id": str(uuid4()),
                "target_id": t.target_id,
                "identity": RuntimeProfileRegistry.identity(t),
                "work": "execute",
                "input_bucket": "x" * 7080,
                "occupancy_ns": 123456,
                "measured_at": time.time(),
            }
        )
    core._measurement_sequence = 3
    cursor, seen = 0, []
    while cursor < 3:
        page = core.read_measurements(cursor)
        assert len(encode_control(page).encode()) <= 32768
        assert len(page["samples"]) == 1
        assert page["next_sequence"] > cursor
        seen.extend(sample["sequence"] for sample in page["samples"])
        cursor = page["next_sequence"]
    assert seen == [1, 2, 3]


def test_profile_writer_enforces_serialized_size_and_preserves_existing_file(tmp_path, monkeypatch):
    from inference_service.scheduler import runtime_profiles

    t = target(hardware_profile_fingerprint="hardware", runtime_version="mock")
    path = tmp_path / "models" / "_work" / "size" / "profiling" / "runtime.json"
    # Escaped keys exercise actual JSON size, not just raw bucket lengths.
    samples = [
        {
            "operation_id": f"{bucket}-{sample}",
            "target_id": t.target_id,
            "identity": runtime_profiles.RuntimeProfileRegistry.identity(t),
            "work": "execute",
            "input_bucket": f"{bucket}" + '"' * 7000,
            "occupancy_ns": 100,
            "measured_at": time.time(),
        }
        for bucket in range(4)
        for sample in range(20)
    ]
    runtime_profiles.write_runtime_profile(t, samples, path, min_samples=20)
    payload = path.read_bytes()
    t = replace(t, profile_path=str(path))
    monkeypatch.setattr(runtime_profiles, "MAX_PROFILE_BYTES", len(payload))
    runtime_profiles.write_runtime_profile(t, samples, path, min_samples=20)
    assert len(runtime_profiles.RuntimeProfileRegistry(t, min_samples=20).estimates) == 4
    monkeypatch.setattr(runtime_profiles, "MAX_PROFILE_BYTES", len(payload) - 1)
    with pytest.raises(ValueError, match="size limit"):
        runtime_profiles.write_runtime_profile(t, samples, path, min_samples=20)
    assert path.read_bytes() == payload
    with pytest.raises(ValueError, match="size limit"):
        runtime_profiles.RuntimeProfileRegistry(t, min_samples=20)


@pytest.mark.parametrize("expire_at", ["before_submit", "before_dispatch"])
def test_predictive_close_needs_no_profile_bucket_and_survives_expiry(tmp_path, expire_at):
    from inference_service.scheduler.runtime_profiles import RuntimeProfileRegistry, write_runtime_profile

    t = target(hardware_profile_fingerprint="hardware", runtime_version="mock")
    path = tmp_path / "models" / "_work" / "close" / "profiling" / "runtime.json"
    measured = time.time()
    samples = [
        {
            "operation_id": str(i),
            "target_id": t.target_id,
            "identity": RuntimeProfileRegistry.identity(t),
            "work": "execute",
            "input_bucket": "inference-only",
            "occupancy_ns": 100,
            "measured_at": measured,
        }
        for i in range(20)
    ]
    write_runtime_profile(t, samples, path, min_samples=20)
    t = replace(t, predictive_admission=True, profile_path=str(path))
    core = RuntimeScheduler([t], resources=attested_resources(t), profile_min_samples=20)
    clock = [measured]
    core._profiles[t.target_id] = RuntimeProfileRegistry(t, min_samples=20, now=lambda: clock[0])
    host, _ = controller(t)
    try:
        connect(core, host)
        close = host.prepare(ModelRequest(), ExecutionContext("close", Deadline.after(1)), work="close")
        if expire_at == "before_submit":
            clock[0] += 31 * 86400
        assert core.submit(close)["state"] == "queued"
        clock[0] += 31 * 86400
        assert core.dispatchable() == [(t, close)]
        host.execute(close, core.epoch)
        host.completion(close).result(timeout=1)
        assert host.handle.closed and host.query(close, core.epoch)["state"] == "completed"
    finally:
        host.shutdown()


@pytest.mark.parametrize("samples,age", [(0, 30), (-1, 30), (True, 30), (20, 0), (20, -1), (20, float("nan"))])
def test_invalid_profile_options_rejected_before_io(samples, age):
    from inference_service.scheduler.runtime_profiles import RuntimeProfileRegistry

    with pytest.raises(ValueError):
        RuntimeProfileRegistry(target(), min_samples=samples, max_age_days=age)
    with pytest.raises(ValueError):
        RuntimeScheduler([], profile_min_samples=samples, profile_max_age_days=age)


def test_measurement_cursor_survives_eviction_and_reports_gaps():
    from collections import deque

    core = RuntimeScheduler([])
    core.measurements = deque(({"sequence": i} for i in range(1, 5)), maxlen=4)
    core._measurement_sequence = 4
    first = core.read_measurements()
    assert [s["sequence"] for s in first["samples"]] == [1, 2]
    core.measurements.append({"sequence": 5})
    core._measurement_sequence = 5
    second = core.read_measurements(first["next_sequence"])
    assert [s["sequence"] for s in second["samples"]] == [3, 4]
    assert second["dropped"] == 0
    assert core.read_measurements()["dropped"] == 1
    with pytest.raises(ControlError, match="INVALID_MEASUREMENT_CURSOR"):
        core.read_measurements(6)


def test_priority_zero_prediction_accounts_for_active_low_priority_but_skips_queued_low_priority(tmp_path):
    import json

    from inference_service.scheduler.runtime_controller import freeze_inputs
    from inference_service.scheduler.runtime_profiles import RuntimeProfileRegistry

    clock = [time.monotonic_ns()]
    path = tmp_path / "runtime.json"
    t = target(
        predictive_admission=True,
        profile_path=str(path),
        hardware_profile_fingerprint="hardware",
        runtime_version="mock",
    )
    _, _, _, bucket = freeze_inputs(ModelRequest({"image": np.zeros((2, 3), dtype=np.uint8)}), 1024)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "scope": "runtime_execution",
                "identity": RuntimeProfileRegistry.identity(t),
                "measured_at": time.time(),
                "buckets": [{"work": "execute", "input_bucket": bucket, "samples": 20, "upper_bound_ns": 10_000_000}],
            }
        )
    )
    core = RuntimeScheduler([t], resources=attested_resources(t), profile_min_samples=20, now_ns=lambda: clock[0])
    host, _ = controller(t, now_ns=lambda: clock[0])
    try:
        connect(core, host)
        active = prepare(host, priority=3)
        core.submit(active)
        assert core.dispatchable() == [(t, active)]
        low = prepare(host, priority=2)
        core.submit(low)
        # Remaining active occupancy (10 ms) + own cost (10 ms) + margin (5 ms).
        too_soon = replace(prepare(host), deadline_ns=clock[0] + 24_000_000)
        with pytest.raises(ControlError, match="NO_FEASIBLE_DEADLINE"):
            core.submit(too_soon)
        high = replace(prepare(host), deadline_ns=clock[0] + 25_000_000)
        assert core.submit(high)["state"] == "queued"
        # A peer at priority zero must include the first high-priority request.
        peer = replace(prepare(host), deadline_ns=clock[0] + 34_000_000)
        with pytest.raises(ControlError, match="NO_FEASIBLE_DEADLINE"):
            core.submit(peer)
        assert core.dispatchable() == []
        clock[0] += 11_000_000
        with pytest.raises(ControlError, match="PREDICTION_UNAVAILABLE"):
            core.submit(prepare(host))
        core.report({"submission": active.to_dict(), "state": "completed"})
        # Overrun invalidates high's prediction; it is rejected before execution.
        assert core.dispatchable() == [(t, low)]
        assert core.query(high)["error"] == "NO_FEASIBLE_DEADLINE"
        core.report({"submission": low.to_dict(), "state": "completed"})
        # Lower priorities have no predictive completion admission, only expiry.
        best_effort = replace(prepare(host, priority=7), deadline_ns=clock[0] + 1_000_000)
        assert core.submit(best_effort)["state"] == "queued"
        assert core.dispatchable() == [(t, best_effort)]
    finally:
        host.shutdown()


def test_runtime_profile_boundary_input_identity_and_overrun(tmp_path):
    import json
    import time

    from inference_service.scheduler.runtime_controller import freeze_inputs
    from inference_service.scheduler.runtime_profiles import RuntimeProfileRegistry, write_runtime_profile

    t = target(hardware_profile_fingerprint="hardware-v1", runtime_version="mock-1")
    _, _, _, bucket = freeze_inputs(ModelRequest({"image": np.zeros((2, 3), dtype=np.uint8)}), 1024)
    samples = [
        {
            "operation_id": str(uuid4()),
            "target_id": t.target_id,
            "identity": RuntimeProfileRegistry.identity(t),
            "work": "execute",
            "input_bucket": bucket,
            "occupancy_ns": 10000000,
            "measured_at": time.time(),
        }
        for _ in range(20)
    ]
    path = tmp_path / "models" / "_work" / "sam" / "profiling" / "runtime.json"
    write_runtime_profile(t, samples, path, min_samples=20)
    t = replace(t, predictive_admission=True, profile_path=str(path))
    core = RuntimeScheduler([t], resources=attested_resources(t), profile_min_samples=20)
    host, _ = controller(t)
    try:
        connect(core, host)
        sub = prepare(host)
        core.submit(sub)
        core.dispatchable()
        core._records[sub.operation_id].estimated_finish_ns = time.monotonic_ns() - 1
        with pytest.raises(ControlError, match="PREDICTION_UNAVAILABLE"):
            core.submit(prepare(host))
        assert core.query(sub)["state"] == "dispatching"
        other_shape = host.prepare(
            ModelRequest({"image": np.zeros((3, 3), dtype=np.uint8)}), ExecutionContext("other", Deadline.after(1))
        )
        with pytest.raises(ControlError, match="PROFILE_INPUT_OUT_OF_RANGE"):
            core.submit(other_shape)
    finally:
        host.shutdown()
    data = json.loads(path.read_text())
    data["scope"] = "global_proxy"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="global_proxy is not compatible"):
        RuntimeProfileRegistry(t, min_samples=20)


@pytest.mark.parametrize("work", ["reset", "drain"])
@pytest.mark.parametrize("blocker", ["none", "active", "other_group"])
def test_lifecycle_prediction_excludes_only_queued_work_its_barrier_cancels(tmp_path, work, blocker):
    import json

    from inference_service.scheduler.runtime_controller import freeze_inputs
    from inference_service.scheduler.runtime_profiles import RuntimeProfileRegistry

    now = time.monotonic_ns()
    image = ModelRequest({"image": np.zeros((2, 3), dtype=np.uint8)})
    bucket = freeze_inputs(image, 4096)[3]
    lifecycle_bucket = freeze_inputs(ModelRequest(), 4096)[3]
    targets = []
    for name in ("first", "other"):
        t = target(
            name,
            predictive_admission=True,
            profile_path=str(tmp_path / f"{name}.json"),
            hardware_profile_fingerprint="hardware",
            runtime_version="mock",
        )
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "scope": "runtime_execution",
                    "identity": RuntimeProfileRegistry.identity(t),
                    "measured_at": time.time(),
                    "buckets": [
                        {"work": "execute", "input_bucket": bucket, "samples": 20, "upper_bound_ns": 2_000_000_000},
                        {"work": work, "input_bucket": lifecycle_bucket, "samples": 20, "upper_bound_ns": 100_000_000},
                    ],
                }
            )
        )
        targets.append(t)
    core = RuntimeScheduler(targets, resources=attested_resources(*targets), profile_min_samples=20, now_ns=lambda: now)
    hosts = [controller(t, now_ns=lambda: now)[0] for t in targets]
    first, other = hosts
    try:
        for host in hosts:
            connect(core, host)

        def queued(host):
            sub = host.prepare(image, ExecutionContext(str(uuid4()), Deadline.after(10)))
            core.submit(sub)
            return sub

        blocking = None
        if blocker != "none":
            blocking = queued(first if blocker == "active" else other)
            if blocker == "active":
                assert core.dispatchable() == [(first.target, blocking)]
        discarded = [queued(first), queued(first)]
        control = first.prepare(ModelRequest(), ExecutionContext("control", Deadline.after(1)), work=work)
        if blocker != "none":
            with pytest.raises(ControlError, match="NO_FEASIBLE_DEADLINE"):
                core.submit(control)
            assert all(core.query(sub)["state"] == "queued" for sub in discarded)
            assert not core.has_operation(control.operation_id)
            if blocker == "active":
                core.report({"submission": blocking.to_dict(), "state": "completed"})
            else:
                core.cancel(blocking)
        assert core.submit(control)["state"] == "queued"
        assert all(core.query(sub)["state"] == "not_started" for sub in discarded)
        assert core.dispatchable() == [(first.target, control)]
    finally:
        for host in hosts:
            host.shutdown()


def test_profile_cli_accepts_production_sample_count_above_32_mib(tmp_path, monkeypatch):
    import json
    import sys

    from inference_service.scheduler.runtime_profiles import RuntimeProfileRegistry, main

    t = target(hardware_profile_fingerprint="hardware-v1", runtime_version="mock-1")
    bucket = "x" * 4096
    samples = tmp_path / "samples.jsonl"
    with samples.open("w") as stream:
        for index in range(10000):
            stream.write(
                json.dumps(
                    {
                        "operation_id": str(index),
                        "target_id": t.target_id,
                        "identity": RuntimeProfileRegistry.identity(t),
                        "work": "execute",
                        "input_bucket": bucket,
                        "occupancy_ns": index + 1,
                        "measured_at": time.time(),
                    }
                )
                + "\n"
            )
    assert samples.stat().st_size > 32 * 1024 * 1024
    target_file = tmp_path / "target.json"
    target_file.write_text(json.dumps(t.to_dict()))
    output = tmp_path / "models" / "_work" / "sam" / "profiling" / "runtime.json"
    monkeypatch.setattr(
        sys, "argv", ["profiles", "--target", str(target_file), "--samples", str(samples), "--output", str(output)]
    )
    main()
    registry = RuntimeProfileRegistry(replace(t, profile_path=str(output)), min_samples=10000)
    assert registry.estimates[("execute", bucket)] == 12000


@pytest.mark.parametrize("clock_rollback", [False, True])
def test_runtime_profile_rechecks_age_at_admission_and_dispatch(tmp_path, clock_rollback):
    import json

    from inference_service.scheduler.runtime_controller import freeze_inputs
    from inference_service.scheduler.runtime_profiles import RuntimeProfileRegistry

    path = tmp_path / "runtime.json"
    t = target(
        predictive_admission=True,
        profile_path=str(path),
        hardware_profile_fingerprint="hardware-v1",
        runtime_version="mock-1",
    )
    measured = time.time()
    clock = [measured]
    _, _, _, bucket = freeze_inputs(ModelRequest({"image": np.zeros((2, 3), dtype=np.uint8)}), 1024)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "scope": "runtime_execution",
                "identity": RuntimeProfileRegistry.identity(t),
                "measured_at": measured,
                "buckets": [{"work": "execute", "input_bucket": bucket, "samples": 20, "upper_bound_ns": 10000000}],
            }
        )
    )
    other = target("other", resource="cpu")
    core = RuntimeScheduler([t, other], resources=attested_resources(t, other), profile_min_samples=20)
    core._profiles[t.target_id] = RuntimeProfileRegistry(t, min_samples=20, now=lambda: clock[0])
    host, _ = controller(t)
    peer, _ = controller(other)
    try:
        connect(core, host)
        connect(core, peer)
        # The exact age limit is still valid; crossing it must invalidate the
        # already-loaded registry, just as loading a stale profile would.
        clock[0] = measured + 30 * 86400
        active = prepare(host)
        core.submit(active)
        assert core.dispatchable() == [(t, active)]
        queued = prepare(host)
        core.submit(queued)
        clock[0] = measured - 1 if clock_rollback else clock[0] + 1
        rejected = prepare(host)
        with pytest.raises(ControlError, match="PROFILE_EXPIRED"):
            core.submit(rejected)
        assert rejected.operation_id not in core._records
        assert core.dispatchable() == []
        assert core.query(active)["state"] == "dispatching"
        core.report({"submission": active.to_dict(), "state": "completed"})
        independent = prepare(peer)
        core.submit(independent)
        assert core.dispatchable() == [(other, independent)]
        report = core.query(queued)
        assert report["state"] == "not_started"
        assert report["error"] == "PROFILE_EXPIRED"
        assert any(sub == queued and verb == "cancel" for _, sub, verb in core.pending_reports())
    finally:
        host.shutdown()
        peer.shutdown()


def test_torch_cpu_freeze_and_transient_metadata_profile_bucket():
    import torch

    from inference_service.scheduler.runtime_controller import freeze_inputs

    tensor = torch.ones((2, 2))
    first, digest, size, bucket = freeze_inputs(
        ModelRequest(
            {"image": tensor},
            {"request_id": "a", "observation_monotonic_ns": 1},
            frozenset({"request_id", "observation_monotonic_ns"}),
        ),
        4096,
    )
    tensor.zero_()
    assert first.inputs["image"].sum().item() == 4
    _, other_digest, _, other_bucket = freeze_inputs(
        ModelRequest(
            {"image": torch.ones((2, 2))},
            {"request_id": "b", "observation_monotonic_ns": 2},
            frozenset({"request_id", "observation_monotonic_ns"}),
        ),
        4096,
    )
    assert other_digest != digest
    assert other_bucket == bucket
    assert size >= 16


@pytest.mark.parametrize("tensor_kind", ["numpy", "torch"])
@pytest.mark.parametrize("payloads", [(10, 30), ([97, 98], [228, 184]), ([[0.0, 1.0]], [[1.0, 0.0]])])
def test_prediction_rejects_same_shape_with_different_tensor_work(tmp_path, tensor_kind, payloads):
    from inference_service.scheduler.runtime_controller import freeze_inputs
    from inference_service.scheduler.runtime_profiles import RuntimeProfileRegistry, write_runtime_profile

    convert = np.asarray if tensor_kind == "numpy" else pytest.importorskip("torch").tensor
    inputs = [convert(value) for value in payloads]
    assert inputs[0].shape == inputs[1].shape
    t = target(hardware_profile_fingerprint="hardware", runtime_version="mock")
    bucket = freeze_inputs(ModelRequest({"work": inputs[0]}), 1024)[3]
    path = tmp_path / "models" / "_work" / "sam" / "profiling" / "runtime.json"
    write_runtime_profile(
        t,
        [
            {
                "operation_id": str(uuid4()),
                "target_id": t.target_id,
                "identity": RuntimeProfileRegistry.identity(t),
                "work": "execute",
                "input_bucket": bucket,
                "occupancy_ns": 1000000,
                "measured_at": time.time(),
            }
            for _ in range(20)
        ],
        path,
        min_samples=20,
    )
    t = replace(t, predictive_admission=True, profile_path=str(path))
    host, _ = controller(t)
    core = RuntimeScheduler([t], resources=attested_resources(t), profile_min_samples=20)
    try:
        connect(core, host)
        other = host.prepare(ModelRequest({"work": inputs[1]}), ExecutionContext("other", Deadline.after(1)))
        with pytest.raises(ControlError, match="PROFILE_INPUT_OUT_OF_RANGE"):
            core.submit(other)
        same = host.prepare(ModelRequest({"work": inputs[0]}), ExecutionContext("same", Deadline.after(1)))
        core.submit(same)
        assert [sub.operation_id for _, sub in core.dispatchable()] == [same.operation_id]
    finally:
        host.shutdown()


@pytest.mark.parametrize("tensor_kind", ["numpy", "torch"])
def test_observation_profile_accepts_live_frames_but_keeps_work_and_identity(tmp_path, tensor_kind):
    import json

    from inference_service.scheduler.runtime_controller import freeze_inputs
    from inference_service.scheduler.runtime_profiles import RuntimeProfileRegistry, write_runtime_profile

    convert = np.asarray if tensor_kind == "numpy" else pytest.importorskip("torch").as_tensor
    t = target(
        hardware_profile_fingerprint="hardware",
        runtime_version="mock",
        profile_observation_inputs=("image", "state"),
    )

    def request(pixel, *, shape=(3, 4, 3), dtype=np.float32, steps=10):
        pixels = np.zeros(shape, dtype=dtype)
        pixels.flat[0] = pixel
        return ModelRequest(
            {"image": convert(pixels), "state": convert(np.full(6, pixel, dtype=np.float32)), "steps": steps}
        )

    samples = []
    for index in range(20):
        _, _, _, bucket = freeze_inputs(request(index), 4096, t.profile_observation_inputs)
        samples.append(
            {
                "operation_id": str(uuid4()),
                "target_id": t.target_id,
                "identity": RuntimeProfileRegistry.identity(t),
                "work": "execute",
                "input_bucket": bucket,
                "occupancy_ns": (index + 1) * 1000000,
                "measured_at": time.time(),
            }
        )
    path = tmp_path / "models" / "_work" / "sam" / "profiling" / "runtime.json"
    write_runtime_profile(t, samples, path, margin=1.5, min_samples=20)
    t = replace(t, predictive_admission=True, profile_path=str(path))
    host, _ = controller(t)
    core = RuntimeScheduler([t], resources=attested_resources(t), profile_min_samples=20)
    try:
        connect(core, host)
        first = host.prepare(request(0), ExecutionContext("first", Deadline.after(2)))
        changed = host.prepare(request(99), ExecutionContext("changed", Deadline.after(2)))
        assert first.digest != changed.digest
        assert first.input_bucket == changed.input_bucket
        assert freeze_inputs(request(0), 4096)[1] == first.digest
        assert RuntimeProfileRegistry(t, min_samples=20).estimate(changed) == 30000000
        assert core.submit(first)["state"] == core.submit(changed)["state"] == "queued"
        for options in ({"shape": (4, 4, 3)}, {"dtype": np.uint8}, {"steps": 20}):
            other = host.prepare(request(99, **options), ExecutionContext("other", Deadline.after(2)))
            with pytest.raises(ControlError, match="PROFILE_INPUT_OUT_OF_RANGE"):
                core.submit(other)
        # A profile is evidence for a specific classification contract.
        with pytest.raises(ValueError, match="identity mismatch"):
            RuntimeProfileRegistry(replace(t, profile_observation_inputs=("image",)), min_samples=20)
        document = json.loads(path.read_text())
        document["schema_version"] = 1
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError, match="remeasure"):
            RuntimeProfileRegistry(t, min_samples=20)
    finally:
        host.shutdown()
