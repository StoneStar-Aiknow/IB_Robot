import gc
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from inference_service.scheduler.runtime_protocol import (
    ControlError,
)
from inference_service.scheduler.runtime_scheduler import RuntimeScheduler
from inference_service.unified_runtime import Deadline, ExecutionContext, ModelRequest

from .runtime_scheduler_helpers import connect, controller, prepare, target


@pytest.mark.parametrize("per_channel", [False, True])
def test_quantized_inputs_are_rejected_before_copy_or_capacity_reservation(monkeypatch, per_channel):
    import torch

    from inference_service.scheduler.runtime_controller import freeze_inputs, input_bytes

    value = (
        torch.quantize_per_channel(
            torch.empty((1000, 0)),
            torch.ones(1000, dtype=torch.float64),
            torch.zeros(1000, dtype=torch.int64),
            axis=0,
            dtype=torch.qint8,
        )
        if per_channel
        else torch.quantize_per_tensor(torch.tensor([1.0]), 0.1, 0, torch.qint8)
    )
    request = ModelRequest({"quantized": value})
    monkeypatch.setattr(torch.Tensor, "clone", lambda *_args, **_kwargs: pytest.fail("quantized input was copied"))
    host, model = controller(target(max_retained_input_bytes=1024))
    host.accept_epoch("test")
    try:
        for measure in (input_bytes, freeze_inputs):
            with pytest.raises(ControlError, match="OPAQUE_INPUT"):
                measure(request, 1024)
        with pytest.raises(ControlError, match="OPAQUE_INPUT"):
            host.prepare(request, ExecutionContext("quantized", Deadline.after(1)))
        assert not host._records and not host._preparing and host._input_bytes == 0
        assert not model.calls
    finally:
        host.shutdown()


def test_host_new_admission_and_heartbeat_skip_history_until_expiry():
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

    now = [time.monotonic_ns()]
    host, _ = controller(target(max_records=32, max_pending_requests=1), now_ns=lambda: now[0])
    host.accept_epoch("test")
    try:
        history = []
        for _ in range(31):
            sub = prepare(host)
            host.cancel(sub, "test")
            host.acknowledge(sub, "test")
            host.release_result_waiter(sub)
            history.append(sub)
        records = CountedRecords(host._records)
        host._records = records
        fresh = prepare(host)
        with pytest.raises(ControlError, match="REQUEST_CAPACITY"):
            prepare(host)
        for _ in range(20):
            host.status()
            host.collect()
            host.release_result_waiter(history[0])
        assert records.visits == 0
        host.cancel(fresh, "test")
        host.acknowledge(fresh, "test")
        host.release_result_waiter(fresh)
        with pytest.raises(ControlError, match="REQUEST_CAPACITY"):
            prepare(host)
        now[0] = fresh.accept_until_ns
        host.collect()
        assert not host._records
        prepare(host)
    finally:
        host.shutdown()


@pytest.mark.parametrize(
    "left,right",
    [
        ({"a.b": 10}, {"a": {"b": 10}}),
        ({"a[0]": 10}, {"a": [10]}),
        ({"a": []}, {"a": {}}),
        ({"a": []}, {"a": [[], []]}),
        ({"cfg": {"steps": 1}, "cfg.steps": 1}, {"cfg": {"steps": 100000}, "cfg.steps": 1}),
    ],
)
def test_workload_bucket_preserves_paths_and_empty_structure(left, right):
    from inference_service.scheduler.runtime_controller import freeze_inputs

    first = freeze_inputs(ModelRequest(left), 65536)
    second = freeze_inputs(ModelRequest(right), 65536)
    assert first[1] != second[1]
    assert first[3] != second[3]


def test_identity_metadata_excludes_only_its_actual_top_level_field():
    from inference_service.scheduler.runtime_controller import freeze_inputs

    def freeze(trace, steps):
        return freeze_inputs(
            ModelRequest(metadata={"trace": {"id": trace}, "trace.steps": steps}, identity_metadata_keys={"trace"}),
            4096,
        )

    first, second, third = freeze("a", 1), freeze("b", 1), freeze("a", 100000)
    assert first[1] != second[1] and first[3] == second[3]
    assert first[3] != third[3]


def test_identity_metadata_names_obey_input_capacity_and_close_has_no_payload():
    from inference_service.scheduler.runtime_controller import freeze_inputs, input_bytes

    request = ModelRequest(identity_metadata_keys={"x" * 1024 * 1024})
    host, _ = controller(target(max_retained_input_bytes=1024))
    host.accept_epoch("test")
    try:
        for measure in (input_bytes, freeze_inputs):
            with pytest.raises(ControlError, match="INPUT_CAPACITY"):
                measure(request, 1024)
        with pytest.raises(ControlError, match="INPUT_CAPACITY"):
            host.prepare(request, ExecutionContext("identity", Deadline.after(1)))
        with pytest.raises(ControlError, match="CLOSE_PAYLOAD_UNSUPPORTED"):
            host.prepare(ModelRequest(identity_metadata_keys={"unused"}), ExecutionContext("close"), work="close")
        small = ModelRequest(metadata={"trace": "id"}, identity_metadata_keys={"trace", "unused"})
        frozen, _, charged, _ = freeze_inputs(small, 4096)
        assert charged == input_bytes(small) == input_bytes(frozen)
        assert charged > input_bytes(ModelRequest(metadata=small.metadata))
        assert not host._records and host._input_bytes == 0
    finally:
        host.shutdown()


@pytest.mark.parametrize("inputs", [{"items": [{} for _ in range(1000)]}, {str(i): [] for i in range(1000)}])
def test_empty_container_snapshots_cannot_bypass_input_quota(inputs):
    from inference_service.scheduler.runtime_controller import freeze_inputs, input_bytes

    request = ModelRequest(inputs)
    assert input_bytes(request) > 1024
    with pytest.raises(ControlError, match="INPUT_CAPACITY"):
        freeze_inputs(request, 1024)
    host, _ = controller(target(max_retained_input_bytes=1024))
    host.accept_epoch("test")
    try:
        with pytest.raises(ControlError, match="INPUT_CAPACITY"):
            host.prepare(request, ExecutionContext("empty-containers", Deadline.after(1)))
        assert not host._records and host._input_bytes == 0
    finally:
        host.shutdown()


@pytest.mark.parametrize("kind", ["numpy", "torch"])
def test_retained_views_charge_shared_backing_storage_once(kind):
    from inference_service.scheduler.runtime_controller import retain_result

    capacity = 16 * 1024 * 1024
    if kind == "numpy":
        allocation = np.empty(capacity, dtype=np.uint8)
    else:
        import torch

        allocation = torch.empty(capacity, dtype=torch.uint8)
    views = [allocation[:1], allocation[-1:]]
    assert capacity <= retain_result(views[0])[1] < capacity + 4096
    assert capacity <= retain_result(views)[1] < capacity + 4096


def test_external_cpu_tensor_is_copied_without_retaining_its_hidden_allocation():
    import torch

    from inference_service.scheduler.runtime_controller import retain_result

    allocation = np.empty(16 * 1024 * 1024, dtype=np.uint8)
    view = torch.from_numpy(allocation[:1])
    owned, charged = retain_result(view, byte_limit=1024)
    assert charged < 1024 and owned.untyped_storage().resizable()
    assert owned.data_ptr() != view.data_ptr()
    reference = weakref.ref(allocation)
    del allocation, view
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("location", ["input", "nested_input", "metadata", "nested_metadata", "selector"])
def test_input_string_subclasses_cannot_retain_hidden_payload(location):
    from inference_service.scheduler.runtime_controller import freeze_inputs

    class Key(str):
        pass

    host, _ = controller(target(max_retained_input_bytes=4096))
    host.accept_epoch("test")

    def attempt():
        key = Key("image")
        key.payload = np.empty(2 * 1024 * 1024, dtype=np.uint8)
        reference = weakref.ref(key.payload)
        request = {
            "input": lambda: ModelRequest({key: np.zeros(1, dtype=np.uint8)}),
            "nested_input": lambda: ModelRequest({"nested": {key: 1}}),
            "metadata": lambda: ModelRequest(metadata={key: 1}),
            "nested_metadata": lambda: ModelRequest(metadata={"nested": {key: 1}}),
            "selector": lambda: ModelRequest(identity_metadata_keys={key}),
        }[location]()
        with pytest.raises(ControlError, match="INVALID_INPUT_KEY"):
            freeze_inputs(request, 4096)
        with pytest.raises(ControlError, match="INVALID_INPUT_KEY"):
            host.prepare(request, ExecutionContext("hidden-key"))
        return reference

    try:
        reference = attempt()
        gc.collect()
        assert reference() is None
        assert not host._records and host._input_bytes == 0
    finally:
        host.shutdown()


def _array_with_dtype_payload(kind):
    payload = np.empty(2 * 1024 * 1024, dtype=np.uint8)
    dtype = np.dtype("i4", metadata={"cache": payload})
    if kind in {"field", "subarray"}:
        field = np.dtype((dtype, (2,))) if kind == "subarray" else dtype
        dtype = np.dtype([("nested", np.dtype([("value", field)]))])
    array = np.zeros(1, dtype=dtype)
    return array, weakref.ref(payload)


def test_request_id_subclass_cannot_retain_hidden_payload():
    class RequestId(str):
        pass

    host, _ = controller(target(max_records=1, max_pending_requests=1))
    host.accept_epoch("test")

    def attempt():
        request_id = RequestId("req")
        request_id.payload = np.empty(1024 * 1024, dtype=np.uint8)
        reference = weakref.ref(request_id.payload)
        with pytest.raises(ValueError, match="parent ID"):
            host.prepare(ModelRequest(), ExecutionContext(request_id))
        return reference

    try:
        reference = attempt()
        gc.collect()
        assert reference() is None
        assert not host._records and not host._preparing and host._input_bytes == 0
        prepare(host)
    finally:
        host.shutdown()


@pytest.mark.parametrize("field_name", ["x", "x" * (1 << 20)], ids=["plain", "oversized"])
def test_structured_numpy_inputs_are_rejected_even_without_data(field_name):
    from inference_service.scheduler.runtime_controller import freeze_inputs

    request = ModelRequest({"x": np.empty(0, dtype=[(field_name, "u1")])})
    host, _ = controller(target(max_retained_input_bytes=1024))
    host.accept_epoch("test")
    try:
        with pytest.raises(ControlError, match="OPAQUE_INPUT"):
            freeze_inputs(request, 1024, require_profile=False)
        with pytest.raises(ControlError, match="OPAQUE_INPUT"):
            host.prepare(request, ExecutionContext("structured"))
        assert not host._records and not host._preparing and host._input_bytes == 0
    finally:
        host.shutdown()


@pytest.mark.parametrize("view", [False, True])
def test_structured_numpy_results_and_bases_are_rejected_even_without_data(view):
    from inference_service.scheduler.runtime_controller import retain_result

    array = np.empty(0, dtype=[("x" * (1 << 20), "u1")])
    with pytest.raises(ControlError, match="OPAQUE_RESULT"):
        retain_result(array.view(np.uint8) if view else array, byte_limit=1024)


@pytest.mark.parametrize("kind", ["plain", "field", "subarray"])
def test_input_dtype_metadata_is_rejected_without_retaining_its_payload(kind):
    from inference_service.scheduler.runtime_controller import freeze_inputs

    host, _ = controller(target(max_retained_input_bytes=1024))
    host.accept_epoch("test")

    def attempt():
        array, reference = _array_with_dtype_payload(kind)
        request = ModelRequest({"x": array})
        with pytest.raises(ControlError, match="OPAQUE_INPUT"):
            freeze_inputs(request, 1024)
        with pytest.raises(ControlError, match="OPAQUE_INPUT"):
            host.prepare(request, ExecutionContext("hidden-dtype"))
        return reference

    try:
        reference = attempt()
        gc.collect()
        assert reference() is None
        assert not host._records and host._input_bytes == 0
    finally:
        host.shutdown()


@pytest.mark.parametrize("kind", ["plain", "field", "subarray"])
@pytest.mark.parametrize("view", [False, True])
def test_result_dtype_metadata_in_array_or_base_cannot_bypass_quota(kind, view):
    host, executor = controller(target(max_retained_result_bytes=8192))
    host.accept_epoch("test")
    references = []

    def execute(*_):
        array, reference = _array_with_dtype_payload(kind)
        references.append(reference)
        return {"value": array.view(np.uint8) if view else array}

    executor.execute = execute
    try:
        sub = prepare(host)
        host.execute(sub, "test")
        host.completion(sub).result(timeout=5)
        with pytest.raises(ControlError, match="OPAQUE_RESULT"):
            host.result(sub)
        gc.collect()
        assert references[0]() is None
        assert host._result_bytes == 0
    finally:
        host.shutdown()


@pytest.mark.parametrize("per_channel", [False, True])
def test_quantized_results_are_outside_the_supported_storage_boundary(per_channel):
    import torch

    host, executor = controller(target(max_retained_result_bytes=8192))
    host.accept_epoch("test")

    def execute(*_):
        data = torch.zeros((4096, 1))
        if per_channel:
            result = torch.quantize_per_channel(
                data, torch.ones(4096), torch.zeros(4096, dtype=torch.int64), 0, torch.qint8
            )
        else:
            result = torch.quantize_per_tensor(data, 1.0, 0, torch.qint8)
        return {"value": result}

    executor.execute = execute
    try:
        sub = prepare(host)
        host.execute(sub, "test")
        host.completion(sub).result(timeout=5)
        with pytest.raises(ControlError, match="OPAQUE_RESULT"):
            host.result(sub)
        assert host._result_bytes == 0
    finally:
        host.shutdown()


@pytest.mark.parametrize("output", ["numpy_export", "exported_tensor", "input", "input_view"])
def test_standard_torch_outputs_survive_managed_result_delivery(output):
    import torch

    host, executor = controller(target(max_retained_result_bytes=4096))
    host.accept_epoch("test")

    def execute(request, context):
        if output in {"input", "input_view"}:
            value = request.inputs["x"]
            return {"value": value if output == "input" else value[:1]}
        value = torch.arange(4, dtype=torch.float32)
        array = value.numpy()
        assert not value.untyped_storage().resizable()
        return {"value": array if output == "numpy_export" else value}

    executor.execute = execute
    try:
        sub = host.prepare(ModelRequest({"x": torch.arange(4, dtype=torch.float32)}), ExecutionContext("identity"))
        host.execute(sub, "test")
        host.completion(sub).result(timeout=5)
        result = host.result(sub).outputs["value"]
        np.testing.assert_array_equal(result, [0] if output == "input_view" else [0, 1, 2, 3])
        assert host._input_bytes == host._result_bytes == 0
    finally:
        host.shutdown()


@pytest.mark.parametrize("kind", ["dataclass", "list", "dict"])
def test_domain_objects_cannot_hide_unaccounted_payloads(kind):
    from dataclasses import dataclass

    @dataclass
    class Score:
        score: float = 1.0

    class Scores(list):
        pass

    class Labels(dict):
        pass

    host, executor = controller(target(max_retained_result_bytes=4096))
    host.accept_epoch("test")
    references = []

    def execute(*_):
        value = {"dataclass": Score, "list": Scores, "dict": Labels}[kind]()
        value._cache = np.empty(1024 * 1024, dtype=np.uint8)
        references.append(weakref.ref(value._cache))
        return {"value": value}

    executor.execute = execute
    try:
        sub = prepare(host)
        host.execute(sub, "test")
        host.completion(sub).result(timeout=5)
        with pytest.raises(ControlError, match="OPAQUE_RESULT"):
            host.result(sub)
        gc.collect()
        assert references[0]() is None
        assert host._result_bytes == 0
    finally:
        host.shutdown()


@pytest.mark.parametrize("cancel_before_start", [False, True])
def test_terminal_history_drops_caller_context_closures(cancel_before_start):
    from datetime import datetime, timezone

    from inference_service.unified_runtime import CancellationToken

    host, _ = controller(target())
    host.accept_epoch("test")

    def request():
        payload = np.empty(1024 * 1024, np.uint8)
        token = CancellationToken()
        token.add_callback(lambda: payload.size)
        deadline = Deadline.after(5, clock=lambda: datetime.now(timezone.utc) if payload.size else None)
        sub = host.prepare(ModelRequest(), ExecutionContext("closure", deadline, token))
        return sub, weakref.ref(payload)

    try:
        sub, reference = request()
        if cancel_before_start:
            host.cancel(sub, "test")
        else:
            host.execute(sub, "test")
            host.completion(sub).result(timeout=5)
        host.release_result_waiter(sub)
        assert sub.operation_id in host._records
        assert host._input_bytes == host._result_bytes == 0
        gc.collect()
        assert reference() is None
    finally:
        host.shutdown()


def test_result_accounting_keeps_control_responsive_and_shutdown_owned(monkeypatch):
    from inference_service.scheduler import runtime_controller
    from inference_service.unified_runtime.errors import DeadlineExceeded

    host, executor = controller(target())
    host.accept_epoch("test")
    executor.execute = lambda *_: {"coordinates": list(range(10000))}
    entered, release = threading.Event(), threading.Event()
    original = runtime_controller.retain_result

    def measure(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime_controller, "retain_result", measure)
    sub = prepare(host)
    workers = ThreadPoolExecutor(1)
    try:
        host.execute(sub, "test")
        assert entered.wait(2)

        def control():
            assert host.query(sub, "test")["state"] == "running"
            assert host.cancel(sub, "test")["state"] == "running"
            host.acknowledge(sub, "test")
            assert not host.status()["clean"]
            with pytest.raises(DeadlineExceeded):
                host.shutdown(ExecutionContext("shutdown", Deadline.after(0.02)))
            assert host._running == 1
            host.release_result_waiter(sub)

        workers.submit(control).result(timeout=0.5)
    finally:
        release.set()
        workers.shutdown(wait=True)
        host.shutdown()
    assert host._input_bytes == host._result_bytes == host._running == 0


def test_failure_delivery_copies_diagnostics_without_blocking_control(monkeypatch):
    from inference_service.scheduler import runtime_controller
    from inference_service.unified_runtime import ExecutionFailure, OutcomeEvidence

    host, executor = controller(target())
    host.accept_epoch("test")
    executor.error = ExecutionFailure(
        code="domain_failed",
        message="domain failed",
        evidence=OutcomeEvidence.completed(),
        details={"values": list(range(10000))},
    )
    entered, release = threading.Event(), threading.Event()
    workers = ThreadPoolExecutor(2)
    try:
        sub = prepare(host)
        host.execute(sub, "test")
        host.completion(sub).result(timeout=5)
        original = runtime_controller._copy_failure

        def copy(failure):
            entered.set()
            assert release.wait(5)
            return original(failure)

        monkeypatch.setattr(runtime_controller, "_copy_failure", copy)
        delivery = workers.submit(host.result, sub)
        assert entered.wait(2)

        def control():
            assert host.query(sub, "test")["state"] == "completed"
            assert host.cancel(sub, "test")["state"] == "completed"
            assert host.status()["ready"]
            host.acknowledge(sub, "test")
            host.release_result_waiter(sub)

        workers.submit(control).result(timeout=0.5)
        release.set()
        with pytest.raises(ExecutionFailure) as failed:
            delivery.result(timeout=2)
        assert failed.value is not executor.error
        assert failed.value.details["values"] == tuple(range(10000))
        assert host._records[sub.operation_id].failure is None
    finally:
        release.set()
        workers.shutdown(wait=True)
        host.shutdown()


def test_concurrent_result_preparation_rechecks_shared_quota_at_commit(monkeypatch):
    from inference_service.scheduler import runtime_controller

    host, executor = controller(target(max_in_flight=2, max_retained_result_bytes=4096))
    host.accept_epoch("test")
    executor.execute = lambda *_: {"payload": b"x" * 2048}
    barrier = threading.Barrier(2)
    original = runtime_controller.retain_result

    def measure(*args, **kwargs):
        result = original(*args, **kwargs)
        barrier.wait(timeout=3)
        return result

    monkeypatch.setattr(runtime_controller, "retain_result", measure)
    try:
        submissions = [prepare(host), prepare(host)]
        for sub in submissions:
            host.execute(sub, "test")
        for sub in submissions:
            host.completion(sub).result(timeout=5)
        reports = [host.query(sub, "test") for sub in submissions]
        assert all(report["state"] == "completed" for report in reports)
        assert sorted(report["error"] for report in reports) == ["", "RESULT_CAPACITY"]
        assert 2048 < host._result_bytes <= 4096
    finally:
        host.shutdown()


@pytest.mark.parametrize("abandon_before_completion", [False, True])
def test_abandoned_terminal_payload_is_released_before_retry_record(abandon_before_completion):
    host, executor = controller(target())
    host.accept_epoch("test")
    references = []
    release = threading.Event()

    def execute(request, context):
        assert release.wait(2)
        payload = np.empty(8 * 1024 * 1024, dtype=np.uint8)
        references.append(weakref.ref(payload))
        return {"payload": payload}

    executor.execute = execute
    try:
        sub = prepare(host, retain_for_caller=True)
        host.execute(sub, "test")
        if abandon_before_completion:
            host.release_result_waiter(sub)
        release.set()
        host.completion(sub).result(timeout=2)
        host.acknowledge(sub, "test")
        host.release_result_waiter(sub)
        gc.collect()
        assert references[0]() is None
        assert host._result_bytes == 0
        assert host.query(sub, "test")["state"] == "completed"
        assert sub.operation_id in host._records
    finally:
        release.set()
        host.shutdown()


def test_host_execution_deadline_keeps_monotonic_clock(monkeypatch):
    from datetime import datetime, timedelta, timezone

    from inference_service.scheduler import runtime_controller

    monotonic = [time.monotonic_ns()]
    wall = [datetime.now(timezone.utc)]

    class WallClock:
        @staticmethod
        def now(*_):
            return wall[0]

    monkeypatch.setattr(runtime_controller, "datetime", WallClock)
    host, executor = controller(target(), now_ns=lambda: monotonic[0])
    host.accept_epoch("test")
    remaining = []

    def execute(request, context):
        remaining.append(context.deadline.remaining_seconds())
        wall[0] += timedelta(days=1)
        monotonic[0] += 100_000_000
        remaining.append(context.deadline.remaining_seconds())
        return {}

    executor.execute = execute
    try:
        sub = prepare(host)
        host.execute(sub, "test")
        host.completion(sub).result(timeout=2)
        assert remaining[0] - remaining[1] == pytest.approx(0.1, abs=0.00001)
    finally:
        host.shutdown()


@pytest.mark.parametrize("interruption", ["cancel", "fence", "failure", "none"])
def test_snapshot_preparation_releases_control_lock_and_reserves_capacity(monkeypatch, interruption):
    from inference_service.scheduler import runtime_controller

    host, _ = controller(target(max_pending_requests=2))
    host.accept_epoch("test")
    existing = prepare(host)
    entered, release = threading.Event(), threading.Event()
    original = runtime_controller.freeze_inputs

    def freeze(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        if interruption == "failure":
            raise ValueError("copy failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime_controller, "freeze_inputs", freeze)
    context = ExecutionContext("preparing", Deadline.after(5))
    with ThreadPoolExecutor(2) as workers:
        pending = workers.submit(host.prepare, ModelRequest({"x": np.arange(10)}), context)
        try:
            assert entered.wait(1)

            def controls():
                assert not host.status()["clean"]
                assert host.query(existing, host.epoch)["state"] == "prepared"
                with pytest.raises(ControlError, match="REQUEST_CAPACITY"):
                    prepare(host)
                host.cancel(existing, host.epoch)
                if interruption == "cancel":
                    host.cancel_pending("preparing")
                elif interruption == "fence":
                    host.fence()

            workers.submit(controls).result(timeout=1)
            release.set()
            if interruption == "none":
                sub = pending.result(timeout=1)
                assert host.query(sub, host.epoch)["state"] == "prepared"
                host.cancel(sub, host.epoch)
            else:
                with pytest.raises(
                    Exception,
                    match={
                        "cancel": "cancel",
                        "fence": "RUNTIME_FENCED",
                        "failure": "copy failed",
                    }[interruption],
                ):
                    pending.result(timeout=1)
            assert host._input_bytes == 0 and not host._preparing
            context.check("reuse-after-input-preparation")
        finally:
            release.set()
            host.shutdown()


@pytest.mark.parametrize("kind", ["numpy", "torch"])
@pytest.mark.parametrize("shape", [(), (0,), (3, 4)])
def test_snapshot_hashes_exact_contiguous_private_tensor_bytes(kind, shape):
    from inference_service.scheduler.runtime_controller import freeze_inputs, input_bytes

    values = np.arange(np.prod(shape, dtype=int), dtype=np.float32).reshape(shape)
    if len(shape) == 2:
        values = values.T
    if kind == "torch":
        import torch

        values = torch.from_numpy(values)
    request = ModelRequest({"tensor": values})
    frozen, digest, size, bucket = freeze_inputs(request, 4096, ("tensor",))
    assert input_bytes(request) == size
    assert digest == freeze_inputs(request, 4096, ("tensor",))[1]
    expected = np.asarray(values).copy()
    values[...] = 9
    assert np.array_equal(np.asarray(frozen.inputs["tensor"]), expected)
    changed = freeze_inputs(request, 4096, ("tensor",))
    assert changed[3] == bucket
    if expected.size:
        assert changed[1] != digest


@pytest.mark.parametrize("copy_ns", [150_000_000, 600_000_000])
def test_input_copy_preserves_original_deadline_and_rejects_expired_input(monkeypatch, copy_ns):
    from inference_service.scheduler import runtime_controller

    clock = [time.monotonic_ns()]
    initial = clock[0]
    host, _ = controller(target(), now_ns=lambda: clock[0])
    host.accept_epoch("test")
    original = runtime_controller.freeze_inputs

    def freeze(*args, **kwargs):
        clock[0] += copy_ns
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime_controller, "freeze_inputs", freeze)
    context = ExecutionContext("copy-deadline", Deadline.after(0.5))
    monkeypatch.setattr(Deadline, "remaining_seconds", lambda self: 0.5)
    try:
        if copy_ns > 500_000_000:
            with pytest.raises(ControlError, match="DEADLINE_EXPIRED"):
                host.prepare(ModelRequest(), context)
            assert not host._records and host._input_bytes == 0
        else:
            sub = host.prepare(ModelRequest(), context)
            assert sub.deadline_ns == initial + 500_000_000
    finally:
        host.shutdown()


@pytest.mark.parametrize(
    ("started", "known", "expected"),
    [(False, True, "not_started"), (True, True, "completed"), (True, False, "unknown")],
)
def test_backend_failure_preserves_outcome_certainty(started, known, expected):
    from inference_service.backends.errors import BackendInferenceError
    from inference_service.unified_runtime.errors import ExecutionFailure

    t = target()
    core = RuntimeScheduler([t])
    host, executor = controller(t)
    try:
        connect(core, host)
        executor.error = BackendInferenceError("backend failed", operation_started=started, outcome_known=known)
        sub = prepare(host)
        core.submit(sub)
        core.dispatchable()
        host.execute(sub, core.epoch)
        host._records[sub.operation_id].done.result(timeout=1)
        report = host.query(sub, core.epoch)
        assert report["state"] == expected
        assert report["error"]
        core.report(report)
        assert core.resource_ready(t.hardware_resource_id) == known
        assert host.status()["ready"] == known
        assert (host._input_bytes == 0) == known
        with pytest.raises(ExecutionFailure) as failure:
            host.result(sub)
        assert failure.value.evidence.outcome_known == known
        if known:
            executor.error = None
            next_sub = prepare(host)
            core.submit(next_sub)
            assert core.dispatchable() == [(t, next_sub)]
            host.execute(next_sub, core.epoch)
            host._records[next_sub.operation_id].done.result(timeout=1)
            assert host.result(next_sub).outputs["answer"][0] == 1
    finally:
        host.shutdown()


@pytest.mark.parametrize("started, known", [(False, True), (True, True), (True, False)])
def test_failed_result_does_not_retain_business_inputs(started, known):
    from inference_service.backends.errors import BackendInferenceError
    from inference_service.unified_runtime.errors import ExecutionFailure

    host, executor = controller(target())
    host.accept_epoch("test")
    executor.error = BackendInferenceError("backend failed", operation_started=started, outcome_known=known)

    def business_waiter():
        image = np.zeros((1024, 1024), dtype=np.uint8)
        request = ModelRequest({"image": image})
        sub = host.prepare(request, ExecutionContext("business", Deadline.after(10)))
        host.execute(sub, "test")
        host._records[sub.operation_id].done.result(timeout=2)
        expected_failure = host._records[sub.operation_id].failure.to_dict()
        for _ in range(2):
            try:
                host.result(sub)
            except ExecutionFailure as exc:
                assert exc.to_dict() == expected_failure
            else:
                pytest.fail("backend failure was not delivered")
        host.release_result_waiter(sub)
        return weakref.ref(image), sub

    try:
        image_ref, sub = business_waiter()
        gc.collect()
        # Replay/diagnostic records remain alive, but must not own the caller's
        # stack. UNKNOWN still owns its frozen input and device reservation.
        assert image_ref() is None
        assert host.query(sub, "test")["error"]
        assert host.handle.last_failure.evidence.outcome_known == known
        assert (host._input_bytes == 0) == known
    finally:
        host.shutdown()


def test_unexpected_failure_retains_only_structured_evidence_without_caller_frames(monkeypatch):
    from inference_service.unified_runtime.errors import ExecutionFailure

    host, _ = controller(target())
    host.accept_epoch("test")

    def fail(*args, **kwargs):
        raise ValueError("unexpected runtime failure")

    monkeypatch.setattr(host.handle, "execute", fail)

    def business_waiter(sub):
        image = np.zeros((1024, 1024), dtype=np.uint8)
        try:
            host.result(sub)
        except ExecutionFailure as exc:
            assert exc.code == "ValueError"
            assert exc.message == "unexpected runtime failure"
            assert exc.cause is None and not exc.evidence.outcome_known
        else:
            pytest.fail("unexpected failure was not delivered")
        return weakref.ref(image)

    try:
        sub = prepare(host)
        host.execute(sub, "test")
        host._records[sub.operation_id].done.result(timeout=2)
        image_ref = business_waiter(sub)
        gc.collect()
        assert image_ref() is None
        assert host.query(sub, "test")["state"] == "unknown"
        with pytest.raises(ExecutionFailure, match="unexpected runtime failure"):
            host.result(sub)
    finally:
        host.shutdown()


@pytest.mark.parametrize("diagnostic", ["message", "details", "small_details", "evidence", "cause"])
@pytest.mark.parametrize("known", [False, True])
def test_failure_payload_obeys_result_capacity_and_does_not_retain_cause(diagnostic, known):
    from inference_service.scheduler.runtime_controller import retain_result
    from inference_service.unified_runtime import ExecutionFailure, OutcomeEvidence

    host, executor = controller(target(max_retained_result_bytes=4096))
    host.accept_epoch("test")
    references = []

    def execute(request, context):
        data = np.empty(1024 * 1024, dtype=np.uint8)
        references.append(weakref.ref(data))
        cause = ValueError(data) if diagnostic == "cause" else None
        raise ExecutionFailure(
            code="domain_failed",
            message="x" * (1024 * 1024 if diagnostic == "message" else 4),
            details={"large": "x" * (1024 * 1024 if diagnostic == "details" else 128)}
            if diagnostic in {"details", "small_details"}
            else {},
            cause=cause,
            evidence=OutcomeEvidence.started(
                outcome_known=known, **({"large": "x" * 1024 * 1024} if diagnostic == "evidence" else {})
            ),
        )

    executor.execute = execute
    try:
        sub = prepare(host)
        host.execute(sub, "test")
        host.completion(sub).result(timeout=2)
        gc.collect()
        assert references[0]() is None
        assert (host._input_bytes == 0) is known
        assert host._result_bytes <= 4096
        saved = host.handle.last_failure
        assert retain_result((saved.code, saved.message, saved.details, saved.evidence, saved.recovery))[1] <= 4096
        assert saved.code == "domain_failed" and saved.evidence.outcome_known is known
        assert saved.recovery.action.value == ("none" if known else "reload")
        assert host.handle.local_ready is known
        assert host.handle.health.failure_count == 1
        assert retain_result(host.handle.diagnostics().last_outcome)[1] < 4096
        assert "large" not in saved.details and not saved.evidence.details
        if diagnostic in {"cause", "small_details"}:
            with pytest.raises(ExecutionFailure) as error:
                host.result(sub)
            assert error.value.cause is None and error.value.code == "domain_failed"
            if diagnostic == "small_details":
                assert error.value.details["large"] == "x" * 128
            host.release_result_waiter(sub)
            # UNKNOWN still owns its charged failure until completion is proven.
            assert (host._result_bytes == 0) is known
        else:
            assert host.query(sub, "test")["error"] == "RESULT_CAPACITY"
            # Payload rejection must not turn an uncertain operation terminal.
            with pytest.raises(ControlError, match="RESULT_CAPACITY" if known else "OUTCOME_UNKNOWN"):
                host.result(sub)
    finally:
        host.shutdown()


def test_opaque_output_is_rejected_without_losing_completion_evidence():
    host, executor = controller(target())
    host.accept_epoch("test")
    executor.execute = lambda *_: {"sdk_object": object()}
    try:
        sub = prepare(host)
        host.execute(sub, "test")
        host.completion(sub).result(timeout=2)
        report = host.query(sub, "test")
        assert report["state"] == "completed" and report["error"] == "OPAQUE_RESULT"
        assert host._result_bytes == 0
    finally:
        host.shutdown()


@pytest.mark.parametrize("dtype", ["float32", "int64", "bfloat16", "bool"])
@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_torch_freeze_preserves_scalar_empty_and_strided_inputs(dtype, shape):
    import torch

    from inference_service.scheduler.runtime_controller import freeze_inputs

    tensor = torch.ones(shape, dtype=getattr(torch, dtype))
    if tensor.ndim == 2:
        tensor = tensor.t()
    original = tensor.clone()
    frozen, digest, _, bucket = freeze_inputs(ModelRequest({"value": tensor}), 4096)
    tensor.zero_()
    assert torch.equal(frozen.inputs["value"], original)
    assert frozen.inputs["value"].shape == original.shape
    assert frozen.inputs["value"].dtype == original.dtype
    assert freeze_inputs(ModelRequest({"value": original}), 4096)[1] == digest
    assert (freeze_inputs(ModelRequest({"value": tensor}), 4096)[3] != bucket) == bool(tensor.numel())
    if tensor.numel():
        assert freeze_inputs(ModelRequest({"value": tensor}), 4096)[1] != digest


def test_input_copy_runtime_error_is_not_started_and_does_not_quarantine(monkeypatch):
    from inference_service.scheduler import runtime_controller
    from inference_service.scheduler.runtime_protocol import NotStartedControlError
    from inference_service.scheduler.runtime_ros import SchedulerRuntimeAdapter

    host, executor = controller(target())
    host.accept_epoch("test")
    adapter = SchedulerRuntimeAdapter.__new__(SchedulerRuntimeAdapter)
    adapter.controller = host
    sent = []
    adapter._call = lambda *args: sent.append(args)

    copy_error = RuntimeError("tensor copy failed")

    def broken_copy(*args, **kwargs):
        raise copy_error

    monkeypatch.setattr(runtime_controller, "freeze_inputs", broken_copy)
    try:
        with pytest.raises(NotStartedControlError, match="INVALID_REQUEST") as failure:
            adapter.execute(ModelRequest({}), ExecutionContext.create("copy", timeout=1))
        assert failure.value.operation_started is False
        assert failure.value.outcome_known is True
        assert failure.value.__cause__ is copy_error
        assert not sent and not executor.calls and not host._records
        assert host.status()["ready"] and host.status()["clean"]
    finally:
        host.shutdown()


@pytest.mark.parametrize("enqueued", [False, True])
@pytest.mark.parametrize("work", ["execute", "reset", "close"])
def test_worker_submission_failure_cannot_start_enqueued_work(monkeypatch, enqueued, work):
    host, model = controller(target())
    host.accept_epoch("test")
    actual_submit = host._workers.submit
    submitted = []

    def failing_submit(*args):
        if enqueued:
            submitted.append(actual_submit(*args))
        raise RuntimeError("cannot start worker")

    monkeypatch.setattr(host._workers, "submit", failing_submit)
    try:
        sub = host.prepare(ModelRequest(), ExecutionContext("submit-failure", Deadline.after(1)), work=work)
        assert host.execute(sub, "test")["state"] == "not_started"
        for future in submitted:
            future.result(timeout=1)
        assert not model.calls and not host.handle.closed
        assert host.completion(sub).done()
        assert host._input_bytes == host._running == 0
        assert host.ready
        # A subsequent operation uses ordinary admission after a proven unsent failure.
        monkeypatch.setattr(host._workers, "submit", actual_submit)
        successful = prepare(host)
        host.execute(successful, "test")
        host.completion(successful).result(timeout=1)
        assert len(model.calls) == 1
    finally:
        host.shutdown()
        host.handle.close()


def test_unsupported_drain_does_not_fence_a_healthy_handle():
    host, model = controller(target())
    host.accept_epoch("test")
    try:
        sub = host.prepare(ModelRequest(), ExecutionContext("drain", Deadline.after(1)), work="drain")
        host.execute(sub, "test")
        host.completion(sub).result(timeout=1)
        assert host.query(sub, "test")["state"] == "not_started"
        assert host.query(sub, "test")["error"] == "backend_drain_unsupported"
        assert host.ready and host.generation == 1
        normal = prepare(host)
        host.execute(normal, "test")
        host.completion(normal).result(timeout=1)
        assert len(model.calls) == 1
        close = host.prepare(ModelRequest(), ExecutionContext("close", Deadline.after(1)), work="close")
        host.execute(close, "test")
        host.completion(close).result(timeout=1)
        assert host.handle.closed
    finally:
        host.shutdown()
        host.handle.close()


def test_optional_profile_stops_scalar_hashing_but_keeps_exact_digest(monkeypatch):
    from inference_service.scheduler import runtime_controller

    sha256 = runtime_controller.hashlib.sha256
    calls = []

    def counted_hash(*args):
        calls.append(None)
        return sha256(*args)

    monkeypatch.setattr(runtime_controller.hashlib, "sha256", counted_hash)
    items = list(range(1000))
    first = runtime_controller.freeze_inputs(ModelRequest({"items": items}), 1000000, require_profile=False)
    assert first[3] == "" and len(calls) < 100
    calls.clear()
    items[-1] += 1
    second = runtime_controller.freeze_inputs(ModelRequest({"items": items}), 1000000, require_profile=False)
    assert second[1] != first[1] and second[3] == "" and len(calls) < 100


def test_managed_execution_skips_domain_wait_but_retains_instance_guard():
    from inference_service.unified_runtime.admission import NativeAdmission

    class Domain:
        acquired = 0

        def acquire(self, deadline):
            self.acquired += 1

        def release(self):
            pass

        def close(self):
            pass

    domain = Domain()
    provider = SimpleNamespace(
        register_instance=lambda *_: SimpleNamespace(close=lambda: None), register=lambda *_: domain
    )
    admission = NativeAdmission(provider, "test", SimpleNamespace(resource_domain="gpu", max_in_flight_per_instance=1))
    with admission.admit(managed=True):
        assert domain.acquired == 0
        with pytest.raises(RuntimeError, match="capacity mismatch"), admission.admit(managed=True):
            pass
    with admission.admit():
        assert domain.acquired == 1
    admission.close()


@pytest.mark.parametrize("kind", ["deep", "mapping", "autograd", "leaf_gradient"])
def test_invalid_result_releases_device_and_preserves_known_completion(kind):
    from collections.abc import Mapping

    class BrokenMapping(Mapping):
        def __len__(self):
            return 1

        def __iter__(self):
            raise ValueError("result property failed")

        def __getitem__(self, key):
            raise ValueError("result property failed")

    if kind == "deep":
        value = 0
        for _ in range(2000):
            value = [value]
        expected = "RESULT_NESTING_TOO_DEEP"
    elif kind == "mapping":
        value, expected = BrokenMapping(), "OPAQUE_RESULT"
    else:
        import torch

        value = torch.ones(1024 * 1024, requires_grad=True)
        if kind == "leaf_gradient":
            value.grad = torch.ones_like(value)
            assert value.grad_fn is None and value._base is None
        else:
            value = value.square().sum()
        expected = "RESULT_RETAINS_AUTOGRAD"
    host, executor = controller(target(max_retained_result_bytes=6 * 1024 * 1024))
    core = RuntimeScheduler([host.target])
    connect(core, host)
    executor.execute = lambda *_: {"value": value}
    try:
        sub = prepare(host)
        core.submit(sub)
        assert core.dispatchable() == [(host.target, sub)]
        host.execute(sub, core.epoch)
        host.completion(sub).result(timeout=2)
        report = host.query(sub, core.epoch)
        assert report["state"] == "completed" and report["error"] == expected
        core.report(report)
        assert host._input_bytes == host._result_bytes == 0
        assert host.handle.local_ready
        next_sub = prepare(host)
        core.submit(next_sub)
        assert core.dispatchable() == [(host.target, next_sub)]
    finally:
        host.shutdown()


def test_result_budget_rejects_container_before_visiting_members():
    from inference_service.scheduler.runtime_controller import retain_result

    # An opaque member would raise OPAQUE_RESULT if visited before the
    # container/string charge rejects the whole result.
    with pytest.raises(ControlError, match="RESULT_CAPACITY"):
        retain_result([object()] * 10000, byte_limit=1024)
    with pytest.raises(ControlError, match="RESULT_CAPACITY"):
        retain_result([b"x" * 2048, object()], byte_limit=1024)


def test_detached_torch_result_has_bounded_storage():
    import torch

    from inference_service.scheduler.runtime_controller import retain_result

    result = torch.ones(1024 * 1024, requires_grad=True).square().sum()
    assert retain_result(result.detach(), byte_limit=1024)[1] < 1024


def test_no_grad_result_view_cannot_retain_unaccounted_autograd_inputs():
    import torch

    host, executor = controller(target(max_retained_result_bytes=4096))
    host.accept_epoch("test")
    references = []

    def execute(*_):
        allocation = torch.ones(4 * 1024 * 1024, requires_grad=True)
        references.append(weakref.ref(allocation))
        result = allocation.square().sum()
        with torch.no_grad():
            view = result.view(1)
        assert view.grad_fn is None and view._base.grad_fn is not None
        return {"value": view}

    executor.execute = execute
    try:
        sub = prepare(host)
        host.execute(sub, "test")
        host.completion(sub).result(timeout=5)
        report = host.query(sub, "test")
        assert report["state"] == "completed" and report["error"] == "RESULT_RETAINS_AUTOGRAD"
        assert host._result_bytes == 0 and host.handle.local_ready
        gc.collect()
        assert references[0]() is None
    finally:
        host.shutdown()


@pytest.mark.parametrize("interruption", ["deadline", "cancel"])
def test_preparation_checks_context_between_measurement_and_freeze(monkeypatch, interruption):
    from datetime import datetime, timedelta, timezone

    from inference_service.scheduler import runtime_controller
    from inference_service.unified_runtime.errors import CancellationRequested, DeadlineExceeded

    host, _ = controller(target())
    host.accept_epoch("test")
    clock = [datetime.now(timezone.utc)]
    context = ExecutionContext("prepare", Deadline.after(1, now=clock[0], clock=lambda: clock[0]))
    original = runtime_controller.input_bytes

    def measure(*args, **kwargs):
        result = original(*args, **kwargs)
        if interruption == "deadline":
            clock[0] += timedelta(seconds=2)
        else:
            context.cancellation_token.cancel("stop preparation")
        return result

    monkeypatch.setattr(runtime_controller, "input_bytes", measure)
    monkeypatch.setattr(runtime_controller, "freeze_inputs", lambda *a, **kw: pytest.fail("interrupted input copied"))
    try:
        with pytest.raises(DeadlineExceeded if interruption == "deadline" else CancellationRequested):
            host.prepare(ModelRequest({"values": list(range(1000))}), context)
        assert not host._preparing and not host._records and host._input_bytes == 0
    finally:
        host.shutdown()


@pytest.mark.parametrize("interruption", ["deadline", "cancel"])
@pytest.mark.parametrize("stage", ["measurement", "freeze"])
@pytest.mark.parametrize("container", ["sequence", "mapping"])
def test_large_input_traversal_stops_after_context_interruption(monkeypatch, interruption, stage, container):
    from datetime import datetime, timedelta, timezone

    from inference_service.scheduler import runtime_controller
    from inference_service.unified_runtime.errors import CancellationRequested, DeadlineExceeded

    host, _ = controller(target())
    host.accept_epoch("test")
    clock = [datetime.now(timezone.utc)]
    context = ExecutionContext("prepare", Deadline.after(1, now=clock[0], clock=lambda: clock[0]))
    original_measure = runtime_controller.input_bytes
    overhead = runtime_controller._input_overhead
    active = [stage == "measurement"]
    visits = [0]

    def measure(*args, **kwargs):
        result = original_measure(*args, **kwargs)
        active[0] = stage == "freeze"
        return result

    def counted(value):
        if active[0]:
            visits[0] += 1
            if visits[0] == 300:
                if interruption == "deadline":
                    clock[0] += timedelta(seconds=2)
                else:
                    context.cancellation_token.cancel("stop preparation")
        return overhead(value)

    monkeypatch.setattr(runtime_controller, "input_bytes", measure)
    monkeypatch.setattr(runtime_controller, "_input_overhead", counted)
    values = list(range(4000)) if container == "sequence" else {str(i): i for i in range(4000)}
    try:
        with pytest.raises(DeadlineExceeded if interruption == "deadline" else CancellationRequested):
            host.prepare(ModelRequest({"values": values}), context)
        assert 300 <= visits[0] < 600
        assert not host._preparing and not host._records and host._input_bytes == 0
    finally:
        host.shutdown()


def test_scalar_identity_has_boundaries_even_in_excluded_metadata():
    from inference_service.scheduler.runtime_controller import freeze_inputs

    def snapshot(values):
        return freeze_inputs(ModelRequest(metadata={"trace": values}, identity_metadata_keys={"trace"}), 4096)

    left, right = snapshot([1, 23]), snapshot([12, 3])
    assert left[1] != right[1]
    assert left[3] == right[3]


def test_fifo_input_remains_usable_when_optional_profile_is_too_large():
    from inference_service.scheduler.runtime_controller import freeze_inputs

    request = ModelRequest({"values": list(range(1000))})
    with pytest.raises(ControlError, match="INPUT_BUCKET_TOO_LARGE"):
        freeze_inputs(request, 1024 * 1024)
    host, executor = controller(target())
    host.accept_epoch("test")
    executor.execute = lambda request, _: {"sum": sum(request.inputs["values"])}
    try:
        sub = host.prepare(request, ExecutionContext("large-fifo", Deadline.after(2)))
        assert sub.input_bucket == "" and len(sub.digest) == 64
        host.execute(sub, "test")
        host.completion(sub).result(timeout=2)
        assert host.result(sub).outputs == {"sum": sum(range(1000))}
    finally:
        host.shutdown()
