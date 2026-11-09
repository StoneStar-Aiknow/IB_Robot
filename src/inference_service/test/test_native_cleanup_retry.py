"""Native resource owners retain failed cleanup for retry."""

from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("failure_at", ["host", "buffer", "dataset", "device", "desc", "model"])
def test_acl_close_checks_status_and_resumes_without_double_free(failure_at):
    from inference_manifest import ArtifactBindings, TensorBinding
    from inference_service.backends.ascend.model import AclModel, _DatasetBuffer

    released = set()
    failed = False

    def release(kind):
        def call(pointer):
            nonlocal failed
            key = (kind, pointer)
            assert key not in released, "double release"
            if kind == failure_at and not failed:
                failed = True
                return 1
            released.add(key)
            return 0

        return call

    acl = SimpleNamespace(
        rt=SimpleNamespace(free_host=release("host"), free=release("device")),
        destroy_data_buffer=release("buffer"),
        mdl=SimpleNamespace(destroy_dataset=release("dataset"), destroy_desc=release("desc"), unload=release("model")),
    )
    model = AclModel(
        SimpleNamespace(acl=acl, bind_current_thread=lambda: None),
        "test",
        Path("x.om"),
        ArtifactBindings(
            inputs=(TensorBinding(semantic="input", index=0, dtype="float32", shape=(1,)),),
            outputs=(TensorBinding(semantic="output", index=0, dtype="float32", shape=(1,)),),
        ),
    )
    model.model_id = 1
    model.model_desc = 2
    model.output_host_buffers = [3]
    model.output_dataset = 4
    model.output_buffers = [_DatasetBuffer(5, 6, 4, True)]
    with pytest.raises(RuntimeError, match="ACL error"):
        model.close()
    assert not model._closed
    model.close()
    model.close()
    assert released == {("host", 3), ("buffer", 6), ("device", 5), ("dataset", 4), ("desc", 2), ("model", 1)}


@pytest.mark.parametrize("failure_at", ["destroy_context", "reset_device", "finalize"])
def test_acl_lease_retains_reference_counts_until_release_succeeds(failure_at):
    from inference_service.backends.ascend.acl_runtime import AclRuntimeManager

    calls = []
    failed = False

    def release(name):
        def call(*args):
            nonlocal failed
            calls.append(name)
            if name == failure_at and not failed:
                failed = True
                return 1
            return 0

        return call

    acl = SimpleNamespace(
        init=lambda: 0,
        finalize=release("finalize"),
        rt=SimpleNamespace(
            set_device=lambda _: 0,
            set_context=lambda _: 0,
            create_context=lambda _: ("context", 0),
            destroy_context=release("destroy_context"),
            reset_device=release("reset_device"),
        ),
    )
    manager = AclRuntimeManager(module_loader=lambda: acl)
    lease = manager.acquire(0)
    with pytest.raises(RuntimeError, match="ACL error"):
        lease.close()
    assert manager._references == 1 and not lease._closed
    lease.close()
    lease.close()
    assert manager._references == 0 and manager._device_references == {}
    assert calls.count(failure_at) == 2
    assert all(calls.count(name) == 1 for name in ("destroy_context", "reset_device", "finalize") if name != failure_at)


@pytest.mark.parametrize("failure_at", ["reset_device", "finalize"])
def test_jpeg_close_retries_lease_without_rebinding_a_destroyed_context(failure_at):
    from inference_service.backends.ascend.acl_runtime import AclRuntimeManager
    from observation_transport.ascend_jpeg_codec import AscendJpegDecoder

    calls, blocked = [], [True]

    def release(name):
        def call(*_):
            calls.append(name)
            return int(blocked[0] and name == failure_at)

        return call

    acl = SimpleNamespace(
        init=lambda: 0,
        finalize=release("finalize"),
        rt=SimpleNamespace(
            set_device=lambda _: 0,
            create_context=lambda _: ("context", 0),
            set_context=lambda context: 1 if context is None else 0,
            destroy_context=release("context"),
            reset_device=release("reset_device"),
            create_stream=lambda: ("stream", 0),
            destroy_stream=release("stream"),
        ),
    )
    media = SimpleNamespace(
        dvpp_create_channel_desc=lambda: "channel",
        dvpp_create_channel=lambda _: 0,
        dvpp_destroy_channel=release("channel"),
        dvpp_destroy_channel_desc=release("descriptor"),
    )
    manager = AclRuntimeManager(lambda: acl)
    decoder = AscendJpegDecoder(manager.acquire(0), media=media, owns_lease=True)
    decoder._open()
    try:
        with pytest.raises(RuntimeError, match=failure_at):
            decoder.close()
        assert manager._references == 1
    finally:
        blocked[0] = False
        decoder.close()
    decoder.close()
    assert manager._references == 0
    assert all(calls.count(name) == 1 for name in ("context", "stream", "channel", "descriptor"))


def test_priority_stream_drain_failure_does_not_destroy_stream_or_forget_owner():
    from inference_service.backends.ascend.acl_runtime import AclPriorityStreamPool

    destroyed = []
    sync = [1]
    lease = SimpleNamespace(
        bind_current_thread=lambda: None,
        acl=SimpleNamespace(
            rt=SimpleNamespace(
                synchronize_stream=lambda _: sync[0], destroy_stream=lambda stream: (destroyed.append(stream), 0)[1]
            )
        ),
    )
    pool = AclPriorityStreamPool(lease, 0, 0, {0: "stream"})
    with pytest.raises(RuntimeError, match="synchronize_stream"):
        pool.close()
    assert destroyed == [] and pool.supports(0)
    sync[0] = 0
    pool.close()
    pool.close()
    assert destroyed == ["stream"]


@pytest.mark.parametrize("failure_at", ["destroy_context", "reset_device", "finalize"])
def test_failed_acl_acquisition_keeps_owner_and_dependencies_until_retry(failure_at):
    from inference_service.backends.ascend.acl_runtime import AclRuntimeManager

    events = []
    blocked = [True]

    def call(name):
        def run(*_):
            events.append(name)
            return int(blocked[0] and name == failure_at)

        return run

    acl = SimpleNamespace(
        init=lambda: 0,
        finalize=call("finalize"),
        rt=SimpleNamespace(
            set_device=lambda _: 0,
            set_context=lambda _: 0,
            create_context=lambda _: ("partial", 1),
            destroy_context=call("destroy_context"),
            reset_device=call("reset_device"),
        ),
    )
    manager = AclRuntimeManager(lambda: acl)
    with pytest.raises(Exception, match="cleanup pending"):
        manager.acquire(0)
    order = ["destroy_context", "reset_device", "finalize"]
    assert events == order[: order.index(failure_at) + 1]
    assert manager._acl is acl and manager._references == 1
    blocked[0] = False
    manager.close()
    assert events == order[: order.index(failure_at) + 1] + order[order.index(failure_at) :]
    assert manager._references == 0 and manager._acl is None


@pytest.mark.parametrize("kind", ["pool", "model", "lease"])
def test_native_cleanup_stops_between_sdk_calls_when_budget_is_consumed(kind):
    from datetime import datetime, timedelta, timezone

    from inference_service.backends.ascend.acl_runtime import AclPriorityStreamPool, AclRuntimeManager
    from inference_service.backends.ascend.model import AclModel
    from inference_service.backends.errors import BackendLifecycleError
    from inference_service.unified_runtime import Deadline, ExecutionContext
    from inference_service.unified_runtime.errors import DeadlineExceeded
    from inference_service.unified_runtime.lifecycle import lifecycle_scope

    now = [datetime(2026, 10, 29, tzinfo=timezone.utc)]
    deadline = Deadline.after(1, now=now[0], clock=lambda: now[0])
    calls = []

    def release(name):
        def run(*_):
            calls.append(name)
            now[0] += timedelta(seconds=2)
            return 0

        return run

    acl = SimpleNamespace(
        init=lambda: 0,
        finalize=release("finalize"),
        rt=SimpleNamespace(
            set_device=lambda _: 0,
            create_context=lambda _: ("context", 0),
            set_context=lambda _: 0,
            destroy_context=release("context"),
            reset_device=release("device"),
            synchronize_stream=release("sync"),
            destroy_stream=release("stream"),
            free_host=release("host"),
        ),
        mdl=SimpleNamespace(destroy_dataset=release("dataset")),
    )
    if kind == "lease":
        owner = AclRuntimeManager(lambda: acl).acquire(0)
    else:
        lease = SimpleNamespace(acl=acl, bind_current_thread=lambda: None)
        owner = (
            AclPriorityStreamPool(lease, 1, 0, {0: "s0", 1: "s1"})
            if kind == "pool"
            else AclModel(lease, "m", Path("m.om"), None)
        )
        if kind == "model":
            owner.output_host_buffers = ["host"]
            owner.output_dataset = "dataset"
    with (
        lifecycle_scope(ExecutionContext("cleanup", deadline)),
        pytest.raises(BackendLifecycleError if kind == "lease" else DeadlineExceeded),
    ):
        owner.close()
    assert calls == [{"pool": "sync", "model": "host", "lease": "context"}[kind]]
    owner.close()
    assert owner._closed
    if kind != "pool":
        assert calls.count(calls[0]) == 1


@pytest.mark.parametrize("failure_at", ["output", "host"])
def test_bank_resources_are_registered_before_later_allocation_can_fail(monkeypatch, failure_at):
    from inference_service.backends.ascend.model import AclModel, AclTensorDescriptor

    released = []
    created = []

    def create(*_):
        if created and failure_at == "output":
            raise RuntimeError("output allocation failed")
        name = f"dataset-{len(created)}"
        created.append(name)
        return name, []

    allocations = []

    def host(_):
        if allocations:
            return None, 1
        allocations.append("host")
        return "host", 0

    acl = SimpleNamespace(rt=SimpleNamespace(malloc_host=host, free_host=lambda ptr: (released.append(ptr), 0)[1]))
    owner = AclModel(SimpleNamespace(acl=acl, bind_current_thread=lambda: None), "m", Path("m.om"), None)
    descriptor = AclTensorDescriptor(0, "x", None, (1,), 4)
    owner.input_descriptors = (descriptor,)
    owner.output_descriptors = (descriptor, AclTensorDescriptor(1, "y", None, (1,), 4))
    monkeypatch.setattr(owner, "_create_dataset", create)
    monkeypatch.setattr(owner, "_destroy_dataset", lambda dataset, _: released.append(dataset) if dataset else None)
    with pytest.raises(RuntimeError):
        owner.prepare_dataset_banks(({},), ({},), host_output_indices={0, 1})
    assert set(released) == set(created + allocations)
    assert not owner._dataset_banks and not owner._dataset_bank_host_buffers


def test_isolated_output_host_free_failure_remains_owned_until_drain():
    import numpy as np

    from inference_service.backends.ascend.model import AclModel, AclTensorDescriptor, _DatasetBuffer

    frees = []

    def free(pointer):
        frees.append(pointer)
        return int(len(frees) == 1)

    acl = SimpleNamespace(
        rt=SimpleNamespace(malloc_host=lambda _: ("host", 0), memcpy=lambda *_: 0, free_host=free),
        util=SimpleNamespace(ptr_to_bytes=lambda *_: np.ones(1, dtype=np.float32).tobytes()),
    )
    owner = AclModel(SimpleNamespace(acl=acl, bind_current_thread=lambda: None), "m", Path("m.om"), None)
    owner.output_descriptors = (AclTensorDescriptor(0, "x", np.dtype("float32"), (1,), 4),)
    with pytest.raises(RuntimeError, match="free_host"):
        owner._read_dataset_outputs([_DatasetBuffer("device", None, 4, False)], None)
    assert owner._quarantined_host_buffers == ["host"]
    owner.drain_quarantined()
    assert frees == ["host", "host"] and not owner._quarantined_host_buffers
