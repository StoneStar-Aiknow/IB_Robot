"""Builder, load and actual OM calls must agree on hardware stream priority."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from inference_manifest import CompiledDeployment, SemanticTensor
from inference_service.backends import RuntimeContext
from inference_service.unified_runtime import ExecutionContext, ModelRequest, RuntimeProviders
from perception_service.model_session_builders import build_perception_session


def binding(semantic, shape, dtype="float32", index=0):
    return {"semantic": semantic, "shape": shape, "dtype": dtype, "index": index}


@pytest.mark.parametrize("family", ["siglip2", "sam2"])
@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize(
    "stop",
    [
        "none",
        "after_first",
        "after_second",
        "after_last",
        "before_first_cancel",
        "before_first_deadline",
        "later_input_failure",
        "invalid_input",
        "host_before",
        "host_between",
        "host_after",
        "native_unknown",
    ],
)
def test_priority_builder_loads_valid_roles_and_uses_selected_stream_for_every_call(
    tmp_path, monkeypatch, family, scheduled, stop
):
    from datetime import datetime, timedelta, timezone

    from inference_service.backends import ResourceDomainAdmissions
    from inference_service.unified_runtime import (
        Deadline,
        ExecutionFailure,
        ModelRuntimeHandle,
        OutcomeState,
        RuntimeAssembly,
    )

    now = [datetime(2026, 10, 31, tzinfo=timezone.utc)]
    execution = ExecutionContext("priority", Deadline.after(1, now=now[0], clock=lambda: now[0]))
    if family == "siglip2":
        roles = {
            "vision": {
                "inputs": [binding("host.siglip2.image", [1, 3, 2, 2])],
                "outputs": [binding("host.siglip2.image_embedding", [1, 3])],
            },
            "text": {
                "inputs": [binding("host.siglip2.input_ids", [1, 4], "int64")],
                "outputs": [binding("host.siglip2.text_embeddings", [1, 3])],
            },
        }
        inputs = {
            "masked_images": np.ones((2, 3, 2, 2), np.float32),
            "text_tokens": np.ones((2, 4), np.int64),
            "text_attention_mask": np.ones((2, 4), np.int64),
        }
    else:
        shared = [
            binding(name, [1, 1, 2, 2], index=i)
            for i, name in enumerate(("internal.image_embed", "internal.high_res_feats_0", "internal.high_res_feats_1"))
        ]
        roles = {
            "encoder": {"inputs": [binding("host.sam2.image", [1, 3, 1024, 1024])], "outputs": shared},
            "decoder": {
                "inputs": shared
                + [
                    binding("host.sam2.point_coords", [1, 2, 2], index=3),
                    binding("host.sam2.point_labels", [1, 2], "int8", 4),
                    binding("host.sam2.mask_input", [1, 1, 256, 256], index=5),
                    binding("host.sam2.has_mask_input", [1], "int8", 6),
                ],
                "outputs": [
                    binding("host.sam2.mask_logits", [1, 1, 256, 256]),
                    binding("host.sam2.iou_predictions", [1, 1], index=1),
                ],
            },
        }
        inputs = {"observation.image": np.ones((2, 2, 3), np.uint8)}
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets/adapter.json").write_text(json.dumps({"points_per_side": 2}))
    for role in roles:
        (tmp_path / f"{role}.om").write_bytes(b"fake SDK artifact")
    deployment = CompiledDeployment.model_validate(
        {
            "execution_contract": {
                "state_scope": "request",
                "execution_structure": "direct",
                "cancellation_granularity": "request_boundary",
            },
            "runtime_profile": {"backend": "ascend", "target": {"runtime": "acl"}, "profile": {"device_id": 0}},
            "execution": list(roles),
            "artifacts": {role: {"path": f"{role}.om", "format": "om"} for role in roles},
            "bindings": roles,
        }
    )
    acl = SimpleNamespace(
        rt=SimpleNamespace(
            device_get_stream_priority_range=lambda: (7, 0, 0),
            create_stream_with_config=lambda priority, _: (f"stream-{priority}", 0),
            synchronize_stream=lambda _: 0,
            destroy_stream=lambda _: 0,
        )
    )
    lease = SimpleNamespace(acl=acl, bind_current_thread=lambda: None, close=lambda: None)
    providers = RuntimeProviders(SimpleNamespace(acquire=lambda _: lease), ResourceDomainAdmissions())
    identity = SimpleNamespace(model_type=family, interface="tensor_model", operation="automatic")
    output_specs = (
        {"image_embeddings": ((-1, 3), "float32"), "text_embeddings": ((-1, 3), "float32")}
        if family == "siglip2"
        else {
            "masks": ((-1, 2, 2), "uint8"),
            "boxes": ((-1, 4), "float32"),
            "scores": ((-1,), "float32"),
            "stability_scores": ((-1,), "float32"),
        }
    )
    outputs = tuple(
        SemanticTensor(semantic=name, shape=shape, dtype=dtype) for name, (shape, dtype) in output_specs.items()
    )
    context = RuntimeContext(
        SimpleNamespace(
            deployment=deployment,
            bundle_root=tmp_path,
            top_level_identity=identity,
            manifest=SimpleNamespace(model=SimpleNamespace(outputs=outputs)),
        ),
        priority_scheduling=scheduled,
    )
    session = build_perception_session(
        context, adapter=SimpleNamespace(identity=identity, compiled_abi_finalized=True), providers=providers
    )
    calls = []

    class Model:
        def __init__(self, lease, role, path, bindings, **kwargs):
            self.role, self.bindings = role, bindings

        def load_descriptor(self):
            pass

        def prepare_datasets(self, **kwargs):
            pass

        def execute(self, inputs, **kwargs):
            if stop == "later_input_failure" and self.role == list(roles)[1]:
                from inference_service.backends.errors import BackendInferenceError

                raise BackendInferenceError(
                    "input copy failed", code="input_preparation_failed", operation_started=False, outcome_known=True
                )
            calls.append((self.role, kwargs.get("stream")))
            if stop == "native_unknown":
                raise RuntimeError("native completion unknown")
            stop_after = {"after_first": 1, "after_second": 2, "after_last": 4 if family == "siglip2" else 5}
            if len(calls) == stop_after.get(stop):
                execution.token.cancel("stop between completed roles")
            return {b.index: np.zeros(b.shape, dtype=b.dtype) for b in self.bindings.outputs}

        def close(self):
            pass

    if stop.startswith("before_first"):
        original_asarray = np.asarray
        trigger = inputs["masked_images" if family == "siglip2" else "observation.image"]

        def preprocessing(value, *args, **kwargs):
            result = original_asarray(value, *args, **kwargs)
            if value is trigger:
                if stop == "before_first_cancel":
                    execution.token.cancel("cancel during host preprocessing")
                else:
                    now[0] += timedelta(seconds=2)
            return result

        monkeypatch.setattr(np, "asarray", preprocessing)
    session._model_factory = Model
    session.load(context)
    host_work = {"broadcast": 0, "postprocess": 0}
    if family == "sam2":
        broadcast = np.broadcast_to
        postprocess = session._postprocess

        def count_broadcast(*args, **kwargs):
            host_work["broadcast"] += 1
            return broadcast(*args, **kwargs)

        def count_postprocess(*args, **kwargs):
            host_work["postprocess"] += 1
            return postprocess(*args, **kwargs)

        monkeypatch.setattr(np, "broadcast_to", count_broadcast)
        monkeypatch.setattr(session, "_postprocess", count_postprocess)
    owner = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=SimpleNamespace(execute=session.execute, close=session.close),
            session=SimpleNamespace(name=session.name, capabilities=session.capabilities),
            providers=providers,
        )
    )
    owner.load()
    request = ModelRequest(inputs, metadata={"priority": 7 if scheduled else 0})
    try:
        if stop.startswith("host_"):
            with monkeypatch.context() as patch:

                def fail(*args, **kwargs):
                    raise MemoryError("host allocation failed")

                if family == "sam2":
                    target, name = (
                        (session, "_letterbox")
                        if stop == "host_before"
                        else (np, "broadcast_to")
                        if stop == "host_between"
                        else (session, "_postprocess")
                    )
                    patch.setattr(target, name, fail)
                else:
                    name = {"host_before": "asarray", "host_between": "concatenate", "host_after": "ascontiguousarray"}[
                        stop
                    ]
                    original = getattr(np, name)

                    def processing(value, *args, **kwargs):
                        if (
                            (stop == "host_before" and value is inputs["masked_images"])
                            or (stop == "host_between" and calls)
                            or (stop == "host_after" and len(calls) == 4 and value.shape == (2, 3))
                        ):
                            fail()
                        return original(value, *args, **kwargs)

                    patch.setattr(np, name, processing)
                with pytest.raises(ExecutionFailure) as failure:
                    owner.execute(request, execution)
            assert failure.value.evidence.state is (
                OutcomeState.NOT_STARTED if stop == "host_before" else OutcomeState.COMPLETED
            )
            assert failure.value.evidence.outcome_known and owner.ready and not owner._admission._retained
            assert bool(calls) is (stop != "host_before")
            calls.clear()
            owner.execute(request, ExecutionContext("valid", Deadline.after(1)))
            assert calls and owner.ready
        elif stop == "native_unknown":
            with pytest.raises(ExecutionFailure) as failure:
                owner.execute(request, execution)
            assert not failure.value.evidence.outcome_known and not owner.ready and owner._admission._retained
        elif stop == "invalid_input":
            invalid = dict(inputs)
            if family == "siglip2":
                invalid.update(
                    masked_images=np.empty((0, 3, 2, 2), np.float32),
                    text_tokens=np.ones((2, 64), np.int64),
                    text_attention_mask=np.ones((1, 64), np.int64),
                )
            else:
                invalid["observation.image"] = np.zeros((2, 2), np.uint8)
            bad_requests = [invalid]
            if family == "sam2":
                bad_requests.extend(
                    {"observation.image": np.zeros(shape, np.uint8)} for shape in ((0, 0, 3), (0, 2, 3), (2, 0, 3))
                )
            for bad in bad_requests:
                with pytest.raises(ExecutionFailure) as failure:
                    owner.execute(ModelRequest(bad, request.metadata), execution)
                assert failure.value.code == "input_shape_mismatch"
                assert failure.value.evidence.state is OutcomeState.NOT_STARTED
                assert owner.ready and calls == [] and not owner._admission._retained
            peer = ModelRuntimeHandle(
                RuntimeAssembly(
                    runtime_executor=SimpleNamespace(execute=lambda *_: {}),
                    session=SimpleNamespace(name="peer", capabilities=session.capabilities),
                    providers=providers,
                )
            )
            peer.load()
            try:
                peer.execute(ModelRequest(), ExecutionContext("peer", Deadline.after(0.1)))
            finally:
                peer.close()
            owner.execute(request, execution)
            assert calls and owner.ready
        elif stop != "none":
            with pytest.raises(ExecutionFailure) as failure:
                owner.execute(request, execution)
            assert failure.value.evidence.state is (
                OutcomeState.COMPLETED
                if stop.startswith("after_") or stop == "later_input_failure"
                else OutcomeState.NOT_STARTED
            )
            assert failure.value.evidence.outcome_known and owner.ready
            expected_calls = 2 if stop == "later_input_failure" and family == "siglip2" else 1
            expected_calls = {"after_second": 2, "after_last": 4 if family == "siglip2" else 5}.get(
                stop, expected_calls
            )
            assert len(calls) == (expected_calls if stop.startswith("after_") or stop == "later_input_failure" else 0)
            if family == "sam2" and stop.startswith("after_"):
                assert host_work == {"broadcast": 3 * (expected_calls - 1), "postprocess": 0}
        else:
            owner.execute(request, execution)
            expected = ["vision", "vision", "text", "text"] if family == "siglip2" else ["encoder"] + ["decoder"] * 4
            assert calls == [(role, "stream-7" if scheduled else None) for role in expected]
    finally:
        owner.close()
        providers.close()
