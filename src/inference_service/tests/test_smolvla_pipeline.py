from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from inference_manifest import (
    ArtifactBindings,
    BundleFile,
    DeviceLink,
    TensorBinding,
    canonical_bundle_digest,
    load_inference_manifest,
)
from inference_service.codecs import build_execution_plan
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.pipeline import (
    ExecutionControl,
    ExecutionError,
    SmolVLAEmbeddingWeights,
    SmolVLAFamilyResource,
    SmolVLATopologyError,
    StageFrame,
    create_smolvla_executor,
    derive_smolvla_topology,
    load_smolvla_embedding_weights,
)
from inference_service.pipeline.smolvla import _embedding_operation
from tests.manifest_fixtures import TEST_BUNDLE_UUID, TEST_DEPLOYMENT_UUID, create_policy_bundle, write_manifest

_HIDDEN = 4
_STATE_DIM = 8
_CHUNK = 2
_NUM_STEPS = 2
_PREFIX_LENGTH = 6
_ACTION_SHAPE = (1, _CHUNK, _STATE_DIM)


def _tensor(semantic: str, index: int, dtype: str, shape: tuple[int, ...]) -> TensorBinding:
    is_image = semantic.startswith("observation.images.") or semantic.startswith("observation.image.")
    layout = "NCHW" if len(shape) == 4 and is_image else None
    return TensorBinding(semantic=semantic, runtime_name=semantic, index=index, dtype=dtype, shape=shape, layout=layout)


def _image_tensor(semantic: str, index: int, dtype: str, shape: tuple[int, ...]) -> TensorBinding:
    return TensorBinding(semantic=semantic, runtime_name=semantic, index=index, dtype=dtype, shape=shape, layout="NCHW")


class _ResultAdapter:
    @staticmethod
    def adapt(frame: StageFrame) -> object:
        return frame.values["action"]

    @staticmethod
    def adapt_error(error: ExecutionError) -> object:
        raise RuntimeError(error.message)


class _FakeExecution:
    def __init__(self, calls: list[tuple[str, frozenset[str]]]) -> None:
        self.calls = calls

    def invoke(self, role: str, values: dict[str, object]) -> dict[str, np.ndarray]:
        self.calls.append((role, frozenset(values)))
        if role in {"vision", "vision_top"}:
            return {"internal.image_embedding.top": np.zeros((1, 2, _HIDDEN), dtype=np.float32)}
        if role == "prefill":
            return {
                "internal.past_key.0": np.zeros((1, _PREFIX_LENGTH, 1, 2), dtype=np.float32),
                "internal.past_value.0": np.zeros((1, _PREFIX_LENGTH, 1, 2), dtype=np.float32),
            }
        return {"action": np.full(_ACTION_SHAPE, 2.0, dtype=np.float32)}


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, frozenset[str]]] = []

    @contextmanager
    def execution(self, request: NamedTensorRequest):
        yield _FakeExecution(self.calls)


def _vision_bindings() -> ArtifactBindings:
    return ArtifactBindings(
        inputs=(_image_tensor("observation.images.top", 0, "float16", (1, 3, 4, 4)),),
        outputs=(_tensor("internal.image_embedding.top", 0, "float16", (1, 2, _HIDDEN)),),
    )


def _embedding_bindings() -> ArtifactBindings:
    return ArtifactBindings(
        inputs=(
            _tensor("internal.image_embedding.top", 0, "float16", (1, 2, _HIDDEN)),
            _tensor("observation.language.tokens", 1, "int64", (1, 3)),
            _tensor("observation.language.attention_mask", 2, "bool", (1, 3)),
            _tensor("observation.state", 3, "float32", (1, _STATE_DIM)),
        ),
        outputs=(
            _tensor("internal.prefix_embeddings", 0, "float16", (1, _PREFIX_LENGTH, _HIDDEN)),
            _tensor("internal.prefix_pad_masks", 1, "bool", (1, _PREFIX_LENGTH)),
            _tensor("internal.attention_mask", 2, "int32", (1, _PREFIX_LENGTH, _PREFIX_LENGTH)),
            _tensor("internal.position_ids", 3, "int32", (1, _PREFIX_LENGTH)),
        ),
    )


def _prefill_bindings() -> ArtifactBindings:
    return ArtifactBindings(
        inputs=(
            _tensor("internal.prefix_embeddings", 0, "float16", (1, _PREFIX_LENGTH, _HIDDEN)),
            _tensor("internal.attention_mask", 1, "int32", (1, _PREFIX_LENGTH, _PREFIX_LENGTH)),
            _tensor("internal.position_ids", 2, "int32", (1, _PREFIX_LENGTH)),
        ),
        outputs=(
            _tensor("internal.past_key.0", 0, "float16", (1, _PREFIX_LENGTH, 1, 2)),
            _tensor("internal.past_value.0", 1, "float16", (1, _PREFIX_LENGTH, 1, 2)),
        ),
    )


def _action_bindings() -> ArtifactBindings:
    return ArtifactBindings(
        inputs=(
            _tensor("noise", 0, "float16", _ACTION_SHAPE),
            _tensor("time", 1, "float16", (1,)),
            _tensor("internal.prefix_pad_masks", 2, "bool", (1, _PREFIX_LENGTH)),
            _tensor("internal.past_key.0", 3, "float16", (1, _PREFIX_LENGTH, 1, 2)),
            _tensor("internal.past_value.0", 4, "float16", (1, _PREFIX_LENGTH, 1, 2)),
        ),
        outputs=(_tensor("action", 0, "float16", _ACTION_SHAPE),),
    )


def _smolvla_plan(*, device_links: bool) -> object:
    bindings = {
        "vision": _vision_bindings(),
        "embedding": _embedding_bindings(),
        "prefill": _prefill_bindings(),
        "action": _action_bindings(),
    }
    if device_links:
        links = (
            DeviceLink(
                semantic="internal.past_key.0",
                producer="prefill",
                consumer="action",
                transport="device_pointer",
                owner="producer",
            ),
            DeviceLink(
                semantic="internal.past_value.0",
                producer="prefill",
                consumer="action",
                transport="device_pointer",
                owner="producer",
            ),
        )
    else:
        links = ()
    return build_execution_plan(("vision", "embedding", "prefill", "action"), bindings, links)


def _request() -> NamedTensorRequest:
    return NamedTensorRequest(
        "smolvla",
        {
            "observation.images.top": np.zeros((1, 3, 4, 4), dtype=np.float32),
            "observation.language.tokens": np.zeros((1, 3), dtype=np.int64),
            "observation.language.attention_mask": np.ones((1, 3), dtype=bool),
            "observation.state": np.zeros((1, _STATE_DIM), dtype=np.float32),
            "noise": np.ones(_ACTION_SHAPE, dtype=np.float32),
        },
    )


def _resource() -> SmolVLAFamilyResource:
    token_weight = np.arange(40, dtype=np.float32).reshape(10, _HIDDEN) / 10.0
    state_weight = np.zeros((_HIDDEN, _STATE_DIM), dtype=np.float32)
    state_weight[:, :_HIDDEN] = np.eye(_HIDDEN, dtype=np.float32)
    state_bias = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    weights = SmolVLAEmbeddingWeights(token_weight=token_weight, state_weight=state_weight, state_bias=state_bias)
    resource = SmolVLAFamilyResource.__new__(SmolVLAFamilyResource)
    resource._deployment = None
    resource._configured_policy = {}
    resource._weights = weights
    resource._policy_config = {"num_steps": _NUM_STEPS}
    return resource


@pytest.mark.parametrize("device_links", [True, False], ids=["hmm-device-link", "rknn-host-link"])
def test_smolvla_executor_runs_device_link_and_host_link_plans(device_links):
    plan = _smolvla_plan(device_links=device_links)
    session = _FakeSession()
    executor = create_smolvla_executor(
        plan,
        session,
        _resource(),
        _ResultAdapter(),
        num_inference_steps=_NUM_STEPS,
    )

    result = executor.execute(_request(), deadline=None, control=ExecutionControl("smolvla"))

    np.testing.assert_allclose(result, -1.0)
    action_calls = [role for role, _values in session.calls if role == "action"]
    assert len(action_calls) == _NUM_STEPS
    for role, values in session.calls:
        if role == "action":
            if device_links:
                assert "internal.past_key.0" not in values
                assert "internal.past_value.0" not in values
            else:
                assert "internal.past_key.0" in values
                assert "internal.past_value.0" in values


def test_smolvla_executor_euler_update_matches_reference():
    plan = _smolvla_plan(device_links=False)
    trace: list[np.ndarray] = []
    executor = create_smolvla_executor(
        plan,
        _FakeSession(),
        _resource(),
        _ResultAdapter(),
        num_inference_steps=_NUM_STEPS,
        velocity_trace=trace,
    )

    result = executor.execute(_request(), deadline=None, control=ExecutionControl("smolvla"))

    np.testing.assert_allclose(result, -1.0)
    assert len(trace) == _NUM_STEPS
    for velocity in trace:
        np.testing.assert_allclose(velocity, np.full(_ACTION_SHAPE, 2.0, dtype=np.float32))


def test_smolvla_topology_rejects_invalid_role_layout():
    bindings = {
        "encoder": ArtifactBindings(
            inputs=(_image_tensor("observation.images.top", 0, "float16", (1, 3, 4, 4)),),
            outputs=(_tensor("action", 0, "float16", _ACTION_SHAPE),),
        ),
    }
    plan = build_execution_plan(("encoder",), bindings)

    with pytest.raises(SmolVLATopologyError, match="vision role"):
        derive_smolvla_topology(plan)


def test_smolvla_topology_rejects_invalid_vision_role_name():
    bindings = {
        "encoder": ArtifactBindings(
            inputs=(_image_tensor("observation.images.top", 0, "float16", (1, 3, 4, 4)),),
            outputs=(_tensor("internal.image_embedding.top", 0, "float16", (1, 2, _HIDDEN)),),
        ),
        "embedding": _embedding_bindings(),
        "prefill": _prefill_bindings(),
        "action": _action_bindings(),
    }
    plan = build_execution_plan(("encoder", "embedding", "prefill", "action"), bindings)

    with pytest.raises(SmolVLATopologyError, match="invalid vision roles"):
        derive_smolvla_topology(plan)


def test_smolvla_topology_rejects_missing_action_semantics():
    bindings = {
        "vision": _vision_bindings(),
        "embedding": _embedding_bindings(),
        "prefill": _prefill_bindings(),
        "action": ArtifactBindings(
            inputs=(
                _tensor("internal.prefix_pad_masks", 0, "bool", (1, _PREFIX_LENGTH)),
                _tensor("internal.past_key.0", 1, "float16", (1, _PREFIX_LENGTH, 1, 2)),
                _tensor("internal.past_value.0", 2, "float16", (1, _PREFIX_LENGTH, 1, 2)),
            ),
            outputs=(_tensor("action", 0, "float16", _ACTION_SHAPE),),
        ),
    }
    plan = build_execution_plan(("vision", "embedding", "prefill", "action"), bindings)

    with pytest.raises(SmolVLATopologyError, match="noise"):
        derive_smolvla_topology(plan)


def test_smolvla_embedding_operation_matches_reference_and_declared_dtypes():
    bindings = _embedding_bindings()
    token_weight = np.arange(2 * _HIDDEN, dtype=np.float32).reshape(2, _HIDDEN) / 3.0
    state_weight = np.eye(_HIDDEN, dtype=np.float32)
    state_bias = np.linspace(-1.0, 1.0, _HIDDEN, dtype=np.float32)
    weights = SmolVLAEmbeddingWeights(token_weight=token_weight, state_weight=state_weight, state_bias=state_bias)
    operation = _embedding_operation(bindings, lambda: weights)

    tokens = np.array([[0, 1, 0]], dtype=np.int64)
    mask = np.array([[True, True, False]], dtype=bool)
    state = np.array([[0.5] * _HIDDEN], dtype=np.float32)
    image = np.arange(1 * 2 * _HIDDEN, dtype=np.float32).reshape(1, 2, _HIDDEN)
    outputs = operation(
        {
            "observation.language.tokens": tokens,
            "observation.language.attention_mask": mask,
            "observation.state": state,
            "internal.image_embedding.top": image,
        }
    )

    sqrt_hidden = math.sqrt(_HIDDEN)
    expected_language = token_weight[tokens] * sqrt_hidden
    expected_state = (state @ state_weight.T + state_bias)[:, None, :] * 1.0
    expected_prefix = np.concatenate(
        (image * sqrt_hidden, expected_language, expected_state.astype(expected_language.dtype)), axis=1
    )
    np.testing.assert_allclose(outputs["internal.prefix_embeddings"], expected_prefix.astype(np.float16), atol=1e-3)
    assert outputs["internal.prefix_embeddings"].dtype == np.dtype(np.float16)

    pad_masks = outputs["internal.prefix_pad_masks"]
    assert pad_masks.dtype == np.dtype(np.bool_)
    expected_pad = np.concatenate((np.ones((1, 2), dtype=bool), mask, np.ones((1, 1), dtype=bool)), axis=1)
    np.testing.assert_array_equal(pad_masks, expected_pad)

    attention = outputs["internal.attention_mask"]
    assert attention.dtype == np.dtype(np.int32)
    state_index = _PREFIX_LENGTH - 1
    image_index = 0
    assert attention[0, image_index, state_index] == 0
    assert attention[0, state_index, image_index] == 1
    assert attention[0, state_index, state_index] == 1

    position_ids = outputs["internal.position_ids"]
    assert position_ids.dtype == np.dtype(np.int32)
    expected_positions = np.where(expected_pad, np.cumsum(expected_pad.astype(np.int32), axis=1) - 1, 0)
    np.testing.assert_array_equal(position_ids, expected_positions)


def _binding_dict(
    semantic: str, name: str, index: int, dtype: str, shape: list[int], *, layout: str | None = None
) -> dict[str, object]:
    is_image = semantic.startswith("observation.images.") or semantic.startswith("observation.image.")
    if layout is None and len(shape) == 4 and is_image:
        layout = "NCHW"
    result: dict[str, object] = {
        "semantic": semantic,
        "runtime_name": name,
        "index": index,
        "dtype": dtype,
        "shape": shape,
    }
    if layout is not None:
        result["layout"] = layout
    return result


def _artifact_dict(root: Path, role: str, artifact_format: str) -> dict[str, str]:
    suffix = ".pt" if artifact_format in {"pt", "pytorch"} else ".bin"
    path = root / "artifacts" / f"{role}{suffix}"
    return {"path": str(path.relative_to(root)), "format": artifact_format}


def _smolvla_context(tmp_path: Path) -> object:
    from inference_service.backends.types import RuntimeContext

    tmp_path.mkdir(parents=True, exist_ok=True)
    bundle_paths = create_policy_bundle(tmp_path, "smolvla", include_weights=False)
    import json

    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    config.update(
        {
            "chunk_size": _CHUNK,
            "max_state_dim": _STATE_DIM,
            "max_action_dim": _STATE_DIM,
            "num_steps": _NUM_STEPS,
            "empty_cameras": 0,
            "add_image_special_tokens": False,
        }
    )
    config["input_features"].update(
        {
            "observation.language.tokens": {"type": "LANGUAGE", "shape": [3]},
            "observation.language.attention_mask": {"type": "LANGUAGE", "shape": [3]},
        }
    )
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    for role in ("vision", "prefill", "action"):
        (artifact_dir / f"{role}.bin").write_bytes(role.encode())
    token_weight = np.arange(10 * _HIDDEN, dtype=np.float32).reshape(10, _HIDDEN) / 10.0
    state_weight = np.zeros((_HIDDEN, _STATE_DIM), dtype=np.float32)
    state_weight[:, :_HIDDEN] = np.eye(_HIDDEN, dtype=np.float32)
    state_bias = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    torch = pytest.importorskip("torch")
    torch.save({"token_embedding.weight": torch.from_numpy(token_weight)}, artifact_dir / "embedding.pt")
    torch.save(
        {"state_proj.weight": torch.from_numpy(state_weight), "state_proj.bias": torch.from_numpy(state_bias)},
        artifact_dir / "state_projection.pt",
    )

    vision_bindings = {
        "inputs": [_binding_dict("observation.images.top", "pixel_values", 0, "float16", [1, 3, 4, 4], layout="NCHW")],
        "outputs": [_binding_dict("internal.image_embedding.top", "image_embeddings", 0, "float16", [1, 2, _HIDDEN])],
    }
    embedding_bindings = {
        "inputs": [
            _binding_dict("internal.image_embedding.top", "image", 0, "float16", [1, 2, _HIDDEN]),
            _binding_dict("observation.language.tokens", "tokens", 1, "int64", [1, 3]),
            _binding_dict("observation.language.attention_mask", "mask", 2, "bool", [1, 3]),
            _binding_dict("observation.state", "state", 3, "float32", [1, _STATE_DIM]),
        ],
        "outputs": [
            _binding_dict("internal.prefix_embeddings", "prefix_embs", 0, "float16", [1, _PREFIX_LENGTH, _HIDDEN]),
            _binding_dict("internal.prefix_pad_masks", "prefix_pad_masks", 1, "bool", [1, _PREFIX_LENGTH]),
            _binding_dict("internal.attention_mask", "attention_mask", 2, "int32", [1, _PREFIX_LENGTH, _PREFIX_LENGTH]),
            _binding_dict("internal.position_ids", "position_ids", 3, "int32", [1, _PREFIX_LENGTH]),
        ],
    }
    prefill_bindings = {
        "inputs": [
            _binding_dict("internal.prefix_embeddings", "prefix_embs", 0, "float16", [1, _PREFIX_LENGTH, _HIDDEN]),
            _binding_dict("internal.attention_mask", "attention_mask", 1, "int32", [1, _PREFIX_LENGTH, _PREFIX_LENGTH]),
            _binding_dict("internal.position_ids", "position_ids", 2, "int32", [1, _PREFIX_LENGTH]),
        ],
        "outputs": [
            _binding_dict("internal.past_key.0", "past_key_0", 0, "float16", [1, _PREFIX_LENGTH, 1, 2]),
            _binding_dict("internal.past_value.0", "past_value_0", 1, "float16", [1, _PREFIX_LENGTH, 1, 2]),
        ],
    }
    action_bindings = {
        "inputs": [
            _binding_dict("noise", "x_t", 0, "float16", list(_ACTION_SHAPE)),
            _binding_dict("time", "timestep", 1, "float16", [1]),
            _binding_dict("internal.prefix_pad_masks", "prefix_pad_masks", 2, "bool", [1, _PREFIX_LENGTH]),
            _binding_dict("internal.past_key.0", "past_key_0", 3, "float16", [1, _PREFIX_LENGTH, 1, 2]),
            _binding_dict("internal.past_value.0", "past_value_0", 4, "float16", [1, _PREFIX_LENGTH, 1, 2]),
        ],
        "outputs": [_binding_dict("action", "v_t", 0, "float16", list(_ACTION_SHAPE))],
    }
    deployment = {
        "backend": "hmm",
        "target": {"soc": "lq50", "runtime": "tcim-lite"},
        "artifacts": {
            "vision": _artifact_dict(tmp_path, "vision", "bin"),
            "embedding": _artifact_dict(tmp_path, "embedding", "pt"),
            "state_projection": _artifact_dict(tmp_path, "state_projection", "pt"),
            "prefill": _artifact_dict(tmp_path, "prefill", "bin"),
            "action": _artifact_dict(tmp_path, "action", "bin"),
        },
        "execution": ["vision", "embedding", "prefill", "action"],
        "bindings": {
            "vision": vision_bindings,
            "embedding": embedding_bindings,
            "prefill": prefill_bindings,
            "action": action_bindings,
        },
        "device_links": [
            {
                "semantic": semantic,
                "producer": "prefill",
                "consumer": "action",
                "transport": "device_pointer",
                "owner": "producer",
                "lifetime": "inference",
            }
            for semantic in ("internal.past_key.0", "internal.past_value.0")
        ],
    }
    entries = tuple(BundleFile(path=path) for path in bundle_paths)
    write_manifest(
        tmp_path,
        {
            "schema_version": 2,
            "bundle": {
                "uuid": TEST_BUNDLE_UUID,
                "revision": 1,
                "name": "smolvla-test",
                "files": [entry.model_dump(mode="json") for entry in entries],
                "digest": {
                    "algorithm": "sha256",
                    "scope": "structure",
                    "value": canonical_bundle_digest(TEST_BUNDLE_UUID, 1, "smolvla-test", entries),
                },
            },
            "deployments": {"houmo": {"uuid": TEST_DEPLOYMENT_UUID, "revision": 1, **deployment}},
        },
    )
    return RuntimeContext(load_inference_manifest(tmp_path, "houmo"))


def test_smolvla_family_resource_loads_weights_once_and_closes(tmp_path):
    context = _smolvla_context(tmp_path)
    deployment = context.deployment
    policy_config = {"chunk_size": _CHUNK, "max_action_dim": _STATE_DIM, "num_steps": _NUM_STEPS}
    resource = SmolVLAFamilyResource(deployment, policy_config)

    resource.load(context)
    weights = resource.embedding
    np.testing.assert_allclose(
        weights.token_weight, np.arange(10 * _HIDDEN, dtype=np.float32).reshape(10, _HIDDEN) / 10.0
    )
    assert weights.state_weight.shape == (_HIDDEN, _STATE_DIM)
    assert resource.policy_config["num_steps"] == _NUM_STEPS

    resource.close()
    with pytest.raises(Exception, match="not loaded"):
        _ = resource.embedding
    with pytest.raises(Exception, match="not loaded"):
        _ = resource.policy_config


def test_smolvla_load_embedding_weights_reads_manifest_artifacts(tmp_path):
    context = _smolvla_context(tmp_path)

    weights = load_smolvla_embedding_weights(context)

    assert weights.token_weight.shape == (10, _HIDDEN)
    assert weights.token_weight.dtype == np.float32
    assert weights.state_weight.shape == (_HIDDEN, _STATE_DIM)
    assert weights.state_bias.shape == (_HIDDEN,)
    np.testing.assert_allclose(weights.state_bias, np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32))


def test_smolvla_executor_components_include_session_and_resource():
    plan = _smolvla_plan(device_links=True)
    session = _FakeSession()
    resource = _resource()
    executor = create_smolvla_executor(
        plan,
        session,
        resource,
        _ResultAdapter(),
        num_inference_steps=_NUM_STEPS,
    )

    assert executor._components[0] is session
    assert executor._components[1] is resource
