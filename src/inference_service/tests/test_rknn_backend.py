from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from inference_manifest import BundleFile, canonical_bundle_digest, load_inference_manifest
from inference_manifest.models import DeviceLink
from inference_service.backends import (
    STATIC_BACKEND_DESCRIPTORS,
    BackendInferenceError,
    BackendLoadError,
    BackendRegistry,
    BackendRegistryError,
    InferenceRequest,
    RuntimeContext,
)
from inference_service.backends.rknn.runtime import validate_runtime_options
from inference_service.model_sessions import RKNNModelSession
from inference_service.pipeline import PipelineState
from inference_service.pipeline import factory as pipeline_factory
from tests.manifest_fixtures import (
    TEST_BUNDLE_UUID,
    TEST_DEPLOYMENT_UUID,
    create_policy_bundle,
    policy_model,
    v3_runtime_deployment,
    write_manifest,
)

_STATIC_BACKEND_REGISTRY = BackendRegistry(STATIC_BACKEND_DESCRIPTORS)


@dataclass(frozen=True)
class FakeRKNNModel:
    callback: object


class FakeRKNNEnvironment:
    def __init__(self, model_specs: dict[str, FakeRKNNModel]) -> None:
        self.model_specs = model_specs
        self.fail_load_paths: set[str] = set()
        self.fail_init_paths: set[str] = set()
        self.load_order: list[str] = []
        self.init_calls: list[tuple[str, str | None, int]] = []
        self.inference_inputs: dict[str, list[tuple[np.ndarray, ...]]] = {}
        self.inference_formats: dict[str, list[str | None]] = {}
        self.release_calls: list[str | None] = []
        self.instances: list[object] = []

    def runtime_type(self) -> type:
        owner = self

        class FakeRKNNLite:
            NPU_CORE_AUTO = 0
            NPU_CORE_0 = 1
            NPU_CORE_1 = 2
            NPU_CORE_2 = 4
            NPU_CORE_ALL = 7

            def __init__(self) -> None:
                self.path: str | None = None
                self.released = False
                owner.instances.append(self)

            def load_rknn(self, path: str) -> int:
                self.path = path
                owner.load_order.append(path)
                return 1 if path in owner.fail_load_paths else 0

            def init_runtime(self, *, target: str | None, core_mask: int) -> int:
                assert self.path is not None
                owner.init_calls.append((self.path, target, core_mask))
                return 1 if self.path in owner.fail_init_paths else 0

            def inference(self, *, inputs: list[np.ndarray], data_format: str | None = None):
                assert self.path is not None
                copied = tuple(np.array(value, copy=True) for value in inputs)
                owner.inference_inputs.setdefault(self.path, []).append(copied)
                owner.inference_formats.setdefault(self.path, []).append(data_format)
                callback = owner.model_specs[self.path].callback
                return callback(copied)

            def release(self) -> None:
                if self.released:
                    return
                self.released = True
                owner.release_calls.append(self.path)

        return FakeRKNNLite


@pytest.fixture(autouse=True)
def _identity_lerobot_processors(monkeypatch):
    monkeypatch.setattr(
        pipeline_factory,
        "create_lerobot_processor_views",
        lambda: (lambda inputs: inputs, lambda action: action),
    )


def _bundle_entries(root: Path, paths: tuple[str, ...]) -> list[BundleFile]:
    del root
    return [BundleFile(path=path) for path in paths]


def _write_compiled_manifest(root: Path, bundle_paths: tuple[str, ...], deployment: dict) -> None:
    entries = _bundle_entries(root, bundle_paths)
    deployment = {"uuid": TEST_DEPLOYMENT_UUID, "revision": 1, **deployment}
    deployment = v3_runtime_deployment(deployment)
    policy_type = json.loads((root / "config.json").read_text(encoding="utf-8")).get("type", "act")
    write_manifest(
        root,
        {
            "schema_version": 3,
            "bundle": {
                "uuid": TEST_BUNDLE_UUID,
                "revision": 1,
                "name": "rknn-test",
                "files": [entry.model_dump(mode="json") for entry in entries],
                "digest": {
                    "algorithm": "sha256",
                    "scope": "structure",
                    "value": canonical_bundle_digest(TEST_BUNDLE_UUID, 1, "rknn-test", entries),
                },
            },
            "model": policy_model(policy_type),
            "deployments": {"rk3588": deployment},
        },
    )


def _act_context(tmp_path: Path, *, runtime_options=None) -> RuntimeContext:
    tmp_path.mkdir(parents=True, exist_ok=True)
    bundle_paths = create_policy_bundle(tmp_path, include_weights=False)
    model = tmp_path / "artifacts" / "custom-act.rknn"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"act-rknn")
    _write_compiled_manifest(
        tmp_path,
        bundle_paths,
        {
            "backend": "rknn",
            "target": {"soc": "rk3588", "runtime": "rknn-lite"},
            "artifacts": {"policy": {"path": "artifacts/custom-act.rknn", "format": "rknn"}},
            "execution": ["policy"],
            "bindings": {
                "policy": {
                    "inputs": [
                        {
                            "semantic": "observation.state",
                            "runtime_name": "state",
                            "index": 0,
                            "dtype": "float32",
                            "shape": [1, 6],
                        },
                        {
                            "semantic": "observation.images.top",
                            "runtime_name": "image",
                            "index": 1,
                            "dtype": "float32",
                            "shape": [1, 16, 24, 3],
                            "layout": "NHWC",
                        },
                    ],
                    "outputs": [
                        {
                            "semantic": "action",
                            "runtime_name": "action",
                            "index": 0,
                            "dtype": "float32",
                            "shape": [1, 4, 6],
                        }
                    ],
                }
            },
        },
    )
    return RuntimeContext(load_inference_manifest(tmp_path, "rk3588"), runtime_options=runtime_options or {})


def _act_pipeline(context: RuntimeContext, environment: FakeRKNNEnvironment):
    def session_factory(ctx, options):
        del ctx, options
        return RKNNModelSession(rknn_loader=environment.runtime_type)

    return pipeline_factory.create_inference_pipeline(
        "policy",
        context.validated_manifest,
        runtime_options=dict(context.runtime_options),
        model_session_factory=session_factory,
    )


def _smolvla_context(tmp_path: Path, *, runtime_options=None) -> RuntimeContext:
    torch = pytest.importorskip("torch")
    tmp_path.mkdir(parents=True, exist_ok=True)
    bundle_paths = create_policy_bundle(tmp_path, "smolvla", include_weights=False)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "chunk_size": 2,
            "max_state_dim": 8,
            "max_action_dim": 8,
            "num_steps": 2,
            "empty_cameras": 0,
            "add_image_special_tokens": False,
        }
    )
    config["input_features"].update(
        {
            "observation.images.wrist": {"type": "VISUAL", "shape": [3, 16, 24]},
            "observation.language.tokens": {"type": "LANGUAGE", "shape": [3]},
            "observation.language.attention_mask": {"type": "LANGUAGE", "shape": [3]},
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True)
    artifact_paths = {
        "vision_top": artifact_dir / "vision-top.rknn",
        "vision_wrist": artifact_dir / "vision-top.rknn",
        "prefill": artifact_dir / "prefill.rknn",
        "action": artifact_dir / "action.rknn",
        "embedding": artifact_dir / "token-embedding.pt",
        "state_projection": artifact_dir / "state-projection.pt",
    }
    artifact_paths["vision_top"].write_bytes(b"shared-vision-rknn")
    artifact_paths["vision_wrist"].write_bytes(b"shared-vision-rknn")
    artifact_paths["prefill"].write_bytes(b"prefill-rknn")
    artifact_paths["action"].write_bytes(b"action-rknn")
    token_weight = np.arange(40, dtype=np.float32).reshape(10, 4) / 10.0
    state_weight = np.zeros((4, 8), dtype=np.float32)
    state_weight[:, :4] = np.eye(4, dtype=np.float32)
    state_bias = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    torch.save({"weight": torch.from_numpy(token_weight)}, artifact_paths["embedding"])
    torch.save(
        {"weight": torch.from_numpy(state_weight), "bias": torch.from_numpy(state_bias)},
        artifact_paths["state_projection"],
    )

    def artifact(role: str, artifact_format: str) -> dict[str, str]:
        path = artifact_paths[role]
        value = {
            "path": str(path.relative_to(tmp_path)),
            "format": artifact_format,
        }
        if role.startswith("vision_"):
            value["share_group"] = "vision"
        return value

    def vision_bindings(semantic, output):
        return {
            "inputs": [
                {
                    "semantic": semantic,
                    "runtime_name": "pixel_values",
                    "index": 0,
                    "dtype": "float32",
                    "shape": [1, 4, 4, 3],
                    "layout": "NHWC",
                }
            ],
            "outputs": [
                {
                    "semantic": output,
                    "runtime_name": "image_embeddings",
                    "index": 0,
                    "dtype": "float32",
                    "shape": [1, 2, 4],
                }
            ],
        }

    deployment = {
        "backend": "rknn",
        "target": {"soc": "rk3588", "runtime": "rknn-lite2"},
        "artifacts": {
            "vision_top": artifact("vision_top", "rknn"),
            "vision_wrist": artifact("vision_wrist", "rknn"),
            "embedding": artifact("embedding", "pt"),
            "prefill": artifact("prefill", "rknn"),
            "action": artifact("action", "rknn"),
            "state_projection": artifact("state_projection", "pt"),
        },
        "execution": ["vision_top", "vision_wrist", "embedding", "prefill", "action"],
        "bindings": {
            "vision_top": vision_bindings("observation.images.top", "internal.image_embedding.top"),
            "vision_wrist": vision_bindings("observation.images.wrist", "internal.image_embedding.wrist"),
            "embedding": {
                "inputs": [
                    {
                        "semantic": "internal.image_embedding.top",
                        "runtime_name": "image_top",
                        "index": 0,
                        "dtype": "float32",
                        "shape": [1, 2, 4],
                    },
                    {
                        "semantic": "internal.image_embedding.wrist",
                        "runtime_name": "image_wrist",
                        "index": 1,
                        "dtype": "float32",
                        "shape": [1, 2, 4],
                    },
                    {
                        "semantic": "observation.language.tokens",
                        "runtime_name": "tokens",
                        "index": 2,
                        "dtype": "int64",
                        "shape": [1, 3],
                    },
                    {
                        "semantic": "observation.language.attention_mask",
                        "runtime_name": "language_mask",
                        "index": 3,
                        "dtype": "bool",
                        "shape": [1, 3],
                    },
                    {
                        "semantic": "observation.state",
                        "runtime_name": "state",
                        "index": 4,
                        "dtype": "float32",
                        "shape": [1, 8],
                    },
                ],
                "outputs": [
                    {
                        "semantic": "internal.prefix_embeddings",
                        "runtime_name": "prefix_embeddings",
                        "index": 0,
                        "dtype": "float32",
                        "shape": [1, 8, 4],
                    },
                    {
                        "semantic": "internal.prefix_pad_masks",
                        "runtime_name": "prefix_pad_masks",
                        "index": 1,
                        "dtype": "bool",
                        "shape": [1, 8],
                    },
                    {
                        "semantic": "internal.attention_mask",
                        "runtime_name": "attention_mask",
                        "index": 2,
                        "dtype": "bool",
                        "shape": [1, 8, 8],
                    },
                    {
                        "semantic": "internal.position_ids",
                        "runtime_name": "position_ids",
                        "index": 3,
                        "dtype": "int64",
                        "shape": [1, 8],
                    },
                ],
            },
            "prefill": {
                "inputs": [
                    {
                        "semantic": "internal.prefix_embeddings",
                        "runtime_name": "prefix_embeddings",
                        "index": 0,
                        "dtype": "float32",
                        "shape": [1, 8, 4],
                    },
                    {
                        "semantic": "internal.attention_mask",
                        "runtime_name": "attention_mask",
                        "index": 1,
                        "dtype": "bool",
                        "shape": [1, 8, 8],
                    },
                    {
                        "semantic": "internal.position_ids",
                        "runtime_name": "position_ids",
                        "index": 2,
                        "dtype": "int64",
                        "shape": [1, 8],
                    },
                ],
                "outputs": [
                    {
                        "semantic": "internal.past_key.0",
                        "runtime_name": "past_key_0",
                        "index": 0,
                        "dtype": "float32",
                        "shape": [1, 8, 1, 2],
                    },
                    {
                        "semantic": "internal.past_value.0",
                        "runtime_name": "past_value_0",
                        "index": 1,
                        "dtype": "float32",
                        "shape": [1, 8, 1, 2],
                    },
                ],
            },
            "action": {
                "inputs": [
                    {
                        "semantic": "noise",
                        "runtime_name": "x_t",
                        "index": 0,
                        "dtype": "float32",
                        "shape": [1, 2, 8],
                    },
                    {
                        "semantic": "time",
                        "runtime_name": "timestep",
                        "index": 1,
                        "dtype": "float32",
                        "shape": [1],
                    },
                    {
                        "semantic": "internal.prefix_pad_masks",
                        "runtime_name": "prefix_pad_masks",
                        "index": 2,
                        "dtype": "bool",
                        "shape": [1, 8],
                    },
                    {
                        "semantic": "internal.past_key.0",
                        "runtime_name": "past_key_0",
                        "index": 3,
                        "dtype": "float32",
                        "shape": [1, 8, 1, 2],
                    },
                    {
                        "semantic": "internal.past_value.0",
                        "runtime_name": "past_value_0",
                        "index": 4,
                        "dtype": "float32",
                        "shape": [1, 8, 1, 2],
                    },
                ],
                "outputs": [
                    {
                        "semantic": "action",
                        "runtime_name": "v_t",
                        "index": 0,
                        "dtype": "float32",
                        "shape": [1, 2, 8],
                    }
                ],
            },
        },
    }
    _write_compiled_manifest(tmp_path, bundle_paths, deployment)
    return RuntimeContext(load_inference_manifest(tmp_path, "rk3588"), runtime_options=runtime_options or {})


def test_rknn_act_pipeline_uses_only_manifest_artifact_and_nhwc_binding(tmp_path):
    context = _act_context(tmp_path, runtime_options={"target": "rk3588", "core_mask": "0"})
    model_path = str(context.resolved_artifacts["policy"])

    def execute(inputs):
        state, image = inputs
        assert image.shape == (1, 16, 24, 3)
        return [np.repeat(state[:, None, :], 4, axis=1).reshape(-1)]

    environment = FakeRKNNEnvironment({model_path: FakeRKNNModel(execute)})
    pipeline = _act_pipeline(context, environment)
    pipeline.load()

    state = np.arange(6, dtype=np.float32)
    result = pipeline.infer(
        InferenceRequest(
            request_id="act",
            inputs={
                "observation.state": state,
                "observation.images.top": np.ones((3, 16, 24), dtype=np.float32),
            },
        )
    )

    np.testing.assert_array_equal(result.action, np.repeat(state[None, None, :], 4, axis=1))
    assert result.actual_chunk_size == 4
    assert environment.load_order == [model_path]
    assert environment.init_calls == [(model_path, "rk3588", 1)]
    assert environment.inference_formats[model_path] == ["nhwc"]
    assert pipeline._backend is None
    assert pipeline.health().state is PipelineState.READY
    pipeline.close()
    pipeline.close()
    assert environment.release_calls == [model_path]


def _patch_identity_processors(monkeypatch):
    from inference_service import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module.factory,
        "create_lerobot_processor_views",
        lambda: (lambda inputs: inputs, lambda action: action),
    )


def _rknn_smolvla_session_factory(environment):
    from inference_service.model_sessions import RKNNModelSession

    def factory(ctx, options):
        del ctx, options
        return RKNNModelSession(rknn_loader=environment.runtime_type)

    return factory


def test_rknn_smolvla_pipeline_runs_through_session_with_host_links_and_shared_vision(tmp_path, monkeypatch):
    from inference_service.pipeline import create_inference_pipeline

    _patch_identity_processors(monkeypatch)
    context = _smolvla_context(tmp_path, runtime_options={"random_seed": 7})
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}
    observed_times: list[float] = []
    observed_prefix: list[np.ndarray] = []

    def execute_vision(inputs):
        (image,) = inputs
        assert image.shape == (1, 4, 4, 3)
        assert np.all(image[:, :2] == 0.0)
        value = float(image.mean())
        return [np.full((1, 2, 4), value, dtype=np.float32)]

    def execute_prefill(inputs):
        prefix, attention, position_ids = inputs
        observed_prefix.append(prefix)
        assert attention.dtype == np.dtype("bool")
        assert not bool(attention[0, 0, -1])
        assert bool(attention[0, -1, 0])
        np.testing.assert_array_equal(position_ids, np.array([[0, 1, 2, 3, 4, 5, 0, 6]], dtype=np.int64))
        key = np.full((1, 8, 1, 2), 3.0, dtype=np.float32)
        value = np.full((1, 8, 1, 2), 4.0, dtype=np.float32)
        return [key, value]

    def execute_action(inputs):
        noise, time_value, pad_masks, key, value = inputs
        observed_times.append(float(time_value[0]))
        np.testing.assert_array_equal(pad_masks, np.array([[True, True, True, True, True, True, False, True]]))
        np.testing.assert_array_equal(key, np.full((1, 8, 1, 2), 3.0, dtype=np.float32))
        np.testing.assert_array_equal(value, np.full((1, 8, 1, 2), 4.0, dtype=np.float32))
        return [np.ones_like(noise)]

    environment = FakeRKNNEnvironment(
        {
            paths["vision_top"]: FakeRKNNModel(execute_vision),
            paths["prefill"]: FakeRKNNModel(execute_prefill),
            paths["action"]: FakeRKNNModel(execute_action),
        }
    )
    pipeline = create_inference_pipeline(
        "smolvla",
        context.validated_manifest,
        runtime_options={"random_seed": 7},
        model_session_factory=_rknn_smolvla_session_factory(environment),
    )
    pipeline.load()

    result = pipeline.infer(
        InferenceRequest(
            request_id="smolvla",
            inputs={
                "observation.state": np.arange(1, 7, dtype=np.float32),
                "observation.images.top": np.ones((3, 2, 4), dtype=np.float32),
                "observation.images.wrist": np.full((3, 2, 4), 2.0, dtype=np.float32),
                "observation.language.tokens": np.array([1, 2, 3], dtype=np.int64),
                "observation.language.attention_mask": np.array([True, True, False]),
                "noise": np.zeros((1, 2, 8), dtype=np.float32),
            },
        )
    )

    np.testing.assert_array_equal(result.action, np.full((1, 2, 6), -1.0, dtype=np.float32))
    assert result.actual_chunk_size == 2
    # RKNN host links: prefill KV caches are materialized on host for action.
    assert environment.load_order == [paths["vision_top"], paths["prefill"], paths["action"]]
    assert environment.inference_formats[paths["vision_top"]] == ["nhwc", "nhwc"]
    assert environment.inference_formats[paths["prefill"]] == [None]
    assert environment.inference_formats[paths["action"]] == [None, None]
    assert len(environment.inference_inputs[paths["vision_top"]]) == 2
    assert observed_times == [1.0, 0.5]
    assert len(observed_prefix) == 1
    np.testing.assert_allclose(observed_prefix[0][0, -1], np.array([1.25, 2.5, 3.75, 5.0], dtype=np.float32))
    pipeline.close()
    assert environment.release_calls == [paths["action"], paths["prefill"], paths["vision_top"]]


def test_rknn_smolvla_partial_load_failure_releases_every_created_runtime(tmp_path, monkeypatch):
    from inference_service.pipeline import create_inference_pipeline

    _patch_identity_processors(monkeypatch)
    context = _smolvla_context(tmp_path, runtime_options={"random_seed": 7})
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}
    environment = FakeRKNNEnvironment(
        {
            paths["vision_top"]: FakeRKNNModel(lambda inputs: inputs),
            paths["prefill"]: FakeRKNNModel(lambda inputs: inputs),
            paths["action"]: FakeRKNNModel(lambda inputs: inputs),
        }
    )
    environment.fail_init_paths.add(paths["action"])
    pipeline = create_inference_pipeline(
        "smolvla",
        context.validated_manifest,
        runtime_options={"random_seed": 7},
        model_session_factory=_rknn_smolvla_session_factory(environment),
    )

    with pytest.raises(BackendLoadError, match="init_runtime"):
        pipeline.load()

    assert environment.load_order == [paths["vision_top"], paths["prefill"], paths["action"]]
    assert environment.release_calls == [paths["action"], paths["prefill"], paths["vision_top"]]
    assert all(instance.released for instance in environment.instances)
    pipeline.close()


def test_rknn_smolvla_rejects_device_pointer_links_before_loading_sdk(tmp_path, monkeypatch):
    from inference_service.pipeline import create_inference_pipeline

    _patch_identity_processors(monkeypatch)
    context = _smolvla_context(tmp_path, runtime_options={"target": "rk3588", "core_mask": "0"})
    deployment = context.deployment.model_copy(
        update={
            "device_links": (
                DeviceLink(
                    semantic="internal.past_key.0",
                    producer="prefill",
                    consumer="action",
                    transport="device_pointer",
                    owner="producer",
                    lifetime="inference",
                ),
            )
        }
    )
    validated = replace(context.validated_manifest, deployment=deployment)
    sdk_loaded = False

    def load_sdk():
        nonlocal sdk_loaded
        sdk_loaded = True
        raise AssertionError("RKNN SDK must not load when device links are declared")

    def factory(ctx, options):
        del ctx, options
        from inference_service.model_sessions import RKNNModelSession

        return RKNNModelSession(rknn_loader=load_sdk)

    pipeline = create_inference_pipeline(
        "smolvla",
        validated,
        runtime_options={"target": "rk3588", "core_mask": "0"},
        model_session_factory=factory,
    )

    with pytest.raises(BackendLoadError) as error:
        pipeline.load()

    assert error.value.code == "unsupported_device_links"
    assert sdk_loaded is False
    pipeline.close()


def test_rknn_smolvla_repeated_inference_is_deterministic_with_seed(tmp_path, monkeypatch):
    from inference_service.pipeline import create_inference_pipeline

    _patch_identity_processors(monkeypatch)
    context = _smolvla_context(tmp_path, runtime_options={"random_seed": 42})
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}
    environment = FakeRKNNEnvironment(
        {
            paths["vision_top"]: FakeRKNNModel(lambda inputs: [np.zeros((1, 2, 4), dtype=np.float32)]),
            paths["prefill"]: FakeRKNNModel(
                lambda inputs: [
                    np.zeros((1, 8, 1, 2), dtype=np.float32),
                    np.zeros((1, 8, 1, 2), dtype=np.float32),
                ]
            ),
            paths["action"]: FakeRKNNModel(lambda inputs: [np.ones((1, 2, 8), dtype=np.float32)]),
        }
    )
    pipeline = create_inference_pipeline(
        "smolvla",
        context.validated_manifest,
        runtime_options={"random_seed": 42},
        model_session_factory=_rknn_smolvla_session_factory(environment),
    )
    pipeline.load()

    request = InferenceRequest(
        request_id="smolvla",
        inputs={
            "observation.state": np.arange(1, 7, dtype=np.float32),
            "observation.images.top": np.ones((3, 2, 4), dtype=np.float32),
            "observation.images.wrist": np.full((3, 2, 4), 2.0, dtype=np.float32),
            "observation.language.tokens": np.array([1, 2, 3], dtype=np.int64),
            "observation.language.attention_mask": np.array([True, True, False]),
            "noise": np.zeros((1, 2, 8), dtype=np.float32),
        },
    )

    first = pipeline.infer(request)
    second = pipeline.infer(request)

    np.testing.assert_array_equal(first.action, second.action)
    assert first.actual_chunk_size == second.actual_chunk_size
    pipeline.close()


def test_rknn_rejects_runtime_output_shape_mismatch(tmp_path):
    context = _act_context(tmp_path)
    model_path = str(context.resolved_artifacts["policy"])
    environment = FakeRKNNEnvironment(
        {model_path: FakeRKNNModel(lambda inputs: [np.zeros((1, 3, 6), dtype=np.float32)])}
    )
    pipeline = _act_pipeline(context, environment)
    pipeline.load()

    with pytest.raises(BackendInferenceError) as error:
        pipeline.infer(
            InferenceRequest(
                request_id="bad-output",
                inputs={
                    "observation.state": np.zeros(6, dtype=np.float32),
                    "observation.images.top": np.zeros((3, 16, 24), dtype=np.float32),
                },
            )
        )

    assert error.value.code == "runtime_output_shape_mismatch"
    assert pipeline.health().state is PipelineState.FAILED
    pipeline.close()


def test_rknn_registry_descriptor_is_session_only(tmp_path):
    context = _act_context(tmp_path)
    descriptor = _STATIC_BACKEND_REGISTRY.validate(context)

    assert descriptor.factory is None
    with pytest.raises(BackendRegistryError) as error:
        _STATIC_BACKEND_REGISTRY._create_legacy_backend(context)
    assert error.value.code == "legacy_backend_unavailable"


@pytest.mark.parametrize(
    "runtime_options",
    [
        {"target": ""},
        {"core_mask": -1},
        {"core_mask": "invalid"},
        {"random_seed": 1.5},
        {"unknown": True},
    ],
)
def test_rknn_rejects_invalid_runtime_options(tmp_path, runtime_options):
    context = _act_context(tmp_path, runtime_options=runtime_options)

    with pytest.raises(BackendLoadError) as error:
        validate_runtime_options(context.runtime_options)

    assert error.value.code == "invalid_runtime_options"
