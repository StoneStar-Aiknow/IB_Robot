from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from inference_manifest import BundleFile, canonical_bundle_digest, load_inference_manifest
from inference_service.backends import (
    BACKEND_REGISTRY,
    BackendCapabilityError,
    BackendLoadError,
    InferenceRequest,
    RuntimeContext,
)
from inference_service.backends.hmm import HMMBackend, create_backend
from tests.manifest_fixtures import TEST_BUNDLE_UUID, TEST_DEPLOYMENT_UUID, create_policy_bundle, write_manifest


@dataclass(frozen=True)
class FakeTensorSpec:
    name: str
    dtype: np.dtype
    shape: tuple[int, ...]


@dataclass(frozen=True)
class FakeModuleSpec:
    inputs: tuple[FakeTensorSpec, ...]
    outputs: tuple[FakeTensorSpec, ...]
    callback: object


@dataclass
class FakeDeviceBuffer:
    module: str
    direction: str
    name: str
    data: np.ndarray | None = None


@dataclass(frozen=True)
class FakeTensorInfo:
    shape: tuple[int, ...]
    dtype: np.dtype


class FakeOutput:
    def __init__(self, value: np.ndarray) -> None:
        self._value = value

    def numpy(self) -> np.ndarray:
        return self._value


class FakeWeightManager:
    def __init__(self, owner: FakeTCIMEnvironment, device: int) -> None:
        self.owner = owner
        self.device = device
        self.released = False
        owner.weight_managers.append(self)

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.owner.weight_manager_releases += 1


class FakeOption:
    def __init__(self, weight_manager: FakeWeightManager) -> None:
        self.weight_manager = weight_manager


class FakeModule:
    def __init__(self, owner: FakeTCIMEnvironment, path: str, spec: FakeModuleSpec) -> None:
        self.owner = owner
        self.path = path
        self.spec = spec
        self.set_inputs: dict[str, np.ndarray] = {}
        self.device_inputs: dict[str, FakeDeviceBuffer] = {}
        self.input_buffers = {item.name: FakeDeviceBuffer(path, "input", item.name) for item in spec.inputs}
        self.output_buffers = {item.name: FakeDeviceBuffer(path, "output", item.name) for item in spec.outputs}
        self.outputs: dict[str, np.ndarray] = {}
        self.runs = 0
        self.released = False

    def get_num_inputs(self) -> int:
        return len(self.spec.inputs)

    def get_input_name(self, index: int) -> str:
        return self.spec.inputs[index].name

    def get_input_info(self, name: str) -> FakeTensorInfo:
        spec = next(item for item in self.spec.inputs if item.name == name)
        return FakeTensorInfo(spec.shape, spec.dtype)

    def get_num_outputs(self) -> int:
        return len(self.spec.outputs)

    def get_output_name(self, index: int) -> str:
        return self.spec.outputs[index].name

    def get_output_info(self, name: str) -> FakeTensorInfo:
        spec = next(item for item in self.spec.outputs if item.name == name)
        return FakeTensorInfo(spec.shape, spec.dtype)

    def set_input(self, name: str, value: np.ndarray) -> None:
        if isinstance(value, FakeDeviceBuffer):
            raise TypeError("device buffers must use set_dev_input")
        spec = next(item for item in self.spec.inputs if item.name == name)
        array = np.asarray(value)
        if array.shape != spec.shape or array.dtype != spec.dtype:
            raise ValueError(f"{name} expected {spec.shape}/{spec.dtype}, got {array.shape}/{array.dtype}")
        self.set_inputs[name] = np.array(array, copy=True)

    def set_dev_input(self, name: str, handle: FakeDeviceBuffer) -> None:
        if not isinstance(handle, FakeDeviceBuffer):
            raise TypeError("set_dev_input requires a device buffer")
        self.device_inputs[name] = handle
        self.owner.device_links.append((self.path, name, handle))

    def get_dev_input(self, name: str) -> FakeDeviceBuffer:
        return self.input_buffers[name]

    def get_dev_output(self, name: str) -> FakeDeviceBuffer:
        return self.output_buffers[name]

    def run(self) -> None:
        values: list[np.ndarray] = []
        for spec in self.spec.inputs:
            if spec.name in self.device_inputs:
                value = self.device_inputs[spec.name].data
                if value is None:
                    raise RuntimeError(f"device input {spec.name} is uninitialized")
            elif spec.name in self.set_inputs:
                value = self.set_inputs[spec.name]
            else:
                handle = self.input_buffers[spec.name]
                if handle.data is None:
                    handle.data = np.zeros(spec.shape, dtype=spec.dtype)
                value = handle.data
            values.append(value)

        produced = self.spec.callback(tuple(values))
        if len(produced) != len(self.spec.outputs):
            raise RuntimeError("fake callback returned the wrong output count")
        self.outputs = {}
        for spec, value in zip(self.spec.outputs, produced, strict=True):
            array = np.ascontiguousarray(value, dtype=spec.dtype)
            if array.shape != spec.shape:
                raise RuntimeError(f"fake output {spec.name} has shape {array.shape}, expected {spec.shape}")
            self.outputs[spec.name] = array
            self.output_buffers[spec.name].data = array
        for handle in self.input_buffers.values():
            if handle.data is None:
                handle.data = np.ones((1,), dtype=np.int8)
        self.runs += 1

    def sync(self) -> None:
        pass

    def get_output(self, name: str) -> FakeOutput:
        return FakeOutput(self.outputs[name])

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.owner.release_order.append(self.path)


class FakeTCIMEnvironment:
    def __init__(self, model_specs: dict[str, FakeModuleSpec]) -> None:
        self.model_specs = model_specs
        self.fail_paths: set[str] = set()
        self.load_order: list[str] = []
        self.release_order: list[str] = []
        self.device_links: list[tuple[str, str, FakeDeviceBuffer]] = []
        self.modules: dict[str, FakeModule] = {}
        self.weight_managers: list[FakeWeightManager] = []
        self.weight_manager_releases = 0
        owner = self

        class Runtime:
            def WeightManager(self, *, device):
                return FakeWeightManager(owner, device)

            def Option(self, weight_manager):
                return FakeOption(weight_manager)

            def load(self, path: str, option=None) -> FakeModule:
                del option
                owner.load_order.append(path)
                if path in owner.fail_paths:
                    raise RuntimeError("fake TCIM load failure")
                module = FakeModule(owner, path, owner.model_specs[path])
                owner.modules[path] = module
                return module

        self.runtime = Runtime()


def _tensor(name: str, dtype: str, shape: tuple[int, ...]) -> FakeTensorSpec:
    return FakeTensorSpec(name, np.dtype(dtype), shape)


def _bundle_entries(root: Path, paths: tuple[str, ...]) -> list[BundleFile]:
    del root
    return [BundleFile(path=path) for path in paths]


def _write_compiled_manifest(root: Path, bundle_paths: tuple[str, ...], deployment: dict) -> None:
    entries = _bundle_entries(root, bundle_paths)
    deployment = {"uuid": TEST_DEPLOYMENT_UUID, "revision": 1, **deployment}
    write_manifest(
        root,
        {
            "schema_version": 2,
            "bundle": {
                "uuid": TEST_BUNDLE_UUID,
                "revision": 1,
                "name": "hmm-test",
                "files": [entry.model_dump(mode="json") for entry in entries],
                "digest": {
                    "algorithm": "sha256",
                    "scope": "structure",
                    "value": canonical_bundle_digest(TEST_BUNDLE_UUID, 1, "hmm-test", entries),
                },
            },
            "deployments": {"houmo": deployment},
        },
    )


def _artifact(root: Path, role: str, artifact_format: str = "hmm") -> dict[str, str]:
    suffix = ".pt" if artifact_format in {"pt", "pytorch"} else ".hmm"
    path = root / "artifacts" / f"{role}{suffix}"
    return {
        "path": str(path.relative_to(root)),
        "format": artifact_format,
    }


def _binding(
    semantic: str,
    name: str,
    index: int,
    dtype: str,
    shape: list[int],
    *,
    layout: str | None = None,
) -> dict[str, object]:
    if layout is None and len(shape) == 4:
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


def _write_embedding(path: Path, weight: np.ndarray) -> None:
    torch = pytest.importorskip("torch")
    torch.save({"weight": torch.from_numpy(weight)}, path)


def test_hmm_converts_bfloat16_torch_weights_to_float32(tmp_path):
    torch = pytest.importorskip("torch")
    value = torch.tensor([[1.25, -2.5]], dtype=torch.bfloat16)

    result = HMMBackend._to_numpy_weight(value, tmp_path / "embedding.pt", "weight")

    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, np.array([[1.25, -2.5]], dtype=np.float32))


def _pi05_context(tmp_path: Path, *, runtime_options=None) -> RuntimeContext:
    tmp_path.mkdir(parents=True, exist_ok=True)
    bundle_paths = create_policy_bundle(tmp_path, "pi05", include_weights=False)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "chunk_size": 2,
            "max_action_dim": 8,
            "num_inference_steps": 2,
            "min_period": 0.004,
            "max_period": 4.0,
        }
    )
    config["input_features"].update(
        {
            "observation.language.tokens": {"type": "LANGUAGE", "shape": [3]},
            "observation.language.attention_mask": {"type": "LANGUAGE", "shape": [3]},
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    roles = (
        "vision_top",
        "embedding",
        "prefill",
        "action_in_proj",
        "time_mlp",
        "decode",
        "action_out_proj",
    )
    for role in roles:
        if role == "embedding":
            continue
        (artifact_dir / f"{role}.hmm").write_bytes(role.encode())
    token_weight = np.arange(40, dtype=np.float32).reshape(10, 4) / 10.0
    _write_embedding(artifact_dir / "embedding.pt", token_weight)

    cache_source = _binding("internal.past_key.0", "past_key_0", 0, "float16", [1, 1])
    cache_target = _binding("internal.past_key.0", "past_key_0", 4, "float16", [1, 1])
    deployment = {
        "backend": "hmm",
        "target": {"soc": "lq50", "runtime": "tcim-lite"},
        "artifacts": {role: _artifact(tmp_path, role, "pt" if role == "embedding" else "hmm") for role in roles},
        "execution": list(roles),
        "bindings": {
            "vision_top": {
                "inputs": [
                    _binding(
                        "observation.images.top",
                        "pixel_values",
                        0,
                        "float16",
                        [1, 3, 4, 4],
                        layout="NCHW",
                    )
                ],
                "outputs": [_binding("internal.image_embedding.top", "image_features", 0, "float16", [1, 2, 4])],
            },
            "embedding": {
                "inputs": [
                    _binding("internal.image_embedding.top", "image", 0, "float16", [1, 2, 4]),
                    _binding("observation.language.tokens", "tokens", 1, "int64", [1, 3]),
                    _binding("observation.language.attention_mask", "mask", 2, "bool", [1, 3]),
                ],
                "outputs": [
                    _binding("internal.prefix_embeddings", "prefix", 0, "float16", [1, 6, 4]),
                    _binding("internal.prefix_attention", "prefix_attention", 1, "float16", [1, 1, 6, 8]),
                    _binding("internal.prefix_positions", "prefix_positions", 2, "int64", [1, 6]),
                    _binding("internal.decode_attention", "decode_attention", 3, "float16", [1, 1, 2, 8]),
                    _binding("internal.decode_positions", "decode_positions", 4, "int64", [1, 2]),
                ],
            },
            "prefill": {
                "inputs": [
                    _binding("internal.prefix_embeddings", "prefix_embs", 0, "float16", [1, 6, 4]),
                    _binding("internal.prefix_attention", "attention_mask", 1, "float16", [1, 1, 6, 8]),
                    _binding("internal.prefix_positions", "position_ids", 2, "int64", [1, 6]),
                ],
                "outputs": [cache_source],
            },
            "action_in_proj": {
                "inputs": [_binding("noise", "action_in", 0, "float16", [1, 2, 8])],
                "outputs": [_binding("internal.action_embedding", "action_in_proj_out", 0, "float16", [1, 2, 4])],
            },
            "time_mlp": {
                "inputs": [_binding("time", "time_emb", 0, "float16", [1, 4])],
                "outputs": [_binding("internal.time_condition", "time_mlp_out", 0, "float16", [1, 4])],
            },
            "decode": {
                "inputs": [
                    _binding("internal.action_embedding", "action_embs", 0, "float16", [1, 2, 4]),
                    _binding("internal.decode_attention", "attention_mask", 1, "float16", [1, 1, 2, 8]),
                    _binding("internal.decode_positions", "position_ids", 2, "int64", [1, 2]),
                    _binding("internal.time_condition", "condition", 3, "float16", [1, 4]),
                    cache_target,
                ],
                "outputs": [_binding("internal.suffix_hidden", "last_hidden_state", 0, "float16", [1, 2, 4])],
            },
            "action_out_proj": {
                "inputs": [_binding("internal.suffix_hidden", "action_out", 0, "float16", [1, 2, 4])],
                "outputs": [_binding("action", "action_out_proj_out", 0, "float16", [1, 2, 8])],
            },
        },
        "device_links": [
            {
                "semantic": "internal.past_key.0",
                "producer": "prefill",
                "consumer": "decode",
                "transport": "device_pointer",
                "owner": "producer",
                "lifetime": "inference",
            }
        ],
    }
    _write_compiled_manifest(tmp_path, bundle_paths, deployment)
    return RuntimeContext(load_inference_manifest(tmp_path, "houmo"), runtime_options=runtime_options or {})


def _smolvla_context(tmp_path: Path, *, runtime_options=None) -> RuntimeContext:
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
            "observation.language.tokens": {"type": "LANGUAGE", "shape": [3]},
            "observation.language.attention_mask": {"type": "LANGUAGE", "shape": [3]},
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    for role in ("vision_top", "prefill", "action"):
        (artifact_dir / f"{role}.hmm").write_bytes(role.encode())
    token_weight = np.arange(40, dtype=np.float32).reshape(10, 4) / 10.0
    state_weight = np.zeros((4, 8), dtype=np.float32)
    state_weight[:, :4] = np.eye(4, dtype=np.float32)
    state_bias = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    _write_embedding(artifact_dir / "embedding.pt", token_weight)
    torch = pytest.importorskip("torch")
    torch.save(
        {"weight": torch.from_numpy(state_weight), "bias": torch.from_numpy(state_bias)},
        artifact_dir / "state_projection.pt",
    )

    deployment = {
        "backend": "hmm",
        "target": {"soc": "lq50", "runtime": "tcim-lite"},
        "artifacts": {
            "vision_top": _artifact(tmp_path, "vision_top"),
            "embedding": _artifact(tmp_path, "embedding", "pt"),
            "state_projection": _artifact(tmp_path, "state_projection", "pt"),
            "prefill": _artifact(tmp_path, "prefill"),
            "action": _artifact(tmp_path, "action"),
        },
        "execution": ["vision_top", "embedding", "prefill", "action"],
        "bindings": {
            "vision_top": {
                "inputs": [
                    _binding(
                        "observation.images.top",
                        "pixel_values",
                        0,
                        "float16",
                        [1, 3, 4, 4],
                        layout="NCHW",
                    )
                ],
                "outputs": [_binding("internal.image_embedding.top", "image_embeddings", 0, "float16", [1, 2, 4])],
            },
            "embedding": {
                "inputs": [
                    _binding("internal.image_embedding.top", "image", 0, "float16", [1, 2, 4]),
                    _binding("observation.language.tokens", "tokens", 1, "int64", [1, 3]),
                    _binding("observation.language.attention_mask", "mask", 2, "bool", [1, 3]),
                    _binding("observation.state", "state", 3, "float32", [1, 8]),
                ],
                "outputs": [
                    _binding("internal.prefix_embeddings", "prefix_embs", 0, "float16", [1, 7, 4]),
                    _binding("internal.prefix_pad_masks", "prefix_pad_masks", 1, "bool", [1, 7]),
                    _binding("internal.attention_mask", "attention_mask", 2, "int32", [1, 7, 7]),
                    _binding("internal.position_ids", "position_ids", 3, "int32", [1, 7]),
                ],
            },
            "prefill": {
                "inputs": [
                    _binding("internal.prefix_embeddings", "prefix_embs", 0, "float16", [1, 7, 4]),
                    _binding("internal.attention_mask", "attention_mask", 1, "int32", [1, 7, 7]),
                    _binding("internal.position_ids", "position_ids", 2, "int32", [1, 7]),
                ],
                "outputs": [
                    _binding("internal.past_key.0", "past_key_0", 0, "float16", [1, 7, 1, 2]),
                    _binding("internal.past_value.0", "past_value_0", 1, "float16", [1, 7, 1, 2]),
                ],
            },
            "action": {
                "inputs": [
                    _binding("noise", "x_t", 0, "float16", [1, 2, 8]),
                    _binding("time", "timestep", 1, "float16", [1]),
                    _binding("internal.prefix_pad_masks", "prefix_pad_masks", 2, "bool", [1, 7]),
                    _binding("internal.past_key.0", "past_key_0", 3, "float16", [1, 7, 1, 2]),
                    _binding("internal.past_value.0", "past_value_0", 4, "float16", [1, 7, 1, 2]),
                ],
                "outputs": [_binding("action", "v_t", 0, "float16", [1, 2, 8])],
            },
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
    _write_compiled_manifest(tmp_path, bundle_paths, deployment)
    return RuntimeContext(load_inference_manifest(tmp_path, "houmo"), runtime_options=runtime_options or {})


def _pi05_environment(context: RuntimeContext, observed_times: list[float]) -> FakeTCIMEnvironment:
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}

    def vision(inputs):
        (image,) = inputs
        return [np.full((1, 2, 4), float(image.mean()), dtype=np.float16)]

    def prefill(inputs):
        prefix, attention, positions = inputs
        assert prefix.shape == (1, 6, 4)
        assert attention.shape == (1, 1, 6, 8)
        np.testing.assert_array_equal(positions, np.arange(6, dtype=np.int64)[None, :])
        return [np.full((1, 1), 3.0, dtype=np.float16)]

    def action_in(inputs):
        (noise,) = inputs
        return [noise[..., :4]]

    def time_mlp(inputs):
        (time_embedding,) = inputs
        observed_times.append(float(time_embedding[0, 0]))
        return [time_embedding]

    def decode(inputs):
        action_embedding, attention, positions, condition, cache = inputs
        assert condition.shape == (1, 4)
        assert attention.shape == (1, 1, 2, 8)
        np.testing.assert_array_equal(positions, np.array([[6, 7]], dtype=np.int64))
        np.testing.assert_array_equal(cache, np.full((1, 1), 3.0, dtype=np.float16))
        return [action_embedding]

    def action_out(inputs):
        (hidden,) = inputs
        return [np.ones((*hidden.shape[:-1], 8), dtype=np.float16)]

    return FakeTCIMEnvironment(
        {
            paths["vision_top"]: FakeModuleSpec(
                inputs=(_tensor("pixel_values", "float16", (1, 3, 4, 4)),),
                outputs=(_tensor("image_features", "float16", (1, 2, 4)),),
                callback=vision,
            ),
            paths["prefill"]: FakeModuleSpec(
                inputs=(
                    _tensor("prefix_embs", "float16", (1, 6, 4)),
                    _tensor("attention_mask", "float16", (1, 1, 6, 8)),
                    _tensor("position_ids", "int64", (1, 6)),
                ),
                outputs=(_tensor("past_key_0", "float16", (1, 1)),),
                callback=prefill,
            ),
            paths["action_in_proj"]: FakeModuleSpec(
                inputs=(_tensor("action_in", "float16", (1, 2, 8)),),
                outputs=(_tensor("action_in_proj_out", "float16", (1, 2, 4)),),
                callback=action_in,
            ),
            paths["time_mlp"]: FakeModuleSpec(
                inputs=(_tensor("time_emb", "float16", (1, 4)),),
                outputs=(_tensor("time_mlp_out", "float16", (1, 4)),),
                callback=time_mlp,
            ),
            paths["decode"]: FakeModuleSpec(
                inputs=(
                    _tensor("action_embs", "float16", (1, 2, 4)),
                    _tensor("attention_mask", "float16", (1, 1, 2, 8)),
                    _tensor("position_ids", "int64", (1, 2)),
                    _tensor("condition", "float16", (1, 4)),
                    _tensor("past_key_0", "float16", (1, 1)),
                ),
                outputs=(_tensor("last_hidden_state", "float16", (1, 2, 4)),),
                callback=decode,
            ),
            paths["action_out_proj"]: FakeModuleSpec(
                inputs=(_tensor("action_out", "float16", (1, 2, 4)),),
                outputs=(_tensor("action_out_proj_out", "float16", (1, 2, 8)),),
                callback=action_out,
            ),
        }
    )


def _smolvla_environment(context: RuntimeContext, observed_times: list[float]) -> FakeTCIMEnvironment:
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}

    def vision(inputs):
        (image,) = inputs
        return [np.full((1, 2, 4), float(image.mean()), dtype=np.float16)]

    def prefill(inputs):
        prefix, attention, position_ids = inputs
        assert prefix.shape == (1, 7, 4)
        assert attention.dtype == np.dtype("int32")
        assert position_ids.dtype == np.dtype("int32")
        key = np.full((1, 7, 1, 2), 3.0, dtype=np.float16)
        value = np.full((1, 7, 1, 2), 4.0, dtype=np.float16)
        return [key, value]

    def action(inputs):
        noise, time_value, masks, key, value = inputs
        observed_times.append(float(time_value[0]))
        assert time_value.dtype == np.dtype("float16")
        assert masks.dtype == np.dtype("bool")
        np.testing.assert_array_equal(key, np.full((1, 7, 1, 2), 3.0, dtype=np.float16))
        np.testing.assert_array_equal(value, np.full((1, 7, 1, 2), 4.0, dtype=np.float16))
        return [np.ones_like(noise)]

    return FakeTCIMEnvironment(
        {
            paths["vision_top"]: FakeModuleSpec(
                inputs=(_tensor("pixel_values", "float16", (1, 3, 4, 4)),),
                outputs=(_tensor("image_embeddings", "float16", (1, 2, 4)),),
                callback=vision,
            ),
            paths["prefill"]: FakeModuleSpec(
                inputs=(
                    _tensor("prefix_embs", "float16", (1, 7, 4)),
                    _tensor("attention_mask", "int32", (1, 7, 7)),
                    _tensor("position_ids", "int32", (1, 7)),
                ),
                outputs=(
                    _tensor("past_key_0", "float16", (1, 7, 1, 2)),
                    _tensor("past_value_0", "float16", (1, 7, 1, 2)),
                ),
                callback=prefill,
            ),
            paths["action"]: FakeModuleSpec(
                inputs=(
                    _tensor("x_t", "float16", (1, 2, 8)),
                    _tensor("timestep", "float16", (1,)),
                    _tensor("prefix_pad_masks", "bool", (1, 7)),
                    _tensor("past_key_0", "float16", (1, 7, 1, 2)),
                    _tensor("past_value_0", "float16", (1, 7, 1, 2)),
                ),
                outputs=(_tensor("v_t", "float16", (1, 2, 8)),),
                callback=action,
            ),
        }
    )


def test_hmm_pi05_pipeline_uses_prefill_output_cache_and_all_projection_modules(tmp_path, monkeypatch):
    from inference_service import pipeline as pipeline_module
    from inference_service.model_sessions import HMMModelSession
    from inference_service.pipeline import create_inference_pipeline

    monkeypatch.setattr(
        pipeline_module.factory,
        "create_lerobot_processor_views",
        lambda: (lambda inputs: inputs, lambda action: action),
    )

    context = _pi05_context(tmp_path, runtime_options={"device_id": 0, "random_seed": 7})
    observed_times: list[float] = []
    environment = _pi05_environment(context, observed_times)

    def session_factory(ctx, options):
        return HMMModelSession(
            device_id=int(options["device_id"]),
            runtime_loader=lambda: environment.runtime,
        )

    pipeline = create_inference_pipeline(
        "pi05",
        context.validated_manifest,
        runtime_options={"device_id": 0, "random_seed": 7},
        model_session_factory=session_factory,
    )
    pipeline.load()

    result = pipeline.infer(
        InferenceRequest(
            request_id="pi05",
            inputs={
                "observation.images.top": np.ones((3, 4, 4), dtype=np.float32),
                "observation.language.tokens": np.array([1, 2, 3], dtype=np.int64),
                "observation.language.attention_mask": np.array([True, True, False]),
                "noise": np.zeros((1, 2, 8), dtype=np.float32),
            },
        )
    )

    np.testing.assert_array_equal(result.action, np.full((1, 2, 6), -1.0, dtype=np.float16))
    assert result.actual_chunk_size == 2
    assert pipeline.state.value == "ready"
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}
    assert environment.modules[paths["prefill"]].runs == 1
    for role in ("action_in_proj", "time_mlp", "decode", "action_out_proj"):
        assert environment.modules[paths[role]].runs == 2
    assert len(environment.device_links) == 1
    _, target_name, handle = environment.device_links[0]
    assert target_name == "past_key_0"
    assert handle.direction == "output"
    assert handle.name == "past_key_0"
    assert len(observed_times) == 2

    pipeline.close()
    pipeline.close()
    assert environment.release_order == [
        paths["action_out_proj"],
        paths["decode"],
        paths["time_mlp"],
        paths["action_in_proj"],
        paths["prefill"],
        paths["vision_top"],
    ]
    assert environment.weight_manager_releases == 1


def _patch_identity_processors(monkeypatch):
    from inference_service import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module.factory,
        "create_lerobot_processor_views",
        lambda: (lambda inputs: inputs, lambda action: action),
    )


def _smolvla_session_factory(environment):
    from inference_service.model_sessions import HMMModelSession

    def factory(ctx, options):
        return HMMModelSession(
            device_id=int(options["device_id"]),
            runtime_loader=lambda: environment.runtime,
        )

    return factory


def test_hmm_smolvla_pipeline_runs_through_session_with_device_links_and_host_embedding(tmp_path, monkeypatch):
    from inference_service.pipeline import create_inference_pipeline

    _patch_identity_processors(monkeypatch)
    context = _smolvla_context(tmp_path, runtime_options={"device_id": 0, "random_seed": 7})
    observed_times: list[float] = []
    environment = _smolvla_environment(context, observed_times)
    pipeline = create_inference_pipeline(
        "smolvla",
        context.validated_manifest,
        runtime_options={"device_id": 0, "random_seed": 7},
        model_session_factory=_smolvla_session_factory(environment),
    )
    pipeline.load()

    result = pipeline.infer(
        InferenceRequest(
            request_id="smolvla",
            inputs={
                "observation.state": np.arange(1, 7, dtype=np.float32),
                "observation.images.top": np.ones((3, 4, 4), dtype=np.float32),
                "observation.language.tokens": np.array([1, 2, 3], dtype=np.int64),
                "observation.language.attention_mask": np.array([True, True, False]),
                "noise": np.zeros((1, 2, 8), dtype=np.float32),
            },
        )
    )

    np.testing.assert_array_equal(result.action, np.full((1, 2, 6), -1.0, dtype=np.float16))
    assert result.actual_chunk_size == 2
    assert observed_times == [1.0, 0.5]
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}
    assert environment.modules[paths["prefill"]].runs == 1
    assert environment.modules[paths["action"]].runs == 2
    # HMM device links stay device-resident: KV handles are producer outputs.
    assert all(link[2].direction == "output" for link in environment.device_links)
    prefill_inputs = environment.modules[paths["prefill"]].set_inputs
    np.testing.assert_allclose(
        prefill_inputs["prefix_embs"][0, 5],
        np.array([1.25, 2.5, 3.75, 5.0], dtype=np.float16),
        atol=1e-3,
    )

    pipeline.close()
    assert environment.release_order == [paths["action"], paths["prefill"], paths["vision_top"]]
    assert environment.weight_manager_releases == 1


def test_hmm_smolvla_partial_load_failure_releases_loaded_resources(tmp_path, monkeypatch):
    from inference_service.pipeline import create_inference_pipeline

    _patch_identity_processors(monkeypatch)
    context = _smolvla_context(tmp_path, runtime_options={"device_id": 0, "random_seed": 7})
    environment = _smolvla_environment(context, [])
    action_path = str(context.resolved_artifacts["action"])
    environment.fail_paths.add(action_path)
    pipeline = create_inference_pipeline(
        "smolvla",
        context.validated_manifest,
        runtime_options={"device_id": 0, "random_seed": 7},
        model_session_factory=_smolvla_session_factory(environment),
    )

    with pytest.raises(BackendLoadError, match="fake TCIM load failure"):
        pipeline.load()

    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}
    assert environment.release_order == [paths["prefill"], paths["vision_top"]]
    assert environment.weight_manager_releases == 1
    pipeline.close()


def test_hmm_smolvla_rejects_runtime_descriptor_mismatch_before_inference(tmp_path, monkeypatch):
    from inference_service.pipeline import create_inference_pipeline

    _patch_identity_processors(monkeypatch)
    context = _smolvla_context(tmp_path, runtime_options={"device_id": 0})
    environment = _smolvla_environment(context, [])
    vision_path = str(context.resolved_artifacts["vision_top"])
    spec = environment.model_specs[vision_path]
    environment.model_specs[vision_path] = FakeModuleSpec(
        inputs=(_tensor("wrong_name", "float16", (1, 3, 4, 4)),),
        outputs=spec.outputs,
        callback=spec.callback,
    )
    pipeline = create_inference_pipeline(
        "smolvla",
        context.validated_manifest,
        runtime_options={"device_id": 0},
        model_session_factory=_smolvla_session_factory(environment),
    )

    with pytest.raises(BackendLoadError) as error:
        pipeline.load()

    assert error.value.code == "runtime_name_mismatch"
    pipeline.close()


def test_hmm_backend_fails_closed_for_smolvla_after_factory_cutover(tmp_path):
    context = _smolvla_context(tmp_path)
    environment = _smolvla_environment(context, [])
    backend = HMMBackend(0, runtime_loader=lambda: environment.runtime)

    with pytest.raises(BackendLoadError, match="no longer hosts SmolVLA") as error:
        backend.load(context)

    assert error.value.code == "unsupported_policy_backend_pair"
    backend.close()


def test_hmm_smolvla_repeated_inference_is_deterministic_with_seed(tmp_path, monkeypatch):
    from inference_service.pipeline import create_inference_pipeline

    _patch_identity_processors(monkeypatch)
    context = _smolvla_context(tmp_path, runtime_options={"device_id": 0, "random_seed": 42})
    environment = _smolvla_environment(context, [])
    pipeline = create_inference_pipeline(
        "smolvla",
        context.validated_manifest,
        runtime_options={"device_id": 0, "random_seed": 42},
        model_session_factory=_smolvla_session_factory(environment),
    )
    pipeline.load()

    request = InferenceRequest(
        request_id="smolvla",
        inputs={
            "observation.state": np.arange(1, 7, dtype=np.float32),
            "observation.images.top": np.ones((3, 4, 4), dtype=np.float32),
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


def test_hmm_smolvla_seed_reproduces_sampled_noise_across_instances(tmp_path, monkeypatch):
    from inference_service.pipeline import create_inference_pipeline

    _patch_identity_processors(monkeypatch)

    counter = 0

    def build_and_infer(seed: int) -> np.ndarray:
        nonlocal counter
        counter += 1
        root = tmp_path / f"seed-{seed}-{counter}"
        context = _smolvla_context(root, runtime_options={"device_id": 0, "random_seed": seed})
        environment = _smolvla_environment(context, [])
        pipeline = create_inference_pipeline(
            "smolvla",
            context.validated_manifest,
            runtime_options={"device_id": 0, "random_seed": seed},
            model_session_factory=_smolvla_session_factory(environment),
        )
        pipeline.load()
        try:
            result = pipeline.infer(
                InferenceRequest(
                    request_id="smolvla",
                    inputs={
                        "observation.state": np.arange(1, 7, dtype=np.float32),
                        "observation.images.top": np.ones((3, 4, 4), dtype=np.float32),
                        "observation.language.tokens": np.array([1, 2, 3], dtype=np.int64),
                        "observation.language.attention_mask": np.array([True, True, False]),
                    },
                )
            )
            return np.asarray(result.action)
        finally:
            pipeline.close()

    np.testing.assert_array_equal(build_and_infer(42), build_and_infer(42))


def test_hmm_registry_factory_is_lazy_and_reset_is_unsupported(tmp_path):
    context = _smolvla_context(tmp_path)
    backend = BACKEND_REGISTRY.create(context)

    assert isinstance(backend, HMMBackend)
    with pytest.raises(BackendCapabilityError):
        backend.reset()
    backend.close()


@pytest.mark.parametrize(
    "runtime_options",
    [
        {"device_id": -1},
        {"random_seed": 1.5},
        {"unknown": True},
    ],
)
def test_hmm_rejects_invalid_runtime_options(tmp_path, runtime_options):
    context = _smolvla_context(tmp_path, runtime_options=runtime_options)

    with pytest.raises(BackendLoadError) as error:
        create_backend(context)

    assert error.value.code == "invalid_runtime_options"


def test_hmm_backend_fails_closed_for_pi05_after_factory_cutover(tmp_path):
    context = _pi05_context(tmp_path)
    environment = _pi05_environment(context, [])
    backend = HMMBackend(0, runtime_loader=lambda: environment.runtime)

    with pytest.raises(BackendLoadError, match="no longer hosts PI0.5") as error:
        backend.load(context)

    assert error.value.code == "unsupported_policy_backend_pair"
    backend.close()


def test_hmm_pi05_repeated_inference_is_deterministic_with_seed(tmp_path, monkeypatch):
    from inference_service import pipeline as pipeline_module
    from inference_service.model_sessions import HMMModelSession
    from inference_service.pipeline import create_inference_pipeline

    monkeypatch.setattr(
        pipeline_module.factory,
        "create_lerobot_processor_views",
        lambda: (lambda inputs: inputs, lambda action: action),
    )

    context = _pi05_context(tmp_path, runtime_options={"device_id": 0, "random_seed": 42})
    environment = _pi05_environment(context, [])

    def session_factory(ctx, options):
        return HMMModelSession(
            device_id=int(options["device_id"]),
            runtime_loader=lambda: environment.runtime,
        )

    pipeline = create_inference_pipeline(
        "pi05",
        context.validated_manifest,
        runtime_options={"device_id": 0, "random_seed": 42},
        model_session_factory=session_factory,
    )
    pipeline.load()

    request = InferenceRequest(
        request_id="pi05",
        inputs={
            "observation.images.top": np.ones((3, 4, 4), dtype=np.float32),
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
