from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from inference_manifest import BundleFile, canonical_bundle_digest, load_inference_manifest
from inference_manifest.models import DeviceLink
from inference_service.backends import (
    BACKEND_REGISTRY,
    BackendCapabilityError,
    BackendDescriptor,
    BackendLoadError,
    BackendRegistry,
    BackendState,
    InferenceRequest,
    RuntimeContext,
)
from inference_service.backends.ascend import AscendBackend, create_backend
from inference_service.backends.ascend.acl_runtime import AclRuntimeManager
from inference_service.codecs import CodecRequest, build_execution_plan, create_policy_codec
from inference_service.core.pure_inference_engine import PureInferenceEngine
from inference_service.pi05_schedule import load_pi05_schedule
from inference_service.pipeline import InferencePipeline
from inference_service.pipeline import factory as pipeline_factory
from tests.manifest_fixtures import TEST_BUNDLE_UUID, TEST_DEPLOYMENT_UUID, create_policy_bundle, write_manifest

_DTYPE_CODES = {
    np.dtype("float32"): 0,
    np.dtype("float16"): 1,
    np.dtype("int8"): 2,
    np.dtype("int32"): 3,
    np.dtype("uint8"): 4,
    np.dtype("int16"): 6,
    np.dtype("int64"): 9,
    np.dtype("float64"): 11,
    np.dtype("bool"): 12,
}


@dataclass(frozen=True)
class FakeTensorSpec:
    name: str
    dtype: np.dtype
    shape: tuple[int, ...]

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize


@dataclass(frozen=True)
class FakeModelSpec:
    inputs: tuple[FakeTensorSpec, ...]
    outputs: tuple[FakeTensorSpec, ...]
    callback: object


class FakeAclRT:
    def __init__(self, owner: FakeAcl) -> None:
        self.owner = owner

    def set_device(self, device_id):
        self.owner.set_device_calls.append(device_id)
        return 0

    def reset_device(self, device_id):
        self.owner.reset_device_calls.append(device_id)
        return 0

    def create_context(self, device_id):
        context = ("context", self.owner.next_identifier())
        self.owner.contexts.add(context)
        return context, 0

    def set_context(self, context):
        if context not in self.owner.contexts:
            return 1
        self.owner.context_bindings.append(context)
        return 0

    def destroy_context(self, context):
        self.owner.contexts.remove(context)
        self.owner.destroyed_contexts.append(context)
        return 0

    def malloc(self, size, policy):
        del policy
        pointer = ("device", self.owner.next_identifier())
        self.owner.memory[pointer] = bytearray(size)
        return pointer, 0

    def free(self, pointer):
        self.owner.memory.pop(pointer)
        self.owner.freed_pointers.append(pointer)
        return 0

    def malloc_host(self, size):
        pointer = ("host", self.owner.next_identifier())
        self.owner.memory[pointer] = bytearray(size)
        return pointer, 0

    def free_host(self, pointer):
        self.owner.memory.pop(pointer)
        self.owner.freed_host_pointers.append(pointer)
        return 0

    def memcpy(self, destination, destination_size, source, count, kind):
        self.owner.memcpy_calls.append((destination, source, count, kind))
        payload = self.owner.read_pointer(source, count)
        if count > destination_size:
            return 1
        self.owner.memory[destination][:count] = payload
        return 0


class FakeAclUtil:
    def __init__(self, owner: FakeAcl) -> None:
        self.owner = owner

    @staticmethod
    def bytes_to_ptr(payload):
        return payload

    def ptr_to_bytes(self, pointer, size):
        return self.owner.read_pointer(pointer, size)


class FakeAclMDL:
    def __init__(self, owner: FakeAcl) -> None:
        self.owner = owner

    def load_from_file(self, path):
        if path in self.owner.fail_model_paths:
            return None, 1
        model_id = ("model", self.owner.next_identifier())
        self.owner.loaded_models[model_id] = self.owner.model_specs[path]
        self.owner.model_paths[model_id] = path
        return model_id, 0

    @staticmethod
    def create_desc():
        return {}

    def get_desc(self, descriptor, model_id):
        descriptor["spec"] = self.owner.loaded_models[model_id]
        return 0

    @staticmethod
    def destroy_desc(descriptor):
        descriptor.clear()
        return 0

    def unload(self, model_id):
        self.owner.unloaded_models.append(model_id)
        self.owner.loaded_models.pop(model_id)
        self.owner.model_paths.pop(model_id)
        return 0

    @staticmethod
    def get_num_inputs(descriptor):
        return len(descriptor["spec"].inputs)

    @staticmethod
    def get_num_outputs(descriptor):
        return len(descriptor["spec"].outputs)

    @staticmethod
    def get_input_size_by_index(descriptor, index):
        return descriptor["spec"].inputs[index].size

    @staticmethod
    def get_output_size_by_index(descriptor, index):
        return descriptor["spec"].outputs[index].size

    @staticmethod
    def get_input_name_by_index(descriptor, index):
        return descriptor["spec"].inputs[index].name.encode()

    @staticmethod
    def get_output_name_by_index(descriptor, index):
        return descriptor["spec"].outputs[index].name.encode()

    @staticmethod
    def get_input_data_type(descriptor, index):
        return _DTYPE_CODES[descriptor["spec"].inputs[index].dtype]

    @staticmethod
    def get_output_data_type(descriptor, index):
        return _DTYPE_CODES[descriptor["spec"].outputs[index].dtype]

    @staticmethod
    def get_input_dims(descriptor, index):
        return {"dims": descriptor["spec"].inputs[index].shape}, 0

    @staticmethod
    def get_output_dims(descriptor, index):
        return {"dims": descriptor["spec"].outputs[index].shape}, 0

    @staticmethod
    def create_dataset():
        return []

    @staticmethod
    def add_dataset_buffer(dataset, data_buffer):
        dataset.append(data_buffer)
        return len(dataset) - 1, 0

    @staticmethod
    def destroy_dataset(dataset):
        dataset.clear()
        return 0

    def execute(self, model_id, input_dataset, output_dataset):
        spec = self.owner.loaded_models[model_id]
        self.owner.executions.append(
            (
                self.owner.model_paths[model_id],
                tuple(item["pointer"] for item in input_dataset),
                tuple(item["pointer"] for item in output_dataset),
            )
        )
        inputs = [
            np.frombuffer(self.owner.read_pointer(item["pointer"], tensor.size), dtype=tensor.dtype)
            .reshape(tensor.shape)
            .copy()
            for item, tensor in zip(input_dataset, spec.inputs, strict=True)
        ]
        with self.owner.execution_lock:
            self.owner.active_executions += 1
            self.owner.max_active_executions = max(self.owner.max_active_executions, self.owner.active_executions)
        try:
            outputs = spec.callback(inputs)
            if self.owner.execution_delay:
                time.sleep(self.owner.execution_delay)
        finally:
            with self.owner.execution_lock:
                self.owner.active_executions -= 1
        for value, item, tensor in zip(outputs, output_dataset, spec.outputs, strict=True):
            array = np.ascontiguousarray(value, dtype=tensor.dtype).reshape(tensor.shape)
            self.owner.memory[item["pointer"]][:] = array.tobytes()
        return 0


class FakeAcl:
    def __init__(self, model_specs: dict[str, FakeModelSpec]) -> None:
        self.model_specs = model_specs
        self.fail_model_paths: set[str] = set()
        self.loaded_models: dict[object, FakeModelSpec] = {}
        self.model_paths: dict[object, str] = {}
        self.unloaded_models: list[object] = []
        self.executions: list[tuple[str, tuple[object, ...], tuple[object, ...]]] = []
        self.memcpy_calls: list[tuple[object, object, int, int]] = []
        self.memory: dict[object, bytearray] = {}
        self.freed_pointers: list[object] = []
        self.freed_host_pointers: list[object] = []
        self.contexts: set[object] = set()
        self.destroyed_contexts: list[object] = []
        self.context_bindings: list[object] = []
        self.set_device_calls: list[int] = []
        self.reset_device_calls: list[int] = []
        self.init_calls: list[object] = []
        self.finalize_calls = 0
        self.destroyed_data_buffers = 0
        self.execution_delay = 0.0
        self.active_executions = 0
        self.max_active_executions = 0
        self.execution_lock = threading.Lock()
        self._identifier = 0
        self.rt = FakeAclRT(self)
        self.util = FakeAclUtil(self)
        self.mdl = FakeAclMDL(self)

    def next_identifier(self) -> int:
        self._identifier += 1
        return self._identifier

    def init(self, config_path=None):
        self.init_calls.append(config_path)
        return 0

    def finalize(self):
        self.finalize_calls += 1
        return 0

    @staticmethod
    def create_data_buffer(pointer, size):
        return {"pointer": pointer, "size": size}

    def destroy_data_buffer(self, data_buffer):
        data_buffer.clear()
        self.destroyed_data_buffers += 1
        return 0

    def read_pointer(self, pointer, size):
        if isinstance(pointer, bytes | bytearray):
            return bytes(pointer[:size])
        return bytes(self.memory[pointer][:size])


def _tensor(name: str, dtype: str, shape: tuple[int, ...]) -> FakeTensorSpec:
    return FakeTensorSpec(name, np.dtype(dtype), shape)


def _bundle_entries(root: Path, paths: tuple[str, ...]) -> list[BundleFile]:
    del root
    return [BundleFile(path=path) for path in paths]


def _write_compiled_manifest(
    root: Path, bundle_paths: tuple[str, ...], deployment: dict, *, deployment_name: str = "ascend"
) -> None:
    entries = _bundle_entries(root, bundle_paths)
    deployment = {"uuid": TEST_DEPLOYMENT_UUID, "revision": 1, **deployment}
    write_manifest(
        root,
        {
            "schema_version": 2,
            "bundle": {
                "uuid": TEST_BUNDLE_UUID,
                "revision": 1,
                "name": "ascend-test",
                "files": [entry.model_dump(mode="json") for entry in entries],
                "digest": {
                    "algorithm": "sha256",
                    "scope": "structure",
                    "value": canonical_bundle_digest(TEST_BUNDLE_UUID, 1, "ascend-test", entries),
                },
            },
            "deployments": {deployment_name: deployment},
        },
    )


def _act_context(tmp_path: Path, *, runtime_options=None) -> RuntimeContext:
    tmp_path.mkdir(parents=True, exist_ok=True)
    bundle_paths = create_policy_bundle(tmp_path, include_weights=False)
    model = tmp_path / "artifacts" / "policy.om"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"act-om")
    _write_compiled_manifest(
        tmp_path,
        bundle_paths,
        {
            "backend": "ascend",
            "target": {"soc": "ascend310", "runtime": "acl"},
            "artifacts": {"policy": {"path": "artifacts/policy.om", "format": "om"}},
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
                            "shape": [1, 3, 16, 24],
                            "layout": "NCHW",
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
    return RuntimeContext(load_inference_manifest(tmp_path, "ascend"), runtime_options=runtime_options or {})


def _pi05_context(
    tmp_path: Path,
    *,
    runtime_options=None,
    deployment_name: str = "ascend",
    chunk_size: int = 2,
    max_action_dim: int = 8,
    num_inference_steps: int = 2,
    action_runtime_name: str = "action",
    schedule: dict[str, object] | str | None = None,
) -> RuntimeContext:
    tmp_path.mkdir(parents=True, exist_ok=True)
    bundle_paths = create_policy_bundle(tmp_path, "pi05", include_weights=False)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "chunk_size": chunk_size,
            "max_action_dim": max_action_dim,
            "num_inference_steps": num_inference_steps,
        }
    )
    config["input_features"].update(
        {
            "observation.language.tokens": {"type": "LANGUAGE", "shape": [4]},
            "observation.language.attention_mask": {"type": "LANGUAGE", "shape": [4]},
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    vlm = tmp_path / "artifacts" / "vlm.om"
    action_expert = tmp_path / "artifacts" / "action_expert.om"
    vlm.parent.mkdir(parents=True)
    vlm.write_bytes(b"vlm-om")
    action_expert.write_bytes(b"action-expert-om")
    artifacts = {
        "vlm": {"path": "artifacts/vlm.om", "format": "om"},
        "action_expert": {
            "path": "artifacts/action_expert.om",
            "format": "om",
        },
    }
    if schedule is not None:
        schedule_path = tmp_path / "artifacts" / "denoising_schedule.json"
        if isinstance(schedule, str):
            schedule_path.write_text(schedule, encoding="utf-8")
        else:
            schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
        artifacts["denoising_schedule"] = {
            "path": "artifacts/denoising_schedule.json",
            "format": "json",
        }
    _write_compiled_manifest(
        tmp_path,
        bundle_paths,
        {
            "backend": "ascend",
            "target": {"soc": "ascend310", "runtime": "acl"},
            "artifacts": artifacts,
            "execution": ["vlm", "action_expert"],
            "bindings": {
                "vlm": {
                    "inputs": [
                        {
                            "semantic": "observation.images.top",
                            "runtime_name": "image",
                            "index": 0,
                            "dtype": "float32",
                            "shape": [1, 3, 16, 24],
                            "layout": "NCHW",
                        },
                        {
                            "semantic": "observation.language.tokens",
                            "runtime_name": "tokens",
                            "index": 1,
                            "dtype": "int64",
                            "shape": [1, 4],
                        },
                        {
                            "semantic": "observation.language.attention_mask",
                            "runtime_name": "masks",
                            "index": 2,
                            "dtype": "bool",
                            "shape": [1, 4],
                        },
                        {
                            "semantic": "prefix_att_2d_masks_4d",
                            "runtime_name": "prefix_mask",
                            "index": 3,
                            "dtype": "float32",
                            "shape": [1, 1, 8, 8],
                            "layout": "NCHW",
                        },
                    ],
                    "outputs": [
                        {
                            "semantic": "internal.past_kv",
                            "runtime_name": "past_kv",
                            "index": 0,
                            "dtype": "float32",
                            "shape": [1, 2],
                        },
                        {
                            "semantic": "internal.prefix_pad_masks",
                            "runtime_name": "prefix_pad_masks",
                            "index": 1,
                            "dtype": "bool",
                            "shape": [1, 4],
                        },
                    ],
                },
                "action_expert": {
                    "inputs": [
                        {
                            "semantic": "internal.past_kv",
                            "runtime_name": "past_kv",
                            "index": 0,
                            "dtype": "float32",
                            "shape": [1, 2],
                        },
                        {
                            "semantic": "internal.prefix_pad_masks",
                            "runtime_name": "prefix_pad_masks",
                            "index": 1,
                            "dtype": "bool",
                            "shape": [1, 4],
                        },
                        {
                            "semantic": "time",
                            "runtime_name": "time",
                            "index": 2,
                            "dtype": "float32",
                            "shape": [1],
                        },
                        {
                            "semantic": "noise",
                            "runtime_name": "noise",
                            "index": 3,
                            "dtype": "float32",
                            "shape": [1, 2, 8],
                        },
                    ],
                    "outputs": [
                        {
                            "semantic": "action",
                            "runtime_name": action_runtime_name,
                            "index": 0,
                            "dtype": "float32",
                            "shape": [1, 2, 8],
                        }
                    ],
                },
            },
            "device_links": [
                {
                    "semantic": "internal.past_kv",
                    "producer": "vlm",
                    "consumer": "action_expert",
                    "transport": "device_pointer",
                    "owner": "producer",
                    "lifetime": "inference",
                },
                {
                    "semantic": "internal.prefix_pad_masks",
                    "producer": "vlm",
                    "consumer": "action_expert",
                    "transport": "device_pointer",
                    "owner": "producer",
                    "lifetime": "inference",
                },
            ],
        },
        deployment_name=deployment_name,
    )
    return RuntimeContext(load_inference_manifest(tmp_path, deployment_name), runtime_options=runtime_options or {})


def _act_acl(
    context: RuntimeContext,
    *,
    state_name: str = "state",
    state_dtype: str = "float32",
    state_shape: tuple[int, ...] = (1, 6),
) -> FakeAcl:
    model_path = str(context.resolved_artifacts["policy"])

    def execute(inputs):
        state, _image = inputs
        if state.shape != (1, 6) or state.dtype != np.dtype("float32"):
            return [np.zeros((1, 4, 6), dtype=np.float32)]
        return [np.repeat(state[:, None, :], 4, axis=1)]

    return FakeAcl(
        {
            model_path: FakeModelSpec(
                inputs=(
                    _tensor(state_name, state_dtype, state_shape),
                    _tensor("image", "float32", (1, 3, 16, 24)),
                ),
                outputs=(_tensor("action", "float32", (1, 4, 6)),),
                callback=execute,
            )
        }
    )


def _pi05_acl(
    context: RuntimeContext,
    observed_times: list[float],
    observed_noise: list[np.ndarray] | None = None,
    action_callback=None,
) -> FakeAcl:
    vlm_path = str(context.resolved_artifacts["vlm"])
    action_path = str(context.resolved_artifacts["action_expert"])

    def execute_vlm(inputs):
        _image, _tokens, masks, prefix = inputs
        assert prefix.shape == (1, 1, 8, 8)
        assert prefix[0, 0, 0, 0] == 0.0
        assert prefix[0, 0, -1, 0] < -1e30
        return [np.array([[3.0, 4.0]], dtype=np.float32), masks]

    def execute_action(inputs):
        past_kv, masks, time_value, noise = inputs
        np.testing.assert_array_equal(past_kv, np.array([[3.0, 4.0]], dtype=np.float32))
        np.testing.assert_array_equal(masks, np.array([[True, True, False, False]]))
        observed_times.append(float(time_value[0]))
        if observed_noise is not None:
            observed_noise.append(noise.copy())
        return [(action_callback(noise) if action_callback is not None else noise - 1.0)]

    return FakeAcl(
        {
            vlm_path: FakeModelSpec(
                inputs=(
                    _tensor("image", "float32", (1, 3, 16, 24)),
                    _tensor("tokens", "int64", (1, 4)),
                    _tensor("masks", "bool", (1, 4)),
                    _tensor("prefix_mask", "float32", (1, 1, 8, 8)),
                ),
                outputs=(
                    _tensor("past_kv", "float32", (1, 2)),
                    _tensor("prefix_pad_masks", "bool", (1, 4)),
                ),
                callback=execute_vlm,
            ),
            action_path: FakeModelSpec(
                inputs=(
                    _tensor("past_kv", "float32", (1, 2)),
                    _tensor("prefix_pad_masks", "bool", (1, 4)),
                    _tensor("time", "float32", (1,)),
                    _tensor("noise", "float32", (1, 2, 8)),
                ),
                outputs=(
                    _tensor(
                        context.deployment.bindings["action_expert"].outputs[0].runtime_name,
                        "float32",
                        (1, 2, 8),
                    ),
                ),
                callback=execute_action,
            ),
        }
    )


def _pure_engine_registry(monkeypatch, acl: FakeAcl) -> BackendRegistry:
    module = ModuleType("tests.fake_ascend_engine_backend")

    def fake_create_backend(context: RuntimeContext) -> AscendBackend:
        device_id = context.runtime_options.get("device_id", 0)
        return AscendBackend(int(device_id), runtime_manager=AclRuntimeManager(lambda: acl))

    module.create_backend = fake_create_backend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return BackendRegistry(
        {
            "ascend": BackendDescriptor(
                name="ascend",
                factory=f"{module.__name__}:create_backend",
                supported_policy_families=frozenset({"pi05"}),
                target_validator=lambda deployment: None,
            )
        }
    )


def test_ascend_act_pipeline_uses_manifest_bindings_and_matches_reference(tmp_path):
    context = _act_context(tmp_path, runtime_options={"device_id": 0})
    acl = _act_acl(context)
    backend = AscendBackend(0, runtime_manager=AclRuntimeManager(lambda: acl))
    pipeline = InferencePipeline("policy", context, backend, codec=create_policy_codec(context.policy))
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
    assert result.metadata["device_id"] == 0
    assert backend.health().state is BackendState.READY
    pipeline.close()
    pipeline.close()
    assert acl.init_calls == [None]
    assert acl.finalize_calls == 1
    assert acl.reset_device_calls == [0]
    assert acl.memory == {}


def test_ascend_pi05_keeps_device_links_internal_and_runs_denoising_loop(tmp_path):
    context = _pi05_context(tmp_path, runtime_options={"random_seed": 7})
    observed_times: list[float] = []
    acl = _pi05_acl(context, observed_times)
    backend = AscendBackend(0, runtime_manager=AclRuntimeManager(lambda: acl))
    pipeline = InferencePipeline("pi05", context, backend, codec=create_policy_codec(context.policy))
    pipeline.load()

    result = pipeline.infer(
        InferenceRequest(
            request_id="pi05",
            inputs={
                "observation.images.top": np.ones((3, 16, 24), dtype=np.float32),
                "observation.language.tokens": np.array([1, 2, 3, 4], dtype=np.int64),
                "observation.language.attention_mask": np.array([True, True, False, False]),
                "noise": np.zeros((1, 2, 8), dtype=np.float32),
            },
        )
    )

    np.testing.assert_array_equal(result.action, np.full((1, 2, 6), -2.0, dtype=np.float32))
    assert result.actual_chunk_size == 2
    assert observed_times == [1.0, 0.5]
    assert "denoising_schedule" not in result.metadata
    pipeline.close()
    assert acl.memory == {}
    assert acl.finalize_calls == 1


def test_ascend_pi05_integrates_uniform_velocity_schedule(tmp_path):
    schedule = {
        "format": "pi05-denoising-schedule-v1",
        "name": "uniform-two-step",
        "algorithm": "euler",
        "model_output": "velocity",
        "timesteps": [1.0, 0.5, 0.0],
    }
    context = _pi05_context(tmp_path, action_runtime_name="/Identity:0:velocity", schedule=schedule)
    observed_times: list[float] = []
    observed_noise: list[np.ndarray] = []
    acl = _pi05_acl(
        context,
        observed_times,
        observed_noise,
        action_callback=lambda noise: np.ones_like(noise),
    )
    backend = AscendBackend(0, runtime_manager=AclRuntimeManager(lambda: acl))
    pipeline = InferencePipeline("pi05", context, backend, codec=create_policy_codec(context.policy))
    pipeline.load()

    result = pipeline.infer(
        InferenceRequest(
            request_id="velocity-uniform",
            inputs={
                "observation.images.top": np.ones((3, 16, 24), dtype=np.float32),
                "observation.language.tokens": np.array([1, 2, 3, 4], dtype=np.int64),
                "observation.language.attention_mask": np.array([True, True, False, False]),
                "noise": np.zeros((1, 2, 8), dtype=np.float32),
            },
        )
    )

    np.testing.assert_array_equal(result.action, np.full((1, 2, 6), -1.0, dtype=np.float32))
    assert observed_times == [1.0, 0.5]
    np.testing.assert_array_equal(observed_noise[0], np.zeros((1, 2, 8), dtype=np.float32))
    np.testing.assert_array_equal(observed_noise[1], np.full((1, 2, 8), -0.5, dtype=np.float32))
    assert result.metadata["denoising_schedule"] == {
        "name": "uniform-two-step",
        "step_count": 2,
        "source": "artifacts/denoising_schedule.json",
    }
    pipeline.close()


def test_ascend_pi05_integrates_non_uniform_velocity_schedule_and_captures_diagnostics(tmp_path):
    schedule = {
        "format": "pi05-denoising-schedule-v1",
        "name": "non-uniform",
        "algorithm": "euler",
        "model_output": "velocity",
        "timesteps": [1.0, 0.8, 0.2, 0.0],
    }
    context = _pi05_context(tmp_path, action_runtime_name="v_t", schedule=schedule)
    observed_times: list[float] = []
    observed_noise: list[np.ndarray] = []
    captured: dict[str, np.ndarray] = {}
    acl = _pi05_acl(
        context,
        observed_times,
        observed_noise,
        action_callback=lambda noise: np.full_like(noise, 2.0),
    )
    backend = AscendBackend(
        0,
        runtime_manager=AclRuntimeManager(lambda: acl),
        diagnostic_capture=lambda name, value: captured.setdefault(name, np.asarray(value).copy()),
    )
    pipeline = InferencePipeline("pi05", context, backend, codec=create_policy_codec(context.policy))
    pipeline.load()

    result = pipeline.infer(
        InferenceRequest(
            request_id="velocity-non-uniform",
            inputs={
                "observation.images.top": np.ones((3, 16, 24), dtype=np.float32),
                "observation.language.tokens": np.array([1, 2, 3, 4], dtype=np.int64),
                "observation.language.attention_mask": np.array([True, True, False, False]),
                "noise": np.zeros((1, 2, 8), dtype=np.float32),
            },
        )
    )

    np.testing.assert_allclose(result.action, np.full((1, 2, 6), -2.0, dtype=np.float32))
    np.testing.assert_allclose(observed_times, [1.0, 0.8, 0.2])
    np.testing.assert_allclose(captured["timesteps"], [1.0, 0.8, 0.2, 0.0])
    np.testing.assert_allclose(captured["dt_step00"], -0.2)
    np.testing.assert_allclose(captured["dt_step01"], -0.6)
    np.testing.assert_allclose(captured["velocity_step01"], 2.0)
    np.testing.assert_allclose(captured["x_t_step01"], -1.6)
    pipeline.close()


def test_ascend_rejects_invalid_schedule_before_acl_initialization(tmp_path):
    invalid_schedule = {
        "format": "pi05-denoising-schedule-v1",
        "name": "invalid",
        "algorithm": "euler",
        "model_output": "velocity",
        "timesteps": [1.0, 0.5, 0.6, 0.0],
    }
    context = _pi05_context(tmp_path, action_runtime_name="velocity", schedule=invalid_schedule)
    acl = _pi05_acl(context, [])
    backend = AscendBackend(0, runtime_manager=AclRuntimeManager(lambda: acl))

    with pytest.raises(BackendLoadError) as error:
        backend.load(context)

    assert error.value.code == "invalid_denoising_schedule"
    assert acl.init_calls == []
    backend.close()


def test_ascend_diagnostic_schedule_takes_precedence_and_is_reported(tmp_path):
    manifest_schedule = {
        "format": "pi05-denoising-schedule-v1",
        "name": "manifest",
        "algorithm": "euler",
        "model_output": "velocity",
        "timesteps": [1.0, 0.5, 0.0],
    }
    override_path = tmp_path / "override.json"
    override_path.write_text(
        json.dumps({**manifest_schedule, "name": "override", "timesteps": [1.0, 0.75, 0.25, 0.0]}),
        encoding="utf-8",
    )
    context = _pi05_context(tmp_path, action_runtime_name="velocity", schedule=manifest_schedule)
    observed_times: list[float] = []
    acl = _pi05_acl(context, observed_times, action_callback=lambda noise: np.ones_like(noise))
    backend = AscendBackend(
        0,
        runtime_manager=AclRuntimeManager(lambda: acl),
        diagnostic_schedule=load_pi05_schedule(override_path),
        diagnostic_schedule_source=str(override_path.resolve()),
    )
    pipeline = InferencePipeline("override", context, backend, codec=create_policy_codec(context.policy))
    pipeline.load()

    result = pipeline.infer(
        InferenceRequest(
            request_id="override",
            inputs={
                "observation.images.top": np.ones((3, 16, 24), dtype=np.float32),
                "observation.language.tokens": np.array([1, 2, 3, 4], dtype=np.int64),
                "observation.language.attention_mask": np.array([True, True, False, False]),
                "noise": np.zeros((1, 2, 8), dtype=np.float32),
            },
        )
    )

    assert observed_times == [1.0, 0.75, 0.25]
    assert result.metadata["denoising_schedule"] == {
        "name": "override",
        "step_count": 3,
        "source": str(override_path.resolve()),
        "override": True,
    }
    pipeline.close()


def test_ascend_rejects_schedule_diagnostics_for_legacy_action_output(tmp_path):
    override_path = tmp_path / "override.json"
    override_path.write_text(
        json.dumps(
            {
                "format": "pi05-denoising-schedule-v1",
                "name": "override",
                "algorithm": "euler",
                "model_output": "velocity",
                "timesteps": [1.0, 0.0],
            }
        ),
        encoding="utf-8",
    )
    context = _pi05_context(tmp_path)
    acl = _pi05_acl(context, [])
    backend = AscendBackend(
        0,
        runtime_manager=AclRuntimeManager(lambda: acl),
        diagnostic_schedule=load_pi05_schedule(override_path),
        diagnostic_schedule_source=str(override_path),
    )

    with pytest.raises(BackendLoadError) as error:
        backend.load(context)

    assert error.value.code == "invalid_runtime_options"
    assert acl.init_calls == []
    backend.close()


def test_ascend_curvature_log_records_strict_schedule_and_adjacent_velocity_scores(tmp_path):
    schedule = {
        "format": "pi05-denoising-schedule-v1",
        "name": "dense",
        "algorithm": "euler",
        "model_output": "velocity",
        "timesteps": [1.0, 0.75, 0.25, 0.0],
    }
    curvature_path = tmp_path / "diagnostics" / "curvature.jsonl"
    context = _pi05_context(
        tmp_path,
        action_runtime_name="v_t",
        schedule=schedule,
        runtime_options={"curvature_log_path": str(curvature_path)},
    )
    velocities = iter((1.0, 2.0, 4.0))
    acl = _pi05_acl(
        context,
        [],
        action_callback=lambda noise: np.full_like(noise, next(velocities)),
    )
    backend = AscendBackend(0, runtime_manager=AclRuntimeManager(lambda: acl))
    pipeline = InferencePipeline("curvature", context, backend, codec=create_policy_codec(context.policy))
    pipeline.load()

    pipeline.infer(
        InferenceRequest(
            request_id="curvature",
            inputs={
                "observation.images.top": np.ones((3, 16, 24), dtype=np.float32),
                "observation.language.tokens": np.array([1, 2, 3, 4], dtype=np.int64),
                "observation.language.attention_mask": np.array([True, True, False, False]),
                "noise": np.zeros((1, 2, 8), dtype=np.float32),
            },
        )
    )
    pipeline.close()

    record = json.loads(curvature_path.read_text(encoding="utf-8"))
    assert record["schedule"] == schedule
    np.testing.assert_allclose(record["curvature_scores"], [1.0, 1.0, 1.0], rtol=1e-5)


@pytest.mark.parametrize("name", ["curvature_log_path"])
def test_ascend_runtime_diagnostic_paths_must_be_nonempty(name):
    with pytest.raises(BackendLoadError, match="non-empty"):
        AscendBackend._validate_runtime_options({name: "   "})


def test_ascend_rejects_schedule_override_runtime_option():
    with pytest.raises(BackendLoadError, match="unknown Ascend runtime options"):
        AscendBackend._validate_runtime_options({"schedule_override_path": "/tmp/schedule.json"})


def test_pure_engine_runs_named_pi05_ascend_deployment_end_to_end_with_fake_acl(monkeypatch, tmp_path):
    deployment_name = "warehouse-arm-om"
    context = _pi05_context(
        tmp_path,
        deployment_name=deployment_name,
        runtime_options={"device_id": 3, "random_seed": 99},
    )
    observed_times: list[float] = []
    observed_noise: list[np.ndarray] = []
    acl = _pi05_acl(context, observed_times, observed_noise)

    def create_processor_views():
        return (lambda inputs: inputs), (lambda action: np.asarray(action) + np.float32(10.0))

    monkeypatch.setattr(pipeline_factory, "create_lerobot_processor_views", create_processor_views)
    engine = PureInferenceEngine(
        tmp_path,
        deployment_name,
        pipeline_id="named-pi05",
        runtime_options={"device_id": 3, "random_seed": 99},
        registry=_pure_engine_registry(monkeypatch, acl),
    )
    external_noise = np.full((1, 2, 8), 5.0, dtype=np.float32)

    result = engine(
        {
            "observation.images.top": np.ones((3, 16, 24), dtype=np.float32),
            "observation.language.tokens": np.array([1, 2, 3, 4], dtype=np.int64),
            "observation.language.attention_mask": np.array([True, True, False, False]),
        },
        request_id="pure-pi05",
        control_inputs={"noise": external_noise},
        capture_raw_action=True,
    )

    assert result.policy_type == "pi05"
    assert result.backend_type == "ascend"
    assert result.chunk_size == 2
    assert engine.chunk_size == 2
    assert result.shape == (1, 2, 6)
    np.testing.assert_array_equal(result.raw_action, np.full((1, 2, 6), 3.0, dtype=np.float32))
    np.testing.assert_array_equal(result.action, np.full((1, 2, 6), 13.0, dtype=np.float32))
    assert observed_times == [1.0, 0.5]
    np.testing.assert_array_equal(observed_noise[0], external_noise)
    np.testing.assert_array_equal(observed_noise[1], external_noise - 1.0)

    vlm_path = str(context.resolved_artifacts["vlm"])
    action_path = str(context.resolved_artifacts["action_expert"])
    vlm_executions = [execution for execution in acl.executions if execution[0] == vlm_path]
    action_executions = [execution for execution in acl.executions if execution[0] == action_path]
    assert len(vlm_executions) == 1
    assert len(action_executions) == 2
    assert action_executions[0][1][:2] == vlm_executions[0][2]
    assert action_executions[1][1][:2] == vlm_executions[0][2]

    engine.close()
    engine.close()
    assert acl.init_calls == [None]
    assert acl.set_device_calls == [3]
    assert acl.reset_device_calls == [3]
    assert acl.finalize_calls == 1
    assert acl.loaded_models == {}
    assert acl.model_paths == {}
    assert acl.contexts == set()
    assert acl.memory == {}


def test_ascend_rejects_input_sourced_device_link_before_acl_initialization(tmp_path):
    context = _pi05_context(tmp_path)
    deployment = context.deployment.model_copy(
        update={
            "device_links": (
                DeviceLink(
                    semantic="internal.past_kv",
                    producer="vlm",
                    consumer="action_expert",
                    producer_binding="input",
                    transport="device_pointer",
                    owner="producer",
                ),
            )
        }
    )
    invalid_context = RuntimeContext(replace(context.validated_manifest, deployment=deployment))
    acl = _pi05_acl(context, [])
    backend = AscendBackend(0, runtime_manager=AclRuntimeManager(lambda: acl))

    with pytest.raises(BackendLoadError) as error:
        backend.load(invalid_context)

    assert error.value.code == "unsupported_device_link_source"
    assert acl.init_calls == []
    backend.close()


def test_ascend_multiple_instances_share_acl_lifecycle_and_serialize_device(tmp_path):
    first_context = _act_context(tmp_path / "first")
    second_context = _act_context(tmp_path / "second")
    first_path = str(first_context.resolved_artifacts["policy"])
    second_path = str(second_context.resolved_artifacts["policy"])

    def execute(inputs):
        state, _image = inputs
        return [np.repeat(state[:, None, :], 4, axis=1)]

    spec = FakeModelSpec(
        inputs=(_tensor("state", "float32", (1, 6)), _tensor("image", "float32", (1, 3, 16, 24))),
        outputs=(_tensor("action", "float32", (1, 4, 6)),),
        callback=execute,
    )
    acl = FakeAcl({first_path: spec, second_path: spec})
    acl.execution_delay = 0.05
    manager = AclRuntimeManager(lambda: acl)
    first = AscendBackend(0, runtime_manager=manager)
    second = AscendBackend(0, runtime_manager=manager)
    first.load(first_context)
    second.load(second_context)
    errors: list[Exception] = []

    def infer(backend, context):
        try:
            deployment = context.deployment
            plan = build_execution_plan(deployment.execution, deployment.bindings, deployment.device_links)
            codec = create_policy_codec(context.policy)
            bound = codec.encode_inputs(
                request=CodecRequest(
                    {
                        "observation.state": np.zeros((1, 6), dtype=np.float32),
                        "observation.images.top": np.zeros((1, 3, 16, 24), dtype=np.float32),
                    }
                ),
                bindings=deployment.bindings["policy"],
            )
            backend.infer(
                InferenceRequest(
                    request_id="shared",
                    inputs={"execution_plan": plan, "role_inputs": {"policy": bound}},
                )
            )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first_thread = threading.Thread(target=infer, args=(first, first_context))
    second_thread = threading.Thread(target=infer, args=(second, second_context))
    first_thread.start()
    second_thread.start()
    first_thread.join(timeout=2.0)
    second_thread.join(timeout=2.0)

    assert errors == []
    assert acl.max_active_executions == 1
    assert acl.init_calls == [None]
    assert acl.set_device_calls == [0]
    first.close()
    assert acl.finalize_calls == 0
    assert acl.reset_device_calls == []
    second.close()
    assert acl.finalize_calls == 1
    assert acl.reset_device_calls == [0]


def test_ascend_partial_load_failure_rolls_back_models_context_device_and_acl(tmp_path):
    context = _pi05_context(tmp_path)
    acl = _pi05_acl(context, [])
    acl.fail_model_paths.add(str(context.resolved_artifacts["action_expert"]))
    backend = AscendBackend(0, runtime_manager=AclRuntimeManager(lambda: acl))

    with pytest.raises(BackendLoadError, match="load_from_file"):
        backend.load(context)

    assert backend.health().state is BackendState.FAILED
    assert len(acl.unloaded_models) == 1
    assert acl.destroyed_contexts
    assert acl.reset_device_calls == [0]
    assert acl.finalize_calls == 1
    assert acl.memory == {}
    backend.close()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("chunk_size", 0),
        ("max_action_dim", True),
        ("num_inference_steps", "2"),
    ],
)
def test_ascend_rejects_invalid_pi05_policy_config_without_leaks(tmp_path, key, value):
    context = _pi05_context(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[key] = value
    config_path.write_text(json.dumps(config), encoding="utf-8")
    acl = _pi05_acl(context, [])
    backend = AscendBackend(0, runtime_manager=AclRuntimeManager(lambda: acl))

    with pytest.raises(BackendLoadError) as error:
        backend.load(context)

    assert error.value.code == "invalid_policy_config"
    assert backend.health().state is BackendState.FAILED
    assert acl.loaded_models == {}
    assert acl.contexts == set()
    assert acl.memory == {}
    assert acl.reset_device_calls == [0]
    assert acl.finalize_calls == 1
    backend.close()


def test_ascend_runtime_manager_rejects_conflicting_acl_config_paths():
    acl = FakeAcl({})
    manager = AclRuntimeManager(lambda: acl)
    first = manager.acquire(0, "first.json")

    with pytest.raises(BackendLoadError) as error:
        manager.acquire(0, "second.json")

    assert error.value.code == "acl_config_conflict"
    assert acl.init_calls == ["first.json"]
    first.close()
    assert acl.finalize_calls == 1


@pytest.mark.parametrize(
    ("acl_options", "code"),
    [
        ({"state_name": "wrong_state"}, "runtime_name_mismatch"),
        ({"state_dtype": "float16"}, "runtime_dtype_mismatch"),
        ({"state_shape": (1, 7)}, "runtime_shape_mismatch"),
    ],
)
def test_ascend_rejects_runtime_descriptor_mismatch_before_dataset_allocation(tmp_path, acl_options, code):
    context = _act_context(tmp_path)
    acl = _act_acl(context, **acl_options)
    backend = AscendBackend(0, runtime_manager=AclRuntimeManager(lambda: acl))

    with pytest.raises(BackendLoadError) as error:
        backend.load(context)

    assert error.value.code == code
    assert acl.memory == {}
    assert acl.finalize_calls == 1
    backend.close()


def test_ascend_registry_factory_is_lazy_and_reset_is_unsupported(tmp_path):
    context = _act_context(tmp_path)
    backend = BACKEND_REGISTRY.create(context)

    assert isinstance(backend, AscendBackend)
    with pytest.raises(BackendCapabilityError):
        backend.reset()
    backend.close()


@pytest.mark.parametrize(
    "runtime_options",
    [
        {"device_id": -1},
        {"device_id": "0"},
        {"acl_config_path": ""},
        {"random_seed": 1.5},
        {"unknown": True},
    ],
)
def test_ascend_rejects_invalid_runtime_options(tmp_path, runtime_options):
    context = _act_context(tmp_path, runtime_options=runtime_options)

    with pytest.raises(BackendLoadError) as error:
        create_backend(context)

    assert error.value.code == "invalid_runtime_options"


_GRASPGEN_ROLE_SHAPES = {
    "generator_sa1": ((1, 3, 256, 64), (1, 64, 256)),
    "generator_sa2": ((1, 131, 64, 128), (1, 128, 64)),
    "generator_encoder_head": ((1, 259, 1, 64), (1, 512)),
    "discriminator_sa1": ((1, 3, 256, 64), (1, 64, 256)),
    "discriminator_sa2": ((1, 131, 64, 128), (1, 128, 64)),
    "discriminator_encoder_head": ((1, 259, 1, 64), (1, 512)),
    "denoiser": ((1, 512), (1000, 6), (1,), (1000, 6)),
    "discriminator_head": ((1, 512), (1000, 6)),
}


def _graspgen_input_bindings(role: str, geometry: dict):
    npoints = geometry["npoints"]
    nsamples = geometry["nsamples"]
    if role == "generator_sa1" or role == "discriminator_sa1":
        return [
            {
                "semantic": "observation.object_points",
                "runtime_name": "grouped_features",
                "index": 0,
                "dtype": "float32",
                "shape": [1, 3, npoints[0], nsamples[0]],
                "layout": "NCHW",
            }
        ]
    if role == "generator_sa2" or role == "discriminator_sa2":
        return [
            {
                "semantic": "stage2_features",
                "runtime_name": "grouped_features",
                "index": 0,
                "dtype": "float32",
                "shape": [1, 131, npoints[1], nsamples[1]],
                "layout": "NCHW",
            }
        ]
    if role == "generator_encoder_head" or role == "discriminator_encoder_head":
        return [
            {
                "semantic": "global_features",
                "runtime_name": "grouped_features",
                "index": 0,
                "dtype": "float32",
                "shape": [1, 259, 1, npoints[1]],
                "layout": "NCHW",
            }
        ]
    if role == "denoiser":
        return [
            {
                "semantic": "generator_embedding",
                "runtime_name": "object_embedding",
                "index": 0,
                "dtype": "float32",
                "shape": [1, 512],
            },
            {
                "semantic": "diffusion.sample",
                "runtime_name": "sample",
                "index": 1,
                "dtype": "float32",
                "shape": [1000, 6],
            },
            {"semantic": "diffusion.time", "runtime_name": "timestep", "index": 2, "dtype": "float32", "shape": [1]},
        ]
    if role == "discriminator_head":
        return [
            {
                "semantic": "discriminator_embedding",
                "runtime_name": "object_embedding",
                "index": 0,
                "dtype": "float32",
                "shape": [1, 512],
            },
            {"semantic": "sample_rt", "runtime_name": "grasp_rt", "index": 1, "dtype": "float32", "shape": [1000, 6]},
        ]
    raise ValueError(f"unknown graspgen role {role!r}")


def _graspgen_output_bindings(role: str, geometry: dict):
    npoints = geometry["npoints"]
    if role in {"generator_sa1", "discriminator_sa1"}:
        return [
            {
                "semantic": "features",
                "runtime_name": "PartitionedCall_/ReduceMax_ReduceMax_1:0:features",
                "index": 0,
                "dtype": "float32",
                "shape": [1, 128, npoints[0]],
            }
        ]
    if role in {"generator_sa2", "discriminator_sa2"}:
        return [
            {
                "semantic": "features",
                "runtime_name": "PartitionedCall_/ReduceMax_ReduceMax_1:0:features",
                "index": 0,
                "dtype": "float32",
                "shape": [1, 256, npoints[1]],
            }
        ]
    if role in {"generator_encoder_head", "discriminator_encoder_head"}:
        return [
            {
                "semantic": "object_embedding",
                "runtime_name": "PartitionedCall_/prediction_head/Gemm:0:object_embedding",
                "index": 0,
                "dtype": "float32",
                "shape": [1, 512],
            }
        ]
    if role == "denoiser":
        return [
            {
                "semantic": "predicted_noise",
                "runtime_name": "PartitionedCall_/prediction_head/Gemm:0:predicted_noise",
                "index": 0,
                "dtype": "float32",
                "shape": [1000, 6],
            }
        ]
    if role == "discriminator_head":
        return [
            {
                "semantic": "grasp.poses",
                "runtime_name": "logits",
                "index": 0,
                "dtype": "float32",
                "shape": [1000, 4, 4],
            },
            {
                "semantic": "grasp.confidence",
                "runtime_name": "confidence",
                "index": 1,
                "dtype": "float32",
                "shape": [1000],
            },
        ]
    raise ValueError(f"unknown graspgen role {role!r}")


def _graspgen_context(tmp_path: Path, *, runtime_options=None, device_link_embeddings: bool = False) -> RuntimeContext:
    tmp_path.mkdir(parents=True, exist_ok=True)
    bundle_paths = create_policy_bundle(tmp_path, "graspgen", include_weights=False)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "kappa": 2.02217,
            "diffusion_steps": 2,
            "grasp_batch_size": 4,
            "point_count": 128,
            "geometry": {"npoints": [16, 8], "radii": [0.05, 0.10], "nsamples": [4, 8]},
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    artifacts: dict[str, object] = {}
    for role in _GRASPGEN_ROLE_SHAPES:
        artifact_path = tmp_path / "artifacts" / f"{role}.om"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(f"{role}-om".encode())
        artifacts[role] = {"path": f"artifacts/{role}.om", "format": "om"}
    execution = list(_GRASPGEN_ROLE_SHAPES.keys())
    geometry = config["geometry"]
    bindings = {
        role: {"inputs": _graspgen_input_bindings(role, geometry), "outputs": _graspgen_output_bindings(role, geometry)}
        for role in execution
    }
    device_links = []
    if device_link_embeddings:
        generator_semantic = "internal.graspgen.generator_embedding"
        discriminator_semantic = "internal.graspgen.discriminator_embedding"
        bindings["generator_encoder_head"]["outputs"][0]["semantic"] = generator_semantic
        bindings["denoiser"]["inputs"][0]["semantic"] = generator_semantic
        bindings["discriminator_encoder_head"]["outputs"][0]["semantic"] = discriminator_semantic
        bindings["discriminator_head"]["inputs"][0]["semantic"] = discriminator_semantic
        device_links = [
            {
                "semantic": generator_semantic,
                "producer": "generator_encoder_head",
                "consumer": "denoiser",
                "transport": "device_pointer",
                "owner": "producer",
            },
            {
                "semantic": discriminator_semantic,
                "producer": "discriminator_encoder_head",
                "consumer": "discriminator_head",
                "transport": "device_pointer",
                "owner": "producer",
            },
        ]
    _write_compiled_manifest(
        tmp_path,
        bundle_paths,
        {
            "backend": "ascend",
            "target": {"soc": "Ascend310P3", "runtime": "acl"},
            "artifacts": artifacts,
            "execution": execution,
            "bindings": bindings,
            "device_links": device_links,
        },
    )
    return RuntimeContext(
        validated_manifest=load_inference_manifest(tmp_path, "ascend"),
        runtime_options=runtime_options or {"device_id": 0, "random_seed": 0},
    )


def _graspgen_acl(context: RuntimeContext) -> FakeAcl:
    specs: dict[str, FakeModelSpec] = {}
    deployment = context.deployment

    def make_callback(role: str):
        def callback(inputs):
            outputs = []
            for binding in deployment.bindings[role].outputs:
                outputs.append(np.full(binding.shape, 0.5, dtype=np.float32))
            return outputs

        return callback

    for role in deployment.execution:
        path = str(context.resolved_artifacts[role])
        inputs = tuple(
            _tensor(b.runtime_name or f"input_{b.index}", b.dtype, tuple(b.shape))
            for b in deployment.bindings[role].inputs
        )
        outputs = tuple(
            _tensor(b.runtime_name or f"output_{b.index}", b.dtype, tuple(b.shape))
            for b in deployment.bindings[role].outputs
        )
        specs[path] = FakeModelSpec(inputs=inputs, outputs=outputs, callback=make_callback(role))
    return FakeAcl(specs)


def test_ascend_graspgen_loads_eight_role_deployment_and_runs_inference(tmp_path):
    context = _graspgen_context(tmp_path, runtime_options={"random_seed": 7})
    acl = _graspgen_acl(context)
    backend = AscendBackend(0, runtime_manager=AclRuntimeManager(lambda: acl))
    backend.load(context)

    assert backend.health().state is BackendState.READY
    assert set(backend._models.keys()) == set(_GRASPGEN_ROLE_SHAPES.keys())

    deployment = context.deployment
    plan = build_execution_plan(deployment.execution, deployment.bindings, deployment.device_links)
    from inference_service.codecs import BoundInputs, BoundTensor

    points = np.random.randn(200, 3).astype(np.float32)
    first_role = deployment.execution[0]
    role_inputs = {role: BoundInputs(tensors=()) for role in deployment.execution}
    role_inputs[first_role] = BoundInputs(
        tensors=(BoundTensor(semantic="observation.object_points", runtime_name=None, index=0, value=points),)
    )
    result = backend.infer(
        InferenceRequest(
            request_id="graspgen-1",
            inputs={"execution_plan": plan, "role_inputs": role_inputs},
        )
    )

    assert result.actual_chunk_size > 0
    assert result.backend_latency_ms >= 0.0
    assert result.metadata["graspgen_denoiser_steps"] == 2
    timing_ms = result.metadata["graspgen_timing_ms"]
    assert isinstance(timing_ms, Mapping)
    assert set(timing_ms) == {
        "input_prepare",
        "pointnet_geometry",
        "generator_sa1",
        "generator_encoder_host",
        "generator_sa2",
        "generator_encoder_head",
        "discriminator_sa1",
        "discriminator_encoder_host",
        "discriminator_sa2",
        "discriminator_encoder_head",
        "denoiser_setup",
        "denoiser_execute",
        "denoiser_host",
        "pose_conversion",
        "discriminator_head",
        "output_finalize",
        "total",
    }
    assert all(value >= 0.0 for value in timing_ms.values())
    backend.close()
    assert acl.memory == {}


def test_ascend_graspgen_keeps_embeddings_on_device_and_skips_unused_logits(tmp_path):
    context = _graspgen_context(
        tmp_path,
        runtime_options={"random_seed": 7},
        device_link_embeddings=True,
    )
    acl = _graspgen_acl(context)
    backend = AscendBackend(0, runtime_manager=AclRuntimeManager(lambda: acl))
    backend.load(context)

    deployment = context.deployment
    plan = build_execution_plan(deployment.execution, deployment.bindings, deployment.device_links)
    from inference_service.codecs import BoundInputs, BoundTensor

    role_inputs = {role: BoundInputs(tensors=()) for role in deployment.execution}
    role_inputs[deployment.execution[0]] = BoundInputs(
        tensors=(
            BoundTensor(
                semantic="observation.object_points",
                runtime_name=None,
                index=0,
                value=np.random.randn(200, 3).astype(np.float32),
            ),
        )
    )
    result = backend.infer(
        InferenceRequest(
            request_id="graspgen-device-links",
            inputs={"execution_plan": plan, "role_inputs": role_inputs},
        )
    )

    executions_by_role = {
        Path(path).stem: (inputs, outputs) for path, inputs, outputs in acl.executions if Path(path).stem != "denoiser"
    }
    denoiser_executions = [execution for execution in acl.executions if Path(execution[0]).stem == "denoiser"]
    assert denoiser_executions
    assert executions_by_role["generator_encoder_head"][1][0] == denoiser_executions[0][1][0]
    assert executions_by_role["discriminator_encoder_head"][1][0] == (executions_by_role["discriminator_head"][0][0])
    assert backend._models["generator_encoder_head"].output_host_buffers == [None]
    assert backend._models["discriminator_encoder_head"].output_host_buffers == [None]
    assert backend._models["discriminator_head"].output_host_buffers[0] is None
    assert backend._models["discriminator_head"].output_host_buffers[1] is not None
    assert result.actual_chunk_size > 0

    backend.close()
    assert acl.memory == {}
