from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from inference_manifest import BundleFile, canonical_bundle_digest, load_inference_manifest, sha256_file
from inference_manifest.models import DeviceLink
from inference_service.backends import (
    BACKEND_REGISTRY,
    BackendCapabilityError,
    BackendLoadError,
    BackendState,
    InferenceRequest,
    RuntimeContext,
)
from inference_service.backends.ascend import AscendBackend, create_backend
from inference_service.backends.ascend.acl_runtime import AclRuntimeManager
from inference_service.codecs import CodecRequest, build_execution_plan, create_policy_codec
from inference_service.pipeline import InferencePipeline
from tests.manifest_fixtures import create_policy_bundle, write_manifest

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
        del kind
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
    return [BundleFile(path=path, sha256=sha256_file(root / path)) for path in paths]


def _write_compiled_manifest(root: Path, bundle_paths: tuple[str, ...], deployment: dict) -> None:
    entries = _bundle_entries(root, bundle_paths)
    write_manifest(
        root,
        {
            "schema_version": 1,
            "bundle": {
                "name": "ascend-test",
                "files": [entry.model_dump(mode="json") for entry in entries],
                "digest": {"algorithm": "sha256", "value": canonical_bundle_digest(entries)},
            },
            "deployments": {"ascend": deployment},
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
            "artifacts": {"policy": {"path": "artifacts/policy.om", "format": "om", "sha256": sha256_file(model)}},
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


def _pi05_context(tmp_path: Path, *, runtime_options=None) -> RuntimeContext:
    tmp_path.mkdir(parents=True, exist_ok=True)
    bundle_paths = create_policy_bundle(tmp_path, "pi05", include_weights=False)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({"chunk_size": 2, "max_action_dim": 8, "num_inference_steps": 2})
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
    _write_compiled_manifest(
        tmp_path,
        bundle_paths,
        {
            "backend": "ascend",
            "target": {"soc": "ascend310", "runtime": "acl"},
            "artifacts": {
                "vlm": {"path": "artifacts/vlm.om", "format": "om", "sha256": sha256_file(vlm)},
                "action_expert": {
                    "path": "artifacts/action_expert.om",
                    "format": "om",
                    "sha256": sha256_file(action_expert),
                },
            },
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
                            "runtime_name": "action",
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
    )
    return RuntimeContext(load_inference_manifest(tmp_path, "ascend"), runtime_options=runtime_options or {})


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


def _pi05_acl(context: RuntimeContext, observed_times: list[float]) -> FakeAcl:
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
        return [noise - 1.0]

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
                outputs=(_tensor("action", "float32", (1, 2, 8)),),
                callback=execute_action,
            ),
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
    pipeline.close()
    assert acl.memory == {}
    assert acl.finalize_calls == 1


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
