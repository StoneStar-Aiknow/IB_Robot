"""Manifest-validated OM model resources and execution through Ascend ACL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inference_manifest import ArtifactBindings, TensorBinding
from inference_service.backends.ascend.acl_runtime import AclRuntimeLease, check_acl_ret
from inference_service.backends.errors import BackendInferenceError, BackendLoadError
from inference_service.codecs import BoundInputs

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

_ACL_DTYPES = {
    0: np.dtype("float32"),
    1: np.dtype("float16"),
    2: np.dtype("int8"),
    3: np.dtype("int32"),
    4: np.dtype("uint8"),
    6: np.dtype("int16"),
    9: np.dtype("int64"),
    11: np.dtype("float64"),
    12: np.dtype("bool"),
}


def numpy_dtype(dtype: str) -> np.dtype:
    if dtype != "bfloat16":
        return np.dtype(dtype)
    try:
        return np.dtype(dtype)
    except TypeError:
        try:
            import ml_dtypes
        except ImportError as exc:
            raise BackendLoadError(
                "Ascend bfloat16 bindings require NumPy bfloat16 support or ml_dtypes",
                code="unsupported_runtime_dtype",
            ) from exc
        return np.dtype(ml_dtypes.bfloat16)


def _result_value(value: object, operation: str) -> object:
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], int):
        result, ret = value
        check_acl_ret(ret, operation)
        return result
    return value


def _binding_by_index(bindings: tuple[TensorBinding, ...], direction: str) -> dict[int, TensorBinding]:
    result: dict[int, TensorBinding] = {}
    for binding in bindings:
        if binding.index is None:
            raise BackendLoadError(
                f"Ascend {direction} binding {binding.semantic!r} requires an explicit runtime index",
                code=f"invalid_{direction}_bindings",
            )
        result[int(binding.index)] = binding
    return result


@dataclass(frozen=True)
class AclTensorDescriptor:
    index: int
    name: str | None
    dtype: np.dtype | None
    shape: tuple[int, ...] | None
    size: int


@dataclass(frozen=True)
class AclDeviceBuffer:
    pointer: object
    size: int


@dataclass
class _DatasetBuffer:
    pointer: object
    data_buffer: object
    size: int
    owned: bool


class AclModel:
    """One loaded OM model with deterministic datasets, buffers, and host staging."""

    def __init__(self, lease: AclRuntimeLease, role: str, path: Path, bindings: ArtifactBindings) -> None:
        self._lease = lease
        self._acl = lease.acl
        self.role = role
        self.path = path
        self.bindings = bindings
        self.model_id: object | None = None
        self.model_desc: object | None = None
        self.input_descriptors: tuple[AclTensorDescriptor, ...] = ()
        self.output_descriptors: tuple[AclTensorDescriptor, ...] = ()
        self.input_dataset: object | None = None
        self.output_dataset: object | None = None
        self.input_buffers: list[_DatasetBuffer] = []
        self.output_buffers: list[_DatasetBuffer] = []
        self.output_host_buffers: list[object] = []
        self._closed = False

    def load_descriptor(self) -> None:
        try:
            self._lease.bind_current_thread()
            self.model_id, ret = self._acl.mdl.load_from_file(str(self.path))
            check_acl_ret(ret, f"acl.mdl.load_from_file({self.role})")
            self.model_desc = self._acl.mdl.create_desc()
            if self.model_desc is None:
                raise RuntimeError(f"acl.mdl.create_desc({self.role}) returned no descriptor")
            check_acl_ret(
                self._acl.mdl.get_desc(self.model_desc, self.model_id),
                f"acl.mdl.get_desc({self.role})",
            )
            self.input_descriptors = self._describe("input")
            self.output_descriptors = self._describe("output")
            self._validate_bindings()
        except Exception:
            self.close()
            raise

    def prepare_datasets(
        self,
        *,
        input_overrides: dict[int, AclDeviceBuffer] | None = None,
        output_overrides: dict[int, AclDeviceBuffer] | None = None,
    ) -> None:
        try:
            self.input_dataset, self.input_buffers = self._create_dataset(self.input_descriptors, input_overrides or {})
            self.output_dataset, self.output_buffers = self._create_dataset(
                self.output_descriptors, output_overrides or {}
            )
            for descriptor in self.output_descriptors:
                host_buffer, ret = self._acl.rt.malloc_host(descriptor.size)
                check_acl_ret(ret, f"acl.rt.malloc_host({self.role} output {descriptor.index})")
                self.output_host_buffers.append(host_buffer)
        except Exception:
            self.close()
            raise

    def execute(
        self, inputs: BoundInputs | dict[int, np.ndarray], *, read_outputs: set[int] | None = None
    ) -> dict[int, np.ndarray]:
        if self.input_dataset is None or self.output_dataset is None or self.model_id is None:
            raise BackendInferenceError(f"Ascend role {self.role!r} is not fully loaded", code="runtime_not_loaded")
        self._lease.bind_current_thread()
        values = self._indexed_inputs(inputs)
        for descriptor, buffer in zip(self.input_descriptors, self.input_buffers, strict=True):
            if not buffer.owned:
                continue
            try:
                value = values[descriptor.index]
            except KeyError as exc:
                raise BackendInferenceError(
                    f"Ascend role {self.role!r} is missing runtime input index {descriptor.index}",
                    code="missing_runtime_input",
                ) from exc
            payload = np.ascontiguousarray(value).tobytes()
            if len(payload) != buffer.size:
                raise BackendInferenceError(
                    f"Ascend role {self.role!r} input {descriptor.index} has {len(payload)} bytes, "
                    f"runtime requires {buffer.size}",
                    code="input_size_mismatch",
                )
            source = self._acl.util.bytes_to_ptr(payload)
            check_acl_ret(
                self._acl.rt.memcpy(buffer.pointer, buffer.size, source, len(payload), ACL_MEMCPY_HOST_TO_DEVICE),
                f"acl.rt.memcpy H2D({self.role} input {descriptor.index})",
            )

        check_acl_ret(
            self._acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset),
            f"acl.mdl.execute({self.role})",
        )
        selected = read_outputs if read_outputs is not None else set(range(len(self.output_descriptors)))
        outputs: dict[int, np.ndarray] = {}
        for descriptor, buffer, host_buffer in zip(
            self.output_descriptors, self.output_buffers, self.output_host_buffers, strict=True
        ):
            if descriptor.index not in selected:
                continue
            check_acl_ret(
                self._acl.rt.memcpy(host_buffer, buffer.size, buffer.pointer, buffer.size, ACL_MEMCPY_DEVICE_TO_HOST),
                f"acl.rt.memcpy D2H({self.role} output {descriptor.index})",
            )
            payload = self._acl.util.ptr_to_bytes(host_buffer, buffer.size)
            dtype = descriptor.dtype or np.dtype("float32")
            value = np.frombuffer(payload, dtype=dtype).copy()
            if descriptor.shape is not None and all(dimension > 0 for dimension in descriptor.shape):
                value = value.reshape(descriptor.shape)
            outputs[descriptor.index] = value
        return outputs

    def output_buffer(self, index: int) -> AclDeviceBuffer:
        try:
            buffer = self.output_buffers[index]
        except IndexError as exc:
            raise BackendLoadError(
                f"Ascend role {self.role!r} has no output buffer at index {index}",
                code="invalid_device_link",
            ) from exc
        return AclDeviceBuffer(pointer=buffer.pointer, size=buffer.size)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        acl = self._acl
        self._lease.bind_current_thread()
        for host_buffer in reversed(self.output_host_buffers):
            acl.rt.free_host(host_buffer)
        self.output_host_buffers.clear()
        self._destroy_dataset(self.output_dataset, self.output_buffers)
        self._destroy_dataset(self.input_dataset, self.input_buffers)
        self.output_dataset = None
        self.input_dataset = None
        if self.model_desc is not None:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None
        if self.model_id is not None:
            acl.mdl.unload(self.model_id)
            self.model_id = None

    def _describe(self, direction: str) -> tuple[AclTensorDescriptor, ...]:
        mdl = self._acl.mdl
        if direction == "input":
            count = int(mdl.get_num_inputs(self.model_desc))
            size_fn = mdl.get_input_size_by_index
        else:
            count = int(mdl.get_num_outputs(self.model_desc))
            size_fn = mdl.get_output_size_by_index
        return tuple(
            AclTensorDescriptor(
                index=index,
                name=self._optional_name(direction, index),
                dtype=self._optional_dtype(direction, index),
                shape=self._optional_shape(direction, index),
                size=int(size_fn(self.model_desc, index)),
            )
            for index in range(count)
        )

    def _optional_name(self, direction: str, index: int) -> str | None:
        callback = getattr(self._acl.mdl, f"get_{direction}_name_by_index", None)
        if not callable(callback):
            return None
        value = _result_value(callback(self.model_desc, index), f"ACL {direction} name query")
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value) if value is not None else None

    def _optional_dtype(self, direction: str, index: int) -> np.dtype | None:
        callback = getattr(self._acl.mdl, f"get_{direction}_data_type", None)
        if not callable(callback):
            return None
        value = int(_result_value(callback(self.model_desc, index), f"ACL {direction} dtype query"))
        try:
            return _ACL_DTYPES[value]
        except KeyError as exc:
            raise BackendLoadError(
                f"Ascend role {self.role!r} {direction} {index} exposes unsupported ACL dtype code {value}",
                code="unsupported_runtime_dtype",
            ) from exc

    def _optional_shape(self, direction: str, index: int) -> tuple[int, ...] | None:
        callback = getattr(self._acl.mdl, f"get_{direction}_dims", None)
        if not callable(callback):
            return None
        value = _result_value(callback(self.model_desc, index), f"ACL {direction} shape query")
        if isinstance(value, dict):
            value = value.get("dims")
        if not isinstance(value, list | tuple):
            return None
        return tuple(int(dimension) for dimension in value)

    def _validate_bindings(self) -> None:
        self._validate_direction("input", self.bindings.inputs, self.input_descriptors)
        self._validate_direction("output", self.bindings.outputs, self.output_descriptors)

    def _validate_direction(
        self,
        direction: str,
        bindings: tuple[TensorBinding, ...],
        descriptors: tuple[AclTensorDescriptor, ...],
    ) -> None:
        indexed = _binding_by_index(bindings, direction)
        if len(indexed) != len(descriptors):
            raise BackendLoadError(
                f"Ascend role {self.role!r} declares {len(indexed)} {direction} bindings but runtime exposes "
                f"{len(descriptors)}",
                code=f"{direction}_binding_count_mismatch",
            )
        for descriptor in descriptors:
            try:
                binding = indexed[descriptor.index]
            except KeyError as exc:
                raise BackendLoadError(
                    f"Ascend role {self.role!r} has no manifest {direction} binding for runtime index "
                    f"{descriptor.index}",
                    code=f"missing_{direction}_binding",
                ) from exc
            if (
                descriptor.name is not None
                and binding.runtime_name is not None
                and descriptor.name != binding.runtime_name
            ):
                raise BackendLoadError(
                    f"Ascend role {self.role!r} {direction} {descriptor.index} runtime name "
                    f"{descriptor.name!r} does not match manifest name {binding.runtime_name!r}",
                    code="runtime_name_mismatch",
                )
            binding_dtype = numpy_dtype(binding.dtype)
            if descriptor.dtype is not None and descriptor.dtype != binding_dtype:
                raise BackendLoadError(
                    f"Ascend role {self.role!r} {direction} {descriptor.index} runtime dtype "
                    f"{descriptor.dtype.name!r} does not match manifest dtype {binding.dtype!r}",
                    code="runtime_dtype_mismatch",
                )
            if descriptor.shape is not None and not self._shapes_compatible(binding.shape, descriptor.shape):
                raise BackendLoadError(
                    f"Ascend role {self.role!r} {direction} {descriptor.index} runtime shape "
                    f"{descriptor.shape} does not match manifest shape {binding.shape}",
                    code="runtime_shape_mismatch",
                )
            if all(dimension > 0 for dimension in binding.shape):
                expected_size = int(np.prod(binding.shape, dtype=np.int64)) * binding_dtype.itemsize
                if expected_size != descriptor.size:
                    raise BackendLoadError(
                        f"Ascend role {self.role!r} {direction} {descriptor.index} runtime buffer size "
                        f"{descriptor.size} does not match manifest size {expected_size}",
                        code="runtime_size_mismatch",
                    )

    def _create_dataset(
        self,
        descriptors: tuple[AclTensorDescriptor, ...],
        overrides: dict[int, AclDeviceBuffer],
    ) -> tuple[object, list[_DatasetBuffer]]:
        dataset = self._acl.mdl.create_dataset()
        if dataset is None:
            raise RuntimeError(f"acl.mdl.create_dataset({self.role}) returned no dataset")
        buffers: list[_DatasetBuffer] = []
        try:
            for descriptor in descriptors:
                override = overrides.get(descriptor.index)
                if override is not None:
                    if override.size != descriptor.size:
                        raise BackendLoadError(
                            f"Ascend device link for role {self.role!r} index {descriptor.index} has size "
                            f"{override.size}, expected {descriptor.size}",
                            code="device_link_size_mismatch",
                        )
                    pointer = override.pointer
                    owned = False
                else:
                    pointer, ret = self._acl.rt.malloc(descriptor.size, ACL_MEM_MALLOC_HUGE_FIRST)
                    check_acl_ret(ret, f"acl.rt.malloc({self.role} index {descriptor.index})")
                    owned = True
                data_buffer = self._acl.create_data_buffer(pointer, descriptor.size)
                if data_buffer is None:
                    if owned:
                        self._acl.rt.free(pointer)
                    raise RuntimeError(f"acl.create_data_buffer({self.role} index {descriptor.index}) failed")
                result = self._acl.mdl.add_dataset_buffer(dataset, data_buffer)
                if isinstance(result, tuple):
                    check_acl_ret(result[-1], f"acl.mdl.add_dataset_buffer({self.role})")
                else:
                    check_acl_ret(result, f"acl.mdl.add_dataset_buffer({self.role})")
                buffers.append(_DatasetBuffer(pointer, data_buffer, descriptor.size, owned))
        except Exception:
            self._destroy_dataset(dataset, buffers)
            raise
        return dataset, buffers

    def _destroy_dataset(self, dataset: object | None, buffers: list[_DatasetBuffer]) -> None:
        for buffer in reversed(buffers):
            self._acl.destroy_data_buffer(buffer.data_buffer)
            if buffer.owned:
                self._acl.rt.free(buffer.pointer)
        buffers.clear()
        if dataset is not None:
            self._acl.mdl.destroy_dataset(dataset)

    @staticmethod
    def _indexed_inputs(inputs: BoundInputs | dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        if isinstance(inputs, BoundInputs):
            return {int(tensor.index): tensor.value for tensor in inputs.tensors if tensor.index is not None}
        return inputs

    @staticmethod
    def _shapes_compatible(manifest_shape: tuple[int, ...], runtime_shape: tuple[int, ...]) -> bool:
        return len(manifest_shape) == len(runtime_shape) and all(
            declared == -1 or actual == -1 or declared == actual
            for declared, actual in zip(manifest_shape, runtime_shape, strict=True)
        )
