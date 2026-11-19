"""CUDA ownership tests with an explicit fake C ABI, never a real GPU."""

from __future__ import annotations

import ctypes as ct
from types import SimpleNamespace

import numpy as np
import pytest

from perception_service.hand_gesture_adapter import INPUT_SHAPE, OUTPUT_NAME, OUTPUT_SHAPE
from perception_service.hand_gesture_session import _TensorRtEngine, resource_headroom_reason


class Function:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


class Cuda:
    """Actual host buffers behind fake cudaMalloc pointers, so copies are checked."""

    def __init__(self):
        self.allocations, self.calls = {}, []
        self.fail_sync = False
        self.fail_allocation = 0
        self.allocation_count = 0
        self.cudaSetDevice = Function(lambda device: self.record("device", device))
        self.cudaMalloc = Function(lambda pointer, size: self.allocate(pointer, size, "device"))
        self.cudaMallocHost = Function(lambda pointer, size: self.allocate(pointer, size, "host"))
        self.cudaFree = Function(lambda pointer: self.free(pointer, "device"))
        self.cudaFreeHost = Function(lambda pointer: self.free(pointer, "host"))
        self.cudaMemcpyAsync = Function(self.copy)
        self.cudaStreamCreateWithFlags = Function(self.stream)
        self.cudaStreamSynchronize = Function(self.synchronize)
        self.cudaStreamDestroy = Function(lambda stream: self.record("destroy_stream", stream.value))
        self.cudaGetErrorString = Function(lambda _code: b"explicit fake CUDA error")

    def record(self, *values):
        self.calls.append(values)
        return 0

    def allocate(self, pointer, size, kind):
        self.allocation_count += 1
        if self.fail_allocation == self.allocation_count:
            return 2
        buffer = ct.create_string_buffer(size)
        address = ct.addressof(buffer)
        self.allocations[address] = (kind, buffer)
        ct.cast(pointer, ct.POINTER(ct.c_void_p))[0] = address
        return self.record("allocate", kind, address)

    def free(self, pointer, kind):
        assert self.allocations.pop(pointer.value)[0] == kind
        return self.record("free", kind, pointer.value)

    def stream(self, pointer, flags):
        assert flags == 1  # Nonblocking stream: no implicit legacy-default-stream dependency.
        ct.cast(pointer, ct.POINTER(ct.c_void_p))[0] = 123
        return self.record("create_stream", flags)

    def copy(self, destination, source, size, direction, stream):
        assert stream.value == 123
        ct.memmove(destination, source, size)
        return self.record("copy", direction, size)

    def synchronize(self, stream):
        assert stream.value == 123
        self.record("sync")
        return 9 if self.fail_sync else 0


@pytest.fixture
def fake_cuda(monkeypatch):
    cuda = Cuda()
    addresses = {}
    control = SimpleNamespace(execute_ok=True, names=["images", OUTPUT_NAME], closed=False)

    class Context:
        def set_tensor_address(self, name, value):
            addresses[name] = value
            return True

        def execute_async_v3(self, *, stream_handle):
            assert stream_handle == 123
            cuda.record("execute")
            # Reflect input content into the output score to prove copy direction.
            incoming = np.frombuffer(cuda.allocations[addresses["images"]][1], dtype=np.float32)
            outgoing = np.frombuffer(cuda.allocations[addresses[OUTPUT_NAME]][1], dtype=np.float32).reshape(
                OUTPUT_SHAPE
            )
            outgoing.fill(0)
            outgoing[0, 12, 0] = incoming[0]
            return control.execute_ok

    class Engine:
        num_io_tensors = 2
        get_tensor_name = staticmethod(lambda index: control.names[index])
        get_tensor_shape = staticmethod(lambda name: INPUT_SHAPE if name == "images" else OUTPUT_SHAPE)
        get_tensor_mode = staticmethod(lambda name: "input" if name == "images" else "output")
        get_tensor_dtype = staticmethod(lambda _name: "float32")
        create_execution_context = staticmethod(Context)

    class Runtime:
        def __init__(self, _logger):
            pass

        @staticmethod
        def deserialize_cuda_engine(blob):
            assert blob == b"reviewed fake bytes"
            return Engine()

    class Logger:
        WARNING = 1

        def __init__(self, _level):
            pass

    module = SimpleNamespace(
        __version__="10.7.0",
        Runtime=Runtime,
        Logger=Logger,
        float32="float32",
        TensorIOMode=SimpleNamespace(INPUT="input", OUTPUT="output"),
    )
    monkeypatch.setattr("perception_service.hand_gesture_session.importlib.import_module", lambda _name: module)
    monkeypatch.setattr("perception_service.hand_gesture_session.ct.CDLL", lambda _name: cuda)
    return cuda, control


def test_buffers_are_reused_outputs_are_isolated_and_device_is_bound(fake_cuda):
    cuda, _control = fake_cuda
    runner = _TensorRtEngine(b"reviewed fake bytes", device_id=0, engine_version="10.7.0")
    try:
        first = runner.infer(np.ones(INPUT_SHAPE, np.float32))
        second = runner.infer(np.zeros(INPUT_SHAPE, np.float32))
        assert first[0, 12, 0] == 1 and second[0, 12, 0] == 0
        assert not np.shares_memory(first, second) and cuda.allocation_count == 4
        assert sum(call == ("device", 0) for call in cuda.calls) >= 3
        assert runner.quiescent
        assert [call[1] for call in cuda.calls if call[0] == "copy"] == [1, 2, 1, 2]
    finally:
        runner.close()
    assert not cuda.allocations
    frees = [index for index, call in enumerate(cuda.calls) if call[0] == "free"]
    synchronizations = [index for index, call in enumerate(cuda.calls) if call[0] == "sync"]
    assert max(synchronizations) < min(frees)
    runner.close()


def test_execution_failure_still_waits_for_stream_completion(fake_cuda):
    cuda, control = fake_cuda
    runner = _TensorRtEngine(b"reviewed fake bytes", device_id=0, engine_version="10.7.0")
    control.execute_ok = False
    try:
        with pytest.raises(RuntimeError, match="execute_async_v3"):
            runner.infer(np.zeros(INPUT_SHAPE, np.float32))
        assert runner.quiescent and cuda.calls[-1] == ("sync",)
    finally:
        runner.close()
    assert not cuda.allocations


def test_failed_sync_preserves_unknown_execution_and_does_not_free_buffers(fake_cuda):
    cuda, _control = fake_cuda
    runner = _TensorRtEngine(b"reviewed fake bytes", device_id=0, engine_version="10.7.0")
    cuda.fail_sync = True
    with pytest.raises(RuntimeError, match="CUDA"):
        runner.infer(np.zeros(INPUT_SHAPE, np.float32))
    assert not runner.quiescent
    with pytest.raises(RuntimeError, match="CUDA"):
        runner.close()
    assert len(cuda.allocations) == 4
    cuda.fail_sync = False
    runner.close()
    assert not cuda.allocations


@pytest.mark.parametrize("allocation", [1, 2, 3, 4])
def test_partial_allocation_failure_rolls_back_exact_owned_resources(fake_cuda, allocation):
    cuda, _control = fake_cuda
    cuda.fail_allocation = allocation
    with pytest.raises(RuntimeError, match="CUDA"):
        _TensorRtEngine(b"reviewed fake bytes", device_id=0, engine_version="10.7.0")
    assert not cuda.allocations


def test_engine_version_mismatch_does_not_allocate(fake_cuda):
    cuda, _control = fake_cuda
    with pytest.raises(RuntimeError, match="requires TensorRT"):
        _TensorRtEngine(b"reviewed fake bytes", device_id=0, engine_version="9.0")
    assert not cuda.allocations and not cuda.calls


@pytest.mark.parametrize(
    "memory,temp,allowed",
    [
        (3 * 1024 * 1024, 79000, True),
        (2 * 1024 * 1024 - 1, 65000, False),
        (3 * 1024 * 1024, 80000, False),
        (3 * 1024 * 1024, None, False),
    ],
)
def test_resource_thresholds_fail_closed(monkeypatch, memory, temp, allowed):
    class FakePath:
        def __init__(self, path):
            self.path = path

        def read_text(self):
            return f"MemAvailable: {memory} kB\n" if self.path == "/proc/meminfo" else str(temp)

        def glob(self, _pattern):
            return [] if temp is None else [FakePath("temperature")]

    monkeypatch.setattr("perception_service.hand_gesture_session.Path", FakePath)
    assert bool(resource_headroom_reason()) is not allowed
