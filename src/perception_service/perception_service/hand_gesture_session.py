"""Narrow serialized TensorRT engine resource under the existing native runtime.

No engine builder, CUDA reset, background inference thread, or model host lives
here. TensorRT is imported lazily only for the real, manifest-selected session.
"""

from __future__ import annotations

import ctypes as ct
import ctypes.util
import fcntl
import hashlib
import importlib
from pathlib import Path

import numpy as np

from inference_manifest import TensorRTRuntimeProfile
from inference_service.backends import BackendCapabilities
from inference_service.model_sessions import ModelSession

from .hand_gesture_adapter import INPUT_SEMANTIC, INPUT_SHAPE, OUTPUT_NAME, OUTPUT_SEMANTIC, OUTPUT_SHAPE


def resource_headroom_reason():
    """Read-only admission check; never change clocks, power or thermal policy.

    This stops accepting optional gesture work under pressure. It is a guard,
    not a CPU/GPU reservation or a guarantee of vendor application performance.
    """
    try:
        values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        if int(values["MemAvailable"].split()[0]) < 2 * 1024 * 1024:
            return "gesture admission paused: less than 2 GiB MemAvailable"
        temperatures = [int(path.read_text()) for path in Path("/sys/class/thermal").glob("thermal_zone*/temp")]
        if not temperatures:
            return "gesture admission paused: temperature monitoring unavailable"
        if max(temperatures) >= 80_000:
            return "gesture admission paused: temperature is at least 80 C"
    except (OSError, ValueError, KeyError) as exc:
        return f"gesture admission paused: resource monitoring failed ({type(exc).__name__})"
    return ""


class _TensorRtEngine:
    """One fixed engine/context and explicit nonblocking CUDA stream.

    I/O buffers are allocated once. CUDA synchronization is the only proof that
    queued work is finished. A failed synchronization leaves quiescent false;
    callers must not acknowledge retirement or free possibly in-use buffers.
    """

    def __init__(self, blob, *, device_id, engine_version):
        self._stream = ct.c_void_p()
        self._buffers = {}
        self._runtime = self._engine = self._context = None
        self.quiescent = True
        self._device_id = device_id
        trt = importlib.import_module("tensorrt")
        self.runtime_version = trt.__version__
        if self.runtime_version != engine_version:
            raise RuntimeError(f"engine requires TensorRT {engine_version}, got {self.runtime_version}")
        self._cuda = ct.CDLL(ctypes.util.find_library("cudart") or "libcudart.so.12")
        signatures = {
            "cudaSetDevice": [ct.c_int],
            "cudaMalloc": [ct.POINTER(ct.c_void_p), ct.c_size_t],
            "cudaFree": [ct.c_void_p],
            "cudaMallocHost": [ct.POINTER(ct.c_void_p), ct.c_size_t],
            "cudaFreeHost": [ct.c_void_p],
            "cudaMemcpyAsync": [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_void_p],
            "cudaStreamCreateWithFlags": [ct.POINTER(ct.c_void_p), ct.c_uint],
            "cudaStreamSynchronize": [ct.c_void_p],
            "cudaStreamDestroy": [ct.c_void_p],
        }
        for name, args in signatures.items():
            function = getattr(self._cuda, name)
            function.argtypes, function.restype = args, ct.c_int
        self._cuda.cudaGetErrorString.argtypes = [ct.c_int]
        self._cuda.cudaGetErrorString.restype = ct.c_char_p
        try:
            self._bind()
            self._logger = trt.Logger(trt.Logger.WARNING)
            self._runtime = trt.Runtime(self._logger)
            self._engine = self._runtime.deserialize_cuda_engine(blob)
            if self._engine is None:
                raise RuntimeError("TensorRT could not deserialize the selected engine on this device")
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError("TensorRT could not create an execution context")
            names = {self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)}
            if names != {"images", OUTPUT_NAME}:
                raise ValueError(f"unexpected engine I/O names: {sorted(names)}")
            self._check(self._cuda.cudaStreamCreateWithFlags(ct.byref(self._stream), 1))
            for name, shape, mode in (
                ("images", INPUT_SHAPE, trt.TensorIOMode.INPUT),
                (OUTPUT_NAME, OUTPUT_SHAPE, trt.TensorIOMode.OUTPUT),
            ):
                if (
                    tuple(self._engine.get_tensor_shape(name)) != shape
                    or self._engine.get_tensor_dtype(name) != trt.float32
                    or self._engine.get_tensor_mode(name) != mode
                ):
                    raise ValueError(f"unexpected engine tensor contract for {name}")
                size = int(np.prod(shape)) * 4
                host, device = ct.c_void_p(), ct.c_void_p()
                self._buffers[name] = [None, host, device]
                self._check(self._cuda.cudaMallocHost(ct.byref(host), size))
                self._check(self._cuda.cudaMalloc(ct.byref(device), size))
                value = np.frombuffer((ct.c_ubyte * size).from_address(host.value), dtype=np.float32).reshape(shape)
                self._buffers[name][0] = value
                if not self._context.set_tensor_address(name, device.value):
                    raise RuntimeError(f"could not bind engine tensor {name}")
        except BaseException:
            self.close()
            raise

    def _check(self, status):
        if status:
            raise RuntimeError(f"CUDA {status}: {self._cuda.cudaGetErrorString(status).decode()}")

    def _bind(self):
        # ModelServiceNode loads on the main thread and calls on executor threads.
        self._check(self._cuda.cudaSetDevice(self._device_id))

    def infer(self, x):
        self._bind()
        if not self.quiescent:
            raise RuntimeError("previous CUDA work has no synchronization proof")
        host, host_ptr, device = self._buffers["images"]
        np.copyto(host, x, casting="no")
        self.quiescent = False
        try:
            self._check(self._cuda.cudaMemcpyAsync(device, host_ptr, host.nbytes, 1, self._stream))
            if not self._context.execute_async_v3(stream_handle=self._stream.value):
                raise RuntimeError("TensorRT execute_async_v3 failed")
            out, out_host, out_device = self._buffers[OUTPUT_NAME]
            self._check(self._cuda.cudaMemcpyAsync(out_host, out_device, out.nbytes, 2, self._stream))
        finally:
            # A successful enqueue, cancellation by a client, or Python exception
            # alone is not proof that the stream is idle.
            self._check(self._cuda.cudaStreamSynchronize(self._stream))
            self.quiescent = True
        return out.copy()

    def close(self):
        if not hasattr(self, "_cuda"):
            return
        self._bind()
        if self._stream.value:
            self._check(self._cuda.cudaStreamSynchronize(self._stream))
            self.quiescent = True
        self._context = self._engine = self._runtime = None
        for name, (_, host, device) in list(self._buffers.items()):
            if device.value:
                self._check(self._cuda.cudaFree(device))
                device.value = None
            if host.value:
                self._check(self._cuda.cudaFreeHost(host))
                host.value = None
            del self._buffers[name]
        if self._stream.value:
            self._check(self._cuda.cudaStreamDestroy(self._stream))
            self._stream = ct.c_void_p()


class TensorRtHandGestureSession(ModelSession):
    """A model resource, with lifecycle/admission owned by ModelRuntimeHandle."""

    def __init__(self, *, runner_factory=None):
        super().__init__("model-session:x2-hand-gesture-tensorrt", BackendCapabilities(max_in_flight_per_instance=1))
        self._factory = runner_factory or _TensorRtEngine
        self._runner = self._device_lock = None
        self.engine_sha256 = ""

    @property
    def quiescent(self):
        return self._runner is None or self._runner.quiescent

    @property
    def runtime_version(self):
        return self._runner.runtime_version if self._runner is not None else ""

    def resource_headroom_reason(self):
        return resource_headroom_reason()

    def _load(self, context, rollback):
        if reason := self.resource_headroom_reason():
            raise RuntimeError(reason)
        profile = context.backend_profile
        if (context.interface, context.model_type, context.operation, context.backend, context.target_runtime) != (
            "tensor_model",
            "hand_gesture",
            "classify",
            "tensorrt",
            "tensorrt",
        ):
            raise ValueError("gesture session requires tensor_model/hand_gesture/classify/tensorrt")
        if not isinstance(profile, TensorRTRuntimeProfile) or profile.precision != "fp32" or not profile.engine_version:
            raise ValueError("gesture session requires a typed FP32 TensorRT engine profile")
        deployment = context.deployment
        if deployment.execution != ("model",) or deployment.device_links or deployment.execution_contract.stateful:
            raise ValueError("gesture engine requires a single stateless model role without device links")
        artifact = deployment.artifacts["model"]
        if artifact.format != "engine" or not artifact.sha256:
            raise ValueError("gesture engine requires an actual artifact SHA-256")
        bindings = deployment.bindings["model"]
        if len(bindings.inputs) != 1 or len(bindings.outputs) != 1:
            raise ValueError("gesture engine requires one input and one output")
        for binding, semantic, name, shape in (
            (bindings.inputs[0], INPUT_SEMANTIC, "images", INPUT_SHAPE),
            (bindings.outputs[0], OUTPUT_SEMANTIC, OUTPUT_NAME, OUTPUT_SHAPE),
        ):
            if (
                binding.semantic != semantic
                or binding.runtime_name != name
                or binding.dtype != "float32"
                or tuple(binding.shape) != shape
            ):
                raise ValueError("gesture engine manifest tensor bindings do not match its fixed ABI")
        path = context.resolved_artifacts["model"]
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("gesture engine exceeds the 64 MiB artifact bound")
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != artifact.sha256:
            raise ValueError("gesture engine actual SHA-256 differs from the selected manifest")
        device_id = profile.device_id or 0
        lock_dir = Path.home() / ".cache" / "ibrobot" / "runtime-locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock = (lock_dir / f"hand-gesture-cuda-{device_id}.lock").open("a")
        rollback.defer(lock.close)
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another gesture engine already owns this user's CUDA device lease") from exc
        runner = self._factory(blob, device_id=device_id, engine_version=profile.engine_version)
        rollback.defer(runner.close)
        self._runner, self._device_lock, self.engine_sha256 = runner, lock, digest

    def _execute(self, request, context):
        context.check("hand_gesture")
        x = np.asarray(request.inputs[INPUT_SEMANTIC])
        if x.shape != INPUT_SHAPE or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("gesture engine input must be finite float32 [1,3,320,320]")
        return {OUTPUT_SEMANTIC: self._runner.infer(x)}

    def _close(self):
        if self._runner is not None:
            self._runner.close()
            self._runner = None
        if self._device_lock is not None:
            self._device_lock.close()
            self._device_lock = None
