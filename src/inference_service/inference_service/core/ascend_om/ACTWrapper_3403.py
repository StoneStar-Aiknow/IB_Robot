"""SD3403 ACT worker wrapper (persistent worker + binary protocol)."""

import contextlib
import logging
import os
import re
import struct
import subprocess
import threading
import time
from collections import deque
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor

from inference_service.core.ascend_om._sd3403_action import (
    DEFAULT_ACTION_DIM,
    _validate_positive_int,
    decode_sd3403_action_array,
)

PROTOCOL_MAGIC = 0x53565031
# The little-endian wire marker is b"1PVS" (0x31='1', 0x50='P', 0x56='V', 0x53='S');
# it is not the log-friendly spelling "SVP1".
_MAGIC_BYTES = struct.pack("<I", PROTOCOL_MAGIC)
PROTOCOL_VERSION = 1
WORKER_STATUS_OK = 0

WORKER_ELEM_FLOAT32 = 1
WORKER_ELEM_FLOAT16 = 2
WORKER_ELEM_INT8 = 3
WORKER_ELEM_UINT8 = 4
WORKER_ELEM_INT32 = 5
WORKER_ELEM_INT64 = 6

PREFERRED_ACTION_OUTPUT_INDEX = 1
MAX_WORKER_OUTPUT_COUNT = 32
MAX_WORKER_OUTPUT_DIM_COUNT = 8
MAX_WORKER_OUTPUT_BYTE_SIZE = 1 << 28
MAX_WORKER_ERROR_MSG_SIZE = 1 << 16
MAX_RESPONSE_PREFIX_BYTES = 1 << 20
RESPONSE_PREFIX_TAIL_BYTES = 120
DEFAULT_GRACEFUL_CLOSE_TIMEOUT = 5.0
WORKER_TERMINATE_TIMEOUT = 1.0
WORKER_KILL_TIMEOUT = 1.0
STDERR_THREAD_JOIN_TIMEOUT = 0.2

REQUEST_HEADER_STRUCT = struct.Struct("<IHHII")
INPUT_ENTRY_STRUCT = struct.Struct("<III")
RESPONSE_HEADER_STRUCT = struct.Struct("<IHHIIIiI")
OUTPUT_ENTRY_STRUCT = struct.Struct("<IIIIII")
DIM_STRUCT = struct.Struct("<Q")
MODEL_LOAD_MS_RE = re.compile(r"model_load_ms=([0-9]+(?:\.[0-9]+)?)")
LEGACY_WORKER_ENV_KEYS = (
    "SVP_MODEL_PATH",
    "SVP_WORKER_EXECUTABLE",
    "SVP_CPP_EXECUTABLE",
    "SVP_IMAGE_HEIGHT",
    "SVP_IMAGE_WIDTH",
    "SVP_PERF_LOG",
    "SVP_PERF_LOG_EVERY",
    "SVP_WORKER_GRACEFUL_CLOSE_TIMEOUT",
    "SVP_WORKER_FORCE_CLOSE",
    "SVP_WORKER_DETACH",
)
LOGGER = logging.getLogger(__name__)


def _is_om_path(path: str) -> bool:
    return path.lower().endswith(".om")


def _guess_worker_path_from_model(model_path: str) -> str:
    model_dir = os.path.dirname(model_path)
    if os.path.basename(model_dir) == "model":
        return os.path.normpath(os.path.join(model_dir, "../out/main"))
    return os.path.normpath(os.path.join(model_dir, "main"))


def _read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise RuntimeError("worker stream closed unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


# Chunk size for buffered reads while scanning the worker response stream.
# Larger blocks reduce the number of read() syscalls on the hot predict() path;
# 4 KiB comfortably covers a typical response prefix in a single read while
# staying well below MAX_RESPONSE_PREFIX_BYTES.
_RESPONSE_READ_CHUNK = 4096


class _PrefixedStream:
    """Buffered reader over a raw stream that preserves over-read bytes.

    ``read_response_header`` scans for the magic prefix in blocks; any bytes
    read past the header belong to the following output entries and are kept
    here so the subsequent ``_read_exact``/``read`` calls consume them first.
    """

    def __init__(self, stream, chunk_size: int = _RESPONSE_READ_CHUNK):
        self._stream = stream
        self._chunk_size = max(1, int(chunk_size))
        self._buffer = bytearray()

    def _refill(self) -> bool:
        if not self._buffer:
            chunk = self._stream.read(self._chunk_size)
            if chunk:
                self._buffer.extend(chunk)
            return bool(chunk)
        return True

    def read(self, size: int) -> bytes:
        remaining = size
        out = bytearray()
        while remaining > 0:
            if not self._buffer and not self._refill():
                break
            take = self._buffer[:remaining]
            del self._buffer[: len(take)]
            out += take
            remaining -= len(take)
        return bytes(out)

    def scan_for_prefix(self, prefix: bytes, max_skip: int, tail_size: int) -> bytes:
        """Scan forward until ``prefix`` is found, skipping preceding bytes.

        Returns the matched ``prefix`` bytes. Any over-read bytes beyond the
        prefix remain buffered internally for subsequent reads.
        Raises ``RuntimeError`` if the stream closes or ``max_skip`` is exceeded.
        ``tail_size`` controls how many trailing skipped bytes are reported.
        """
        window = bytearray()
        skipped_tail = deque(maxlen=tail_size)
        skipped_count = 0
        plen = len(prefix)

        while True:
            if not self._buffer and not self._refill():
                raise RuntimeError(
                    f"worker stream closed while waiting for response header, skipped_tail={bytes(skipped_tail)!r}"
                )
            byte = self._buffer[0]
            del self._buffer[0]
            window.append(byte)
            if len(window) > plen:
                skipped_tail.append(window.pop(0))
                skipped_count += 1
                if skipped_count > max_skip:
                    raise RuntimeError(
                        "worker response header was not found after "
                        f"{max_skip} skipped bytes, skipped_tail={bytes(skipped_tail)!r}"
                    )
            if len(window) == plen and bytes(window) == prefix:
                return bytes(window)


def _read_response_header(stream: _PrefixedStream) -> tuple[int, int, int, int, int, int, int, int]:
    window = stream.scan_for_prefix(_MAGIC_BYTES, MAX_RESPONSE_PREFIX_BYTES, RESPONSE_PREFIX_TAIL_BYTES)
    rest = _read_exact(stream, RESPONSE_HEADER_STRUCT.size - len(_MAGIC_BYTES))
    return RESPONSE_HEADER_STRUCT.unpack(window + rest)


def _dtype_from_elem_type(elem_type: int):
    if elem_type == WORKER_ELEM_FLOAT32:
        return np.float32
    if elem_type == WORKER_ELEM_FLOAT16:
        return np.float16
    if elem_type == WORKER_ELEM_INT8:
        return np.int8
    if elem_type == WORKER_ELEM_UINT8:
        return np.uint8
    if elem_type == WORKER_ELEM_INT32:
        return np.int32
    if elem_type == WORKER_ELEM_INT64:
        return np.int64
    raise RuntimeError(f"unsupported worker element type: {elem_type}")


def _to_numpy_float32(t: Tensor) -> np.ndarray:
    # Fast path: no copy when already CPU float32 contiguous tensor.
    if t.device.type == "cpu" and t.dtype == torch.float32 and t.is_contiguous():
        return t.detach().numpy()
    return np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32, copy=False))


def _preferred_action_output(
    outputs: list[tuple[int, np.ndarray]],
    action_output_index: int = PREFERRED_ACTION_OUTPUT_INDEX,
) -> np.ndarray | None:
    if not outputs:
        return None

    for index, data in outputs:
        if index == action_output_index:
            return data

    return None


def _validate_non_negative_int(name: str, value: int) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _validate_output_entry(elem_type: int, elem_count: int, byte_size: int, dim_count: int) -> np.dtype:
    if dim_count > MAX_WORKER_OUTPUT_DIM_COUNT or byte_size > MAX_WORKER_OUTPUT_BYTE_SIZE:
        raise RuntimeError(f"unreasonable response output: dim_count={dim_count}, byte_size={byte_size}")
    dtype = np.dtype(_dtype_from_elem_type(elem_type))
    if elem_count > MAX_WORKER_OUTPUT_BYTE_SIZE // dtype.itemsize:
        raise RuntimeError(f"unreasonable response output: elem_count={elem_count}, dtype={dtype.name}")
    expected_byte_size = elem_count * dtype.itemsize
    if byte_size != expected_byte_size:
        raise RuntimeError(
            "response output byte size does not match element count: "
            f"byte_size={byte_size}, expected={expected_byte_size}"
        )
    return dtype


class ACT3403Policy:
    def __init__(
        self,
        cpp_executable: str,
        model_path: str | None = None,
        *,
        action_dim: int = DEFAULT_ACTION_DIM,
        action_output_index: int = PREFERRED_ACTION_OUTPUT_INDEX,
        image_height: int | None = None,
        image_width: int | None = None,
        perf_enabled: bool = False,
        perf_log_every: int = 1,
        graceful_close_timeout: float = DEFAULT_GRACEFUL_CLOSE_TIMEOUT,
        force_close: bool = True,
    ):
        super().__init__()
        self.cpp_executable, resolved_model_path = self._resolve_paths(cpp_executable, model_path)
        self.model_path = resolved_model_path
        self.cpp_dir = os.path.dirname(self.cpp_executable)
        self._action_dim = _validate_positive_int("action_dim", action_dim)
        self._action_output_index = _validate_non_negative_int("action_output_index", action_output_index)

        legacy_env_keys = sorted(key for key in LEGACY_WORKER_ENV_KEYS if key in os.environ)
        if legacy_env_keys:
            LOGGER.warning(
                "[ACT3403Policy] ignoring legacy worker environment variables: %s; "
                "use config.json/config.om.json metadata instead",
                ", ".join(legacy_env_keys),
            )
        self._worker_env = {key: value for key, value in os.environ.items() if key not in LEGACY_WORKER_ENV_KEYS}
        # image_height/width are optional: when None, the worker trusts the
        # incoming tensor size as-is (the resolution is the model's input
        # contract, derived from config.json input_features, not a hardcoded
        # value). When supplied, inputs are resized to that target.
        self._image_height = _validate_positive_int("image_height", image_height) if image_height is not None else None
        self._image_width = _validate_positive_int("image_width", image_width) if image_width is not None else None
        self._resize_warned = False
        self._perf_enabled = bool(perf_enabled)
        self._perf_log_every = _validate_positive_int("perf_log_every", perf_log_every)
        self._graceful_close_timeout = max(0.0, float(graceful_close_timeout))
        self._force_close = bool(force_close)
        self._predict_count = 0
        self._process_start_ts = 0.0
        self._model_load_ms: float | None = None
        self._model_load_logged = False
        self._model_load_reported_unavailable = False

        self._request_id = 0
        self._process: subprocess.Popen | None = None
        self._stderr_thread: threading.Thread | None = None
        self._io_lock = threading.Lock()
        self._stderr_tail = deque(maxlen=80)
        self._start_process()

    def _resolve_paths(self, cpp_executable: str, model_path: str | None) -> tuple[str, str]:
        arg = cpp_executable.strip()

        resolved_worker = ""
        resolved_model = ""
        if model_path:
            resolved_worker = arg
            resolved_model = model_path
        elif _is_om_path(arg):
            resolved_model = arg
            resolved_worker = _guess_worker_path_from_model(resolved_model)
        else:
            resolved_worker = arg

        if not resolved_worker:
            raise RuntimeError("missing worker executable path: pass binary path to ACT3403Policy(...)")
        if not resolved_model:
            raise RuntimeError(
                "missing model path: pass model_path to ACT3403Policy(...); "
                "image dims are derived from config.json input_features, not guessed"
            )

        resolved_worker = os.path.abspath(resolved_worker)
        resolved_model = os.path.abspath(resolved_model)

        if not os.path.isfile(resolved_worker):
            raise FileNotFoundError(f"worker executable not found: {resolved_worker}")
        if not os.access(resolved_worker, os.X_OK):
            raise PermissionError(f"worker executable is not executable: {resolved_worker}")
        if not os.path.isfile(resolved_model):
            raise FileNotFoundError(f"OM model file not found: {resolved_model}")
        return resolved_worker, resolved_model

    def _start_process(self):
        self._process_start_ts = time.perf_counter()
        self._model_load_ms = None
        self._model_load_logged = False
        self._model_load_reported_unavailable = False
        # Hi3403 workers must accept "--model <path>"; legacy SVP_* env knobs
        # are filtered out so stale shell state cannot select a different OM.
        self._process = subprocess.Popen(
            [self.cpp_executable, "--model", self.model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cpp_dir,
            env=self._worker_env,
            text=False,
            bufsize=0,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self._process.stderr,),
            daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self, pipe):
        # Keep the whole loop defensive: decode/search rarely raise, but a crash
        # here would silently drop subsequent worker stderr. Log and stop instead.
        try:
            while True:
                line = pipe.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                if decoded:
                    self._stderr_tail.append(decoded)
                    match = MODEL_LOAD_MS_RE.search(decoded)
                    if match is not None:
                        with contextlib.suppress(ValueError):
                            self._model_load_ms = float(match.group(1))
        except Exception as exc:  # noqa: BLE001 - stderr reader must never crash the worker
            LOGGER.debug("stderr reader stopped: %s", exc)

    def close(self):
        if self._process is None:
            return
        process = self._process
        with contextlib.suppress(Exception):
            if process.stdin:
                process.stdin.close()
        with contextlib.suppress(Exception):
            if process.stdout:
                process.stdout.close()
        with contextlib.suppress(Exception):
            if process.stderr:
                process.stderr.close()

        with contextlib.suppress(Exception):
            process.wait(timeout=self._graceful_close_timeout)

        if process.poll() is None and self._force_close:
            with contextlib.suppress(Exception):
                process.terminate()
                process.wait(timeout=WORKER_TERMINATE_TIMEOUT)
            if process.poll() is None:
                with contextlib.suppress(Exception):
                    process.kill()
                    process.wait(timeout=WORKER_KILL_TIMEOUT)

        if process.poll() is None:
            if self._force_close:
                LOGGER.warning(
                    "[ACT3403Policy] worker did not exit after stdin EOF, SIGTERM, or SIGKILL; pid=%s",
                    getattr(process, "pid", "<unknown>"),
                )
            else:
                LOGGER.warning(
                    "[ACT3403Policy] worker did not exit after stdin EOF; "
                    "left it running because force_close=False. pid=%s",
                    getattr(process, "pid", "<unknown>"),
                )

        thread = self._stderr_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=STDERR_THREAD_JOIN_TIMEOUT)
        self._stderr_thread = None
        self._process = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _worker_exit_message(self) -> str:
        if self._process is None:
            return "worker process is not running"
        return_code = self._process.poll()
        msg = "worker exited unexpectedly"
        if return_code is not None:
            msg += f" (returncode={return_code})"
        if self._stderr_tail:
            msg += "\nworker stderr tail:\n" + "\n".join(self._stderr_tail)
        return msg

    def _ensure_process(self):
        if self._process is None:
            self._start_process()
            return
        if self._process.poll() is not None:
            self.close()
            self._start_process()

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _normalize_image_tensor(self, image: Tensor, name: str) -> Tensor:
        if image.ndim != 4:
            raise RuntimeError(f"{name} must be NCHW tensor, got shape={tuple(image.shape)}")

        if image.dtype != torch.float32:
            image = image.to(dtype=torch.float32)

        # No explicit target dims: trust the incoming tensor size as-is. The
        # image resolution is the model's input contract (config.json
        # input_features.<key>.shape, baked into the OM via ATC --input_shape),
        # so the caller is expected to supply correctly-sized tensors.
        if self._image_height is None or self._image_width is None:
            return image.contiguous()

        height, width = int(image.shape[-2]), int(image.shape[-1])
        if height != self._image_height or width != self._image_width:
            image = functional.interpolate(
                image,
                size=(self._image_height, self._image_width),
                mode="bilinear",
                align_corners=False,
            )
            if not self._resize_warned:
                LOGGER.warning(
                    "[ACT3403Policy] resize image inputs to "
                    f"{self._image_height}x{self._image_width} before worker inference"
                )
                self._resize_warned = True

        return image.contiguous()

    def _build_inputs(self, batch: dict[str, Tensor]) -> list[np.ndarray]:
        # Prefer explicit keys expected by exported ACT model.
        state = batch.get("observation.state")
        top = batch.get("observation.images.top")
        wrist = batch.get("observation.images.wrist")

        if isinstance(state, Tensor) and isinstance(top, Tensor) and isinstance(wrist, Tensor):
            top = self._normalize_image_tensor(top, "observation.images.top")
            wrist = self._normalize_image_tensor(wrist, "observation.images.wrist")
            return [_to_numpy_float32(state), _to_numpy_float32(top), _to_numpy_float32(wrist)]

        # Fallback 1: merged image tensor in observation.images (camera order assumed top,wrist).
        merged_images = batch.get("observation.images")
        if isinstance(state, Tensor) and isinstance(merged_images, Tensor) and merged_images.ndim >= 2:
            if merged_images.shape[1] < 2:
                raise RuntimeError("observation.images must contain at least 2 cameras for SD3403")
            top_tensor = self._normalize_image_tensor(merged_images[:, 0, ...], "observation.images[0]")
            wrist_tensor = self._normalize_image_tensor(merged_images[:, 1, ...], "observation.images[1]")
            top_arr = _to_numpy_float32(top_tensor)
            wrist_arr = _to_numpy_float32(wrist_tensor)
            return [_to_numpy_float32(state), top_arr, wrist_arr]

        # Fallback 2: observation.images can be a list/tuple of image tensors in ACT pipeline.
        if (
            isinstance(state, Tensor)
            and isinstance(merged_images, list | tuple)
            and len(merged_images) >= 2
            and isinstance(merged_images[0], Tensor)
            and isinstance(merged_images[1], Tensor)
        ):
            top_tensor = self._normalize_image_tensor(merged_images[0], "observation.images[0]")
            wrist_tensor = self._normalize_image_tensor(merged_images[1], "observation.images[1]")
            return [
                _to_numpy_float32(state),
                _to_numpy_float32(top_tensor),
                _to_numpy_float32(wrist_tensor),
            ]

        raise RuntimeError(
            "missing required inputs: need observation.state + "
            "(observation.images.top & observation.images.wrist or observation.images)"
        )

    def _write_request(self, input_arrays: Sequence[np.ndarray]) -> int:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("worker process is not running")
        if self._process.poll() is not None:
            raise RuntimeError(self._worker_exit_message())

        request_id = self._next_request_id()
        header = REQUEST_HEADER_STRUCT.pack(
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            len(input_arrays),
            request_id,
            0,
        )
        try:
            self._process.stdin.write(header)
            for idx, arr in enumerate(input_arrays):
                contiguous = arr
                if contiguous.dtype != np.float32 or not contiguous.flags["C_CONTIGUOUS"]:
                    contiguous = np.ascontiguousarray(contiguous, dtype=np.float32)
                payload_size = int(contiguous.nbytes)
                self._process.stdin.write(INPUT_ENTRY_STRUCT.pack(idx, payload_size, 0))
                # Write directly from numpy memory to avoid an extra tobytes() copy.
                self._process.stdin.write(memoryview(contiguous).cast("B"))
            self._process.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(self._worker_exit_message()) from exc
        return request_id

    def _read_response(self, expected_request_id: int) -> tuple[np.ndarray, int]:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("worker process is not running")
        if self._process.poll() is not None:
            raise RuntimeError(self._worker_exit_message())

        # Wrap stdout in a buffered reader so the prefix scan reads in blocks
        # and any over-read bytes feed the following output-entry reads.
        stream = _PrefixedStream(self._process.stdout)
        magic, version, status, request_id, output_count, latency_us, error_code, error_msg_size = (
            _read_response_header(stream)
        )
        if magic != PROTOCOL_MAGIC or version != PROTOCOL_VERSION:
            raise RuntimeError(f"unexpected response header: magic=0x{magic:x}, version={version}")
        if request_id != expected_request_id:
            raise RuntimeError(f"mismatched response id: expected {expected_request_id}, got {request_id}")
        if output_count > MAX_WORKER_OUTPUT_COUNT or error_msg_size > MAX_WORKER_ERROR_MSG_SIZE:
            raise RuntimeError(
                f"unreasonable response header: output_count={output_count}, error_msg_size={error_msg_size}"
            )

        outputs: list[tuple[int, np.ndarray]] = []
        for _ in range(output_count):
            entry = _read_exact(stream, OUTPUT_ENTRY_STRUCT.size)
            output_index, elem_type, elem_count, byte_size, dim_count, _reserved = OUTPUT_ENTRY_STRUCT.unpack(entry)
            dtype = _validate_output_entry(elem_type, elem_count, byte_size, dim_count)
            dims = [DIM_STRUCT.unpack(_read_exact(stream, DIM_STRUCT.size))[0] for _ in range(dim_count)]
            payload = _read_exact(stream, byte_size)
            data = np.frombuffer(payload, dtype=dtype, count=elem_count)
            if dims:
                data = data.reshape(tuple(int(d) for d in dims))
            outputs.append((output_index, data))

        error_msg = ""
        if error_msg_size:
            error_msg = _read_exact(stream, error_msg_size).decode("utf-8", errors="replace")
        if status != WORKER_STATUS_OK:
            raise RuntimeError(f"worker inference failed (error_code={error_code}): {error_msg or 'unknown error'}")
        target_output = _preferred_action_output(outputs, self._action_output_index)
        if target_output is None:
            available = ", ".join(f"index={index}, shape={tuple(data.shape)}" for index, data in outputs) or "<none>"
            raise RuntimeError(f"worker response does not contain an action output; available outputs: {available}")
        return target_output, int(latency_us)

    def _execute_prepared_arrays(self, input_arrays: Sequence[np.ndarray]) -> tuple[np.ndarray, int, int]:
        request_id = self._write_request(input_arrays)
        data, worker_latency_us = self._read_response(request_id)
        return data, worker_latency_us, request_id

    def execute_arrays(self, input_arrays: Sequence[np.ndarray]) -> np.ndarray:
        """Run worker inference for already prepared arrays."""
        with self._io_lock:
            self._ensure_process()
            data, _worker_latency_us, _request_id = self._execute_prepared_arrays(input_arrays)
        return data

    def predict(self, batch: dict[str, Tensor]) -> tuple[Tensor] | None:
        t0 = time.perf_counter()
        with self._io_lock:
            self._ensure_process()
            t1 = time.perf_counter()
            input_arrays = self._build_inputs(batch)
            t2 = time.perf_counter()
            request_id = self._write_request(input_arrays)
            t3 = time.perf_counter()
            data, worker_latency_us = self._read_response(request_id)
            t4 = time.perf_counter()

        action_tensor = torch.from_numpy(decode_sd3403_action_array(data, self._action_dim))

        t5 = time.perf_counter()
        self._predict_count += 1
        if self._perf_enabled and (self._predict_count % self._perf_log_every == 0):
            e2e_ms = (t5 - t0) * 1000.0
            worker_infer_ms = worker_latency_us / 1000.0
            prepare_ms = (t2 - t1) * 1000.0
            write_ms = (t3 - t2) * 1000.0
            wait_resp_ms = (t4 - t3) * 1000.0
            post_ms = (t5 - t4) * 1000.0
            print(
                "[ACT3403Policy][PERF] "
                f"request_id={request_id} "
                f"e2e_ms={e2e_ms:.3f} "
                f"worker_infer_ms={worker_infer_ms:.3f} "
                f"prepare_ms={prepare_ms:.3f} "
                f"ipc_write_ms={write_ms:.3f} "
                f"wait_response_ms={wait_resp_ms:.3f} "
                f"post_ms={post_ms:.3f} "
                f"non_worker_ms={max(0.0, e2e_ms - worker_infer_ms):.3f}"
            )

        if self._perf_enabled and not self._model_load_logged:
            if self._model_load_ms is not None:
                print(f"[ACT3403Policy][PERF] model_load_ms={self._model_load_ms:.3f} (from worker)")
                self._model_load_logged = True
            elif self._predict_count == 1 and not self._model_load_reported_unavailable:
                print(
                    "[ACT3403Policy][PERF] precise model_load_ms not available from worker log yet; "
                    "please ensure the worker binary includes [PERF] model_load_ms logging"
                )
                self._model_load_reported_unavailable = True

        return (action_tensor,)
