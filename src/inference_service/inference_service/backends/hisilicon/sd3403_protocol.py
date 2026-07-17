"""Persistent SD3403 worker process and binary protocol implementation."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import struct
import subprocess
import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Protocol

import numpy as np

PROTOCOL_MAGIC = 0x53565031
PROTOCOL_VERSION = 1
WORKER_STATUS_OK = 0

WORKER_ELEM_FLOAT32 = 1
WORKER_ELEM_FLOAT16 = 2
WORKER_ELEM_INT8 = 3
WORKER_ELEM_UINT8 = 4
WORKER_ELEM_INT32 = 5
WORKER_ELEM_INT64 = 6

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
_MAGIC_BYTES = struct.pack("<I", PROTOCOL_MAGIC)
_RESPONSE_READ_CHUNK = 4096
_MODEL_LOAD_MS_RE = re.compile(r"model_load_ms=([0-9]+(?:\.[0-9]+)?)")
_LEGACY_WORKER_ENV_KEYS = frozenset(
    {
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
    }
)
LOGGER = logging.getLogger(__name__)


class SD3403ProtocolError(RuntimeError):
    """Raised when worker protocol data is malformed or inconsistent."""


class SD3403WorkerExitedError(SD3403ProtocolError):
    """Raised when the persistent worker is unavailable during a request."""


class SD3403WorkerError(SD3403ProtocolError):
    """Raised when the worker returns a structured inference failure."""

    def __init__(self, error_code: int, message: str) -> None:
        super().__init__(f"worker inference failed (error_code={error_code}): {message or 'unknown error'}")
        self.error_code = error_code


@dataclass(frozen=True)
class SD3403Response:
    outputs: Mapping[int, np.ndarray]
    worker_latency_us: int
    request_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))


class _Process(Protocol):
    stdin: BinaryIO | None
    stdout: BinaryIO | None
    stderr: BinaryIO | None
    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


ProcessFactory = Callable[..., _Process]


def _read_exact(stream: BinaryIO | _PrefixedStream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise SD3403WorkerExitedError("worker stream closed unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class _PrefixedStream:
    """Buffered reader that preserves bytes read beyond the protocol magic."""

    def __init__(self, stream: BinaryIO, chunk_size: int = _RESPONSE_READ_CHUNK) -> None:
        self._stream = stream
        self._chunk_size = max(1, int(chunk_size))
        self._buffer = bytearray()

    def _refill(self) -> bool:
        if self._buffer:
            return True
        chunk = self._stream.read(self._chunk_size)
        if chunk:
            self._buffer.extend(chunk)
        return bool(chunk)

    def read(self, size: int) -> bytes:
        remaining = size
        output = bytearray()
        while remaining > 0:
            if not self._buffer and not self._refill():
                break
            take = self._buffer[:remaining]
            del self._buffer[: len(take)]
            output += take
            remaining -= len(take)
        return bytes(output)

    def scan_for_prefix(self, prefix: bytes, max_skip: int, tail_size: int) -> bytes:
        window = bytearray()
        skipped_tail = deque(maxlen=tail_size)
        skipped_count = 0
        while True:
            if not self._buffer and not self._refill():
                raise SD3403WorkerExitedError(
                    f"worker stream closed while waiting for response header, skipped_tail={bytes(skipped_tail)!r}"
                )
            window.append(self._buffer.pop(0))
            if len(window) > len(prefix):
                skipped_tail.append(window.pop(0))
                skipped_count += 1
                if skipped_count > max_skip:
                    raise SD3403ProtocolError(
                        f"worker response header was not found after {max_skip} skipped bytes, "
                        f"skipped_tail={bytes(skipped_tail)!r}"
                    )
            if bytes(window) == prefix:
                return bytes(window)


def _dtype_from_element_type(element_type: int) -> np.dtype:
    types = {
        WORKER_ELEM_FLOAT32: np.dtype(np.float32),
        WORKER_ELEM_FLOAT16: np.dtype(np.float16),
        WORKER_ELEM_INT8: np.dtype(np.int8),
        WORKER_ELEM_UINT8: np.dtype(np.uint8),
        WORKER_ELEM_INT32: np.dtype(np.int32),
        WORKER_ELEM_INT64: np.dtype(np.int64),
    }
    try:
        return types[element_type]
    except KeyError as exc:
        raise SD3403ProtocolError(f"unsupported worker element type: {element_type}") from exc


class SD3403Protocol:
    """Own one persistent worker and perform serialized request/response I/O."""

    def __init__(
        self,
        worker_path: Path,
        model_path: Path,
        *,
        graceful_close_timeout: float = DEFAULT_GRACEFUL_CLOSE_TIMEOUT,
        force_close: bool = True,
        process_factory: ProcessFactory | None = None,
    ) -> None:
        self.worker_path = worker_path.resolve(strict=True)
        self.model_path = model_path.resolve(strict=True)
        self._graceful_close_timeout = graceful_close_timeout
        self._force_close = force_close
        self._process_factory = process_factory
        self._process: _Process | None = None
        self._stdout_stream: _PrefixedStream | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: deque[str] = deque(maxlen=80)
        self._request_id = 0
        self._io_lock = threading.Lock()
        self._model_load_ms: float | None = None

    @property
    def stderr_tail(self) -> tuple[str, ...]:
        return tuple(self._stderr_tail)

    @property
    def model_load_ms(self) -> float | None:
        return self._model_load_ms

    def start(self) -> None:
        with self._io_lock:
            self._ensure_process_locked()

    def execute(self, input_arrays: Sequence[np.ndarray]) -> SD3403Response:
        if not input_arrays:
            raise SD3403ProtocolError("worker request requires at least one input array")
        with self._io_lock:
            self._ensure_process_locked()
            request_id = self._write_request_locked(input_arrays)
            try:
                outputs, worker_latency_us = self._read_response_locked(request_id)
            except SD3403ProtocolError as exc:
                if self._process is None or self._process.poll() is not None:
                    raise SD3403WorkerExitedError(self._worker_exit_message()) from exc
                raise
        return SD3403Response(outputs=outputs, worker_latency_us=worker_latency_us, request_id=request_id)

    def close(self) -> None:
        with self._io_lock:
            process = self._process
            thread = self._stderr_thread
            self._process = None
            self._stdout_stream = None
            self._stderr_thread = None
            if process is None:
                return
            self._close_process(process)
        if thread is not None and thread.is_alive():
            thread.join(timeout=STDERR_THREAD_JOIN_TIMEOUT)

    def _ensure_process_locked(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        if self._process is not None:
            self._close_process(self._process)
            self._process = None
            self._stdout_stream = None
            self._stderr_thread = None
        self._start_process_locked()

    def _start_process_locked(self) -> None:
        environment = {key: value for key, value in os.environ.items() if key not in _LEGACY_WORKER_ENV_KEYS}
        factory = self._process_factory or subprocess.Popen
        process = factory(
            [str(self.worker_path), "--model", str(self.model_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.worker_path.parent),
            env=environment,
            text=False,
            bufsize=0,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            self._close_process(process)
            raise SD3403ProtocolError("worker process did not expose binary stdin/stdout/stderr pipes")
        self._process = process
        self._stdout_stream = _PrefixedStream(process.stdout)
        self._model_load_ms = None
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(process.stderr,),
            name="hisilicon-worker-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self, pipe: BinaryIO) -> None:
        try:
            while True:
                line = pipe.readline()
                if not line:
                    return
                decoded = line.decode("utf-8", errors="replace").rstrip()
                if not decoded:
                    continue
                self._stderr_tail.append(decoded)
                match = _MODEL_LOAD_MS_RE.search(decoded)
                if match is not None:
                    with contextlib.suppress(ValueError):
                        self._model_load_ms = float(match.group(1))
        except Exception as exc:  # The diagnostic reader must not terminate the worker.
            LOGGER.debug("Hisilicon worker stderr reader stopped: %s", exc)

    def _write_request_locked(self, input_arrays: Sequence[np.ndarray]) -> int:
        process = self._require_running_process()
        assert process.stdin is not None
        self._request_id += 1
        request_id = self._request_id
        try:
            process.stdin.write(
                REQUEST_HEADER_STRUCT.pack(PROTOCOL_MAGIC, PROTOCOL_VERSION, len(input_arrays), request_id, 0)
            )
            for index, array in enumerate(input_arrays):
                contiguous = np.ascontiguousarray(array, dtype=np.float32)
                process.stdin.write(INPUT_ENTRY_STRUCT.pack(index, contiguous.nbytes, 0))
                process.stdin.write(memoryview(contiguous).cast("B"))
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise SD3403WorkerExitedError(self._worker_exit_message()) from exc
        return request_id

    def _read_response_locked(self, expected_request_id: int) -> tuple[dict[int, np.ndarray], int]:
        self._require_running_process()
        stream = self._stdout_stream
        if stream is None:
            raise SD3403WorkerExitedError("worker stdout stream is not available")
        prefix = stream.scan_for_prefix(_MAGIC_BYTES, MAX_RESPONSE_PREFIX_BYTES, RESPONSE_PREFIX_TAIL_BYTES)
        header = RESPONSE_HEADER_STRUCT.unpack(prefix + _read_exact(stream, RESPONSE_HEADER_STRUCT.size - len(prefix)))
        magic, version, status, request_id, output_count, latency_us, error_code, error_msg_size = header
        if magic != PROTOCOL_MAGIC or version != PROTOCOL_VERSION:
            raise SD3403ProtocolError(f"unexpected response header: magic=0x{magic:x}, version={version}")
        if request_id != expected_request_id:
            raise SD3403ProtocolError(f"mismatched response id: expected {expected_request_id}, got {request_id}")
        if output_count > MAX_WORKER_OUTPUT_COUNT or error_msg_size > MAX_WORKER_ERROR_MSG_SIZE:
            raise SD3403ProtocolError(
                f"unreasonable response header: output_count={output_count}, error_msg_size={error_msg_size}"
            )

        outputs: dict[int, np.ndarray] = {}
        for _ in range(output_count):
            entry = OUTPUT_ENTRY_STRUCT.unpack(_read_exact(stream, OUTPUT_ENTRY_STRUCT.size))
            output_index, element_type, element_count, byte_size, dimension_count, _reserved = entry
            if output_index in outputs:
                raise SD3403ProtocolError(f"worker response contains duplicate output index {output_index}")
            if dimension_count > MAX_WORKER_OUTPUT_DIM_COUNT or byte_size > MAX_WORKER_OUTPUT_BYTE_SIZE:
                raise SD3403ProtocolError(
                    f"unreasonable response output: dim_count={dimension_count}, byte_size={byte_size}"
                )
            dtype = _dtype_from_element_type(element_type)
            expected_size = element_count * dtype.itemsize
            if element_count > MAX_WORKER_OUTPUT_BYTE_SIZE // dtype.itemsize or byte_size != expected_size:
                raise SD3403ProtocolError(
                    "response output byte size does not match element count: "
                    f"byte_size={byte_size}, expected={expected_size}"
                )
            dimensions = tuple(
                int(DIM_STRUCT.unpack(_read_exact(stream, DIM_STRUCT.size))[0]) for _ in range(dimension_count)
            )
            if dimensions and int(np.prod(dimensions)) != element_count:
                raise SD3403ProtocolError(
                    f"response output dimensions {dimensions} do not contain {element_count} elements"
                )
            data = np.frombuffer(_read_exact(stream, byte_size), dtype=dtype, count=element_count).copy()
            outputs[output_index] = data.reshape(dimensions) if dimensions else data

        error_message = ""
        if error_msg_size:
            error_message = _read_exact(stream, error_msg_size).decode("utf-8", errors="replace")
        if status != WORKER_STATUS_OK:
            raise SD3403WorkerError(error_code, error_message)
        return outputs, int(latency_us)

    def _require_running_process(self) -> _Process:
        if self._process is None or self._process.poll() is not None:
            raise SD3403WorkerExitedError(self._worker_exit_message())
        return self._process

    def _worker_exit_message(self) -> str:
        process = self._process
        if process is None:
            return "worker process is not running"
        message = "worker exited unexpectedly"
        return_code = process.poll()
        if return_code is not None:
            message += f" (returncode={return_code})"
        if self._stderr_tail:
            message += "\nworker stderr tail:\n" + "\n".join(self._stderr_tail)
        return message

    def _close_process(self, process: _Process) -> None:
        for stream in (process.stdin, process.stdout, process.stderr):
            with contextlib.suppress(Exception):
                if stream is not None:
                    stream.close()
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
            mode = "after EOF, SIGTERM, and SIGKILL" if self._force_close else "after EOF with force_close disabled"
            LOGGER.warning("Hisilicon worker did not exit %s; pid=%s", mode, getattr(process, "pid", "<unknown>"))
