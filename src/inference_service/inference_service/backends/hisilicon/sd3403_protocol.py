"""Persistent SD3403 worker process and binary protocol implementation."""

from __future__ import annotations

import contextlib
import io
import logging
import math
import os
import re
import select
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

from inference_service.unified_runtime import Deadline, OutcomeEvidence, OutcomeState
from inference_service.unified_runtime.errors import CancellationRequested, DeadlineExceeded
from inference_service.unified_runtime.lifecycle import (
    POLL_INTERVAL_SECONDS,
    check_lifecycle,
    lifecycle_lock,
    lifecycle_remaining,
)

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
_RESPONSE_READ_CHUNK = 65536
_PIPE_WRITE_CHUNK_BYTES = 65536  # Bound each pipe write; not a protocol payload limit.
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

    def __init__(self, stream: BinaryIO, chunk_size: int = _RESPONSE_READ_CHUNK, *, check=check_lifecycle) -> None:
        self._stream = stream
        self._check = check
        self._chunk_size = max(1, int(chunk_size))
        self._buffer = bytearray()

    def _refill(self) -> bool:
        if self._buffer:
            return True
        self._check()
        try:
            descriptor = self._stream.fileno()
        except (AttributeError, io.UnsupportedOperation):
            # In-memory protocol fixtures have no OS pipe to poll.
            chunk = self._stream.read(self._chunk_size)
        else:
            poller = select.poll()
            poller.register(descriptor, select.POLLIN)
            while not poller.poll(POLL_INTERVAL_SECONDS * 1000):
                self._check()
            self._check()
            chunk = os.read(descriptor, self._chunk_size)
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
        if not prefix:
            raise ValueError("response prefix must not be empty")
        window = bytearray()
        skipped_tail = deque(maxlen=tail_size)
        skipped_count = 0
        while True:
            if not self._buffer and not self._refill():
                raise SD3403WorkerExitedError(
                    f"worker stream closed while waiting for response header, skipped_tail={bytes(skipped_tail)!r}"
                )
            window.extend(self._buffer)
            self._buffer.clear()
            match = window.find(prefix)
            discard = match if match >= 0 else max(0, len(window) - len(prefix) + 1)
            skipped_tail.extend(window[:discard])
            skipped_count += discard
            if skipped_count > max_skip:
                raise SD3403ProtocolError(
                    f"worker response header was not found after {max_skip} skipped bytes, "
                    f"skipped_tail={bytes(skipped_tail)!r}"
                )
            if match >= 0:
                self._buffer.extend(window[match + len(prefix) :])
                return prefix
            del window[:discard]


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
        if (
            type(graceful_close_timeout) not in {int, float}
            or not math.isfinite(graceful_close_timeout)
            or graceful_close_timeout < 0
        ):
            raise ValueError("graceful_close_timeout must be a finite non-negative number")
        self.worker_path = worker_path.resolve(strict=True)
        self.model_path = model_path.resolve(strict=True)
        self._graceful_close_timeout = graceful_close_timeout
        self._force_close = force_close
        self._process_factory = process_factory
        self._process: _Process | None = None
        self._stdout_stream: _PrefixedStream | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_stop = threading.Event()
        self._stderr_tail: deque[str] = deque(maxlen=80)
        self._request_id = 0
        self._io_lock = threading.Lock()
        self._closing = False
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
        # Complete every conversion before the first protocol byte is sent.
        try:
            if not input_arrays or len(input_arrays) > 0xFFFF:
                raise ValueError("worker request requires between 1 and 65535 input arrays")
            input_arrays = tuple(np.ascontiguousarray(array, dtype=np.float32) for array in input_arrays)
            if any(array.nbytes > 0xFFFFFFFF for array in input_arrays):
                raise ValueError("worker input exceeds the protocol byte-size limit")
        except Exception as exc:
            exc.evidence = OutcomeEvidence.not_started("worker_inputs")
            raise
        writing = False
        try:
            with lifecycle_lock(self._io_lock):
                self._ensure_process_locked()
                writing = True
                request_id = self._write_request_locked(input_arrays)
                try:
                    outputs, worker_latency_us = self._read_response_locked(request_id)
                except (DeadlineExceeded, CancellationRequested):
                    # The request was sent, but this response is not consumed.
                    # Reuse would lose framing or read it as the next response.
                    self._closing = True
                    raise
                except SD3403ProtocolError as exc:
                    if self._process is None or self._process.poll() is not None:
                        raise SD3403WorkerExitedError(self._worker_exit_message()) from exc
                    raise
        except (DeadlineExceeded, CancellationRequested) as exc:
            if not writing:
                exc.evidence = OutcomeEvidence.not_started(exc.phase)
            raise
        return SD3403Response(outputs=outputs, worker_latency_us=worker_latency_us, request_id=request_id)

    def _check_io_wait(self, phase: str) -> None:
        check_lifecycle(phase=phase)
        if self._closing:
            raise SD3403ProtocolError("worker I/O interrupted by Close")

    def _check_response_wait(self) -> None:
        self._check_io_wait("worker_response")

    def close(self) -> None:
        self._closing = True
        with lifecycle_lock(self._io_lock):
            process = self._process
            thread = self._stderr_thread
            if process is not None:
                check_lifecycle()
                self._close_process(process)
            self._stderr_stop.set()
            if thread is not None and thread.is_alive():
                deadline = Deadline.after(STDERR_THREAD_JOIN_TIMEOUT)
                while thread.is_alive() and not deadline.expired:
                    check_lifecycle()
                    thread.join(timeout=min(POLL_INTERVAL_SECONDS, lifecycle_remaining(deadline.remaining_seconds())))
                if thread.is_alive():
                    raise SD3403ProtocolError("Hisilicon stderr worker close is pending")
            if process is not None and process.stderr is not None:
                process.stderr.close()
            self._process = None
            self._stdout_stream = None
            self._stderr_thread = None

    def _ensure_process_locked(self) -> None:
        if self._closing:
            raise SD3403ProtocolError("worker Close is irreversible")
        if self._process is not None and self._process.poll() is None:
            return
        if self._process is not None:
            self._close_process(self._process)
            self._stderr_stop.set()
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=STDERR_THREAD_JOIN_TIMEOUT)
                if self._stderr_thread.is_alive():
                    raise SD3403ProtocolError("Hisilicon stderr worker close is pending")
            if self._process.stderr is not None:
                self._process.stderr.close()
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
        self._process = process
        if process.stdin is None or process.stdout is None or process.stderr is None:
            self._close_process(process)
            raise SD3403ProtocolError("worker process did not expose binary stdin/stdout/stderr pipes")
        # In-memory protocol fixtures have no OS pipe.
        with contextlib.suppress(AttributeError, io.UnsupportedOperation):
            os.set_blocking(process.stdin.fileno(), False)
        self._stdout_stream = _PrefixedStream(process.stdout, check=self._check_response_wait)
        self._model_load_ms = None
        self._stderr_stop.clear()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(process.stderr,),
            name="hisilicon-worker-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self, pipe: BinaryIO) -> None:
        try:
            try:
                descriptor = pipe.fileno()
            except (AttributeError, io.UnsupportedOperation):
                descriptor = None
            poller = select.poll()
            if descriptor is not None:
                poller.register(descriptor, select.POLLIN)
            pending = bytearray()
            while not self._stderr_stop.is_set():
                if descriptor is not None and not poller.poll(POLL_INTERVAL_SECONDS * 1000):
                    continue
                chunk = (
                    pipe.read(_RESPONSE_READ_CHUNK) if descriptor is None else os.read(descriptor, _RESPONSE_READ_CHUNK)
                )
                pending.extend(chunk)
                lines = pending.split(b"\n")
                pending = bytearray(lines.pop())
                if not chunk or len(pending) >= _RESPONSE_READ_CHUNK:
                    lines.append(bytes(pending))
                    pending.clear()
                for line in lines:
                    decoded = line.decode("utf-8", errors="replace").rstrip()
                    if decoded:
                        self._stderr_tail.append(decoded)
                        match = _MODEL_LOAD_MS_RE.search(decoded)
                        if match is not None:
                            with contextlib.suppress(ValueError):
                                self._model_load_ms = float(match.group(1))
                if not chunk:
                    return
        except Exception as exc:  # The diagnostic reader must not terminate the worker.
            LOGGER.debug("Hisilicon worker stderr reader stopped: %s", exc)

    def _write_request_locked(self, input_arrays: Sequence[np.ndarray]) -> int:
        process = self._require_running_process()
        assert process.stdin is not None
        request_id = self._request_id + 1
        try:
            header = REQUEST_HEADER_STRUCT.pack(PROTOCOL_MAGIC, PROTOCOL_VERSION, len(input_arrays), request_id, 0)
            payloads = tuple(
                (INPUT_ENTRY_STRUCT.pack(index, array.nbytes, 0), memoryview(array).cast("B") if array.nbytes else b"")
                for index, array in enumerate(input_arrays)
            )
        except Exception as exc:
            exc.evidence = OutcomeEvidence.not_started("worker_inputs")
            raise
        self._request_id = request_id
        try:
            self._write_all(process.stdin, header, request_start=True)
            for entry, payload in payloads:
                self._write_all(process.stdin, entry)
                self._write_all(process.stdin, payload)
        except (BrokenPipeError, OSError) as exc:
            self._closing = True
            raise SD3403WorkerExitedError(self._worker_exit_message()) from exc
        except BaseException as exc:
            # A partial request cannot be retried on this byte stream. Keep the
            # existing process owner for Close to terminate and reap it.
            evidence = getattr(exc, "evidence", None)
            if evidence is None or evidence.state is not OutcomeState.NOT_STARTED:
                self._closing = True
            raise
        return request_id

    def _write_all(self, stream: BinaryIO, data, *, request_start=False) -> None:
        try:
            descriptor = stream.fileno()
        except (AttributeError, io.UnsupportedOperation):
            descriptor = None
        poller = select.poll()
        if descriptor is not None:
            poller.register(descriptor, select.POLLOUT)
        remaining = memoryview(data)
        try:
            while remaining:
                self._check_io_wait("worker_request")
                if descriptor is not None:
                    if not poller.poll(POLL_INTERVAL_SECONDS * 1000):
                        continue
                    self._check_io_wait("worker_request")
                try:
                    written = (
                        os.write(descriptor, remaining[:_PIPE_WRITE_CHUNK_BYTES])
                        if descriptor is not None
                        else stream.write(remaining)
                    )
                except BlockingIOError:
                    continue
                if not written:
                    raise SD3403ProtocolError("worker request pipe stopped accepting input")
                remaining = remaining[written:]
        except (DeadlineExceeded, CancellationRequested) as exc:
            if request_start and len(remaining) == len(data):
                exc.evidence = OutcomeEvidence.not_started(exc.phase)
            raise

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
        # stderr is polled by a stoppable reader; an inherited writer must not
        # hold Close hostage after the owned worker has been reaped.
        for stream in (process.stdin, process.stdout):
            with contextlib.suppress(Exception):
                if stream is not None:
                    stream.close()
        self._wait_process(process, self._graceful_close_timeout)
        if process.poll() is None and self._force_close:
            check_lifecycle()
            with contextlib.suppress(Exception):
                process.terminate()
            self._wait_process(process, WORKER_TERMINATE_TIMEOUT)
            if process.poll() is None:
                check_lifecycle()
                with contextlib.suppress(Exception):
                    process.kill()
                self._wait_process(process, WORKER_KILL_TIMEOUT)
        if process.poll() is None:
            mode = "after EOF, SIGTERM, and SIGKILL" if self._force_close else "after EOF with force_close disabled"
            raise SD3403ProtocolError(
                f"Hisilicon worker did not exit {mode}; pid={getattr(process, 'pid', '<unknown>')}"
            )

    @staticmethod
    def _wait_process(process: _Process, timeout: float) -> None:
        deadline = Deadline.after(timeout)
        while process.poll() is None:
            check_lifecycle()
            if deadline.expired:
                return
            try:
                process.wait(timeout=min(POLL_INTERVAL_SECONDS, lifecycle_remaining(deadline.remaining_seconds())))
            except subprocess.TimeoutExpired:
                continue
