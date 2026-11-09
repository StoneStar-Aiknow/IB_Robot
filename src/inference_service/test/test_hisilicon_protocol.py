from __future__ import annotations

import io
import subprocess

import numpy as np
import pytest

from inference_service.backends.hisilicon import sd3403_protocol as protocol_module
from inference_service.backends.hisilicon.sd3403_protocol import (
    DIM_STRUCT,
    INPUT_ENTRY_STRUCT,
    OUTPUT_ENTRY_STRUCT,
    PROTOCOL_MAGIC,
    PROTOCOL_VERSION,
    REQUEST_HEADER_STRUCT,
    RESPONSE_HEADER_STRUCT,
    WORKER_ELEM_FLOAT32,
    WORKER_STATUS_OK,
    SD3403Protocol,
    SD3403ProtocolError,
    SD3403WorkerError,
    SD3403WorkerExitedError,
)


class FakeProcess:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode: int | None = None
        self.pid = 1234
        self.terminated = False
        self.killed = False
        self.wait_timeouts: list[float | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class NonExitingProcess(FakeProcess):
    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        raise subprocess.TimeoutExpired("worker", timeout)

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def test_close_cancellation_after_process_wait_starts_retains_worker_for_retry(tmp_path, monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from inference_service.unified_runtime import ExecutionContext
    from inference_service.unified_runtime.errors import CancellationRequested
    from inference_service.unified_runtime.lifecycle import POLL_INTERVAL_SECONDS, lifecycle_scope

    waiting = threading.Event()

    class WaitingProcess(FakeProcess):
        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            waiting.set()
            time.sleep(timeout)
            raise subprocess.TimeoutExpired("worker", timeout)

    process = WaitingProcess()
    protocol, _ = create_protocol(tmp_path, process, monkeypatch)
    protocol.start()
    context = ExecutionContext("close")

    def close():
        with lifecycle_scope(context):
            protocol.close()

    try:
        with ThreadPoolExecutor(1) as pool:
            closing = pool.submit(close)
            assert waiting.wait(1)
            context.token.cancel()
            with pytest.raises(CancellationRequested):
                closing.result(timeout=0.5)
        assert protocol._process is process and protocol._closing
        assert not process.terminated and not process.killed
        assert process.wait_timeouts and max(process.wait_timeouts) <= POLL_INTERVAL_SECONDS
    finally:
        process.returncode = 0
        protocol.close()
    assert protocol._process is None


@pytest.mark.parametrize("stop", ["deadline", "cancel", "close"])
@pytest.mark.parametrize("blocked_phase", ["response", "request"])
def test_live_worker_with_blocked_io_can_be_interrupted_and_reaped(tmp_path, monkeypatch, stop, blocked_phase):
    import sys
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from inference_manifest import ArtifactBindings, TensorBinding
    from inference_service.backends import BackendState
    from inference_service.model_sessions.hisilicon import HisiliconModelSession
    from inference_service.unified_runtime import (
        Deadline,
        ExecutionContext,
        ExecutionFailure,
        ModelRequest,
        ModelRuntimeHandle,
        RecoveryAction,
    )
    from inference_service.unified_runtime.lifecycle import lifecycle_budget

    model = tmp_path / "model.om"
    model.write_bytes(b"om")
    processes = []
    waiting = threading.Event()

    def spawn(*_args, **kwargs):
        # Stop consuming either side of the protocol. A request larger than
        # pipe capacity must stay cancellable even after a partial write.
        # SIGTERM is ignored so force_close must reach SIGKILL.
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "os.write(2, b'R'); "
                + ("os.read(0, 4096); " if blocked_phase == "response" else "")
                + "time.sleep(30)",
            ],
            **kwargs,
        )
        processes.append(child)
        assert child.stderr.read(1) == b"R"  # SIGTERM handler is installed before cancellation/Close races.
        return child

    protocol = SD3403Protocol(tmp_path / "model.om", model, process_factory=spawn, graceful_close_timeout=0)
    context = ExecutionContext("hung", Deadline.after(0.15) if stop == "deadline" else Deadline.unbounded())
    original_check = protocol._check_response_wait

    def check():
        waiting.set()
        original_check()

    if blocked_phase == "response":
        protocol._check_response_wait = check
    else:
        poll = protocol_module.select.poll

        class ObservedPoll:
            def __init__(self):
                self.poller = poll()
                self.writing = False

            def register(self, descriptor, events):
                self.writing = events == protocol_module.select.POLLOUT
                self.poller.register(descriptor, events)

            def poll(self, timeout):
                result = self.poller.poll(timeout)
                if self.writing and not result:
                    waiting.set()
                return result

        monkeypatch.setattr(protocol_module.select, "poll", ObservedPoll)
    session = HisiliconModelSession()
    input_size = 1 if blocked_phase == "response" else 1 << 18
    binding = TensorBinding(semantic="input", index=0, dtype="float32", shape=(input_size,))
    session._context = SimpleNamespace(
        deployment=SimpleNamespace(bindings={"policy": ArtifactBindings(inputs=(binding,), outputs=(binding,))})
    )
    session._state, session._action_binding, session._protocol = BackendState.READY, binding, protocol
    owner = ModelRuntimeHandle(SimpleNamespace(execute=session.execute, close=session.close))
    owner.load()

    try:
        with ThreadPoolExecutor(1) as pool:
            active = pool.submit(owner.execute, ModelRequest({"input": np.ones(input_size, np.float32)}), context)
            assert waiting.wait(1)
            if stop == "cancel":
                owner.cancel(context.request_id)
            elif stop == "close":
                with lifecycle_budget(3):
                    protocol.close()
            with pytest.raises(ExecutionFailure) as caught:
                active.result(timeout=1)
            assert not caught.value.evidence.outcome_known and not owner.ready
            if stop != "close":
                assert caught.value.recovery.action is RecoveryAction.RELOAD
            with lifecycle_budget(3):
                owner.close()
        assert processes[0].poll() == -9 and protocol._process is None
        assert protocol._stderr_thread is None
    finally:
        context.token.cancel()
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=1)
        protocol.close()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), -1.0])
def test_invalid_close_timeout_is_rejected_before_starting_non_exiting_worker(tmp_path, monkeypatch, invalid):
    from inference_service.backends.errors import BackendLoadError
    from inference_service.model_sessions.hisilicon import validate_runtime_options
    from inference_service.unified_runtime.lifecycle import lifecycle_budget

    process = NonExitingProcess()
    with lifecycle_budget(0.1):
        with pytest.raises(ValueError, match="finite non-negative"):
            create_protocol(tmp_path, process, monkeypatch, graceful_close_timeout=invalid)
        with pytest.raises(BackendLoadError, match="finite non-negative"):
            validate_runtime_options({"graceful_close_timeout": invalid})
    assert not process.wait_timeouts and not process.terminated


def response_frame(
    outputs: list[tuple[int, np.ndarray]],
    *,
    request_id: int,
    prefix: bytes = b"",
    status: int = WORKER_STATUS_OK,
    error_code: int = 0,
    error_message: str = "",
) -> bytes:
    message = error_message.encode()
    frame = bytearray(prefix)
    frame.extend(
        RESPONSE_HEADER_STRUCT.pack(
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            status,
            request_id,
            len(outputs),
            1250,
            error_code,
            len(message),
        )
    )
    for output_index, value in outputs:
        array = np.ascontiguousarray(value, dtype=np.float32)
        frame.extend(
            OUTPUT_ENTRY_STRUCT.pack(
                output_index,
                WORKER_ELEM_FLOAT32,
                array.size,
                array.nbytes,
                array.ndim,
                0,
            )
        )
        for dimension in array.shape:
            frame.extend(DIM_STRUCT.pack(dimension))
        frame.extend(memoryview(array).cast("B"))
    frame.extend(message)
    return bytes(frame)


def create_protocol(
    tmp_path,
    process: FakeProcess,
    monkeypatch,
    **protocol_options,
) -> tuple[SD3403Protocol, dict[str, object]]:
    worker = tmp_path / "worker"
    model = tmp_path / "model.om"
    worker.write_text("#!/bin/sh\n", encoding="utf-8")
    worker.chmod(0o755)
    model.write_bytes(b"om")
    for key in protocol_module._LEGACY_WORKER_ENV_KEYS:
        monkeypatch.setenv(key, f"legacy-{key}")
    call: dict[str, object] = {}

    def process_factory(args, **kwargs):
        call["args"] = args
        call["kwargs"] = kwargs
        return process

    return SD3403Protocol(worker, model, process_factory=process_factory, **protocol_options), call


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
@pytest.mark.parametrize("partial", [False, True])
def test_interrupted_response_cannot_send_another_request(tmp_path, monkeypatch, stop, partial):
    from inference_service.unified_runtime.errors import CancellationRequested, DeadlineExceeded

    frame = response_frame([(0, np.ones(1, np.float32))], request_id=1)
    process = FakeProcess(frame + response_frame([(0, np.ones(1, np.float32))], request_id=2))
    protocol, _ = create_protocol(tmp_path, process, monkeypatch)
    protocol.start()
    read = protocol._read_response_locked

    def interrupted(_request_id):
        if partial:
            protocol._stdout_stream.read(RESPONSE_HEADER_STRUCT.size + 1)
        raise (CancellationRequested if stop == "cancel" else DeadlineExceeded)(phase="worker_response")

    monkeypatch.setattr(protocol, "_read_response_locked", interrupted)
    try:
        with pytest.raises(CancellationRequested if stop == "cancel" else DeadlineExceeded):
            protocol.execute((np.ones(1, np.float32),))
        sent = process.stdin.getvalue()
        assert sent
        monkeypatch.setattr(protocol, "_read_response_locked", read)
        with pytest.raises(SD3403ProtocolError, match="irreversible"):
            protocol.execute((np.ones(1, np.float32),))
        assert process.stdin.getvalue() == sent
    finally:
        protocol.close()
    assert process.returncode is not None


@pytest.mark.parametrize("short_writes", [False, True])
def test_protocol_preserves_back_to_back_frames_sparse_outputs_and_request_ids(tmp_path, monkeypatch, short_writes):
    first_action = np.arange(12, dtype=np.float32).reshape(1, 2, 6)
    second_action = np.arange(18, dtype=np.float32).reshape(1, 3, 6)
    process = FakeProcess(
        response_frame([(1, first_action)], request_id=1, prefix=b"load runtime OK\n")
        + response_frame([(1, second_action)], request_id=2)
    )
    if short_writes:

        class ShortWriter(io.BytesIO):
            def write(self, data):
                return super().write(data[:3])

        process.stdin = ShortWriter()
    protocol, call = create_protocol(tmp_path, process, monkeypatch)
    protocol.start()

    first = protocol.execute((np.ones((1, 6), dtype=np.float64),))
    second = protocol.execute((np.zeros((1, 6), dtype=np.float32),))

    assert first.request_id == 1
    assert second.request_id == 2
    assert first.worker_latency_us == 1250
    np.testing.assert_array_equal(first.outputs[1], first_action)
    np.testing.assert_array_equal(second.outputs[1], second_action)
    assert call["args"] == [str(protocol.worker_path), "--model", str(protocol.model_path)]
    environment = call["kwargs"]["env"]
    assert all(key not in environment for key in protocol_module._LEGACY_WORKER_ENV_KEYS)

    request_bytes = process.stdin.getvalue()
    offset = 0
    for request_id in (1, 2):
        magic, version, input_count, actual_request_id, reserved = REQUEST_HEADER_STRUCT.unpack_from(
            request_bytes, offset
        )
        offset += REQUEST_HEADER_STRUCT.size
        assert (magic, version, input_count, actual_request_id, reserved) == (
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            1,
            request_id,
            0,
        )
        input_index, byte_size, reserved = INPUT_ENTRY_STRUCT.unpack_from(request_bytes, offset)
        offset += INPUT_ENTRY_STRUCT.size
        assert input_index == 0
        assert byte_size == 24
        assert reserved == 0
        offset += byte_size
    assert offset == len(request_bytes)
    protocol.close()


def test_protocol_reports_worker_error_and_mismatched_request_id(tmp_path, monkeypatch):
    process = FakeProcess(
        response_frame([], request_id=1, status=1, error_code=7, error_message="extract output failed")
    )
    protocol, _ = create_protocol(tmp_path, process, monkeypatch)
    protocol.start()

    with pytest.raises(SD3403WorkerError, match="extract output failed") as error:
        protocol.execute((np.zeros((1, 6), dtype=np.float32),))
    assert error.value.error_code == 7
    protocol.close()

    mismatched = FakeProcess(response_frame([(1, np.zeros((1, 1, 6)))], request_id=9))
    protocol, _ = create_protocol(tmp_path, mismatched, monkeypatch)
    protocol.start()
    with pytest.raises(SD3403ProtocolError, match="mismatched response id"):
        protocol.execute((np.zeros((1, 6), dtype=np.float32),))
    protocol.close()


def test_protocol_restarts_dead_worker_before_request_without_replaying_partial_io(tmp_path, monkeypatch):
    dead = FakeProcess()
    dead.returncode = 3
    live = FakeProcess(response_frame([(1, np.zeros((1, 1, 6)))], request_id=1))
    processes = iter((dead, live))
    protocol, _ = create_protocol(tmp_path, dead, monkeypatch)
    protocol._process_factory = lambda *args, **kwargs: next(processes)
    protocol.start()

    response = protocol.execute((np.zeros((1, 6), dtype=np.float32),))

    assert response.request_id == 1
    protocol.close()


def test_protocol_maps_stream_loss_to_worker_exit_and_force_closes(tmp_path, monkeypatch):
    process = NonExitingProcess()
    protocol, _ = create_protocol(tmp_path, process, monkeypatch)
    protocol.start()

    with pytest.raises(SD3403WorkerExitedError, match="stream closed"):
        protocol.execute((np.zeros((1, 6), dtype=np.float32),))

    protocol.close()
    protocol.close()
    assert process.terminated is True
    assert process.killed is True


def test_protocol_drains_stderr_and_records_model_load_time(tmp_path, monkeypatch):
    process = FakeProcess(stderr=b"worker booted\nmodel_load_ms=12.75\n")
    protocol, _ = create_protocol(tmp_path, process, monkeypatch)
    protocol.start()
    stderr_thread = protocol._stderr_thread
    assert stderr_thread is not None
    stderr_thread.join(timeout=1.0)

    assert protocol.stderr_tail == ("worker booted", "model_load_ms=12.75")
    assert protocol.model_load_ms == 12.75
    protocol.close()


def test_protocol_retains_non_exiting_worker_and_retries_close(tmp_path, monkeypatch):
    process = NonExitingProcess()
    protocol, _ = create_protocol(
        tmp_path,
        process,
        monkeypatch,
        graceful_close_timeout=0.0,
        force_close=False,
    )
    protocol.start()

    with pytest.raises(SD3403ProtocolError, match="did not exit"):
        protocol.close()
    assert protocol._process is process
    with pytest.raises(SD3403ProtocolError, match="irreversible"):
        protocol.start()
    with pytest.raises(SD3403ProtocolError, match="irreversible"):
        protocol.execute((np.zeros((1, 6), dtype=np.float32),))
    assert process.terminated is False
    assert process.killed is False
    process.returncode = 0
    protocol.close()
    protocol.close()
    assert protocol._process is None


def test_protocol_rejects_unreasonable_output_metadata(tmp_path, monkeypatch):
    frame = bytearray(RESPONSE_HEADER_STRUCT.pack(PROTOCOL_MAGIC, PROTOCOL_VERSION, WORKER_STATUS_OK, 1, 1, 0, 0, 0))
    frame.extend(OUTPUT_ENTRY_STRUCT.pack(1, WORKER_ELEM_FLOAT32, 2, 4, 0, 0))
    process = FakeProcess(bytes(frame))
    protocol, _ = create_protocol(tmp_path, process, monkeypatch)
    protocol.start()

    with pytest.raises(SD3403ProtocolError, match="byte size does not match"):
        protocol.execute((np.zeros((1, 6), dtype=np.float32),))
    protocol.close()


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
@pytest.mark.parametrize("boundary", ["lock", "first_write", "partial_write"])
def test_session_control_before_first_request_byte_keeps_worker_usable(tmp_path, monkeypatch, stop, boundary):
    from contextlib import contextmanager
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    from inference_manifest import ArtifactBindings, TensorBinding
    from inference_service.backends import BackendState
    from inference_service.model_sessions.hisilicon import HisiliconModelSession
    from inference_service.unified_runtime import (
        Deadline,
        ExecutionContext,
        ExecutionFailure,
        ModelRequest,
        ModelRuntimeHandle,
        OutcomeState,
        RecoveryAction,
    )

    next_request = 1 if boundary == "lock" else 2
    process = FakeProcess(response_frame([(0, np.ones(1, np.float32))], request_id=next_request))
    protocol, _ = create_protocol(tmp_path, process, monkeypatch)
    protocol.start()
    now = [datetime(2026, 11, 2, tzinfo=timezone.utc)]
    context = ExecutionContext("interrupted", Deadline.after(1, now=now[0], clock=lambda: now[0]))

    def interrupt():
        if stop == "cancel":
            context.token.cancel()
        else:
            now[0] += timedelta(seconds=2)

    lock = protocol_module.lifecycle_lock
    if boundary == "lock":

        @contextmanager
        def enter(lock_object):
            interrupt()
            with lock(lock_object):
                yield

        monkeypatch.setattr(protocol_module, "lifecycle_lock", enter)
    else:
        check = protocol._check_io_wait

        def check_write(phase):
            if phase == "worker_request" and (boundary == "first_write" or process.stdin.tell() > 0):
                interrupt()
            check(phase)

        monkeypatch.setattr(protocol, "_check_io_wait", check_write)
    binding = TensorBinding(semantic="input", index=0, dtype="float32", shape=(1,))
    output = binding.model_copy(update={"semantic": "action"})
    session = HisiliconModelSession()
    session._state, session._protocol, session._action_binding = BackendState.READY, protocol, output
    session._options = {"perf_enabled": False}
    session._context = SimpleNamespace(
        deployment=SimpleNamespace(bindings={"policy": ArtifactBindings(inputs=(binding,), outputs=(output,))}),
        validated_manifest=SimpleNamespace(manifest=SimpleNamespace(model=SimpleNamespace(outputs=(output,)))),
        target=SimpleNamespace(soc="test"),
        device_id=0,
    )
    owner = ModelRuntimeHandle(SimpleNamespace(execute=session.execute, close=session.close))
    owner.load()
    partial = boundary == "partial_write"
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest({"input": np.ones(1, np.float32)}), context)
        assert caught.value.code == ("request_canceled" if stop == "cancel" else "deadline_exceeded")
        assert caught.value.evidence.state is (OutcomeState.STARTED if partial else OutcomeState.NOT_STARTED)
        assert caught.value.evidence.outcome_known is not partial
        assert caught.value.recovery.action is (RecoveryAction.RELOAD if partial else RecoveryAction.NONE)
        assert bool(process.stdin.getvalue()) is partial
        assert owner.ready is not partial
        if not partial:
            monkeypatch.setattr(protocol_module, "lifecycle_lock", lock)
            if boundary != "lock":
                monkeypatch.setattr(protocol, "_check_io_wait", check)
            result = owner.execute(ModelRequest({"input": np.ones(1, np.float32)}), ExecutionContext("next"))
            np.testing.assert_array_equal(result.outputs["action"], np.ones(1, np.float32))
    finally:
        monkeypatch.setattr(protocol_module, "lifecycle_lock", lock)
        owner.close()


def test_protocol_reads_and_writes_high_numbered_pipe_descriptors(tmp_path, monkeypatch):
    import fcntl
    import os

    process = FakeProcess()
    request_read, request_write = os.pipe()
    response_read, response_write = os.pipe()
    high_write = fcntl.fcntl(request_write, fcntl.F_DUPFD, 1100)
    high_read = fcntl.fcntl(response_read, fcntl.F_DUPFD, 1100)
    os.close(request_write)
    os.close(response_read)
    process.stdin = os.fdopen(high_write, "wb", buffering=0)
    process.stdout = os.fdopen(high_read, "rb", buffering=0)
    protocol, _ = create_protocol(tmp_path, process, monkeypatch)
    try:
        expected = np.ones(1, np.float32)
        os.write(response_write, response_frame([(0, expected)], request_id=1))
        protocol.start()
        result = protocol.execute((expected,))
        np.testing.assert_array_equal(result.outputs[0], expected)
        request = os.read(request_read, 4096)
        assert REQUEST_HEADER_STRUCT.unpack(request[: REQUEST_HEADER_STRUCT.size])[0] == PROTOCOL_MAGIC
    finally:
        protocol.close()
        process.stdin.close()
        process.stdout.close()
        os.close(request_read)
        os.close(response_write)


@pytest.mark.parametrize("bad", [["bad"], [[1], [1, 2]], np.array([object()], dtype=object)])
def test_invalid_later_input_never_writes_request_header(tmp_path, monkeypatch, bad):
    from inference_service.unified_runtime import OutcomeState

    process = FakeProcess(response_frame([(0, np.ones(1, np.float32))], request_id=1))
    protocol, _ = create_protocol(tmp_path, process, monkeypatch)
    protocol.start()
    try:
        with pytest.raises((ValueError, TypeError)) as caught:
            protocol.execute((np.ones(1, np.float32), bad))
        assert caught.value.evidence.state is OutcomeState.NOT_STARTED
        assert process.stdin.getvalue() == b"" and not protocol._closing
        assert protocol.execute((np.ones(1, np.float32),)).request_id == 1
    finally:
        protocol.close()


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 7, 65536])
def test_prefix_scan_preserves_split_magic_and_payload(chunk_size):
    prefix = b"SVP1"
    data = b"noiseS" * 37 + prefix + b"payload"
    stream = protocol_module._PrefixedStream(io.BytesIO(data), chunk_size=chunk_size)
    assert stream.scan_for_prefix(prefix, len(data), 8) == prefix
    assert stream.read(7) == b"payload"


def test_stderr_reader_stops_after_worker_exit_with_an_inherited_writer(tmp_path, monkeypatch):
    import os
    import time

    process = FakeProcess()
    read_fd, inherited_writer = os.pipe()
    process.stderr = os.fdopen(read_fd, "rb", buffering=0)
    protocol, _ = create_protocol(tmp_path, process, monkeypatch)
    protocol.start()
    reader = protocol._stderr_thread
    os.write(inherited_writer, b"partial diagnostic without newline")
    try:
        started = time.monotonic()
        protocol.close()
        assert time.monotonic() - started < 1
        assert not reader.is_alive()
        assert process.poll() is not None and process.stderr.closed
        assert protocol._process is None
    finally:
        os.close(inherited_writer)
        protocol.close()


def test_zero_length_input_has_a_complete_wire_frame(tmp_path, monkeypatch):
    process = FakeProcess(response_frame([(0, np.ones(1, np.float32))], request_id=1))
    protocol, _ = create_protocol(tmp_path, process, monkeypatch)
    protocol.start()
    try:
        protocol.execute((np.zeros((0, 2), dtype=np.float32),))
        assert len(process.stdin.getvalue()) == REQUEST_HEADER_STRUCT.size + INPUT_ENTRY_STRUCT.size
        assert not protocol._closing
    finally:
        protocol.close()
