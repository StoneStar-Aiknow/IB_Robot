"""Provenance setup must not turn observation calls into initialization waits."""

import io
import json
import logging
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ibrobot_tracing import instrumentation


class Collector(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(json.loads(record.getMessage().removeprefix("IBTRACE1 ")))


@pytest.fixture(autouse=True)
def fresh_identity(monkeypatch):
    monkeypatch.setattr(instrumentation, "_provenance_state", None)
    monkeypatch.setattr(instrumentation, "_provenance_lock", threading.Lock())


def logger_and_records():
    logger = logging.Logger("provenance-test", logging.INFO)
    sink = Collector()
    logger.addHandler(sink)
    return logger, sink.records


def test_disabled_construction_and_calls_do_not_prepare_or_read_identity(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("disabled tracing attempted provenance work")

    monkeypatch.setattr(instrumentation, "_prepare_process_provenance", forbidden)
    monkeypatch.setattr(instrumentation, "_process_provenance", forbidden)
    monkeypatch.setattr(instrumentation.Path, "open", forbidden)
    monkeypatch.setattr(instrumentation.socket, "gethostname", forbidden)
    monkeypatch.setattr(instrumentation.uuid, "uuid4", forbidden)
    monkeypatch.setenv("IB_TRACE_ENABLED", "0")
    trace = instrumentation.get_trace_emitter("ib_trace.disabled")
    with trace.trace_context("request"), trace.span("work"):
        trace.event("event")
    assert trace.start_span("work") is None
    trace.end_span(None)
    assert trace.logger is None


def test_hot_path_uses_prepared_identity_without_io_or_lock(monkeypatch):
    logger, rows = logger_and_records()
    trace = instrumentation.TraceEmitter(logger)
    assert instrumentation._process_provenance(os.getpid()) is not None

    def forbidden(*args, **kwargs):
        pytest.fail("hot path attempted identity initialization")

    class ForbiddenLock:
        acquire = release = forbidden

    monkeypatch.setattr(instrumentation, "_provenance_lock", ForbiddenLock())
    monkeypatch.setattr(instrumentation, "_load_process_provenance", forbidden)
    monkeypatch.setattr(instrumentation.Path, "open", forbidden)
    monkeypatch.setattr(instrumentation.socket, "gethostname", forbidden)
    with trace.trace_context("business-request"), trace.span("work"):
        trace.event("one", machine_id="business attribute")
        trace.event("two")
    assert [row["event"] for row in rows] == ["span_begin", "one", "two", "span_end"]
    assert [row["provenance"]["event_sequence"] for row in rows] == list(range(4))
    assert all(row["fields"]["request_id"] == "business-request" for row in rows)
    assert rows[1]["fields"]["machine_id"] == "business attribute"
    assert rows[1]["provenance"]["machine_id"] != "business attribute"
    assert instrumentation._process_provenance(os.getpid() + 1) is None


def test_busy_initialization_does_not_hold_up_other_constructor_or_event(monkeypatch):
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    original = instrumentation._load_process_provenance
    logger, rows = logger_and_records()
    calls, errors, emitters = [], [], []

    def slow_load(pid):
        calls.append(pid)
        entered.set()
        assert release.wait(5), "test did not release setup"
        return original(pid)

    monkeypatch.setattr(instrumentation, "_load_process_provenance", slow_load)

    def construct(name):
        try:
            trace = instrumentation.TraceEmitter(logger)
            emitters.append(trace)
            trace.event(name)
        except BaseException as error:
            errors.append(error)
        finally:
            if name == "while-busy":
                completed.set()

    first = threading.Thread(target=construct, args=("initialized",), daemon=True)
    second = threading.Thread(target=construct, args=("while-busy",), daemon=True)
    first.start()
    try:
        assert entered.wait(2)
        second.start()
        # Complete the second call while setup is still paused, rather than
        # relying on a machine-dependent microsecond performance threshold.
        assert completed.wait(2), "another caller waited on identity setup"
        assert not release.is_set()
        assert rows[0]["event"] == "while-busy" and "provenance" not in rows[0]
    finally:
        release.set()
        first.join(3)
        if second.ident is not None:
            second.join(3)
    assert not first.is_alive() and not second.is_alive() and not errors
    for trace in emitters:
        trace.event("after-setup")
    assert calls == [os.getpid()]
    assert len({row["provenance"]["process_instance"] for row in rows[1:]}) == 1
    assert [row["provenance"]["event_sequence"] for row in rows[1:]] == [0, 1, 2]


def test_concurrent_emitters_share_one_ready_identity_and_unique_sequence():
    logger, rows = logger_and_records()
    instrumentation.TraceEmitter(logger)
    barrier = threading.Barrier(4)

    def emit(_):
        trace = instrumentation.TraceEmitter(logger)
        barrier.wait(timeout=2)
        for _ in range(25):
            trace.event("concurrent")

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(emit, range(4)))
    assert len(rows) == 100
    assert len({row["provenance"]["process_instance"] for row in rows}) == 1
    assert sorted(row["provenance"]["event_sequence"] for row in rows) == list(range(100))


def test_setup_failure_keeps_local_records_and_only_explicit_setup_retries(monkeypatch):
    logger, rows = logger_and_records()
    original = instrumentation._load_process_provenance
    attempts = []

    def failed_load(pid):
        attempts.append(pid)
        raise OSError("identity unavailable")

    monkeypatch.setattr(instrumentation, "_load_process_provenance", failed_load)
    trace = instrumentation.TraceEmitter(logger)
    error = ValueError("business error")
    with pytest.raises(ValueError) as caught, trace.trace_context("request"), trace.span("work"):
        trace.event("event")
        raise error
    assert caught.value is error
    assert len(rows) == 3 and all("provenance" not in row for row in rows)
    assert rows[-1]["fields"]["status"] == "error"
    assert attempts == [os.getpid()]
    monkeypatch.setattr(instrumentation, "_load_process_provenance", original)
    instrumentation.TraceEmitter(logger)
    trace.event("after-explicit-setup")
    assert rows[-1]["provenance"]["process_id"] == os.getpid()


@pytest.mark.parametrize("session", ["", " ", " padded", "x" * 257, "x" * 1025])
def test_invalid_session_does_not_drop_local_records(monkeypatch, session):
    monkeypatch.setenv("IB_TRACE_SESSION_ID", session)
    logger, rows = logger_and_records()
    trace = instrumentation.TraceEmitter(logger)
    with trace.trace_context("business"), trace.span("work"):
        trace.event("local")
    assert len(rows) == 3
    assert all(row["provenance"]["capture_session"] == "" for row in rows)


def test_session_is_fixed_at_process_setup_not_each_event(monkeypatch):
    monkeypatch.setenv("IB_TRACE_SESSION_ID", "x" * 256)
    logger, rows = logger_and_records()
    trace = instrumentation.TraceEmitter(logger)
    monkeypatch.setenv("IB_TRACE_SESSION_ID", "changed-after-setup")
    trace.event("one")
    instrumentation.TraceEmitter(logger).event("two")
    assert [row["provenance"]["capture_session"] for row in rows] == ["x" * 256] * 2


@pytest.mark.parametrize("contents", ["", "x" * 129, OSError("no file"), UnicodeError("invalid encoding")])
def test_unavailable_identity_file_is_cached_as_unknown(monkeypatch, contents):
    reads = []

    def open_identity(path, **kwargs):
        reads.append(str(path))
        if isinstance(contents, Exception):
            raise contents
        return io.StringIO(contents)

    monkeypatch.setattr(instrumentation.Path, "open", open_identity)
    logger, rows = logger_and_records()
    trace = instrumentation.TraceEmitter(logger)
    trace.event("one")
    trace.event("two")
    assert len(reads) == 2
    assert len(rows) == 2
    assert all(row["provenance"]["machine_id"] == row["provenance"]["boot_id"] == "" for row in rows)


def run_isolated(code):
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "IB_TRACE_ENABLED": "0"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires Linux/POSIX fork")
def test_fork_never_reuses_parent_identity_or_initializes_during_emission():
    run_isolated("""
import json, logging, os, signal, traceback
from ibrobot_tracing import instrumentation as m
rows = []
class Sink(logging.Handler):
    def emit(self, record):
        rows.append(json.loads(record.getMessage().removeprefix("IBTRACE1 ")))
logger = logging.Logger("fork", logging.INFO)
logger.addHandler(Sink())
trace = m.TraceEmitter(logger)
trace.event("parent-before")
parent_id = rows[-1]["provenance"]["process_instance"]
m._provenance_lock.acquire()
pid = os.fork()
if pid == 0:
    signal.alarm(4)
    try:
        original = m._load_process_provenance
        unexpected_calls = []
        def forbidden(*args):
            unexpected_calls.append(args)
            raise AssertionError("inherited emission attempted setup")
        m._load_process_provenance = forbidden
        trace.event("child-before-setup")
        assert rows[-1]["event"] == "child-before-setup"
        assert "provenance" not in rows[-1]
        assert m._process_provenance(os.getpid()) is None
        m.TraceEmitter(logger, enabled=False)
        assert unexpected_calls == [], "inherited/disabled emitter attempted setup"
        assert m._process_provenance(os.getpid()) is None
        m._load_process_provenance = original
        other = m.TraceEmitter(logger)
        other.event("child-ready")
        trace.event("child-inherited-ready")
        child = rows[-2:]
        assert all(r["provenance"]["process_id"] == os.getpid() for r in child)
        assert all(r["provenance"]["process_instance"] != parent_id for r in child)
        assert child[0]["provenance"]["process_instance"] == child[1]["provenance"]["process_instance"]
        assert [r["provenance"]["event_sequence"] for r in child] == [0, 1]
    except BaseException:
        traceback.print_exc()
        os._exit(1)
    os._exit(0)
try:
    trace.event("parent-after")
finally:
    m._provenance_lock.release()
assert os.waitpid(pid, 0)[1] == 0
assert rows[-1]["provenance"]["process_instance"] == parent_id
assert rows[-1]["provenance"]["event_sequence"] == 1
""")


@pytest.mark.skipif(sys.platform != "linux", reason="uses SIGUSR1")
def test_signal_reentrant_setup_and_emission_fall_back_without_waiting():
    run_isolated("""
import json, logging, os, signal
from ibrobot_tracing import instrumentation as m
rows, calls = [], []
class Sink(logging.Handler):
    def emit(self, record):
        rows.append(json.loads(record.getMessage().removeprefix("IBTRACE1 ")))
logger = logging.Logger("reentrant", logging.INFO)
logger.addHandler(Sink())
original = m._load_process_provenance
def handler(signum, frame):
    m.TraceEmitter(logger).event("reentrant")
def load(pid):
    calls.append(pid)
    os.kill(os.getpid(), signal.SIGUSR1)
    return original(pid)
previous = signal.signal(signal.SIGUSR1, handler)
try:
    m._load_process_provenance = load
    trace = m.TraceEmitter(logger)
    trace.event("ready")
finally:
    signal.signal(signal.SIGUSR1, previous)
assert calls == [os.getpid()]
assert [r["event"] for r in rows] == ["reentrant", "ready"]
assert "provenance" not in rows[0]
assert rows[1]["provenance"]["event_sequence"] == 0
""")
