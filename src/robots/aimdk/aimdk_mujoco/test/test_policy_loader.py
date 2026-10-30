"""Bounded, single-flight loader lifecycle without ROS or policy assets."""

import subprocess
import sys
import threading
from pathlib import Path

import pytest
from aimdk_mujoco.policy_loader import PolicyLoader


def test_loader_returns_candidate_and_propagates_preparation_error():
    loader = PolicyLoader()
    candidate = object()
    try:
        assert loader.submit(lambda value: value, candidate).result(timeout=5) is candidate
        # The result may be published just before the worker thread returns.
        assert loader.close(timeout=5)
    finally:
        loader.close(timeout=5)
    with pytest.raises(RuntimeError, match="closed"):
        loader.submit(lambda: None)

    loader = PolicyLoader()
    try:

        def fail():
            raise ValueError("invalid policy")

        with pytest.raises(ValueError, match="invalid policy"):
            loader.submit(fail).result(timeout=5)
    finally:
        assert loader.close(timeout=5)


def test_running_cancel_cannot_admit_overlapping_load_and_close_is_bounded():
    loader = PolicyLoader()
    started, release = threading.Event(), threading.Event()

    def prepare():
        started.set()
        assert release.wait(5)
        return "candidate"

    try:
        pending = loader.submit(prepare)
        assert started.wait(5)
        assert not pending.cancel()
        assert loader.busy
        with pytest.raises(RuntimeError, match="busy"):
            loader.submit(lambda: None)
        assert not loader.close(timeout=0.01)
        with pytest.raises(RuntimeError, match="closed"):
            loader.submit(lambda: None)
        release.set()
        assert pending.result(timeout=5) == "candidate"
        assert loader.close(timeout=5)
        assert not loader.busy
    finally:
        release.set()
        assert loader.close(timeout=5)


def test_canceled_pending_future_still_blocks_admission_until_worker_exits(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    run = PolicyLoader._run
    prepared = []

    def delayed_run(*args):
        entered.set()
        assert release.wait(5)
        run(*args)

    monkeypatch.setattr(PolicyLoader, "_run", staticmethod(delayed_run))
    loader = PolicyLoader()
    try:
        pending = loader.submit(lambda: prepared.append("candidate"))
        assert entered.wait(5)
        assert pending.cancel() and pending.done()
        assert loader.busy
        with pytest.raises(RuntimeError, match="busy"):
            loader.submit(lambda: prepared.append("overlapping candidate"))
        assert not loader.close(timeout=0.01)
        release.set()
        assert loader.close(timeout=5)
        assert pending.cancelled() and not prepared
    finally:
        release.set()
        assert loader.close(timeout=5)


def test_sleeping_loader_does_not_join_at_interpreter_exit():
    script = """
import threading
import time
from aimdk_mujoco.policy_loader import PolicyLoader

started = threading.Event()
def prepare():
    started.set()
    time.sleep(60)

loader = PolicyLoader()
future = loader.submit(prepare)
assert started.wait(2)
assert not future.cancel()
start = time.perf_counter()
assert not loader.close()
assert time.perf_counter() - start < 1.5
assert loader.busy and not future.done()
print("unfinished daemon abandoned", flush=True)
"""
    # A fresh interpreter must actually exit, not just return from close().
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    assert result.stdout.strip() == "unfinished daemon abandoned"
