"""Bounded, single-flight loader lifecycle without ROS or policy assets."""

import threading

import pytest
from aimdk_mujoco.policy_loader import PolicyLoader


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


def test_canceled_pending_load_keeps_admission_closed_until_worker_exits(monkeypatch):
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
