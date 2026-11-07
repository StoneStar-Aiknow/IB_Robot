"""Behavioral tests for client-side RT-Infer periodic publishing."""

from __future__ import annotations

import ctypes
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_service.model_sessions.ascend import AscendOmModelSession
from inference_service.rt_infer_periodic import (
    RELEASE_MONOTONIC_NS_KEY,
    PeriodicAdmission,
    PeriodicAdmissionError,
)
from inference_service.unified_runtime import ModelRequest

_STUB_SOURCE = r"""
#include <stdint.h>
#include <stdlib.h>

static int armed = 0;
static int release_count = 0;
static int disarm_count = 0;
static int close_count = 0;
static int arm_count = 0;
static uint64_t epoch_rec = 0;
static uint64_t last_release = 0;

int stub_release_count(void) { return release_count; }
int stub_disarm_count(void) { return disarm_count; }
int stub_close_count(void) { return close_count; }
int stub_arm_count(void) { return arm_count; }
uint64_t stub_last_release(void) { return last_release; }

void *periodic_shared_create(const char *path) { (void)path; return (void *)1; }
void *periodic_shared_open(const char *path, int mode) {
    (void)path; (void)mode; return (void *)1;
}
int periodic_shared_arm(void *h, uint64_t epoch, uint64_t period, uint64_t budget) {
    if (arm_count) return -114;
    (void)h; (void)period; (void)budget; armed = 1; arm_count += 1; epoch_rec = epoch; return 0;
}
int periodic_shared_release(void *h, uint64_t ns) {
    (void)h;
    if (!armed || ns < epoch_rec || ns < last_release) return -22;
    release_count += 1; last_release = ns; return 0;
}
int periodic_shared_disarm(void *h) { (void)h; disarm_count += 1; armed = 0; return 0; }
void periodic_shared_close(void *h) { (void)h; close_count += 1; }
"""


def _start_ticks(pid: int) -> str:
    return (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[19]


@pytest.fixture()
def owner_session(tmp_path, monkeypatch, request):
    prefix = tmp_path / "prefix"
    library_dir = prefix / "lib"
    library_dir.mkdir(parents=True)
    source = library_dir / "stub.c"
    source.write_text(_STUB_SOURCE, encoding="utf-8")
    bridge = library_dir / "periodic_control_bridge.so"
    if os.system(f"gcc -shared -fPIC -o {bridge} {source}") != 0:
        pytest.skip("gcc unavailable")

    library = ctypes.CDLL(str(bridge))
    library.periodic_shared_create.restype = ctypes.c_void_p
    library.periodic_shared_create.argtypes = [ctypes.c_char_p]
    library.periodic_shared_arm.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
    control = tmp_path / "control.bin"
    handle = library.periodic_shared_create(str(control).encode())
    period_hz = 18
    period_ns = (1_000_000_000 + period_hz - 1) // period_hz
    budget_ns = 38_000_000
    epoch_ns = time.monotonic_ns()
    unarmed = getattr(request, "param", None) == "unarmed"
    if not unarmed:
        assert library.periodic_shared_arm(handle, epoch_ns, period_ns, budget_ns) == 0

    identity = {"pid": os.getpid(), "start_ticks": _start_ticks(os.getpid())}
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({"status": "running", "prefix": str(prefix), "scheduler": identity, "boot_id": boot_id}),
        encoding="utf-8",
    )
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    token = "test-launch-token"
    session = tmp_path / "periodic-session.json"
    session.write_text(
        json.dumps(
            {
                "boot_id": boot_id,
                "prefix": str(prefix),
                "control_path": str(control),
                "launch_token": token,
                "scheduler": identity,
                "owner": identity,
                "epoch_ns": 0 if unarmed else epoch_ns,
                "phase": "initializing" if unarmed else "active",
                "expected_roles": ["pear", "tts"],
                "period_ns": period_ns,
                "budget_ns": budget_ns,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RTINFER_PREFIX", str(prefix))
    monkeypatch.setenv("RTINFER_LAUNCH_TOKEN", token)
    monkeypatch.setenv("RTINFER_STATE_FILE", str(state))
    monkeypatch.setenv("RT_INFER_PERIODIC_PATH", str(control))
    monkeypatch.setenv("RT_INFER_PERIODIC_SESSION", str(session))
    monkeypatch.setenv("RT_INFER_PERIODIC_PERIOD_HZ", str(period_hz))
    monkeypatch.setenv("RT_INFER_PERIODIC_BUDGET_NS", str(budget_ns))
    yield {
        "library": library,
        "epoch_ns": epoch_ns,
        "session": session,
        "state": state,
        "prefix": prefix,
        "control": control,
        "identity": identity,
        "token": token,
        "boot": boot_id,
    }


def _counter(library, name: str) -> int:
    function = getattr(library, name)
    function.restype = ctypes.c_uint64 if name == "stub_last_release" else ctypes.c_int
    return int(function())


def test_explicit_release_metadata_is_published_unchanged(owner_session):
    admission = PeriodicAdmission.from_launch_environment()
    admission.arm()
    release = owner_session["epoch_ns"] + 123
    assert admission.publish_release({RELEASE_MONOTONIC_NS_KEY: release}) == release
    assert _counter(owner_session["library"], "stub_last_release") == release
    admission.close()


def test_missing_release_metadata_uses_current_monotonic_time_without_sleep(owner_session):
    admission = PeriodicAdmission.from_launch_environment()
    admission.arm()
    before = time.monotonic_ns()
    published = admission.publish_release({})
    after = time.monotonic_ns()
    assert before <= published <= after
    assert after - before < 50_000_000
    admission.close()


def test_future_owner_epoch_skips_cold_start_without_client_rearm(owner_session, monkeypatch):
    document = json.loads(owner_session["session"].read_text(encoding="utf-8"))
    future = document["epoch_ns"] + 60_000_000_000
    document["epoch_ns"] = future
    owner_session["session"].write_text(json.dumps(document), encoding="utf-8")
    admission = PeriodicAdmission.from_launch_environment()
    admission.arm()
    before_releases = _counter(owner_session["library"], "stub_release_count")
    before_arms = _counter(owner_session["library"], "stub_arm_count")
    monkeypatch.setattr("inference_service.rt_infer_periodic.time.monotonic_ns", lambda: future - 1)
    assert admission.publish_release({RELEASE_MONOTONIC_NS_KEY: future - 1}) == future - 1
    assert _counter(owner_session["library"], "stub_release_count") == before_releases
    monkeypatch.setattr("inference_service.rt_infer_periodic.time.monotonic_ns", lambda: future + 1)
    assert admission.publish_release({RELEASE_MONOTONIC_NS_KEY: future + 1}) == future + 1
    assert _counter(owner_session["library"], "stub_arm_count") == before_arms
    assert _counter(owner_session["library"], "stub_release_count") == before_releases + 1
    admission.close()


def test_client_close_does_not_disarm_and_next_client_can_publish(owner_session):
    first = PeriodicAdmission.from_launch_environment()
    first.arm()
    first.publish_release({RELEASE_MONOTONIC_NS_KEY: owner_session["epoch_ns"] + 1})
    first.close()

    second = PeriodicAdmission.from_launch_environment()
    second.arm()
    second.publish_release({RELEASE_MONOTONIC_NS_KEY: owner_session["epoch_ns"] + 2})
    second.close()

    library = owner_session["library"]
    assert _counter(library, "stub_release_count") == 2
    assert _counter(library, "stub_disarm_count") == 0
    assert _counter(library, "stub_close_count") == 2


def test_sparse_and_future_off_grid_releases_are_not_rewritten(owner_session):
    admission = PeriodicAdmission.from_launch_environment()
    admission.arm()
    future_off_grid = owner_session["epoch_ns"] + 60_000_000_001
    assert admission.publish_release({RELEASE_MONOTONIC_NS_KEY: future_off_grid}) == future_off_grid
    admission.close()


def test_decreasing_or_invalid_release_is_rejected(owner_session):
    admission = PeriodicAdmission.from_launch_environment()
    admission.arm()
    release = owner_session["epoch_ns"] + 10
    admission.publish_release({RELEASE_MONOTONIC_NS_KEY: release})
    with pytest.raises(PeriodicAdmissionError, match="nondecreasing"):
        admission.publish_release({RELEASE_MONOTONIC_NS_KEY: release - 1})
    with pytest.raises(PeriodicAdmissionError, match="positive integer"):
        admission.publish_release({RELEASE_MONOTONIC_NS_KEY: "bad"})
    admission.close()


def test_session_identity_mismatch_fails_before_open(owner_session):
    document = json.loads(owner_session["session"].read_text(encoding="utf-8"))
    document["scheduler"] = {"pid": 999999, "start_ticks": "1"}
    owner_session["session"].write_text(json.dumps(document), encoding="utf-8")
    admission = PeriodicAdmission.from_launch_environment()
    with pytest.raises(PeriodicAdmissionError, match="scheduler identity"):
        admission.arm()


def test_missing_or_corrupt_live_state_fails_closed(owner_session):
    owner_session["state"].write_text("not-json", encoding="utf-8")
    admission = PeriodicAdmission.from_launch_environment()
    with pytest.raises(PeriodicAdmissionError, match="vendor state"):
        admission.arm()


def test_no_environment_means_native_inert(monkeypatch):
    monkeypatch.delenv("RT_INFER_PERIODIC_PATH", raising=False)
    assert PeriodicAdmission.from_launch_environment() is None


def test_missing_prefix_is_a_hard_error(tmp_path, monkeypatch):
    monkeypatch.setenv("RT_INFER_PERIODIC_PATH", str(tmp_path / "control.bin"))
    monkeypatch.delenv("RTINFER_PREFIX", raising=False)
    with pytest.raises(PeriodicAdmissionError, match="RTINFER_PREFIX"):
        PeriodicAdmission.from_launch_environment()


def test_model_session_checks_deadline_or_cancellation_before_publish(monkeypatch):
    class Admission:
        calls = []

        def arm(self):
            self.calls.append("arm")

        def publish_release(self, metadata, **kwargs):
            self.calls.append(("publish", metadata))

    class RejectedContext:
        def check(self, phase):
            raise RuntimeError(f"rejected at {phase}")

    admission = Admission()
    session = AscendOmModelSession(runtime_manager=object())
    monkeypatch.setattr(session, "_loaded_deployment", lambda: SimpleNamespace(execution=()))
    monkeypatch.setattr(session, "_periodic_admission_for", lambda _deployment: admission)
    with pytest.raises(RuntimeError, match="rejected at backend"):
        session._execute(ModelRequest(metadata={RELEASE_MONOTONIC_NS_KEY: 123}), RejectedContext())
    assert admission.calls == []


def test_model_session_forwards_release_metadata_without_rewriting(monkeypatch):
    class Admission:
        calls = []

        def arm(self):
            self.calls.append("arm")

        def publish_release(self, metadata, **kwargs):
            self.calls.append(("publish", metadata))

    admission = Admission()
    session = AscendOmModelSession(runtime_manager=object())
    monkeypatch.setattr(session, "_loaded_deployment", lambda: SimpleNamespace(execution=()))
    monkeypatch.setattr(session, "_periodic_admission_for", lambda _deployment: admission)
    metadata = {RELEASE_MONOTONIC_NS_KEY: 123}
    result = session._execute(ModelRequest(metadata=metadata), SimpleNamespace(check=lambda _phase: None))
    assert result == {}
    assert admission.calls == ["arm", ("publish", metadata)]


def test_close_real_session_then_publish_from_replacement(owner_session, monkeypatch):
    for index in (1, 2):
        session = AscendOmModelSession(runtime_manager=object())
        admission = PeriodicAdmission.from_launch_environment()
        session._periodic_admission = admission
        monkeypatch.setattr(session, "_loaded_deployment", lambda: SimpleNamespace(execution=()))
        session._execute(
            ModelRequest(metadata={RELEASE_MONOTONIC_NS_KEY: owner_session["epoch_ns"] + index}),
            SimpleNamespace(check=lambda _phase: None),
        )
        session.close()
    assert _counter(owner_session["library"], "stub_release_count") == 2
    assert _counter(owner_session["library"], "stub_arm_count") == 1
    assert _counter(owner_session["library"], "stub_disarm_count") == 0


def test_cancel_during_mapping_open_prevents_publication(monkeypatch):
    from inference_service.unified_runtime import ExecutionContext

    context = ExecutionContext("cancel-during-open")
    admission = SimpleNamespace(
        arm=lambda: context.cancellation_token.cancel(),
        publish_release=lambda *_args, **_kw: pytest.fail("cancelled request published"),
    )
    session = AscendOmModelSession(runtime_manager=object())
    monkeypatch.setattr(session, "_loaded_deployment", lambda: SimpleNamespace(execution=()))
    monkeypatch.setattr(session, "_periodic_admission_for", lambda _deployment: admission)
    with pytest.raises(Exception, match="cancel"):
        session._execute(ModelRequest(), context)


def test_mapping_close_failure_still_releases_model_and_lease():
    closed = []
    session = AscendOmModelSession(runtime_manager=object())
    session._models = {"pear": SimpleNamespace(close=lambda: closed.append("model"))}
    session._lease = SimpleNamespace(close=lambda: closed.append("lease"))
    session._periodic_admission = SimpleNamespace(close=lambda: (_ for _ in ()).throw(RuntimeError("mapping close")))
    with pytest.raises(Exception, match="mapping close"):
        session.close()
    assert closed == ["model", "lease"]


@pytest.mark.parametrize("owner_session", ["unarmed"], indirect=True)
def test_owner_waits_for_bound_readiness_then_arms_once_and_drains(owner_session, monkeypatch):
    import subprocess

    from robot_config import rt_infer_scheduler_supervisor as coordinator

    root = owner_session["state"].parent
    monkeypatch.setattr(coordinator, "STATE_FILE", owner_session["state"])
    (root / "launch-owner.lock").write_text(
        json.dumps(
            {
                "identity": owner_session["identity"],
                "token": owner_session["token"],
                "boot_id": owner_session["boot"],
            }
        )
    )
    children = [subprocess.Popen(["sleep", "30"]) for _ in range(2)]
    try:
        state = json.loads(owner_session["state"].read_text())
        state["clients"] = [{"pid": child.pid, "start_ticks": _start_ticks(child.pid)} for child in children]
        owner_session["state"].write_text(json.dumps(state))
        for role, client in zip(("pear", "tts"), state["clients"], strict=True):
            (root / f"client-{client['pid']}.token").write_text(
                json.dumps(
                    {
                        **client,
                        "role": role,
                        "ready": role == "pear",
                        "token": owner_session["token"],
                        "boot_id": owner_session["boot"],
                    }
                )
            )
        arguments = (
            str(owner_session["control"]),
            owner_session["token"],
            str(owner_session["prefix"]),
            owner_session["identity"],
            owner_session["boot"],
        )
        admission = PeriodicAdmission.from_launch_environment()
        admission.arm()
        # Initialization executes without a reservation and without a client arm.
        admission.publish_release({})
        assert _counter(owner_session["library"], "stub_arm_count") == 0
        assert _counter(owner_session["library"], "stub_release_count") == 0
        assert coordinator._coordinate_periodic(*arguments) is False
        path = root / f"client-{children[1].pid}.token"
        record = json.loads(path.read_text())
        record.update(ready=True, token="foreign")
        path.write_text(json.dumps(record))
        assert coordinator._coordinate_periodic(*arguments) is False
        record["token"] = owner_session["token"]
        path.write_text(json.dumps(record))
        assert coordinator._coordinate_periodic(*arguments) is True
        assert coordinator._coordinate_periodic(*arguments) is False
        assert _counter(owner_session["library"], "stub_arm_count") == 1
        epoch = json.loads(owner_session["session"].read_text())["epoch_ns"]
        monkeypatch.setattr("inference_service.rt_infer_periodic.time.monotonic_ns", lambda: epoch + 10)
        assert admission.publish_release({RELEASE_MONOTONIC_NS_KEY: epoch + 10}) == epoch + 10
        assert coordinator._coordinate_periodic(*arguments, stopping=True) is True
        admission.publish_release({RELEASE_MONOTONIC_NS_KEY: epoch + 20})
        assert _counter(owner_session["library"], "stub_release_count") == 1
        assert _counter(owner_session["library"], "stub_disarm_count") == 1
        admission.close()
    finally:
        for child in children:
            child.terminate()
            child.wait()
