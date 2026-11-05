"""Tests for the RT-Infer launch owner builder.

Review #3/#4 contracts under test:

* participation and failure semantics: enabled only with a hardware target;
  bring-up failure aborts the launch (never silent native); a non-Ascend
  host aborts BEFORE any vendor CLI call touches the device.
* liveness, not the status string: readiness requires the vendor identity to
  match a live /proc process; a stale "running" file with a dead pid is a
  hard error, not a ready signal.
* ownership: a launch requires exclusive scheduler ownership and refuses a
  foreign running scheduler.
* ordering: stop waits for the vendor client list to drain before the vendor
  stop call.
* the periodic mapping is created BEFORE the vendor start carries
  ``--pfp-periodic``.
* the install step runs ahead of everything (review #5) with the canonical
  prefix and the repository bundle dir.

The vendor CLI is faked; no device, no root.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from launch.actions import EmitEvent, OpaqueFunction
from launch.event_handlers import OnShutdown

from robot_config.launch_builders import rt_infer_scheduler as builder
from robot_config.launch_registry.registry import LaunchBuildContext
from robot_config.runtime_target import RuntimeTarget

_REAL_COVERAGE_PREFLIGHT = builder._coverage_preflight


class _Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture()
def rt_infer_yaml():
    return {
        "rt_infer": {
            "enable": True,
            "bundle_version": "2026.11.05",
            "pear_periodic": {"period_hz": 18, "budget_ns": 38_000_000},
        }
    }


def _context(robot_config, **overrides) -> LaunchBuildContext:
    return LaunchBuildContext(robot_config=robot_config, arguments={}, **overrides)


def _opaque_body(result):
    opaque = next(action for action in result.actions if isinstance(action, OpaqueFunction))
    return opaque._OpaqueFunction__function


def _shutdown_callback(result):
    registered = next(
        action.event_handler
        for action in result.actions
        if isinstance(getattr(action, "event_handler", None), OnShutdown)
    )
    return registered._OnShutdown__on_shutdown


@pytest.fixture()
def ctl(monkeypatch, tmp_path):
    """Fake vendor CLI + state file + live process identity."""
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(builder, "STATE_FILE", state_file)
    monkeypatch.setattr(builder, "STATE_DIR", tmp_path)
    monkeypatch.setattr(builder, "OWNER_LOCK", tmp_path / "launch-owner.lock")
    calls = []
    results: dict[str, _Completed] = {}

    def _fake(script, prefix, arguments, *, timeout):
        action = "verify-bundle" if "verify-bundle" in arguments else arguments[0]
        calls.append((action, list(arguments)))
        if action == "status":
            state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.is_file() else {}
            return _Completed(0, stdout=json.dumps({"state": state}))
        if action == "start":
            # The real vendor start writes the live running state on success.
            if action in results:
                result = results[action]
                return result() if callable(result) else result
            state_file.write_text(json.dumps(_live_running_state(prefix)), encoding="utf-8")
            return _Completed()
        if action in results:
            result = results[action]
            return result() if callable(result) else result
        return _Completed()

    # The bundle resolver must not read the real filesystem in these tests.
    monkeypatch.setattr(
        builder,
        "resolve_vendor_bundle",
        lambda version, override="": Path(f"/tmp/rt-infer-test-bundle-{version}"),
    )
    monkeypatch.setattr(
        builder,
        "bootstrap_control_script",
        lambda bundle: Path(f"{bundle}/scripts/priorityctl.py"),
    )
    monkeypatch.setattr(builder, "_run_ctl", _fake)
    monkeypatch.setattr(builder, "_probe_svm", lambda *_args: calls.append(("svm", [])))
    monkeypatch.setattr(builder, "_prepare_install_parent", lambda *_args: None)
    monkeypatch.setattr(builder, "_coverage_preflight", lambda *_args: {"pear"})
    monkeypatch.setattr(
        builder,
        "_write_periodic_session",
        lambda _config, _ownership, _mapping: None,
    )
    return {
        "calls": calls,
        "results": results,
        "state_file": state_file,
        "set_state": lambda payload: state_file.write_text(json.dumps(payload), encoding="utf-8"),
    }


def _live_running_state(prefix: str, pid: int | None = None) -> dict:
    """A vendor state whose scheduler identity matches a live process."""
    pid = pid if pid is not None else __import__("os").getpid()
    ticks = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]
    return {
        "status": "running",
        "prefix": prefix,
        "strategy": "pfp",
        "cpu": 15,
        "clients": [],
        "scheduler": {"pid": pid, "start_ticks": ticks},
        "boot_id": builder._boot_id(),
    }


def test_disabled_config_produces_no_actions():
    result = builder.build_rt_infer_scheduler(_context({"rt_infer": {"enable": False}}))
    assert result.actions == []


def test_simulated_target_never_touches_the_device(rt_infer_yaml, ctl):
    """Review #6: use_sim/benchmark refuses before any CLI call (review #7)."""
    with pytest.raises(RuntimeError, match="not compatible with this launch target"):
        builder.build_rt_infer_scheduler(_context(rt_infer_yaml, use_sim=True, runtime_target=RuntimeTarget.SIMULATION))
    assert ctl["calls"] == [], "the refusal happens before any vendor CLI call"


def test_non_hardware_target_refuses_at_enable_time(rt_infer_yaml):
    from robot_config.launch_builders.rt_infer_scheduler import rt_infer_enabled

    with pytest.raises(RuntimeError, match="not compatible"):
        rt_infer_enabled(_context(rt_infer_yaml, use_sim=True, runtime_target=RuntimeTarget.SIMULATION))


def test_start_failure_aborts_launch_without_native_fallback(rt_infer_yaml, ctl):
    ctl["results"]["install"] = _Completed(returncode=1, stderr="hash mismatch")
    result = builder.build_rt_infer_scheduler(_context(rt_infer_yaml))
    with pytest.raises(RuntimeError, match="RT-Infer bring-up failed"):
        _opaque_body(result)(None)


def test_non_ascend_host_aborts_before_any_cli_call(rt_infer_yaml, ctl, monkeypatch):
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    result = builder.build_rt_infer_scheduler(_context(rt_infer_yaml))
    with pytest.raises(RuntimeError, match="aarch64"):
        _opaque_body(result)(None)
    assert ctl["calls"] == [], "the refusal must happen before any vendor CLI call"


def test_ready_requires_a_live_scheduler_identity(rt_infer_yaml, ctl, monkeypatch):
    """Review #3: a stale 'running' file with a dead pid is not ready."""
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    ctl["set_state"](
        {"status": "running", "prefix": "/opt/rt-infer/2026.11.05", "scheduler": {"pid": 999999999, "start_ticks": "1"}}
    )
    rt_infer_yaml["rt_infer"]["readiness_timeout_sec"] = 0.2
    result = builder.build_rt_infer_scheduler(_context(rt_infer_yaml))
    with pytest.raises(RuntimeError, match="RT-Infer bring-up failed"):
        _opaque_body(result)(None)


def test_successful_start_installs_creates_mapping_then_starts(rt_infer_yaml, ctl, monkeypatch):
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    mapping_created = []
    monkeypatch.setattr(
        builder,
        "_create_periodic_mapping",
        lambda config: (
            mapping_created.append(config.prefix) or {"epoch_ns": 1, "period_ns": 55_555_556, "budget_ns": 38_000_000}
        ),
    )
    result = builder.build_rt_infer_scheduler(_context(rt_infer_yaml))
    actions = _opaque_body(result)(None)
    assert len(actions) == 2
    assert not isinstance(actions[0], EmitEvent)
    order = [call[0] for call in ctl["calls"]]
    assert order[:3] == ["verify-bundle", "svm", "install"]
    assert "doctor" in order and "status" in order and "start" in order
    assert mapping_created, "the periodic mapping is created by the owner"
    assert order.index("start") > order.index("status")


def test_start_carries_candidate10_profile(rt_infer_yaml, ctl, monkeypatch):
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    monkeypatch.setattr(
        builder,
        "_create_periodic_mapping",
        lambda config: {"epoch_ns": 1, "period_ns": 55_555_556, "budget_ns": 38_000_000},
    )
    result = builder.build_rt_infer_scheduler(_context(rt_infer_yaml))
    _opaque_body(result)(None)
    start = next(call for call in ctl["calls"] if call[0] == "start")
    arguments = start[1]
    # The _run_ctl wrapper adds --prefix; the recorded arguments are the
    # vendor subcommand options themselves.
    assert arguments[0] == "start"
    assert arguments[arguments.index("--scheduler") + 1] == "pfp"
    assert "--pfp-exclusive" in arguments and arguments[arguments.index("--pfp-exclusive") + 1] == "on"
    assert "--pfp-batch-select" in arguments and arguments[arguments.index("--pfp-batch-select") + 1] == "on"
    assert "--local-pool" in arguments and arguments[arguments.index("--local-pool") + 1] == "on"
    assert "--pfp-min-cache" in arguments and arguments[arguments.index("--pfp-min-cache") + 1] == "on"
    assert "--pfp-hi-priority" in arguments and arguments[arguments.index("--pfp-hi-priority") + 1] == "0"
    assert "--pfp-periodic" in arguments


def test_foreign_scheduler_is_refused_before_owner_claim(rt_infer_yaml, ctl, monkeypatch):
    """A live foreign scheduler is never adopted or stopped."""
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    ctl["set_state"](_live_running_state("/opt/rt-infer/2026.11.05"))
    result = builder.build_rt_infer_scheduler(_context(rt_infer_yaml))
    with pytest.raises(RuntimeError, match="refusing"):
        _opaque_body(result)(None)
    _shutdown_callback(result)(None, None)
    assert not any(call[0] == "stop" for call in ctl["calls"]), "never stop a scheduler we did not start"


@pytest.mark.parametrize("failure", ["racing_vendor", "foreign_after_nonzero", "unknown_after_timeout"])
def test_unproven_start_never_stops_state_and_releases_launch_lock(rt_infer_yaml, ctl, monkeypatch, failure):
    """A launch lease is not ownership of a concurrent vendor CLI attempt."""
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    rt_infer_yaml["rt_infer"].pop("pear_periodic")
    observed = None

    def rejected_start():
        nonlocal observed
        assert any(call[0] == "status" for call in ctl["calls"])
        observed = _live_running_state("/opt/rt-infer/2026.11.05")
        if failure == "foreign_after_nonzero":
            observed["prefix"] = "/opt/rt-infer/other-owner"
        elif failure == "unknown_after_timeout":
            observed.update(status="starting", scheduler={})
        ctl["set_state"](observed)
        if failure == "unknown_after_timeout":
            raise subprocess.TimeoutExpired("priorityctl start", 1)
        return _Completed(returncode=1, stderr="Stop the current scheduler cleanly first")

    ctl["results"]["start"] = rejected_start
    result = builder.build_rt_infer_scheduler(_context(rt_infer_yaml))
    with pytest.raises(RuntimeError, match="RT-Infer bring-up failed"):
        _opaque_body(result)(None)
    assert not any(call[0] == "stop" for call in ctl["calls"]), "unproven start cannot stop a daemon"
    assert json.loads(ctl["state_file"].read_text()) == observed
    # The failed start itself must release its lease, even before Shutdown.
    reclaimed = builder._acquire_owner_lock("next-launch")
    reclaimed.close()
    _shutdown_callback(result)(None, None)
    assert not any(call[0] == "stop" for call in ctl["calls"])


def test_shutdown_callback_is_nonblocking_and_supervisor_drains_clients(rt_infer_yaml, ctl, monkeypatch):
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    """Review #4: the owner starts, then stops only after the clients drain."""
    monkeypatch.setattr(
        builder,
        "_create_periodic_mapping",
        lambda config: {"epoch_ns": 1, "period_ns": 55_555_556, "budget_ns": 38_000_000},
    )
    result = builder.build_rt_infer_scheduler(_context(rt_infer_yaml))
    _opaque_body(result)(None)
    import fcntl as _fcntl

    # While the launch runs (before shutdown), the starter holds the lock:
    # a non-blocking EXCLUSIVE take from this side must FAIL.
    lock = builder.OWNER_LOCK.open("a+")
    try:
        _fcntl.flock(lock.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
    except OSError:
        held = True
    else:
        _fcntl.flock(lock.fileno(), _fcntl.LOCK_UN)
        held = False
    finally:
        lock.close()
    assert held, "the starter must hold the owner lock while running"
    # A LIVE client carrying THIS launch's token is draining; cleanup
    # SIGTERMs it (pidfd-verified, own token), then the vendor stop runs
    # once the live list is empty. The client here is a real helper process
    # that the cleanup must signal.

    import subprocess as _sp

    helper = _sp.Popen(["sleep", "60"])
    token = rt_infer_yaml["_rt_infer_launch"]["token"]
    token_file = builder.STATE_DIR / f"client-{helper.pid}.token"
    state = json.loads(ctl["state_file"].read_text(encoding="utf-8"))
    stat = (Path("/proc") / str(helper.pid) / "stat").read_text().rsplit(")", 1)[1].split()
    token_file.write_text(
        json.dumps({"token": token, "pid": helper.pid, "start_ticks": stat[19], "boot_id": builder._boot_id()}),
        encoding="utf-8",
    )
    state["clients"] = [{"pid": helper.pid, "start_ticks": stat[19]}]
    ctl["set_state"](state)
    signalled = {"done": False}

    def _tick(_seconds):
        # Only reached if the SIGTERMed helper is still tearing down on the
        # first re-read; the fast path (helper gone) drains without ticking.
        signalled["done"] = True
        helper.wait()
        state["clients"] = []
        ctl["set_state"](state)

    monkeypatch.setattr(builder.time, "sleep", _tick)
    _shutdown_callback(result)(None, None)
    assert not any(call[0] == "stop" for call in ctl["calls"]), "launch callback delegates teardown"

    from robot_config import rt_infer_scheduler_supervisor as supervisor

    monkeypatch.setattr(supervisor, "STATE_FILE", ctl["state_file"])
    monkeypatch.setattr(supervisor, "_cleanup_periodic", lambda *_args: None)

    stop_calls = []

    def _stop(command, **kwargs):
        stop_calls.append((command, kwargs))
        final = json.loads(ctl["state_file"].read_text())
        final["status"] = "stopped"
        ctl["set_state"](final)
        return _Completed()

    monkeypatch.setattr(supervisor.subprocess, "run", _stop)
    monkeypatch.setattr(
        supervisor,
        "expected_toolkit_home",
        lambda _prefix: "/usr/local/Ascend/ascend-toolkit/8.1.RC1",
    )
    assert supervisor._teardown(
        token=token,
        prefix="/opt/rt-infer/2026.11.05",
        control_tool="/opt/rt-infer/2026.11.05/scripts/priorityctl.py",
        timeout_sec=2.0,
        periodic_control_path="",
        expected_identity=state["scheduler"],
        expected_boot_id=state["boot_id"],
    )
    assert helper.poll() is not None, "cleanup must have SIGTERMed its own token-bearing client"
    state_now = json.loads(ctl["state_file"].read_text(encoding="utf-8"))
    live = [c for c in state_now.get("clients", []) if (Path("/proc") / str(c["pid"])).is_dir()]
    assert not live, "no live own client remains before the vendor stop"
    assert stop_calls and stop_calls[0][0][-1] == "stop", "supervisor stops after the live list drains"
    assert stop_calls[0][1]["env"]["ASCEND_TOOLKIT_HOME"] == "/usr/local/Ascend/ascend-toolkit/8.1.RC1"


def test_supervisor_never_signals_foreign_client(ctl, monkeypatch):
    import subprocess as _sp

    from robot_config import rt_infer_scheduler_supervisor as supervisor

    helper = _sp.Popen(["sleep", "60"])
    lease = builder._acquire_owner_lock("this-launch")
    try:
        stat = (Path("/proc") / str(helper.pid) / "stat").read_text().rsplit(")", 1)[1].split()
        state = _live_running_state("/opt/rt-infer/2026.11.05")
        state["clients"] = [{"pid": helper.pid, "start_ticks": stat[19]}]
        ctl["set_state"](state)
        (builder.STATE_DIR / f"client-{helper.pid}.token").write_text(
            json.dumps(
                {"token": "other-launch", "pid": helper.pid, "start_ticks": stat[19], "boot_id": builder._boot_id()}
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(supervisor, "STATE_FILE", ctl["state_file"])
        monkeypatch.setattr(
            supervisor.subprocess,
            "run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("foreign client must block stop")),
        )
        assert not supervisor._teardown(
            token="this-launch",
            prefix="/opt/rt-infer/2026.11.05",
            control_tool="/opt/rt-infer/2026.11.05/scripts/priorityctl.py",
            timeout_sec=0.1,
            periodic_control_path="",
            expected_identity=state["scheduler"],
            expected_boot_id=state["boot_id"],
        )
        assert helper.poll() is None
    finally:
        lease.close()
        helper.terminate()
        helper.wait()


def test_teardown_timeout_retains_state_and_does_not_report_success(ctl, monkeypatch, tmp_path):
    import subprocess as _sp

    from robot_config import rt_infer_scheduler_supervisor as supervisor

    ready = tmp_path / "sigterm-handler-ready"
    helper = _sp.Popen(
        [
            sys.executable,
            "-c",
            "import pathlib,signal,sys,time; "
            "signal.signal(signal.SIGINT, lambda *_: None); "
            "pathlib.Path(sys.argv[1]).touch(); time.sleep(60)",
            str(ready),
        ]
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "helper must install its SIGTERM handler before teardown starts"
        stat = (Path("/proc") / str(helper.pid) / "stat").read_text().rsplit(")", 1)[1].split()
        token = "owned-launch"
        state = _live_running_state("/opt/rt-infer/2026.11.05")
        state["clients"] = [{"pid": helper.pid, "start_ticks": stat[19]}]
        ctl["set_state"](state)
        lease = builder._acquire_owner_lock(token)
        (builder.STATE_DIR / f"client-{helper.pid}.token").write_text(
            json.dumps({"token": token, "pid": helper.pid, "start_ticks": stat[19], "boot_id": builder._boot_id()}),
            encoding="utf-8",
        )
        monkeypatch.setattr(supervisor, "STATE_FILE", ctl["state_file"])
        monkeypatch.setattr(supervisor, "_cleanup_periodic", lambda *_args: None)
        monkeypatch.setattr(supervisor, "expected_toolkit_home", lambda _prefix: None)
        monkeypatch.setattr(supervisor.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("undrained stop"))
        assert not supervisor._teardown(
            token=token,
            prefix="/opt/rt-infer/2026.11.05",
            control_tool="/opt/rt-infer/2026.11.05/scripts/priorityctl.py",
            timeout_sec=0.2,
            periodic_control_path="",
            expected_identity=state["scheduler"],
            expected_boot_id=state["boot_id"],
        )
        assert helper.poll() is None
        assert ctl["state_file"].exists()
        lease.close()
    finally:
        if helper.poll() is None:
            helper.kill()
            helper.wait()


def test_supervisor_rejects_missing_corrupt_and_swapped_state(ctl, monkeypatch):
    from robot_config import rt_infer_scheduler_supervisor as supervisor

    identity = _live_running_state("/opt/rt-infer/2026.11.05")["scheduler"]
    monkeypatch.setattr(supervisor, "STATE_FILE", ctl["state_file"])
    assert supervisor._vendor_state() == {}
    assert not supervisor._state_matches_expected({}, "/opt/rt-infer/2026.11.05", identity, builder._boot_id())

    ctl["state_file"].write_text("not-json", encoding="utf-8")
    assert supervisor._vendor_state() == {}

    swapped = _live_running_state("/opt/rt-infer/2026.11.05")
    swapped["scheduler"] = {"pid": 999999999, "start_ticks": "1"}
    assert not supervisor._state_matches_expected(swapped, "/opt/rt-infer/2026.11.05", identity, builder._boot_id())


def test_builder_registered_in_builtin_table():
    from robot_config.launch_registry.builtin_registry import BUILTIN_BUILDERS

    assert "robot_config.launch_builders.rt_infer_scheduler:RT_INFER_SCHEDULER_BUILDER" in BUILTIN_BUILDERS, (
        "the owner must be a builtin so it precedes extension model builders"
    )


# ---------------------------------------------------------------------------
# Round-3 review contracts
# ---------------------------------------------------------------------------


def test_second_owner_is_refused_conservatively(rt_infer_yaml, ctl, monkeypatch):
    """Review #D: an exclusive owner lock — two owners can never race."""
    import fcntl

    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    # A live foreign lock holder.
    holder = builder.OWNER_LOCK.open("a+")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    import os as _os

    ticks = (Path("/proc") / str(_os.getpid()) / "stat").read_text().rsplit(")", 1)[1].split()[19]
    holder.seek(0)
    holder.write(json.dumps({"identity": {"pid": _os.getpid(), "start_ticks": ticks}, "token": "other"}))
    holder.flush()
    result = builder.build_rt_infer_scheduler(_context(rt_infer_yaml))
    with pytest.raises(RuntimeError, match="another RT-Infer launch owner is active"):
        _opaque_body(result)(None)
    fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
    holder.close()


def test_dead_lock_holder_is_reclaimed(rt_infer_yaml, ctl, monkeypatch):
    """A stale lock from a dead launch is reclaimed, not a permanent brick."""
    import fcntl

    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    holder = builder.OWNER_LOCK.open("a+")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    holder.seek(0)
    holder.write(json.dumps({"identity": {"pid": 999999999, "start_ticks": "1"}, "token": "dead"}))
    holder.flush()
    # Keep the fd open but RELEASE the lock: a dead holder cannot flock.
    fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
    holder.close()
    reclaimed = builder._acquire_owner_lock("new-owner")
    try:
        document = json.loads(builder.OWNER_LOCK.read_text())
        assert document["token"] == "new-owner"
        assert document["identity"]["pid"] == os.getpid()
    finally:
        reclaimed.close()


def test_unmapped_ascend_service_refuses_before_any_device_work(rt_infer_yaml, ctl, monkeypatch):
    """Review #F: the coverage preflight runs in the BUILD phase, before the
    owner's start could ever insmod — the refusal precedes every ctl call."""

    from types import SimpleNamespace as NS

    profile = NS(backend="ascend")
    manifest = NS(
        deployment=NS(runtime_profile=profile, role_runtime_profiles={}),
        manifest=NS(model=NS(model_type="some_unmapped_ascend_model")),
    )
    service = NS(instance_id="hri_new_model", validated_manifest=manifest)
    monkeypatch.setattr(
        "robot_config.perception_runtime_config.parse_perception_runtime_config",
        lambda *_a: NS(enabled_services=[service]),
    )
    monkeypatch.setattr(builder, "_coverage_preflight", _REAL_COVERAGE_PREFLIGHT)
    robot_config = dict(rt_infer_yaml)
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    with pytest.raises(RuntimeError, match="hri_new_model"):
        builder.build_rt_infer_scheduler(_context(robot_config))
    assert ctl["calls"] == [], "the preflight refusal must precede every vendor CLI call"
