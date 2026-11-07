"""Process and launch-event boundary tests; no vendor binary or NPU access."""

import asyncio
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from launch import LaunchDescription, LaunchService
from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessIO, OnShutdown
from launch.events import Shutdown

from robot_config import rt_infer_scheduler_supervisor as supervisor
from robot_config.launch_builders import rt_infer_scheduler as owner
from robot_config.launch_registry.registry import LaunchBuildContext
from robot_config.rt_infer_config import RtInferConfig


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(owner, "STATE_DIR", tmp_path)
    monkeypatch.setattr(owner, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(owner, "OWNER_LOCK", tmp_path / "launch-owner.lock")
    monkeypatch.setattr(supervisor, "STATE_FILE", tmp_path / "state.json")
    return tmp_path


def test_two_processes_cannot_claim_even_with_corrupt_lock_header(state_dir):
    handle = owner._acquire_owner_lock("first")
    handle.seek(0)
    handle.truncate()
    handle.write("not-json")
    handle.flush()
    code = """
import sys
from pathlib import Path
from robot_config.launch_builders import rt_infer_scheduler as owner
owner.STATE_DIR = Path(sys.argv[1])
owner.OWNER_LOCK = owner.STATE_DIR / 'launch-owner.lock'
try:
    owner._acquire_owner_lock('second')
except RuntimeError:
    sys.exit(17)
sys.exit(0)
"""
    try:
        contender = subprocess.run([sys.executable, "-c", code, str(state_dir)], timeout=3, check=False)
        assert contender.returncode == 17
    finally:
        handle.close()
    reclaimed = owner._acquire_owner_lock("reclaimed")
    reclaimed.close()
    assert json.loads(owner.OWNER_LOCK.read_text())["token"] == "reclaimed"


def test_async_shutdown_keeps_launch_dispatching_and_waits_for_real_client(state_dir, monkeypatch):
    config = RtInferConfig(enable=True, bundle_version="test", stop_timeout_sec=3)
    ownership = owner._SchedulerOwnership()
    ownership.token = "async-drain-test"
    ownership.started_here = True
    ownership.scheduler_identity = owner._process_identity(os.getpid())
    ownership.lock_handle = owner._acquire_owner_lock(ownership.token)
    state = {
        "boot_id": ownership.boot_id,
        "status": "running",
        "prefix": config.prefix,
        "scheduler": ownership.scheduler_identity,
        "clients": [],
    }
    owner.STATE_FILE.write_text(json.dumps(state))
    child_code = """
import json, os, signal, sys, time
from pathlib import Path
root, token = Path(sys.argv[1]), sys.argv[2]
pid = os.getpid()
ticks = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
identity = dict(pid=pid, start_ticks=ticks)
state = json.loads((root/'state.json').read_text())
state['clients'] = [identity]
(root/'state.json').write_text(json.dumps(state))
(root/f'client-{pid}.token').write_text(json.dumps(dict(identity, token=token, boot_id=state['boot_id'])))
def close(*_args):
    (root/'client-closed').write_text('closed')
    sys.exit(0)
signal.signal(signal.SIGINT, close)
print('READY', flush=True)
while True: time.sleep(0.02)
"""
    stop_calls = []

    def vendor_stop(command, **_kwargs):
        assert (state_dir / "client-closed").exists()
        assert not any(owner._identity_alive(item) for item in owner._vendor_state()["clients"])
        stop_calls.append(command)
        final = owner._vendor_state()
        final["status"] = "stopped"
        owner.STATE_FILE.write_text(json.dumps(final))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(supervisor.subprocess, "run", vendor_stop)
    child = ExecuteProcess(cmd=[sys.executable, "-c", child_code, str(state_dir), ownership.token])
    stop = owner._StopScheduler(config, ownership)
    service = LaunchService()
    service.include_launch_description(
        LaunchDescription(
            [
                RegisterEventHandler(OnShutdown(on_shutdown=lambda *_a: [stop])),
                RegisterEventHandler(
                    OnProcessIO(
                        target_action=child,
                        on_stdout=lambda event: (
                            [EmitEvent(event=Shutdown(reason="client ready"))] if b"READY" in event.text else []
                        ),
                    )
                ),
                child,
            ]
        )
    )
    result = asyncio.run(asyncio.wait_for(service.run_async(), timeout=10))
    assert result == 0
    assert len(stop_calls) == 1
    assert ownership.lock_handle is None
    assert stop.get_asyncio_future().done()


def test_empty_prefix_install_uses_bundle_bootstrap_and_prepares_parent(state_dir, monkeypatch):
    bundle = state_dir / "bundle"
    (bundle / "scripts").mkdir(parents=True)
    script = bundle / "scripts/priorityctl.py"
    script.write_text("""
import sys
from pathlib import Path
if 'install' in sys.argv:
    prefix = Path(sys.argv[-1])
    assert prefix.parent.is_dir()
    assert not prefix.exists()
    prefix.mkdir()
    (prefix/'bootstrap-used').write_text(__file__)
""")
    (bundle / "svm_bind_probe.py").write_text("print('guarded-test-probe')\n")
    prefix = state_dir / "new-opt/rt-infer/test"
    config = SimpleNamespace(prefix=str(prefix), bundle_version="test", bundle_dir=str(bundle), verify_timeout_sec=3)
    monkeypatch.setattr(owner, "resolve_vendor_bundle", lambda *_a: bundle)
    owner._install_bundle(config)
    assert (prefix / "bootstrap-used").read_text() == str(script)


def test_manifest_backend_not_deployment_name_controls_policy_refusal(monkeypatch):
    from robot_config import inference_config

    profile = SimpleNamespace(backend="ascend")
    validated = SimpleNamespace(deployment=SimpleNamespace(runtime_profile=profile, role_runtime_profiles={}))
    pipeline = SimpleNamespace(pipeline_id="new_policy", deployment="innocent_name", validated_manifest=validated)
    monkeypatch.setattr(
        inference_config,
        "parse_inference_config",
        lambda *_a, **_k: SimpleNamespace(pipelines={"new_policy": pipeline}),
    )
    context = LaunchBuildContext(robot_config={}, with_inference=True, control_mode="model_inference")
    with pytest.raises(RuntimeError, match="new_policy"):
        owner._coverage_preflight(context, RtInferConfig(enable=True))


def test_no_pear_profile_does_not_create_periodic_reservation(monkeypatch):
    context = LaunchBuildContext(
        robot_config={
            "rt_infer": {
                "enable": True,
                "bundle_version": "test",
                "pear_periodic": {},
            }
        }
    )
    started = []
    context.robot_config["visual_interaction"] = {"enabled": True, "vl": {"implementation": "internvl"}}
    monkeypatch.setattr(owner, "_start_scheduler", lambda config, _ownership: started.append(config))
    result = owner.build_rt_infer_scheduler(context)
    for action in result.actions:
        function = getattr(action, "_OpaqueFunction__function", None)
        if function is not None:
            function(None)
    assert len(started) == 1
    assert started[0].pear_periodic is None
    assert context.robot_config["_rt_infer_launch"]["periodic"] is False


def test_start_failure_returns_nonzero_without_starting_dependent_node(state_dir, monkeypatch):
    context = LaunchBuildContext(robot_config={"rt_infer": {"enable": True, "bundle_version": "test"}})
    marker = state_dir / "dependent-started"
    monkeypatch.setattr(owner, "_coverage_preflight", lambda *_a: {"vl"})
    monkeypatch.setattr(owner, "_start_scheduler", lambda *_a: (_ for _ in ()).throw(RuntimeError("install refused")))
    result = owner.build_rt_infer_scheduler(context)
    child = ExecuteProcess(
        cmd=[sys.executable, "-c", "from pathlib import Path; Path(__import__('sys').argv[1]).touch()", str(marker)]
    )
    service = LaunchService()
    service.include_launch_description(LaunchDescription([*result.actions, child]))
    assert asyncio.run(service.run_async()) == 1
    assert not marker.exists()


def test_no_ascend_models_never_start_owner(monkeypatch):
    context = LaunchBuildContext(robot_config={"rt_infer": {"enable": True, "bundle_version": "test"}})
    monkeypatch.setattr(owner, "_start_scheduler", lambda *_a: pytest.fail("all-native config touched scheduler"))
    assert owner.build_rt_infer_scheduler(context).actions == []


@pytest.mark.parametrize(
    "code",
    [
        "pass",
        "import os,signal; os.kill(os.getpid(),signal.SIGINT)",
        "import os,signal; os.kill(os.getpid(),signal.SIGTERM)",
    ],
)
def test_unexpected_supervisor_exit_fails_the_real_launch(monkeypatch, code):
    ownership = owner._SchedulerOwnership()
    ownership.scheduler_identity = owner._process_identity(os.getpid())
    monkeypatch.setattr(owner, "Node", lambda **_kw: ExecuteProcess(cmd=[sys.executable, "-c", code]))
    registration, supervisor_action = owner._supervise_action(RtInferConfig(bundle_version="test"), ownership)
    service = LaunchService()
    service.include_launch_description(LaunchDescription([registration, supervisor_action]))
    assert asyncio.run(service.run_async()) == 1


def test_vendor_stop_waits_for_own_wrapper_bookkeeping(state_dir, monkeypatch):
    import threading
    import time

    token = "bookkeeping-owner"
    lease = owner._acquire_owner_lock(token)
    identity = owner._process_identity(os.getpid())
    boot = owner._boot_id()
    state = {"boot_id": boot, "prefix": "/opt/rt-infer/test", "status": "running", "scheduler": identity, "clients": []}
    owner.STATE_FILE.write_text(json.dumps(state))
    wrapper = subprocess.Popen(["sleep", "30"])
    wrapper_identity = owner._process_identity(wrapper.pid)
    wrapper_identity["boot_id"] = boot
    (state_dir / "client-999999999.token").write_text(
        json.dumps(
            {
                "pid": 999999999,
                "start_ticks": "1",
                "token": token,
                "boot_id": boot,
                "wrapper": wrapper_identity,
            }
        )
    )
    stopped = []

    def vendor_stop(*_args, **_kw):
        assert wrapper.poll() is not None
        stopped.append(True)
        final = owner._vendor_state()
        final["status"] = "stopped"
        owner.STATE_FILE.write_text(json.dumps(final))
        return SimpleNamespace(returncode=0)

    def finish_bookkeeping():
        time.sleep(0.2)
        wrapper.terminate()

    monkeypatch.setattr(supervisor.subprocess, "run", vendor_stop)
    waiter = threading.Thread(target=finish_bookkeeping)
    waiter.start()
    try:
        assert supervisor._teardown(
            token=token,
            prefix=state["prefix"],
            control_tool="unused",
            timeout_sec=2,
            periodic_control_path="",
            expected_identity=identity,
            expected_boot_id=boot,
        )
        assert stopped == [True]
    finally:
        waiter.join()
        wrapper.wait()
        lease.close()


def test_stop_exit_zero_with_recovery_state_is_not_success(state_dir, monkeypatch):
    token, identity, boot = "not-clean", owner._process_identity(os.getpid()), owner._boot_id()
    lease = owner._acquire_owner_lock(token)
    state = {"boot_id": boot, "prefix": "/opt/rt-infer/test", "status": "running", "scheduler": identity, "clients": []}
    owner.STATE_FILE.write_text(json.dumps(state))

    def vendor_stop(*_args, **_kwargs):
        state["status"] = "recovery-required"
        owner.STATE_FILE.write_text(json.dumps(state))
        return SimpleNamespace(returncode=0, stderr="", stdout="Scheduler exited unexpectedly")

    monkeypatch.setattr(supervisor.subprocess, "run", vendor_stop)
    try:
        assert not supervisor._teardown(
            token=token,
            prefix=state["prefix"],
            control_tool="unused",
            timeout_sec=1,
            periodic_control_path="",
            expected_identity=identity,
            expected_boot_id=boot,
        )
        assert json.loads(owner.STATE_FILE.read_text())["status"] == "recovery-required"
    finally:
        lease.close()
