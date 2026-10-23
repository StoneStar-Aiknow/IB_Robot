"""Cancellation concurrency and real child-process tests, without robot hardware."""

from __future__ import annotations

import errno
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_skill_cli.hermes_motion_plugin import MotionAdapter
from robot_skill_cli.hermes_robot_runner import RobotRunner, terminal_state, workflow_arguments
from robot_skill_cli.hermes_stop_state import StopStateError, StopStore
from robot_skill_cli.hermes_workflow_stop import managed_stop, record_admission_error
from robot_skill_cli.stop_contract import is_stop


def begin(store, session="one", turn="turn"):
    store.user_input(session, stop=False)
    store.begin_turn(session, turn, time.time())


def wait_for(predicate, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not converge")


@pytest.fixture
def store(tmp_path):
    return StopStore(tmp_path / "private")


def test_closed_stop_vocabulary():
    assert is_stop(" 别动 ") and is_stop("STOP")
    assert not is_stop("别动，等一下再跳")
    assert not is_stop("不停")


def test_stop_before_turn_and_submission_stays_latched(store):
    arrival = time.time()
    store.user_input("one", stop=True)
    store.begin_turn("one", "old", arrival)
    with pytest.raises(StopStateError, match="STOP_LATCHED"):
        store.claim("one", "old", "token")
    store.user_input("one", stop=False)
    with pytest.raises(StopStateError, match="STOP_LATCHED"):
        store.claim("one", "old", "token")
    store.begin_turn("one", "fresh", time.time())
    store.claim("one", "fresh", "fresh-token")


def test_same_session_race_has_one_owner_and_other_sessions_are_independent(store):
    begin(store)

    def claim(token):
        try:
            store.claim("one", "turn", token)
            return True
        except StopStateError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(claim, ["a", "b"])) == 1
    begin(store, "two")
    store.claim("two", "turn", "two-token")
    store.user_input("one", stop=True)
    assert not store.stopped("two", "two-token")


def test_unknown_owner_cannot_be_cleared_by_new_user_request(store):
    begin(store)
    store.claim("one", "turn", "a")
    store.update("one", "a", state="unknown")
    store.user_input("one", stop=False)
    store.begin_turn("one", "new", time.time())
    with pytest.raises(StopStateError, match="UNKNOWN_OR_BUSY"):
        store.claim("one", "new", "b")


def test_fresh_input_after_cancel_does_not_depend_on_stale_ui_busy_flag(store):
    begin(store)
    store.claim("one", "turn", "a")
    store.user_input("one", stop=True)
    store.update("one", "a", state="stopped")
    store.user_input("one", stop=False, busy=True)
    with pytest.raises(StopStateError, match="STOP_LATCHED"):
        store.claim("one", "turn", "old-retry")
    store.begin_turn("one", "new", time.time())
    store.claim("one", "new", "b")


def test_workflow_poll_latches_stop_before_and_during_execution(store, monkeypatch):
    begin(store)
    store.claim("one", "turn", "token")
    monkeypatch.setenv("IBROBOT_STOP_ROOT", str(store.root))
    monkeypatch.setenv("IBROBOT_STOP_SESSION", "one")
    monkeypatch.setenv("IBROBOT_STOP_TOKEN", "token")
    event = threading.Event()
    with managed_stop(event):
        store.user_input("one", stop=True)
        assert event.wait(0.5)
    with managed_stop(threading.Event()):
        assert store.stopped("one", "token")


def test_unmanaged_hermes_workflow_is_rejected(monkeypatch):
    monkeypatch.delenv("IBROBOT_STOP_ROOT", raising=False)
    monkeypatch.setenv("IBROBOT_HERMES_MANAGED_REQUIRED", "1")
    with pytest.raises(ValueError, match="managed workflow"), managed_stop(threading.Event()):
        pytest.fail("must not reach execution")


def test_pre_admission_error_does_not_leave_motion_unknown(store, monkeypatch):
    begin(store)
    store.claim("one", "turn", "token")
    monkeypatch.setenv("IBROBOT_STOP_ROOT", str(store.root))
    monkeypatch.setenv("IBROBOT_STOP_SESSION", "one")
    monkeypatch.setenv("IBROBOT_STOP_TOKEN", "token")
    record_admission_error("SERVER_UNAVAILABLE", "test unavailable")
    assert store.snapshot("one")["owner"]["state"] == "not_submitted"
    begin(store, turn="new")
    store.claim("one", "new", "new-token")


@pytest.mark.parametrize(
    "fresh,submit,expected", [(True, False, "not_submitted"), (True, True, "starting"), (False, False, "starting")]
)
def test_error_after_presentation_tracks_submission_not_presentation(store, monkeypatch, fresh, submit, expected):
    begin(store)
    store.claim("one", "turn", "token")
    for key, value in {"ROOT": str(store.root), "SESSION": "one", "TOKEN": "token"}.items():
        monkeypatch.setenv("IBROBOT_STOP_" + key, value)
    with managed_stop(threading.Event(), fresh_request=fresh) as guard:
        store.update("one", "token", presentation={"task_id": "task"})
        if submit:
            with pytest.raises(RuntimeError), guard():
                raise RuntimeError("transport could have submitted before raising")
        record_admission_error("SKILL_LIMIT_VIOLATION", "invalid navigation direction")
    assert store.snapshot("one")["owner"]["state"] == expected
    if expected == "not_submitted":
        store.user_input("one", stop=True)
        begin(store, turn="fresh-turn")
        store.claim("one", "fresh-turn", "fresh-token")


@pytest.mark.parametrize("suffix", ["; true", "&& robot-skill execute x", "--config-name other", "--text again"])
def test_shell_composition_and_config_override_are_rejected(suffix):
    with pytest.raises(StopStateError):
        workflow_arguments(f"robot-skill run-workflow --text x --workflow-json '[]' {suffix}", "/bound/robot-skill")


def test_command_uses_bound_executable_and_literal_argv():
    args = workflow_arguments("robot-skill run-workflow --text '$(false)' --workflow-json '[]'", "/bound/robot-skill")
    assert args[2] == "$(false)"
    assert workflow_arguments("robot-skill status", "/bound/robot-skill") is None


def test_model_generated_unsupported_flags_are_rejected():
    with pytest.raises(StopStateError, match="use run-workflow --text"):
        workflow_arguments(
            "robot-skill run-workflow --raw-command dance --task-id dance-$(date +%s)",
            "/bound/robot-skill",
        )


def test_exact_help_is_read_only():
    assert workflow_arguments("robot-skill run-workflow --help", "/bound/robot-skill") is None
    with pytest.raises(StopStateError):
        workflow_arguments("robot-skill run-workflow --help ; true", "/bound/robot-skill")


@pytest.fixture
def child(tmp_path):
    helpers = Path(__file__).with_name("test_interactive_control.py")
    executable = tmp_path / "robot-skill"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import runpy, threading, time, os\n"
        "from types import SimpleNamespace\n"
        "from pathlib import Path\n"
        "from robot_skill_cli import cli, interactive_control as ic\n"
        f"h = runpy.run_path({str(helpers)!r})\n"
        "bridge = h['FakeBridge'](h['FakeClock']())\n"
        f"bridge.send_hook = lambda: Path({str(executable.with_suffix('.ready'))!r}).touch()\n"
        "original = ic.InteractiveController\n"
        "def factory(*args, **kwargs):\n"
        "    return original(*args, **kwargs, view_resolver=lambda *_: h['_capability_view']())\n"
        "ic.InteractiveController = factory\n"
        "def cancel(task_id, **kwargs):\n"
        "    time.sleep(float(os.environ.get('TEST_CANCEL_DELAY', '0')))\n"
        "    bridge.plan_result_status = 5\n"
        "    bridge.plan_result.update(success=False, error_code='SKILL_CANCELLED', completed_step_count=0)\n"
        "    bridge.result_future._done = True\n"
        "    return {'accepted': True}\n"
        "bridge.cancel_agent_plan = cancel\n"
        'args = SimpleNamespace(raw_command=\'test\', workflow_json=\'[{"schema_version":1,"skill_name":"nod_yes"}]\')\n'
        "code = cli._run_workflow(args, SimpleNamespace(view={'timeout_policy': {'rpc_timeout_sec': 2}}), bridge)\n"
        "raise SystemExit(code.exit_code)\n"
    )
    executable.chmod(0o700)
    probe = tmp_path / "exec-probe"
    probe.write_text(f"#!{sys.executable}\nraise SystemExit(0)\n", encoding="utf-8")
    probe.chmod(0o700)
    try:
        subprocess.run([str(probe)], check=True, capture_output=True, timeout=2)
    except OSError as exc:
        if exc.errno == errno.EACCES:
            pytest.skip("temporary test directory is mounted noexec")
        raise
    return executable


@pytest.mark.parametrize("delay", [0, 1.5])
def test_real_cli_child_converges_without_hermes_force_kill(store, child, monkeypatch, delay):
    monkeypatch.setenv("TEST_CANCEL_DELAY", str(delay))
    begin(store)
    runner = RobotRunner(store, str(child), cancel_timeout=5)
    adapter = MotionAdapter(store, str(child))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(runner.run, [], "one", "turn")
        wait_for(lambda: child.with_suffix(".ready").exists() or future.done())
        assert not future.done(), future.result() if future.done() else ""
        before = time.monotonic()
        response = adapter.user_input("别动", "one", busy=True)
        assert "正在停止" in response["message"]
        result = json.loads(future.result(timeout=7))
    assert time.monotonic() - before >= delay
    assert result["exit_code"] == 13
    assert result["error"] is None
    owner = store.snapshot("one")["owner"]
    assert owner["state"] in {"stopped", "not_submitted"}
    assert owner["terminal"]["result"]["error_code"] == "SKILL_CANCELLED"
    assert owner["stop_observed_at"] - owner["stop_received_at"] < 0.5
    assert not store.owner_alive(owner["token"])


def test_lost_or_mismatched_terminal_remains_unknown():
    assert terminal_state(None, None) == "unknown"
    assert terminal_state({"task_id": "a"}, {"task_id": "b"}) == "unknown"
    assert terminal_state({"task_id": "a"}, {"task_id": "a", "result": {"success": "false"}}) == "unknown"


def test_presentation_stop_cannot_submit_goal(store, child):
    begin(store)
    seen = []

    def present(text):
        seen.append(json.loads(text))
        store.user_input("one", stop=True)

    result = json.loads(RobotRunner(store, str(child)).run([], "one", "turn", presentation_callback=present))
    assert result["error"] is None
    assert len(seen) == 1
    assert not child.with_suffix(".ready").exists()
    assert store.snapshot("one")["owner"]["state"] == "not_submitted"


def test_cancel_timeout_retains_unknown_and_blocks_new_motion(store, child, monkeypatch):
    monkeypatch.setenv("TEST_CANCEL_DELAY", "2")
    begin(store)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(RobotRunner(store, str(child), cancel_timeout=0.2).run, [], "one", "turn")
        wait_for(lambda: child.with_suffix(".ready").exists() or future.done())
        assert not future.done(), future.result() if future.done() else ""
        store.user_input("one", stop=True)
        result = json.loads(future.result(timeout=3))
    assert result["error"].startswith("ROBOT_STOP_UNKNOWN")
    assert store.snapshot("one")["owner"]["state"] == "unknown"
    begin(store, turn="later")
    with pytest.raises(StopStateError):
        store.claim("one", "later", "new")


def test_recovery_does_not_cancel_a_live_owner(store, monkeypatch):
    begin(store)
    store.claim("one", "turn", "token")
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: pytest.fail("must not send a second cancel"))
    with store.owner_lock("token"):
        assert store.owner_alive("token")
        RobotRunner(store, "/bound/robot-skill").recover("one")


def test_terminal_cannot_release_execution_ownership_before_parent_exits(store):
    begin(store)
    with store.owner_lock("first"):
        store.claim("one", "turn", "first")
        store.update("one", "first", state="stopped")
        with pytest.raises(StopStateError, match="UNKNOWN_OR_BUSY"):
            store.claim("one", "turn", "second")


def test_stop_is_checked_at_exact_goal_submission(store):
    begin(store)
    store.claim("one", "turn", "token")
    event = threading.Event()
    store.user_input("one", stop=True)
    with store.submission_guard("one", "token", event):
        assert event.is_set()


def test_recovery_uses_complete_plan_identity_and_retains_unknown(store, monkeypatch):
    begin(store)
    store.claim("one", "turn", "token")
    presentation = {
        "task_id": "task",
        "plan_id": "plan",
        "plan_digest": "digest",
        "steps": [{}, {}],
        "registry_identity": {"registry_epoch": "epoch", "registry_generation": 2, "registry_digest": "registry"},
    }
    store.update("one", "token", state="unknown", presentation=presentation)
    calls = []

    def run(args, **_kwargs):
        calls.append(args)
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "data": {
                        "goal_status": 5,
                        "result": {
                            "success": False,
                            "error_code": "SKILL_CANCELLED",
                            "plan_id": "wrong",
                            "plan_digest": "digest",
                            "actual_registry_epoch": "epoch",
                            "actual_registry_generation": 2,
                            "actual_registry_digest": "registry",
                            "completed_step_count": 0,
                        },
                    }
                }
            )
        )

    monkeypatch.setattr("subprocess.run", run)
    RobotRunner(store, "/bound/robot-skill").recover("one")
    assert calls[0] == [
        "/bound/robot-skill",
        "cancel-plan",
        "--task-id",
        "task",
        "--plan-id",
        "plan",
        "--plan-digest",
        "digest",
        "--registry-epoch",
        "epoch",
        "--registry-generation",
        "2",
        "--registry-digest",
        "registry",
        "--expected-step-count",
        "2",
    ]
    assert store.snapshot("one")["owner"]["state"] == "unknown"


def test_runtime_installer_is_idempotent_and_refuses_modified_files(tmp_path, monkeypatch):
    from robot_skill_cli import hermes_runtime_adapter as runtime

    monkeypatch.setattr(
        runtime.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=0, stdout=runtime.SUPPORTED_REVISION)
    )
    monkeypatch.setattr(runtime, "_base_replacements", lambda: {"entry.py": [("value = 1", "value = 2")]})
    monkeypatch.setattr(runtime, "_upgrade_replacements", lambda: {})
    monkeypatch.setattr(runtime, "_feishu_replacements", lambda: {})
    entry = tmp_path / "entry.py"
    entry.write_text("value = 1\n")
    (tmp_path / "hermes_cli").mkdir()
    helper = tmp_path / "helper.py"
    helper.write_text("API = 1\n")
    backups = []
    kwargs = {"write": lambda path, text: path.write_text(text), "backup": backups.append}
    assert runtime.sync_runtime(tmp_path, helper, dry_run=True, **kwargs)
    assert entry.read_text() == "value = 1\n" and backups == []
    assert runtime.sync_runtime(tmp_path, helper, dry_run=False, **kwargs)
    assert not runtime.sync_runtime(tmp_path, helper, dry_run=False, **kwargs)
    entry.write_text("user_change = True\n")
    with pytest.raises(ValueError, match="changed since installation"):
        runtime.sync_runtime(tmp_path, helper, dry_run=False, **kwargs)
    assert entry.read_text() == "user_change = True\n"


def test_unsupported_runtime_is_rejected_before_mutation(tmp_path, monkeypatch):
    from robot_skill_cli import hermes_runtime_adapter as runtime

    monkeypatch.setattr(runtime.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=0, stdout="other"))
    with pytest.raises(ValueError, match="unsupported Hermes"):
        runtime.sync_runtime(
            tmp_path,
            tmp_path / "helper",
            dry_run=False,
            write=lambda *_: pytest.fail(),
            backup=lambda *_: pytest.fail(),
        )
