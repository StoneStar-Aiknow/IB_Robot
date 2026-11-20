"""x2_dataplane_nice.py against stand-in dataplanes: python processes with threads whose
argv[0] ends in a per-test path, so no real dataplane on this host can match.

Unprivileged runs can raise a nice but not lower it again; the tests use a positive
nice, and a restore of the nice is checked to fail out loud there (not silently).

    python3 -m pytest -q scripts/x2_dataplane_sched/test_x2_dataplane_nice.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("x2_dataplane_nice.py")
ROOT = os.geteuid() == 0
STAND_IN = """
import sys, threading, time
def spare():
    time.sleep(3600)
for _ in range(3):
    threading.Thread(target=spare, daemon=True).start()
for line in sys.stdin:  # "thread": one more, after the start
    threading.Thread(target=spare, daemon=True).start()
"""


class Env(dict):
    """The scripts' environment; not printed in a failure report (it holds the session's)."""

    def __repr__(self):
        return "<env>"


@pytest.fixture
def env(tmp_path):
    exe = tmp_path / "install" / "lib" / "aimdk_dataplane" / "aimdk_dataplane"
    exe.parent.mkdir(parents=True)
    exe.symlink_to(sys.executable)
    values = Env(os.environ)
    values.update(
        X2_DATAPLANE_EXE=str(exe),
        X2_DATAPLANE_NICE_STATE=str(tmp_path / "state"),
        X2_DATAPLANE_NICE_VALUE="7",
    )
    values.pop("X2_DATAPLANE_NICE", None)
    values.pop("X2_DATAPLANE_CPUS", None)
    started = []

    def stand_in():
        process = subprocess.Popen([str(exe), "-c", STAND_IN], stdin=subprocess.PIPE, text=True)
        started.append(process)
        wait_for(lambda: len(tids(process.pid)) == 4, "the stand-in's threads")
        return process

    def run(*args, extra=None, check=True):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *args], env=Env({**values, **(extra or {})}),
            capture_output=True, text=True, timeout=30,
        )
        if check:
            assert result.returncode == 0, result.stdout + result.stderr
        return result

    yield stand_in, run, values
    run("restore", check=False)  # stops a watcher a test left
    for process in started:
        process.kill()
        process.wait()


def wait_for(predicate, what, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def tids(pid):
    try:
        return sorted(int(t) for t in os.listdir(f"/proc/{pid}/task"))
    except OSError:
        return []


def nices(pid):
    return {tid: os.getpriority(os.PRIO_PROCESS, tid) for tid in tids(pid)}


def affinities(pid):
    return {tid: os.sched_getaffinity(tid) for tid in tids(pid)}


def more_threads(process):
    process.stdin.write("thread\n")
    process.stdin.flush()


def test_apply_reaches_every_thread_and_restore_puts_back_the_cpus(env):
    stand_in, run, _ = env
    process = stand_in()
    before = affinities(process.pid)
    one_cpu = min(os.sched_getaffinity(0))
    result = run("apply", "--cpus", str(one_cpu))
    assert "失败 0" in result.stdout
    assert set(nices(process.pid).values()) == {7}, "every thread, not only the main one"
    assert all(cpus == {one_cpu} for cpus in affinities(process.pid).values())
    assert "nice=7" in run("status").stdout

    result = run("restore", check=False)
    assert affinities(process.pid) == before
    if ROOT:
        assert result.returncode == 0
        assert set(nices(process.pid).values()) == {0}
    else:
        # Lowering a nice needs privileges: it must say so and keep the originals.
        assert result.returncode == 1
        assert "还原不了" in result.stdout
        assert list((Path(env[2]["X2_DATAPLANE_NICE_STATE"])).glob("*-*.json"))


def test_two_dataplanes_are_refused_without_a_pid(env):
    stand_in, run, _ = env
    first, second = stand_in(), stand_in()
    result = run("apply", check=False)
    assert result.returncode == 1
    assert "找到 2 个 dataplane" in result.stdout
    assert set(nices(first.pid).values()) == set(nices(second.pid).values()) == {0}
    run("apply", "--pid", str(second.pid))
    assert set(nices(first.pid).values()) == {0}
    assert set(nices(second.pid).values()) == {7}


def test_the_hook_is_off_by_default(env):
    stand_in, run, values = env
    process = stand_in()
    result = run("hook")
    assert "关" in result.stdout
    time.sleep(1.5)
    assert set(nices(process.pid).values()) == {0}
    assert not (Path(values["X2_DATAPLANE_NICE_STATE"]) / "watch.pid").exists()


def test_the_watcher_follows_new_threads_and_a_restarted_dataplane(env):
    stand_in, run, values = env
    state = Path(values["X2_DATAPLANE_NICE_STATE"])
    result = run("hook", extra={"X2_DATAPLANE_NICE": "1"})
    assert "开" in result.stdout
    wait_for(lambda: (state / "watch.pid").exists(), "the watcher")
    # Started before the dataplane, as from a start script.
    process = stand_in()
    wait_for(lambda: set(nices(process.pid).values()) == {7}, "the first dataplane reniced")
    more_threads(process)
    wait_for(lambda: len(tids(process.pid)) == 5, "a thread born later")
    wait_for(lambda: set(nices(process.pid).values()) == {7}, "the later thread reniced")
    process.kill()
    process.wait()
    restarted = stand_in()
    wait_for(lambda: set(nices(restarted.pid).values()) == {7}, "the restarted dataplane reniced")
    assert "监视进程：" in run("status").stdout

    # Off again: the watcher stops, so a later dataplane is left as it is.
    run("hook")
    assert not (state / "watch.pid").exists()
    restarted.kill()
    restarted.wait()
    untouched = stand_in()
    time.sleep(1.5)
    assert set(nices(untouched.pid).values()) == {0}


def test_one_watcher_at_a_time_and_it_ends_with_its_dataplane(env):
    stand_in, run, values = env
    process = stand_in()
    watch = subprocess.Popen(
        [sys.executable, str(SCRIPT), "watch", "--linger", "1", "--interval", "0.2"],
        env=values, stdout=subprocess.PIPE, text=True,
    )
    try:
        wait_for(lambda: set(nices(process.pid).values()) == {7}, "reniced")
        assert "已有监视进程在跑" in run("watch").stdout
        process.kill()
        process.wait()
        assert watch.wait(timeout=10) == 0
        assert "dataplane 已退出" in watch.stdout.read()
    finally:
        if watch.poll() is None:
            watch.kill()
            watch.wait()


def test_a_thread_born_after_the_change_gets_the_main_threads_original_back(env):
    """It inherits the changed nice and CPUs; a later apply must not save those as its original."""
    stand_in, run, values = env
    process = stand_in()
    original = affinities(process.pid)[process.pid]
    one_cpu = min(os.sched_getaffinity(0))
    run("apply", "--cpus", str(one_cpu))
    more_threads(process)
    wait_for(lambda: len(tids(process.pid)) == 5, "a thread born after the change")
    born = max(set(tids(process.pid)))
    assert nices(process.pid)[born] == 7 and affinities(process.pid)[born] == {one_cpu}  # inherited
    run("apply", "--cpus", str(one_cpu))  # as the watcher does every second

    # The born thread lives on through the restore.
    result = run("restore", check=False)
    assert born in tids(process.pid)
    assert affinities(process.pid) == {tid: original for tid in tids(process.pid)}
    if ROOT:
        assert result.returncode == 0
        assert set(nices(process.pid).values()) == {0}
    else:
        # The nice cannot come back unprivileged; what was saved for it must be the original.
        assert result.returncode == 1
        (state,) = Path(values["X2_DATAPLANE_NICE_STATE"]).glob("*-*.json")
        saved = json.loads(state.read_text())
        assert saved["threads"][str(born)] == {"nice": 0, "cpus": sorted(original)}
        assert f"线程 {born}" in result.stdout


def load_script(values, monkeypatch):
    for key in ("X2_DATAPLANE_EXE", "X2_DATAPLANE_NICE_STATE", "X2_DATAPLANE_NICE_VALUE"):
        monkeypatch.setenv(key, values[key])
    spec = importlib.util.spec_from_file_location("x2_dataplane_nice_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(os, "setpriority", lambda *a: calls.append(("setpriority", a)))
    monkeypatch.setattr(os, "sched_setaffinity", lambda *a: calls.append(("sched_setaffinity", a)))
    return module, calls


def test_an_explicit_pid_that_is_not_a_dataplane_is_refused_untouched(env, monkeypatch, capsys):
    stand_in, _, values = env
    process = stand_in()
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        module, calls = load_script(values, monkeypatch)
        assert module.main(["apply", "--pid", str(other.pid), "--nice", "7"]) == 1
        assert "不是 dataplane" in capsys.readouterr().out
        # A dataplane chosen earlier that is not the same process now (another start time).
        assert module.apply(process.pid, 7, None, started="0") == (0, 0)
        assert calls == []
        assert not list(Path(values["X2_DATAPLANE_NICE_STATE"]).glob("*-*.json"))
        # The check does not refuse the real one.
        assert module.main(["apply", "--pid", str(process.pid), "--nice", "7"]) == 0
        assert {name for name, _ in calls} == {"setpriority"}
    finally:
        other.kill()
        other.wait()


def test_apply_when_up_waits_for_the_dataplane_and_reads_every_thread_back(env, tmp_path):
    stand_in, _, values = env
    result = tmp_path / "result"
    waiting = subprocess.Popen(
        [sys.executable, str(SCRIPT), "apply-when-up", "--wait", "20", "--settle", "0.3", "--interval", "0.1",
         "--result", str(result)],
        env=values, stdout=subprocess.PIPE, text=True,
    )
    try:
        time.sleep(0.5)  # started before the dataplane, as from a start script
        process = stand_in()
        assert waiting.wait(timeout=20) == 0
        line = result.read_text()
        assert line.startswith("ok nice=7") and "全部 4 个线程" in line, line
        assert line.strip() in waiting.stdout.read()
        assert set(nices(process.pid).values()) == {7}
    finally:
        if waiting.poll() is None:
            waiting.kill()
            waiting.wait()


def test_apply_when_up_fails_out_loud_without_one_dataplane(env, tmp_path):
    stand_in, run, _ = env
    result = tmp_path / "result"
    nothing = run("apply-when-up", "--wait", "0.5", "--interval", "0.1", "--result", str(result), check=False)
    assert nothing.returncode == 1
    assert result.read_text().startswith("FAIL nice=7") and "没等到 dataplane" in result.read_text()

    first, second = stand_in(), stand_in()
    two = run("apply-when-up", "--wait", "1", "--settle", "0.1", "--result", str(result), check=False)
    assert two.returncode == 1
    assert "找到 2 个 dataplane" in result.read_text()
    assert set(nices(first.pid).values()) == set(nices(second.pid).values()) == {0}


@pytest.mark.skipif(ROOT, reason="as root the nice can be lowered")
def test_apply_when_up_says_fail_when_the_threads_are_not_changed(env, tmp_path):
    stand_in, run, _ = env
    process = stand_in()
    result = tmp_path / "result"
    lowered = run("apply-when-up", "--nice", "-5", "--wait", "1", "--settle", "0.1", "--result", str(result),
                  check=False)
    assert lowered.returncode == 1
    assert result.read_text().startswith("FAIL nice=-5") and "4/4 个线程没改成" in result.read_text()
    assert set(nices(process.pid).values()) == {0}


def test_apply_when_up_fails_when_the_dataplane_ends_before_it_is_done(env, tmp_path):
    stand_in, _, values = env
    process = stand_in()
    result = tmp_path / "result"
    waiting = subprocess.Popen(
        [sys.executable, str(SCRIPT), "apply-when-up", "--wait", "5", "--settle", "1.5", "--result", str(result)],
        env=values, stdout=subprocess.PIPE, text=True,
    )
    try:
        wait_for(lambda: "已起" in (waiting.stdout.readline() or ""), "the dataplane found")
        process.kill()
        process.wait()
        assert waiting.wait(timeout=10) == 1
        assert result.read_text().startswith("FAIL") and "已退出" in result.read_text()
    finally:
        if waiting.poll() is None:
            waiting.kill()
            waiting.wait()
