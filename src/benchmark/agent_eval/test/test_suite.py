"""Suite orchestration: plugin lookup, process lifecycle, secrets pass-through, report on disk."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml
from agent_eval.platform import PlatformNotFound, PlatformPlugin, ProcessSpec, load_platform
from agent_eval.suite import load_suite, parse_suite, run_suite, seed_runs, with_env_file
from agent_eval_testkit import CONFIG, FIXTURES


def suite_file(tmp_path, **overrides):
    data = {
        "schema_version": 1,
        "kind": "suite",
        "name": "fake_suite",
        "platform": "fake",
        "robot_config": str(FIXTURES / "robots" / "dual_arm_fixture.yaml"),
        "runtime_profile": str(FIXTURES / "profiles" / "dual_arm_fixture.yaml"),
        "task": str(CONFIG / "tasks" / "general_pickup.yaml"),
        "timeouts": {"stack_ready_s": 5, "episode_wait_s": 5, "native_result_wait_s": 5, "client_exit_s": 10},
        "layouts": [{"name": "l1", "seed_start": 10, "episodes": 2}, {"name": "l2", "seed_start": 0, "episodes": 1}],
    }
    data.update(overrides)
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_suite(str(path))


class CpuFakePlatform(PlatformPlugin):
    """CPU stand-in: the "stack" and "client" are short Python processes.

    ``options``: ``stack_seconds`` (stack lifetime, default 60), ``client_seconds`` (default 0).
    """

    name = "fake"

    def stack_command(self, suite) -> ProcessSpec:
        seconds = float(self.options.get("stack_seconds", 60.0))
        return ProcessSpec(
            [sys.executable, "-c", f"import time; print('fake stack up', flush=True); time.sleep({seconds})"]
        )

    def client_command(self, suite, *, layout, seed, episodes, run_id, pass_id) -> ProcessSpec:
        seconds = float(self.options.get("client_seconds", 0.0))
        code = f"import sys, time; print('fake client', sys.argv[1:], flush=True); time.sleep({seconds})"
        return ProcessSpec(
            [sys.executable, "-c", code, str(layout.get("name")), str(seed), str(episodes), run_id, pass_id]
        )

    def native_artifact_dir(self, suite, *, run_id, layout) -> str | None:
        return f"/tmp/agent_eval_fake/{run_id}/{layout.get('name')}"


class FakeSession:
    """Stands in for RunnerSession: hands back one record per requested episode."""

    instances: list[FakeSession] = []

    def __init__(self, body, recipe_facts, settings):
        self.settings, self.calls = settings, []
        self.entered = self.exited = False
        FakeSession.instances.append(self)

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True

    def wait_agent_ready(self, timeout_s):
        return True

    def run_episodes(self, episodes, episode_wait_s, abort=None):
        self.calls.append(episodes)
        out = []
        for i in range(episodes):
            n = sum(self.calls) - episodes + i
            out.append(
                {
                    "session_id": f"s{n:04d}",
                    "generation": n,
                    "finish_reason": "agent_done",
                    "decisions": [],
                    "native_result": {"native_success": n % 2 == 0, "native_score": None},
                }
            )
        return out


class SecretAwarePlatform(CpuFakePlatform):
    def stack_command(self, suite):
        code = "import os; print('has secret:', bool(os.environ.get('FAKE_PLANNER_KEY')), flush=True)"
        return ProcessSpec([sys.executable, "-c", code])


def test_only_registered_plugins_load():
    # Registration by entry point is covered by agent_eval_robodojo's test_platform.
    with pytest.raises(PlatformNotFound):
        load_platform("os.system", {})


def test_seed_overrides_group_consecutive_seeds():
    layout = {"seed_start": 5, "episodes": 3}
    assert seed_runs(layout) == [(5, 3)]
    assert seed_runs(layout, episodes=1) == [(5, 1)]
    assert seed_runs(layout, seeds=[0, 1, 2, 7, 9, 10]) == [(0, 3), (7, 1), (9, 2)]


def test_suite_schema_is_strict(tmp_path):
    with pytest.raises(ValueError, match="unknown keys"):
        suite_file(tmp_path, api_key="sk-123")
    with pytest.raises(ValueError, match="unknown keys"):
        suite_file(tmp_path, context_mode="legacy")
    with pytest.raises(ValueError):
        parse_suite({"schema_version": 1, "kind": "suite"})


def test_env_file_is_sourced_by_the_child_only(tmp_path):
    argv = with_env_file(["prog", "--x"], "/path/to.env")
    assert argv[0] == "bash" and argv[-2:] == ["prog", "--x"] and "/path/to.env" in argv
    assert with_env_file(["prog"], None) == ["prog"]


def test_run_suite_orchestrates_processes_and_writes_the_report(tmp_path):
    env_file = tmp_path / "planner.env"
    env_file.write_text("FAKE_PLANNER_KEY=topsecret-value\n", encoding="utf-8")
    suite = suite_file(tmp_path, env_file=str(env_file))
    FakeSession.instances.clear()
    lines = []
    report, path = run_suite(
        suite,
        out_dir=tmp_path / "runs",
        plugin=SecretAwarePlatform({"client_seconds": 0}),
        session_factory=FakeSession,
        run_id="run1",
        repo_dir=tmp_path,
        log=lines.append,
    )
    session = FakeSession.instances[0]
    assert session.entered and session.exited
    assert session.calls == [2, 1]
    assert session.settings.max_decisions == 40 and session.settings.run_id == "run1"
    assert report["error"] is None
    assert [(e["layout"], e["seed"]) for e in report["episodes"]] == [("l1", 10), ("l1", 11), ("l2", 0)]
    counters = report["aggregate"]["counters"]
    assert counters["successful_native_episodes"] == 2 and counters["completed_native_episodes"] == 3
    assert report["aggregate"]["run_status"] == "complete"
    canonical = tmp_path / "runs" / "run1" / "canonical"
    assert len((canonical / "episodes.jsonl").read_text().splitlines()) == 3
    assert json.loads((canonical / "summary.json").read_text())["counters"]["planned_episodes"] == 3
    assert report["planner"]["model"] == "fixture-model"
    assert [layout["client_exit"] for layout in report["layouts"]] == [0, 0]
    stack_log = (tmp_path / "runs" / "run1" / "logs" / "stack.log").read_text()
    assert "has secret: True" in stack_log
    everything = (
        path.read_text()
        + "\n".join(lines)
        + "".join(p.read_text() for p in (tmp_path / "runs" / "run1" / "logs").glob("client_*.log"))
    )
    assert "topsecret" not in everything
    saved = json.loads(path.read_text())
    assert saved["env_file"] == str(env_file)


def test_a_failing_session_still_writes_a_report_and_stops_the_stack(tmp_path):
    class NotReady(FakeSession):
        def wait_agent_ready(self, timeout_s):
            return False

    suite = suite_file(tmp_path)
    report, path = run_suite(
        suite,
        out_dir=tmp_path / "runs",
        plugin=CpuFakePlatform({"stack_seconds": 30}),
        session_factory=NotReady,
        run_id="run2",
        repo_dir=tmp_path,
        log=lambda _t: None,
    )
    assert "did not become ready" in report["error"]
    assert report["episodes"] == [] and path.is_file()


class WaitingSession(FakeSession):
    """Waits for episodes on a real EventBook, as RunnerSession does, so process exits matter."""

    def run_episodes(self, episodes, episode_wait_s, abort=None):
        from agent_eval.runner import EventBook
        from agent_eval_testkit import fixture_body

        book = EventBook(fixture_body())
        book.set_abort_check(abort)
        book.next_episode(episode_wait_s)
        raise AssertionError("no episode can start in this test")


def test_a_client_that_exits_before_any_episode_fails_the_suite_fast(tmp_path):
    import time

    suite = suite_file(
        tmp_path,
        timeouts={"stack_ready_s": 5, "episode_wait_s": 120, "native_result_wait_s": 5, "client_exit_s": 5},
    )
    t0 = time.monotonic()
    report, _ = run_suite(
        suite,
        out_dir=tmp_path / "out",
        plugin=CpuFakePlatform({"client_seconds": 0, "stack_seconds": 120}),
        session_factory=WaitingSession,
        log=lambda _m: None,
    )
    assert time.monotonic() - t0 < 30.0  # not the 120 s episode wait
    assert "client exited with code 0" in report["error"]
    stack_log = next((tmp_path / "out").glob("*/logs/stack.log"))
    assert "fake stack up" in stack_log.read_text()


def test_a_stack_that_dies_before_the_agent_is_ready_fails_fast(tmp_path):
    import time

    class NeverReady(FakeSession):
        def wait_agent_ready(self, timeout_s):
            time.sleep(min(timeout_s, 0.2))
            return False

    suite = suite_file(
        tmp_path,
        timeouts={"stack_ready_s": 120, "episode_wait_s": 5, "native_result_wait_s": 5, "client_exit_s": 5},
    )
    t0 = time.monotonic()
    report, _ = run_suite(
        suite,
        out_dir=tmp_path / "out",
        plugin=CpuFakePlatform({"stack_seconds": 0}),
        session_factory=NeverReady,
        log=lambda _m: None,
    )
    assert time.monotonic() - t0 < 30.0
    assert "stack exited" in report["error"]


def test_the_platform_client_gets_a_clean_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/ros/venv/lib/python3.10/site-packages")
    monkeypatch.setenv("VIRTUAL_ENV", "/ros/venv")
    monkeypatch.setenv("AMENT_PREFIX_PATH", "/ros/install")
    monkeypatch.setenv("https_proxy", "http://proxy.local:1")

    class EnvDumpPlatform(CpuFakePlatform):
        def client_command(self, suite, *, layout, seed, episodes, run_id, pass_id):
            code = "import json, os; print('ENV=' + json.dumps(dict(os.environ)), flush=True)"
            return ProcessSpec([sys.executable, "-I", "-c", code], env={"PLATFORM_ROOT": "/p"})

    suite = suite_file(tmp_path, layouts=[{"name": "l1", "seed_start": 0, "episodes": 1}])
    run_suite(
        suite,
        out_dir=tmp_path / "out",
        session_factory=FakeSession,
        plugin=EnvDumpPlatform({}),
        log=lambda _m: None,
    )
    log = next((tmp_path / "out").glob("*/logs/client_p01_l1_0.log")).read_text()
    env = json.loads(log.split("ENV=", 1)[1].splitlines()[0])
    assert not {"PYTHONPATH", "VIRTUAL_ENV", "AMENT_PREFIX_PATH"} & set(env)
    assert env["https_proxy"] == "http://proxy.local:1" and env["PLATFORM_ROOT"] == "/p"
    assert not [name for name in env if name.startswith("AGENT_EVAL")]


def test_the_run_starts_the_runtime_with_a_profile_copy_that_lists_the_evaluation_layer(tmp_path):
    from agent_eval.suite import EVALUATION_EXTENSION

    source = FIXTURES / "profiles" / "dual_arm_fixture.yaml"
    report, _ = run_suite(
        suite_file(tmp_path),
        out_dir=tmp_path / "out",
        plugin=CpuFakePlatform({}),
        session_factory=FakeSession,
        run_id="r",
        log=lambda _m: None,
    )
    copy = report["runtime_profile_copy"]
    assert copy["source"] == str(source.resolve()) and copy["source_sha256"]
    profile = yaml.safe_load(Path(copy["copy"]).read_text(encoding="utf-8"))
    assert profile["simulation"]["fixture"]["extensions"] == [EVALUATION_EXTENSION]
    assert profile["motion_groups"] == yaml.safe_load(source.read_text())["motion_groups"]
    # The run-local robot config the stack launches points at the copy; the runner reads the same file.
    derived = tmp_path / "out" / "r" / "robot_config.dual_arm_fixture.yaml"
    assert yaml.safe_load(derived.read_text())["robot"]["runtime"]["profile"] == copy["copy"]
    assert report["runtime_profile"]["profile"] == copy["copy"]
    assert yaml.safe_load(source.read_text()).get("simulation") is None  # the shipped profile is untouched


def test_a_profile_without_a_simulation_platform_cannot_get_the_evaluation_layer(tmp_path):
    from agent_eval.task_spec import SpecError

    profile = tmp_path / "plain.yaml"
    profile.write_text(
        yaml.safe_dump({"motion_groups": {"left_arm": {"joints": ["a"]}, "right_arm": {"joints": ["b"]}}})
    )
    with pytest.raises(SpecError, match="simulation platform"):
        run_suite(
            suite_file(tmp_path, runtime_profile=str(profile)),
            out_dir=tmp_path / "out",
            plugin=CpuFakePlatform({}),
            session_factory=FakeSession,
            log=lambda _m: None,
        )


def test_two_passes_of_one_layout_keep_separate_logs_and_pass_ids(tmp_path):
    layouts = [{"name": "l1", "seed_start": 0, "episodes": 1}, {"name": "l1", "seed_start": 0, "episodes": 1}]
    report, _ = run_suite(
        suite_file(tmp_path, layouts=layouts),
        out_dir=tmp_path / "out",
        plugin=CpuFakePlatform({}),
        session_factory=FakeSession,
        log=lambda _m: None,
    )
    assert [e["pass"] for e in report["episodes"]] == ["p01", "p02"]
    assert len({layout["client_log"] for layout in report["layouts"]}) == 2
    assert report["recipe_task_hash"] and report["full_prompt_hash"] is None and report["full_prompt_hash_reason"]
