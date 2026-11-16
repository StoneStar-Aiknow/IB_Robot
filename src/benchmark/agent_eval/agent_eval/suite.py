"""One command for an evaluation suite: stack + platform client + in-process episode runner.

    ros2 run agent_eval run_suite --suite <name-or-yaml> [--episodes N] [--seeds S ...] [--out DIR]

The suite file names a platform plugin (entry point group ``agent_eval.platforms``) that
builds the process commands; see :mod:`agent_eval.platform`. A suite is found by path, or by
name under ``share/<package>/config/suites`` of agent_eval or of a package that registers a
plugin (agent_eval_robodojo ships the RoboDojo suites).

The runtime gets the evaluation layer through its profile: the run copies the robot config's
runtime profile into the run directory, lists :data:`EVALUATION_EXTENSION` in the copy's
``simulation.<platform>.extensions`` and points the run-local robot config's
``runtime.profile`` at the copy (a profile with relative file references cannot be copied
this way). Runs outside agent_eval start the shipped profile, without the layer.

Model keys reach the stack only through the environment: the suite may name an
``env_file``, which a shell wrapper sources in the stack process. This module never opens,
reads or prints that file.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .body import load_body, load_robot_document
from .platform import PlatformPlugin, ProcessSpec, load_platform, plugin_config_dirs
from .report import CanonicalRun, build_report, episode_entry, file_sha256, write_report
from .runner import RunnerSettings
from .task_spec import SpecError, _check_keys, config_path, load_recipe, load_task_spec

#: The runtime extension a run lists in its profile copy (see the module doc).
EVALUATION_EXTENSION = "agent_eval.runtime_extension:attach"

_TIMEOUT_DEFAULTS = {
    "stack_ready_s": 300.0,
    "episode_wait_s": 1800.0,
    "native_result_wait_s": 600.0,
    "client_exit_s": 120.0,
}


@dataclass(frozen=True)
class Suite:
    name: str
    platform: str
    robot_config: str
    task: str
    layouts: tuple[dict[str, Any], ...]
    episode_tag: bool = True
    env_file: str | None = None
    # A runtime profile file to use instead of the robot config's (the run copies it as any other).
    runtime_profile: str | None = None
    timeouts: dict[str, float] = field(default_factory=lambda: dict(_TIMEOUT_DEFAULTS))
    platform_options: dict[str, Any] = field(default_factory=dict)
    # Deep-merged over the robot config's ``robot:`` mapping into a run-local copy; None deletes a
    # key. String values may name environment variables (``${VAR}``), e.g. a planner base_url.
    robot_config_overrides: dict[str, Any] = field(default_factory=dict)
    path: str | None = None


def _expand(path: str | None) -> str | None:
    return None if not path else os.path.expandvars(os.path.expanduser(str(path)))


def parse_suite(data: Any, path: str | None = None) -> Suite:
    top = _check_keys(
        "suite",
        data,
        {"schema_version", "kind", "name", "platform", "robot_config", "task", "layouts"},
        {
            "episode_tag",
            "env_file",
            "runtime_profile",
            "timeouts",
            "platform_options",
            "robot_config_overrides",
        },
    )
    if top["schema_version"] != 1 or top["kind"] != "suite":
        raise SpecError("suite: expected schema_version 1 and kind suite")
    timeouts = dict(_TIMEOUT_DEFAULTS)
    for key, value in _check_keys("suite.timeouts", top.get("timeouts") or {}, set(), set(_TIMEOUT_DEFAULTS)).items():
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            raise SpecError(f"suite.timeouts.{key}: expected a positive number")
        timeouts[key] = float(value)
    if not isinstance(top["layouts"], list) or not top["layouts"]:
        raise SpecError("suite.layouts: expected a non-empty list")
    layouts = []
    for index, layout in enumerate(top["layouts"]):
        entry = _check_keys(f"suite.layouts[{index}]", layout, {"name", "seed_start", "episodes"}, {"platform_options"})
        for key in ("seed_start", "episodes"):
            value = entry[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if key == "episodes" else 0):
                raise SpecError(f"suite.layouts[{index}].{key}: expected an integer >= {1 if key == 'episodes' else 0}")
        layouts.append({"platform_options": {}, **dict(entry)})
    return Suite(
        name=str(top["name"]),
        platform=str(top["platform"]),
        robot_config=str(top["robot_config"]),
        task=str(top["task"]),
        layouts=tuple(layouts),
        episode_tag=bool(top.get("episode_tag", True)),
        env_file=_expand(top.get("env_file")),
        runtime_profile=_expand(top.get("runtime_profile")),
        timeouts=timeouts,
        platform_options=dict(top.get("platform_options") or {}),
        robot_config_overrides=_overrides(top.get("robot_config_overrides")),
        path=path,
    )


def _overrides(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SpecError("suite.robot_config_overrides: expected a mapping")
    return dict(value)


def _merged(base: Any, override: Any) -> Any:
    if isinstance(base, Mapping) and isinstance(override, Mapping):
        out = dict(base)
        for key, value in override.items():
            if value is None:
                out.pop(key, None)
            else:
                out[key] = _merged(out.get(key), value)
        return out
    if isinstance(override, str):
        return os.path.expandvars(override)
    if isinstance(override, list):
        return [_merged(None, v) for v in override]
    return override


def with_overrides(suite: Suite, items: Sequence[str]) -> Suite:
    """``PATH=VALUE`` command-line overrides on top of the suite's robot_config_overrides."""
    overrides = dict(suite.robot_config_overrides)
    for item in items:
        path, sep, raw = item.partition("=")
        if not sep or not path.strip():
            raise SpecError(f"--override {item!r}: expected PATH=VALUE")
        node: Any = {}
        leaf = node
        keys = path.strip().split(".")
        for key in keys[:-1]:
            leaf[key] = {}
            leaf = leaf[key]
        leaf[keys[-1]] = yaml.safe_load(raw)
        overrides = _merged_overrides(overrides, node)
    return replace(suite, robot_config_overrides=overrides)


def _merged_overrides(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merged_overrides(out[key], value)
        else:
            out[key] = value
    return out


def derived_robot_config(suite: Suite, out_dir: Path) -> Suite:
    """Write the run-local robot config when the suite overrides any of it; the suite then points at it."""
    if not suite.robot_config_overrides:
        return suite
    source, robot = load_robot_document(suite.robot_config)
    with open(source, encoding="utf-8") as fh:
        document = yaml.safe_load(fh) or {}
    document["robot"] = _merged(robot, suite.robot_config_overrides)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"robot_config.{robot.get('name', 'robot')}.yaml"
    path.write_text(
        f"# Derived from {source} with suite {suite.name} robot_config_overrides.\n"
        + yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    return replace(suite, robot_config=str(path))


def suite_path(name_or_path: str) -> Path:
    """A suite file: a path, or a name in agent_eval's or a platform plugin package's config/suites."""
    try:
        return config_path("suites", name_or_path)
    except FileNotFoundError as missing:
        file_name = name_or_path if name_or_path.endswith(".yaml") else f"{name_or_path}.yaml"
        for directory in plugin_config_dirs("suites"):
            if (directory / file_name).is_file():
                return (directory / file_name).resolve()
        raise FileNotFoundError(f"{missing} nor in the platform plugins' config/suites") from None


def load_suite(name_or_path: str) -> Suite:
    path = suite_path(name_or_path)
    with open(path, encoding="utf-8") as fh:
        return parse_suite(yaml.safe_load(fh), str(path))


def evaluation_profile(source: str | os.PathLike, robot: Mapping[str, Any], out_dir: Path) -> Path:
    """Copy the runtime profile into the run with the evaluation extension listed for its platform."""
    source = Path(source)
    with open(source, encoding="utf-8") as fh:
        document = yaml.safe_load(fh) or {}
    platform = str((robot.get("simulation") or {}).get("platform") or document.get("sim_platform") or "")
    if not platform:
        raise SpecError(
            f"cannot attach the evaluation layer: neither the robot config (simulation.platform) nor {source} "
            "(sim_platform) names the simulation platform"
        )
    simulation = document.setdefault("simulation", {})
    section = simulation.setdefault(platform, {})
    if not isinstance(section, dict):
        raise SpecError(f"{source}: simulation.{platform} must be a mapping")
    extensions = [str(e) for e in section.get("extensions") or []]
    section["extensions"] = extensions + ([EVALUATION_EXTENSION] if EVALUATION_EXTENSION not in extensions else [])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"runtime_profile.{source.stem}.yaml"
    path.write_text(
        f"# Copied from {source} by agent_eval run_suite: simulation.{platform}.extensions lists the "
        "evaluation layer.\n" + yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    return path


def evaluation_robot_config(suite: Suite, out_dir: Path) -> tuple[Suite, dict[str, Any]]:
    """The run's suite: its robot config file derived with the overrides and the profile copy.

    Returns the suite pointing at the run-local robot config and what the copy came from.
    """
    configured = derived_robot_config(suite, out_dir)
    source = Path(load_body(configured.robot_config, profile_path=suite.runtime_profile).runtime_profile_path)
    _, robot = load_robot_document(configured.robot_config)
    copy = evaluation_profile(source, robot, out_dir)
    overrides = _merged_overrides(suite.robot_config_overrides, {"runtime": {"profile": str(copy)}})
    evaluated = derived_robot_config(replace(suite, robot_config_overrides=overrides, runtime_profile=None), out_dir)
    return evaluated, {"source": str(source), "source_sha256": file_sha256(source), "copy": str(copy)}


def seed_runs(layout: Mapping[str, Any], episodes: int | None = None, seeds: Sequence[int] | None = None):
    """(seed, count) client runs for a layout: its range, or the given seeds grouped when consecutive."""
    if not seeds:
        return [(int(layout["seed_start"]), int(episodes or layout["episodes"]))]
    runs: list[list[int]] = []
    for seed in seeds:
        if runs and seed == runs[-1][0] + runs[-1][1]:
            runs[-1][1] += 1
        else:
            runs.append([int(seed), 1])
    return [tuple(run) for run in runs]


def with_env_file(argv: Sequence[str], env_file: str | None) -> list[str]:
    """Wrap ``argv`` so a shell sources ``env_file`` (exporting its variables) and execs it."""
    if not env_file:
        return list(argv)
    script = 'set -a; . "$1" || exit 97; set +a; shift; exec "$@"'
    return ["bash", "-c", script, "agent_eval-env", env_file, *argv]


class ManagedProcess:
    """A child in its own process group, logged to a file, stopped with SIGINT then SIGTERM/SIGKILL."""

    def __init__(self, label: str, spec: ProcessSpec, log_path: Path, base_env: Mapping[str, str]):
        self.label = label
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_path, "ab")  # noqa: SIM115 - closed in stop()
        env = {**base_env, **spec.env}
        self.popen = subprocess.Popen(
            spec.argv, cwd=spec.cwd, env=env, stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True
        )
        self.log_path = log_path

    def wait(self, timeout_s: float) -> int | None:
        try:
            return self.popen.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return None

    def stop(self, grace_s: float = 10.0) -> int | None:
        for sig, wait_s in ((signal.SIGINT, grace_s), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
            if self.popen.poll() is not None:
                break
            try:
                os.killpg(self.popen.pid, sig)
            except ProcessLookupError:
                break
            self.wait(wait_s)
        self._log.close()
        return self.popen.poll()


SessionFactory = Callable[..., Any]

# Variables a platform client may inherit; everything else (PYTHONPATH, VIRTUAL_ENV, ROS
# overlays, AMENT/COLCON prefixes, stale CONDA_* of another env) is dropped.
CLIENT_ENV_BASE = (
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    "LANG",
    "LC_ALL",
    "TZ",
    "DISPLAY",
    "XAUTHORITY",
    "XDG_RUNTIME_DIR",
    "TMPDIR",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "all_proxy",
    "ALL_PROXY",
)
CLIENT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def isolated_env(environ: Mapping[str, str], passthrough: Iterable[str] = ()) -> dict[str, str]:
    """A clean environment for a process of another interpreter: allowlisted variables only."""
    keep = set(CLIENT_ENV_BASE) | set(passthrough)
    env = {k: v for k, v in environ.items() if k in keep}
    env["PATH"] = CLIENT_PATH
    return env


def _planner_cameras(robot: Mapping[str, Any]) -> tuple[str, ...]:
    """Camera topics the Agent's planner attaches (robot config ``embodied.agent.planner.vision_topics``)."""
    planner = ((robot.get("embodied") or {}).get("agent") or {}).get("planner") or {}
    return tuple(str(t) for t in planner.get("vision_topics") or ())


def _exit_check(**processes: ManagedProcess) -> Callable[[], str | None]:
    """Why waiting for episodes is pointless: one of these processes has exited."""

    def check() -> str | None:
        for label, process in processes.items():
            code = process.popen.poll()
            if code is not None:
                return f"{label} exited with code {code} (log {process.log_path})"
        return None

    return check


def _wait_ready(session, stack: ManagedProcess, timeout_s: float) -> bool:
    """Wait for the Agent, but stop as soon as the stack process has died."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        code = stack.popen.poll()
        if code is not None:
            raise RuntimeError(f"stack exited with code {code} before the Agent was ready (log {stack.log_path})")
        if session.wait_agent_ready(min(5.0, max(0.1, deadline - time.monotonic()))):
            return True
    return False


def _default_session_factory(body, recipe_facts, settings):
    from .runner_node import RunnerSession

    return RunnerSession(body, recipe_facts, settings)


def new_run_id(suite_name: str) -> str:
    return f"{suite_name}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"


def run_suite(
    suite: Suite,
    *,
    out_dir: str | os.PathLike,
    episodes: int | None = None,
    seeds: Sequence[int] | None = None,
    repo_dir: str | os.PathLike | None = None,
    plugin: PlatformPlugin | None = None,
    session_factory: SessionFactory = _default_session_factory,
    run_id: str | None = None,
    log: Callable[[str], None] = print,
) -> tuple[dict[str, Any], Path]:
    """Run the suite and write its report; returns (report, report path)."""
    run_id = run_id or new_run_id(suite.name)
    out = Path(out_dir) / run_id
    suite, profile_copy = evaluation_robot_config(suite, out)
    spec = load_task_spec(suite.task)
    recipe = load_recipe(spec.recipe)
    body = load_body(suite.robot_config)
    _, robot = load_robot_document(suite.robot_config)
    plugin = plugin or load_platform(suite.platform, suite.platform_options)
    logs = out / "logs"
    base_env = dict(os.environ)
    # The platform client runs another interpreter (its own Python / conda env): it gets a
    # clean environment, never this process's ROS / venv paths.
    client_env = isolated_env(base_env, plugin.client_env_passthrough())
    timeouts = suite.timeouts
    entries: list[dict[str, Any]] = []
    layouts_run: list[dict[str, Any]] = []
    error = None
    # One canonical task per layout entry of the suite (a layout listed twice is two tasks).
    planned = {
        index: {
            "task_name": str(layout["name"]),
            "planned_episodes": sum(count for _seed, count in seed_runs(layout, episodes, seeds)),
        }
        for index, layout in enumerate(suite.layouts)
    }
    canonical = CanonicalRun(out / "canonical", run_id=run_id, suite=suite.name, planned=planned)
    stack_spec = plugin.stack_command(suite)
    stack_spec = ProcessSpec(
        with_env_file(stack_spec.argv, suite.env_file), stack_spec.env, stack_spec.cwd, stack_spec.ready_timeout_s
    )
    log(f"[{run_id}] stack: {stack_spec.argv} (log {logs / 'stack.log'})")
    stack = ManagedProcess("stack", stack_spec, logs / "stack.log", base_env)
    try:
        settings = RunnerSettings(
            out_dir=str(out / "episodes"),
            max_decisions=spec.max_decisions,
            decision_timeout_s=spec.decision_timeout_s,
            observation_fields=spec.observation_fields,
            episode_tag=suite.episode_tag,
            run_id=run_id,
            end_wait_s=timeouts["native_result_wait_s"],
            camera_topics=_planner_cameras(robot),
        )
        with session_factory(body, recipe.facts(), settings) as session:
            if not _wait_ready(session, stack, timeouts["stack_ready_s"]):
                raise RuntimeError("the Agent node did not become ready")
            pass_index = 0
            for task_id, layout in enumerate(suite.layouts):
                for seed, count in seed_runs(layout, episodes, seeds):
                    pass_index += 1
                    # A layout may run twice (E2): every client pass gets its own id, so platform
                    # artifacts, videos and logs of the two passes never share a path.
                    pass_id = f"p{pass_index:02d}"
                    client_spec = plugin.client_command(
                        suite, layout=layout, seed=seed, episodes=count, run_id=run_id, pass_id=pass_id
                    )
                    client_log = logs / f"client_{pass_id}_{layout['name']}_{seed}.log"
                    log(f"[{run_id}] client {layout['name']} seeds {seed}..{seed + count - 1}: {client_spec.argv}")
                    client = ManagedProcess("client", client_spec, client_log, client_env)
                    try:
                        records = session.run_episodes(
                            count, timeouts["episode_wait_s"], abort=_exit_check(client=client, stack=stack)
                        )
                    finally:
                        if client.wait(timeouts["client_exit_s"]) is None:
                            log(f"[{run_id}] client did not exit in time; stopping it")
                        exit_code = client.stop()
                    video_dir = plugin.video_dir(suite, run_id=run_id, layout=layout, seed=seed, pass_id=pass_id)
                    for offset, record in enumerate(records):
                        entry = episode_entry(
                            record, layout=str(layout["name"]), seed=seed + offset, video_dir=video_dir
                        )
                        entry["pass"] = pass_id
                        entry["artifacts"]["video_episode_index"] = offset
                        entries.append(entry)
                        canonical.append(entry, task_id)
                    layouts_run.append(
                        {
                            "pass": pass_id,
                            "name": layout["name"],
                            "seed_start": seed,
                            "episodes": count,
                            "client_exit": exit_code,
                            "client_log": str(client_log),
                            "native_artifact_dir": plugin.native_artifact_dir(suite, run_id=run_id, layout=layout),
                        }
                    )
    except Exception as exc:  # the report still records what ran
        error = f"{type(exc).__name__}: {exc}"
        log(f"[{run_id}] suite stopped: {error}")
    finally:
        stack.stop()
    summary = canonical.finalize(stopped=error is not None)
    report = build_report(
        run_id=run_id,
        repo_dir=repo_dir or Path.cwd(),
        body=body,
        robot=robot,
        task_spec=spec,
        recipe=recipe,
        layouts=layouts_run,
        episodes=entries,
        summary=summary,
        suite_path=suite.path,
        platform_identity=plugin.platform_identity(suite),
        extra={
            "suite": suite.name,
            "platform": suite.platform,
            "env_file": suite.env_file,  # the path only; never its content
            "logs_dir": str(logs),
            "runtime_profile_copy": profile_copy,
            "canonical_dir": str(canonical.writer.run_root),
            "error": error,
        },
    )
    report_path = write_report(report, out)
    agg = report["aggregate"]
    counters = agg["counters"]
    log(
        f"[{run_id}] report: {report_path} ({agg['run_status']}): {counters['successful_native_episodes']} "
        f"successes of {counters['completed_native_episodes']} judged, {counters['planned_episodes']} planned; "
        f"{agg['interrupted']} interrupted, {agg['unjudged']} unjudged"
    )
    return report, report_path


def _int_list(values: Iterable[str]) -> list[int]:
    return [int(v) for v in values]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an agent_eval suite (stack, platform client, runner).")
    parser.add_argument(
        "--suite", required=True, help="suite name (config/suites of agent_eval or a platform plugin) or yaml path"
    )
    parser.add_argument("--episodes", type=int, default=None, help="episodes per layout (overrides the suite)")
    parser.add_argument("--seeds", nargs="+", default=None, help="explicit seeds for every layout")
    parser.add_argument("--out", default="agent_eval_runs")
    parser.add_argument("--repo-dir", default=None, help="git work tree recorded in the report (default: cwd)")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="robot config override below robot:, e.g. embodied.agent.planner.model=m (VALUE is YAML; "
        "repeatable; recorded in the report through the derived config's hash)",
    )
    args = parser.parse_args(argv)
    suite = with_overrides(load_suite(args.suite), args.override)
    report, _ = run_suite(
        suite,
        out_dir=args.out,
        episodes=args.episodes,
        seeds=_int_list(args.seeds) if args.seeds else None,
        repo_dir=args.repo_dir,
    )
    return 1 if report.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
