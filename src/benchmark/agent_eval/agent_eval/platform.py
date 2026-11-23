"""Platform plugins: how a suite starts the robot stack and the platform's episode client.

A plugin is a trusted, installed class registered under the Python entry point group
``agent_eval.platforms`` (``setup.py``: ``entry_points={"agent_eval.platforms": ["name =
pkg.module:Class"]}``). The suite names the plugin; it is never loaded from Agent text or
from a module path in a config file.

What a plugin (e.g. RoboDojo's, shipped by ``agent_eval_robodojo``) must provide:

* ``stack_command(suite)``: the IB-Robot stack launch (runtime + embodied pipeline + Agent)
  of ``suite.robot_config`` (the run-local file whose runtime profile lists agent_eval's
  runtime extension), started once per suite and stopped at the end. The stack must expose
  the direct-control runtime's session surface (``/simulation/session``,
  ``/simulation/clock``, ``/simulation/finish_session``, ``/simulation/abort_session``), the
  extension's ``/evaluation/admit_episode`` and ``/evaluation/native_result``, and the Agent
  public entry.
* ``client_command(suite, layout=..., seed=..., episodes=..., run_id=..., pass_id=...)``: the
  platform's evaluation client for one layout pass (``pass_id`` is unique per pass of the
  run): it must run ``episodes`` consecutive seeds starting at ``seed`` and exit.
* ``native_artifact_dir(suite, run_id=..., layout=...)``: where the platform writes its own
  artifacts (score files, videos), or ``None``.
* optionally ``video_dir(suite, run_id=..., layout=..., seed=..., pass_id=...)``.

A plugin's package may ship suites in ``share/<package>/config/suites`` (found by name).

Commands are argv lists (no shell). Secrets are not the plugin's business: the suite's
``env_file`` is sourced by a wrapper around the stack command only.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

ENTRY_POINT_GROUP = "agent_eval.platforms"


@dataclass(frozen=True)
class ProcessSpec:
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)  # added to the suite's environment
    cwd: str | None = None
    ready_timeout_s: float = 0.0  # seconds to wait after start before continuing


class PlatformPlugin:
    """Base class; subclasses override the command builders."""

    name = ""

    def __init__(self, options: Mapping[str, Any] | None = None):
        self.options = dict(options or {})

    def stack_command(self, suite) -> ProcessSpec:
        raise NotImplementedError

    def client_command(
        self, suite, *, layout: Mapping[str, Any], seed: int, episodes: int, run_id: str, pass_id: str
    ) -> ProcessSpec:
        raise NotImplementedError

    def native_artifact_dir(self, suite, *, run_id: str, layout: Mapping[str, Any]) -> str | None:
        return None

    def platform_identity(self, suite) -> dict[str, Any]:
        """What identifies the platform build the run used (repo commit, task config hashes...)."""
        return {}

    def client_env_passthrough(self) -> tuple[str, ...]:
        """Extra variables the client may inherit; it otherwise starts from a clean environment."""
        return ()

    def video_dir(self, suite, *, run_id: str, layout: Mapping[str, Any], seed: int, pass_id: str) -> str | None:
        return None


class PlatformNotFound(LookupError):
    pass


def plugin_config_dirs(kind: str) -> list[Path]:
    """``share/<package>/config/<kind>`` of every package that registers a platform plugin."""
    dirs = []
    with suppress(ImportError):
        from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

        for package in sorted({ep.value.split(":")[0].split(".")[0] for ep in entry_points(group=ENTRY_POINT_GROUP)}):
            with suppress(PackageNotFoundError):
                directory = Path(get_package_share_directory(package)) / "config" / kind
                if directory.is_dir():
                    dirs.append(directory)
    return dirs


def load_platform(name: str, options: Mapping[str, Any] | None = None) -> PlatformPlugin:
    """Instantiate the installed plugin registered as ``name``."""
    matches = [ep for ep in entry_points(group=ENTRY_POINT_GROUP) if ep.name == name]
    if not matches:
        known = sorted(ep.name for ep in entry_points(group=ENTRY_POINT_GROUP))
        raise PlatformNotFound(f"no platform plugin {name!r} in {ENTRY_POINT_GROUP} (installed: {known})")
    cls = matches[0].load()
    if not (isinstance(cls, type) and issubclass(cls, PlatformPlugin)):
        raise PlatformNotFound(f"{ENTRY_POINT_GROUP}:{name} is not a PlatformPlugin subclass")
    return cls(options)
