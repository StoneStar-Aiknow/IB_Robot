"""agent_eval platform plugin for RoboDojo (entry point ``agent_eval.platforms: robodojo``).

Builds the two processes of a RoboDojo suite:

* the IB-Robot stack: ``embodied_pipeline.launch.py`` with the suite's run-local robot config,
  which starts robodojo_robot's runtime (robot_runtime_sim's direct-control runtime with the
  RoboDojo adapter, and agent_eval's evaluation extension listed in the run's profile copy)
  through ``runtime.provider``, plus the Gateway, safety_guard, task_dispatch and the Agent;
* the RoboDojo eval client (Isaac Sim, its own conda env): robodojo_robot's
  ``run_eval_client.sh`` for one layout pass, ``EVAL_NUM`` episodes from the layout's ``SEED``.

The plugin lives here, not in robodojo_robot, so the platform package never depends on the
evaluation layer.

Machine-specific locations come from options or the environment, never from code:
``robodojo_root`` / ``ROBODOJO_ROOT``, ``conda_sh`` / ``CONDA_SH``, ``conda_env``, ``gpu``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agent_eval.platform import PlatformPlugin, ProcessSpec

CLIENT_SCRIPT = "run_eval_client.sh"
# Must match run_eval_client.sh (POLICY_NAME default, embodiment and additional_info arguments).
POLICY_NAME = "ibrobot_direct_control"
EMBODIMENT = "arx_x5"
ADDITIONAL_INFO = "ckpt_name=ibrobot,action_type=joint"


def _client_dir() -> Path:
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("robodojo_robot")) / "robodojo_client"


class RoboDojoPlatform(PlatformPlugin):
    name = "robodojo"

    def _option(self, layout: Mapping[str, Any] | None, key: str, env: str | None = None, default: Any = None):
        layered = {**self.options, **dict((layout or {}).get("platform_options") or {})}
        if layered.get(key) not in (None, ""):
            return layered[key]
        if env and os.environ.get(env):
            return os.environ[env]
        return default

    def _robodojo_root(self, layout=None) -> str:
        root = self._option(layout, "robodojo_root", "ROBODOJO_ROOT")
        if not root:
            raise RuntimeError("RoboDojo suite needs platform_options.robodojo_root or ROBODOJO_ROOT")
        return os.path.expanduser(str(root))

    def stack_command(self, suite) -> ProcessSpec:
        # run_suite points the suite at its run-local robot config file: launch it by path.
        path = Path(os.path.expanduser(str(suite.robot_config)))
        config = [f"config_path:={path}"] if path.is_file() else [f"robot_config:={suite.robot_config}"]
        argv = [
            "ros2",
            "launch",
            "embodied_bringup",
            "embodied_pipeline.launch.py",
            *config,
            "control_mode:=moveit_planning",
            "with_embodied:=true",
            "entry_mode:=agent",
            "authorize_motion:=true",
        ]
        return ProcessSpec(argv=argv, env={"ROBODOJO_ROOT": self._robodojo_root()}, ready_timeout_s=10.0)

    def client_command(
        self, suite, *, layout: Mapping[str, Any], seed: int, episodes: int, run_id: str, pass_id: str
    ) -> ProcessSpec:
        out = {
            "ROBODOJO_ROOT": self._robodojo_root(layout),
            "TASK": self._task(suite, layout),
            "SEED": str(seed),
            "EVAL_NUM": str(episodes),
            "GPU": str(self._option(layout, "gpu", default=0)),
            "PORT": str(self._option(layout, "port", default=34950)),
            "CONDA_ENV": str(self._option(layout, "conda_env", default="RoboDojo")),
            # One RoboDojo run id per client pass: its videos, _result.json and resume manifest
            # live under it, so a layout run twice never overwrites the first pass.
            "ROBODOJO_RUN_ID": f"{run_id}_{pass_id}_{layout['name']}_{seed}",
            "IBROBOT_RUNTIME_ARTIFACT_DIR": str(self._artifact_root(layout, run_id)),
        }
        conda_sh = self._option(layout, "conda_sh", "CONDA_SH")
        if conda_sh:
            out["CONDA_SH"] = os.path.expanduser(str(conda_sh))
        return ProcessSpec(argv=["bash", str(_client_dir() / CLIENT_SCRIPT)], env=out)

    def _artifact_root(self, layout, run_id: str) -> Path:
        base = self._option(layout, "artifact_dir")
        if base:
            return Path(os.path.expanduser(str(base))) / run_id
        return Path(self._robodojo_root(layout)) / "eval_result" / POLICY_NAME

    def native_artifact_dir(self, suite, *, run_id: str, layout: Mapping[str, Any]) -> str | None:
        return str(self._artifact_root(layout, run_id))

    def video_dir(self, suite, *, run_id: str, layout: Mapping[str, Any], seed: int, pass_id: str) -> str | None:
        """RoboDojo's directory of this pass: episode_<index>_<camera>_<success|fail>.mp4 + _result.json."""
        task = self._task(suite, layout)
        return str(
            Path(self._robodojo_root(layout))
            / "eval_result"
            / "RoboDojo"
            / task
            / POLICY_NAME
            / EMBODIMENT
            / f"{seed}_{ADDITIONAL_INFO}"
            / f"{run_id}_{pass_id}_{layout['name']}_{seed}"
        )

    def _task(self, suite, layout) -> str:
        from agent_eval.task_spec import load_task_spec

        return str(self._option(layout, "task", default=load_task_spec(suite.task).platform_task))

    def platform_identity(self, suite) -> dict[str, Any]:
        """RoboDojo checkout commit/dirty state and the hashes of the task's native config and code."""
        import hashlib
        import subprocess

        root = Path(self._robodojo_root())
        task = self._task(suite, {})

        def git(*args) -> str | None:
            try:
                return subprocess.run(
                    ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10, check=True
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                return None

        def sha(path: Path) -> str | None:
            return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

        status = git("status", "--porcelain")
        return {
            "platform": "robodojo",
            "commit": git("rev-parse", "HEAD"),
            "dirty": None if status is None else bool(status),
            "task": task,
            "task_config_sha256": sha(root / "task" / "RoboDojo" / "config" / f"{task}.yml"),
            "task_code_sha256": sha(root / "task" / "RoboDojo" / "tasks" / f"{task}.py"),
            "embodiment": EMBODIMENT,
            "client_protocol": "robodojo-ws-v2",
        }
