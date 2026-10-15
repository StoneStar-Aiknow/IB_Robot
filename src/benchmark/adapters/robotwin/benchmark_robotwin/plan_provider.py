"""RoboTwin plan provider: lazy seam to the pinned ``libs/robotwin`` source.

Reads the canonical 50-task manifest from ``env_cfg/eval/all_tasks.yml``
and the per-task step limit from ``env_cfg/task_config/_eval_step_limit.yml``.

Heavy Sapien/RoboTwin imports are deferred to the provider's
``resolve_tasks`` method; the provider itself is cheap to construct.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from benchmark_robotwin.version_probe import default_robotwin_root


@dataclass(frozen=True, slots=True)
class RobotwinProviderTask:
    """One resolved task from the RoboTwin source."""

    task_name: str
    instruction: str
    step_limit: int

    def __post_init__(self) -> None:
        if not isinstance(self.task_name, str) or not self.task_name.strip():
            raise ValueError("task_name must be a non-empty string")
        if not isinstance(self.instruction, str):
            raise ValueError("instruction must be a string")
        if not isinstance(self.step_limit, int) or isinstance(self.step_limit, bool) or self.step_limit <= 0:
            raise ValueError("step_limit must be a positive int")


class RobotwinProviderError(RuntimeError):
    """Raised when the RoboTwin provider cannot resolve tasks."""


class RobotwinPlanProvider:
    """Lazy pinned-``libs/robotwin`` plan provider."""

    def __init__(self, robotwin_root: str | Path | None = None) -> None:
        if robotwin_root is None:
            robotwin_root = default_robotwin_root()
        self._root = Path(robotwin_root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _load_all_tasks(self) -> list[str]:
        manifest = self._root / "env_cfg" / "eval" / "all_tasks.yml"
        if not manifest.is_file():
            raise RobotwinProviderError(f"task manifest not found: {manifest}")
        with manifest.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise RobotwinProviderError(f"task manifest must be a Mapping, got {type(data).__name__}")
        tasks = data.get("tasks")
        if not isinstance(tasks, list):
            raise RobotwinProviderError("task manifest 'tasks' must be a list")
        result: list[str] = []
        for entry in tasks:
            if not isinstance(entry, str) or not entry.strip():
                raise RobotwinProviderError(f"task entry must be a non-empty string, got {entry!r}")
            result.append(entry.strip())
        return result

    def _load_step_limits(self) -> dict[str, int]:
        path = self._root / "env_cfg" / "task_config" / "_eval_step_limit.yml"
        if not path.is_file():
            raise RobotwinProviderError(f"step limit file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise RobotwinProviderError("step limit file must be a Mapping")
        result: dict[str, int] = {}
        for key, val in data.items():
            if isinstance(val, int) and not isinstance(val, bool) and val > 0:
                result[str(key)] = int(val)
        return result

    def _load_default_instruction(self, task_name: str) -> str:
        path = self._root / "description" / "task_instruction" / f"{task_name}.json"
        if not path.is_file():
            return task_name.replace("_", " ")
        try:
            import json

            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                full = data.get("full_description")
                if isinstance(full, str) and full.strip():
                    return full.strip()
        except Exception:
            pass
        return task_name.replace("_", " ")

    def resolve_single_task(self, task_name: str) -> RobotwinProviderTask:
        if not isinstance(task_name, str) or not task_name.strip():
            raise RobotwinProviderError("task_name must be a non-empty string")
        all_tasks = self._load_all_tasks()
        if task_name not in all_tasks:
            raise RobotwinProviderError(
                f"task '{task_name}' not in canonical 50-task manifest; available: {len(all_tasks)} tasks"
            )
        step_limits = self._load_step_limits()
        step_limit = step_limits.get(task_name, 1000)
        instruction = self._load_default_instruction(task_name)
        return RobotwinProviderTask(
            task_name=task_name,
            instruction=instruction,
            step_limit=step_limit,
        )
