"""Seed poses and the memory fingerprint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from pose_composer.memory import PoseMemory
from pose_composer.model import load

SEED_VERSION_KEY = "seed_version"


def fingerprint(prompt_version: str, model_name: str, mjcf_path: str) -> str:
    """What a remembered pose was solved under: prompt, model, robot model file, joint travel and rest pose.

    A change to any of them makes remembered poses stale until they are re-solved.
    """
    robot = load()
    mjcf_digest = hashlib.sha256(Path(mjcf_path).expanduser().read_bytes()).hexdigest()
    payload = {
        "prompt": prompt_version,
        "model": model_name,
        "mjcf": mjcf_digest,
        "limits": {k: [round(v[0], 4), round(v[1], 4)] for k, v in sorted(robot.safe_limits.items())},
        "rest": {k: round(v, 4) for k, v in sorted(robot.rest.items())},
    }
    if robot.task_poses:
        payload["task_poses_digest"] = robot.task_poses["digest"]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def read_seed_file(path: str | Path) -> tuple[str, list[dict]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    entries = data.get("poses") or []
    for entry in entries:
        if not {"id", "phrases", "description"} <= set(entry):
            raise ValueError(f"seed {entry.get('id')!r} needs id, phrases and description")
    return str(data.get("version", "")), entries


def load_seeds(memory: PoseMemory, path: str) -> int:
    """Load the seed file if it is newer than the one the memory holds; the poses are stored stale (to be re-solved)."""
    if not path:
        return 0
    version, entries = read_seed_file(path)
    if memory.meta(SEED_VERSION_KEY) == version:
        return 0
    return memory.add_described(entries, source="seed", pinned=True, meta={SEED_VERSION_KEY: version})
