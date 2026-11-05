"""Resolve the fixed action artifact into the startup configuration identity."""

import hashlib
from pathlib import Path


def resolve_fixed_poses(robot, config_path):
    config = robot.get("pose_composer") or {}
    if config.get("mode", "generated") != "fixed":
        return
    raw = config.get("fixed_library", "")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("fixed pose mode requires fixed_library")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(config_path).parent / path
    path = path.resolve()
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = config.get("fixed_library_sha256")
    if expected and expected != observed:
        raise ValueError("fixed pose artifact changed since the configuration snapshot")
    config.update(fixed_library=str(path), fixed_library_sha256=observed)
