"""Resolve the fixed action artifact into the startup configuration identity."""

import hashlib
import os
import re
from pathlib import Path

from robot_config.utils import resolve_ros_path


def fixed_library_path(robot, config_path) -> Path:
    """Resolve the configured artifact path without requiring a pre-existing artifact."""
    config = robot.get("pose_composer") or {}
    raw = config.get("fixed_library", "")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("fixed pose mode requires fixed_library")
    for variable in re.findall(r"\$\(env\s+(\w+)\)", raw):
        if not os.environ.get(variable):
            raise ValueError(f"fixed_library requires non-empty environment variable {variable}")
    resolved = resolve_ros_path(raw)
    if "$(" in resolved:
        raise ValueError("fixed_library contains an unresolved path substitution")
    path = Path(resolved).expanduser()
    if not path.is_absolute():
        path = Path(config_path).parent / path
    return path.resolve()


def resolve_fixed_poses(robot, config_path):
    config = robot.get("pose_composer") or {}
    if config.get("mode", "generated") != "fixed":
        return
    path = fixed_library_path(robot, config_path)
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = config.get("fixed_library_sha256")
    if expected and expected != observed:
        raise ValueError("fixed pose artifact changed since the configuration snapshot")
    config.update(fixed_library=str(path), fixed_library_sha256=observed)
