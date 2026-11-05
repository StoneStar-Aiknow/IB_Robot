"""Resolve a local task-pose file once through the robot configuration loader."""

from pathlib import Path

import yaml

from robot_runtime.task_poses import normalize_task_poses


def resolve_task_pose_file(robot_config: dict, config_path) -> None:
    block = robot_config.get("task_poses")
    if block is None:
        return
    if isinstance(block, dict) and isinstance(block.get("digest"), str) and block["digest"]:
        # Live launch serializes the normalized block into its consumer snapshot.
        # Revalidate its digest without reopening the mutable source pose file.
        snapshot = normalize_task_poses(block)
    else:
        if (
            not isinstance(block, dict)
            or set(block) != {"file"}
            or not isinstance(block["file"], str)
            or not block["file"].strip()
        ):
            raise ValueError("task_poses requires one non-empty file path or a digest-bound snapshot")
        path = Path(block["file"]).expanduser()
        if not path.is_absolute():
            path = Path(config_path).parent / path
        snapshot = normalize_task_poses(yaml.safe_load(path.read_text(encoding="utf-8")))
    if snapshot["robot_type"] != robot_config.get("type"):
        raise ValueError("task-pose robot_type does not match robot configuration")
    if (robot_config.get("pose_composer") or {}).get("enabled") is not True:
        raise ValueError("task_poses requires pose_composer.enabled")
    robot_config["task_poses"] = snapshot
