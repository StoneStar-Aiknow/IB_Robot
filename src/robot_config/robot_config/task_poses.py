"""Resolve a local task-pose file once through the robot configuration loader."""

from pathlib import Path

import yaml

from robot_runtime.hand_mounts import installed_hands
from robot_runtime.task_poses import normalize_task_poses


def _select_installed_hands(snapshot: dict, robot_config: dict) -> dict:
    options = (robot_config.get("grasp_execution") or {}).get("backend_options") or {}
    if "installed_hands" not in options:
        raise ValueError("task_poses.hands: installed requires explicit backend_options.installed_hands")
    mounted = installed_hands(options["installed_hands"])
    if not mounted:
        raise ValueError("task_poses.hands: installed requires at least one installed hand")
    # The input was normalized first, so invalid actions cannot disappear through filtering.
    snapshot.pop("digest")
    for name, pose in snapshot["poses"].items():
        if "hands" not in pose:
            continue
        if not set(mounted) <= set(pose["hands"]):
            raise ValueError(f"{name}: task_poses.hands: installed requires actions for every installed hand")
        pose["hands"] = {side: pose["hands"][side] for side in mounted}
    return normalize_task_poses(snapshot)


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
            or set(block) - {"file", "hands"}
            or "file" not in block
            or not isinstance(block["file"], str)
            or not block["file"].strip()
        ):
            raise ValueError("task_poses requires one non-empty file path or a digest-bound snapshot")
        if "hands" in block and block["hands"] != "installed":
            raise ValueError("task_poses.hands must be installed when specified")
        path = Path(block["file"]).expanduser()
        if not path.is_absolute():
            path = Path(config_path).parent / path
        snapshot = normalize_task_poses(yaml.safe_load(path.read_text(encoding="utf-8")))
        if block.get("hands") == "installed":
            snapshot = _select_installed_hands(snapshot, robot_config)
    if snapshot["robot_type"] != robot_config.get("type"):
        raise ValueError("task-pose robot_type does not match robot configuration")
    if (robot_config.get("pose_composer") or {}).get("enabled") is not True:
        raise ValueError("task_poses requires pose_composer.enabled")
    robot_config["task_poses"] = snapshot
