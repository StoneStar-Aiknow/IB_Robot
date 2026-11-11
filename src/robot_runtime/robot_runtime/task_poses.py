"""Immutable task-pose snapshots shared by configuration, catalog and executors.

Angles are radians. Targets contain every declared arm joint. No file is reread
while a task is running; the digest binds a named request to its loaded snapshot.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping

RETURN_TASK_POSE = "return_to_task_pose"
#: Hands a pose may open, in the order they are opened (left before right).
HAND_SIDES = ("left", "right")
#: The only hand action a pose can declare; it is only ever applied with nothing held.
HAND_ACTIONS = ("open",)


def _number(value, name, low, high):
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    return float(value)


def _hands(pose: str, hands) -> dict:
    if (
        not isinstance(hands, Mapping)
        or not hands
        or set(hands) - set(HAND_SIDES)
        or any(action not in HAND_ACTIONS for action in hands.values())
    ):
        raise ValueError(f"{pose}: hands must map left and/or right to {list(HAND_ACTIONS)}")
    return {side: hands[side] for side in HAND_SIDES if side in hands}


def pose_hand_sides(pose: Mapping) -> tuple[str, ...]:
    """The hands a normalized pose opens, left first; empty for a pose without hands."""
    return tuple(pose.get("hands") or {})


def _transition_route(name, route, joints):
    """Freeze route provenance separately from endpoint hardware approval."""
    if (
        not isinstance(route, Mapping)
        or set(route) - {"waypoints", "supported_starts", "hardware_approved", "source", "table_bounds"}
        or not {"waypoints", "supported_starts", "hardware_approved", "source"} <= set(route)
    ):
        raise ValueError(f"{name}: invalid transition_route keys")
    if not isinstance(route["hardware_approved"], bool):
        raise ValueError(f"{name}: route hardware_approved must be a boolean")

    def positions(q):
        if not isinstance(q, Mapping) or set(q) != set(joints):
            raise ValueError(f"{name}: route positions must cover exactly joint_names")
        return {j: _number(q[j], j, -10, 10) for j in joints}

    waypoints, starts = route["waypoints"], route["supported_starts"]
    if not isinstance(waypoints, list) or not 1 <= len(waypoints) <= 32:
        raise ValueError(f"{name}: route requires 1..32 waypoints")
    if not isinstance(starts, list) or not 1 <= len(starts) <= 32:
        raise ValueError(f"{name}: route requires explicit supported starts")
    normalized_starts = []
    for start in starts:
        if not isinstance(start, Mapping) or set(start) != {"joint_positions", "tolerance_rad"}:
            raise ValueError(f"{name}: invalid supported start")
        normalized_starts.append(
            {
                "joint_positions": positions(start["joint_positions"]),
                "tolerance_rad": _number(start["tolerance_rad"], "route start tolerance_rad", 0.001, 0.2),
            }
        )
    source = route["source"]
    if (
        not isinstance(source, Mapping)
        or set(source) != {"path", "sha256", "sample_indices"}
        or not isinstance(source["path"], str)
        or not source["path"].strip()
        or not isinstance(source["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
        or not isinstance(source["sample_indices"], list)
        or len(source["sample_indices"]) != len(waypoints)
        or any(type(i) is not int or i < 0 for i in source["sample_indices"])
    ):
        raise ValueError(f"{name}: route requires source SHA256 and one sample index per waypoint")
    result = {
        "waypoints": [positions(q) for q in waypoints],
        "supported_starts": normalized_starts,
        "hardware_approved": route["hardware_approved"],
        "source": copy.deepcopy(dict(source)),
    }
    if "table_bounds" in route:
        bounds = route["table_bounds"]
        if not isinstance(bounds, Mapping) or set(bounds) != {"frame", "min_xyz_m", "max_xyz_m", "clearance_m"}:
            raise ValueError(f"{name}: invalid table_bounds")
        # Robot root frame: +X forward, +Y left, +Z up. No implicit world transform.
        if bounds["frame"] != "base_link":
            raise ValueError(f"{name}: table_bounds frame must be base_link")
        normalized = {"frame": "base_link"}
        for key in ("min_xyz_m", "max_xyz_m"):
            if not isinstance(bounds[key], list) or len(bounds[key]) != 3:
                raise ValueError(f"{name}: table bounds require three coordinates")
            normalized[key] = [_number(v, key, -10, 10) for v in bounds[key]]
        if any(a >= b for a, b in zip(normalized["min_xyz_m"], normalized["max_xyz_m"], strict=True)):
            raise ValueError(f"{name}: table bounds must have positive extent")
        normalized["clearance_m"] = _number(bounds["clearance_m"], "clearance_m", 0, 1)
        result["table_bounds"] = normalized
    return result


def normalize_task_poses(data: Mapping) -> dict:
    if not isinstance(data, Mapping) or type(data.get("schema_version")) is not int or data.get("schema_version") != 1:
        raise ValueError("task poses require schema_version: 1")
    if set(data) - {"schema_version", "robot_type", "joint_names", "poses", "skill_entry_poses", "digest"}:
        raise ValueError("unknown task-pose file keys")
    if not isinstance(data.get("robot_type"), str) or not data["robot_type"].strip():
        raise ValueError("task poses require robot_type")
    joints = data.get("joint_names")
    if (
        not isinstance(joints, list)
        or not joints
        or any(not isinstance(j, str) or not j for j in joints)
        or len(set(joints)) != len(joints)
    ):
        raise ValueError("joint_names must be unique non-empty joint names")
    poses = data.get("poses")
    if not isinstance(poses, Mapping) or not poses:
        raise ValueError("poses must be a non-empty mapping")
    out = {"schema_version": 1, "robot_type": data["robot_type"], "joint_names": list(joints), "poses": {}}
    for name, pose in poses.items():
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name)
            or not isinstance(pose, Mapping)
        ):
            raise ValueError("invalid task-pose name or mapping")
        if set(pose) - {
            "joint_positions",
            "position_tolerance_rad",
            "stable_sec",
            "timeout_sec",
            "max_state_age_sec",
            "hardware_approved",
            "transition_route",
            "arrival_wait_sec",
            "hands",
            "unloaded_operation_approved",
            "description",
            "aliases",
        }:
            raise ValueError(f"unknown task-pose keys: {name}")
        q = pose.get("joint_positions")
        if not isinstance(q, Mapping) or set(q) != set(joints):
            raise ValueError(f"{name}: joint_positions must cover exactly joint_names")
        approved = pose.get("hardware_approved", False)
        if not isinstance(approved, bool):
            raise ValueError("hardware_approved must be a boolean")
        hands = pose.get("hands")
        if "hands" in pose and hands is None:
            raise ValueError(f"{name}: hands must be a non-empty mapping when declared")
        if hands is None and "unloaded_operation_approved" in pose:
            raise ValueError(f"{name}: unloaded_operation_approved only applies to a pose with hands")
        out["poses"][name] = {
            "joint_positions": {j: _number(q[j], j, -10, 10) for j in joints},
            "position_tolerance_rad": _number(
                pose.get("position_tolerance_rad", 0.03), "position_tolerance_rad", 0.001, 0.1
            ),
            "stable_sec": _number(pose.get("stable_sec", 0.3), "stable_sec", 0.05, 5),
            "timeout_sec": _number(pose.get("timeout_sec", 30), "timeout_sec", 0.1, 300),
            "max_state_age_sec": _number(pose.get("max_state_age_sec", 0.5), "max_state_age_sec", 0.02, 2),
            "hardware_approved": approved,
        }
        # Optional fields preserve the identity of legacy endpoint-only snapshots.
        if "transition_route" in pose:
            out["poses"][name]["transition_route"] = _transition_route(name, pose["transition_route"], joints)
        if "arrival_wait_sec" in pose:
            out["poses"][name]["arrival_wait_sec"] = _number(pose["arrival_wait_sec"], "arrival_wait_sec", 0.0, 10.0)
        for key in ("description", "aliases"):
            if key not in pose:
                continue
            value = pose[key]
            if key == "description":
                valid = isinstance(value, str) and bool(value.strip())
            else:
                valid = isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value)
            if not valid:
                raise ValueError(f"{name}: invalid {key}")
            out["poses"][name][key] = copy.deepcopy(value)
        if hands is not None:
            # Absent hands keep the normalized pose, and so the digest, exactly as before.
            out["poses"][name]["hands"] = _hands(name, hands)
            unloaded = pose.get("unloaded_operation_approved", False)
            if not isinstance(unloaded, bool):
                raise ValueError("unloaded_operation_approved must be a boolean")
            out["poses"][name]["unloaded_operation_approved"] = unloaded
    entries = data.get("skill_entry_poses", {})
    if not isinstance(entries, Mapping) or set(entries) - {"pick_object"}:
        raise ValueError("v1 skill_entry_poses supports pick_object only")
    if any(not isinstance(name, str) or name not in out["poses"] for name in entries.values()):
        raise ValueError("skill_entry_poses references an unknown pose")
    out["skill_entry_poses"] = dict(entries)
    out["digest"] = hashlib.sha256(
        json.dumps(out, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if "digest" in data and data["digest"] != out["digest"]:
        raise ValueError("task-pose snapshot digest mismatch")
    return out


def pose_catalog(config: Mapping | None, composer: Mapping | None = None) -> dict:
    """Freeze canonical execution IDs and their planner-only descriptions at startup.

    Aliases describe choices; they are deliberately not accepted by the enum.
    Generated composition remains freeform. Fixed artifacts are content-bound just
    like the executor's startup input, never read again during a running workflow.
    """
    from pathlib import Path

    result = {}
    if config:
        snapshot = normalize_task_poses(config)
        result[RETURN_TASK_POSE] = [
            {
                "target_name": name,
                "description": pose.get("description", name),
                "aliases": list(pose.get("aliases", [])),
            }
            for name, pose in sorted(snapshot["poses"].items())
        ]
    composer = composer or {}
    if composer.get("mode", "generated") == "fixed":
        raw = Path(composer["fixed_library"]).expanduser().read_bytes()
        expected = composer.get("fixed_library_sha256")
        if not expected or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("fixed pose artifact changed since the configuration snapshot")
        library = json.loads(raw)
        poses = library.get("poses")
        if not isinstance(poses, dict) or not poses:
            raise ValueError("fixed pose artifact requires poses")
        result["free_pose"] = [
            {
                "target_name": name,
                "description": record["user_message"],
                "aliases": list(record["phrases"]),
            }
            for name, record in sorted(poses.items())
        ]
    return result


def task_pose_identity(config: Mapping | None, composer: Mapping | None = None) -> dict:
    """Keep deployments without task poses byte-for-byte compatible."""
    from .path_validation import path_validation_identity

    return {**({"task_poses_digest": config["digest"]} if config else {}), **path_validation_identity(composer)}


def task_pose_arm_targets(description: Mapping, snapshot: Mapping) -> tuple[dict, dict]:
    """Resolve exact per-arm named anchors from an already frozen task snapshot."""
    targets, references = {}, {}
    for side in HAND_SIDES:
        arm = description.get(side) or {}
        if not isinstance(arm, Mapping) or "task_pose" not in arm:
            continue
        if set(arm) != {"task_pose"} or not isinstance(arm["task_pose"], str):
            raise ValueError(f"{side}: task_pose is exclusive with other arm instructions")
        name = arm["task_pose"]
        pose = (snapshot.get("poses") or {}).get(name)
        if pose is None:
            raise ValueError(f"{side}: task pose {name!r} is not configured")
        references[side] = name
        targets.update({n: v for n, v in pose["joint_positions"].items() if n.startswith(side + "_")})
    return targets, references


def task_pose_target(config: Mapping, name: str, digest: str) -> dict:
    if not config or digest != config["digest"]:
        raise ValueError("task-pose snapshot changed or is unavailable")
    if name not in config["poses"]:
        raise ValueError(f"unknown task pose: {name}")
    return copy.deepcopy(config["poses"][name])
