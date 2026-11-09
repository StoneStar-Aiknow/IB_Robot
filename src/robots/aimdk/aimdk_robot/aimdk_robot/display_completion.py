"""Explicit playback-only opt-in; strict completion remains the default."""

from __future__ import annotations

from collections.abc import Mapping

from aimdk_robot.projection import DEXTEROUS_JOINT_ORDER, VENDOR_ARM_ORDER


def display_completion_policy(value, trajectories):
    if value is None:
        return None
    keys = {"motions", "mode"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"display_completion requires exactly {sorted(keys)}")
    names = value["motions"]
    if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names):
        raise ValueError("display_completion.motions must be a non-empty list of recording names")
    allowed = {"short_wave", "invite", "self_intro", "explain", "wave", "raise_hand", "blow_kiss", "clap"}
    if len(names) != len(set(names)) or any(name not in trajectories or name not in allowed for name in names):
        raise ValueError("display_completion.motions must name distinct supported empty-hand displays")
    if value["mode"] != "playback_only":
        raise ValueError("display_completion.mode must be playback_only")
    return {"motions": tuple(names), "mode": "playback_only"}


def goal_acceptance_disabled(tolerances, joints):
    """Honor the standard JointTolerance -1 opt-out only for a complete explicit joint set."""
    return bool(
        len(tolerances) == len(joints)
        and {item.name for item in tolerances} == set(joints)
        and all(item.position == item.velocity == item.acceleration == -1.0 for item in tolerances)
    )


def final_pose_evidence(arm_sample, hand_sample, arms, hands, final_at, now):
    """Keep missing/stale feedback separate from actual joint tracking errors."""
    result = {"feedback_after_target": False}
    for label, sample, desired, names in (
        ("arm", arm_sample, arms, VENDOR_ARM_ORDER),
        (
            "hand",
            hand_sample,
            hands,
            [f"{side}_{joint}" for side in ("left", "right") for joint in DEXTEROUS_JOINT_ORDER],
        ),
    ):
        result[f"{label}_age_s"] = None if sample is None else now - sample[1]
        result[f"{label}_max_error_rad"] = None
        result[f"{label}_worst_joint"] = ""
        if sample is not None:
            errors = [abs(a - b) for a, b in zip(sample[0], desired, strict=True)]
            worst = max(range(len(errors)), key=errors.__getitem__)
            result[f"{label}_max_error_rad"] = errors[worst]
            result[f"{label}_worst_joint"] = names[worst]
    result["feedback_after_target"] = bool(
        arm_sample
        and hand_sample
        and final_at <= min(arm_sample[1], hand_sample[1])
        and max(arm_sample[1], hand_sample[1]) <= now
    )
    return result


def final_pose_reached(evidence, arm_tolerance, hand_tolerance):
    return bool(
        evidence["feedback_after_target"]
        and evidence["arm_max_error_rad"] <= arm_tolerance
        and evidence["hand_max_error_rad"] <= hand_tolerance
    )
