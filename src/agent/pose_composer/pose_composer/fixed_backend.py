"""Compatibility facade; X2 physical execution belongs to its robot package."""

from aimdk_grasp.fixed_pose_backend import EFFORT, GRAVITY_SHARE, JOINTS, MAX_SPEED, FixedBackend, digest

__all__ = ["EFFORT", "GRAVITY_SHARE", "JOINTS", "MAX_SPEED", "FixedBackend", "digest"]
