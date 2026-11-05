"""Compatibility facade; X2 implementation lives in aimdk_robot.pose_backend."""

from aimdk_robot.pose_backend import motion_limits as _backend


def __getattr__(name):
    return getattr(_backend, name)
