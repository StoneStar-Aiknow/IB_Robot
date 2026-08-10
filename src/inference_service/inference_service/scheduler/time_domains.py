"""Conversions at scheduler wire boundaries between monotonic and ROS time."""

from __future__ import annotations


def monotonic_expiry_to_ros_ns(expiry_ns: int, *, monotonic_now_ns: int, ros_now_ns: int) -> int:
    """Express a monotonic expiry as an absolute timestamp in the ROS clock domain."""
    return ros_now_ns + max(0, expiry_ns - monotonic_now_ns)
