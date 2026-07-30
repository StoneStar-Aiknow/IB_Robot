"""Fail-closed SLAM contract gate for fusion and semantic actions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SlamReadiness:
    ready: bool
    reason: str = ""


def evaluate_slam_readiness(
    *,
    expected_map_hash: str,
    active_map_hash: str,
    localization_ready: bool,
    authoritative_map_odom: bool,
    cloud_map_ready: bool,
    timestamped_tf_ready: bool,
) -> SlamReadiness:
    if not expected_map_hash or active_map_hash != expected_map_hash:
        return SlamReadiness(False, "active SLAM map identity is incompatible")
    if not localization_ready:
        return SlamReadiness(False, "global localization is not ready")
    if not authoritative_map_odom:
        return SlamReadiness(False, "authoritative map-to-odom contract is not ready")
    if not cloud_map_ready:
        return SlamReadiness(False, "cloud_map contract is not ready")
    if not timestamped_tf_ready:
        return SlamReadiness(False, "timestamped camera transform is not ready")
    return SlamReadiness(True)
