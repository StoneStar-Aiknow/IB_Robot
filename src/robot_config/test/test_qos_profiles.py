"""The QoS mapping a runtime description declares, parsed without the contract datamodel."""

from __future__ import annotations

from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

from robot_config.qos_profiles import qos_profile_from_dict


def test_declared_policies_become_the_matching_rclpy_profile():
    profile = qos_profile_from_dict(
        {"reliability": "best_effort", "history": "keep_all", "durability": "transient_local", "depth": 3}
    )

    assert profile is not None
    assert profile.reliability == ReliabilityPolicy.BEST_EFFORT
    assert profile.history == HistoryPolicy.KEEP_ALL
    assert profile.durability == DurabilityPolicy.TRANSIENT_LOCAL
    assert profile.depth == 3


def test_an_absent_mapping_has_no_profile():
    """``None`` lets a caller keep its own default instead of inheriting a made-up one."""

    assert qos_profile_from_dict(None) is None
    assert qos_profile_from_dict({}) is None
