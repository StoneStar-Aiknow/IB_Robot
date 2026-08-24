import pytest

from semantic_mapping.slam_readiness import evaluate_slam_readiness


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"active_map_hash": "other"}, "map identity"),
        ({"localization_ready": False}, "localization"),
        ({"authoritative_map_odom": False}, "map-to-odom"),
        ({"cloud_map_ready": False}, "cloud_map"),
        ({"timestamped_tf_ready": False}, "timestamped camera transform"),
    ],
)
def test_slam_gate_fails_closed_for_each_required_contract(override, reason):
    values = {
        "expected_map_hash": "map-hash",
        "active_map_hash": "map-hash",
        "localization_ready": True,
        "authoritative_map_odom": True,
        "cloud_map_ready": True,
        "timestamped_tf_ready": True,
    }
    values.update(override)

    readiness = evaluate_slam_readiness(**values)

    assert not readiness.ready
    assert reason in readiness.reason


def test_slam_gate_accepts_complete_compatible_contract():
    assert evaluate_slam_readiness(
        expected_map_hash="map-hash",
        active_map_hash="map-hash",
        localization_ready=True,
        authoritative_map_odom=True,
        cloud_map_ready=True,
        timestamped_tf_ready=True,
    ).ready


def test_slam_gate_accepts_missing_active_hash_for_legacy_publishers():
    assert evaluate_slam_readiness(
        expected_map_hash="map-hash",
        active_map_hash="",
        localization_ready=True,
        authoritative_map_odom=True,
        cloud_map_ready=True,
        timestamped_tf_ready=True,
    ).ready
