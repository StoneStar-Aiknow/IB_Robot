"""Placement palm goals: which orientations are tried, in what order, and where the object goes."""

from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.grasp_request import palm_reach
from aimdk_grasp.place_targets import (
    PLACE_EXTRA_POSITIONS,
    PLACE_TILT_DIRECTIONS,
    PLACE_TILTS_DEG,
    PLACE_YAWS_DEG,
    PlaceTargetConfig,
    place_orientations,
    place_targets,
)
from scipy.spatial.transform import Rotation

from manipulation_execution.grasp_pipeline.contracts import GraspFailure

UP = np.array([0.0, 0.0, 1.0])


def test_orientations_cover_every_yaw_and_the_tilt_cone_least_turn_first():
    rotations = place_orientations(UP)
    expected = len(PLACE_YAWS_DEG) * (1 + PLACE_TILT_DIRECTIONS * (len(PLACE_TILTS_DEG) - 1))
    assert len(rotations) == expected
    np.testing.assert_allclose(rotations[0], np.eye(3), atol=1e-12)  # the carrying palm itself first
    turned = [Rotation.from_matrix(r).magnitude() for r in rotations]
    assert np.all(np.diff(turned) >= -1e-12)
    tilt = np.degrees([np.arccos(np.clip((r @ UP) @ UP, -1, 1)) for r in rotations])
    assert tilt.max() <= max(PLACE_TILTS_DEG) + 1e-9
    # The five yaws placement used before are still there, untilted.
    for yaw in (0, 30, -30, 60, -60):
        target = Rotation.from_rotvec(UP * np.radians(yaw)).as_matrix()
        assert min(np.abs(r - target).max() for r in rotations) < 1e-9


def test_targets_keep_the_object_over_the_opening_and_start_from_the_carrying_palm():
    shoulder, reach = palm_reach("right")
    centre = np.asarray(shoulder, float) + [0.2, 0.0, -0.25]
    assert np.linalg.norm(centre - shoulder) < reach - 0.1
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    opening = centre + 0.12 * np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
    container = SimpleNamespace(normal=UP, opening=opening, report={}, rim_height_m=float(centre[2]))
    cloud = 0.03 * np.random.default_rng(0).normal(size=(64, 3)) + [0.0, 0.04, 0.10]
    held = SimpleNamespace(arm_side="right", points_in_palm=tuple(cloud.ravel()), radius_m=0.04)
    palm = np.eye(4)
    palm[:3, 3] = centre + [0.0, 0.0, 0.2]

    targets = place_targets(container, held, palm)

    assert 0 < len(targets) <= len(place_orientations(UP)) * (1 + PLACE_EXTRA_POSITIONS)
    np.testing.assert_allclose(targets[0][1][:3, :3], palm[:3, :3], atol=1e-12)
    for above, release in targets:
        np.testing.assert_allclose(above[:3, :3], release[:3, :3])  # the descent keeps its orientation
        np.testing.assert_allclose(above[:3, 3] - release[:3, 3], 0.08 * UP, atol=1e-12)
        lowest = np.min((cloud @ release[:3, :3].T + release[:3, 3]) @ UP)
        assert lowest >= container.rim_height_m + 0.045 - 1e-9
        assert max(np.linalg.norm(pose[:3, 3] - shoulder) for pose in (above, release)) <= reach


def _scene(opening_radius=0.12, held_radius=0.04):
    """A round opening well inside the right arm's reach, a held cloud and a palm above it."""
    shoulder, reach = palm_reach("right")
    centre = np.asarray(shoulder, float) + [0.2, 0.0, -0.25]
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    opening = centre + opening_radius * np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
    container = SimpleNamespace(normal=UP, opening=opening, report={}, rim_height_m=float(centre[2]))
    cloud = 0.03 * np.random.default_rng(0).normal(size=(64, 3)) + [0.0, 0.04, 0.10]
    held = SimpleNamespace(arm_side="right", points_in_palm=tuple(cloud.ravel()), radius_m=held_radius)
    palm = np.eye(4)
    palm[:3, 3] = centre + [0.0, 0.0, 0.2]
    return container, held, palm, cloud, centre


def _object_centres(targets, cloud):
    """Where each release pose puts the held cloud's centre, in the opening's plane."""
    return np.array([(cloud @ release[:3, :3].T + release[:3, 3]).mean(axis=0)[:2] for _, release in targets])


def test_default_config_reproduces_the_fixed_constants():
    container, held, palm, _, _ = _scene()
    default = place_targets(container, held, palm)
    explicit = place_targets(container, held, palm, PlaceTargetConfig())
    assert len(default) == len(explicit)
    for (above, release), (above2, release2) in zip(default, explicit, strict=True):
        np.testing.assert_array_equal(above, above2)
        np.testing.assert_array_equal(release, release2)


def test_configured_clearances_set_the_release_height_and_the_transfer_rise():
    container, held, palm, cloud, _ = _scene()
    config = PlaceTargetConfig(bottom_clearance_m=0.02, transfer_clearance_m=0.15)

    targets = place_targets(container, held, palm, config)

    for above, release in targets:
        np.testing.assert_allclose(above[:3, 3] - release[:3, 3], 0.15 * UP, atol=1e-12)
        lowest = np.min((cloud @ release[:3, :3].T + release[:3, 3]) @ UP)
        np.testing.assert_allclose(lowest, container.rim_height_m + 0.02 + 0.01, atol=1e-9)


def test_zero_extra_positions_releases_at_the_centre_only():
    container, held, palm, cloud, centre = _scene()

    spread = place_targets(container, held, palm)
    centred = place_targets(container, held, palm, PlaceTargetConfig(extra_positions=0))

    assert len(centred) < len(spread)
    # Every orientation samples the same xy in the opening's basis, though the final world position
    # varies slightly because different rotations put the object's lowest point at different places.
    released = _object_centres(centred, cloud)
    assert np.linalg.norm(released - released.mean(axis=0), axis=1).max() < 0.02
    assert np.linalg.norm(_object_centres(spread, cloud) - released.mean(axis=0), axis=1).max() > 0.02


def test_edge_fraction_moves_the_extra_positions_toward_the_rim():
    container, held, palm, cloud, _ = _scene(opening_radius=0.2)

    def spread(fraction):
        targets = place_targets(container, held, palm, PlaceTargetConfig(edge_fraction=fraction))
        released = _object_centres(targets, cloud)
        return np.linalg.norm(released - released[0], axis=1).max()

    assert spread(0.5) > spread(0.25) + 0.02


def test_no_release_position_clear_of_the_rim_is_reported_as_too_small():
    container, held, palm, _, _ = _scene(opening_radius=0.05, held_radius=0.04)
    with pytest.raises(GraspFailure) as failure:
        place_targets(container, held, palm, PlaceTargetConfig(extra_positions=0))
    assert failure.value.code == "CONTAINER_TOO_SMALL"


@pytest.mark.parametrize(
    "bad",
    [
        {"bottom_clearance_m": -0.01},
        {"bottom_clearance_m": float("nan")},
        {"transfer_clearance_m": 0.0},
        {"transfer_clearance_m": 0.5},
        {"extra_positions": -1},
        {"extra_positions": 2.0},
        {"extra_positions": True},
        {"edge_fraction": 0.0},
        {"edge_fraction": 1.0},
        {"bottom_clearance_m": "0.05"},
    ],
)
def test_config_refuses_values_outside_its_ranges(bad):
    with pytest.raises(ValueError, match="backend_options.place"):
        PlaceTargetConfig(**bad)
