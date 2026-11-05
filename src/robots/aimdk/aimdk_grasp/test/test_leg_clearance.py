"""Refine conservative boxes without accepting solid intersections or losing margins."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.hand_restore import RestoreError
from aimdk_grasp.hand_restore_geometry import DualHandCollision
from aimdk_grasp.leg_clearance import ConvexLeg
from scipy.spatial.transform import Rotation


def test_convex_distance_matches_analytic_rotated_solid_box():
    rng = np.random.default_rng(163)
    half = np.array([1.0, 0.7, 0.4])
    rotation = Rotation.from_rotvec([0.3, -0.7, 0.2]).as_matrix()
    origin = np.array([0.1, -0.2, 0.8])
    hull = ConvexLeg(np.array(list(product([-1, 1], repeat=3))) * half @ rotation.T + origin)
    # Random points plus exact face/edge/vertex, interior and corner cases.
    points = np.r_[rng.uniform(-2, 2, (256, 3)), [[0, 0, 0], half, [1, 0, 0], [1.1, 0.8, 0.5]]]
    radii = rng.uniform(0, 0.1, len(points))
    spheres = np.c_[points @ rotation.T + origin, radii]
    gap = np.abs(points) - half
    exact = np.linalg.norm(np.maximum(gap, 0), axis=1) + np.minimum(np.max(gap, axis=1), 0) - radii
    np.testing.assert_allclose(hull.clearance(spheres), exact, atol=1e-10)
    assert np.all(hull.lower_bound(spheres) <= exact + 1e-10)


def test_tetrahedron_face_edge_vertex_and_inside_distances():
    hull = ConvexLeg([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    spheres = np.array([[0.8, 0.8, 0.8, 0], [0.6, 0.6, 0, 0], [-0.1, -0.1, -0.1, 0], [0.1, 0.1, 0.1, 0.02]])
    expected = [1.4 / np.sqrt(3), np.sqrt(0.02), np.sqrt(0.03), -0.12]
    np.testing.assert_allclose(hull.clearance(spheres), expected, atol=1e-10)


@pytest.mark.parametrize("ball", [[np.nan, 0, 0, 1], [0, 0, 0, -0.1]])
def test_invalid_sphere_is_rejected(ball):
    hull = ConvexLeg(list(product([-1, 1], repeat=3)))
    with pytest.raises(ValueError):
        hull.clearance([ball])


def geometry_with_spheres(left, right, hull, margins):
    geometry = object.__new__(DualHandCollision)
    geometry.leg_hulls = {"left_leg_2": hull} if hull else {}
    geometry.checks, geometry.indices, geometry.bounds = {}, {}, {}
    for side, balls in (("left", left), ("right", right)):
        names = ["R_palm", "R_thumb"][: len(balls)]
        geometry.indices[side] = list(range(len(balls)))
        geometry.checks[side] = SimpleNamespace(
            place=lambda *_a, balls=balls, **_k: np.array([balls], float),
            check_spheres=lambda *_a, **_k: 1.0,
            model=SimpleNamespace(links=names, sphere_links=names, joints=[]),
        )
        geometry.bounds[side] = SimpleNamespace(
            angles=lambda _feedback, delta, *_radii: np.asarray(delta),
            world=lambda link, angles: margins[link] * np.max(angles),
            levers=dict.fromkeys(names, {}),
        )
    return geometry


HANDS = {side: np.zeros(10) for side in ("left", "right")}


def test_coarse_box_overlap_gets_convex_refinement():
    # Triangular prism: the PCA box contains empty space outside its x=0 face.
    hull = ConvexLeg([[x, y, z] for x, y in ((0, 0), (1, 0), (0, 1)) for z in (-0.2, 0.2)])
    left = [[-0.1, 0.5, 0, 0.01]]
    assert hull.lower_bound(np.array(left))[0] < 0
    assert hull.clearance(left)[0] == pytest.approx(0.09)
    geometry = geometry_with_spheres(left, [[3, 3, 3, 0.01]], hull, {"R_palm": 0})
    assert geometry.check(None, HANDS) > 0


@pytest.mark.parametrize("x", [0.0, 1.005, 1.02])
def test_real_overlap_and_insufficient_clearance_still_reject_with_diagnostics(x):
    hull = ConvexLeg(list(product([-1, 1], repeat=3)))
    geometry = geometry_with_spheres([[x, 0, 0, 0.01]], [[3, 3, 3, 0.01]], hull, {"R_palm": 0.001})
    with pytest.raises(RestoreError, match=r"R_palm vs left_leg_2: clearance=.*required=0.011000.*sample=41"):
        geometry.check(None, HANDS, arm_delta=np.ones(14), sample_offset=41)


def test_distant_thumb_margin_does_not_reject_safe_palm_near_leg():
    hull = ConvexLeg(list(product([-1, 1], repeat=3)))
    geometry = geometry_with_spheres(
        [[1.03, 0, 0, 0.01], [3, 0, 0, 0.01]], [[4, 4, 4, 0.01]], hull, {"R_palm": 0.001, "R_thumb": 0.04}
    )
    assert geometry.check(None, HANDS, arm_delta=np.ones(14)) > 0


def test_opposite_hands_use_the_two_participating_link_margins():
    geometry = geometry_with_spheres(
        [[0, 0, 0, 0.01], [10, 0, 0, 0.01]], [[0.04, 0, 0, 0.01]], None, {"R_palm": 0.001, "R_thumb": 0.04}
    )
    assert geometry.check(None, HANDS, arm_delta=np.ones(14)) > 0
    geometry.checks["right"].place = lambda *_a, **_k: np.array([[[0.021, 0, 0, 0.01]]])
    with pytest.raises(RestoreError, match="left/right hands"):
        geometry.check(None, HANDS, arm_delta=np.ones(14))
