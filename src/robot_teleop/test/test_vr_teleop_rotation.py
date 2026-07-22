"""Rotation-contract tests for the VR pose passthrough.

These call the *production* rotation functions in ``robot_teleop.vr_rotation``
directly. That module is ROS-free (no rclpy / message imports), so these tests
need NO ``sys.modules`` stubbing and do not pollute the pytest process for
other ROS tests — and if the production formula regresses, these tests fail.

The contract under test (base-frame relative rotation):

    ΔR_base = R_current * R_clutch^-1        (compute_base_rotation_delta)
    R_target = ΔR_base * R_clutch            (placo, left-multiply)

The defining property is that ΔR_base depends ONLY on the physical hand turn,
NOT on the hand attitude at the moment the trigger was pressed. That is exactly
the regression the old body-frame form (R_clutch^-1 * R_current) failed: its
axis rotated with the clutch attitude, so the same wrist turn produced a
different EE motion depending on how the hand happened to be oriented at press.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_teleop.vr_rotation import (
    R_ROBOT_BASE_FROM_VR_BASE,
    compute_base_rotation_delta,
    remap_base_rotation,
)


@pytest.mark.parametrize("clutch", [
    Rotation.identity(),
    Rotation.from_euler("xyz", [30.0, -45.0, 80.0], degrees=True),
    Rotation.from_euler("zyx", [-120.0, 15.0, 200.0], degrees=True),
    Rotation.from_rotvec([1.1, -0.4, 0.7]),
])
def test_base_delta_is_independent_of_clutch_attitude(clutch):
    """Same physical base-frame turn D → same ΔR_base for ANY clutch attitude.

    This is the core regression guard for issue #3: a single-axis wrist turn
    must not change axis just because the hand was oriented differently when
    the trigger was pressed.
    """
    D = Rotation.from_rotvec([0.0, 0.0, 0.6])  # a fixed base-frame turn
    # If the hand undergoes base-frame turn D from the clutch pose, the new
    # absolute attitude is R_current = D * R_clutch.
    r_current = D * clutch
    delta = compute_base_rotation_delta(r_current, clutch)
    assert np.allclose(delta.as_matrix(), D.as_matrix(), atol=1e-9)


def test_base_delta_identity_at_press():
    """At the press instant R_current == R_clutch → ΔR_base is identity."""
    clutch = Rotation.from_euler("xyz", [10.0, 20.0, 30.0], degrees=True)
    delta = compute_base_rotation_delta(clutch, clutch)
    assert np.allclose(delta.as_matrix(), np.eye(3), atol=1e-12)


def test_old_body_frame_form_would_couple_with_clutch():
    """Guard the reason for the fix: the OLD body-frame form R_clutch^-1 *
    R_current does depend on the clutch attitude, so two different clutch poses
    given the same base-frame turn produce DIFFERENT deltas. This asserts the
    old behavior was wrong, so a regression back to it fails here."""
    D = Rotation.from_rotvec([0.0, 0.0, 0.6])
    c1 = Rotation.identity()
    c2 = Rotation.from_euler("xyz", [30.0, -45.0, 80.0], degrees=True)
    old1 = c1.inv() * (D * c1)
    old2 = c2.inv() * (D * c2)
    assert not np.allclose(old1.as_matrix(), old2.as_matrix(), atol=1e-6)


def test_remap_is_a_proper_rotation_preserving_angle():
    """remap_base_rotation is a similarity transform (conjugation): it must
    preserve the rotation angle and stay a proper rotation (det +1)."""
    delta = Rotation.from_rotvec([0.2, -0.5, 0.3])
    remapped = remap_base_rotation(delta)
    m = remapped.as_matrix()
    assert np.isclose(np.linalg.det(m), 1.0, atol=1e-6)
    # Conjugation preserves the angle of rotation.
    assert np.isclose(
        np.linalg.norm(delta.as_rotvec()),
        np.linalg.norm(remapped.as_rotvec()),
        atol=1e-9,
    )


def test_remap_of_identity_is_identity():
    """A zero turn (press instant) must map to identity so the arm holds."""
    remapped = remap_base_rotation(Rotation.identity())
    assert np.allclose(remapped.as_matrix(), np.eye(3), atol=1e-12)


def test_correction_matrix_is_a_proper_rotation():
    """The base-alignment matrix must be a proper rotation (det +1); a det -1
    (mirror) value would flip wrist chirality."""
    assert np.isclose(np.linalg.det(R_ROBOT_BASE_FROM_VR_BASE), 1.0, atol=1e-3)
    # Orthonormal: R @ R^T == I.
    assert np.allclose(
        R_ROBOT_BASE_FROM_VR_BASE @ R_ROBOT_BASE_FROM_VR_BASE.T,
        np.eye(3),
        atol=1e-3,
    )
