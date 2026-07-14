"""Pure rotation math for the VR pose passthrough — no ROS dependency.

These functions implement the *exact* rotation formulas the VR teleop node runs
at 50 Hz. They are pulled out of ``vr_teleop`` so tests can import and exercise
the real production code directly (rather than re-deriving the formula in the
test file, where a regression in the node would go undetected), and so those
tests never have to stub ``rclpy`` / message packages into ``sys.modules``.

Nothing here touches a ROS node, topic, parameter, socket, or global mutable
state: output depends only on the arguments.

Contract (base-frame relative rotation):

    ΔR_base = R_current * R_clutch^-1        (compute_base_rotation_delta)
    R_target = ΔR_base * R_clutch            (placo, left-multiply)

ΔR_base depends ONLY on the physical hand turn, not on the hand attitude at the
moment the trigger was pressed — that is the property the old body-frame form
(R_clutch^-1 * R_current) failed.
"""

import numpy as np
from scipy.spatial.transform import Rotation

# Base-frame alignment VR-base -> robot-base (proper rotation, det +1): ~180 deg
# about ~base-x. Face-to-face teleop puts the VR base ~180 deg about the forward
# axis from the arm base.
#
# Contract this matrix serves:
#     delta_R  = R_current * R_clutch.inv()      (compute_base_rotation_delta)
#     R_target = delta_R @ R_ee_clutch            (placo, left-multiply)
#
# Measured axis mapping (VR base -> robot base), R @ axis:
#     VR +X -> Robot +X   (EE roll  — tracked)
#     VR +Y -> Robot -Y   (EE pitch — tracked)
#     VR +Z -> Robot -Z   (EE yaw   — NOT tracked, see below)
#
# 5-DOF LIMITATION — rotation about robot base +Z is NOT reproducible.
# SO-101 has only 5 revolute joints, so the EE cannot independently realise all
# 6 Cartesian DOF. placo pins the 3 position DOF with a hard PositionTask and
# follows orientation with a low-weight (0.01) soft OrientationTask, leaving only
# ~2 attainable orientation DOF. For this kinematic layout (shoulder-pan + four
# pitch/wrist joints) the unattainable one is base-Z EE yaw: hand roll/pitch map
# to EE roll/pitch, but a hand yaw about base +Z produces (almost) no EE motion.
# This is a structural property of the arm, NOT a calibration error — no choice
# of this matrix recovers base-Z yaw without sacrificing a position DOF.
#
# The X/Y rotation mapping is validated on sim (correct axis, usable sign). If a
# task needs a firm wrist attitude and the soft 2-DOF follow is undesirable, set
# position_only=true to teleop position only; otherwise rotation stays ON.
R_ROBOT_BASE_FROM_VR_BASE = np.array([[0.90098, -0.43079, -0.05147],
                                      [-0.43079, -0.90238, 0.01166],
                                      [-0.05147, 0.01166, -0.99861]])


def compute_base_rotation_delta(current: Rotation, clutch: Rotation) -> Rotation:
    """Base-frame rotation delta ``current * clutch^-1``.

    At the press instant ``current == clutch`` so the delta is identity and the
    EE attitude does not jump. The delta is a BASE-frame increment (same frame
    as the position delta and the ``pose_cmd_base`` topic): its axis is fixed in
    the base frame, independent of the hand attitude when the trigger was
    pressed. The body/tool-frame form ``clutch^-1 * current`` would instead make
    the axis rotate with the clutch attitude — the coupling bug this avoids.
    """
    return current * clutch.inv()


def remap_base_rotation(
    delta: Rotation, robot_base_from_vr_base: np.ndarray = R_ROBOT_BASE_FROM_VR_BASE
) -> Rotation:
    """Re-express a base-frame rotation delta from the VR base frame into the
    robot base frame with a SIMILARITY transform ``R * delta * R^T``
    (conjugation).

    This is a pure change of basis: it rotates the delta's axis by ``R`` while
    leaving its angle untouched (never a mirror, since ``R`` is a proper
    rotation). See :data:`R_ROBOT_BASE_FROM_VR_BASE`.
    """
    matrix = delta.as_matrix()
    corrected = robot_base_from_vr_base @ matrix @ robot_base_from_vr_base.T
    return Rotation.from_matrix(corrected)
