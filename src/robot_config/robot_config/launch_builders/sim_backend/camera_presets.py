"""Platform-specific default camera poses for simulation.

When a camera peripheral in the robot YAML has ``use_default_transform: true``,
the launch system looks up default transform and fovy values from this module
instead of using the (zeroed-out) values in the YAML.

Two mount categories:
  "wrist" — gripper-mounted palm-plate camera, follows the end-effector.
  "base"  — generic workspace camera fixed on the robot base body.
             Used by any camera whose name is NOT "wrist" (top, front, etc.).

Convention:
  Values are stored in each platform's native coordinate convention:
    - Gazebo (gz-sim): camera forward = +X, up = +Z
    - MuJoCo:          camera forward = -Z, up = +Y
  No cross-platform conversion is needed at runtime — each adapter reads its
  own column directly.
"""

PRESETS = {
    "gazebo": {
        "wrist": {
            "parent_frame": "gripper",
            "x": 0.002,
            "y": 0.061,
            "z": -0.025,
            "roll": -1.5708,
            "pitch": 1.1708,
            "yaw": -1.5708,
            "fovy": 65,
        },
        "base": {
            "parent_frame": "base",
            "x": 0.0,
            "y": -0.28,
            "z": 0.5,
            "roll": 0.0,
            "pitch": 1.5708,
            "yaw": -1.5708,
            "fovy": 70,
        },
    },
    "mujoco": {
        "wrist": {
            "parent_frame": "gripper",
            "x": 0.002,
            "y": 0.061,
            "z": -0.025,
            "roll": 0.0,
            "pitch": -0.4,
            "yaw": -1.5708,
            "fovy": 65,
        },
        "base": {
            "parent_frame": "base",
            "x": 0.0,
            "y": -0.28,
            "z": 0.55,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 3.1415,
            "fovy": 70,
        },
    },
}


def get_preset(platform: str, camera_name: str) -> dict | None:
    """Look up default camera transform for a given platform and camera name.

    Args:
        platform: "gazebo" or "mujoco".
        camera_name: The ``name`` field from the YAML peripheral entry.
                     "wrist" → wrist preset; anything else → base preset.

    Returns:
        A dict with keys (parent_frame, x, y, z, roll, pitch, yaw, fovy),
        or None if the platform is unknown.
    """
    platform_presets = PRESETS.get(platform)
    if not platform_presets:
        return None
    if camera_name == "wrist":
        return platform_presets.get("wrist")
    return platform_presets.get("base")


# ---------------------------------------------------------------------------
# Coordinate-system conversion: Gazebo camera frame → MuJoCo camera frame
# ---------------------------------------------------------------------------
#
# Gazebo SDF camera link convention:   optical = +X,  image-up = +Z,  image-right = -Y
# MuJoCo camera body convention:       optical = -Z,  image-up = +Y,  image-right = +X
#
# R_fix maps MuJoCo camera basis into Gazebo camera basis (columns = e_x^M,
# e_y^M, e_z^M expressed in Gazebo coords).  With R_WG the calibrated Gazebo
# camera rotation in world, the equivalent MuJoCo camera rotation is
# R_WM = R_WG @ R_fix.  det(R_fix) = +1 (proper rotation).
#
# This conversion relies on the URDF/MJCF body hierarchy being consistent
# (same parent frame orientation in both backends), which is true for the
# SO101 model because mujoco_adapter uses URDF-derived body origins with
# eulerseq="XYZ" (matching URDF rpy extrinsic XYZ).
#
# Used by ``mujoco_adapter`` to translate ``sim_camera_adjuster`` output
# (always saved in Gazebo convention) into MuJoCo ``<camera euler="...">``.

_R_FIX_GZ_TO_MJ = (
    (0.0, 0.0, -1.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)


def gazebo_rpy_to_mujoco_rpy(gz_roll: float, gz_pitch: float, gz_yaw: float) -> tuple[float, float, float]:
    """Convert (roll, pitch, yaw) from Gazebo camera convention to MuJoCo.

    Input and output are extrinsic XYZ Euler angles in radians, matching both
    URDF ``rpy`` and MuJoCo ``eulerseq="XYZ"``.
    """
    import math

    def rpy_to_R(r, p, y):
        cr, sr = math.cos(r), math.sin(r)
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)
        # R = Rz(y) @ Ry(p) @ Rx(r)
        return (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )

    def mat_mul_3x3(A, B):
        return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)) for i in range(3))

    def R_to_rpy(R):
        # Inverse of R = Rz(y) @ Ry(p) @ Rx(r).
        # R[2][0] = -sin(p);  R[2][1] = cos(p)*sin(r);  R[2][2] = cos(p)*cos(r)
        # R[1][0] = cos(p)*sin(y);  R[0][0] = cos(p)*cos(y)
        sp = max(-1.0, min(1.0, -R[2][0]))
        pitch = math.asin(sp)
        cp = math.cos(pitch)
        if abs(cp) > 1e-6:
            roll = math.atan2(R[2][1], R[2][2])
            yaw = math.atan2(R[1][0], R[0][0])
        else:
            # Gimbal lock (pitch = ±pi/2); roll and yaw are coupled.
            # Convention: set roll = 0, resolve yaw.
            roll = 0.0
            yaw = math.atan2(-R[0][1], R[1][1])
        return roll, pitch, yaw

    R_WG = rpy_to_R(gz_roll, gz_pitch, gz_yaw)
    R_WM = mat_mul_3x3(R_WG, _R_FIX_GZ_TO_MJ)
    return R_to_rpy(R_WM)
