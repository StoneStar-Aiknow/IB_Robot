"""Platform-specific default camera poses for simulation.

When a camera peripheral in the robot YAML has ``use_default_transform: true``,
the launch system looks up default transform and fovy values from this module
instead of using the (zeroed-out) values in the YAML.

The Gazebo column was removed (openspec ``so101-gazebo-runtime-transport``
D6.2): those values were SO-101 defaults and are now runtime-owned, declared
per-camera as ``simulation.gazebo.transform`` (+ ``fovy``) in the SO-101
runtime profile. The MuJoCo column was removed the same way
(so101-mujoco-runtime-platform D5): the SO-101 MuJoCo defaults live in the
runtime profile as ``simulation.mujoco.transform`` (+ ``fovy``), and the
gazebo→mujoco orientation conversion moved to ``robot_runtime.mujoco``
(``gazebo_rpy_to_mujoco_rpy``), next to the shared Gazebo camera code.

``get_preset`` stays as the generic default-pose loader hook: provider-less
generic Gazebo users declare their poses explicitly, so it now returns None
for every platform (the ``sim_camera_adjuster``'s guarded lookups keep
working, exactly as they already do for the removed Gazebo column).
"""


def get_preset(platform: str, camera_name: str) -> dict | None:
    """Look up a default camera transform for a given platform and camera name.

    Args:
        platform: "gazebo" or "mujoco".
        camera_name: The ``name`` field from the YAML peripheral entry.

    Returns:
        None — both preset columns were removed (Gazebo: runtime-owned since
        ``so101-gazebo-runtime-transport`` D6.2; MuJoCo: runtime-owned since
        ``so101-mujoco-runtime-platform`` D5). The function remains as the
        default-pose loader hook ``robot_runtime.gazebo`` callers inject.
    """
    return None
