"""SO-101 robot suite: robot-specific grasp geometry and wrist guards.

Contains the SO-101-specific parts extracted from the former generic-package
SO-101 modules (design D8):

- Gripper mesh geometry (STL loading, convex hull, jaw motion)
- Joint-5 wrist roll guards (branch continuity, closing-axis correction)

The generic manipulation pipeline in ``manipulation_execution`` consumes
these through configuration-declared providers. A robot config that enables
the orientation guard or the tabletop filter without declaring this suite's
providers fails fast at node startup, naming the missing configuration
field. The suite is a dependency leaf: it depends on no IB-Robot generic
package, so a robot runtime (e.g. ``lekiwi_robot``) can install it without
the manipulation/inference closure.
"""

from so101_suite.gripper_geometry import (
    gripper_geometry_metrics_batch,
    gripper_mesh_min_z,
    gripper_mesh_vertices,
    tabletop_clearance,
)
from so101_suite.wrist_guard import (
    Joint5RetryResult,
    apply_joint5_retry,
    canonicalize_joint5,
    joint5_branch_continuity_check,
    joint5_branch_delta,
    joint5_branch_filter_check,
    joint5_closing_axis_correction,
    joint5_within_abs_limit,
)

__all__ = [
    "Joint5RetryResult",
    "apply_joint5_retry",
    "canonicalize_joint5",
    "gripper_geometry_metrics_batch",
    "gripper_mesh_min_z",
    "gripper_mesh_vertices",
    "joint5_branch_continuity_check",
    "joint5_branch_delta",
    "joint5_branch_filter_check",
    "joint5_closing_axis_correction",
    "joint5_within_abs_limit",
    "tabletop_clearance",
]
