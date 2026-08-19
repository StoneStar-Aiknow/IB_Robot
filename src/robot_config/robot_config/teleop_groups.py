"""Compatibility imports for the teleoperation command-group contract.

New code imports from :mod:`robot_teleop.teleop_groups`; this module remains
for downstream callers during the package-boundary migration.
"""

from robot_teleop.teleop_groups import (
    LEGACY_TARGET_KEYS,
    PublishGroup,
    legacy_publish_groups,
    parse_publish_groups,
    resolve_node_publish_groups,
    resolve_target_publish_groups,
)

__all__ = [
    "LEGACY_TARGET_KEYS",
    "PublishGroup",
    "legacy_publish_groups",
    "parse_publish_groups",
    "resolve_node_publish_groups",
    "resolve_target_publish_groups",
]
