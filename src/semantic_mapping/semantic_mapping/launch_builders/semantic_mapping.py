"""Launch the online semantic-mapping node from the robot_config SSOT.

This module is ``semantic_mapping``'s contribution to the launch-builder
registry (openspec change ``migrate-business-launch-builders``): the
``semantic_mapping`` entry is declared as an entry point in the
``ibrobot.launch_builders`` group (see ``setup.py``), so the node's owning
package registers its own launch wiring without any robot_config change.
The content is a verbatim migration of the former
``robot_config/launch_builders/semantic_mapping.py``.

The generic model services (SAM2 masks, RAM++ tags, SigLIP2 image/text) are
already started by ``generate_perception_model_nodes`` because they are plain
``perception_services.services`` entries.  This builder adds the remaining
piece: the ``semantic_mapping_node`` itself, with the validated SSOT sections
flattened into its parameter contract.

The parameter flattening intentionally lives in ``semantic_mapping`` and is
imported here at launch time (not at module import time) so ``robot_config``
keeps no static dependency on ``semantic_mapping``; the dependency direction
stays ``semantic_mapping`` -> ``robot_config``.
"""

from typing import Any

from launch_ros.actions import Node

from robot_config.launch_registry.registry import BuilderSpec, BuildResult, LaunchBuildContext
from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.semantic_mapping")


def generate_semantic_mapping_nodes(robot_config: dict[str, Any]) -> list[Node]:
    """Generate the online semantic mapping node when it is enabled.

    Args:
        robot_config: Loaded (stage-resolved) robot configuration.

    Returns:
        A single-element list with the mapping node, or an empty list when
        ``robot.semantic_mapping.enabled`` is false.
    """
    if not robot_config.get("semantic_mapping", {}).get("enabled", False):
        return []

    # Deferred import: keeps robot_config importable without semantic_mapping.
    from semantic_mapping.configuration import semantic_mapping_parameters

    return [
        Node(
            package="semantic_mapping",
            executable="semantic_mapping_node",
            name="semantic_mapping",
            output="screen",
            parameters=[semantic_mapping_parameters(robot_config)],
        )
    ]


__all__ = ["generate_semantic_mapping_nodes"]


# ---------------------------------------------------------------------------
# Registry entry (openspec change ``decentralize-launch-builder-registration``):
# the BuilderSpec constant and build function moved here from the retired
# ``adapters.py``; the body is a verbatim migration. Openspec change
# ``migrate-business-launch-builders`` moved this module from
# ``robot_config/launch_builders/semantic_mapping.py`` to its owning package.
# ---------------------------------------------------------------------------


def build_semantic_mapping(context: LaunchBuildContext) -> BuildResult:
    from semantic_mapping.launch_builders.semantic_mapping import generate_semantic_mapping_nodes

    nodes = generate_semantic_mapping_nodes(context.robot_config)
    if nodes:
        logger.info(f"Added {len(nodes)} semantic mapping node(s)")
    return BuildResult(actions=list(nodes))


SEMANTIC_MAPPING_BUILDER = BuilderSpec(
    id="semantic_mapping",
    factory=f"{__name__}:build_semantic_mapping",
    timing="runtime_ready",
)
