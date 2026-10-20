"""Import-budget test for visual_interaction's launch-builder contribution.

Resolving the ``ibrobot.launch_builders`` entry point imports this module on
every launch description generation, whether or not the feature is enabled, so
it must not drag in a non-stdlib top-level package beyond the registry core
budget and this package itself.

This is what the function-local imports in the builder are for: the contract
module, the endpoint constants and the session package are read inside
``generate_visual_interaction_nodes``, not at module scope.
"""

from __future__ import annotations

from robot_config.launch_registry.import_budget import (
    HARD_DEPENDENCY_ALLOWANCE,
    core_import_budget,
    imported_top_level_packages,
)


def test_the_launch_builder_stays_within_the_core_import_budget():
    core = core_import_budget()
    loaded = imported_top_level_packages("import visual_interaction.launch_builders.visual_interaction\n")

    overflow = sorted(loaded - core - HARD_DEPENDENCY_ALLOWANCE - {"visual_interaction"})

    assert overflow == [], (
        f"importing visual_interaction.launch_builders pulled non-core top-level packages: {overflow}; "
        "extension modules must stay import-light"
    )


def test_importing_the_builder_does_not_pull_in_the_session_or_the_contract():
    """The session and its contract load when the feature runs, not at parse time.

    This is what the function-local imports buy: a deployment with the feature
    disabled pays for neither. ``rclpy`` is not checked here — ``launch_ros``
    is part of the core budget and brings it along, so its presence says
    nothing about this module.
    """
    loaded = imported_top_level_packages("import visual_interaction.launch_builders.visual_interaction\n")

    assert "embodied_common" not in loaded
    assert "ibrobot_msgs" not in loaded


def test_the_node_module_is_not_imported_at_parse_time():
    """Importing the node would load rclpy's node machinery for every launch."""
    loaded = imported_top_level_packages(
        "import sys\n"
        "import visual_interaction.launch_builders.visual_interaction\n"
        "assert 'visual_interaction.visual_interaction_node' not in sys.modules\n"
        "assert 'visual_interaction.session.core' not in sys.modules\n"
    )

    assert loaded
