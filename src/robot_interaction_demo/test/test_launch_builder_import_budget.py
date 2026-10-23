"""Import-budget test for robot_interaction_demo's launch-builder contribution.

Openspec change ``migrate-business-launch-builders``, design decision 6:
importing this package's ``launch_builders`` modules (which the
``ibrobot.launch_builders`` entry point resolution does on every launch
description generation) must not pull in non-stdlib top-level packages
beyond the registry core budget and this package itself. The subprocess
probing and the core-budget computation are shared through
``robot_config.launch_registry.import_budget``.
"""

from __future__ import annotations

from robot_config.launch_registry.import_budget import (
    HARD_DEPENDENCY_ALLOWANCE,
    core_import_budget,
    imported_top_level_packages,
)


def test_launch_builder_modules_stay_within_the_core_import_budget():
    core = core_import_budget()
    loaded = imported_top_level_packages("import robot_interaction_demo.launch_builders.interaction_demo\n")
    overflow = sorted(loaded - core - HARD_DEPENDENCY_ALLOWANCE - {"robot_interaction_demo"})
    assert overflow == [], (
        f"importing robot_interaction_demo.launch_builders pulled non-core top-level packages: {overflow}; "
        "extension modules must stay import-light (design decision 6)"
    )
