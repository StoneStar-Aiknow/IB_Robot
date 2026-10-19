"""Participation predicate of the navigation entry (robot_navigation's contribution).

Migrated from robot_config's registry suite (openspec change
``migrate-business-launch-builders``, task B.7): the ``with_navigation``
CLI-over-YAML resolution and the mock-backend skip are declared on
``NAVIGATION_BUILDER``, so their tests follow the entry. The entry point is
injected into ``build_launch_registry`` so the plan stays hermetic.
"""

from importlib.metadata import EntryPoint

from robot_config.launch_builders.builtin_registry import build_launch_registry
from robot_config.launch_builders.registry import LaunchBuildContext
from robot_config.runtime_target import RuntimeTarget


def _plan_ids(**overrides) -> list[str]:
    registry = build_launch_registry(
        entry_points=[
            EntryPoint(
                name="navigation",
                value="robot_navigation.launch_builders.navigation:NAVIGATION_BUILDER",
                group="ibrobot.launch_builders",
            )
        ]
    )
    defaults = {
        "robot_config": {"name": "test_robot"},
        "runtime_target": RuntimeTarget.HARDWARE,
        "use_sim": False,
        "sim_platform": "",
        "control_mode": "model_inference",
        "with_inference": True,
        "arguments": {},
    }
    defaults.update(overrides)
    return [spec.id for spec in registry.plan(LaunchBuildContext(**defaults)).specs]


def test_navigation_follows_cli_switch_and_yaml_fallback():
    # CLI false beats YAML enabled=true.
    assert "navigation" not in _plan_ids(
        arguments={"with_navigation": "false"},
        robot_config={"name": "r", "navigation": {"enabled": True}},
    )
    # CLI true beats YAML enabled=false.
    assert "navigation" in _plan_ids(arguments={"with_navigation": "true"})
    # Empty CLI falls back to the prepared YAML flag.
    assert "navigation" not in _plan_ids(robot_config={"name": "r", "navigation": {"enabled": False}})
    assert "navigation" in _plan_ids(robot_config={"name": "r", "navigation": {"enabled": True}})


def test_mock_backend_skips_navigation_at_plan_time():
    assert "navigation" not in _plan_ids(
        arguments={"with_navigation": "true"},
        use_sim=True,
        sim_platform="mock",
        runtime_target=RuntimeTarget.SIMULATION,
    )
