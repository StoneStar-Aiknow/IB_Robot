"""Participation predicate of the teleop entry (robot_teleop's contribution).

Migrated from robot_config's registry suite (openspec change
``migrate-business-launch-builders``, task B.3): ``TELEOP_BUILDER`` declares
that it only participates in the teleop control mode, so the test follows the
entry. The entry point is injected into ``build_launch_registry`` so the plan
stays hermetic.
"""

from importlib.metadata import EntryPoint

from robot_config.launch_registry.builtin_registry import build_launch_registry
from robot_config.launch_registry.registry import LaunchBuildContext
from robot_config.runtime_target import RuntimeTarget


def _plan_ids(**overrides) -> list[str]:
    registry = build_launch_registry(
        entry_points=[
            EntryPoint(
                name="teleop",
                value="robot_teleop.launch_builders.teleop:TELEOP_BUILDER",
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


def test_teleop_only_participates_in_teleop_mode():
    assert "teleop" not in _plan_ids(control_mode="model_inference")
    assert "teleop" in _plan_ids(control_mode="teleop")
