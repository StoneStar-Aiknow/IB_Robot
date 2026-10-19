"""Mock-backend skip of the speech_direction entry (voice_asr_service's contribution).

Migrated from robot_config's registry suite (openspec change
``migrate-business-launch-builders``, task B.4): ``SPEECH_DIRECTION_BUILDER``
declares ``mock_skip_key="speech_direction"``, so the plan-time skip test
follows the entry. The entry point is injected into ``build_launch_registry``
so the plan stays hermetic.
"""

from importlib.metadata import EntryPoint

from robot_config.launch_builders.builtin_registry import build_launch_registry
from robot_config.launch_builders.registry import LaunchBuildContext
from robot_config.runtime_target import RuntimeTarget


def _plan_ids(**overrides) -> list[str]:
    registry = build_launch_registry(
        entry_points=[
            EntryPoint(
                name="speech_direction",
                value="voice_asr_service.launch_builders.speech_direction:SPEECH_DIRECTION_BUILDER",
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


def test_speech_direction_participates_outside_mock():
    assert "speech_direction" in _plan_ids()


def test_mock_backend_skips_speech_direction_at_plan_time():
    assert "speech_direction" not in _plan_ids(
        use_sim=True,
        sim_platform="mock",
        runtime_target=RuntimeTarget.SIMULATION,
    )
