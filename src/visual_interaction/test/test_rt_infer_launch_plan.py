"""Full plugin-plan coverage lives in a leaf package, avoiding test cycles."""

from inference_service.launch_builders.perception_models import PERCEPTION_MODELS_BUILDER
from robot_config.launch_builders.rt_infer_scheduler import RT_INFER_SCHEDULER_BUILDER
from robot_config.launch_registry.registry import LaunchBuildContext, LaunchBuilderRegistry
from visual_interaction.launch_builders.visual_interaction import VISUAL_INTERACTION_BUILDER
from voice_tts_service.launch_builders.voice_tts import VOICE_TTS_BUILDER


def test_rt_infer_plan_orders_models_after_owner_and_drops_disabled_edge():
    registry = LaunchBuilderRegistry()
    registry.add_all(
        (PERCEPTION_MODELS_BUILDER, VOICE_TTS_BUILDER, VISUAL_INTERACTION_BUILDER, RT_INFER_SCHEDULER_BUILDER)
    )
    enabled = LaunchBuildContext(robot_config={"rt_infer": {"enable": True, "bundle_version": "2026.11.05"}})
    assert [spec.id for spec in registry.plan(enabled).specs] == [
        "rt_infer_scheduler",
        "perception_models",
        "voice_tts",
        "visual_interaction",
    ]
    disabled = LaunchBuildContext(robot_config={"rt_infer": {"enable": False}})
    assert [spec.id for spec in registry.plan(disabled).specs] == [
        "perception_models",
        "voice_tts",
        "visual_interaction",
    ]
