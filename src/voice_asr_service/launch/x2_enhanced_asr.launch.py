"""Compare raw X2 audio and FullSubNet-enhanced audio using the same ASR.

This debugging entry consumes an already-running runtime audio source. All
results and control inputs stay under /asr_test; no robot runtime is launched.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap


def _launch_setup(context):
    models_root = Path(LaunchConfiguration("models_root").perform(context)).expanduser().resolve()
    bundle = str(Path(LaunchConfiguration("asr_bundle_path").perform(context)).expanduser().resolve())
    deployment = LaunchConfiguration("asr_deployment").perform(context)
    audio_topic = LaunchConfiguration("audio_topic").perform(context)
    channel = int(LaunchConfiguration("raw_channel").perform(context))
    channels = int(LaunchConfiguration("audio_channels").perform(context))
    if not 0 <= channel < channels:
        raise ValueError("raw_channel must be within audio_channels")
    # FullSubNet preserves channel ordering. Keep both ASRs on the same source.
    enhancement_channels = [channel] + [i for i in [*range(1, channels), 0] if i != channel][:3]
    if len(enhancement_channels) != 4:
        raise ValueError("FullSubNet requires at least four input channels")
    enhanced_topic = "/asr_test/audio_enhanced"
    share = Path(get_package_share_directory("voice_asr_service"))
    enhancer = GroupAction(
        [
            SetRemap(src="/voice/speech_direction", dst="/asr_test/speech_direction"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(share / "launch" / "speech_direction.launch.py")),
                launch_arguments={
                    "profile": LaunchConfiguration("profile").perform(context),
                    "models_root": str(models_root),
                    "speech_direction_audio_topic": audio_topic,
                    "speech_direction_enhanced_audio_topic": enhanced_topic,
                    "microphone_channels": str(channels),
                    "microphone_sample_rate": "16000",
                    "microphone_channel_indices": repr(enhancement_channels),
                }.items(),
            ),
        ]
    )
    actions = [enhancer]
    for name, topic, count, selected in (
        ("enhanced", enhanced_topic, 4, 0),
        ("raw", audio_topic, channels, channel),
    ):
        prefix = f"/asr_test/{name}"
        actions.append(
            Node(
                package="voice_asr_service",
                executable="voice_asr_node",
                name=f"asr_test_{name}",
                output="screen",
                parameters=[
                    {
                        "active_mode": "continuous",
                        "bundle_path": bundle,
                        "deployment": deployment,
                        "vad_bundle_path": str(models_root / "silero-vad"),
                        "vad_deployment": "torch_cpu",
                        "audio_topic": topic,
                        "audio_channels": count,
                        "audio_input_channel": selected,
                        "output_topic": f"{prefix}/text",
                    }
                ],
                remappings=[
                    (f"/voice_{suffix}", f"{prefix}/{suffix}")
                    for suffix in (
                        "partial",
                        "status",
                        "confidence",
                        "file_progress",
                        "control",
                        "file_input",
                    )
                ],
                condition=IfCondition(LaunchConfiguration("compare_raw")) if name == "raw" else None,
            )
        )
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("models_root", description="Root containing fullsubnet and silero-vad bundles"),
            DeclareLaunchArgument("asr_bundle_path", description="Schema-v3 streaming sherpa ASR bundle"),
            DeclareLaunchArgument("asr_deployment", default_value="torch_cpu"),
            DeclareLaunchArgument("profile", default_value="ascend_310p"),
            DeclareLaunchArgument("audio_topic", default_value="/audio/capture_stamped"),
            DeclareLaunchArgument("audio_channels", default_value="6"),
            DeclareLaunchArgument("raw_channel", default_value="1"),
            DeclareLaunchArgument("compare_raw", default_value="true"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
