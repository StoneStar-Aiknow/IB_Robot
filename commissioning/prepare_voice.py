"""Configure the existing X2 apple Agent with ASR and runtime-owned TTS playback."""

import copy
import json
import shutil
from pathlib import Path

import yaml

root = Path(__file__).resolve().parents[1]
stage = root / "commissioning"
robots = stage / "robots"
robots.mkdir(exist_ok=True)
for source in (root / "src/robot_config/config/robots").glob("*.yaml"):
    shutil.copy2(source, robots / source.name)
state = stage / "voice_state"
state.mkdir(exist_ok=True)
base = {
    "base_config": "aimdk_x2_grasp_310p_agent.yaml",
    "voice_asr": {
        "enabled": True,
        "input_mode": "shared_audio",
        "active_mode": "continuous",
        "bundle_path": "models/voice_asr/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-int8-2026-06-05",
        "deployment": "torch_cpu",
        "vad_bundle_path": "models/silero-vad",
        "vad_deployment": "torch_cpu",
        "audio_input": {"interface": "audio.capture", "type": "audio_common_msgs/msg/AudioDataStamped"},
        "output_topic": "/voice_command",
    },
    "embodied": {
        "execution_store_path": str(state / "executions.sqlite3"),
        "skill_catalog_source_root": str(root / "src/agent/skill_catalog"),
        "agent": {
            "execution_enabled": False,
            "channel_id": "voice_asr",
            "principal_id": "local_operator",
            "test_allowlist": ["pick_object"],
            "ledger_path": str(state / "requests.sqlite3"),
            "conversation_path": str(state / "conversation.sqlite3"),
            "deployment_lock_path": str(state / "agent.lock"),
            "voice_bridge": {
                "enabled": True,
                "feedback_only": False,
                "confirm_mode": "auto",
                "mute_asr_during_tts": True,
                "tts_timeout_sec": 90.0,
            },
        },
    },
}
for name, enabled in [("hz_voice_apple_dryrun", False), ("hz_voice_apple_live", True)]:
    config = copy.deepcopy(base)
    config["embodied"]["agent"]["execution_enabled"] = enabled
    (robots / f"{name}.yaml").write_text(yaml.safe_dump({"robot": config}, sort_keys=False, allow_unicode=True))
smoke = {"robot": {"base_config": "hz_voice_apple_dryrun.yaml", "voice_asr": {"active_mode": "manual"}}}
(robots / "hz_voice_apple_smoke.yaml").write_text(yaml.safe_dump(smoke, sort_keys=False))
head = {"robot": {"base_config": "aimdk_x2_grasp_310p.yaml"}}
(robots / "hz_grasp_head.yaml").write_text(yaml.safe_dump(head, sort_keys=False))
# Site profiles retain reference gestures, introduction and picking, excluding mirror.
site_skills = ["wave_hand", "handshake", "raise_hand", "blow_kiss", "clap_hands", "self_introduction", "pick_object"]
catalog = stage / "site_catalog" / "config"
(catalog / "profiles").mkdir(parents=True, exist_ok=True)
shutil.copytree(root / "src/agent/skill_catalog/config/skills", catalog / "skills", dirs_exist_ok=True)
source_profile = yaml.safe_load(
    (root / "src/agent/skill_catalog/config/profiles/aimdk_x2_voice_mirror_grasp.yaml").read_text()
)
source_profile.update({"name": "hz_site_no_mirror", "robot_name": "aimdk_x2_grasp"})
source_profile["enabled_skills"] = [item for item in source_profile["enabled_skills"] if item["name"] in site_skills]
(catalog / "profiles/hz_site_no_mirror.yaml").write_text(yaml.safe_dump(source_profile, sort_keys=False))
mirror_skills = [*site_skills, "mirror_start", "mirror_stop"]
mirror_profile = yaml.safe_load(
    (root / "src/agent/skill_catalog/config/profiles/aimdk_x2_voice_mirror_grasp.yaml").read_text()
)
mirror_profile.update({"name": "hz_site_mirror", "robot_name": "aimdk_x2_grasp"})
(catalog / "profiles/hz_site_mirror.yaml").write_text(yaml.safe_dump(mirror_profile, sort_keys=False))
voice_reference = yaml.safe_load((robots / "aimdk_x2_voice_mirror_all.yaml").read_text())["robot"]
reference = yaml.safe_load((robots / "aimdk_x2_grasp_310p_agent.yaml").read_text())["robot"]
grasp = yaml.safe_load((robots / "aimdk_x2_grasp.yaml").read_text())["robot"]
models = yaml.safe_load((robots / "aimdk_x2_grasp_310p.yaml").read_text())["robot"]["perception_services"]
for name, enabled in [
    ("hz_site_dryrun", False),
    ("hz_site_live", True),
    ("hz_site_mirror_preview", False),
    ("hz_site_mirror_live", True),
]:
    config = copy.deepcopy(base)
    config.update(
        {
            "base_config": "aimdk_x2_voice_mirror_grasp.yaml",
            "name": "aimdk_x2_grasp",
            "default_control_mode": "moveit_planning",
            "skill_required_control_mode": "moveit_planning",
            "skill_supported_control_modes": ["moveit_planning", "named_motion"],
            "control_modes": {
                "named_motion": {
                    "runtime_mode": "policy_stream",
                    "description": "Recorded gestures on the shared upper-body stream",
                    "description_cn": "在抓取共用的 URS 模式回放已录制手势",
                    "inference": {"enabled": False},
                    "executor": {"enabled": False},
                }
            },
            "arm_limits": grasp["arm_limits"],
            "grasp_execution": copy.deepcopy(grasp["grasp_execution"]),
            "perception_services": copy.deepcopy(models),
            "motion_imitation_stream": {"enabled": False},
            "visual_interaction": {"enabled": False},
            "mirror_session": {
                "enabled": False,
                "publish_enabled": False,
                "voice": {"enabled": False},
                "operator": {"enabled": False, "send_enabled": False},
            },
        }
    )
    config["grasp_execution"]["model_deployment"] = "torch_npu"
    config["grasp_execution"]["backend_options"].update({"segment_service": "/perception/grasp/segment_detections"})
    config["voice_asr"].update(
        {
            "input_mode": "external_vad",
            # X-ASR uses prefixed subword tokens; the character hotword list is incompatible.
            "hotwords_file": "",
            "audio_input": {"interface": "audio.speech", "type": "ibrobot_msgs/msg/SpeechAudioChunk"},
        }
    )
    # Text wake and speaker checks are opt-in for this commissioning deployment.
    config["voice_asr"]["speaker_embedding"] = {"enabled": True}
    config["voice_tts"] = copy.deepcopy(reference["voice_tts"])
    embodied = config["embodied"]
    embodied.update(
        {
            "enabled": True,
            "entry_mode": "agent",
            "arm_surface": False,
            "skill_catalog_profile": "hz_site_no_mirror",
            "skill_catalog_source_mode": "development",
            "skill_catalog_source_root": str(stage / "site_catalog"),
            "runtime_named_motion": copy.deepcopy(voice_reference["embodied"]["runtime_named_motion"]),
            "runtime_play_audio": copy.deepcopy(voice_reference["embodied"]["runtime_play_audio"]),
        }
    )
    # Keep the reference introduction skill without depending on its author's absolute path.
    embodied["runtime_play_audio"]["clips"]["self_intro"] = str(stage / "audio/self_introduction.wav")
    agent = embodied["agent"]
    agent.update(
        {
            "enabled": True,
            "incubation": True,
            "execution_enabled": enabled,
            "planner": copy.deepcopy(reference["embodied"]["agent"]["planner"]),
        }
    )
    agent["test_allowlist"] = site_skills
    agent["voice_bridge"]["silent_skills"] = ["self_introduction"]
    agent["voice_bridge"].update(
        {
            "addressee_trigger": "wake_phrase",
            "wake_window_sec": 60.0,
            "wake_phrases": [
                "小欧小欧",
                "小欧小鸥",
                "小欧小偶",
                "小鸥小欧",
                "小鸥小鸥",
                "小鸥小偶",
                "小偶小欧",
                "小偶小鸥",
                "小偶小偶",
            ],
            "speaker_check_enabled": True,
        }
    )
    agent["voice_bridge"]["confirm_mode"] = "voice_confirm"
    if name.startswith("hz_site_mirror_"):
        # Opt-in mirror profiles retain all seven existing skills. Only the
        # explicitly selected live profile publishes physical arm commands.
        config["default_control_mode"] = "model_inference"
        config["skill_required_control_mode"] = "model_inference"
        config["skill_supported_control_modes"].append("model_inference")
        config["motion_imitation_stream"] = {"enabled": True}
        config["mirror_session"].update({"enabled": True, "control_mode": "model_inference"})
        config["mirror_session"]["publish_enabled"] = enabled
        config["mirror_session"]["operator"] = {"enabled": True, "send_enabled": enabled}
        mirror_models = yaml.safe_load((robots / "aimdk_x2_magicmirror.yaml").read_text())["robot"][
            "perception_services"
        ]["services"]
        config["perception_services"]["services"] = copy.deepcopy(mirror_models) + copy.deepcopy(models["services"])
        embodied["skill_catalog_profile"] = "hz_site_mirror"
        agent["test_allowlist"] = mirror_skills
    site_state = stage / "site_state" / name
    site_state.mkdir(parents=True, exist_ok=True)
    embodied["execution_store_path"] = str(site_state / "executions.sqlite3")
    for field, filename in [
        ("ledger_path", "requests.sqlite3"),
        ("conversation_path", "conversation.sqlite3"),
        ("deployment_lock_path", "agent.lock"),
    ]:
        agent[field] = str(site_state / filename)
    (robots / f"{name}.yaml").write_text(yaml.safe_dump({"robot": config}, sort_keys=False, allow_unicode=True))

(stage / "logs").mkdir(exist_ok=True)
print(
    json.dumps(
        {
            "workspace": str(root),
            "configs": [
                "hz_voice_apple_dryrun",
                "hz_voice_apple_live",
                "hz_grasp_head",
                "hz_site_dryrun",
                "hz_site_live",
                "hz_site_mirror_preview",
                "hz_site_mirror_live",
            ],
            "models": "reused",
            "site_skill_allowlist": site_skills,
        }
    )
)
