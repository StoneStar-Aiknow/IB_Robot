"""Validate speech-to-apple-grasp deployment and its single-skill catalog."""

import json
from pathlib import Path

import aimdk_grasp
import aimdk_robot
from aimdk_robot.trajectory import load_trajectory

import inference_service
import manipulation_execution
import robot_config
import robot_runtime
from manipulation_execution.launch_builders.mirror_session import generate_mirror_session_nodes
from manipulation_execution.launch_builders.motion_imitation_stream import generate_motion_imitation_stream_nodes
from robot_config.interface_binding import bind_robot_interfaces
from robot_config.loader import load_robot_config_dict, validate_robot_config_dict
from robot_runtime.interface_description import build_description
from robot_runtime.profile import load_profile
from robot_skill_cli.catalog import compile_local_snapshot

root = Path(__file__).resolve().parents[1]
for module in (aimdk_grasp, aimdk_robot, inference_service, manipulation_execution, robot_config, robot_runtime):
    path = Path(module.__file__).resolve()
    if root not in path.parents:
        raise RuntimeError(f"wrong checkout imported: {module.__name__}: {path}")
    print(f"IMPORT {module.__name__} {path}", flush=True)
profile = load_profile(str(root / "src/robots/aimdk/aimdk_robot/profiles/x2_ultra_omnihand.yaml"))
results = []
for name in (
    "hz_voice_apple_dryrun",
    "hz_voice_apple_live",
    "hz_voice_apple_smoke",
    "hz_site_dryrun",
    "hz_site_live",
    "hz_site_mirror_preview",
    "hz_site_mirror_live",
):
    path = root / "commissioning/robots" / f"{name}.yaml"
    config = load_robot_config_dict(path, defer_interface_binding=True)
    config = bind_robot_interfaces(config, build_description(profile, use_sim=name.endswith("smoke")))
    errors = validate_robot_config_dict(config)
    if errors:
        raise RuntimeError(f"invalid {name}: {errors}")
    if name.startswith("hz_site_"):
        mirror_enabled = name.startswith("hz_site_mirror_")
        for section in ("mirror_session", "motion_imitation_stream", "visual_interaction"):
            expected_enabled = mirror_enabled and section != "visual_interaction"
            if bool(config.get(section, {}).get("enabled")) != expected_enabled:
                raise RuntimeError(f"unexpected {section} enablement")
        if mirror_enabled:
            expected_live = name == "hz_site_mirror_live"
            if (
                config["mirror_session"]["publish_enabled"] != expected_live
                or config["embodied"]["agent"]["execution_enabled"] != expected_live
            ):
                raise RuntimeError("mirror execution and publication must match the selected preview/live profile")
            if config["control_modes"]["model_inference"]["runtime_mode"] != "policy_stream":
                raise RuntimeError("mirror must share policy_stream")
            generate_mirror_session_nodes(config)
            generate_motion_imitation_stream_nodes(config)
        if config["embodied"]["agent"]["voice_bridge"]["confirm_mode"] != "voice_confirm":
            raise RuntimeError("site profile must require voice confirmation")
        if config["voice_asr"]["input_mode"] != "external_vad":
            raise RuntimeError("site ASR must match the requested reference recipe")
        for mode in ("moveit_planning", "named_motion"):
            if config["control_modes"][mode]["runtime_mode"] != "policy_stream":
                raise RuntimeError(f"site {mode} must share the policy_stream runtime")
        replay = profile["vendor"]["named_motion"]
        if "policy_stream" not in replay["trajectory_modes"]:
            raise RuntimeError("runtime profile does not allow policy_stream gesture replay")
        for motion in ("wave", "handshake", "raise_hand", "blow_kiss", "clap"):
            load_trajectory(
                root / "src/robots/aimdk/aimdk_robot/profiles" / replay["trajectories"][motion],
                max_speed_radps=float(replay["trajectory_max_speed_radps"]),
            )
    catalog = compile_local_snapshot(config, path)
    skills = sorted(catalog.enabled_skill_names)
    expected = (
        sorted(["wave_hand", "handshake", "raise_hand", "blow_kiss", "clap_hands", "self_introduction", "pick_object"])
        if name.startswith("hz_site_")
        else ["pick_object"]
    )
    if name.startswith("hz_site_mirror_"):
        expected = sorted([*expected, "mirror_start", "mirror_stop"])
    if skills != expected:
        raise RuntimeError(f"unexpected skill catalog: {skills}")
    if name.startswith("hz_site_") and "pick_object" not in catalog.planner_visible_skill_names:
        raise RuntimeError("site picking must remain planner-visible after gestures")
    results.append(
        {
            "config": name,
            "skills": skills,
            "registry_digest": catalog.registry_digest,
            "model_services": len(config["perception_services"]["services"]),
            "asr_input_mode": config["voice_asr"]["input_mode"],
            "tts_enabled": config["voice_tts"]["enabled"],
            "execution_enabled": config["embodied"]["agent"]["execution_enabled"],
        }
    )
    print(json.dumps(results[-1]), flush=True)
(root / "commissioning/logs/config_preflight.json").write_text(json.dumps(results, indent=2))
print("RESULT=PASS")
