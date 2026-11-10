"""Generate the local voice rig's robot config and skill catalog from the summit deployment.

    rig_env.sh python3 make_config.py --models <dir> --state <dir> --out <config.yaml> \
        --catalog-out <profile.yaml> [--keep-english]

The summit config (``aimdk_x2_summit.yaml``) is loaded with its base chain, the local
deviations in ``deviate`` are applied, and the result is written as one standalone file
without ``base_config``. A standalone file follows every later summit change, and unlike an
overlay it can also drop skills, direct commands and whole sections.

Every deviation exists because the PC lacks something the robot has. Keep the list short:
whatever is not changed here runs exactly as on the summit deployment.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from robot_config.loader import _load_robot_section

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
SUMMIT = WORKSPACE / "src/robot_config/config/robots/aimdk_x2_summit.yaml"
SUMMIT_CATALOG = WORKSPACE / "src/agent/skill_catalog/config/profiles/aimdk_x2_summit.yaml"
CATALOG_NAME = "aimdk_x2_summit_local_voice"
ASR_BUNDLE = "voice_asr/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-int8-2026-06-05"

#: Mirror needs the head camera and Ascend YOLOX/PEAR. Contact interaction stays disabled.
UNSUPPORTED_SKILLS = frozenset({"mirror_start", "mirror_stop", "handshake"})


def deviate(config: dict, *, models: Path, state: Path, keep_english: bool) -> dict:
    """Apply the local deviations to the merged summit robot section, in place."""
    # Mirror: no camera stream and no Ascend person models on a PC.
    config["mirror_session"]["enabled"] = False
    config["motion_imitation_stream"]["enabled"] = False
    config["motion_imitation_stream"]["pause_during_manipulation"] = False
    config["motion_imitation_stream"]["pause_during_robot_motion"] = False
    config["visual_interaction"]["enabled"] = False
    # model_inference is the mirror's mode; the rest of the summit uses these two.
    config["default_control_mode"] = "moveit_planning"
    config["skill_required_control_mode"] = "moveit_planning"
    config["skill_supported_control_modes"] = ["moveit_planning", "named_motion"]
    # Grounding DINO, SAM2, GraspGen, YOLOX and PEAR are Ascend deployments; the fake
    # pick/place backends need none of them.
    config["perception_services"]["services"] = []
    # Execute the existing fixed poses through the simulated runtime. Keep hardware's
    # commissioning block in the summit config; it does not apply to this localhost mock.
    config["pose_composer"]["enabled"] = True
    config["pose_composer"]["blocked_task_poses"] = []
    config["pose_composer"]["fixed_library"] = str(state.parent / "fixed_poses.json")
    # Pick and place keep their pipelines, custody and Gateway contracts; only perception
    # and motion are fakes (mock_manipulation.py). The bundle carries identity and mock hand presets, no weights.
    config["grasp_execution"].update(
        backend="mock_manipulation:grasp_backends",
        model_bundle_path=str(HERE / "mock_model_bundle"),
        model_deployment="physical_mock",
    )
    config["placement_execution"]["backend"] = "mock_manipulation:place_backends"
    # Named gestures use the vendor mock directly. Fixed poses keep their articulated-hand checks.
    config["embodied"].get("runtime_named_motion", {}).pop("hand_preparation", None)
    # Same models as the robot, from this PC's model root; ZipVoice in its x86 deployment.
    config["voice_asr"]["bundle_path"] = str(models / ASR_BUNDLE)
    config["voice_asr"]["deployment"] = "torch_cpu"
    config["voice_asr"]["speaker_embedding"]["bundle_path"] = str(models / "campplus-speaker")
    config["voice_tts"]["bundle_path"] = str(models / "zipvoice")
    config["voice_tts"]["deployment"] = "ubuntu_onnx"

    embodied = config["embodied"]
    embodied["skill_catalog_profile"] = CATALOG_NAME
    # Ledgers of this rig only; start.sh empties them, so a quarantine lasts one run.
    embodied["execution_store_path"] = str(state / "executions.sqlite3")
    agent = embodied["agent"]
    agent["ledger_path"] = str(state / "requests.sqlite3")
    agent["conversation_path"] = str(state / "conversation.sqlite3")
    agent["deployment_lock_path"] = str(state / "agent.lock")
    agent["test_allowlist"] = [skill for skill in agent["test_allowlist"] if skill not in UNSUPPORTED_SKILLS]
    commands = agent["planner"].get("direct_commands") or {}
    agent["planner"]["direct_commands"] = {
        phrase: skill for phrase, skill in commands.items() if skill not in UNSUPPORTED_SKILLS
    }
    bridge = agent["voice_bridge"]
    bridge["semantic_gate_model_path"] = str(models / "addressee/addressee_zh_v2.json")
    bridge["semantic_gate_log_path"] = str(state / "semantic_gate.jsonl")
    # English words reach ZipVoice only when its English frontend (piper_phonemize) is installed.
    bridge["tts_keep_english"] = keep_english
    return config


def prepare_fixed_library(state: Path) -> None:
    """Recompile stale derived pose assets against this checkout's current motion profile."""
    from aimdk_grasp.fixed_pose_backend import FixedBackend

    from pose_composer.compile_library import compile_library, configuration_inputs

    inputs = configuration_inputs(SUMMIT)
    output = state.parent / "fixed_poses.json"
    if output.is_file():
        try:
            FixedBackend(inputs["profile"], inputs["limits"], output, task_poses=inputs["task_poses"])
            return
        except ValueError:
            pass  # This artifact is derived; rebuild it when its inputs change.
    selected = list(json.loads(Path(inputs["output"]).read_text())["poses"])
    print("compiling local fixed pose library", flush=True)
    library = compile_library(
        inputs["mjcf"],
        inputs["profile"],
        inputs["limits"],
        inputs["seeds"],
        selected,
        task_poses=inputs["task_poses"],
        hand_path_check=inputs["hand_path_check"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(library, ensure_ascii=False), encoding="utf-8")


def catalog(text: str) -> str:
    """The summit catalog profile without the unsupported skills, under the local name."""
    for skill in sorted(UNSUPPORTED_SKILLS):
        text = re.sub(rf"(?m)^- name: {skill}\n(?:  .*\n)*", "", text)
    text = re.sub(r"(?m)^name: aimdk_x2_summit$", f"name: {CATALOG_NAME}", text, count=1)
    return "# Generated by tools/x2_local_voice/make_config.py; removed by stop.sh.\n" + text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--catalog-out", type=Path, required=True)
    parser.add_argument("--keep-english", action="store_true")
    args = parser.parse_args()
    prepare_fixed_library(args.state.resolve())
    _path, config = _load_robot_section(SUMMIT)
    deviate(
        config,
        models=args.models.resolve(),
        state=args.state.resolve(),
        keep_english=args.keep_english,
    )
    header = "# Generated by tools/x2_local_voice/make_config.py from aimdk_x2_summit.yaml. Do not edit.\n"
    body = yaml.safe_dump({"robot": config}, allow_unicode=True, sort_keys=False)
    args.out.write_text(header + body, encoding="utf-8")
    args.catalog_out.write_text(catalog(SUMMIT_CATALOG.read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    main()
