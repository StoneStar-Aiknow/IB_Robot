"""Voice ASR launch builder for robot_config."""

import os
from pathlib import Path
from typing import Any

from launch_ros.actions import Node

from robot_config.audio_contract import find_microphone_params, is_audio_io_enabled
from robot_config.logger_utils import get_colored_logger
from robot_config.utils import resolve_ros_path

logger = get_colored_logger("robot_config.voice_asr")
_VOICE_ASR_REPO_ROOT = Path(__file__).resolve().parents[4]
_VOICE_ASR_REALTIME_MODES = {"continuous", "wake_word"}
_VOICE_ASR_MISSING_ERROR = "voice_asr.enabled=true requires the voice_asr_service package to be installed"


def _load_voice_asr_service():
    try:
        from voice_asr_service.defaults import VOICE_ASR_DEFAULTS
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("voice_asr_service"):
            raise ModuleNotFoundError(_VOICE_ASR_MISSING_ERROR) from exc
        raise

    return VOICE_ASR_DEFAULTS


def resolve_voice_asr_path(path: str) -> str:
    """Resolve Voice ASR paths relative to the workspace root when needed."""
    resolved = resolve_ros_path(path)
    if not resolved:
        return resolved

    resolved_path = Path(resolved).expanduser()
    if resolved_path.is_absolute():
        return str(resolved_path)
    return str((_VOICE_ASR_REPO_ROOT / resolved_path).resolve())


def voice_asr_mode_requires_streaming(active_mode: str) -> bool:
    return active_mode in _VOICE_ASR_REALTIME_MODES


def _validate_audio_contract(validated, *, sample_rate: int, chunk_size: int, require_frame: bool) -> list[str]:
    """Check a deployment's declared audio contract against the node's audio input."""
    contract = getattr(validated.deployment, "audio_contract", None)
    if contract is None:
        return []
    errors: list[str] = []
    if contract.sample_rate_hz is not None and contract.sample_rate_hz != sample_rate:
        errors.append(
            f"audio contract sample_rate_hz={contract.sample_rate_hz} does not match configured sample_rate={sample_rate}"
        )
    if contract.channels is not None and contract.channels != 1:
        errors.append(f"audio contract requires channels={contract.channels}, expected mono input")
    if contract.sample_dtype is not None and contract.sample_dtype != "float32":
        errors.append(f"audio contract requires sample_dtype={contract.sample_dtype}, expected float32")
    if require_frame:
        if contract.frame_size is not None and contract.frame_size != chunk_size:
            errors.append(
                f"audio contract frame_size={contract.frame_size} does not match configured chunk_size={chunk_size}"
            )
        if contract.execution_mode is not None and contract.execution_mode not in {"streaming", "both"}:
            errors.append(f"audio contract execution_mode={contract.execution_mode} is not streaming-capable")
    return errors


def validate_voice_asr_model_config(
    bundle_path: str,
    deployment: str,
    require_streaming: bool = False,
    sample_rate: int = 16000,
    chunk_size: int = 512,
    vad_bundle_path: str = "",
    vad_deployment: str = "",
) -> list[str]:
    if not bundle_path:
        return ["voice_asr.bundle_path is required when voice_asr.enabled is true"]
    if not deployment:
        return ["voice_asr.deployment is required when voice_asr.enabled is true"]
    if not Path(bundle_path).is_dir():
        return [f"Voice ASR bundle path not found: {bundle_path}"]
    try:
        from inference_manifest import load_inference_manifest

        validated = load_inference_manifest(bundle_path, deployment)
    except Exception as exc:
        return [f"Voice ASR bundle/deployment is invalid: {exc}"]
    if require_streaming and not {"encoder", "decoder"}.issubset(validated.resolved_artifacts):
        return ["Voice ASR realtime streaming requires encoder and decoder deployment artifacts"]
    errors = _validate_audio_contract(validated, sample_rate=sample_rate, chunk_size=chunk_size, require_frame=False)
    if vad_bundle_path and vad_deployment:
        try:
            vad_validated = load_inference_manifest(vad_bundle_path, vad_deployment)
        except Exception as exc:
            return errors + [f"Voice ASR VAD bundle/deployment is invalid: {exc}"]
        errors += _validate_audio_contract(
            vad_validated, sample_rate=sample_rate, chunk_size=chunk_size, require_frame=True
        )
    return errors


def _ascend_node_environment(bundle_path: str, deployment: str) -> dict[str, str]:
    """CANN runtime paths for deployments with an Ascend role (same contract as speech_direction).

    The node must not depend on whether the caller sourced set_env.sh; non-Ascend deployments get
    no extra environment.
    """
    from inference_manifest import load_inference_manifest

    try:
        validated = load_inference_manifest(bundle_path, deployment)
    except Exception:  # validate_voice_asr_model_config already reports an unusable bundle
        return {}
    profiles = validated.deployment.role_runtime_profiles or {}
    if not any(profile.backend == "ascend" for profile in profiles.values()):
        return {}
    cann_root = Path(os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest"))

    def prepend(name: str, entries: tuple[Path, ...]) -> str:
        values = [str(path) for path in entries if path.exists()]
        if os.environ.get(name):
            values.append(os.environ[name])
        return os.pathsep.join(values)

    return {
        "ASCEND_HOME_PATH": str(cann_root),
        "PYTHONPATH": prepend("PYTHONPATH", (cann_root / "python" / "site-packages",)),
        "LD_LIBRARY_PATH": prepend("LD_LIBRARY_PATH", (cann_root / "lib64", cann_root / "runtime" / "lib64")),
        "PATH": prepend("PATH", (cann_root / "bin",)),
    }


def generate_voice_asr_nodes(robot_config: dict[str, Any]) -> list[Node]:
    """Generate voice ASR nodes from robot_config YAML."""
    voice_asr_config = robot_config.get("voice_asr", {})
    if not voice_asr_config.get("enabled", False):
        logger.info("Voice ASR disabled, skipping")
        return []

    stale_fields = {"model_path", "tokens_path", "provider", "model_type", "auto_download_model"}
    configured_stale_fields = sorted(stale_fields.intersection(voice_asr_config))
    if configured_stale_fields:
        raise ValueError(
            "voice_asr uses deprecated raw model fields; configure bundle_path/deployment instead: "
            + ", ".join(configured_stale_fields)
        )

    voice_asr_defaults = _load_voice_asr_service()

    active_mode = voice_asr_config.get("active_mode", "manual")
    input_mode = str(voice_asr_config.get("input_mode", voice_asr_defaults["input_mode"]))
    if input_mode not in ("shared_audio", "external_vad"):
        raise ValueError(f"voice_asr.input_mode must be 'shared_audio' or 'external_vad', got {input_mode!r}")
    external_mode = input_mode == "external_vad"
    bundle_path = resolve_voice_asr_path(voice_asr_config.get("bundle_path", voice_asr_defaults["bundle_path"]))
    deployment = str(voice_asr_config.get("deployment", voice_asr_defaults["deployment"]))
    if external_mode:
        # 外部分段输入：VAD 由音源负责，节点不装配本地 Silero，无需 VAD bundle。
        vad_bundle_path = ""
        vad_deployment = ""
    else:
        vad_bundle_path = resolve_voice_asr_path(
            voice_asr_config.get("vad_bundle_path", voice_asr_defaults["vad_bundle_path"])
        )
        vad_deployment = str(voice_asr_config.get("vad_deployment", voice_asr_defaults["vad_deployment"]))
    validation_errors = validate_voice_asr_model_config(
        bundle_path=bundle_path,
        deployment=deployment,
        require_streaming=voice_asr_mode_requires_streaming(active_mode) or external_mode,
        sample_rate=int(voice_asr_config.get("sample_rate", voice_asr_defaults["sample_rate"])),
        chunk_size=int(voice_asr_config.get("chunk_size", voice_asr_defaults["chunk_size"])),
        vad_bundle_path=vad_bundle_path,
        vad_deployment=vad_deployment,
    )
    if validation_errors:
        raise ValueError("; ".join(validation_errors))

    node_params = {
        "active_mode": active_mode,
        "input_mode": input_mode,
        "language": voice_asr_config.get("language", voice_asr_defaults["language"]),
        "bundle_path": bundle_path,
        "deployment": deployment,
        "max_recording_duration": voice_asr_config.get(
            "max_recording_duration", voice_asr_defaults["max_recording_duration"]
        ),
        "exit_on_init_failure": voice_asr_config.get(
            "exit_on_init_failure", voice_asr_defaults["exit_on_init_failure"]
        ),
        "final_tail_padding_seconds": voice_asr_config.get(
            "final_tail_padding_seconds", voice_asr_defaults["final_tail_padding_seconds"]
        ),
        "publish_partial": voice_asr_config.get("publish_partial", voice_asr_defaults["publish_partial"]),
        "output_topic": voice_asr_config.get("output_topic", voice_asr_defaults["output_topic"]),
        "sample_rate": voice_asr_config.get("sample_rate", voice_asr_defaults["sample_rate"]),
    }
    audio_io = robot_config.get("audio_io", {})
    if not is_audio_io_enabled(audio_io):
        raise ValueError("voice_asr.enabled=true requires audio_io.enabled=true")

    if external_mode:
        # SpeechAudioChunk carries the mono S16LE PCM and its boundary in one
        # message, so the external path has exactly one input topic. A
        # runtime-managed deployment must bind it before constructing the
        # consumer; there is no audio_topic in this mode to fall back to.
        audio_input = voice_asr_config.get("audio_input")
        if audio_input is not None:
            if not isinstance(audio_input, dict) or not audio_input.get("_interface_source"):
                raise ValueError("voice_asr.audio_input must be bound to a runtime description before launch")
            if audio_input.get("type") != "ibrobot_msgs/msg/SpeechAudioChunk":
                raise ValueError("voice_asr.audio_input requires SpeechAudioChunk")
            explicit = voice_asr_config.get("chunk_topic")
            if explicit and explicit != audio_input["topic"]:
                raise ValueError("voice_asr.chunk_topic conflicts with the bound runtime endpoint")
            node_params["chunk_topic"] = audio_input["topic"]
        else:
            node_params["chunk_topic"] = str(voice_asr_config.get("chunk_topic", voice_asr_defaults["chunk_topic"]))
        node_params["external_min_samples"] = int(
            voice_asr_config.get("external_min_samples", voice_asr_defaults["external_min_samples"])
        )
        node_params["external_idle_timeout"] = float(
            voice_asr_config.get("external_idle_timeout", voice_asr_defaults["external_idle_timeout"])
        )
    else:
        node_params.update(
            {
                "vad_sensitivity": voice_asr_config.get("vad_sensitivity", voice_asr_defaults["vad_sensitivity"]),
                "vad_bundle_path": resolve_voice_asr_path(
                    voice_asr_config.get("vad_bundle_path", voice_asr_defaults["vad_bundle_path"])
                ),
                "vad_deployment": voice_asr_config.get("vad_deployment", voice_asr_defaults["vad_deployment"]),
                "realtime_pre_roll_seconds": voice_asr_config.get(
                    "realtime_pre_roll_seconds", voice_asr_defaults["realtime_pre_roll_seconds"]
                ),
                "chunk_size": voice_asr_config.get("chunk_size", voice_asr_defaults["chunk_size"]),
                "buffer_seconds": voice_asr_config.get("buffer_seconds", voice_asr_defaults["buffer_seconds"]),
            }
        )
        microphone_name = str(audio_io.get("microphone", ""))
        microphone_params = find_microphone_params(robot_config.get("peripherals", []), microphone_name)
        node_params["audio_topic"] = str(audio_io.get("capture_stamped_topic", "/audio/capture_stamped"))
        node_params["audio_channels"] = int(microphone_params.get("channels", 1))
        node_params["audio_input_channel"] = int(
            voice_asr_config.get("audio_input_channel", voice_asr_defaults.get("audio_input_channel", 1))
        )
        # VAD 独立输入通道（如 ReSpeaker ch0 板载处理通道）；未配置时与 ASR 通道一致。
        vad_input_channel = voice_asr_config.get("vad_input_channel")
        node_params["vad_input_channel"] = -1 if vad_input_channel is None else int(vad_input_channel)

    node_name = voice_asr_config.get("node_name", "voice_asr_node")
    logger.info(f"Voice ASR enabled, launching node '{node_name}' (input_mode={input_mode})")
    logger.info(f"  output_topic: {node_params['output_topic']}")
    logger.info(f"  input: {node_params['chunk_topic'] if external_mode else node_params['audio_topic']}")

    return [
        Node(
            package="voice_asr_service",
            executable="voice_asr_node",
            name=node_name,
            output="screen",
            parameters=[node_params],
            additional_env=_ascend_node_environment(bundle_path, deployment),
        )
    ]
