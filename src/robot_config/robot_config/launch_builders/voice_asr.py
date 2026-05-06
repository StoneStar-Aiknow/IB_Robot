"""Voice ASR launch builder for robot_config."""

from pathlib import Path
from typing import Any

from launch_ros.actions import Node

from robot_config.logger_utils import get_colored_logger
from robot_config.utils import resolve_ros_path
from voice_asr_service.defaults import VOICE_ASR_DEFAULTS
from voice_asr_service.model_manager import (
    STREAMING_ZH_BUNDLE,
    default_model_root,
    infer_model_bundle,
    infer_model_bundle_from_path_hint,
)

logger = get_colored_logger("robot_config.voice_asr")
_VOICE_ASR_REPO_ROOT = Path(__file__).resolve().parents[4]
_VOICE_ASR_REALTIME_MODES = {"continuous", "wake_word"}


def default_voice_asr_model_path(
    model_type: str = "auto",
    active_mode: str = "manual",
    language: str = "zh",
) -> str:
    bundle = infer_model_bundle(
        model_path="",
        model_type=model_type,
        active_mode=active_mode,
        language=language,
    )
    return bundle.resolved_model_path(default_model_root())


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


def _voice_asr_model_dir(model_path: str) -> Path:
    path = Path(model_path)
    return path.parent if path.is_file() else path


def _voice_asr_onnx_files(model_path: str) -> list[Path]:
    path = Path(model_path)
    if path.is_file():
        return [path] if path.suffix == ".onnx" else []
    return sorted(path.glob("*.onnx"))


def _voice_asr_tokens_path(model_path: str, tokens_path: str) -> Path:
    if tokens_path:
        return Path(tokens_path)
    return _voice_asr_model_dir(model_path) / "tokens.txt"


def _voice_asr_has_streaming_artifacts(model_path: str) -> bool:
    model_dir = _voice_asr_model_dir(model_path)
    return all(
        any(model_dir.glob(pattern))
        for pattern in ("encoder*.onnx", "decoder*.onnx", "joiner*.onnx")
    )


def _voice_asr_has_streaming_hint(model_path: str) -> bool:
    hinted_bundle = infer_model_bundle_from_path_hint(model_path)
    return hinted_bundle is not None and hinted_bundle.profile == STREAMING_ZH_BUNDLE.profile


def validate_voice_asr_model_config(
    model_path: str,
    tokens_path: str = "",
    model_type: str = "auto",
    require_streaming: bool = False,
    auto_download_model: bool = True,
) -> list[str]:
    errors: list[str] = []

    if not model_path:
        return ["voice_asr.model_path is required when voice_asr.enabled is true"]

    if require_streaming and model_type == "offline":
        errors.append(
            "Voice ASR realtime streaming requires a streaming-capable model; "
            "model_type=offline is not valid for microphone realtime recognition."
        )

    resolved_model_path = Path(model_path)
    if not resolved_model_path.exists():
        if auto_download_model:
            return errors
        errors.append(f"Voice ASR model path not found: {resolved_model_path}")
        return errors

    onnx_files = _voice_asr_onnx_files(model_path)
    if not onnx_files:
        errors.append(
            f"Voice ASR model path does not contain any .onnx files: {resolved_model_path}"
        )

    resolved_tokens_path = _voice_asr_tokens_path(model_path, tokens_path)
    if not resolved_tokens_path.exists():
        errors.append(f"Voice ASR tokens file not found: {resolved_tokens_path}")

    needs_streaming_layout = require_streaming or model_type == "streaming"
    if needs_streaming_layout and not (
        _voice_asr_has_streaming_artifacts(model_path)
        or _voice_asr_has_streaming_hint(model_path)
    ):
        errors.append(
            "Voice ASR realtime streaming requires either encoder/decoder/joiner ONNX files "
            "or a streaming model path/name hint."
        )

    return errors


def generate_voice_asr_nodes(robot_config: dict[str, Any]) -> list[Node]:
    """Generate voice ASR nodes from robot_config YAML."""
    voice_asr_config = robot_config.get("voice_asr", {})
    if not voice_asr_config.get("enabled", False):
        logger.info("Voice ASR disabled, skipping")
        return []

    model_path = voice_asr_config.get("model_path", "")
    tokens_path = voice_asr_config.get("tokens_path", "")
    auto_download_model = voice_asr_config.get("auto_download_model", VOICE_ASR_DEFAULTS["auto_download_model"])

    if not model_path and auto_download_model:
        model_path = default_voice_asr_model_path(
            model_type=voice_asr_config.get("model_type", "auto"),
            active_mode=voice_asr_config.get("active_mode", "manual"),
            language=voice_asr_config.get("language", "zh"),
        )
        logger.info(f"Voice ASR model_path not set; defaulting to '{model_path}'")

    if not model_path:
        raise ValueError("voice_asr.model_path is required when voice_asr.enabled is true")

    active_mode = voice_asr_config.get("active_mode", "manual")
    resolved_model_path = resolve_voice_asr_path(model_path) if model_path else ""
    resolved_tokens_path = resolve_voice_asr_path(tokens_path) if tokens_path else ""
    validation_errors = validate_voice_asr_model_config(
        model_path=resolved_model_path,
        tokens_path=resolved_tokens_path,
        model_type=voice_asr_config.get("model_type", "auto"),
        require_streaming=voice_asr_mode_requires_streaming(active_mode),
        auto_download_model=auto_download_model,
    )
    if validation_errors:
        raise ValueError("; ".join(validation_errors))

    node_params = {
        "auto_download_model": auto_download_model,
        "active_mode": active_mode,
        "language": voice_asr_config.get("language", VOICE_ASR_DEFAULTS["language"]),
        "model_path": resolved_model_path,
        "tokens_path": resolved_tokens_path,
        "provider": voice_asr_config.get("provider", VOICE_ASR_DEFAULTS["provider"]),
        "model_type": voice_asr_config.get("model_type", VOICE_ASR_DEFAULTS["model_type"]),
        "max_recording_duration": voice_asr_config.get(
            "max_recording_duration", VOICE_ASR_DEFAULTS["max_recording_duration"]
        ),
        "vad_sensitivity": voice_asr_config.get("vad_sensitivity", VOICE_ASR_DEFAULTS["vad_sensitivity"]),
        "realtime_pre_roll_seconds": voice_asr_config.get(
            "realtime_pre_roll_seconds", VOICE_ASR_DEFAULTS["realtime_pre_roll_seconds"]
        ),
        "publish_partial": voice_asr_config.get("publish_partial", VOICE_ASR_DEFAULTS["publish_partial"]),
        "output_topic": voice_asr_config.get("output_topic", VOICE_ASR_DEFAULTS["output_topic"]),
        "sample_rate": voice_asr_config.get("sample_rate", VOICE_ASR_DEFAULTS["sample_rate"]),
        "chunk_size": voice_asr_config.get("chunk_size", VOICE_ASR_DEFAULTS["chunk_size"]),
        "buffer_seconds": voice_asr_config.get("buffer_seconds", VOICE_ASR_DEFAULTS["buffer_seconds"]),
        "device_index": voice_asr_config.get("device_index", VOICE_ASR_DEFAULTS["device_index"]),
        "device_name": voice_asr_config.get("device_name", VOICE_ASR_DEFAULTS["device_name"]),
        "exit_on_init_failure": voice_asr_config.get(
            "exit_on_init_failure", VOICE_ASR_DEFAULTS["exit_on_init_failure"]
        ),
    }

    node_name = voice_asr_config.get("node_name", "voice_asr_node")
    logger.info(f"Voice ASR enabled, launching node '{node_name}'")
    logger.info(f"  output_topic: {node_params['output_topic']}")

    return [
        Node(
            package="voice_asr_service",
            executable="voice_asr_node",
            name=node_name,
            output="screen",
            parameters=[node_params],
        )
    ]
