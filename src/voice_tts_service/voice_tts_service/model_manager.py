"""ZipVoice bundle validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from voice_tts_service.errors import BackendLoadError

if TYPE_CHECKING:
    from inference_manifest import ValidatedManifest


@dataclass(frozen=True)
class TTSBundle:
    """Validated generic ZipVoice bundle."""

    root: Path
    validated: ValidatedManifest


def load_tts_bundle(bundle_path: str | Path, deployment: str) -> TTSBundle:
    """Validate one explicit named deployment before importing backend SDKs."""

    root = Path(bundle_path).expanduser().resolve()
    if not deployment:
        raise BackendLoadError("voice_tts.deployment must select a named manifest deployment")
    try:
        from inference_manifest import load_inference_manifest

        validated = load_inference_manifest(root, deployment)
    except Exception as exc:
        raise BackendLoadError(f"failed to validate Voice TTS bundle {root}: {exc}") from exc
    if validated.manifest.model.kind != "generic" or validated.manifest.model.family != "zipvoice":
        raise BackendLoadError("selected bundle must declare model.kind=generic and model.family=zipvoice")

    return TTSBundle(root=root, validated=validated)
