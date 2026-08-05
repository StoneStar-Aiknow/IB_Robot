"""ZipVoice bundle validation and adapter-factory loading."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from voice_tts_service.errors import BackendLoadError

if TYPE_CHECKING:
    from inference_manifest import ValidatedManifest


@dataclass(frozen=True)
class TTSBundle:
    """Validated generic ZipVoice bundle and its adapter metadata."""

    root: Path
    validated: ValidatedManifest
    adapter: dict[str, Any]


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

    adapter_path = root / "assets" / "adapter.json"
    try:
        adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendLoadError(f"failed to read ZipVoice adapter metadata {adapter_path}: {exc}") from exc
    if not isinstance(adapter, dict):
        raise BackendLoadError("ZipVoice adapter metadata must be a JSON object")
    return TTSBundle(root=root, validated=validated, adapter=adapter)


def deployment_backend(bundle: TTSBundle) -> str:
    """Return the backend declared by the selected manifest deployment."""

    backend = bundle.validated.deployment.backend
    if backend == "torch":
        return "torch"
    if backend == "ascend":
        return "om"
    raise BackendLoadError(f"unsupported Voice TTS deployment backend: {backend!r}")


def load_factory(spec: str, *, field: str) -> Callable[..., Any]:
    """Load a canonical ``module:function`` factory from bundle-owned metadata."""

    if not isinstance(spec, str) or ":" not in spec:
        raise BackendLoadError(f"adapter.json field {field!r} must be a module:function path")
    module_name, function_name = spec.split(":", 1)
    try:
        factory = getattr(importlib.import_module(module_name), function_name)
    except (AttributeError, ImportError) as exc:
        raise BackendLoadError(f"cannot load ZipVoice adapter factory {spec!r}: {exc}") from exc
    if not callable(factory):
        raise BackendLoadError(f"ZipVoice adapter factory {spec!r} is not callable")
    return factory
