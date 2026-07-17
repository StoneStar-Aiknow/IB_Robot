"""Strict JSON parsing helpers shared by manifest and metadata loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inference_manifest.errors import ManifestValidationError


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ManifestValidationError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def loads_json_strict(content: str, source: str | Path) -> Any:
    """Parse JSON while rejecting duplicate object keys."""

    try:
        return json.loads(content, object_pairs_hook=_reject_duplicate_keys)
    except ManifestValidationError as exc:
        raise ManifestValidationError(f"Invalid JSON in {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(f"Invalid JSON in {source}: {exc}") from exc


def load_json_strict(path: Path) -> Any:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestValidationError(f"Unable to read JSON file {path}: {exc}") from exc
    return loads_json_strict(content, path)
