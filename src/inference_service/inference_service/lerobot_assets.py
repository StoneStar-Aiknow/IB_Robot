"""Resolve bundle-local LeRobot semantic asset references without rewriting metadata."""

from __future__ import annotations

import json
from pathlib import Path

TOKENIZER_REFERENCE_KEYS = frozenset(
    {
        "tokenizer_name",
        "action_tokenizer_name",
        "paligemma_tokenizer_name",
    }
)
VLM_REFERENCE_KEYS = frozenset({"vlm_model_name"})


def resolve_local_semantic_reference(
    bundle_root: str | Path,
    metadata_filename: str,
    reference_keys: frozenset[str],
) -> str | None:
    """Return the absolute in-bundle asset path for one local semantic reference."""

    root = Path(bundle_root).resolve(strict=True)
    metadata_path = root / metadata_filename
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read LeRobot semantic asset metadata {metadata_filename!r}: {exc}") from exc

    references = _find_references(metadata, reference_keys)
    if not references:
        return None
    if len(references) != 1:
        raise ValueError(
            f"LeRobot metadata {metadata_filename!r} declares conflicting semantic asset references: "
            f"{sorted(references)}"
        )

    reference = next(iter(references))
    candidate = Path(reference)
    if not candidate.is_absolute():
        candidate = root / candidate
    if not candidate.exists() and not candidate.is_symlink():
        return None

    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"local semantic asset reference escapes the policy bundle: {reference!r}") from exc
    return str(resolved)


def _find_references(value: object, reference_keys: frozenset[str]) -> set[str]:
    references: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in reference_keys:
                if not isinstance(item, str) or not item:
                    raise ValueError(f"LeRobot semantic asset reference {key!r} must be a non-empty string")
                references.add(item)
            references.update(_find_references(item, reference_keys))
    elif isinstance(value, list):
        for item in value:
            references.update(_find_references(item, reference_keys))
    return references
