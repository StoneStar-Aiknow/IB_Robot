"""Shared work-path conventions for model exporters."""

from __future__ import annotations

from pathlib import Path


def export_work_dir(bundle_root: str | Path, exporter: str, override: str | Path | None = None) -> Path:
    """Return an existing exporter work directory inside the policy bundle by default."""

    root = Path(bundle_root).expanduser().resolve(strict=True)
    work_dir = Path(override).expanduser() if override is not None else root / "model_utils_work" / exporter
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def ensure_output_parent(path: str | Path) -> Path:
    """Create the parent directory for an explicit or derived exporter output."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
