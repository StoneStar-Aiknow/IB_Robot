"""Shared work-path conventions for model exporters."""

from __future__ import annotations

from pathlib import Path


def _work_layout(bundle_root: Path) -> tuple[Path, Path]:
    """Return the ``_work`` root and the bundle path below it.

    Bundles live under a ``models`` directory, so intermediates default to
    ``models/_work/<bundle>/...`` and never land inside a releasable bundle.
    Bundles outside a ``models`` tree fall back to ``<bundle>/../_work/<name>/...``.
    """

    for models_root in bundle_root.parents:
        if models_root.name == "models":
            relative = bundle_root.relative_to(models_root)
            if relative.parts[0] == "_work":
                return bundle_root.parent / "_work", Path(bundle_root.name)
            return models_root / "_work", relative
    return bundle_root.parent / "_work", Path(bundle_root.name)


def export_work_dir(bundle_root: str | Path, exporter: str, override: str | Path | None = None) -> Path:
    """Return an exporter work directory outside the policy bundle by default.

    Defaults to ``models/_work/<bundle>/<exporter>``. An explicit ``override``
    wins; exporters must never write intermediates into the bundle itself.
    """

    root = Path(bundle_root).expanduser().resolve(strict=True)
    if override is not None:
        work_dir = Path(override).expanduser()
    else:
        work_root, bundle_rel = _work_layout(root)
        work_dir = work_root / bundle_rel / exporter
    work_dir = resolve_outside_bundle_path(root, work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def resolve_outside_bundle_path(bundle_root: str | Path, path: str | Path) -> Path:
    """Resolve an intermediate path and reject bundle-local destinations."""

    root = Path(bundle_root).expanduser().resolve(strict=True)
    resolved = Path(path).expanduser().resolve()
    if resolved == root or resolved.is_relative_to(root):
        raise ValueError(f"conversion intermediate path must be outside the policy bundle {root}: {resolved}")
    return resolved


def ensure_output_parent(path: str | Path) -> Path:
    """Create the parent directory for an explicit or derived exporter output."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
