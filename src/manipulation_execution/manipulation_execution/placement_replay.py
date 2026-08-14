"""Offline replay and audit helpers for placement verification evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from manipulation_execution.placement_executor_node import evaluate_mask_arrays

CURRENT_PIPELINE = "placement_pipeline"
CURRENT_PIPELINE_VERSION = 3


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_config(manifest: dict[str, Any]) -> dict[str, Any]:
    config = manifest.get("configuration", {})
    if not isinstance(config, dict):
        raise ValueError("placement manifest configuration must be an object")
    return config


def _selected_mask_path(
    root: Path, sample_index: int, kind: str, metadata: dict[str, Any]
) -> tuple[Path, float | None]:
    """Choose the highest-confidence recorded candidate for one sample.

    Runtime verification sorts candidates before writing the evidence files.  The replay
    path still uses the confidence recorded in the sample metadata so it remains correct
    even when evidence was produced by an older writer or manually assembled.
    """
    records = metadata.get(f"{kind}_detections", [])
    if not isinstance(records, list) or not records:
        return root / f"sample_{sample_index:02d}_{kind}_00_mask.npy", None

    def confidence(item: Any) -> float:
        try:
            value = float(item.get("confidence", float("-inf"))) if isinstance(item, dict) else float("-inf")
        except (TypeError, ValueError, OverflowError):
            return float("-inf")
        return value if np.isfinite(value) else float("-inf")

    selected_index, selected_record = max(enumerate(records), key=lambda item: (confidence(item[1]), -item[0]))
    selected_confidence = confidence(selected_record)
    return (
        root / f"sample_{sample_index:02d}_{kind}_{selected_index:02d}_mask.npy",
        None if not np.isfinite(selected_confidence) else selected_confidence,
    )


def audit_placement_evidence(directory: str | Path) -> dict[str, Any]:
    """Audit one placement evidence directory without contacting ROS or hardware."""
    root = Path(directory)
    if root.is_file():
        return {
            "status": "legacy_incompatible",
            "reason": "legacy placement recovery file; current fixed-pose evidence requires a per-request directory",
            "directory": str(root),
        }
    manifest_path = root / "placement_manifest.json"
    result_path = root / "placement_result.json"
    if not manifest_path.exists():
        return {
            "status": "legacy_incompatible",
            "reason": "missing placement_manifest.json; this is not current fixed-pose evidence",
            "directory": str(root),
        }
    manifest = _read_json(manifest_path)
    pipeline = str(manifest.get("pipeline", ""))
    version = int(manifest.get("pipeline_version", 0))
    if pipeline != CURRENT_PIPELINE or version != CURRENT_PIPELINE_VERSION:
        return {
            "status": "legacy_incompatible",
            "reason": f"unsupported evidence pipeline {pipeline!r} v{version}; expected {CURRENT_PIPELINE} v{CURRENT_PIPELINE_VERSION}",
            "directory": str(root),
        }

    config = _load_config(manifest)
    verification = config.get("verification", {})
    if not isinstance(verification, dict):
        raise ValueError("placement manifest verification configuration must be an object")
    required = int(verification.get("required_confirmations", 2))
    minimum_target_pixels = int(verification.get("min_target_mask_pixels", 100))
    minimum_inside_fraction = float(verification.get("min_inside_mask_fraction", 0.70))
    inset_ratio = float(verification.get("container_inset_ratio", 0.05))
    samples: list[dict[str, Any]] = []
    for metadata_path in sorted(root.glob("sample_*.json")):
        if metadata_path.name == "placement_manifest.json" or metadata_path.name == "placement_result.json":
            continue
        metadata = _read_json(metadata_path)
        index = int(metadata.get("sample_index", 0))
        container_mask_path, container_confidence = _selected_mask_path(root, index, "container", metadata)
        target_mask_path, target_confidence = _selected_mask_path(root, index, "target", metadata)
        if not container_mask_path.exists() or not target_mask_path.exists():
            samples.append({"sample_index": index, "outcome": None, "detail": "detection masks unavailable"})
            continue
        try:
            containment = evaluate_mask_arrays(
                np.load(container_mask_path, allow_pickle=False),
                np.load(target_mask_path, allow_pickle=False),
                min_target_pixels=minimum_target_pixels,
                min_inside_fraction=minimum_inside_fraction,
                container_inset_ratio=inset_ratio,
            )
            outcome = bool(containment.inside)
            detail = (
                f"target_pixels={containment.target_pixel_count} inside_pixels={containment.inside_pixel_count} "
                f"inside_fraction={containment.inside_fraction:.3f} center_inside={containment.center_inside}"
            )
            if container_confidence is not None or target_confidence is not None:
                detail += (
                    f" selected_container_confidence={container_confidence if container_confidence is not None else 'unknown'}"
                    f" selected_target_confidence={target_confidence if target_confidence is not None else 'unknown'}"
                )
        except (OSError, ValueError) as exc:
            outcome = None
            detail = str(exc)
        samples.append({"sample_index": index, "outcome": outcome, "detail": detail})

    previous: bool | None = None
    consecutive = 0
    vision_verified = False
    for sample in samples:
        outcome = sample["outcome"]
        if outcome is None:
            previous = None
            consecutive = 0
        elif outcome == previous:
            consecutive += 1
        else:
            previous = outcome
            consecutive = 1
        if outcome is True and consecutive >= required:
            vision_verified = True
            break

    feedback_path = root / "open_gripper_joint_state.json"
    result = _read_json(result_path) if result_path.exists() else {}
    open_feedback_present = feedback_path.exists()
    gripper = manifest.get("gripper", {})
    open_feedback_verified = False
    if open_feedback_present and isinstance(gripper, dict):
        try:
            feedback = _read_json(feedback_path)
            joint_name = str(gripper["joint_name"])
            names = [str(name) for name in feedback.get("name", [])]
            positions = [float(position) for position in feedback.get("position", [])]
            index = names.index(joint_name)
            open_feedback_verified = abs(positions[index] - float(gripper["open_position"])) <= float(
                gripper["position_tolerance"]
            )
        except (IndexError, KeyError, TypeError, ValueError):
            open_feedback_verified = False
    verified = vision_verified and open_feedback_verified
    return {
        "status": "verified" if verified else "not_verified",
        "pipeline": pipeline,
        "pipeline_version": version,
        "directory": str(root),
        "task_id": str(manifest.get("task_id", "")),
        "target_query": str(manifest.get("target_query", "")),
        "container_query": str(manifest.get("container_query", "")),
        "open_feedback_present": open_feedback_present,
        "open_feedback_verified": open_feedback_verified,
        "vision_verified": vision_verified,
        "samples": samples,
        "required_confirmations": required,
        "recorded_result": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay fixed-pose placement evidence offline")
    parser.add_argument("evidence_dir", type=Path)
    args = parser.parse_args()
    report = audit_placement_evidence(args.evidence_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
