"""Analyze a recorded mHandPro calibration sweep without accessing hardware."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .calibrate_glove import fit_captured_task_space
from .devices.aero_compact_retarget import build_aero_compact_calibration
from .devices.aero_hand_retarget import extract_hand_shape_metrics
from .devices.glove_calibration import load_calibration, load_raw_capture, write_calibration_atomic
from .devices.mocap_retarget import FEATURE_SCHEMA_AERO_COMPACT, build_sdk_skeleton_sweep_calibration


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze a replayable mHandPro raw capture")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--side", choices=("left", "right"))
    parser.add_argument(
        "--update-calibration",
        type=Path,
        help="Atomically replace reusable channel/task-space ranges in an existing calibration JSON",
    )
    return parser.parse_args(argv)


def analyze_capture(path: str | Path, side: str | None = None) -> dict:
    phases = load_raw_capture(path, side)
    captured_side = side or json.loads(Path(path).expanduser().read_text(encoding="utf-8"))["side"]
    sweep_positions = [frame.positions for frame in phases["sweep"]]
    metrics = [extract_hand_shape_metrics(positions, captured_side) for positions in sweep_positions]
    metric_rows = [
        [item.thumb_adduction, item.thumb_opposition, item.thumb_curve, *item.finger_curves] for item in metrics
    ]
    spans = [max(row[index] for row in metric_rows) - min(row[index] for row in metric_rows) for index in range(7)]
    quaternion_frames = sum(frame.quaternions is not None for frames in phases.values() for frame in frames)
    result = {
        "side": captured_side,
        "frame_counts": {phase: len(frames) for phase, frames in phases.items()},
        "quaternion_frames": quaternion_frames,
        "metric_spans": {
            "thumb_adduction_rad": spans[0],
            "thumb_opposition_rad": spans[1],
            "thumb_curve": spans[2],
            "finger_curves": spans[3:],
        },
    }
    try:
        result["fitted_task_space"] = fit_captured_task_space(phases["open"], phases["sweep"], captured_side)
    except ValueError as exc:
        result["task_space_error"] = str(exc)
    if all(frame.virtual_positions is not None for frames in phases.values() for frame in frames):
        result["fitted_sdk_skeleton"] = build_sdk_skeleton_sweep_calibration(
            phases["open"],
            phases["sweep"],
            captured_side,
        )
        result["fitted_aero_compact"] = build_aero_compact_calibration(
            phases["open"],
            phases["sweep"],
            captured_side,
        )
    return result


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        result = analyze_capture(args.input, args.side)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Capture analysis failed: {exc}", file=sys.stderr)
        return 1
    if not all(math.isfinite(value) for value in result["metric_spans"]["finger_curves"]):
        print("Capture analysis failed: non-finite metric span", file=sys.stderr)
        return 1
    if args.update_calibration is not None:
        try:
            load_calibration(args.update_calibration, result["side"], require_persistence=False)
            document = json.loads(args.update_calibration.expanduser().read_text(encoding="utf-8"))
            if "fitted_task_space" in result:
                document["task_space"] = result["fitted_task_space"]
            if "fitted_aero_compact" in result:
                document.update(result["fitted_aero_compact"])
                document["feature_schema"] = FEATURE_SCHEMA_AERO_COMPACT
                document.pop("thumb_neutral", None)
            elif "fitted_sdk_skeleton" in result:
                document.update(result["fitted_sdk_skeleton"])
            if "fitted_task_space" not in result and "fitted_aero_compact" not in result:
                raise ValueError("Capture did not produce reusable calibration ranges")
            written = write_calibration_atomic(args.update_calibration, document)
            result["updated_calibration"] = str(written)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"Calibration update failed: {exc}", file=sys.stderr)
            return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
