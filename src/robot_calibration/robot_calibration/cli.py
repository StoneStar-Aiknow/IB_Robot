"""File-only calibration maintenance commands."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from robot_calibration.capture import CaptureError, import_legacy_capture
from robot_calibration.detector import run_detector
from robot_calibration.export import export_capture
from robot_calibration.offline import REQUIRED_SCENES, create_candidate_artifact, solve_joint_calibration
from robot_calibration.store import ArtifactStore, StoreError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    legacy = commands.add_parser("legacy-import")
    legacy.add_argument("source", type=Path)
    legacy.add_argument("--output", type=Path, required=True)
    legacy.add_argument("--lidar-serial", required=True)
    legacy.add_argument("--camera-serial", required=True)
    export = commands.add_parser("capture-export")
    export.add_argument("capture", type=Path)
    export.add_argument("--archive", type=Path, required=True)
    capture_import = commands.add_parser("capture-import")
    capture_import.add_argument("archive", type=Path)
    capture_import.add_argument("--output", type=Path, required=True)
    export_data = commands.add_parser("export")
    export_data.add_argument("capture", type=Path)
    export_data.add_argument("--output", type=Path, required=True)
    detector = commands.add_parser("detect")
    detector.add_argument("--workspace", type=Path, required=True)
    detector.add_argument("--templates", type=Path, required=True)
    detector.add_argument("--exported", type=Path, required=True)
    detector.add_argument("--output", type=Path, required=True)
    solve = commands.add_parser("solve")
    for scene in REQUIRED_SCENES:
        solve.add_argument(f"--{scene}", type=Path, required=True)
    solve.add_argument("--output", type=Path, required=True)
    solve.add_argument("--report", type=Path, required=True)
    solve.add_argument("--max-training-rmse-m", type=float, default=0.04)
    solve.add_argument("--max-test-rmse-m", type=float, default=0.04)
    solve.add_argument("--max-baseline-m", type=float, default=0.5)
    solve.add_argument("--min-correspondence-margin-m", type=float, default=0.05)
    artifact = commands.add_parser("candidate")
    artifact.add_argument("--result", type=Path, required=True)
    artifact.add_argument("--report", type=Path, required=True)
    artifact.add_argument("--mount", type=Path, required=True)
    artifact.add_argument("--capture-manifest", type=Path, required=True)
    artifact.add_argument("--camera-serial", required=True)
    artifact.add_argument("--producer-commit", required=True)
    artifact.add_argument("--parameters-sha256", required=True)
    artifact.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute one offline calibration maintenance operation."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "legacy-import":
            result = {
                "path": str(
                    import_legacy_capture(
                        args.source,
                        args.output,
                        devices={"lidar": args.lidar_serial, "camera": args.camera_serial},
                    )
                )
            }
        elif args.command == "capture-export":
            result = {
                "path": str(args.archive.absolute()),
                "sha256": ArtifactStore.export_capture(args.capture, args.archive),
            }
        elif args.command == "capture-import":
            result = ArtifactStore.import_capture(args.archive, args.output)
        elif args.command == "export":
            result = export_capture(args.capture, args.output)
        elif args.command == "detect":
            result = run_detector(args.workspace, args.templates, args.exported, args.output)
        elif args.command == "solve":
            _, result = solve_joint_calibration(
                observations={scene: getattr(args, scene.replace("-", "_")) for scene in REQUIRED_SCENES},
                output=args.output,
                report=args.report,
                max_training_rmse_m=args.max_training_rmse_m,
                max_test_rmse_m=args.max_test_rmse_m,
                max_baseline_m=args.max_baseline_m,
                min_correspondence_margin_m=args.min_correspondence_margin_m,
            )
        else:
            result = create_candidate_artifact(
                result=args.result,
                report=args.report,
                mount=args.mount,
                capture_manifest=args.capture_manifest,
                camera_serial=args.camera_serial,
                producer_commit=args.producer_commit,
                parameters_sha256=args.parameters_sha256,
                output=args.output,
            )
    except (
        CaptureError,
        StoreError,
        ImportError,
        OSError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
