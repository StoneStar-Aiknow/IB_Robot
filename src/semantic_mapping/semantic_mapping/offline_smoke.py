"""Open a finalized semantic dataset bag and verify one usable RGB-D frame."""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from .offline_bag import OfflineBagSource, OfflineTopicContract, RosbagReader


def _calibration_status(snapshot: dict) -> str:
    if "status" in snapshot:
        return snapshot["status"]
    ready = snapshot["ready"]
    if not isinstance(ready, bool):
        raise TypeError("ready must be boolean")
    return "complete" if ready else "calibration_incomplete"


def _default_source_factory(
    bag_path: Path,
    topics: OfflineTopicContract,
    global_frame: str,
    sync_slop_ns: int,
) -> OfflineBagSource:
    return OfflineBagSource(
        RosbagReader(str(bag_path), storage_id="mcap"),
        topics,
        global_frame=global_frame,
        sync_slop_ns=sync_slop_ns,
    )


def smoke_dataset(
    dataset_path: Path,
    *,
    topics: OfflineTopicContract,
    global_frame: str = "map",
    sync_slop_ns: int = 50_000_000,
    source_factory: Callable = _default_source_factory,
) -> dict[str, int | str | bool]:
    """Validate topic types, build historical TF, and return the first usable frame."""
    dataset_path = dataset_path.expanduser()
    bag_path = dataset_path / "bag" if (dataset_path / "bag").is_dir() else dataset_path
    snapshot_path = bag_path / "calibration_snapshot.json"
    try:
        calibration_status = _calibration_status(json.loads(snapshot_path.read_text(encoding="utf-8")))
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid calibration snapshot {snapshot_path}: {exc}") from exc
    if calibration_status not in {"complete", "calibration_incomplete"}:
        raise ValueError(f"unsupported calibration snapshot status: {calibration_status}")
    source = source_factory(bag_path, topics, global_frame, sync_slop_ns)
    source.validate_topics()
    tf_buffer = source.build_tf_buffer()
    frame = None
    for candidate in source.frames(tf_buffer, require_camera_tf=calibration_status == "complete"):
        try:
            source.lookup_transform(tf_buffer, global_frame, "base_link", int(candidate.stamp_ns))
        except ValueError:
            continue
        frame = candidate
        break
    if frame is None:
        raise ValueError("finalized bag has no synchronized RGB-D frame with required historical TF")
    camera_frame = frame.rgb.header.frame_id or frame.camera_info.header.frame_id
    if calibration_status == "complete":
        source.lookup_transform(tf_buffer, global_frame, camera_frame, int(frame.stamp_ns))
    if calibration_status == "calibration_incomplete":
        return {
            "bag": str(bag_path),
            "calibration_status": "calibration_incomplete",
            "status": "calibration_incomplete",
            "first_frame_stamp_ns": int(frame.stamp_ns),
            "geometry_ready": False,
        }
    return {
        "bag": str(bag_path),
        "calibration_status": "complete",
        "first_frame_stamp_ns": int(frame.stamp_ns),
        "geometry_ready": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test a finalized RGB-D LiDAR MCAP without semantic models.")
    parser.add_argument("dataset", help="Finalized session root or bag directory")
    parser.add_argument("--rgb-topic", default="/camera/front/image_raw")
    parser.add_argument("--depth-topic", default="/camera/front/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/front/camera_info")
    parser.add_argument("--global-frame", default="map")
    parser.add_argument("--sync-slop-ms", type=float, default=50.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    topics = OfflineTopicContract(args.rgb_topic, args.depth_topic, args.camera_info_topic)
    try:
        result = smoke_dataset(
            Path(args.dataset),
            topics=topics,
            global_frame=args.global_frame,
            sync_slop_ns=int(args.sync_slop_ms * 1_000_000),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Semantic dataset smoke failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["geometry_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
