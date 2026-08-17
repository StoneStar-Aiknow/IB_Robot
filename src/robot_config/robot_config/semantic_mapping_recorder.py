"""Supervise one rosbag2 process for an RGB-D LiDAR mapping session."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supervise a semantic dataset rosbag2 recording.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--robot-config", required=True)
    parser.add_argument("--session-root", required=True)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--mount-file", required=True)
    parser.add_argument("--camera-file", default="~/.ros/ibrobot/calib/current/base_to_front_camera.yaml")
    parser.add_argument("--camera-info-topic", required=True)
    parser.add_argument("--topic", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def recorder_command_from_args(args: argparse.Namespace) -> list[str]:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("rosbag command is required after --")
    return command


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pin_calibration_sources(session_root: Path, mount_file: Path, camera_file: Path) -> dict[str, dict[str, str]]:
    """Copy calibration inputs before rosbag starts; finalization never rereads sources."""
    target_root = session_root / "calibration_sources"
    target_root.mkdir(parents=True, exist_ok=True)
    sources: dict[str, dict[str, str]] = {}
    for name, source, required in (
        ("base_to_mid360", mount_file.expanduser(), True),
        ("base_to_front_camera", camera_file.expanduser(), False),
    ):
        if not source.is_file():
            if required:
                raise ValueError(f"required calibration source is missing: {source}")
            sources[name] = {"status": "missing", "source": str(source)}
            continue
        data = source.read_bytes()
        snapshot = target_root / f"{name}.yaml"
        snapshot.write_bytes(data)
        try:
            document = yaml.safe_load(data) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid calibration source {source}: {exc}") from exc
        if not isinstance(document, dict):
            raise ValueError(f"invalid calibration source {source}: expected a mapping")
        status = str(document.get("status", "unknown"))
        if name == "base_to_front_camera" and status != "approved":
            raise ValueError(f"camera calibration source is not approved: {source}")
        sources[name] = {
            "status": status,
            "source": str(source.resolve()),
            "snapshot": str(snapshot.resolve()),
            "sha256": _sha256_bytes(data),
        }
    return sources


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        command = recorder_command_from_args(args)
    except ValueError as exc:
        print(f"Semantic dataset recorder failed: {exc}", file=sys.stderr)
        return 2

    session_root = Path(args.session_root).expanduser()
    bag_path = session_root / "bag"
    map_path = session_root / "map"
    state_path = Path(args.state_file).expanduser()
    try:
        session_root.mkdir(parents=True, exist_ok=False)
        map_path.mkdir()
        calibration_sources = pin_calibration_sources(
            session_root,
            Path(args.mount_file),
            Path(args.camera_file),
        )
    except (OSError, ValueError) as exc:
        print(f"Semantic dataset recorder failed: {exc}", file=sys.stderr)
        return 2
    state = {
        "schema_version": "1.0",
        "session_id": args.session_id,
        "profile": args.profile,
        "robot_config": str(Path(args.robot_config).expanduser().resolve()),
        "session_root": str(session_root.resolve()),
        "bag_path": str(bag_path.resolve()),
        "map_prefix": str((map_path / "map").resolve()),
        "calibration_sources": calibration_sources,
        "camera_info_topic": args.camera_info_topic,
        "topics": list(args.topic),
        "supervisor_pid": os.getpid(),
        "recorder_pid": 0,
        "status": "starting",
        "started_at": _timestamp(),
    }
    _write_state(state_path, state)

    child = None
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        state["status"] = "stopping"
        state["stop_requested_at"] = _timestamp()
        _write_state(state_path, state)
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        child = subprocess.Popen(command, start_new_session=True)
        state["recorder_pid"] = child.pid
        state["status"] = "recording"
        _write_state(state_path, state)
        if stop_requested and child.poll() is None:
            child.send_signal(signal.SIGINT)
        returncode = child.wait()
    except OSError as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
        _write_state(state_path, state)
        print(f"Semantic dataset recorder failed: {exc}", file=sys.stderr)
        return 2

    state["recorder_returncode"] = returncode
    state["recorded_at"] = _timestamp()
    if stop_requested and returncode == 0 and (bag_path / "metadata.yaml").is_file():
        state["status"] = "recorded"
        _write_state(state_path, state)
        print(f"Semantic dataset bag recorded: {bag_path}")
        return 0

    state["status"] = "failed"
    state["error"] = f"rosbag recorder exited with return code {returncode}"
    _write_state(state_path, state)
    print(f"Semantic dataset recorder failed: {state['error']}", file=sys.stderr)
    return returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
