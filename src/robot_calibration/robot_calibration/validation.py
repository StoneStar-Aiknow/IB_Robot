"""User-facing candidate artifact validation helpers."""

import argparse
import contextlib
import fcntl
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterator
from pathlib import Path

from robot_calibration.viewer import start_viewer, stop_viewer
from robot_calibration.workflow import (
    _start_capture_preview,
    _start_sensor_calibration,
    _stop_owned_process,
    default_log_dir,
)


def validate_artifact_archive(value: Path) -> Path:
    """Accept exactly one calibration archive without activating it."""
    path = Path(value).expanduser().absolute()
    if not path.is_file() or not path.name.endswith(".candidate.tar"):
        raise ValueError(f"artifact input must end with .candidate.tar: {path}")
    return path


@contextlib.contextmanager
def validation_lock(path: Path) -> Iterator[None]:
    """Prevent two validation sessions from owning the same hardware graph."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("已有标定验证正在运行，请先结束现有验证") from exc
        yield


def run_validation(value: Path, *, mount: Path | None = None, output_topic: str = "/calib/overlay/compressed") -> int:
    """Extract a candidate into private staging and start the live overlay node."""
    archive = validate_artifact_archive(value)
    log_path = default_log_dir() / f"{archive.name.removesuffix('.candidate.tar')}-validate.log"
    sensor_process = _start_sensor_calibration(log_path)
    preview_process = _start_capture_preview(log_path)
    viewer_process = start_viewer("validate", log_path)
    try:
        with tempfile.TemporaryDirectory(prefix="robot-calibration-validate-") as directory:
            root = Path(directory)
            with tarfile.open(archive, "r") as source:
                for member in source.getmembers():
                    relative = Path(member.name)
                    if (
                        not member.isfile()
                        or relative.is_absolute()
                        or any(part in {"", ".", ".."} for part in relative.parts)
                    ):
                        raise ValueError(f"artifact archive contains unsafe member: {member.name}")
                    extracted = source.extractfile(member)
                    if extracted is None:
                        raise ValueError(f"artifact archive member is unreadable: {member.name}")
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as output:
                        output.write(extracted.read())
            artifact = root / "base_to_front_camera.candidate.yaml"
            if not artifact.is_file():
                raise ValueError("artifact archive does not contain a candidate camera artifact")
            mount_path = (mount or root / "base_to_mid360.yaml").expanduser().absolute()
            if not mount_path.is_file():
                raise ValueError("artifact archive does not contain a candidate MID-360 mount")
            command = [
                "ros2",
                "run",
                "robot_calibration",
                "calib_overlay",
                "--artifact",
                str(artifact),
                "--mount",
                str(mount_path),
                "--output-topic",
                output_topic,
                "--max-fps",
                "5.0",
                "--jpeg-quality",
                "75",
                "--no-display",
            ]
            return subprocess.run(command, check=False).returncode
    finally:
        stop_viewer(viewer_process)
        _stop_owned_process(preview_process)
        _stop_owned_process(sensor_process)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="启动候选标定的只读实时 RGB/LiDAR 叠加验证")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        with validation_lock(default_log_dir().parent / "validate.lock"):
            return run_validation(args.input)
    except (OSError, RuntimeError, subprocess.SubprocessError, TypeError, ValueError) as exc:
        print(f"标定验证失败: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
