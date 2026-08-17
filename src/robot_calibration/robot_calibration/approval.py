"""Explicitly approve a validated candidate calibration for production."""

import argparse
import hashlib
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

from robot_calibration.sensor_calibration import _load_artifact

CANDIDATE_FILES = {
    "base_to_front_camera.candidate.yaml": "base_to_front_camera.yaml",
    "base_to_mid360.yaml": "base_to_mid360.yaml",
    "front_camera_intrinsics.yaml": "front_camera_intrinsics.yaml",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_candidate(archive: Path, root: Path) -> dict[str, Path]:
    with tarfile.open(archive, "r") as source:
        for member in source.getmembers():
            relative = Path(member.name)
            if not member.isfile() or relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                raise ValueError(f"archive contains unsafe member: {member.name}")
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = source.extractfile(member)
            if extracted is None:
                raise ValueError(f"archive member is unreadable: {member.name}")
            target.write_bytes(extracted.read())
    candidates = {name: root / name for name in CANDIDATE_FILES}
    for name, path in candidates.items():
        if not path.is_file():
            raise ValueError(f"archive does not contain {name}")
    return candidates


def approve(archive: Path, root: Path | None = None) -> Path:
    archive = Path(archive).expanduser().absolute()
    if not archive.name.endswith(".candidate.tar"):
        raise ValueError("input must end with .candidate.tar")
    production = Path(root or "~/.ros/ibrobot/calib").expanduser().absolute()
    with tempfile.TemporaryDirectory(prefix="calib-approve-") as directory:
        candidates = _extract_candidate(archive, Path(directory))
        values = {}
        for candidate_name, path in candidates.items():
            artifact_name = CANDIDATE_FILES[candidate_name].removesuffix(".yaml")
            resolved = _load_artifact(artifact_name, path, None)
            if resolved["state"] != "candidate":
                codes = ", ".join(item["code"] for item in resolved["diagnostics"])
                raise ValueError(f"{candidate_name} is not a valid candidate calibration: {codes}")
            values[candidate_name] = resolved["data"]
        current = production / "current"
        current.mkdir(parents=True, exist_ok=True)
        approved_at = datetime.now(timezone.utc).isoformat()
        for candidate_name, output_name in CANDIDATE_FILES.items():
            approved = dict(values[candidate_name])
            approved["status"] = "approved"
            approved["approved_at"] = approved_at
            approved["candidate_sha256"] = _sha256(candidates[candidate_name])
            destination = current / output_name
            destination.unlink(missing_ok=True)
            destination.write_text(yaml.safe_dump(approved, sort_keys=False), encoding="utf-8")
        return current / "base_to_front_camera.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="将已现场确认的 candidate 标定转为 approved production 标定")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(f"approved calibration: {approve(args.input)}")
        return 0
    except (OSError, tarfile.TarError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"标定转正失败: {exc}", file=sys.stderr)
        return 2
