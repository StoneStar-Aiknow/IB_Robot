"""Versioned semantic artifact export without occupancy-grid ownership."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .database import ObjectGeometryRecord, SemanticMapDatabase, SemanticMapManifest


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_xyz_pcd(path: str | Path, points: np.ndarray) -> None:
    """Write deterministic ASCII PCD to avoid optional point-cloud dependencies."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        f"WIDTH {len(points)}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {len(points)}",
        "DATA ascii",
    ]
    rows = [f"{x:.8g} {y:.8g} {z:.8g}" for x, y, z in points]
    path.write_text("\n".join([*header, *rows, ""]), encoding="ascii")


class SemanticArtifactExporter:
    def __init__(self, output_dir: str | Path, database: SemanticMapDatabase):
        self.output_dir = Path(output_dir).expanduser()
        self.database = database

    def export_manifest(self, manifest: SemanticMapManifest) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "manifest.json"
        path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="ascii"
        )
        return path

    def export_geometry(self, object_id: str, object_version: int, points: np.ndarray, created_ns: int) -> Path:
        relative_path = Path("objects") / f"{object_id}.v{object_version}.pcd"
        absolute_path = self.output_dir / relative_path
        write_xyz_pcd(absolute_path, points)
        self.database.upsert_geometry(
            ObjectGeometryRecord(
                object_id=object_id,
                object_version=object_version,
                artifact_type="pointcloud_pcd",
                artifact_path=str(relative_path),
                artifact_hash=sha256_path(absolute_path),
                point_count=int(np.asarray(points).reshape(-1, 3).shape[0]),
                created_ns=created_ns,
            )
        )
        return absolute_path
