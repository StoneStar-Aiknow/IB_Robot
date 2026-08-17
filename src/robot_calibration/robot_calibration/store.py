"""Content-addressed capture exchange and calibration revision storage."""

import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any


class StoreError(ValueError):
    """Raised when an archive or revision fails closed validation."""


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.artifacts = self.root / "artifacts"
        self.current = self.root / "current"
        self.artifacts.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def export_capture(capture: Path, archive: Path) -> str:
        ArtifactStore.verify_capture(capture)
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            raise StoreError("capture archive already exists")
        with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as output:
            for path in sorted(item for item in capture.rglob("*") if item.is_file()):
                relative = path.relative_to(capture).as_posix()
                info = output.gettarinfo(str(path), arcname=relative)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                info.mode = 0o644
                with path.open("rb") as stream:
                    output.addfile(info, stream)
        return ArtifactStore._sha256(archive)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def verify_capture(capture: Path) -> dict[str, Any]:
        if (capture / "FINALIZED").is_symlink() or not (capture / "FINALIZED").is_file():
            raise StoreError("capture is not finalized")
        manifest_path = capture / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise StoreError("capture manifest is missing")
        try:
            manifest = json.loads(manifest_path.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StoreError(f"capture manifest is invalid: {exc}") from exc
        if manifest.get("schema_version") != 1 or not manifest.get("sealed"):
            raise StoreError("capture manifest is not sealed")
        capture_id = manifest.get("capture_id")
        if not isinstance(capture_id, str) or not capture_id or Path(capture_id).name != capture_id:
            raise StoreError("capture_id must be a single safe path component")
        expected_hash = manifest.get("manifest_sha256")
        unhashed = dict(manifest)
        unhashed.pop("manifest_sha256", None)
        canonical = (json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
        if expected_hash != hashlib.sha256(canonical).hexdigest():
            raise StoreError("capture manifest sha256 mismatch")
        declared = set()
        for entry in manifest.get("files", []):
            if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
                raise StoreError("capture manifest file entry is invalid")
            relative = Path(entry["path"])
            if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                raise StoreError("capture manifest file path is unsafe")
            path = capture / relative
            if entry["path"] in declared or path.is_symlink() or not path.is_file():
                raise StoreError(f"capture file is missing or invalid: {entry['path']}")
            declared.add(entry["path"])
            if path.stat().st_size != entry["size"] or ArtifactStore._sha256(path) != entry["sha256"]:
                raise StoreError(f"capture file sha256 mismatch: {entry['path']}")
        actual = {
            path.relative_to(capture).as_posix()
            for path in capture.rglob("*")
            if path.is_file() and path.name not in {"manifest.json", "FINALIZED"}
        }
        if actual != declared:
            raise StoreError("capture files do not match manifest")
        return manifest

    @staticmethod
    def import_capture(archive: Path, destination: Path) -> dict[str, str]:
        digest = ArtifactStore._sha256(archive)
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r") as source, tempfile.TemporaryDirectory(dir=destination) as temporary:
            temporary_path = Path(temporary) / "capture"
            temporary_path.mkdir()
            for member in source.getmembers():
                relative = Path(member.name)
                if (
                    not member.isfile()
                    or relative.is_absolute()
                    or any(part in {"", ".", ".."} for part in relative.parts)
                ):
                    raise StoreError(f"capture archive contains unsafe member: {member.name}")
                extracted = source.extractfile(member)
                if extracted is None:
                    raise StoreError(f"capture archive member is unreadable: {member.name}")
                target_file = temporary_path / relative
                target_file.parent.mkdir(parents=True, exist_ok=True)
                with target_file.open("xb") as output:
                    output.write(extracted.read())
            manifest = ArtifactStore.verify_capture(temporary_path)
            target = destination / manifest["capture_id"]
            if target.exists():
                raise StoreError("capture destination already exists")
            os.rename(temporary_path, target)
        return {"sha256": digest, "path": str(target)}

    def install(self, artifact: Path) -> Path:
        value: dict[str, Any] = json.loads(Path(artifact).read_text(encoding="utf-8"))
        artifact_id = value.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise StoreError("artifact_id is required")
        revision = self.artifacts / artifact_id
        if revision.exists():
            raise StoreError("artifact revision is immutable")
        revision.mkdir()
        (revision / "artifact.json").write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        self._activate(revision)
        return revision

    def _activate(self, revision: Path) -> None:
        link = self.root / ".current.new"
        link.unlink(missing_ok=True)
        link.symlink_to(revision)
        os.replace(link, self.current)

    def current_artifact(self) -> dict[str, Any]:
        if not self.current.is_symlink():
            raise StoreError("no current artifact")
        return json.loads((self.current / "artifact.json").read_text(encoding="utf-8"))

    def rollback(self, artifact_id: str) -> None:
        revision = self.artifacts / artifact_id
        if not (revision / "artifact.json").is_file():
            raise StoreError("artifact revision does not exist")
        self._activate(revision)
