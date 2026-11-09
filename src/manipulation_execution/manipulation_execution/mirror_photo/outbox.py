"""The photos chosen but not yet delivered: one directory, crash-safe, bounded; no ROS.

Every photo is two files in ``<photos_dir>/outbox``: ``<round_id>.jpg`` and
``<round_id>.json`` (its metadata), each written to a temporary name, synced
and renamed, the JPEG first. A JPEG without its JSON is a write cut short and
is read back with minimal metadata; nothing is thrown away on restart. Round
ids start with the local date and time, so name order is round order.

``current.jpg`` / ``current.json`` hold the running round's provisional best.
A normal round end removes them after its photo is in; if they are still
there at startup the round never ended (crash, kill), and they become that
round's photo (``quality: recovered``) unless the round's photo already is in.

The bounds are ``outbox_max_files``, ``outbox_max_mb``: past either the
oldest photo is dropped with an ERROR naming its round, the only way a photo
is ever lost. Below ``min_free_gb`` of free disk a new photo is not written
at all but kept in memory (ERROR as well) and still delivered; it is lost only
if the node stops before that.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .selector import SCHEMA, Photo

CURRENT = "current"
Log = Callable[[str, str], None]


def _write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _meta_bytes(meta: dict) -> bytes:
    return json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=1).encode("utf-8")


class Outbox:
    def __init__(
        self,
        photos_dir: str,
        *,
        max_files: int,
        max_mb: float,
        min_free_gb: float,
        log: Log,
        free_bytes: Callable[[Path], int] | None = None,
    ):
        self.path = Path(photos_dir) / "outbox"
        self.max_files = max_files
        self.max_bytes = int(max_mb * 1024 * 1024)
        self.min_free_bytes = int(min_free_gb * 1024**3)
        self._log = log
        self._free = free_bytes or (lambda path: shutil.disk_usage(path).free)
        self._lock = threading.Lock()
        # Photos that found no disk room: name -> Photo.
        self._memory: dict[str, Photo] = {}
        self.dropped = 0

    # -- startup -------------------------------------------------------------------------

    def prepare(self) -> None:
        """Create the directory, finish what a crash left, and hold the bounds."""
        with self._lock:
            self.path.mkdir(parents=True, exist_ok=True)
            for leftover in self.path.glob("*.tmp"):
                leftover.unlink(missing_ok=True)
            for meta in self.path.glob("*.json"):
                if meta.stem != CURRENT and not meta.with_suffix(".jpg").exists():
                    meta.unlink(missing_ok=True)
            for jpeg in self.path.glob("*.jpg"):
                if jpeg.stem != CURRENT and not jpeg.with_suffix(".json").exists():
                    data = jpeg.read_bytes()
                    meta = {"schema": SCHEMA, "round_id": jpeg.stem, "quality": "recovered"}
                    _write_atomic(jpeg.with_suffix(".json"), _meta_bytes(self._fresh(meta, data)))
                    self._log(
                        "warning",
                        f"photo outbox: {jpeg.stem} had no metadata (write cut short); sending it as recovered",
                    )
            self._recover_current()
            _sync_dir(self.path)
            self._hold_bounds(incoming=0)

    def _recover_current(self) -> None:
        jpeg = self.path / f"{CURRENT}.jpg"
        meta_path = self.path / f"{CURRENT}.json"
        if not jpeg.exists():
            meta_path.unlink(missing_ok=True)
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            name = str(meta["round_id"])
        except (OSError, ValueError, KeyError, TypeError):
            name = time.strftime("%Y%m%d-%H%M%S", time.localtime(jpeg.stat().st_mtime)) + "-rec"
            meta = {"schema": SCHEMA, "round_id": name}
        if (self.path / f"{name}.json").exists():
            jpeg.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return
        data = jpeg.read_bytes()
        meta.update(quality="recovered", end_reason="restart")
        os.replace(jpeg, self.path / f"{name}.jpg")
        _write_atomic(self.path / f"{name}.json", _meta_bytes(self._fresh(meta, data)))
        meta_path.unlink(missing_ok=True)
        self._log(
            "warning", f"photo outbox: round {name} never ended (node stopped mid-round); sending its provisional best"
        )

    @staticmethod
    def _fresh(meta: dict, data: bytes) -> dict:
        return {**meta, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    # -- the running round -----------------------------------------------------------

    def write_current(self, photo: Photo) -> None:
        with self._lock:
            if self._low_disk(len(photo.jpeg)):
                return
            _write_atomic(self.path / f"{CURRENT}.jpg", photo.jpeg)
            _write_atomic(self.path / f"{CURRENT}.json", _meta_bytes(photo.meta))

    def clear_current(self) -> None:
        with self._lock:
            for suffix in (".jpg", ".json"):
                (self.path / f"{CURRENT}{suffix}").unlink(missing_ok=True)

    # -- photos ------------------------------------------------------------------------

    def put(self, photo: Photo) -> str:
        """Queue ``photo``; returns ``disk`` or ``memory`` (no room on disk)."""
        with self._lock:
            if self._low_disk(len(photo.jpeg)):
                self._memory[photo.name] = photo
                self._log(
                    "error",
                    f"photo outbox: less than {self.min_free_bytes / 1024**3:g} GB free on {self.path}; round "
                    f"{photo.name} kept in memory only (lost if the node stops before it is delivered)",
                )
                self._hold_bounds(incoming=0)
                return "memory"
            self._hold_bounds(incoming=len(photo.jpeg))
            _write_atomic(self.path / f"{photo.name}.jpg", photo.jpeg)
            _write_atomic(self.path / f"{photo.name}.json", _meta_bytes(photo.meta))
            _sync_dir(self.path)
            return "disk"

    def names(self) -> list[str]:
        """Queued photos, newest first."""
        with self._lock:
            return sorted(set(self._disk_names()) | set(self._memory), reverse=True)

    def load(self, name: str) -> tuple[bytes, dict] | None:
        with self._lock:
            if name in self._memory:
                photo = self._memory[name]
                return photo.jpeg, dict(photo.meta)
            try:
                data = (self.path / f"{name}.jpg").read_bytes()
            except OSError:
                return None
            try:
                meta = json.loads((self.path / f"{name}.json").read_text(encoding="utf-8"))
                if not isinstance(meta, dict):
                    raise ValueError("not an object")
            except (OSError, ValueError):
                meta = {"schema": SCHEMA, "round_id": name, "quality": "recovered"}
            return data, self._fresh(meta, data)

    def remove(self, name: str) -> None:
        with self._lock:
            self._memory.pop(name, None)
            for suffix in (".json", ".jpg"):
                (self.path / f"{name}{suffix}").unlink(missing_ok=True)

    def stats(self) -> dict:
        with self._lock:
            names = self._disk_names()
            return {
                "queued": len(names) + len(self._memory),
                "in_memory": len(self._memory),
                "bytes": sum(self._size(name) for name in names),
                "dropped": self.dropped,
            }

    # -- internals ---------------------------------------------------------------------

    def _disk_names(self) -> list[str]:
        try:
            return sorted(
                path.stem
                for path in self.path.glob("*.json")
                if path.stem != CURRENT and path.with_suffix(".jpg").exists()
            )
        except OSError:
            return []

    def _size(self, name: str) -> int:
        total = 0
        for suffix in (".jpg", ".json"):
            with contextlib.suppress(OSError):
                total += (self.path / f"{name}{suffix}").stat().st_size
        return total

    def _low_disk(self, incoming: int) -> bool:
        try:
            return self._free(self.path) - incoming < self.min_free_bytes
        except OSError:
            return False

    def _hold_bounds(self, *, incoming: int) -> None:
        """Drop the oldest photos until one more of ``incoming`` bytes fits (0: until the bounds hold)."""
        names = self._disk_names()
        memory = sorted(self._memory)
        room = 1 if incoming else 0
        total = sum(self._size(name) for name in names)
        while names or memory:
            if len(names) + len(memory) + room <= self.max_files and total + incoming <= self.max_bytes:
                return
            oldest_disk = names[0] if names else None
            oldest_memory = memory[0] if memory else None
            if oldest_memory is not None and (oldest_disk is None or oldest_memory < oldest_disk):
                memory.pop(0)
                self._memory.pop(oldest_memory, None)
                name = oldest_memory
            else:
                names.pop(0)
                total -= self._size(oldest_disk)
                for suffix in (".json", ".jpg"):
                    (self.path / f"{oldest_disk}{suffix}").unlink(missing_ok=True)
                name = oldest_disk
            self.dropped += 1
            self._log(
                "error",
                f"photo outbox full ({self.max_files} photos / {self.max_bytes // (1024 * 1024)} MB): "
                f"dropped round {name}, never delivered",
            )
