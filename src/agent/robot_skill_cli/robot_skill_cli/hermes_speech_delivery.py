"""At-most-once speech delivery shared by final, interim and lifecycle events."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path

from .hermes_response import SpeechEvent

_STATE_MAX_AGE_SEC = 7 * 24 * 3600
_STATE_FILENAME = re.compile(r"^[0-9a-f]{64}(?:\.terminal)?$")


def _validate_state_root(root: Path) -> None:
    details = root.lstat()
    if root.is_symlink() or not stat.S_ISDIR(details.st_mode):
        raise OSError(f"speech state path must be a directory: {root}")
    if details.st_uid != os.getuid() or details.st_mode & 0o077:
        raise OSError(f"speech state directory must be private and owned by this user: {root}")


def _cleanup_expired(root: Path, *, now: float | None = None) -> None:
    cutoff = (time.time() if now is None else now) - _STATE_MAX_AGE_SEC
    for entry in root.iterdir():
        if not _STATE_FILENAME.fullmatch(entry.name):
            continue
        try:
            details = entry.lstat()
            if stat.S_ISREG(details.st_mode) and details.st_mtime < cutoff:
                entry.unlink()
        except FileNotFoundError:
            continue


def _state_root() -> Path:
    root = Path(os.environ.get("IBROBOT_SPEECH_STATE", "/tmp/ibrobot-speech-events"))
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _validate_state_root(root)
    _cleanup_expired(root)
    return root


def _terminal_path(session_id: str, turn_id: str) -> Path:
    identity = json.dumps([session_id, turn_id], ensure_ascii=False)
    return _state_root() / (hashlib.sha256(identity.encode()).hexdigest() + ".terminal")


def finish_speech_turn(session_id: str, turn_id: str) -> None:
    """Invalidate queued/synthesizing progress before the final reply is spoken."""
    if session_id and turn_id:
        _terminal_path(session_id, turn_id).touch(mode=0o600, exist_ok=True)


def event_is_current(event: SpeechEvent) -> bool:
    return event.phase == "final" or not _terminal_path(event.session_id, event.turn_id).exists()


def claim_event(event: SpeechEvent) -> bool:
    """Claim before synthesis; unknown playback outcomes are never retried."""
    root = _state_root()
    parts = [event.session_id, event.turn_id, event.phase]
    if event.schema_version == 2:
        parts.append(event.sequence)
    identity = json.dumps(parts, ensure_ascii=False)
    digest = hashlib.sha256(identity.encode()).hexdigest()
    try:
        descriptor = os.open(root / digest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "identity": json.loads(identity),
                "speech_sha256": hashlib.sha256(event.response.speech.encode()).hexdigest(),
            },
            stream,
        )
    return True
