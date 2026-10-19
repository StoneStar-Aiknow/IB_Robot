"""Small cross-process stop ledger, independent of Hermes' conversation lease.

One profile owns the directory; one locked record owns each session. No PID from
this ledger is ever signaled: a runner signals only its own Popen child. An
orphaned record stays unknown until its exact ROS task is reconciled.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


class StopStateError(RuntimeError):
    pass


class StopStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.is_symlink() or self.root.stat().st_uid != os.getuid() or self.root.stat().st_mode & 0o077:
            raise StopStateError("stop state directory must be private and owned by this user")

    @contextmanager
    def owner_lock(self, token: str):
        path = self.root / f"owner-{hashlib.sha256(token.encode()).hexdigest()}.lock"
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield fd
        finally:
            os.close(fd)

    def owner_alive(self, token: str) -> bool:
        try:
            with self.owner_lock(token):
                return False
        except BlockingIOError:
            return True

    @contextmanager
    def locked(self, session: str):
        if not session:
            raise StopStateError("missing Hermes session identity")
        key = hashlib.sha256(session.encode()).hexdigest()
        path = self.root / f"{key}.json"
        descriptor = os.open(self.root / f"{key}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            # Bounded waiting prevents an abandoned application lock from
            # blocking the input thread. OS locks release on process exit.
            deadline = time.monotonic() + 0.15
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise StopStateError("stop state is busy") from None
                    time.sleep(0.005)
            record = (
                json.loads(path.read_text())
                if path.exists()
                else {
                    "session": session,
                    "stop_at": 0.0,
                    "blocked": False,
                    "turns": {},
                    "owner": None,
                }
            )
            if record.get("session") != session:
                raise StopStateError("stop session identity mismatch")
            before = json.dumps(record, sort_keys=True)
            yield record
            after = json.dumps(record, sort_keys=True, ensure_ascii=False)
            if json.dumps(record, sort_keys=True) != before or not path.exists():
                fd, temporary = tempfile.mkstemp(prefix=f".{key}.", dir=self.root)
                try:
                    with os.fdopen(fd, "w") as stream:
                        stream.write(after)
                    os.replace(temporary, path)
                finally:
                    Path(temporary).unlink(missing_ok=True)
        finally:
            os.close(descriptor)

    def user_input(self, session: str, *, stop: bool, busy: bool = False) -> dict:
        with self.locked(session) as record:
            owner = record["owner"]
            if stop:
                record["stop_at"] = time.time()
                record["blocked"] = True
                for turn in record["turns"].values():
                    turn["stopped"] = True
                if owner and owner["state"] in {"starting", "running"}:
                    owner["stop_requested"] = True
                    owner.setdefault("stop_received_at", time.time())
                    return {"state": "stopping"}
                if owner and owner["state"] == "unknown":
                    return {"state": "unknown"}
                return {"state": "no_active_task"}
            terminal_owner = owner and owner["state"] in {"stopped", "succeeded", "failed", "not_submitted"}
            if (not busy and not owner) or (terminal_owner and not self.owner_alive(owner["token"])):
                record["blocked"] = False
            return {"state": "normal"}

    def begin_turn(self, session: str, turn: str, started: float) -> None:
        if not turn:
            raise StopStateError("missing Hermes turn identity")
        with self.locked(session) as record:
            record["turns"][turn] = {
                "stopped": record["blocked"] or started <= record["stop_at"],
                "started": started,
            }

    def end_turn(self, session: str, turn: str) -> None:
        with self.locked(session) as record:
            record["turns"].pop(turn, None)

    def claim(self, session: str, turn: str, token: str) -> None:
        with self.locked(session) as record:
            current = record["turns"].get(turn)
            if current is None or current["stopped"] or record["blocked"]:
                raise StopStateError("STOP_LATCHED: this turn cannot start robot motion")
            owner = record["owner"]
            if owner and (owner["state"] in {"starting", "running", "unknown"} or self.owner_alive(owner["token"])):
                raise StopStateError("ROBOT_STOP_UNKNOWN_OR_BUSY: previous task has not converged")
            record["owner"] = {
                "token": token,
                "turn": turn,
                "state": "starting",
                "stop_requested": False,
                "presentation": None,
                "terminal": None,
                "started_at": time.time(),
            }

    def update(self, session: str, token: str, **values) -> None:
        with self.locked(session) as record:
            owner = record["owner"]
            if not owner or owner["token"] != token:
                raise StopStateError("robot execution ownership changed")
            owner.update(values)

    def stopped(self, session: str, token: str) -> bool:
        with self.locked(session) as record:
            owner = record["owner"]
            return not owner or owner["token"] != token or owner["stop_requested"]

    @contextmanager
    def submission_guard(self, session: str, token: str, stop_event):
        """Serialize the stop latch with the exact nonblocking goal submission."""
        with self.locked(session) as record:
            owner = record["owner"]
            if not owner or owner["token"] != token or owner["stop_requested"]:
                stop_event.set()
            if owner and owner["token"] == token:
                owner["fresh_submission_pending"] = False
            yield

    def snapshot(self, session: str) -> dict:
        with self.locked(session) as record:
            return json.loads(json.dumps(record))
