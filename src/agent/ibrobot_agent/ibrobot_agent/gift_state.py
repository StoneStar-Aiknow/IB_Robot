"""Agent-owned gift rounds, independent of conversation sessions.

Only the Agent writes this store. A validated wake starts a round; execution
owns its reservation until a confirmed result. Old wake events are idempotent
and old execution feedback can only update the round it reserved.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


class GiftStateError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class SQLiteGiftStateStore:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=5)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS gift_rounds (
                robot_scope TEXT NOT NULL, channel_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                round_id TEXT NOT NULL, status TEXT NOT NULL,
                request_id TEXT NOT NULL DEFAULT '', revision INTEGER NOT NULL,
                PRIMARY KEY (robot_scope, channel_id, principal_id, round_id)
            );
            CREATE TABLE IF NOT EXISTS gift_current (
                robot_scope TEXT NOT NULL, channel_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                round_id TEXT NOT NULL,
                PRIMARY KEY (robot_scope, channel_id, principal_id)
            );
        """)

    @staticmethod
    def _scope(scope):
        if len(scope) != 3 or any(not isinstance(value, str) or not value.strip() for value in scope):
            raise ValueError("gift scope requires robot, channel and principal")
        return tuple(scope)

    def _current(self, scope):
        row = self._db.execute(
            "SELECT r.* FROM gift_rounds r JOIN gift_current c USING "
            "(robot_scope, channel_id, principal_id, round_id) "
            "WHERE robot_scope=? AND channel_id=? AND principal_id=?",
            scope,
        ).fetchone()
        if row is None:
            self._db.execute("INSERT INTO gift_rounds VALUES (?, ?, ?, 'initial', 'available', '', 0)", scope)
            self._db.execute("INSERT INTO gift_current VALUES (?, ?, ?, 'initial')", scope)
            return self._current(scope)
        return dict(row)

    def get(self, scope) -> dict:
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            return self._current(self._scope(scope))

    def reset(self, scope, wake_id: str) -> dict:
        if not isinstance(wake_id, str) or not wake_id.strip() or len(wake_id) > 128 or wake_id == "initial":
            raise ValueError("wake_id must be a nonempty event ID of at most 128 characters")
        scope = self._scope(scope)
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            current = self._current(scope)
            previous = self._db.execute(
                "SELECT 1 FROM gift_rounds WHERE robot_scope=? AND channel_id=? AND principal_id=? AND round_id=?",
                (*scope, wake_id),
            ).fetchone()
            if previous:
                if current["round_id"] != wake_id:
                    raise GiftStateError("STALE_GIFT_WAKE", "an old wake cannot replace the current round")
                return current
            if current["status"] in {"in_progress", "uncertain"}:
                raise GiftStateError("GIFT_STATE_UNRESOLVED", "finish or reconcile the current delivery first")
            self._db.execute(
                "INSERT INTO gift_rounds VALUES (?, ?, ?, ?, 'available', '', ?)",
                (*scope, wake_id, current["revision"] + 1),
            )
            self._db.execute(
                "UPDATE gift_current SET round_id=? WHERE robot_scope=? AND channel_id=? AND principal_id=?",
                (wake_id, *scope),
            )
            return self._current(scope)

    def reserve(self, scope, round_id: str, request_id: str) -> dict:
        scope = self._scope(scope)
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            current = self._current(scope)
            if current["round_id"] != round_id or current["status"] != "available":
                raise GiftStateError("GIFT_NOT_AVAILABLE", "this round cannot start another gift")
            self._db.execute(
                "UPDATE gift_rounds SET status='in_progress', request_id=?, revision=revision+1 "
                "WHERE robot_scope=? AND channel_id=? AND principal_id=? AND round_id=?",
                (request_id, *scope, round_id),
            )
            return self._current(scope)

    def finish(self, scope, round_id: str, request_id: str, status: str) -> None:
        if status not in {"available", "delivered", "uncertain"}:
            raise ValueError("invalid gift result status")
        scope = self._scope(scope)
        with self._lock, self._db:
            # A confirmed delivery is monotonic, including when a later return fails.
            self._db.execute(
                "UPDATE gift_rounds SET status=?, revision=revision+1 "
                "WHERE robot_scope=? AND channel_id=? AND principal_id=? AND round_id=? "
                "AND request_id=? AND status IN ('in_progress', 'uncertain') AND status != ?",
                (status, *scope, round_id, request_id, status),
            )

    def recover_interrupted(self, robot_scope: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE gift_rounds SET status='uncertain', revision=revision+1 WHERE status='in_progress' AND robot_scope=?",
                (robot_scope,),
            )

    def close(self) -> None:
        with self._lock:
            self._db.close()
