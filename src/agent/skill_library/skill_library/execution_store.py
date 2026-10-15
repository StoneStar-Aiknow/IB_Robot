"""Durable authoritative execution ledger owned by the Skill Gateway."""

from __future__ import annotations

import contextlib
import fcntl
import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embodied_common.program_contracts import CanonicalProgram

TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"})
_RECOVERY_UNKNOWN_STATES = ("COMMITTING", "ACTIVE", "STOPPING", "FINALIZING")
_RECOVERY_FAILED_STATES = ("RECEIVED",)


class ExecutionConflictError(Exception):
    """The scoped request identity already belongs to another Program."""


class ExecutionBusyError(Exception):
    """Another Program still owns the robot execution slot."""


class ExecutionQuarantinedError(Exception):
    """An UNKNOWN physical effect blocks new motion."""


@dataclass(frozen=True)
class ExecutionRecord:
    execution_id: str
    request_id: str
    caller_scope: str
    robot_id: str
    program_digest: str
    program_json: str
    workflow_digest: str
    state: str
    completed_step_count: int
    stop_requested: bool
    stop_generation: int
    last_event_sequence: int
    error_code: str
    message: str
    actual_registry_epoch: str
    actual_registry_generation: int
    actual_registry_digest: str
    created_at: str
    updated_at: str

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def program(self) -> dict[str, Any]:
        return json.loads(self.program_json)


@dataclass(frozen=True)
class ExecutionEventRecord:
    execution_id: str
    sequence: int
    state: str
    completed_step_count: int
    error_code: str
    message: str
    created_at: str


class SQLiteExecutionStore:
    """SQLite WAL ledger with a durable commit-intent and ordered event outbox."""

    def __init__(self, path: str | Path) -> None:
        normalized = str(path).strip()
        if not normalized:
            raise ValueError("execution_store_path must be non-empty")
        self._path = Path(normalized).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._writer_lock = self._path.with_suffix(self._path.suffix + ".lock").open("a+")
        try:
            fcntl.flock(self._writer_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._writer_lock.close()
            raise RuntimeError("execution store already has an active writer") from None
        self._lock = threading.RLock()
        self._unknown_pending = True
        # Event listener notification happens strictly after commit and outside
        # the store lock, so a slow subscriber can never stall ledger writes.
        self._event_listener = None
        self._pending_event_ids: list[str] = []
        try:
            self._connection = sqlite3.connect(self._path, check_same_thread=False, timeout=5.0)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA busy_timeout=5000")
            self._create_schema()
            self.recover_interrupted()
            self._refresh_quarantine()
        except Exception:
            if hasattr(self, "_connection"):
                self._connection.close()
            self._writer_lock.close()
            raise

    def set_event_listener(self, listener) -> None:
        """Register ``listener(execution_ids)`` invoked after each committed event append."""
        with self._lock:
            self._event_listener = listener

    def _take_committed_event_ids(self) -> tuple[list[str], object]:
        ids, self._pending_event_ids = self._pending_event_ids, []
        return list(dict.fromkeys(ids)), self._event_listener

    def _discard_pending_event_ids(self) -> None:
        self._pending_event_ids = []

    def close(self) -> None:
        with self._lock:
            self._connection.close()
            self._writer_lock.close()

    def _create_schema(self) -> None:
        with self._transaction() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS executions (
                    execution_id TEXT PRIMARY KEY,
                    caller_scope TEXT NOT NULL,
                    robot_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    program_digest TEXT NOT NULL,
                    program_json TEXT NOT NULL,
                    workflow_digest TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL,
                    completed_step_count INTEGER NOT NULL DEFAULT 0,
                    stop_requested INTEGER NOT NULL DEFAULT 0,
                    stop_generation INTEGER NOT NULL DEFAULT 0,
                    last_event_sequence INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT NOT NULL DEFAULT '',
                    message TEXT NOT NULL DEFAULT '',
                    actual_registry_epoch TEXT NOT NULL DEFAULT '',
                    actual_registry_generation INTEGER NOT NULL DEFAULT 0,
                    actual_registry_digest TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (caller_scope, robot_id, request_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_events (
                    execution_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    completed_step_count INTEGER NOT NULL,
                    error_code TEXT NOT NULL DEFAULT '',
                    message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (execution_id, sequence),
                    FOREIGN KEY (execution_id) REFERENCES executions(execution_id)
                )
                """
            )
            columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(executions)").fetchall()}
            if "workflow_digest" not in columns:
                connection.execute("ALTER TABLE executions ADD COLUMN workflow_digest TEXT NOT NULL DEFAULT ''")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS execution_stop_intents (
                    caller_scope TEXT NOT NULL, robot_id TEXT NOT NULL,
                    request_id TEXT NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY (caller_scope, robot_id, request_id))"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS execution_resolutions (
                    execution_id TEXT PRIMARY KEY, operator_scope TEXT NOT NULL,
                    evidence TEXT NOT NULL, resolved_at TEXT NOT NULL)"""
            )

    def recover_interrupted(self) -> int:
        """Quarantine every execution that crossed the commit-intent barrier."""
        with self._transaction() as connection:
            precommit_rows = connection.execute(
                f"SELECT execution_id FROM executions WHERE state IN ({','.join('?' for _ in _RECOVERY_FAILED_STATES)})",
                _RECOVERY_FAILED_STATES,
            ).fetchall()
            for row in precommit_rows:
                self._transition_in_transaction(
                    connection,
                    str(row["execution_id"]),
                    state="FAILED",
                    error_code="EXECUTION_RESTARTED_BEFORE_COMMIT",
                    message="runtime restarted before the motion commit barrier; request was not replayed",
                )
            rows = connection.execute(
                f"SELECT execution_id FROM executions WHERE state IN ({','.join('?' for _ in _RECOVERY_UNKNOWN_STATES)})",
                _RECOVERY_UNKNOWN_STATES,
            ).fetchall()
            for row in rows:
                self._transition_in_transaction(
                    connection,
                    str(row["execution_id"]),
                    state="UNKNOWN",
                    error_code="EXECUTION_RECOVERY_UNKNOWN",
                    message="runtime restarted after commit intent; downstream effects require operator reconciliation",
                )
            return len(precommit_rows) + len(rows)

    def begin(self, program: CanonicalProgram) -> tuple[ExecutionRecord, bool]:
        """Insert one scoped request or return the exact existing execution."""
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM executions WHERE caller_scope=? AND robot_id=? AND request_id=?",
                program.execution_key,
            ).fetchone()
            if existing is not None:
                record = self._record(existing)
                if record.program_digest != program.program_digest:
                    raise ExecutionConflictError("request_id is already bound to a different Program")
                return record, False
            unknown = connection.execute(
                "SELECT execution_id FROM executions WHERE robot_id=? AND state='UNKNOWN' "
                "AND execution_id NOT IN (SELECT execution_id FROM execution_resolutions) LIMIT 1",
                (program.robot_id,),
            ).fetchone()
            if unknown is not None:
                raise ExecutionQuarantinedError("an earlier execution has unresolved physical effects")
            placeholders = ",".join("?" for _ in TERMINAL_STATES)
            active = connection.execute(
                f"SELECT execution_id FROM executions WHERE robot_id=? AND state NOT IN ({placeholders}) LIMIT 1",
                (program.robot_id, *sorted(TERMINAL_STATES)),
            ).fetchone()
            if active is not None:
                raise ExecutionBusyError("another Program owns the robot execution slot")
            now = _utc_now()
            execution_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO executions (
                    execution_id, caller_scope, robot_id, request_id, program_digest,
                    program_json, state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'RECEIVED', ?, ?)
                """,
                (
                    execution_id,
                    program.caller_scope,
                    program.robot_id,
                    program.request_id,
                    program.program_digest,
                    json.dumps(program.to_dict(), sort_keys=True, separators=(",", ":")),
                    now,
                    now,
                ),
            )
            self._append_event_in_transaction(connection, execution_id, "RECEIVED", 0, "", "")
            stopped = connection.execute(
                "SELECT 1 FROM execution_stop_intents WHERE caller_scope=? AND robot_id=? AND request_id=?",
                program.execution_key,
            ).fetchone()
            if stopped is not None:
                connection.execute(
                    "UPDATE executions SET stop_requested=1, stop_generation=1 WHERE execution_id=?", (execution_id,)
                )
                self._transition_in_transaction(
                    connection,
                    execution_id,
                    state="CANCELED",
                    error_code="SKILL_CANCELLED",
                    message="stop intent arrived before this request; no action dispatched",
                )
            return self._get_in_transaction(connection, execution_id), True

    def mark_committing(self, execution_id: str) -> ExecutionRecord:
        return self.transition(execution_id, state="COMMITTING")

    def mark_active(self, execution_id: str) -> ExecutionRecord:
        return self.transition(execution_id, state="ACTIVE")

    def bind_workflow_digest(self, execution_id: str, workflow_digest: str) -> ExecutionRecord:
        normalized = str(workflow_digest).strip()
        if not normalized:
            raise ValueError("workflow_digest must be non-empty")
        with self._transaction() as connection:
            current = self._get_in_transaction(connection, execution_id)
            if current.workflow_digest and current.workflow_digest != normalized:
                raise ExecutionConflictError("workflow digest is immutable")
            connection.execute(
                "UPDATE executions SET workflow_digest=?, updated_at=? WHERE execution_id=?",
                (normalized, _utc_now(), execution_id),
            )
            return self._get_in_transaction(connection, execution_id)

    def finish(
        self,
        execution_id: str,
        *,
        state: str,
        completed_step_count: int,
        error_code: str = "",
        message: str = "",
        actual_registry_epoch: str = "",
        actual_registry_generation: int = 0,
        actual_registry_digest: str = "",
    ) -> ExecutionRecord:
        if state not in TERMINAL_STATES:
            raise ValueError("terminal execution state is invalid")
        return self.transition(
            execution_id,
            state=state,
            completed_step_count=completed_step_count,
            error_code=error_code,
            message=message,
            actual_registry_epoch=actual_registry_epoch,
            actual_registry_generation=actual_registry_generation,
            actual_registry_digest=actual_registry_digest,
        )

    def transition(
        self,
        execution_id: str,
        *,
        state: str,
        completed_step_count: int | None = None,
        error_code: str | None = None,
        message: str | None = None,
        actual_registry_epoch: str | None = None,
        actual_registry_generation: int | None = None,
        actual_registry_digest: str | None = None,
    ) -> ExecutionRecord:
        with self._transaction() as connection:
            return self._transition_in_transaction(
                connection,
                execution_id,
                state=state,
                completed_step_count=completed_step_count,
                error_code=error_code,
                message=message,
                actual_registry_epoch=actual_registry_epoch,
                actual_registry_generation=actual_registry_generation,
                actual_registry_digest=actual_registry_digest,
            )

    def request_stop(self, execution_id: str) -> ExecutionRecord:
        """Persist a monotonic stop request independently from worker progress."""
        with self._transaction() as connection:
            current = self._get_in_transaction(connection, execution_id)
            if current.terminal or current.stop_requested:
                return current
            generation = current.stop_generation + 1
            now = _utc_now()
            state = "STOPPING"
            connection.execute(
                """
                UPDATE executions
                   SET stop_requested=1, stop_generation=?, state=?, updated_at=?
                 WHERE execution_id=?
                """,
                (generation, state, now, execution_id),
            )
            self._append_event_in_transaction(
                connection,
                execution_id,
                state,
                current.completed_step_count,
                "",
                "stop requested",
            )
            return self._get_in_transaction(connection, execution_id)

    def request_stop_by_key(self, caller_scope: str, robot_id: str, request_id: str) -> ExecutionRecord | None:
        """Persist a stop even when the corresponding goal has not arrived yet."""
        if not all(
            isinstance(value, str) and value.strip() == value and value
            for value in (caller_scope, robot_id, request_id)
        ):
            raise ValueError("complete canonical request scope is required")
        with self._transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO execution_stop_intents VALUES (?, ?, ?, ?)",
                (caller_scope, robot_id, request_id, _utc_now()),
            )
            row = connection.execute(
                "SELECT execution_id FROM executions WHERE caller_scope=? AND robot_id=? AND request_id=?",
                (caller_scope, robot_id, request_id),
            ).fetchone()
        # begin() observes the tombstone if it raced the transaction above.
        return None if row is None else self.request_stop(str(row["execution_id"]))

    def resolve_unknown(self, execution_id: str, *, operator_scope: str, evidence: str) -> None:
        """Record offline operator reconciliation; never rewrite the old result.

        The maintenance caller must hold this store's exclusive writer lock
        after stopping the runtime and verifying/resetting downstream hardware.
        """
        if not operator_scope.strip() or not evidence.strip():
            raise ValueError("operator identity and reconciliation evidence are required")
        with self._transaction() as connection:
            current = self._get_in_transaction(connection, execution_id)
            if current.state != "UNKNOWN":
                raise ValueError("only an UNKNOWN execution may be reconciled")
            existing = connection.execute(
                "SELECT operator_scope, evidence FROM execution_resolutions WHERE execution_id=?", (execution_id,)
            ).fetchone()
            if existing is not None:
                if (existing["operator_scope"], existing["evidence"]) != (operator_scope, evidence):
                    raise ExecutionConflictError("execution already has different reconciliation evidence")
                return
            connection.execute(
                "INSERT INTO execution_resolutions VALUES (?, ?, ?, ?)",
                (execution_id, operator_scope, evidence, _utc_now()),
            )
            self._append_event_in_transaction(
                connection,
                execution_id,
                "UNKNOWN",
                current.completed_step_count,
                "EXECUTION_RECONCILED",
                f"operator={operator_scope}; {evidence}",
            )
        self._refresh_quarantine()

    def get(
        self,
        *,
        execution_id: str = "",
        caller_scope: str = "",
        robot_id: str = "",
        request_id: str = "",
    ) -> ExecutionRecord | None:
        with self._lock:
            if execution_id:
                row = self._connection.execute(
                    "SELECT * FROM executions WHERE execution_id=?", (execution_id,)
                ).fetchone()
                if row is not None and any(
                    supplied and str(row[field]) != supplied
                    for field, supplied in (
                        ("caller_scope", caller_scope),
                        ("robot_id", robot_id),
                        ("request_id", request_id),
                    )
                ):
                    return None
            elif caller_scope and robot_id and request_id:
                row = self._connection.execute(
                    "SELECT * FROM executions WHERE caller_scope=? AND robot_id=? AND request_id=?",
                    (caller_scope, robot_id, request_id),
                ).fetchone()
            else:
                raise ValueError("execution_id or complete scoped request identity is required")
            return None if row is None else self._record(row)

    def events_after(
        self, execution_id: str, sequence: int = 0, *, limit: int = 100
    ) -> tuple[ExecutionEventRecord, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("event limit must be between 1 and 100")
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM execution_events WHERE execution_id=? AND sequence>? ORDER BY sequence LIMIT ?",
                (execution_id, sequence, limit),
            ).fetchall()
            return tuple(
                ExecutionEventRecord(
                    execution_id=str(row["execution_id"]),
                    sequence=int(row["sequence"]),
                    state=str(row["state"]),
                    completed_step_count=int(row["completed_step_count"]),
                    error_code=str(row["error_code"]),
                    message=str(row["message"]),
                    created_at=str(row["created_at"]),
                )
                for row in rows
            )

    def has_unresolved_unknown(self) -> bool:
        # Called while Gateway holds its control-state lock: never wait for a
        # SQLite transaction here, otherwise stop could be stalled by disk I/O.
        return self._unknown_pending

    def _refresh_quarantine(self) -> None:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM executions WHERE state='UNKNOWN' "
                "AND execution_id NOT IN (SELECT execution_id FROM execution_resolutions) LIMIT 1"
            ).fetchone()
            self._unknown_pending = row is not None

    def reconciliation(self, execution_id: str) -> dict[str, str] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT operator_scope, evidence, resolved_at FROM execution_resolutions WHERE execution_id=?",
                (execution_id,),
            ).fetchone()
            return dict(row) if row is not None else None

    def _transition_in_transaction(
        self,
        connection: sqlite3.Connection,
        execution_id: str,
        *,
        state: str,
        completed_step_count: int | None = None,
        error_code: str | None = None,
        message: str | None = None,
        actual_registry_epoch: str | None = None,
        actual_registry_generation: int | None = None,
        actual_registry_digest: str | None = None,
    ) -> ExecutionRecord:
        current = self._get_in_transaction(connection, execution_id)
        if state == "UNKNOWN":
            self._unknown_pending = True
        if current.terminal:
            requested = (
                state,
                current.completed_step_count if completed_step_count is None else completed_step_count,
                current.error_code if error_code is None else error_code,
                current.message if message is None else message,
            )
            actual = (current.state, current.completed_step_count, current.error_code, current.message)
            if requested != actual:
                raise ExecutionConflictError("terminal execution result is immutable")
            return current
        values = {
            "completed_step_count": current.completed_step_count
            if completed_step_count is None
            else int(completed_step_count),
            "error_code": current.error_code if error_code is None else str(error_code),
            "message": current.message if message is None else str(message),
            "actual_registry_epoch": current.actual_registry_epoch
            if actual_registry_epoch is None
            else str(actual_registry_epoch),
            "actual_registry_generation": current.actual_registry_generation
            if actual_registry_generation is None
            else int(actual_registry_generation),
            "actual_registry_digest": current.actual_registry_digest
            if actual_registry_digest is None
            else str(actual_registry_digest),
        }
        if current.stop_requested and state not in TERMINAL_STATES:
            state = "STOPPING"
        now = _utc_now()
        connection.execute(
            """
            UPDATE executions
               SET state=?, completed_step_count=?, error_code=?, message=?,
                   actual_registry_epoch=?, actual_registry_generation=?,
                   actual_registry_digest=?, updated_at=?
             WHERE execution_id=?
            """,
            (
                state,
                values["completed_step_count"],
                values["error_code"],
                values["message"],
                values["actual_registry_epoch"],
                values["actual_registry_generation"],
                values["actual_registry_digest"],
                now,
                execution_id,
            ),
        )
        self._append_event_in_transaction(
            connection,
            execution_id,
            state,
            values["completed_step_count"],
            values["error_code"],
            values["message"],
        )
        return self._get_in_transaction(connection, execution_id)

    def _append_event_in_transaction(
        self,
        connection: sqlite3.Connection,
        execution_id: str,
        state: str,
        completed_step_count: int,
        error_code: str,
        message: str,
    ) -> None:
        row = connection.execute(
            "SELECT last_event_sequence FROM executions WHERE execution_id=?", (execution_id,)
        ).fetchone()
        if row is None:
            raise KeyError(execution_id)
        sequence = int(row["last_event_sequence"]) + 1
        created_at = _utc_now()
        connection.execute(
            """
            INSERT INTO execution_events (
                execution_id, sequence, state, completed_step_count, error_code, message, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (execution_id, sequence, state, completed_step_count, error_code, message, created_at),
        )
        connection.execute(
            "UPDATE executions SET last_event_sequence=?, updated_at=? WHERE execution_id=?",
            (sequence, created_at, execution_id),
        )
        self._pending_event_ids.append(execution_id)

    def _get_in_transaction(self, connection: sqlite3.Connection, execution_id: str) -> ExecutionRecord:
        row = connection.execute("SELECT * FROM executions WHERE execution_id=?", (execution_id,)).fetchone()
        if row is None:
            raise KeyError(execution_id)
        return self._record(row)

    @staticmethod
    def _record(row: sqlite3.Row) -> ExecutionRecord:
        return ExecutionRecord(
            execution_id=str(row["execution_id"]),
            request_id=str(row["request_id"]),
            caller_scope=str(row["caller_scope"]),
            robot_id=str(row["robot_id"]),
            program_digest=str(row["program_digest"]),
            program_json=str(row["program_json"]),
            workflow_digest=str(row["workflow_digest"]),
            state=str(row["state"]),
            completed_step_count=int(row["completed_step_count"]),
            stop_requested=bool(row["stop_requested"]),
            stop_generation=int(row["stop_generation"]),
            last_event_sequence=int(row["last_event_sequence"]),
            error_code=str(row["error_code"]),
            message=str(row["message"]),
            actual_registry_epoch=str(row["actual_registry_epoch"]),
            actual_registry_generation=int(row["actual_registry_generation"]),
            actual_registry_digest=str(row["actual_registry_digest"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def _transaction(self):
        return _Transaction(self)


class _Transaction:
    def __init__(self, store: SQLiteExecutionStore) -> None:
        self._store = store
        self._connection = store._connection
        self._lock = store._lock

    def __enter__(self) -> sqlite3.Connection:
        self._lock.acquire()
        self._connection.execute("BEGIN IMMEDIATE")
        return self._connection

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        committed_ids: list[str] = []
        listener = None
        try:
            if exc_type is None:
                self._connection.commit()
                committed_ids, listener = self._store._take_committed_event_ids()
            else:
                self._connection.rollback()
                self._store._discard_pending_event_ids()
        finally:
            self._lock.release()
        if listener is not None and committed_ids:
            # Delivery is best effort: the ledger stays authoritative and
            # GetExecutionEvents replays anything a subscriber missed.
            with contextlib.suppress(Exception):
                listener(committed_ids)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "ExecutionConflictError",
    "ExecutionBusyError",
    "ExecutionEventRecord",
    "ExecutionRecord",
    "ExecutionQuarantinedError",
    "SQLiteExecutionStore",
    "TERMINAL_STATES",
]
