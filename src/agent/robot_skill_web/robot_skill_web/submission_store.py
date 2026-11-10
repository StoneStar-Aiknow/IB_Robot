"""Durable Web submission records with conservative ledger reconciliation.

The base :class:`~robot_skill_web.serialization.SubmissionStore` owns the
private on-disk association file and the atomic "reserve once" rule.  This
module adds the Web adapter's pending-submission lifecycle on top of it:

``submitting``
    The request was recorded before the client started its pre-goal checks.
    No goal may have been sent yet.
``pending``
    The client persisted the reference immediately before sending the goal,
    but no receipt/ledger observation has associated an execution yet.
``active``
    The ledger is known to have accepted the goal (or the execution snapshot
    reports a nonterminal execution).  This is still busy.
``terminal``
    The ledger or the Gateway explicitly reported SUCCEEDED/FAILED/CANCELED,
    or the goal was rejected before it could be sent, or startup recovery
    proved from the fsynced ``goal_may_be_sent=False`` flag that no goal was
    sent (``NOT_SENT``, not a physical cancellation).

Only a read-only observation may move a record forward.  A timeout, a missing
snapshot, an unavailable event service, a late nonterminal reply and a
conflicting reference all leave the record unresolved; none of them unlocks a
second goal.
"""

from __future__ import annotations

import copy
from typing import Any

from robot_client.connection import write_private_json
from robot_client.errors import ClientError

from .serialization import SUBMISSION_TERMINAL_STATES, SubmissionStore

TERMINAL_EXECUTION_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELED"})
IDENTITY_FIELDS = ("request_id", "target_key", "robot_id", "caller_scope")
ASSOCIATION_FIELDS = ("execution_id", "legacy_task_id")
MAX_FAILURE_TEXT = 4000
#: The durable reason written when startup recovery proves that no goal was
#: sent.  ``REJECTED`` is the existing terminal admission state; this code is
#: what distinguishes "never sent" from a physical cancellation or a Gateway
#: rejection.
NOT_SENT_CODE = "NOT_SENT"
NOT_SENT_MESSAGE = "the Web process exited before the goal was sent; no goal was sent"


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _field(value: Any, name: str) -> Any:
    """Read a mapping or a message-like object from the transport."""
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _is_terminal_record(record: dict[str, Any]) -> bool:
    state = _text(record.get("state"))
    return state in SUBMISSION_TERMINAL_STATES or (state == "UNKNOWN" and record.get("reconciled") is True)


def _provably_unsent(record: dict[str, Any]) -> bool:
    """Return whether the saved record itself proves the goal was not sent.

    ``mark_goal_may_be_sent`` fsyncs ``goal_may_be_sent=True`` before the client
    may call ``send_program_goal``, so an explicit boolean ``False`` in the
    ``submitting`` phase is durable evidence that the send boundary was never
    crossed.  A missing field (legacy record), a defaulted false or any
    non-boolean value is ambiguous evidence and must keep the conservative
    lock instead of releasing admission.
    """
    return record.get("phase") == "submitting" and record.get("goal_may_be_sent") is False


def _association_value(record: dict[str, Any], field: str) -> str:
    """Read a mutable execution association, falling back to legacy records."""
    reference = record.get("reference")
    saved = _text(reference.get(field)) if isinstance(reference, dict) else ""
    if saved:
        return saved
    association = record.get("association")
    return _text(association.get(field)) if isinstance(association, dict) else ""


def _association_conflict(stored: dict[str, Any], incoming: Any) -> str | None:
    if not isinstance(incoming, dict):
        return None
    for field in ASSOCIATION_FIELDS:
        saved = _association_value(stored, field)
        supplied = _text(incoming.get(field))
        if saved and supplied and saved != supplied:
            return f"execution reference {field} conflicts with the saved association"
    return None


def _reference_conflict(stored: dict[str, Any], incoming: Any) -> str | None:
    """Return a stable conflict description for one reference, or ``None``.

    Immutable request identity is never overwritten.  An execution identifier
    is a mutable association: empty means "not linked yet", while two different
    nonempty identifiers are a conflict and must not be reconciled.
    """
    if not isinstance(incoming, dict):
        return "execution reference is not an object"
    for field in IDENTITY_FIELDS:
        before, after = _text(stored.get(field)), _text(incoming.get(field))
        if before and after and before != after:
            return f"execution reference {field} differs from the saved submission"
    for field in ASSOCIATION_FIELDS:
        before, after = _text(stored.get(field)), _text(incoming.get(field))
        if before and after and before != after:
            return f"execution reference {field} conflicts with the saved association"
    return None


def _same_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not _text(left.get("request_id")) or not _text(right.get("request_id")):
        return False
    return all(_text(left.get(field)) == _text(right.get(field)) for field in IDENTITY_FIELDS)


def _sequence(event: Any) -> int:
    value = _field(event, "sequence")
    return value if type(value) is int else 0


def _event_list(events: Any) -> list[Any]:
    if not isinstance(events, dict):
        return []
    raw = events.get("events")
    return raw if isinstance(raw, list) else []


def _terminal_events(events: Any) -> tuple[str, bool, str] | None:
    """Return ``(state, reconciled, conflict)`` for terminal event states."""
    by_state: dict[str, int] = {}
    reconciled = False
    for event in _event_list(events):
        state = _text(_field(event, "state"))
        event_type = _text(_field(event, "event_type")) or _text(_field(event, "type"))
        if state in TERMINAL_EXECUTION_STATES:
            by_state[state] = max(by_state.get(state, 0), _sequence(event))
        elif state == "UNKNOWN" and "RECONCIL" in event_type.upper():
            by_state["UNKNOWN"] = max(by_state.get("UNKNOWN", 0), _sequence(event))
            reconciled = True
    if not by_state:
        return None
    if len(by_state) > 1:
        return "", False, "ledger events disagree on the terminal state"
    return next(iter(by_state)), reconciled, ""


def _event_association(events: Any) -> tuple[set[str], str, str]:
    """Return execution ids, the latest nonterminal state and a conflict."""
    if not isinstance(events, dict) or events.get("unavailable"):
        return set(), "", ""
    ids = {
        _text(_field(event, "execution_id")) for event in _event_list(events) if _text(_field(event, "execution_id"))
    }
    if events.get("found") is not False and _text(events.get("execution_id")):
        ids.add(_text(events.get("execution_id")))
    if len(ids) > 1:
        return ids, "", "ledger events refer to multiple executions"
    latest_state = ""
    latest_sequence = -1
    for event in _event_list(events):
        sequence = _field(event, "sequence")
        if type(sequence) is not int:
            continue
        state = _text(_field(event, "state"))
        if state and state not in TERMINAL_EXECUTION_STATES and sequence >= latest_sequence:
            latest_sequence = sequence
            latest_state = state
    return ids, latest_state, ""


class PendingSubmissionStore(SubmissionStore):
    """SubmissionStore plus conservative pending/active/terminal reconciliation."""

    def reserve_intent(
        self,
        preview_id: str,
        session_id: str,
        reference: dict[str, Any],
        *,
        command: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record the client identity before any pre-goal check starts.

        The reference is the request the client will persist immediately
        before sending the goal.  Reserving it first makes a lost acceptance
        response observable even if the Web process dies before the client
        calls back.
        """
        with self._lock:
            record = self.reserve(preview_id, session_id, reference, command=command)
            stored = self._records[preview_id]
            if "phase" not in stored and "goal_may_be_sent" not in stored:
                stored["phase"] = "submitting"
                stored["goal_may_be_sent"] = False
                write_private_json(self._path(preview_id), stored)
                self._records[preview_id] = stored
                record = copy.deepcopy(stored)
            return record

    def mark_goal_may_be_sent(self, preview_id: str, reference: dict[str, Any]) -> dict[str, Any]:
        """Persist the client's pre-send reference and freeze its identity."""
        with self._lock:
            self._load()
            stored = self._records.get(preview_id)
            if stored is None:
                raise ClientError("INVALID_LOCAL_STATE", "submission intent was not reserved")
            if _is_terminal_record(stored):
                raise ClientError("INVALID_LOCAL_STATE", "submission was already finalized")
            if not _text(reference.get("request_id")) or not _text(reference.get("target_key")):
                raise ClientError("INVALID_REFERENCE", "submission requires request_id and target_key")
            conflict = _reference_conflict(stored["reference"], reference)
            if conflict is None:
                conflict = _association_conflict(stored, reference)
            if conflict is not None:
                raise ClientError("EXECUTION_IDENTITY_CONFLICT", conflict, reference=copy.deepcopy(stored["reference"]))
            for field in ASSOCIATION_FIELDS:
                if _text(reference.get(field)):
                    stored.setdefault("association", {})[field] = reference[field]
            stored["phase"] = "pending"
            stored["goal_may_be_sent"] = True
            stored.pop("failure", None)
            write_private_json(self._path(preview_id), stored)
            self._records[preview_id] = stored
            return copy.deepcopy(stored)

    def mark_accepted(self, preview_id: str, result: Any) -> None:
        """Record a successful send without treating accepted as completion."""
        with self._lock:
            self._load()
            stored = self._records.get(preview_id)
            if stored is None or _is_terminal_record(stored):
                return
            if isinstance(result, dict):
                conflict = _reference_conflict(stored["reference"], result)
                if conflict is None:
                    conflict = _association_conflict(stored, result)
                if conflict is not None:
                    stored["failure"] = {
                        "code": "EXECUTION_IDENTITY_CONFLICT",
                        "message": conflict[:MAX_FAILURE_TEXT],
                    }
                    write_private_json(self._path(preview_id), stored)
                    self._records[preview_id] = stored
                    return
                for field in ASSOCIATION_FIELDS:
                    if _text(result.get(field)):
                        stored.setdefault("association", {})[field] = result[field]
                execution_state = _text(result.get("state"))
                if execution_state:
                    stored["execution_state"] = execution_state
            stored["phase"] = "accepted"
            stored["goal_may_be_sent"] = True
            stored.pop("failure", None)
            write_private_json(self._path(preview_id), stored)
            self._records[preview_id] = stored

    def mark_failure(
        self,
        preview_id: str,
        *,
        error_code: str,
        message: str,
        rejected: bool,
    ) -> None:
        """Record a failed submission; only a provably unsent goal is terminal."""
        with self._lock:
            self._load()
            stored = self._records.get(preview_id)
            if stored is None or _is_terminal_record(stored):
                return
            stored["failure"] = {
                "code": _text(error_code)[:256],
                "message": _text(message)[:MAX_FAILURE_TEXT],
            }
            if rejected:
                stored["state"] = "REJECTED"
                stored["reconciled"] = False
                stored["phase"] = "terminal"
                self._unknown.discard(preview_id)
            else:
                stored["phase"] = "pending"
            write_private_json(self._path(preview_id), stored)
            self._records[preview_id] = stored

    def recover_unsent(
        self,
        *,
        target_key: str,
        exclude_request_ids: Any = None,
    ) -> list[str]:
        """Release only the intents this deployment provably never sent.

        Called once while the Web process starts, before it can expose a
        submission worker.  A record is released only when all durable and
        process-local evidence agrees: it still owns admission, it belongs to
        this deployment, it is explicitly in the ``submitting`` phase, its
        ``goal_may_be_sent`` field is the boolean ``False`` written before the
        send boundary, and no live worker of this process owns it.  Everything
        else keeps the conservative lock.

        The record keeps its saved identity and evidence; only the admission
        state moves to the existing terminal ``REJECTED`` with a ``NOT_SENT``
        reason, so history and a late observation can explain the release
        instead of guessing.  This never sends, cancels, deletes or re-links a
        goal.
        """
        excluded = {
            request_id for request_id in (exclude_request_ids or ()) if isinstance(request_id, str) and request_id
        }
        recovered: list[str] = []
        with self._lock:
            self._load()
            for preview_id in sorted(self._records):
                record = self._records[preview_id]
                if preview_id not in self._unknown:
                    continue
                reference = record.get("reference")
                if not isinstance(reference, dict) or _text(reference.get("target_key")) != target_key:
                    continue
                if _text(reference.get("request_id")) in excluded:
                    continue
                if not _provably_unsent(record):
                    continue
                stored = copy.deepcopy(record)
                stored["state"] = "REJECTED"
                stored["reconciled"] = False
                stored["phase"] = "terminal"
                stored["failure"] = {"code": NOT_SENT_CODE, "message": NOT_SENT_MESSAGE}
                # ``goal_may_be_sent=False`` stays as the durable evidence for
                # this outcome; it is not a physical cancellation.
                write_private_json(self._path(preview_id), stored)
                self._records[preview_id] = stored
                self._unknown.discard(preview_id)
                recovered.append(preview_id)
        return recovered

    def adopt_if_unresolved(self, reference: Any, adopt: Any) -> bool:
        """Adopt one saved reference only while its record is still unresolved.

        Running the recheck under the store lock makes "rejection already
        persisted" atomic with ``mark_failure``: a late observation can no
        longer read a nonterminal copy and then adopt it after the rejection
        was written.  Callers must additionally exclude submissions this
        process still owns a live worker for; a provably unsent intent is never
        adopted here either because startup recovery will release it.
        """
        if not isinstance(reference, dict):
            return False
        with self._lock:
            stored = self.find_for_reference(reference)
            if stored is None:
                return False
            if _is_terminal_record(stored) or _provably_unsent(stored):
                return False
            preview_id = _text(stored.get("preview_id"))
            if preview_id and preview_id not in self._unknown:
                return False
            adopt(copy.deepcopy(stored["reference"]))
            return True

    def by_request(self, request_id: str, *, reference: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Look up by request identity, tolerating a later association update."""
        with self._lock:
            self._load()
            for record in self._records.values():
                saved = record["reference"]
                if _text(saved.get("request_id")) != request_id:
                    continue
                if reference is None or _same_identity(saved, reference):
                    return copy.deepcopy(record)
        return None

    def find_for_reference(self, reference: Any) -> dict[str, Any] | None:
        """Find a saved submission by request identity, not by mutable link.

        A browser keeps the reference it received at submit time.  Once the
        ledger associates an ``execution_id``, that original reference still
        has to match the same record, while a deliberately different nonempty
        identifier is rejected as a conflict.
        """
        if not isinstance(reference, dict):
            return None
        request_id = _text(reference.get("request_id"))
        target_key = _text(reference.get("target_key"))
        if not request_id or not target_key:
            return None
        with self._lock:
            self._load()
            candidates = [
                record
                for record in self._records.values()
                if _text(record["reference"].get("request_id")) == request_id
                and _text(record["reference"].get("target_key")) == target_key
            ]
            if not candidates:
                return None
            if len(candidates) > 1:
                raise ClientError(
                    "EXECUTION_IDENTITY_CONFLICT",
                    "multiple saved submissions share this request identity",
                )
            stored = candidates[0]
            conflict = _reference_conflict(stored["reference"], reference)
            if conflict is None:
                # A caller that supplies a nonempty association is claiming a
                # specific execution.  It must match the saved link exactly;
                # an unlinked saved record is not a wildcard for any id.
                for field in ASSOCIATION_FIELDS:
                    supplied = _text(reference.get(field))
                    saved = _association_value(stored, field)
                    if supplied and supplied != saved:
                        conflict = f"execution reference {field} conflicts with the saved association"
                        break
            if conflict is not None:
                raise ClientError(
                    "EXECUTION_IDENTITY_CONFLICT",
                    conflict,
                    reference=copy.deepcopy(stored["reference"]),
                )
            return copy.deepcopy(stored)

    def find_by_request(
        self,
        request_id: str,
        *,
        target_key: str,
        robot_id: str,
        caller_scope: str,
    ) -> dict[str, Any] | None:
        """Return an existing submission for the same request identity."""
        if not request_id:
            return None
        with self._lock:
            self._load()
            matches = [
                record
                for record in self._records.values()
                if _text(record["reference"].get("request_id")) == request_id
                and _text(record["reference"].get("target_key")) == target_key
                and _text(record["reference"].get("robot_id")) == robot_id
                and _text(record["reference"].get("caller_scope")) == caller_scope
            ]
            if len(matches) > 1:
                raise ClientError(
                    "EXECUTION_IDENTITY_CONFLICT",
                    "multiple saved submissions share this request identity",
                )
            return copy.deepcopy(matches[0]) if matches else None

    def reconcile(
        self,
        preview_id: str,
        execution: Any,
        events: Any,
        *,
        reference: Any,
    ) -> dict[str, Any]:
        """Apply one read-only observation to a saved pending submission.

        The record only moves forward.  A terminal result is final; a stale
        nonterminal acknowledgement cannot reopen it, and conflicting
        association identifiers raise instead of overwriting the saved link.

        A missing snapshot is normalized to an empty one: admission requires an
        explicit ``found`` snapshot, so ``None``, ``{}`` and ``found=False``
        are all "not observed" and none of them may release admission or be
        reported as a cancellation.
        """
        if not isinstance(execution, dict):
            execution = {}
        if not isinstance(events, dict):
            events = {}
        with self._lock:
            self._load()
            stored = copy.deepcopy(self._records.get(preview_id))
            if stored is None:
                return {"conflict": "saved submission is missing", "terminal": False}
            conflict = _reference_conflict(stored["reference"], reference)
            if conflict is not None:
                raise ClientError(
                    "EXECUTION_IDENTITY_CONFLICT",
                    conflict,
                    reference=copy.deepcopy(stored["reference"]),
                )
            for field in ASSOCIATION_FIELDS:
                supplied = _text(reference.get(field)) if isinstance(reference, dict) else ""
                saved = _association_value(stored, field)
                if supplied and supplied != saved:
                    raise ClientError(
                        "EXECUTION_IDENTITY_CONFLICT",
                        f"execution reference {field} conflicts with the saved association",
                        reference=copy.deepcopy(stored["reference"]),
                    )
            if _is_terminal_record(stored):
                return {
                    "conflict": None,
                    "state": stored["state"],
                    "terminal": True,
                    "reconciled": bool(stored.get("reconciled")),
                    "execution_id": _association_value(stored, "execution_id"),
                    "phase": "terminal",
                }

            execution_found = bool(_field(execution, "found"))
            execution_state = _text(_field(execution, "state"))
            execution_id = _text(_field(execution, "execution_id")) if execution_found else ""
            execution_reconciled = bool(_field(execution, "reconciled"))
            event_ids, event_state, event_conflict = _event_association(events)
            if event_conflict:
                return {"conflict": event_conflict, "terminal": False}
            observed_ids = event_ids | ({execution_id} if execution_id else set())
            if len(observed_ids) > 1:
                return {"conflict": "observations refer to multiple executions", "terminal": False}
            observed_id = next(iter(observed_ids), "")
            saved_id = _association_value(stored, "execution_id")
            if saved_id and observed_id and saved_id != observed_id:
                raise ClientError(
                    "EXECUTION_IDENTITY_CONFLICT",
                    "observed execution_id conflicts with the saved association",
                    reference=copy.deepcopy(stored["reference"]),
                )

            terminal_state = ""
            reconciled = False
            if execution_found and (
                execution_state in TERMINAL_EXECUTION_STATES or (execution_state == "UNKNOWN" and execution_reconciled)
            ):
                terminal_state = execution_state
                reconciled = execution_reconciled
            event_terminal = _terminal_events(events)
            if event_terminal is not None:
                event_terminal_state, event_reconciled, event_conflict = event_terminal
                if event_conflict:
                    return {"conflict": event_conflict, "terminal": False}
                if terminal_state and event_terminal_state and terminal_state != event_terminal_state:
                    return {"conflict": "execution and ledger events disagree on the terminal state", "terminal": False}
                if not terminal_state:
                    terminal_state = event_terminal_state
                    reconciled = event_reconciled

            # Cache the observation first; the base class owns the bounded,
            # deduplicated event page and its terminal-archive guard.
            self.observe(preview_id, execution, events)
            stored = copy.deepcopy(self._records.get(preview_id))
            if stored is None:
                return {"conflict": "saved submission disappeared", "terminal": False}
            if observed_id and not _association_value(stored, "execution_id"):
                stored.setdefault("association", {})["execution_id"] = observed_id
            if execution_state:
                stored["execution_state"] = execution_state
            elif event_state:
                stored["execution_state"] = event_state

            if terminal_state:
                stored["state"] = terminal_state
                stored["reconciled"] = bool(reconciled) if terminal_state == "UNKNOWN" else False
                stored["phase"] = "terminal"
                stored["goal_may_be_sent"] = True
                self._unknown.discard(preview_id)
            else:
                stored["state"] = "unknown"
                if execution_state or event_state:
                    stored["phase"] = "active"
                elif stored.get("phase") not in {"submitting", "pending"}:
                    stored.setdefault("phase", "pending")
                self._unknown.add(preview_id)
            write_private_json(self._path(preview_id), stored)
            self._records[preview_id] = stored
            return {
                "conflict": None,
                "state": stored["state"],
                "terminal": bool(terminal_state),
                "reconciled": bool(stored.get("reconciled")),
                "execution_id": _association_value(stored, "execution_id"),
                "phase": stored.get("phase"),
            }

    def resolve(self, preview_id: str, state: str, *, reconciled: bool = False) -> None:
        """Apply an explicit resolution, but never downgrade a terminal record."""
        with self._lock:
            self._load()
            stored = self._records.get(preview_id)
            if stored is not None and _is_terminal_record(stored):
                return
        super().resolve(preview_id, state, reconciled=reconciled)

    def unresolved(self) -> list[dict[str, Any]]:
        records = super().unresolved()
        for record in records:
            record.setdefault(
                "phase",
                "submitting" if record.get("goal_may_be_sent") is False else "pending",
            )
            failure = record.get("failure")
            if isinstance(failure, dict) and "error_code" not in record:
                record["error_code"] = _text(failure.get("code"))
        return records
