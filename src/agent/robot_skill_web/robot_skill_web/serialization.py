from __future__ import annotations

import copy
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

from robot_client.connection import read_private_json, write_private_json
from robot_client.errors import ClientError

MAX_JSON_BYTES = 128 * 1024
MAX_NESTING = 16
MAX_COLLECTION_ITEMS = 256
SUBMISSION_TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELED", "REJECTED"})


class RequestError(ValueError):
    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def parse_json_body(stream) -> Any:
    length_header = stream.headers.get("Content-Length")
    if length_header is None:
        raise RequestError("Content-Length is required")
    try:
        length = int(length_header, 10)
    except ValueError as exc:
        raise RequestError("invalid Content-Length") from exc
    if length <= 0 or length > MAX_JSON_BYTES:
        raise RequestError("request body is empty or too large", status=413)
    raw = stream.rfile.read(length)
    if len(raw) != length:
        raise RequestError("incomplete request body")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RequestError("request body is not valid JSON") from exc
    _validate_json(value, 0)
    return value


def _validate_json(value: Any, depth: int) -> None:
    if depth > MAX_NESTING:
        raise RequestError("request body is nested too deeply")
    if value is None or isinstance(value, str | bool) or type(value) in (int, float):
        if isinstance(value, float) and not math.isfinite(value):
            raise RequestError("request body contains non-finite numbers")
        return
    if isinstance(value, list):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise RequestError("request body contains too many list items")
        for item in value:
            _validate_json(item, depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise RequestError("request body contains too many object fields")
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 128:
                raise RequestError("request body contains invalid object keys")
            _validate_json(item, depth + 1)
        return
    raise RequestError("request body contains unsupported JSON values")


def require_object(value: Any, fields: set[str], *, optional: set[str] | None = None) -> dict[str, Any]:
    optional = optional or set()
    if not isinstance(value, dict):
        raise RequestError("request body must be an object")
    unknown = set(value) - fields - optional
    missing = fields - set(value)
    if unknown or missing:
        detail = []
        if missing:
            detail.append("missing: " + ", ".join(sorted(missing)))
        if unknown:
            detail.append("unsupported: " + ", ".join(sorted(unknown)))
        raise RequestError("; ".join(detail))
    return value


def validate_id(value: Any, field: str, *, max_length: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length or any(ord(char) < 32 for char in value):
        raise RequestError(f"{field} must be a nonempty printable string")
    return value


def validate_connection_version(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise RequestError("connection_version must be a positive integer")
    return value


def error_payload(exc: Exception) -> tuple[int, dict[str, Any]]:
    if isinstance(exc, RequestError):
        return exc.status, {"error": {"code": "INVALID_REQUEST", "message": str(exc)}}
    if isinstance(exc, PermissionError):
        status = 429 if "too many" in str(exc) else 401
        return status, {"error": {"code": "SESSION_REQUIRED" if status == 401 else "RATE_LIMITED", "message": str(exc)}}
    if isinstance(exc, ClientError):
        status = 409
        if exc.code in {"NOT_CONNECTED", "TARGET_NOT_FOUND"}:
            status = 404
        elif exc.code == "READ_ONLY":
            status = 403
        elif exc.code in {
            "ROS_UNAVAILABLE",
            "SERVER_UNAVAILABLE",
            "SNAPSHOT_UNAVAILABLE",
            "CONNECTION_FAILED",
        }:
            status = 503
        elif exc.code in {"INVALID_REQUEST", "INVALID_REFERENCE", "INVALID_CURSOR", "STALE_PREVIEW"}:
            status = 400
        payload = {"error": {"code": exc.code, "message": str(exc)}}
        if exc.reference is not None:
            payload["execution"] = exc.reference
        payload["observation_unknown"] = bool(exc.observation_unknown)
        return status, payload
    return 500, {"error": {"code": "INTERNAL_ERROR", "message": "internal adapter error"}}


class SubmissionStore:
    """Private, durable association records used to prevent duplicate goals."""

    def __init__(self, directory: Path | None = None) -> None:
        state_root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
        self.directory = Path(directory) if directory is not None else state_root / "ibrobot" / "robot-skill-web"
        self._lock = threading.RLock()
        self._loaded = False
        self._records: dict[str, dict[str, Any]] = {}
        self._unknown: set[str] = set()

    def _load(self) -> None:
        if self._loaded:
            return
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.directory.stat()
        if not (info.st_mode & 0o170000) == 0o040000 or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ClientError("UNSAFE_LOCAL_STATE", f"{self.directory} must be a private directory owned by this user")
        for path in self.directory.glob("*.json"):
            try:
                value = read_private_json(path)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ClientError("INVALID_LOCAL_STATE", f"submission record {path} is corrupt") from exc
            if not isinstance(value.get("preview_id"), str):
                raise ClientError("INVALID_LOCAL_STATE", f"submission record {path} is incomplete")
            preview_id = value["preview_id"]
            if not preview_id or path.name != f"{preview_id}.json":
                raise ClientError("INVALID_LOCAL_STATE", f"submission record {path} has a mismatched preview_id")
            reference = value.get("reference")
            if (
                not isinstance(reference, dict)
                or not isinstance(reference.get("request_id"), str)
                or not reference["request_id"].strip()
                or not isinstance(reference.get("target_key"), str)
                or not reference["target_key"].strip()
            ):
                raise ClientError("INVALID_LOCAL_STATE", f"submission record {path} has an invalid reference")
            state = value.get("state")
            if state in SUBMISSION_TERMINAL_STATES or (state == "UNKNOWN" and value.get("reconciled") is True):
                pass
            elif state in {"unknown", "UNKNOWN"} and value.get("reconciled", False) is False:
                self._unknown.add(preview_id)
            else:
                raise ClientError(
                    "INVALID_LOCAL_STATE", f"submission record {path} has an invalid state or reconciliation flag"
                )
            self._records[preview_id] = value
        self._loaded = True

    def _path(self, preview_id: str) -> Path:
        separators = {os.sep, os.altsep} if os.altsep else {os.sep}
        if (
            not preview_id
            or "\x00" in preview_id
            or preview_id in {".", ".."}
            or any(separator in preview_id for separator in separators)
        ):
            raise RequestError("preview_id is not a valid local record identifier")
        path = (self.directory / f"{preview_id}.json").resolve(strict=False)
        if path.parent != self.directory.resolve(strict=False):
            raise RequestError("preview_id is not a valid local record identifier")
        return path

    def get(self, preview_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._load()
            record = self._records.get(preview_id)
            return copy.deepcopy(record) if record else None

    def by_request(self, request_id: str, *, reference: dict[str, Any] | None = None) -> dict[str, Any] | None:
        with self._lock:
            self._load()
            for record in self._records.values():
                if record.get("reference", {}).get("request_id") == request_id and (
                    reference is None or record["reference"] == reference
                ):
                    return copy.deepcopy(record)
        return None

    def reserve(
        self,
        preview_id: str,
        session_id: str,
        reference: dict[str, Any],
        *,
        command: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._path(preview_id)
        if not isinstance(reference, dict) or any(
            not isinstance(reference.get(key), str) or not reference[key].strip()
            for key in ("request_id", "target_key")
        ):
            raise ClientError("INVALID_REFERENCE", "submission requires request_id and target_key")
        record = {
            "preview_id": preview_id,
            "session_id": session_id,
            "submitted_at": time.time(),
            "state": "unknown",
            "reference": copy.deepcopy(reference),
        }
        if command is not None:
            archived_command = {
                key: command.get(key, "")[:limit]
                for key, limit in (("skill_name", 256), ("display_name", 120), ("raw_command", 4000))
            }
            archived_command["parameters"] = copy.deepcopy(command.get("parameters", {}))
            # read_private_json caps records at 256 KiB. Archival detail must
            # never make a durable admission record unreadable after restart.
            if len(json.dumps(archived_command, allow_nan=False)) > 64 * 1024:
                archived_command.pop("parameters")
                archived_command["parameters_omitted"] = True
            record["command"] = archived_command
        with self._lock:
            self._load()
            existing = self._records.get(preview_id)
            if existing is not None:
                return copy.deepcopy(existing)
            write_private_json(self._path(preview_id), record)
            self._records[preview_id] = record
            self._unknown.add(preview_id)
            return copy.deepcopy(record)

    @staticmethod
    def _matches(record: dict[str, Any], robot_id: str, caller_scope: str, target_key: str) -> bool:
        reference = record["reference"]
        return (
            reference.get("robot_id") == robot_id
            and reference.get("caller_scope") == caller_scope
            and reference["target_key"] == target_key
        )

    def history(self, *, robot_id: str, caller_scope: str, target_key: str, limit: int = 100) -> dict[str, Any]:
        """Read the local archive, never query a robot or change unresolved state."""
        with self._lock:
            self._load()
            records = [
                record for record in self._records.values() if self._matches(record, robot_id, caller_scope, target_key)
            ]
            records.sort(key=lambda record: (record.get("submitted_at", 0), record["preview_id"]), reverse=True)
            summaries = []
            for record in records[:limit]:
                summary = {
                    key: copy.deepcopy(record[key])
                    for key in ("preview_id", "submitted_at", "state", "reconciled", "reference", "observed_at")
                    if key in record
                }
                command = record.get("command", {})
                summary["command"] = {
                    key: copy.deepcopy(command[key]) for key in ("skill_name", "display_name") if key in command
                }
                summaries.append(summary)
            return {"records": summaries, "total": len(records), "limit": limit}

    def history_record(
        self, preview_id: str, *, robot_id: str, caller_scope: str, target_key: str
    ) -> dict[str, Any] | None:
        with self._lock:
            self._load()
            record = self._records.get(preview_id)
            if record is None or not self._matches(record, robot_id, caller_scope, target_key):
                return None
            return copy.deepcopy({key: value for key, value in record.items() if key != "session_id"})

    def observe(self, preview_id: str, execution: dict[str, Any], events: dict[str, Any]) -> None:
        """Cache observations for inspection without treating them as new admission decisions."""
        with self._lock:
            self._load()
            record = copy.deepcopy(self._records.get(preview_id))
            if record is None:
                return
            terminal = record["state"] in SUBMISSION_TERMINAL_STATES or record.get("reconciled") is True
            if terminal and (
                execution.get("state") != record["state"]
                or (record.get("reconciled") is True and not execution.get("reconciled"))
            ):
                return  # A delayed nonterminal response must not overwrite a cached terminal result.
            previous = record.get("observation", {}).get("events", {})
            merged = {
                event["sequence"]: copy.deepcopy(event)
                for event in [*previous.get("events", []), *events.get("events", [])]
                if isinstance(event, dict) and type(event.get("sequence")) is int
            }

            trimmed = False

            def fields(value, names):
                nonlocal trimmed
                trimmed = trimmed or any(
                    key in names and isinstance(item, str) and len(item) > (4000 if key == "message" else 256)
                    for key, item in value.items()
                )
                return {
                    key: item[: 4000 if key == "message" else 256] if isinstance(item, str) else copy.deepcopy(item)
                    for key, item in value.items()
                    if key in names
                }

            cached_execution = fields(
                execution,
                {"found", "state", "execution_id", "error_code", "message", "reconciled", "submission_rejected"},
            )
            cached_events = fields(events, {"unavailable", "error_code", "message", "has_more", "last_event_sequence"})
            event_fields = {"sequence", "execution_id", "state", "event_type", "type", "message"}
            cached_events["events"] = [fields(merged[sequence], event_fields) for sequence in sorted(merged)[-100:]]
            cached_events["event_gap"] = bool(previous.get("event_gap") or events.get("event_gap"))
            cached_events["truncated"] = bool(previous.get("truncated") or len(merged) > 100 or trimmed)
            record["observed_at"] = time.time()
            record["observation"] = {"execution": cached_execution, "events": cached_events}
            while len(json.dumps(record["observation"], allow_nan=False)) > 96 * 1024 and cached_events["events"]:
                cached_events["events"].pop(0)
                cached_events["truncated"] = True
            write_private_json(self._path(preview_id), record)
            self._records[preview_id] = record

    def unresolved(self) -> list[dict[str, Any]]:
        with self._lock:
            self._load()
            return [copy.deepcopy(self._records[key]) for key in self._unknown if key in self._records]

    def resolve(self, preview_id: str, state: str, *, reconciled: bool = False) -> None:
        with self._lock:
            self._load()
            record = copy.deepcopy(self._records.get(preview_id))
            if record is not None:
                record["state"] = state
                record["reconciled"] = bool(reconciled) if state == "UNKNOWN" else False
                write_private_json(self._path(preview_id), record)
                self._records[preview_id] = record
            if state in SUBMISSION_TERMINAL_STATES or (state == "UNKNOWN" and reconciled):
                self._unknown.discard(preview_id)
