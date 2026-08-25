"""Composable stream contracts and stream-owned diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from .contracts import ExecutionContext, ModelRequest, ModelResult, StreamState, _normalize_datetime


class StreamErrorCode(str, Enum):
    STREAM_NOT_FOUND = "stream_not_found"
    STREAM_CLOSED = "stream_closed"
    STREAM_REENTRANT = "stream_reentrant"
    STREAM_CAPACITY_EXHAUSTED = "stream_capacity_exhausted"


class StreamHandle:
    """Stable identity owned by one streaming runtime.

    The owner token is private and is used by ``ModelRuntimeHandle`` to reject
    forged handles from another runtime even when their string IDs match.
    """

    __slots__ = ("_stream_id", "_owner_token", "_closed_hint", "_state_hint")

    def __init__(self, stream_id: str) -> None:
        if not isinstance(stream_id, str) or not stream_id.strip():
            raise ValueError("stream_id must be a non-empty string")
        self._stream_id = stream_id
        self._owner_token: object | None = None
        self._closed_hint = False
        self._state_hint = StreamState.OPEN

    @property
    def stream_id(self) -> str:
        return self._stream_id

    @property
    def id(self) -> str:
        return self._stream_id

    @property
    def closed(self) -> bool:
        return self._closed_hint or self._state_hint is StreamState.CLOSED

    @property
    def state(self) -> StreamState:
        return self._state_hint

    def _bind_owner(self, owner_token: object) -> None:
        if self._owner_token is not None and self._owner_token is not owner_token:
            raise ValueError("stream handle is owned by another runtime")
        self._owner_token = owner_token

    def _belongs_to(self, owner_token: object) -> bool:
        return self._owner_token is owner_token

    def _set_state(self, state: StreamState) -> None:
        self._state_hint = state
        self._closed_hint = state is StreamState.CLOSED

    def __repr__(self) -> str:
        return f"StreamHandle(stream_id={self.stream_id!r}, state={self.state.value!r})"


@dataclass(frozen=True)
class StreamDiagnostics:
    stream_id: str
    state: StreamState
    active_step: bool
    state_bank_mode: str
    created_at: datetime
    last_step_at: datetime | None = None
    recovery_requirement: object | None = None
    last_failure_code: str | None = None

    def __post_init__(self) -> None:
        if not self.stream_id:
            raise ValueError("stream_id must be non-empty")
        if self.state_bank_mode not in {"per_stream", "runtime_exclusive"}:
            raise ValueError("invalid stream state bank mode")
        if isinstance(self.state, StreamState):
            state = self.state
        else:
            try:
                state = StreamState(str(self.state))
            except ValueError:
                try:
                    state = StreamState[str(self.state)]
                except KeyError as exc:
                    raise ValueError(f"invalid stream state: {self.state!r}") from exc
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "created_at", _normalize_datetime(self.created_at))
        if self.last_step_at is not None:
            object.__setattr__(self, "last_step_at", _normalize_datetime(self.last_step_at))

    def to_dict(self) -> dict[str, object]:
        recovery = self.recovery_requirement
        if hasattr(recovery, "to_dict"):
            recovery = recovery.to_dict()
        return {
            "stream_id": self.stream_id,
            "state": self.state.value,
            "active_step": self.active_step,
            "state_bank_mode": self.state_bank_mode,
            "created_at": self.created_at.isoformat().replace("+00:00", "Z"),
            "last_step_at": self.last_step_at.isoformat().replace("+00:00", "Z")
            if self.last_step_at is not None
            else None,
            "recovery_requirement": recovery,
            "last_failure_code": self.last_failure_code,
        }

    @property
    def active(self) -> bool:
        return self.state is not StreamState.CLOSED


@runtime_checkable
class StreamingRuntime(Protocol):
    """Compositional stream API owned and admitted by a runtime handle."""

    def open_stream(self, context: ExecutionContext) -> StreamHandle: ...

    def step(self, stream_handle: StreamHandle, request: ModelRequest, context: ExecutionContext) -> ModelResult: ...

    def reset_stream(self, stream_handle: StreamHandle, context: ExecutionContext) -> None: ...

    def close_stream(self, stream_handle: StreamHandle, context: ExecutionContext) -> None: ...


StreamingRuntimeProtocol = StreamingRuntime


__all__ = [
    "StreamDiagnostics",
    "StreamErrorCode",
    "StreamHandle",
    "StreamState",
    "StreamingRuntime",
    "StreamingRuntimeProtocol",
]
