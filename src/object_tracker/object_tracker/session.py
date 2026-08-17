"""Fail-closed single-target tracking session state."""

from dataclasses import dataclass
from enum import Enum
from uuid import uuid4


class SessionState(str, Enum):
    ACQUIRING = "acquiring"
    TRACKING = "tracking"
    SEARCHING = "searching"
    LOST = "lost"
    STOPPED = "stopped"


@dataclass(frozen=True)
class Session:
    session_id: str
    object_id: str
    state: SessionState
    reason: str


class SingleTargetSession:
    """Own one session and reject ambiguous or stale state transitions."""

    def __init__(self):
        self._session: Session | None = None

    @property
    def session(self) -> Session | None:
        return self._session

    def start(self, object_id: str, *, navigation_ready: bool, map_ready: bool) -> Session:
        if self._session is not None and self._session.state not in {SessionState.LOST, SessionState.STOPPED}:
            raise RuntimeError("an active tracking session already exists")
        if not object_id.strip():
            raise ValueError("object_id is required")
        if not navigation_ready:
            raise RuntimeError("semantic target is not navigation-ready")
        if not map_ready:
            raise RuntimeError("semantic map contract is not ready")
        self._session = Session(str(uuid4()), object_id, SessionState.ACQUIRING, "awaiting local confirmation")
        return self._session

    def confirm(self, session_id: str) -> Session:
        session = self._require(session_id)
        if session.state != SessionState.ACQUIRING:
            raise RuntimeError(f"cannot confirm session in {session.state.value} state")
        return self._replace(SessionState.TRACKING, "local identity confirmed")

    def begin_search(self, session_id: str, reason: str) -> Session:
        session = self._require(session_id)
        if session.state != SessionState.TRACKING:
            raise RuntimeError(f"cannot search from {session.state.value} state")
        return self._replace(SessionState.SEARCHING, reason)

    def reacquire(self, session_id: str) -> Session:
        session = self._require(session_id)
        if session.state != SessionState.SEARCHING:
            raise RuntimeError(f"cannot reacquire from {session.state.value} state")
        return self._replace(SessionState.TRACKING, "visual identity reacquired")

    def lose(self, session_id: str, reason: str) -> Session:
        session = self._require(session_id)
        if session.state not in {SessionState.SEARCHING, SessionState.TRACKING}:
            raise RuntimeError(f"cannot lose session in {session.state.value} state")
        return self._replace(SessionState.LOST, reason)

    def stop(self, session_id: str, reason: str = "caller requested stop") -> Session:
        self._require(session_id)
        return self._replace(SessionState.STOPPED, reason)

    def _require(self, session_id: str) -> Session:
        if self._session is None:
            raise KeyError("no tracking session exists")
        if session_id != self._session.session_id:
            raise KeyError("unknown tracking session identifier")
        return self._session

    def _replace(self, state: SessionState, reason: str) -> Session:
        assert self._session is not None
        self._session = Session(self._session.session_id, self._session.object_id, state, reason)
        return self._session
