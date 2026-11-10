from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass

HTTP_SESSION_COOKIE = "_robot_skill_web_session"
CSRF_COOKIE = "_robot_skill_web_csrf"
SESSION_TTL_SEC = 8 * 60 * 60
MAX_CONNECTION_FAILURES = 5
CONNECTION_WINDOW_SEC = 60.0


@dataclass
class Session:
    token_hash: str
    csrf: str
    expires_at: float


class SessionStore:
    """Local browser sessions and session-establishment rate limits, without user keys."""

    def __init__(self, *, now=time.time) -> None:
        self._sessions: dict[str, Session] = {}
        self._now = now
        self._lock = threading.RLock()
        self._failed_attempts: deque[float] = deque()

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def issue_csrf(self) -> str:
        return secrets.token_urlsafe(32)

    def check_connection_attempt(self) -> None:
        with self._lock:
            now = self._now()
            while self._failed_attempts and now - self._failed_attempts[0] >= CONNECTION_WINDOW_SEC:
                self._failed_attempts.popleft()
            if len(self._failed_attempts) >= MAX_CONNECTION_FAILURES:
                raise PermissionError("too many failed connection attempts")

    def record_connection_failure(self) -> None:
        with self._lock:
            self._failed_attempts.append(self._now())

    def issue_session(self) -> str:
        """Issue a browser session for the deployment the server is bound to."""
        with self._lock:
            token = secrets.token_urlsafe(32)
            session = Session(self._hash(token), secrets.token_urlsafe(32), self._now() + SESSION_TTL_SEC)
            # A new browser session replaces browser access, not the bound target or its tasks.
            self._sessions = {token: session}
            return token

    def lookup(self, token: str) -> Session:
        with self._lock:
            now = self._now()
            self._sessions = {key: value for key, value in self._sessions.items() if value.expires_at > now}
            session = self._sessions.get(token)
            if session is None or not hmac.compare_digest(session.token_hash, self._hash(token)):
                raise PermissionError("browser session is not active")
            return session

    def close(self, token: str | None) -> None:
        with self._lock:
            if token is not None:
                self._sessions.pop(token, None)
