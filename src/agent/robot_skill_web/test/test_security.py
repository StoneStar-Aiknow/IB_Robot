from __future__ import annotations

import pytest

from robot_skill_web.security import (
    CONNECTION_WINDOW_SEC,
    MAX_CONNECTION_FAILURES,
    SESSION_TTL_SEC,
    SessionStore,
)


def test_new_session_revokes_the_previous_browser_session():
    store = SessionStore(now=lambda: 0)
    first = store.issue_session()
    first_csrf = store.lookup(first).csrf
    second = store.issue_session()
    assert second != first
    assert store.lookup(second).csrf != first_csrf
    with pytest.raises(PermissionError):
        store.lookup(first)


def test_rejected_session_establishment_does_not_revoke_active_session():
    store = SessionStore()
    token = store.issue_session()
    store.check_connection_attempt()
    store.record_connection_failure()
    assert store.lookup(token)


def test_expired_session_can_be_reestablished_without_server_restart():
    now = [0.0]
    store = SessionStore(now=lambda: now[0])
    token = store.issue_session()
    now[0] = SESSION_TTL_SEC
    with pytest.raises(PermissionError):
        store.lookup(token)
    renewed = store.issue_session()
    assert store.lookup(renewed).expires_at == now[0] + SESSION_TTL_SEC


def test_global_connection_rate_limit_expires_without_restarting_server():
    now = [0.0]
    store = SessionStore(now=lambda: now[0])
    for _ in range(MAX_CONNECTION_FAILURES):
        store.check_connection_attempt()
        store.record_connection_failure()
    with pytest.raises(PermissionError, match="too many"):
        store.check_connection_attempt()
    now[0] = CONNECTION_WINDOW_SEC
    store.check_connection_attempt()
    assert store.lookup(store.issue_session())


def test_closed_session_allows_a_new_session():
    store = SessionStore()
    token = store.issue_session()
    store.close(token)
    with pytest.raises(PermissionError):
        store.lookup(token)
    assert store.lookup(store.issue_session())


def test_lookup_requires_the_exact_session_token():
    store = SessionStore()
    token = store.issue_session()
    with pytest.raises(PermissionError):
        store.lookup(token + "x")
    with pytest.raises(PermissionError):
        store.lookup("")
    assert store.lookup(token)
