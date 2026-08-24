import pytest

from object_tracker import SessionState, SingleTargetSession


def test_session_preserves_identity_across_search_and_reacquisition():
    sessions = SingleTargetSession()
    started = sessions.start("object-1", navigation_ready=True, map_ready=True)
    sessions.confirm(started.session_id)
    sessions.begin_search(started.session_id, "quality gate failed")
    reacquired = sessions.reacquire(started.session_id)

    assert reacquired.session_id == started.session_id
    assert reacquired.object_id == "object-1"
    assert reacquired.state is SessionState.TRACKING


def test_session_rejects_second_active_target_and_unknown_id():
    sessions = SingleTargetSession()
    started = sessions.start("object-1", navigation_ready=True, map_ready=True)

    with pytest.raises(RuntimeError, match="already exists"):
        sessions.start("object-2", navigation_ready=True, map_ready=True)
    with pytest.raises(KeyError, match="unknown"):
        sessions.stop("wrong-session")
    sessions.stop(started.session_id)


def test_session_fails_closed_when_dependencies_are_unready():
    sessions = SingleTargetSession()

    with pytest.raises(RuntimeError, match="navigation-ready"):
        sessions.start("object-1", navigation_ready=False, map_ready=True)
    with pytest.raises(RuntimeError, match="map contract"):
        sessions.start("object-1", navigation_ready=True, map_ready=False)
