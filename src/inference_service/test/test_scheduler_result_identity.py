from __future__ import annotations

from types import SimpleNamespace

from inference_service.scheduler.result_identity import result_identity_error


def test_matching_result_identity_is_accepted():
    result = SimpleNamespace(session_id="s1", session_generation=7, request_id="r1", pipeline_id="p1")
    assert not result_identity_error(
        "dispatch",
        result,
        {
            "session_id": "s1",
            "session_generation": 7,
            "request_id": "r1",
            "pipeline_id": "p1",
        },
    )


def test_result_identity_mismatch_is_reported():
    result = SimpleNamespace(session_id="other")
    assert result_identity_error("dispatch", result, {"session_id": "s1"}) == "dispatch_session_id_mismatch"


def test_close_requires_a_higher_drained_generation():
    result = SimpleNamespace(
        session_id="s1",
        pipeline_id="p1",
        closed_session_generation=7,
        drained_generation=7,
    )
    assert (
        result_identity_error(
            "close",
            result,
            {"session_id": "s1", "pipeline_id": "p1", "closed_session_generation": 7},
            require_higher_drained_generation=True,
        )
        == "close_drained_generation_invalid"
    )
