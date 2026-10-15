import json

from robot_skill_cli.output import (
    EXIT_GATEWAY_REJECTED,
    EXIT_INVALID_INPUT,
    EXIT_ROS_UNAVAILABLE,
    EXIT_SIGINT,
    EXIT_SIGTERM,
    EXIT_SUCCESS,
    EXIT_TIMEOUT,
    error_envelope,
    feedback_event,
    json_dumps,
    result_event,
    success_envelope,
)


def test_json_output_is_sorted_compact_and_ascii():
    payload = {"z": "夹爪", "a": {"b": 1}}

    assert json_dumps(payload) == '{"a":{"b":1},"z":"\\u5939\\u722a"}'


def test_command_envelopes_have_stable_shape():
    success = success_envelope("status", {"ready": True})
    failure = error_envelope("describe", "UNKNOWN_SKILL", "unknown skill: missing")

    assert success == {
        "schema_version": 1,
        "ok": True,
        "command": "status",
        "data": {"ready": True},
        "error": None,
    }
    assert failure == {
        "schema_version": 1,
        "ok": False,
        "command": "describe",
        "data": None,
        "error": {"code": "UNKNOWN_SKILL", "message": "unknown skill: missing"},
    }


def test_feedback_and_result_events_expose_only_public_progress():
    feedback = feedback_event("task-1", "abc123", "executing", "step 1 of 2")
    result = result_event(
        "task-1",
        "abc123",
        success=True,
        error_code="",
        message="completed",
        executed_step_count=2,
    )

    assert feedback == {
        "schema_version": 1,
        "event": "feedback",
        "task_id": "task-1",
        "payload_hash": "abc123",
        "data": {"state": "executing", "detail": "step 1 of 2"},
    }
    assert result == {
        "schema_version": 1,
        "event": "result",
        "task_id": "task-1",
        "payload_hash": "abc123",
        "data": {
            "success": True,
            "error_code": "",
            "message": "completed",
            "executed_step_count": 2,
        },
    }
    encoded = json.dumps(result)
    assert "primitive" not in encoded
    assert "executed_primitives" not in encoded


def test_exit_codes_are_stable():
    assert {
        EXIT_SUCCESS,
        EXIT_INVALID_INPUT,
        EXIT_GATEWAY_REJECTED,
        EXIT_ROS_UNAVAILABLE,
        EXIT_TIMEOUT,
        EXIT_SIGINT,
        EXIT_SIGTERM,
    } == {0, 2, 3, 4, 124, 130, 143}
