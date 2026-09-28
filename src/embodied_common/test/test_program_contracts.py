import pytest

from embodied_common.program_contracts import canonicalize_program


def _program(**overrides):
    values = {
        "schema_version": 1,
        "request_id": "request-1",
        "caller_scope": "local-agent",
        "robot_id": "so101",
        "expected_registry_epoch": "epoch",
        "expected_registry_generation": 2,
        "expected_registry_digest": "digest",
        "workflow_steps": [{"schema_version": 1, "skill_name": "wave_hello"}],
        "task_budget_sec": 30.0,
    }
    values.update(overrides)
    return canonicalize_program(**values)


def test_program_digest_covers_execution_payload_but_not_raw_command():
    first = _program()
    assert first == _program()
    assert first.program_digest != _program(task_budget_sec=31.0).program_digest
    assert first.program_digest != _program(robot_id="other").program_digest
    assert (
        first.program_digest
        != _program(workflow_steps=[{"schema_version": 1, "skill_name": "close_gripper_skill"}]).program_digest
    )


def test_program_root_task_id_is_stable_per_scoped_request():
    assert _program().root_task_id == _program().root_task_id
    assert _program().root_task_id != _program(caller_scope="hermes").root_task_id


@pytest.mark.parametrize(
    ("field", "value"),
    [("request_id", ""), ("caller_scope", ""), ("robot_id", ""), ("task_budget_sec", 0.0)],
)
def test_program_rejects_incomplete_identity(field, value):
    with pytest.raises(ValueError):
        _program(**{field: value})
