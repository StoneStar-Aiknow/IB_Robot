from embodied_common.agent_terminal_contract import classify_agent_terminal, stable_agent_execution_error_code


def _expectation():
    return {
        "plan_id": "plan-1",
        "plan_digest": "digest-1",
        "registry_epoch": "epoch-1",
        "registry_generation": 1,
        "registry_digest": "registry-1",
        "step_count": 1,
    }


def _result(**updates):
    result = {
        "success": False,
        "plan_id": "plan-1",
        "plan_digest": "digest-1",
        "completed_step_count": 0,
        "error_code": "CAPABILITY_NOT_READY",
        "actual_registry_epoch": "epoch-1",
        "actual_registry_generation": 1,
        "actual_registry_digest": "registry-1",
    }
    result.update(updates)
    return result


def test_classifier_requires_exact_identity_for_failure():
    result = _result(plan_id="", plan_digest="", actual_registry_epoch="", actual_registry_generation=0)
    result["actual_registry_digest"] = ""

    assert classify_agent_terminal(6, result, _expectation()) is None


def test_classifier_preserves_uncertain_motion_class():
    assert classify_agent_terminal(6, _result(error_code="GATEWAY_FINALIZATION_FAILED"), _expectation()) == "unknown"


def test_stable_error_mapping_is_shared():
    assert stable_agent_execution_error_code("CANCEL_CLEANUP_TIMEOUT") == "SKILL_CANCEL_TIMEOUT"
    assert stable_agent_execution_error_code("NEW_EXECUTOR_VERSION_MISMATCH") == "SKILL_EXECUTOR_IDENTITY_MISMATCH"
    assert stable_agent_execution_error_code("unexpected") == "CAPABILITY_NOT_READY"
