from types import SimpleNamespace

import pytest

from embodied_common.dispatch_binding import new_binding
from embodied_common.workflow_lifecycle import WorkflowLifecycleClient, WorkflowLifecycleError


def _client(response):
    calls = []

    def call_service(client, request, service):
        calls.append((client, request, service))
        return response(request) if callable(response) else response

    return WorkflowLifecycleClient(call_service, "begin-client", "/begin", "finalize-client", "/finalize"), calls


def test_begin_validates_digest_and_freezes_lease_nonce():
    binding = new_binding(task_id="task-1")
    binding.workflow_digest = "digest"
    lifecycle, calls = _client(
        SimpleNamespace(success=True, error_code="", message="", root_lease_nonce="nonce", workflow_digest="digest")
    )

    returned = lifecycle.begin(binding, [])

    assert returned.root_lease_nonce == "nonce"
    assert calls[0][2] == "/begin"


def test_finalize_preserves_gateway_rejection_response():
    lifecycle, _calls = _client(
        SimpleNamespace(success=False, error_code="SKILL_WORKFLOW_STEP_MISMATCH", message="wrong step")
    )

    response = lifecycle.finalize(new_binding(task_id="task-1"), 1, 0)

    assert response.success is False
    assert response.error_code == "SKILL_WORKFLOW_STEP_MISMATCH"


def test_begin_maps_transport_exception_without_code_to_unknown_state():
    def fail(_request):
        raise RuntimeError("transport failed")

    lifecycle, _calls = _client(fail)

    with pytest.raises(WorkflowLifecycleError) as raised:
        lifecycle.begin(new_binding(task_id="task-1"), [])

    assert raised.value.code == "SKILL_CANCEL_TIMEOUT"
