"""Shared Begin/Finalize client for typed Workflow execution scopes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ibrobot_msgs.srv import BeginWorkflowExecution, FinalizeWorkflowExecution


class WorkflowLifecycleError(RuntimeError):
    """Stable failure from Workflow lifecycle transport or admission."""

    def __init__(self, code: str, message: str = "workflow lifecycle request failed") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class WorkflowLifecycleClient:
    """Build and validate Begin/Finalize requests while leaving orchestration to callers."""

    call_service: Callable[[Any, Any, str], Any]
    begin_client: Any
    begin_service: str
    finalize_client: Any
    finalize_service: str

    @staticmethod
    def _error_code(exc: Exception, fallback: str) -> str:
        code = str(getattr(exc, "code", "") or "")
        return code if code else fallback

    def begin(self, binding, workflow_steps: Sequence[Any]):
        request = BeginWorkflowExecution.Request()
        request.dispatch_binding = binding
        request.workflow_steps = list(workflow_steps)
        try:
            response = self.call_service(self.begin_client, request, self.begin_service)
        except Exception as exc:
            raise WorkflowLifecycleError(self._error_code(exc, "SKILL_CANCEL_TIMEOUT"), str(exc)) from exc
        if not response.success:
            raise WorkflowLifecycleError(
                str(response.error_code or "SKILL_REQUEST_ID_CONFLICT"),
                str(response.message or response.error_code or "workflow admission rejected"),
            )
        if not response.root_lease_nonce:
            raise WorkflowLifecycleError("SKILL_WORKFLOW_LEASE_MISMATCH", "workflow lease nonce is missing")
        if response.workflow_digest != binding.workflow_digest:
            raise WorkflowLifecycleError("SKILL_WORKFLOW_DIGEST_MISMATCH", "workflow digest does not match")
        binding.root_lease_nonce = response.root_lease_nonce
        return binding

    def finalize(self, binding, terminal_state: int, completed_step_count: int):
        request = FinalizeWorkflowExecution.Request()
        request.dispatch_binding = binding
        request.terminal_state = terminal_state
        request.completed_step_count = completed_step_count
        try:
            response = self.call_service(self.finalize_client, request, self.finalize_service)
        except Exception as exc:
            raise WorkflowLifecycleError(self._error_code(exc, "SKILL_CANCEL_TIMEOUT"), str(exc)) from exc
        return response
