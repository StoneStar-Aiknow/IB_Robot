"""The Gateway call behind a double cover: ``mirror_start``. No ROS calls.

The node makes the calls; this fills the requests and checks the answers,
written after ``robot_skill_cli`` (status, payload, validate) and
``embodied_agent.sound_orientation_node`` (goal), without importing either:

1. ``GetSkillGatewayStatus``: ``mirror_start`` must be a ready capability;
   remember the registry identity (epoch, generation, digest) and build the
   canonical payload (schema 1, ``timeout_sec`` 5.0).
2. ``ValidateSkill`` with that payload and identity: it must be allowed, and
   the registry identity it validated against must be the one from step 1
   (an empty one does not match), else nothing is sent.
3. ``SkillCommand`` goal ``mirror_start``, goal id ``skill_goal_uuid(task_id)``;
   the result must be a success.

``task_budget`` is left empty: the Gateway fills it for a root goal.
Every refusal raises ``GatewayError`` with the Gateway's error code.
"""

from __future__ import annotations

import uuid
from typing import Any

from embodied_common.dispatch_binding import new_binding
from embodied_common.mirror_session_contract import OPERATOR_NODE_NAME
from embodied_common.skill_request import canonical_skill_payload, skill_goal_uuid, skill_payload_hash

SKILL_NAME = "mirror_start"
SCHEMA_VERSION = 1
TIMEOUT_SEC = 5.0
# From the double cover to the goal result: status, validate and the goal itself.
RESULT_WAIT_SEC = 8.0
REGISTRY_MISMATCH = "SKILL_REGISTRY_VERSION_MISMATCH"


class GatewayError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}" if message else code)


class MirrorStartRequest:
    """One ``mirror_start`` attempt; a new double cover makes a new one."""

    def __init__(self, task_id: str = ""):
        self.task_id = task_id or f"{OPERATOR_NODE_NAME}-{uuid.uuid4()}"
        self.identity: tuple[str, int, str] | None = None
        self.payload: dict[str, Any] | None = None
        self.payload_hash = ""

    @property
    def goal_uuid(self) -> list[int]:
        return list(skill_goal_uuid(self.task_id).bytes)

    def fill_status_request(self, request: Any) -> Any:
        request.schema_version = 1
        request.task_id = ""
        request.payload_hash = ""
        return request

    def accept_status(self, response: Any) -> None:
        capability = next((item for item in response.capabilities if item.name == SKILL_NAME), None)
        if capability is None or not capability.ready:
            reason = str(capability.reason) if capability is not None and capability.reason else ""
            code = reason.partition(":")[0].strip() or "CAPABILITY_NOT_READY"
            raise GatewayError(code, reason or f"{SKILL_NAME} is not a ready capability")
        identity = (
            str(response.registry_epoch),
            int(response.registry_generation),
            str(response.registry_digest),
        )
        if not identity[0] or not identity[2]:
            raise GatewayError(REGISTRY_MISMATCH, "Gateway status has no registry identity")
        payload = canonical_skill_payload(
            SKILL_NAME,
            schema_version=SCHEMA_VERSION,
            motion_distance=0.0,  # as robot_skill_cli: the request field is 0.0, so the hash is of 0.0
            timeout_sec=TIMEOUT_SEC,
            default_timeout_sec=float(response.default_skill_timeout_sec),
        )
        self.identity = identity
        self.payload = payload
        self.payload_hash = skill_payload_hash(payload)

    def _binding(self) -> Any:
        if self.identity is None or self.payload is None:
            raise RuntimeError("accept_status first")
        binding = new_binding(task_id=self.task_id)
        binding.expected_registry_epoch, binding.expected_registry_generation, binding.expected_registry_digest = (
            self.identity
        )
        return binding

    def _fill(self, message: Any) -> Any:
        binding = self._binding()
        payload = self.payload
        message.schema_version = payload["schema_version"]
        message.dispatch_binding = binding
        message.skill_name = payload["skill_name"]
        message.target_name = payload["target_name"]
        message.container_name = payload["container_name"]
        message.place_name = payload["place_name"]
        message.motion_direction = payload["motion_direction"]
        message.motion_distance = float(payload["motion_distance"])
        return message

    def fill_validate_request(self, request: Any) -> Any:
        return self._fill(request)

    def accept_validation(self, response: Any) -> None:
        if not response.allowed:
            raise GatewayError(str(response.error_code) or "SAFETY_REJECTED", str(response.reason))
        actual = (
            str(response.actual_registry_epoch),
            int(response.actual_registry_generation),
            str(response.actual_registry_digest),
        )
        if actual != self.identity:
            raise GatewayError(REGISTRY_MISMATCH, f"validated against {actual}, status reported {self.identity}")

    def fill_goal(self, goal: Any) -> Any:
        self._fill(goal)
        goal.timeout_sec = float(self.payload["timeout_sec"])
        return goal

    def accept_result(self, result: Any) -> str:
        if not result.success:
            raise GatewayError(str(result.error_code) or "SKILL_FAILED", str(result.message))
        return str(result.message)
