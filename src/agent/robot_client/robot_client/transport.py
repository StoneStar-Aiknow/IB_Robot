"""Neutral Gateway transport contract for the public robot client.

The client library never reads robot resources or discovers targets. A caller
that owns a deployment resolves the robot configuration at its own boundary and
hands the already-loaded ``embodied`` mapping to :func:`gateway_transport`.
This module only applies the documented endpoint defaults and derives the
canonical identity every request is scoped to.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from embodied_common.canon import sha256_text, to_canonical_json

#: Endpoint parameter names understood by the ROS transport and their types.
#: A complete map prevents transport defaults from addressing a different
#: deployment; advertising an endpoint does not require it to be ready.
ENDPOINT_TYPES = {
    "status_service": "ibrobot_msgs/srv/GetSkillGatewayStatus",
    "snapshot_service": "ibrobot_msgs/srv/GetSkillSnapshot",
    "reload_service": "ibrobot_msgs/srv/ReloadSkillCatalog",
    "validate_skill_service": "ibrobot_msgs/srv/ValidateSkill",
    "skill_action": "ibrobot_msgs/action/SkillCommand",
    "primitive_action": "ibrobot_msgs/action/PrimitiveCommand",
    "validate_primitive_service": "ibrobot_msgs/srv/ValidatePrimitive",
    "plan_service": "ibrobot_msgs/srv/PlanAgentCommand",
    "prepare_plan_service": "ibrobot_msgs/srv/PrepareAgentPlan",
    "validate_plan_service": "ibrobot_msgs/srv/ValidateAgentPlan",
    "confirm_plan_service": "ibrobot_msgs/srv/ConfirmAgentPlan",
    "execute_plan_action": "ibrobot_msgs/action/ExecuteAgentPlan",
    "execute_program_action": "ibrobot_msgs/action/ExecuteProgram",
    "get_execution_service": "ibrobot_msgs/srv/GetExecution",
    "get_execution_events_service": "ibrobot_msgs/srv/GetExecutionEvents",
    "cancel_execution_service": "ibrobot_msgs/srv/CancelExecution",
    "start_visual_game_service": "ibrobot_msgs/srv/StartVisualGame",
    "get_visual_game_result_service": "ibrobot_msgs/srv/GetVisualGameResult",
}

#: ``bridge_kwargs`` parameter names that differ from the transport field name.
_ENDPOINT_ALIASES = {"skill_action": "skill_action_name", "primitive_action": "primitive_action_name"}


@dataclass(frozen=True)
class GatewayTransport:
    """Resolved, non-secret Gateway endpoints for one deployment."""

    status_service: str
    validate_skill_service: str
    skill_action_name: str
    primitive_action_name: str = "/embodied/execute_primitive"
    validate_primitive_service: str = "/embodied/validate_primitive"
    snapshot_service: str = "/embodied/get_skill_snapshot"
    reload_service: str = "/embodied/reload_skill_catalog"
    plan_service: str = "/embodied/plan_agent_command"
    validate_plan_service: str = "/embodied/validate_agent_plan"
    confirm_plan_service: str = "/embodied/confirm_agent_plan"
    execute_plan_action: str = "/embodied/execute_agent_plan"
    start_visual_game_service: str = "/embodied/start_visual_game"
    get_visual_game_result_service: str = "/embodied/get_visual_game_result"
    execute_program_action: str = "/embodied/execute_program"
    get_execution_service: str = "/embodied/get_execution"
    get_execution_events_service: str = "/embodied/get_execution_events"
    cancel_execution_service: str = "/embodied/cancel_execution"
    robot_id: str = ""
    robot_name: str = ""
    caller_scope: str = "robot_skill_cli"

    @property
    def endpoints(self) -> dict[str, str]:
        """Return the normalized endpoint map used for identity and bridging."""
        return {
            key: getattr(self, _ENDPOINT_ALIASES.get(key, key))
            for key in ENDPOINT_TYPES
            if hasattr(self, _ENDPOINT_ALIASES.get(key, key))
        }

    @property
    def target_name(self) -> str:
        """Return the portion of the identity describing which robot this is."""
        return self.robot_name or self.robot_id

    def identity_digest(self) -> str:
        """Return a canonical digest of the deployment this transport addresses.

        Endpoints, robot identity and caller scope all participate, so two
        deployments that resolve the same file to different endpoints, or that
        share endpoints but route different robots, stay distinguishable.
        """
        return sha256_text(
            to_canonical_json(
                {
                    "schema_version": 1,
                    "robot_id": self.robot_id,
                    "robot_name": self.target_name,
                    "caller_scope": self.caller_scope,
                    "endpoints": self.endpoints,
                }
            )
        )


def gateway_transport(
    embodied: Mapping[str, Any] | None,
    *,
    robot_id: str = "",
    robot_name: str = "",
    caller_scope: str = "robot_skill_cli",
) -> GatewayTransport:
    """Project an already-loaded ``embodied`` config mapping into a transport.

    This function performs no file IO and imports no configuration package:
    the caller decides which robot config is deployed, the defaults below are
    the same ones the launch files use for the Gateway nodes.
    """
    values = dict(embodied or {})
    return GatewayTransport(
        status_service=values.get("skill_gateway_status_service", "/embodied/get_skill_gateway_status"),
        snapshot_service=values.get("skill_catalog_snapshot_service", "/embodied/get_skill_snapshot"),
        reload_service=values.get("skill_catalog_reload_service", "/embodied/reload_skill_catalog"),
        validate_skill_service=values.get("validate_skill_service", "/embodied/validate_skill"),
        skill_action_name=values.get("skill_action_name", "/embodied/execute_skill"),
        primitive_action_name=values.get("primitive_action_name", "/embodied/execute_primitive"),
        validate_primitive_service=values.get("validate_primitive_service", "/embodied/validate_primitive"),
        plan_service=values.get("plan_service", "/embodied/plan_agent_command"),
        validate_plan_service=values.get("validate_plan_service", "/embodied/validate_agent_plan"),
        confirm_plan_service=values.get("confirm_plan_service", "/embodied/confirm_agent_plan"),
        execute_plan_action=values.get("execute_plan_action", "/embodied/execute_agent_plan"),
        start_visual_game_service=values.get("start_visual_game_service", "/embodied/start_visual_game"),
        get_visual_game_result_service=values.get("get_visual_game_result_service", "/embodied/get_visual_game_result"),
        execute_program_action=values.get("execute_program_action", "/embodied/execute_program"),
        get_execution_service=values.get("get_execution_service", "/embodied/get_execution"),
        get_execution_events_service=values.get("get_execution_events_service", "/embodied/get_execution_events"),
        cancel_execution_service=values.get("cancel_execution_service", "/embodied/cancel_execution"),
        robot_id=robot_id,
        robot_name=robot_name,
        caller_scope=caller_scope,
    )
