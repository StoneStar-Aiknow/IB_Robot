"""ROS-independent loading and querying of robot skill capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_common.capability_view import project_capability_timeout_policy
from embodied_common.dispatch_binding import delegated_executor_identity, load_delegated_model_identity
from embodied_common.mirror_session_contract import SESSION_SERVICE as MIRROR_SESSION_SERVICE
from embodied_common.primitive_contracts import primitive_contract_for_version
from embodied_common.visual_game_contracts import build_visual_game_capability_view

# Neutral online projection and helper functions are owned by robot_client and
# must not drift from the public client implementation.
from robot_client.catalog import (
    UnknownGameError as UnknownGameError,
)
from robot_client.catalog import (
    UnknownSkillError as UnknownSkillError,
)
from robot_client.catalog import (
    capability_view_from_snapshot as capability_view_from_snapshot,
)
from robot_client.catalog import (
    describe_game as describe_game,
)
from robot_client.catalog import (
    describe_skill as describe_skill,
)
from robot_client.catalog import (
    list_games as list_games,
)
from robot_client.catalog import (
    list_poses as list_poses,
)
from robot_client.catalog import (
    list_skills as list_skills,
)
from robot_client.catalog import (
    planner_catalog_view as planner_catalog_view,
)
from robot_client.catalog import (
    require_enabled_game as require_enabled_game,
)
from robot_client.transport import (
    GatewayTransport as GatewayTransport,
)
from robot_client.transport import (
    gateway_transport as gateway_transport,
)
from robot_config.config_path import resolve_robot_config_path
from robot_config.loader import (
    get_effective_visual_game_policies,
    load_robot_config_dict,
    load_transport_robot_config,
    robot_config_digest,
    robot_context_schema_version,
    robot_execution_endpoints,
    robot_supported_control_modes,
)
from robot_config.timeout_policy import resolve_embodied_timeout_policy
from skill_catalog.compiler import compile_skill_catalog
from skill_catalog.models import DelegatedExecutorDescriptor, SkillCompileContext, SkillRobotContext
from skill_catalog.source import (
    AmentShareSkillSource,
    DevelopmentStagingSkillSource,
    DirectoryReleaseSkillSource,
    resolve_development_source_root,
)


@dataclass(frozen=True)
class CatalogContext:
    view: dict[str, Any]
    game_view: dict[str, Any]


def _game_view(robot_config: dict[str, Any], timeout_policy: dict[str, float]) -> dict[str, Any]:
    embodied = robot_config.get("embodied", {})
    games = get_effective_visual_game_policies(robot_config)
    return build_visual_game_capability_view(
        str(robot_config.get("name", "")),
        games,
        timeout_sec=timeout_policy["visual_game_timeout_sec"],
        result_retention_sec=timeout_policy["visual_game_result_retention_sec"],
        result_capacity=embodied.get("visual_game_result_capacity", 128),
        start_service=embodied.get("start_visual_game_service", "/embodied/start_visual_game"),
        result_service=embodied.get("get_visual_game_result_service", "/embodied/get_visual_game_result"),
        event_topic=embodied.get("visual_game_event_topic", "/embodied/visual_game_events"),
    )


def load_catalog_context(
    *,
    config_name: str | None = None,
    config_path: str | Path | None = None,
) -> CatalogContext:
    """Load one normalized config for public catalog use (ROS-independent)."""
    resolved_path = resolve_robot_config_path(config_name=config_name, config_path=config_path)
    robot_config = load_robot_config_dict(resolved_path)
    embodied = robot_config.get("embodied", {})
    timeout_policy = resolve_embodied_timeout_policy(embodied)
    return CatalogContext(
        view=_snapshot_capability_view(compile_local_snapshot(robot_config, resolved_path)),
        game_view=_game_view(robot_config, timeout_policy),
    )


def load_runtime_context(
    *,
    config_name: str | None = None,
    config_path: str | Path | None = None,
) -> tuple[CatalogContext, GatewayTransport]:
    """Load both catalog view and ROS transport for runtime Gateway commands.

    Provider-bound robot configs require the live interface description; pass
    the launch-generated consumer snapshot (``/tmp/ibrobot_interfaces_*/robot.yaml``)
    as ``config_path`` when driving a running pipeline.
    """
    resolved_path = resolve_robot_config_path(config_name=config_name, config_path=config_path)
    robot_config = load_robot_config_dict(resolved_path)
    embodied = robot_config.get("embodied", {})
    timeout_policy = resolve_embodied_timeout_policy(embodied)
    return (
        CatalogContext(
            view=_snapshot_capability_view(compile_local_snapshot(robot_config, resolved_path)),
            game_view=_game_view(robot_config, timeout_policy),
        ),
        _gateway_transport(embodied, robot_id=str(robot_config.get("name", ""))),
    )


#: Gateway commands that read the live Gateway status/snapshot and the
#: projected timeout policy, never a locally compiled skill view. Ledger
#: queries and cancellation must not depend on local robot resources being
#: intact; prepare/execute validate the live snapshot's schema and identity, so
#: they must not fail locally with ``description_required`` either.
TRANSPORT_ONLY_COMMANDS = frozenset(
    {
        "status",
        "cancel",
        "reload-catalog",
        "validate",
        "execute",
        "plan-workflow",
        "run-workflow",
        "validate-plan",
        "confirm-plan",
        "execute-plan",
        "cancel-plan",
    }
)


#: Catalog inspection commands that read the live Gateway snapshot in online
#: mode. They bind only the public transport contract (never the local skill
#: catalog or the runtime interface description) and project the exact snapshot
#: the Gateway serves. ``--offline`` deliberately keeps the strict local loader
#: so development trees are still validated before use.
ONLINE_CATALOG_COMMANDS = frozenset({"list-skills", "describe", "list-poses"})


def load_transport_context(
    *,
    config_name: str | None = None,
    config_path: str | Path | None = None,
) -> tuple[CatalogContext, GatewayTransport]:
    """Load the Gateway transport and timeout policy WITHOUT compiling a catalog.

    The compatibility path's ledger, prepare/execute and workflow commands
    (see :data:`TRANSPORT_ONLY_COMMANDS`) and the online catalog commands (see
    :data:`ONLINE_CATALOG_COMMANDS`) need the bound endpoints, the timeout
    policy and the live Gateway status/snapshot, never a locally compiled
    skill view. They also must not depend on the runtime interface description,
    a local model directory or a launch snapshot: only the public ``embodied``
    transport contract is read (``load_transport_robot_config``). A deployment
    with logical runtime interfaces therefore binds the same Gateway endpoints
    as its launch. Mirrors the visual-game loader's split between metadata and
    motion dependencies; ``--offline`` catalog inspection and the visual-game
    commands keep their own strict loaders.
    """
    projection = load_transport_robot_config(config_name=config_name, config_path=config_path)
    embodied = projection.embodied
    timeout_policy = resolve_embodied_timeout_policy(embodied)
    return (
        CatalogContext(
            # The same projection the full context's view carries, so the
            # timeout fields these commands read are identical.
            view={
                "robot_name": projection.name,
                "timeout_policy": project_capability_timeout_policy(timeout_policy),
            },
            game_view=_game_view({"name": projection.name, "embodied": embodied}, timeout_policy),
        ),
        _gateway_transport(embodied, robot_id=projection.name),
    )


def load_visual_game_context(
    *,
    config_name: str | None = None,
    config_path: str | Path | None = None,
) -> CatalogContext:
    """Load visual-game metadata without compiling the motion Skill catalog."""
    resolved_path = resolve_robot_config_path(config_name=config_name, config_path=config_path)
    robot_config = load_robot_config_dict(resolved_path)
    return _visual_game_context(robot_config)


def load_visual_game_runtime_context(
    *,
    config_name: str | None = None,
    config_path: str | Path | None = None,
) -> tuple[CatalogContext, GatewayTransport]:
    """Load visual-game metadata and ROS transport without motion dependencies."""
    resolved_path = resolve_robot_config_path(config_name=config_name, config_path=config_path)
    robot_config = load_robot_config_dict(resolved_path)
    embodied = robot_config.get("embodied", {})
    return _visual_game_context(robot_config), _gateway_transport(embodied, robot_id=str(robot_config.get("name", "")))


def _visual_game_context(robot_config: dict[str, Any]) -> CatalogContext:
    embodied = robot_config.get("embodied", {})
    timeout_policy = resolve_embodied_timeout_policy(embodied)
    return CatalogContext(
        view={"robot_name": robot_config["name"], "timeout_policy": timeout_policy},
        game_view=_game_view(robot_config, timeout_policy),
    )


def _gateway_transport(embodied: dict[str, Any], *, robot_id: str = "") -> GatewayTransport:
    return gateway_transport(embodied, robot_id=robot_id, robot_name=robot_id)


def compile_local_snapshot(robot_config: dict[str, Any], config_path: Path):
    """Compile the profile selected by one normalized robot config."""
    embodied = robot_config["embodied"]
    execution = embodied.get("execution", {})
    source_mode = embodied.get("skill_catalog_source_mode", "installed")
    source_root = embodied.get("skill_catalog_source_root", "")
    if source_mode == "development":
        development_root = resolve_development_source_root(source_root, config_path)
        if development_root is None:
            source = AmentShareSkillSource()
        else:
            source = DevelopmentStagingSkillSource(development_root)
    elif source_mode == "production":
        source = DirectoryReleaseSkillSource(Path(source_root))
    else:
        source = AmentShareSkillSource()
    # Keep the ROS-independent CLI catalog context aligned with the runtime
    # SkillExecutorNode.  Delegated executors are selected by the enabled
    # execution configuration, not by one particular profile name; otherwise
    # the PC grasp profile cannot validate its own ``grasp_pipeline`` entry.
    delegated = {}
    grasp_execution = robot_config.get("grasp_execution", {})
    if grasp_execution.get("enabled", False):
        descriptor = DelegatedExecutorDescriptor(
            **delegated_executor_identity(
                name="grasp_pipeline",
                endpoint_name=grasp_execution.get("action_name", "/manipulation/execute_pick"),
                configuration=grasp_execution,
                **load_delegated_model_identity(grasp_execution),
            )
        )
        delegated[descriptor.name] = descriptor

    placement_execution = robot_config.get("placement_execution", {})
    if placement_execution.get("enabled", False):
        descriptor = DelegatedExecutorDescriptor(
            **delegated_executor_identity(
                name="placement_pipeline",
                endpoint_name=placement_execution.get("action_name", "/manipulation/execute_place"),
                configuration=placement_execution,
            )
        )
        delegated[descriptor.name] = descriptor

    hri_runtime = embodied.get("imitate_human_motion", {})
    if isinstance(hri_runtime, dict) and hri_runtime.get("enabled", False):
        descriptor = DelegatedExecutorDescriptor(
            **delegated_executor_identity(
                name="imitate_human_motion",
                endpoint_name=hri_runtime.get("action_name", "/hri/imitate_human_motion"),
                configuration={"implementation": "mock_v1"},
            )
        )
        delegated[descriptor.name] = descriptor

    sound_orientation = embodied.get("idle_behaviors", {}).get("sound_orientation", {})
    if sound_orientation.get("enabled", False) and sound_orientation.get("mode", "keyword") == "periodic":
        descriptor = DelegatedExecutorDescriptor(
            **delegated_executor_identity(
                name="sound_following",
                endpoint_name="/sound_orientation_node/set_following",
                endpoint_kind="ros_service",
                configuration={},
            )
        )
        delegated[descriptor.name] = descriptor

    named_motion = embodied.get("runtime_named_motion", {})
    if isinstance(named_motion, dict) and named_motion.get("enabled") is True:
        # Mirror SkillExecutorNode: the endpoint and runtime identity come from
        # the bound public description when this config snapshot carries one.
        runtime = robot_config.get("runtime", {}) if isinstance(robot_config.get("runtime"), dict) else {}
        description = runtime.get("interface_description") or {}
        interface = (description.get("interfaces") or {}).get(named_motion.get("interface", "")) or {}
        descriptor = DelegatedExecutorDescriptor(
            **delegated_executor_identity(
                name="runtime_named_motion",
                endpoint_name=interface.get("endpoint", "/motion/execute_named"),
                endpoint_kind="ros_action",
                configuration={
                    "runtime_name": (description.get("robot") or {}).get("runtime_name", runtime.get("provider", ""))
                },
            )
        )
        delegated[descriptor.name] = descriptor

    mirror_session = robot_config.get("mirror_session", {})
    if isinstance(mirror_session, dict) and mirror_session.get("enabled") is True:
        descriptor = DelegatedExecutorDescriptor(
            **delegated_executor_identity(
                name="mirror_session",
                endpoint_name=MIRROR_SESSION_SERVICE,
                endpoint_kind="ros_service",
                configuration={},
            )
        )
        delegated[descriptor.name] = descriptor

    semantic_mapping = robot_config.get("semantic_mapping", {})
    if isinstance(semantic_mapping, dict) and semantic_mapping.get("enabled", False):
        interfaces = semantic_mapping.get("interfaces", {})
        target_service = interfaces.get("query_service", "/semantic_mapping/get_objects")
        target_watch = semantic_mapping.get("target_watch", {})
        stand_off_distance_m = target_watch.get("stand_off_distance_m", 0.3)
        if isinstance(target_service, str) and target_service.strip():
            configuration = {
                "query_service": target_service,
                "stand_off_distance_m": float(stand_off_distance_m),
            }
            descriptor = DelegatedExecutorDescriptor(
                **delegated_executor_identity(
                    name="semantic_map_query",
                    endpoint_name=target_service,
                    endpoint_kind="ros_service",
                    configuration=configuration,
                )
            )
            delegated[descriptor.name] = descriptor
    robot_context = SkillRobotContext(
        robot_name=robot_config["name"],
        context_schema_version=robot_context_schema_version(robot_config),
        robot_config_digest=robot_config_digest(robot_config),
        named_poses=embodied.get("named_poses", {}),
        named_targets=embodied.get("named_targets", {}),
        arm_joint_names=tuple(robot_config.get("joints", {}).get("arm", [])),
        joint_limits=robot_config.get("teleoperation", {}).get("safety", {}).get("joint_limits", {}),
        workspace_limits=embodied.get("safety", {}).get("workspace", {}),
        required_control_mode=robot_config["skill_required_control_mode"],
        timeout_policy=project_capability_timeout_policy(resolve_embodied_timeout_policy(embodied)),
        relative_motion_reference_frame=execution.get("relative_motion_reference_frame", "base"),
        relative_motion_step_m=execution.get("relative_motion_step_m", 0.03),
        relative_motion_direction_mapping=execution.get("relative_motion_direction_mapping", {}),
        gripper_open_position=execution.get("gripper_open_position", 1.0),
        gripper_closed_position=execution.get("gripper_closed_position", 0.0),
        execution_endpoints=robot_execution_endpoints(robot_config),
        supported_control_modes=robot_supported_control_modes(robot_config),
    )
    primitive_contract = primitive_contract_for_version(robot_context.context_schema_version)
    return compile_skill_catalog(
        source,
        profile_name=embodied["skill_catalog_profile"],
        context=SkillCompileContext(
            robot=robot_context,
            primitive_contracts=primitive_contract.descriptors,
            primitive_contract_digest=primitive_contract.digest,
            delegated_executors=delegated,
        ),
    )


def _snapshot_capability_view(snapshot):
    def thaw(value):
        if isinstance(value, Mapping):
            return {str(key): thaw(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [thaw(item) for item in value]
        return value

    skills = [thaw(snapshot.capability_view[name]) for name in sorted(snapshot.capability_view)]
    for skill in skills:
        skill["timeout_sec"] = float(snapshot.templates[skill["name"]]["timeout_sec"])
    return {
        "robot_name": snapshot.robot_name,
        "skills": skills,
        "pose_names": sorted(snapshot.robot_context.named_poses),
        "timeout_policy": project_capability_timeout_policy(snapshot.robot_context.timeout_policy),
        "capability_digest": snapshot.capability_digest,
        "profile_name": snapshot.profile_name,
    }


def load_capability_catalog(
    *,
    config_name: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the normalized config and return its public capability view."""
    return load_catalog_context(config_name=config_name, config_path=config_path).view
