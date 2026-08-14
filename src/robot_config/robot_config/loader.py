"""Configuration loader and validator for robot_config."""

import copy
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import yaml

from embodied_common.skill_templates import (
    SUPPORTED_PRIMITIVES,
    SUPPORTED_SKILL_EXECUTORS,
)
from robot_config.config import (
    CameraConfig,
    ContractAction,
    ContractExtensionConfig,
    ContractObservation,
    EmbodiedConfig,
    PeripheralConfig,
    RobotConfig,
    Ros2ControlConfig,
    SemanticMappingConfig,
    SkillGatewayRuntimeConfig,
    VoiceASRConfig,
    VoiceTTSConfig,
)
from robot_config.grasp_execution_config import validate_grasp_execution_config
from robot_config.observation_transport import (
    parse_observation_transport,
    validate_observation_transports,
    validate_robot_config_observation_transports,
)
from robot_config.perception_runtime_config import PerceptionRuntimeConfigError, parse_perception_runtime_config
from robot_config.placement_execution_config import validate_placement_execution_config
from robot_config.timeout_policy import resolve_embodied_timeout_policy

from .config_path import resolve_robot_config_path
from .utils import resolve_calibration_paths_from_config, resolve_ros_path

logger = logging.getLogger(__name__)

# Controlled vocabularies for the skill `description` contract exposed to Agent callers.
_VALID_MOTION_SCOPES = {"base", "shoulder", "elbow", "wrist", "gripper", "arm"}
_VALID_INTENSITIES = {"subtle", "moderate", "large"}
_SUPPORTED_CONTROL_MODES = {"teleop", "model_inference", "moveit_planning"}
_VALID_RECOVERY_POLICIES = {"never_retry", "ask_user", "recover_safe_pose"}
_PUBLIC_REQUEST_FIELDS = {"target_name", "place_name", "motion_direction", "motion_distance"}
_STRING_REQUEST_FIELDS = {"target_name", "place_name", "motion_direction"}
_VALID_MOTION_DIRECTIONS = {"forward", "backward", "left", "right", "up", "down"}
_PARAMETER_SCHEMA_FIELDS = {"type", "additionalProperties", "properties", "required"}
_STRING_PARAMETER_FIELDS = {"type", "enum", "freeform"}
_DISTANCE_PARAMETER_FIELDS = {"type", "exclusiveMinimum", "unit"}
_VALID_DISTANCE_UNITS = {"meters", "degrees"}


def _normalize_digest_value(value: Any) -> Any:
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("digest preimage mapping keys must be strings")
        return {key: _normalize_digest_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_digest_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("NaN and Infinity are not allowed in digest preimages")
        return 0.0 if value == 0.0 else value
    raise TypeError(f"unsupported type in digest preimage: {type(value).__name__}")


def _canonical_digest_json(value: Any) -> str:
    return json.dumps(
        _normalize_digest_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def robot_config_digest(robot_config: dict[str, Any]) -> str:
    """Return the digest of the closed skill execution context preimage.

    This is deliberately not a digest of the YAML document.  Catalog source
    selection, profile selection and unrelated robot configuration must not
    change the identity used by catalog consumers.
    """
    embodied = robot_config.get("embodied", {})
    execution = embodied.get("execution", {}) if isinstance(embodied, dict) else {}
    safety = embodied.get("safety", {}) if isinstance(embodied, dict) else {}
    if not isinstance(execution, dict):
        execution = {}
    if not isinstance(safety, dict):
        safety = {}
    joints = robot_config.get("joints", {})
    if not isinstance(joints, dict):
        joints = {}
    teleoperation = robot_config.get("teleoperation", {})
    if not isinstance(teleoperation, dict):
        teleoperation = {}
    teleop_safety = teleoperation.get("safety", {})
    if not isinstance(teleop_safety, dict):
        teleop_safety = {}
    preimage = {
        "context_schema_version": 1,
        "robot_name": robot_config.get("name"),
        "named_poses": embodied.get("named_poses", {}) if isinstance(embodied, dict) else {},
        "named_targets": embodied.get("named_targets", {}) if isinstance(embodied, dict) else {},
        "arm_joint_names": joints.get("arm", []),
        "joint_limits": teleop_safety.get("joint_limits", {}),
        "workspace_limits": safety.get("workspace", {}),
        "required_control_mode": robot_config.get("skill_required_control_mode"),
        "timeout_policy": resolve_embodied_timeout_policy(embodied if isinstance(embodied, dict) else {}),
        "relative_motion_reference_frame": execution.get("relative_motion_reference_frame", "base"),
        "relative_motion_step_m": execution.get("relative_motion_step_m", 0.03),
        "relative_motion_direction_mapping": execution.get("relative_motion_direction_mapping", {}),
        "gripper_open_position": execution.get("gripper_open_position", 1.0),
        "gripper_closed_position": execution.get("gripper_closed_position", 0.0),
        "execution_endpoints": {
            "skill_action": embodied.get("skill_action_name", "/embodied/execute_skill"),
            "primitive_action": embodied.get("primitive_action_name", "/embodied/execute_primitive"),
            "validate_skill_service": embodied.get("validate_skill_service", "/embodied/validate_skill"),
            "validate_primitive_service": embodied.get("validate_primitive_service", "/embodied/validate_primitive"),
            "gateway_status_service": embodied.get(
                "skill_gateway_status_service", "/embodied/get_skill_gateway_status"
            ),
            "begin_workflow_service": embodied.get("begin_workflow_service", "/embodied/begin_workflow_execution"),
            "finalize_workflow_service": embodied.get(
                "finalize_workflow_service", "/embodied/finalize_workflow_execution"
            ),
            "task_executor_action": execution.get("task_executor_action_name", "/task_executor/execute_task_plan"),
            "arm_trajectory_action": execution.get(
                "arm_trajectory_action_name", "/arm_trajectory_controller/follow_joint_trajectory"
            ),
            "move_configuration_service": execution.get(
                "move_configuration_service", "/moveit_gateway/move_to_configuration"
            ),
        },
    }
    return hashlib.sha256(_canonical_digest_json(preimage).encode("utf-8")).hexdigest()


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _required_string(section: dict[str, Any], key: str, path: str, errors: list[str]) -> None:
    if not isinstance(section.get(key), str) or not section[key].strip():
        errors.append(f"{path}.{key} must be a non-empty string")


def _active_identity(section: dict[str, Any], key: str, path: str, errors: list[str]) -> None:
    _required_string(section, key, path, errors)
    value = section.get(key)
    if isinstance(value, str) and value.startswith("REPLACE_WITH_"):
        errors.append(f"{path}.{key} must be an active identity")


def _positive_number(section: dict[str, Any], key: str, path: str, errors: list[str]) -> None:
    value = section.get(key)
    if not _is_finite_number(value) or float(value) <= 0.0:
        errors.append(f"{path}.{key} must be a finite number greater than zero")


def _positive_integer(section: dict[str, Any], key: str, path: str, errors: list[str]) -> None:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{path}.{key} must be a positive integer")


def _unit_interval(section: dict[str, Any], key: str, path: str, errors: list[str]) -> None:
    value = section.get(key)
    if not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0:
        errors.append(f"{path}.{key} must be in [0.0, 1.0]")


def validate_semantic_mapping_config(robot_config: dict[str, Any]) -> list[str]:
    """Validate the standalone semantic mapping SSOT contract."""
    config = robot_config.get("semantic_mapping")
    if config is None:
        return []
    if not isinstance(config, dict):
        return ["semantic_mapping must be a mapping"]

    errors: list[str] = []
    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        return ["semantic_mapping.enabled must be a boolean"]

    if not enabled:
        return errors

    section_names = (
        "camera",
        "slam",
        "perception",
        "persistence",
        "filtering",
        "queue",
        "lifecycle",
        "labels",
        "label_refinement",
        "target_watch",
        "interfaces",
    )
    sections: dict[str, dict[str, Any]] = {}
    for name in section_names:
        value = config.get(name)
        if not isinstance(value, dict):
            errors.append(f"semantic_mapping.{name} must be a mapping when semantic mapping is enabled")
            value = {}
        sections[name] = value

    camera = sections["camera"]
    camera_path = "semantic_mapping.camera"
    for key in ("peripheral", "mounting", "parent_frame", "rgb_topic", "depth_topic", "camera_info_topic"):
        _required_string(camera, key, camera_path, errors)
    if camera.get("mounting") != "fixed":
        errors.append("semantic_mapping.camera.mounting must be 'fixed'")
    peripheral_name = camera.get("peripheral")
    peripherals = robot_config.get("peripherals", [])
    matching = [item for item in peripherals if isinstance(item, dict) and item.get("name") == peripheral_name]
    if not matching or matching[0].get("type") != "camera":
        errors.append("semantic_mapping.camera.peripheral must reference a configured camera peripheral")
    else:
        peripheral = matching[0]
        if peripheral.get("driver") != "realsense":
            errors.append("semantic_mapping.camera.peripheral must use the realsense driver")
        if not peripheral.get("align_depth", False):
            errors.append("semantic_mapping.camera.peripheral must enable aligned depth")
        if peripheral.get("transform", {}).get("parent_frame") != camera.get("parent_frame"):
            errors.append(
                "semantic_mapping.camera.parent_frame must match the camera peripheral transform parent_frame"
            )
    if camera.get("depth_topic") == camera.get("rgb_topic"):
        errors.append("semantic_mapping.camera.depth_topic must differ from rgb_topic")

    slam = sections["slam"]
    slam_path = "semantic_mapping.slam"
    for key in (
        "global_frame",
        "cloud_map_topic",
        "active_map_hash_topic",
        "localization_ready_topic",
        "authoritative_map_odom_topic",
        "geometry_map_id",
        "coordinate_convention",
        "map_odom_authority",
    ):
        _required_string(slam, key, slam_path, errors)
    for key in ("geometry_map_hash", "localization_session_id", "calibration_id", "urdf_hash"):
        _active_identity(slam, key, slam_path, errors)
    if slam.get("global_frame") == "odom":
        errors.append("semantic_mapping.slam.global_frame must be a persistent global frame, not 'odom'")
    if slam.get("map_odom_authority") != "slam":
        errors.append("semantic_mapping.slam.map_odom_authority must be 'slam'")

    perception = sections["perception"]
    perception_path = "semantic_mapping.perception"
    mapping_backend = perception.get("mapping_backend")
    if mapping_backend not in {"service", "embedded"}:
        errors.append("semantic_mapping.perception.mapping_backend must be 'service' or 'embedded'")
    for legacy_field in (
        "model_backend",
        "model_identities",
        "sam_service",
        "ram_plus_service",
        "siglip2_service",
        "encode_text_service",
        "grounding_service",
    ):
        if legacy_field in perception:
            errors.append(
                f"semantic_mapping.perception.{legacy_field} is unsupported; bind semantic_roles to perception_services.services IDs"
            )
    _positive_number(perception, "service_wait_sec", perception_path, errors)
    if mapping_backend == "service":
        errors.extend(_validate_semantic_service_roles(robot_config, perception))
    elif mapping_backend == "embedded" and not bool(perception.get("allow_legacy_embedded", False)):
        errors.append("semantic_mapping.perception.allow_legacy_embedded must be true for the embedded backend")

    persistence = sections["persistence"]
    for key in ("database_path", "artifact_output_dir"):
        _required_string(persistence, key, "semantic_mapping.persistence", errors)

    filtering = sections["filtering"]
    filtering_path = "semantic_mapping.filtering"
    _positive_number(filtering, "depth_trunc_m", filtering_path, errors)
    for key in ("min_points", "min_mask_pixels"):
        _positive_integer(filtering, key, filtering_path, errors)
    for key in (
        "min_frame_valid_depth_ratio",
        "min_mask_area_ratio",
        "min_mask_valid_depth_ratio",
        "max_mask_overlap_ratio",
    ):
        _unit_interval(filtering, key, filtering_path, errors)
    if "ground_filter_enabled" in filtering and not isinstance(filtering["ground_filter_enabled"], bool):
        errors.append("semantic_mapping.filtering.ground_filter_enabled must be a boolean")
    if "ground_reference_frame" in filtering:
        _required_string(filtering, "ground_reference_frame", filtering_path, errors)
    if "ground_height_offset_m" in filtering:
        offset = filtering["ground_height_offset_m"]
        if isinstance(offset, bool) or not isinstance(offset, int | float):
            errors.append("semantic_mapping.filtering.ground_height_offset_m must be a number")
    for key in (
        "ground_max_bottom_clearance_m",
        "ground_max_object_height_m",
        "ground_max_footprint_m",
        "max_object_distance_m",
    ):
        if key in filtering:
            _positive_number(filtering, key, filtering_path, errors)

    queue = sections["queue"]
    queue_path = "semantic_mapping.queue"
    for key in ("sync_queue_size", "frame_capacity", "max_masks_per_batch"):
        _positive_integer(queue, key, queue_path, errors)
    if "max_masks_per_frame" in queue:
        _positive_integer(queue, "max_masks_per_frame", queue_path, errors)
    for key in ("sync_slop_sec", "tf_timeout_sec", "processing_interval_sec"):
        _positive_number(queue, key, queue_path, errors)
    if queue.get("policy") not in {"drop_oldest", "drop_newest", "backpressure"}:
        errors.append("semantic_mapping.queue.policy must be 'drop_oldest', 'drop_newest', or 'backpressure'")
    max_masks = queue.get("max_masks_per_batch")
    if isinstance(max_masks, int) and not isinstance(max_masks, bool) and max_masks > 8:
        errors.append("semantic_mapping.queue.max_masks_per_batch must be <= 8")

    lifecycle = sections["lifecycle"]
    lifecycle_path = "semantic_mapping.lifecycle"
    for key in ("association_distance_m", "stale_after_sec", "move_stability_m"):
        _positive_number(lifecycle, key, lifecycle_path, errors)
    _positive_integer(lifecycle, "move_confirmations", lifecycle_path, errors)
    for key in ("association_position_weight", "embedding_similarity_threshold"):
        _unit_interval(lifecycle, key, lifecycle_path, errors)

    labels = sections["labels"]
    labels_path = "semantic_mapping.labels"
    _unit_interval(labels, "min_confidence", labels_path, errors)
    _positive_integer(labels, "max_candidates_per_mask", labels_path, errors)
    excluded_labels = labels.get("excluded_labels")
    if not isinstance(excluded_labels, list) or any(
        not isinstance(label, str) or not label.strip() for label in excluded_labels
    ):
        errors.append(f"{labels_path}.excluded_labels must be a list of non-empty strings")

    label_refinement = sections["label_refinement"]
    refinement_path = "semantic_mapping.label_refinement"
    if not isinstance(label_refinement.get("enabled"), bool):
        errors.append(f"{refinement_path}.enabled must be a boolean")
    if label_refinement.get("enabled") is True:
        for key in ("model", "model_identity", "prompt"):
            _required_string(label_refinement, key, refinement_path, errors)
    _unit_interval(label_refinement, "min_confidence", refinement_path, errors)
    _unit_interval(label_refinement, "trigger_below_confidence", refinement_path, errors)
    _positive_integer(label_refinement, "min_observations", refinement_path, errors)

    target_watch = sections["target_watch"]
    target_watch_path = "semantic_mapping.target_watch"
    _positive_integer(target_watch, "max_attempts", target_watch_path, errors)
    for key in ("stand_off_distance_m", "clearance_m"):
        _positive_number(target_watch, key, target_watch_path, errors)
    for key in (
        "scan_profile",
        "footprint_ready_topic",
        "obstacle_map_ready_topic",
        "reachability_ready_topic",
    ):
        _required_string(target_watch, key, target_watch_path, errors)

    interfaces = sections["interfaces"]
    for key in ("semantic_map_topic", "object_cloud_topic", "query_service", "target_service"):
        _required_string(interfaces, key, "semantic_mapping.interfaces", errors)
    return errors


_SEMANTIC_SERVICE_TYPES = {
    "sam2_masks": "ibrobot_msgs/srv/GenerateMasks",
    "ram_plus_tags": "ibrobot_msgs/srv/RecognizeTags",
    "siglip2_image": "ibrobot_msgs/srv/EncodeEmbeddings",
    "siglip2_text": "ibrobot_msgs/srv/EncodeText",
    "gdino_confirmation": "ibrobot_msgs/srv/GroundingDetect",
}
_CONSTRUCTION_ROLES = frozenset({"sam2_masks", "ram_plus_tags", "siglip2_image"})


def _validate_semantic_service_roles(robot_config: dict[str, Any], perception: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    roles = perception.get("semantic_roles")
    if not isinstance(roles, dict):
        return ["semantic_mapping.perception.semantic_roles must be a mapping"]
    unknown_roles = sorted(set(roles) - set(_SEMANTIC_SERVICE_TYPES))
    if unknown_roles:
        errors.append(f"semantic_mapping.perception.semantic_roles contains unsupported roles: {unknown_roles}")
    for role in _CONSTRUCTION_ROLES | {"siglip2_text"}:
        _required_string(roles, role, "semantic_mapping.perception.semantic_roles", errors)

    try:
        runtime = parse_perception_runtime_config(robot_config)
    except PerceptionRuntimeConfigError:
        return errors
    services = {service.instance_id: service for service in runtime.services}
    bound = {}
    for role, instance_id in roles.items():
        if role not in _SEMANTIC_SERVICE_TYPES or not isinstance(instance_id, str) or not instance_id:
            continue
        service = services.get(instance_id)
        path = f"semantic_mapping.perception.semantic_roles.{role}"
        if service is None:
            errors.append(f"{path} references unknown perception service {instance_id!r}")
            continue
        if not service.enabled:
            errors.append(f"{path} references disabled perception service {instance_id!r}")
            continue
        if service.service_type != _SEMANTIC_SERVICE_TYPES[role]:
            errors.append(f"{path} must reference service type {_SEMANTIC_SERVICE_TYPES[role]}")
        if (role in _CONSTRUCTION_ROLES) != service.required:
            policy = "required" if role in _CONSTRUCTION_ROLES else "optional"
            errors.append(f"{path} must reference an enabled {policy} service")
        manifest = service.validated_manifest.manifest if service.validated_manifest else None
        identity = manifest.model.semantic_identity if manifest else None
        if identity is None:
            errors.append(f"{path} service manifest must declare model.semantic_identity")
        bound[role] = identity

    image_identity = bound.get("siglip2_image")
    text_identity = bound.get("siglip2_text")
    if image_identity is not None and text_identity is not None:
        image_embedding = image_identity.embedding
        text_embedding = text_identity.embedding
        if image_embedding is None or text_embedding is None or image_embedding != text_embedding:
            errors.append("semantic_mapping SigLIP2 image/text services must declare compatible embedding metadata")
    return errors


def _validate_skill_description(
    skill_name: str,
    description: Any,
    valid_skills: set[str],
    named_poses: dict[str, Any],
    errors: list[str],
) -> None:
    """Validate the agent-facing ``description`` contract of a skill template.

    The description is the single source of truth for how an agent (rule parser
    or Hermes/CLI) picks this skill over its near-synonyms, so structural fields
    are enforced here rather than at call time.
    """
    prefix = f"embodied.skill_templates.{skill_name}.description"
    if description is None:
        errors.append(f"{prefix} is required")
        return
    if not isinstance(description, dict):
        errors.append(f"{prefix} must be a mapping")
        return

    summary_value = description.get("summary")
    if not isinstance(summary_value, str) or not summary_value.strip():
        errors.append(f"{prefix}.summary must be a non-empty string")
    elif len(summary_value.strip()) > 120:
        errors.append(f"{prefix}.summary must be at most 120 characters")

    category_value = description.get("category")
    if not isinstance(category_value, str) or not category_value.strip():
        errors.append(f"{prefix}.category must be a non-empty string")

    when_to_use = description.get("when_to_use")
    if not isinstance(when_to_use, list) or not when_to_use:
        errors.append(f"{prefix}.when_to_use must be a non-empty list")
    elif not all(isinstance(item, str) and item.strip() for item in when_to_use):
        errors.append(f"{prefix}.when_to_use entries must be non-empty strings")

    for field in ("aliases_zh", "aliases_en", "motion_scope"):
        value = description.get(field)
        if value is None:
            continue
        if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
            errors.append(f"{prefix}.{field} must be a list of non-empty strings")
        elif field == "motion_scope":
            unknown = sorted(set(value) - _VALID_MOTION_SCOPES)
            if unknown:
                errors.append(f"{prefix}.motion_scope contains unsupported token(s): {', '.join(unknown)}")

    intensity = description.get("intensity")
    if intensity is not None:
        if not isinstance(intensity, str):
            errors.append(f"{prefix}.intensity must be a string when present")
        elif intensity.strip() not in _VALID_INTENSITIES:
            errors.append(f"{prefix}.intensity must be one of {sorted(_VALID_INTENSITIES)} when present")

    anchor_pose = description.get("anchor_pose")
    if anchor_pose is not None:
        if not isinstance(anchor_pose, str):
            errors.append(f"{prefix}.anchor_pose must be a string when present")
        else:
            anchor_pose = anchor_pose.strip()
            if anchor_pose and anchor_pose != "none" and anchor_pose not in named_poses:
                errors.append(f"{prefix}.anchor_pose references undefined pose '{anchor_pose}'")

    duration = description.get("duration_sec_estimate")
    if duration is not None:
        if not _is_finite_number(duration):
            errors.append(f"{prefix}.duration_sec_estimate must be a finite number")
        elif float(duration) <= 0.0:
            errors.append(f"{prefix}.duration_sec_estimate must be greater than zero")

    requires_motion_params = description.get("requires_motion_params")
    if requires_motion_params is not None and not isinstance(requires_motion_params, bool):
        errors.append(f"{prefix}.requires_motion_params must be a boolean when present")

    rule_entry = description.get("rule_entry")
    if rule_entry is not None and not isinstance(rule_entry, bool):
        errors.append(f"{prefix}.rule_entry must be a boolean when present")

    do_not_use = description.get("do_not_use")
    if do_not_use is None:
        return
    if not isinstance(do_not_use, list):
        errors.append(f"{prefix}.do_not_use must be a list")
        return
    for index, entry in enumerate(do_not_use):
        entry_prefix = f"{prefix}.do_not_use[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{entry_prefix} must be a mapping")
            continue
        condition = entry.get("condition")
        if not isinstance(condition, str) or not condition.strip():
            errors.append(f"{entry_prefix}.condition must be a non-empty string")
        instead_use = entry.get("instead_use")
        if not isinstance(instead_use, str) or not instead_use.strip():
            errors.append(f"{entry_prefix}.instead_use must reference a skill name")
        elif instead_use.strip() not in valid_skills:
            errors.append(f"{entry_prefix}.instead_use references unknown skill '{instead_use.strip()}'")
        elif instead_use.strip() == skill_name:
            errors.append(f"{entry_prefix}.instead_use must not reference the same skill")


def _validate_parameter_enum(value: Any, prefix: str, errors: list[str]) -> list[str] | None:
    if not isinstance(value, list) or not value:
        errors.append(f"{prefix} must be a non-empty list of non-empty strings")
        return None
    if not all(isinstance(item, str) and item.strip() for item in value):
        errors.append(f"{prefix} must be a non-empty list of non-empty strings")
        return None
    return value


def _validate_capability_parameter_property(
    property_name: str,
    definition: Any,
    prefix: str,
    errors: list[str],
) -> None:
    if not isinstance(definition, dict):
        errors.append(f"{prefix} must be a mapping")
        return

    allowed_fields = _STRING_PARAMETER_FIELDS if property_name in _STRING_REQUEST_FIELDS else _DISTANCE_PARAMETER_FIELDS
    for key in definition:
        if key not in allowed_fields:
            errors.append(f"{prefix} contains unsupported key '{key}'")

    expected_type = "string" if property_name in _STRING_REQUEST_FIELDS else "number"
    if definition.get("type") != expected_type:
        errors.append(f"{prefix}.type must be '{expected_type}'")

    if property_name in _STRING_REQUEST_FIELDS:
        if property_name == "target_name" and definition.get("freeform") is True:
            return
        enum = _validate_parameter_enum(definition.get("enum"), f"{prefix}.enum", errors)
        if property_name == "motion_direction" and enum is not None:
            unknown_directions = sorted(set(enum) - _VALID_MOTION_DIRECTIONS)
            if unknown_directions:
                errors.append(f"{prefix}.enum contains unsupported direction(s): {', '.join(unknown_directions)}")
        return

    if property_name == "motion_distance":
        exclusive_minimum = definition.get("exclusiveMinimum")
        if not _is_finite_number(exclusive_minimum) or float(exclusive_minimum) != 0.0:
            errors.append(f"{prefix}.exclusiveMinimum must equal 0")
        unit = definition.get("unit")
        if not isinstance(unit, str) or unit not in _VALID_DISTANCE_UNITS:
            errors.append(f"{prefix}.unit must be one of {sorted(_VALID_DISTANCE_UNITS)}")


def _validate_capability_parameters(parameters: dict[str, Any], prefix: str, errors: list[str]) -> None:
    parameter_prefix = f"{prefix}.parameters"
    for key in parameters:
        if key not in _PARAMETER_SCHEMA_FIELDS:
            errors.append(f"{parameter_prefix} contains unsupported key '{key}'")

    if parameters.get("type") != "object":
        errors.append(f"{parameter_prefix}.type must be 'object'")
    if parameters.get("additionalProperties") is not False:
        errors.append(f"{parameter_prefix}.additionalProperties must be false")

    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        errors.append(f"{parameter_prefix}.properties must be a mapping")
        properties = None
    else:
        for property_name, definition in properties.items():
            if not isinstance(property_name, str) or property_name not in _PUBLIC_REQUEST_FIELDS:
                errors.append(f"{parameter_prefix}.properties contains unsupported property '{property_name}'")
                continue
            _validate_capability_parameter_property(
                property_name,
                definition,
                f"{parameter_prefix}.properties.{property_name}",
                errors,
            )

    required = parameters.get("required")
    if not isinstance(required, list):
        errors.append(f"{parameter_prefix}.required must be a list")
        return

    required_names: set[str] = set()
    for index, property_name in enumerate(required):
        required_prefix = f"{parameter_prefix}.required[{index}]"
        if not isinstance(property_name, str) or not property_name.strip():
            errors.append(f"{required_prefix} must be a non-empty string")
            continue
        if property_name in required_names:
            errors.append(f"{parameter_prefix}.required entries must be unique")
            continue
        required_names.add(property_name)
        if properties is not None and property_name not in properties:
            errors.append(f"{required_prefix} must reference a property")


def _validate_skill_capability(
    skill_name: str,
    capability: Any,
    gateway_control_mode: str | None,
    errors: list[str],
) -> None:
    """Validate the public, ROS-independent capability metadata for one skill."""
    prefix = f"embodied.skill_templates.{skill_name}.capability"
    if capability is None:
        errors.append(f"{prefix} is required")
        return
    if not isinstance(capability, dict):
        errors.append(f"{prefix} must be a mapping")
        return

    schema_version = capability.get("schema_version")
    if schema_version is None:
        errors.append(f"{prefix}.schema_version is required")
    elif isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != 1:
        errors.append(f"{prefix}.schema_version must equal 1")

    for field in ("summary", "domain"):
        value = capability.get(field)
        if value is None:
            errors.append(f"{prefix}.{field} is required")
        elif not isinstance(value, str) or not value.strip():
            errors.append(f"{prefix}.{field} must be a non-empty string")

    moves_robot = capability.get("moves_robot")
    if moves_robot is None:
        errors.append(f"{prefix}.moves_robot is required")
    elif not isinstance(moves_robot, bool):
        errors.append(f"{prefix}.moves_robot must be a boolean")

    required_control_mode = capability.get("required_control_mode")
    if required_control_mode is None:
        errors.append(f"{prefix}.required_control_mode is required")
    elif not isinstance(required_control_mode, str) or not required_control_mode.strip():
        errors.append(f"{prefix}.required_control_mode must be a non-empty string")
    elif required_control_mode not in _SUPPORTED_CONTROL_MODES:
        errors.append(f"{prefix}.required_control_mode must be one of {sorted(_SUPPORTED_CONTROL_MODES)}")
    elif gateway_control_mode is not None and required_control_mode != gateway_control_mode:
        errors.append(f"{prefix}.required_control_mode must match skill_required_control_mode '{gateway_control_mode}'")

    parameters = capability.get("parameters")
    if parameters is None:
        errors.append(f"{prefix}.parameters is required")
    elif not isinstance(parameters, dict):
        errors.append(f"{prefix}.parameters must be a mapping")
    else:
        _validate_capability_parameters(parameters, prefix, errors)

    recovery_policy = capability.get("recovery_policy")
    if recovery_policy is None:
        errors.append(f"{prefix}.recovery_policy is required")
    elif not isinstance(recovery_policy, str):
        errors.append(f"{prefix}.recovery_policy must be a string")
    elif recovery_policy not in _VALID_RECOVERY_POLICIES:
        errors.append(f"{prefix}.recovery_policy must be one of {sorted(_VALID_RECOVERY_POLICIES)}")


def _load_robot_section(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load the raw ``robot`` section from a robot_config YAML file."""
    resolved_config_path = Path(config_path).expanduser().resolve()

    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Robot configuration not found: {resolved_config_path}")

    with resolved_config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Invalid robot config: expected mapping in {resolved_config_path}")

    if "robot" not in data:
        raise ValueError(f"Invalid robot config: missing 'robot' section in {resolved_config_path}")

    robot_data = data["robot"]
    if not isinstance(robot_data, dict):
        raise ValueError(f"Invalid robot config: 'robot' section must be a mapping in {resolved_config_path}")

    name = robot_data.get("name")
    if not name:
        raise ValueError(f"Invalid robot config: missing 'name' in {resolved_config_path}")

    return resolved_config_path, robot_data


_GRIPPER_ONLY_PRIMITIVES = {"open_gripper", "close_gripper"}


def _validate_absolute_joint_trajectory_entries(skill_templates: dict[str, Any], errors: list[str]) -> None:
    """Require every absolute waypoint trajectory to start from its first waypoint."""
    for skill_name, template in skill_templates.items():
        if not isinstance(template, dict):
            continue
        primitive_sequence = template.get("primitive_sequence", [])
        if not isinstance(primitive_sequence, list):
            continue

        for index, step in enumerate(primitive_sequence):
            if not isinstance(step, dict) or step.get("primitive_name") != "move_through_joint_positions":
                continue

            prefix = f"embodied.skill_templates.{skill_name}.primitive_sequence[{index}]"
            previous = None
            for candidate in reversed(primitive_sequence[:index]):
                if not isinstance(candidate, dict):
                    previous = candidate
                    break
                if candidate.get("primitive_name") in _GRIPPER_ONLY_PRIMITIVES:
                    continue
                previous = candidate
                break

            if not isinstance(previous, dict) or previous.get("primitive_name") != "move_to_joint_positions":
                errors.append(f"{prefix} must have a preceding move_to_joint_positions arm entry")
                continue

            try:
                duration_sec = float(previous.get("duration_sec", 0.0))
            except (OverflowError, TypeError, ValueError):
                duration_sec = 0.0
            if duration_sec <= 0.0:
                errors.append(f"{prefix} arm entry duration_sec must be greater than zero")

            waypoints = step.get("joint_waypoints", [])
            first_waypoint = waypoints[0] if isinstance(waypoints, list) and waypoints else {}
            first_positions = first_waypoint.get("joint_positions", {}) if isinstance(first_waypoint, dict) else {}
            entry_positions = previous.get("joint_positions", {})
            positions_match = bool(
                isinstance(entry_positions, dict) and isinstance(first_positions, dict) and first_positions
            )
            if positions_match:
                positions_match = set(entry_positions) == set(first_positions)
            if positions_match:
                try:
                    positions_match = all(
                        math.isclose(
                            float(entry_positions[joint_name]),
                            float(first_positions[joint_name]),
                            rel_tol=0.0,
                            abs_tol=1e-9,
                        )
                        for joint_name in first_positions
                    )
                except (OverflowError, TypeError, ValueError):
                    positions_match = False
            if not positions_match:
                errors.append(f"{prefix} arm entry must match the first joint waypoint")


def _validate_skill_primitive_sequence(
    skill_name: str,
    template: dict[str, Any],
    named_poses: dict[str, Any],
    errors: list[str],
) -> None:
    initial_gripper_state = template.get("initial_gripper_state")
    valid_gripper_states = {"open", "closed", "hold", "none"}
    if initial_gripper_state is not None and (
        not isinstance(initial_gripper_state, str) or initial_gripper_state.strip().lower() not in valid_gripper_states
    ):
        errors.append(
            f"embodied.skill_templates.{skill_name}.initial_gripper_state must be one of "
            f"{sorted(valid_gripper_states)} when present"
        )

    executor_name = str(template.get("executor", "")).strip()
    if executor_name:
        prefix = f"embodied.skill_templates.{skill_name}"
        if executor_name not in SUPPORTED_SKILL_EXECUTORS:
            errors.append(f"{prefix} uses unsupported executor '{executor_name}'")
        required_args = template.get("required_args", [])
        if not isinstance(required_args, list) or any(not isinstance(arg, str) for arg in required_args):
            errors.append(f"{prefix}.required_args must be a list of strings")
        timeout_sec = template.get("timeout_sec")
        if not _is_finite_number(timeout_sec) or float(timeout_sec) <= 0.0:
            errors.append(f"{prefix}.timeout_sec must be a finite number greater than zero")
        return

    primitive_sequence = template.get("primitive_sequence", [])
    if skill_name == "inspect_scene" and not primitive_sequence:
        return
    if not isinstance(primitive_sequence, list) or not primitive_sequence:
        errors.append(f"embodied.skill_templates.{skill_name}.primitive_sequence must be a non-empty list")
        return

    valid_directions = {"forward", "backward", "left", "right", "up", "down"}
    for index, step in enumerate(primitive_sequence):
        prefix = f"embodied.skill_templates.{skill_name}.primitive_sequence[{index}]"
        if not isinstance(step, dict):
            errors.append(f"embodied.skill_templates.{skill_name}.primitive_sequence entries must be objects")
            continue
        primitive_name = str(step.get("primitive_name", "")).strip()
        if primitive_name not in SUPPORTED_PRIMITIVES:
            errors.append(f"embodied.skill_templates.{skill_name} uses unsupported primitive '{primitive_name}'")
            continue
        if primitive_name == "move_to_named_pose":
            pose_name = str(step.get("pose_name", "")).strip()
            target_pose_key = str(step.get("target_pose_key", "")).strip()
            place_name_from_request = bool(step.get("place_name_from_request", False))
            if not pose_name and not target_pose_key and not place_name_from_request:
                errors.append(
                    f"embodied.skill_templates.{skill_name} move_to_named_pose step must define pose_name, "
                    "target_pose_key, or enable place_name_from_request"
                )
            elif pose_name and pose_name not in named_poses:
                errors.append(f"embodied.skill_templates.{skill_name} references undefined pose '{pose_name}'")
        if primitive_name == "move_relative_ee":
            literal_direction = str(step.get("motion_direction", "")).strip()
            if not step.get("motion_direction_from_request", False) and literal_direction not in valid_directions:
                errors.append(
                    f"embodied.skill_templates.{skill_name} move_relative_ee step must provide a valid "
                    "motion_direction or enable motion_direction_from_request"
                )
            if not step.get("motion_distance_from_request", False):
                literal_distance = step.get("motion_distance", 0.0)
                if not _is_finite_number(literal_distance) or float(literal_distance) <= 0.0:
                    errors.append(
                        f"embodied.skill_templates.{skill_name} move_relative_ee step must provide a positive "
                        "finite motion_distance or enable motion_distance_from_request"
                    )
        if primitive_name == "move_to_joint_positions":
            joint_positions = step.get("joint_positions")
            joint_position_offsets = step.get("joint_position_offsets")
            if joint_positions and joint_position_offsets:
                errors.append(f"{prefix} cannot define both joint_positions and joint_position_offsets")
                continue
            joint_map = joint_positions or joint_position_offsets
            if not isinstance(joint_map, dict) or not joint_map:
                errors.append(f"{prefix} must define joint_positions or joint_position_offsets")
            else:
                for joint_name, position in joint_map.items():
                    if not _is_finite_number(position):
                        errors.append(f"{prefix}.joint_positions.{joint_name} must be a finite number")
            duration_sec = step.get("duration_sec")
            if duration_sec is not None:
                if not _is_finite_number(duration_sec):
                    errors.append(f"{prefix}.duration_sec must be a finite number")
                elif float(duration_sec) <= 0.0:
                    errors.append(f"{prefix}.duration_sec must be greater than zero")
        if primitive_name == "move_through_joint_positions":
            waypoint_duration_sec = step.get("waypoint_duration_sec")
            if not _is_finite_number(waypoint_duration_sec):
                errors.append(f"{prefix}.waypoint_duration_sec must be a finite number")
            elif float(waypoint_duration_sec) <= 0.0:
                errors.append(f"{prefix}.waypoint_duration_sec must be greater than zero")

            joint_waypoints = step.get("joint_waypoints")
            if not isinstance(joint_waypoints, list) or not joint_waypoints:
                errors.append(f"{prefix}.joint_waypoints must be a non-empty list")
                continue
            for waypoint_index, waypoint in enumerate(joint_waypoints):
                waypoint_prefix = f"{prefix}.joint_waypoints[{waypoint_index}]"
                if not isinstance(waypoint, dict):
                    errors.append(f"{waypoint_prefix} must be an object")
                    continue
                joint_positions = waypoint.get("joint_positions")
                if not isinstance(joint_positions, dict) or not joint_positions:
                    errors.append(f"{waypoint_prefix}.joint_positions must be a non-empty mapping")
                    continue
                for joint_name, position in joint_positions.items():
                    if not _is_finite_number(position):
                        errors.append(f"{waypoint_prefix}.joint_positions.{joint_name} must be a finite number")


def _validate_embodied_skill_contract(robot_config: dict[str, Any]) -> list[str]:
    embodied = robot_config.get("embodied", {})
    if not isinstance(embodied, dict):
        return []
    return (
        ["embodied.skill_templates is removed; use embodied.skill_catalog_profile"]
        if "skill_templates" in embodied
        else []
    )


def _validate_skill_gateway_config(robot_config: dict[str, Any]) -> list[str]:
    """Validate the Gateway fields that cross robot and embodied configuration."""
    errors: list[str] = []
    required_control_mode = robot_config.get("skill_required_control_mode")
    embodied = robot_config.get("embodied", {})
    gateway_enabled = isinstance(embodied, dict) and bool(embodied.get("skill_catalog_profile"))
    if gateway_enabled and (not isinstance(required_control_mode, str) or not required_control_mode.strip()):
        errors.append("skill_required_control_mode is required when embodied.skill_catalog_profile is configured")
    elif required_control_mode is not None:
        if not isinstance(required_control_mode, str) or not required_control_mode.strip():
            errors.append("skill_required_control_mode must be a non-empty control_modes member")
        else:
            control_modes = robot_config.get("control_modes")
            if not isinstance(control_modes, dict) or required_control_mode not in control_modes:
                errors.append("skill_required_control_mode must be a control_modes member")

    if isinstance(embodied, dict):
        status_service = embodied.get("skill_gateway_status_service")
        if status_service is not None and (not isinstance(status_service, str) or not status_service.strip()):
            errors.append("embodied.skill_gateway_status_service must be a non-empty string")
        source_mode = embodied.get("skill_catalog_source_mode", "installed")
        source_root = embodied.get("skill_catalog_source_root", "")
        profile_name = embodied.get("skill_catalog_profile", "")
        if source_mode not in {"installed", "development", "production"}:
            errors.append("embodied.skill_catalog_source_mode must be installed, development, or production")
        elif source_mode in {"development", "production"} and (
            not isinstance(source_root, str) or not source_root.strip()
        ):
            errors.append("embodied.skill_catalog_source_root is required in development and production modes")
        if embodied.get("enabled", False) and (not isinstance(profile_name, str) or not profile_name.strip()):
            errors.append("embodied.skill_catalog_profile is required")
        try:
            resolve_embodied_timeout_policy(embodied)
        except ValueError as exc:
            errors.append(str(exc))
    return errors


def load_robot_config_dict(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load robot configuration as a complete dict.

    This is the canonical loader for launch/builders/runtime consumers. It preserves
    the full YAML schema under ``robot`` and annotates the resolved source path for
    downstream users that need provenance.
    """
    resolved_config_path, robot_data = _load_robot_section(resolve_robot_config_path(config_path=config_path))
    robot_config = copy.deepcopy(robot_data)
    validation_errors = validate_grasp_execution_config(robot_config.get("grasp_execution"))
    validation_errors.extend(validate_placement_execution_config(robot_config.get("placement_execution")))
    validation_errors.extend(_validate_embodied_skill_contract(robot_config))
    validation_errors.extend(_validate_skill_gateway_config(robot_config))
    try:
        parse_perception_runtime_config(robot_config)
    except PerceptionRuntimeConfigError as exc:
        validation_errors.append(str(exc))
    validation_errors.extend(validate_semantic_mapping_config(robot_config))
    try:
        validation_errors.extend(validate_robot_config_observation_transports(robot_config))
    except (TypeError, ValueError) as exc:
        validation_errors.append(str(exc))
    if validation_errors:
        raise ValueError("Invalid robot configuration:\n- " + "\n- ".join(validation_errors))
    robot_config["_config_path"] = str(resolved_config_path)
    return robot_config


def load_camera_config(data: dict[str, Any]) -> CameraConfig:
    """Load camera configuration from dict.

    Example:
    ```yaml
    - type: camera
      name: top
      driver: opencv
      index: 0
      width: 640
      height: 480
      fps: 30
      frame_id: camera_top_frame
      optical_frame_id: camera_top_optical_frame
      camera_info_url: file:///path/to/calibration.yaml
    ```
    """
    driver = data.get("driver", "opencv")

    # Handle different index/port naming conventions
    index_or_port = data.get("index", data.get("port", data.get("serial_number", 0)))
    if driver == "realsense" and "serial_number" in data:
        index_or_port = data["serial_number"]

    return CameraConfig(
        name=data["name"],
        driver=driver,
        index_or_port=index_or_port,
        width=data.get("width", 640),
        height=data.get("height", 480),
        fps=data.get("fps", 30),
        frame_id=data.get("frame_id", f"camera_{data['name']}_frame"),
        optical_frame_id=data.get("optical_frame_id", f"camera_{data['name']}_optical_frame"),
        camera_info_url=data.get("camera_info_url"),
        pixel_format=data.get("pixel_format", "bgr8"),
        depth_width=data.get("depth_width"),
        depth_height=data.get("depth_height"),
        depth_fps=data.get("depth_fps"),
        enable_pointcloud=data.get("enable_pointcloud", False),
        enable_sync=data.get("enable_sync", True),
        align_depth=data.get("align_depth", False),
        direct_topic_remap=data.get("direct_topic_remap", False),
        transform=data.get("transform"),
    )


def load_ros2_control_config(data: dict[str, Any], config_dir: Path | None = None) -> Ros2ControlConfig:
    """Load ros2_control configuration from dict.

    Example:
    ```yaml
    ros2_control:
      hardware_plugin: so101_hardware/SO101SystemHardware
      port: /dev/ttyACM0
      calib_file: $(env HOME)/.calibrate/so101_follower_calibrate.json
      reset_positions: {1: 0.0, 2: 0.0}
      urdf_path: $(find robot_description)/urdf/lerobot/so101/so101.urdf.xacro
    ```
    """
    params = {}
    for key, value in data.items():
        if key not in ["hardware_plugin", "urdf_path"]:
            # Resolve paths in parameters
            if isinstance(value, str):
                value = resolve_ros_path(value)
            params[key] = value

    return Ros2ControlConfig(
        hardware_plugin=data.get("hardware_plugin", ""),
        params=params,
        urdf_path=resolve_ros_path(data.get("urdf_path")),
    )


def load_contract_config(data: dict[str, Any]) -> ContractExtensionConfig:
    """Load contract extension configuration from dict.

    Example:
    ```yaml
    contract:
      base_contract: $(find robot_config)/config/contracts/act_grab_pan.yaml
      observations:
        - key: observation.images.top
          topic: /camera/top
          peripheral: top
      actions:
        - key: action
          publish:
            topic: /joint_commands
            type: sensor_msgs/msg/JointState
    ```
    """
    observations = []
    for obs_data in data.get("observations", []):
        observations.append(
            ContractObservation(
                key=obs_data["key"],
                topic=obs_data.get("topic"),
                type=obs_data.get("type"),
                peripheral=obs_data.get("peripheral"),
                selector=obs_data.get("selector"),
                image=obs_data.get("image"),
                align=obs_data.get("align"),
                qos=obs_data.get("qos"),
                transport=parse_observation_transport(obs_data.get("transport")),
            )
        )

    actions = []
    for action_data in data.get("actions", []):
        actions.append(
            ContractAction(
                key=action_data["key"],
                publish=action_data.get("publish", {}),
                selector=action_data.get("selector"),
                from_tensor=action_data.get("from_tensor"),
                safety_behavior=action_data.get("safety_behavior", "zeros"),
            )
        )

    return ContractExtensionConfig(
        base_contract=data.get("base_contract"),
        observations=observations,
        actions=actions,
        rate_hz=data.get("rate_hz", 20.0),
        max_duration_s=data.get("max_duration_s", 30.0),
    )


def load_voice_asr_config(data: dict[str, Any]) -> VoiceASRConfig:
    """Load voice ASR configuration from dict."""
    defaults = VoiceASRConfig()
    model_path = data.get("model_path", "")
    tokens_path = data.get("tokens_path", "")

    return VoiceASRConfig(
        enabled=data.get("enabled", defaults.enabled),
        auto_download_model=data.get("auto_download_model", defaults.auto_download_model),
        active_mode=data.get("active_mode", defaults.active_mode),
        language=data.get("language", defaults.language),
        model_path=resolve_ros_path(model_path) if model_path else "",
        tokens_path=resolve_ros_path(tokens_path) if tokens_path else "",
        provider=data.get("provider", defaults.provider),
        model_type=data.get("model_type", defaults.model_type),
        max_recording_duration=data.get("max_recording_duration", defaults.max_recording_duration),
        vad_sensitivity=data.get("vad_sensitivity", defaults.vad_sensitivity),
        realtime_pre_roll_seconds=data.get("realtime_pre_roll_seconds", defaults.realtime_pre_roll_seconds),
        publish_partial=data.get("publish_partial", defaults.publish_partial),
        output_topic=data.get("output_topic", defaults.output_topic),
        sample_rate=data.get("sample_rate", defaults.sample_rate),
        chunk_size=data.get("chunk_size", defaults.chunk_size),
        buffer_seconds=data.get("buffer_seconds", defaults.buffer_seconds),
        device_index=data.get("device_index", defaults.device_index),
        device_name=data.get("device_name", defaults.device_name),
        exit_on_init_failure=data.get("exit_on_init_failure", defaults.exit_on_init_failure),
    )


def load_voice_tts_config(data: dict[str, Any]) -> VoiceTTSConfig:
    """Load Voice TTS configuration without selecting a backend implicitly."""

    defaults = VoiceTTSConfig()
    bundle_path = data.get("bundle_path", defaults.bundle_path)
    return VoiceTTSConfig(
        enabled=data.get("enabled", defaults.enabled),
        bundle_path=resolve_ros_path(bundle_path) if bundle_path else "",
        deployment=data.get("deployment", defaults.deployment),
        service_name=data.get("service_name", defaults.service_name),
        playback_service_name=data.get("playback_service_name", defaults.playback_service_name),
        playback_timeout_sec=data.get("playback_timeout_sec", defaults.playback_timeout_sec),
        prompt_profile=data.get("prompt_profile", defaults.prompt_profile),
        segment_max_chars=data.get("segment_max_chars", defaults.segment_max_chars),
        segment_pause_ms=data.get("segment_pause_ms", defaults.segment_pause_ms),
        max_request_chars=data.get("max_request_chars", defaults.max_request_chars),
        max_prompt_audio_bytes=data.get("max_prompt_audio_bytes", defaults.max_prompt_audio_bytes),
        max_prompt_duration_sec=data.get("max_prompt_duration_sec", defaults.max_prompt_duration_sec),
        max_segments=data.get("max_segments", defaults.max_segments),
        max_response_audio_bytes=data.get("max_response_audio_bytes", defaults.max_response_audio_bytes),
        device_id=data.get("device_id", defaults.device_id),
        exit_on_init_failure=data.get("exit_on_init_failure", defaults.exit_on_init_failure),
    )


def load_semantic_mapping_config(data: dict[str, Any]) -> SemanticMappingConfig:
    """Load the standalone semantic mapping section without flattening its contracts."""
    return SemanticMappingConfig(
        enabled=data.get("enabled", False),
        camera=dict(data.get("camera", {})),
        slam=dict(data.get("slam", {})),
        perception=dict(data.get("perception", {})),
        persistence=dict(data.get("persistence", {})),
        filtering=dict(data.get("filtering", {})),
        queue=dict(data.get("queue", {})),
        lifecycle=dict(data.get("lifecycle", {})),
        labels=dict(data.get("labels", {})),
        label_refinement=dict(data.get("label_refinement", {})),
        target_watch=dict(data.get("target_watch", {})),
        interfaces=dict(data.get("interfaces", {})),
    )


def load_embodied_config(data: dict[str, Any]) -> EmbodiedConfig:
    """Load embodied minimal-closure configuration from dict."""
    execution = data.get("execution", {})
    safety = data.get("safety", {})
    direction_mapping = execution.get("relative_motion_direction_mapping", {})
    perception = data.get("perception", {})
    entry = data.get("entry", {})
    timeout_policy = resolve_embodied_timeout_policy(data)

    return EmbodiedConfig(
        enabled=data.get("enabled", False),
        debug_tracing=data.get("debug_tracing", True),
        task_input_topic=data.get("task_input_topic", "/voice_command"),
        task_command_topic=data.get("task_command_topic", "/embodied/task_command"),
        planned_task_topic=data.get("planned_task_topic", "/embodied/planned_task"),
        status_topic=data.get("status_topic", "/embodied/task_status"),
        skill_action_name=data.get("skill_action_name", "/embodied/execute_skill"),
        primitive_action_name=data.get("primitive_action_name", "/embodied/execute_primitive"),
        validate_skill_service=data.get("validate_skill_service", "/embodied/validate_skill"),
        validate_primitive_service=data.get("validate_primitive_service", "/embodied/validate_primitive"),
        skill_gateway_status_service=data.get("skill_gateway_status_service", "/embodied/get_skill_gateway_status"),
        skill_catalog_source_mode=data.get("skill_catalog_source_mode", "installed"),
        skill_catalog_source_root=data.get("skill_catalog_source_root", ""),
        skill_catalog_profile=data.get("skill_catalog_profile", ""),
        default_target_name=data.get("default_target_name", "demo_object"),
        default_place_name=data.get("default_place_name", "tray_right"),
        skill_timeout_sec=execution.get("skill_timeout_sec", 120.0),
        primitive_timeout_sec=execution.get("primitive_timeout_sec", 5.0),
        primitive_wait_sec=execution.get("primitive_wait_sec", 1.0),
        timeouts=timeout_policy,
        relative_motion_step_m=execution.get("relative_motion_step_m", 0.03),
        relative_motion_reference_frame=execution.get("relative_motion_reference_frame", "base"),
        relative_motion_direction_mapping=direction_mapping,
        perception=perception,
        entry=entry,
        gripper_open_position=execution.get("gripper_open_position", 1.0),
        gripper_closed_position=execution.get("gripper_closed_position", 0.0),
        skill_templates=data.get("skill_templates", {}),
        named_poses=data.get("named_poses", {}),
        named_targets=data.get("named_targets", {}),
        workspace=safety.get("workspace", {}),
    )


def load_robot_config(config_path: str | Path | None = None) -> RobotConfig:
    """Load robot configuration from YAML file.

    Args:
        config_path: Path to robot configuration YAML file

    Returns:
        RobotConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    robot_data = load_robot_config_dict(config_path)
    resolved_config_path = Path(robot_data["_config_path"])
    config_dir = resolved_config_path.parent

    # Load required fields
    name = robot_data.get("name")

    robot_type = robot_data.get("robot_type", robot_data.get("type", name))
    type_ = robot_data.get("type", name)

    # Load ros2_control config
    ros2_control_data = robot_data.get("ros2_control", {})
    ros2_control = load_ros2_control_config(ros2_control_data, config_dir)

    # Load peripherals (cameras)
    peripherals = []
    for periph_data in robot_data.get("peripherals", []):
        if periph_data.get("type") == "camera":
            peripherals.append(load_camera_config(periph_data))
        else:
            # Generic peripheral
            peripherals.append(
                PeripheralConfig(
                    type=periph_data["type"],
                    name=periph_data["name"],
                    driver=periph_data.get("driver", "generic"),
                    params=periph_data.get("params", {}),
                    frame_id=periph_data.get("frame_id"),
                )
            )

    # Load contract config
    contract_data = robot_data.get("contract", {})
    contract = load_contract_config(contract_data)

    voice_asr = load_voice_asr_config(robot_data.get("voice_asr", {}))
    voice_tts = load_voice_tts_config(robot_data.get("voice_tts", {}))
    embodied = load_embodied_config(robot_data.get("embodied", {}))
    skill_gateway = SkillGatewayRuntimeConfig(
        status_service=embodied.skill_gateway_status_service,
        required_control_mode=robot_data.get("skill_required_control_mode", ""),
        control_modes=tuple(robot_data.get("control_modes", {}).keys())
        if isinstance(robot_data.get("control_modes"), dict)
        else (),
        default_skill_timeout_sec=embodied.timeouts["default_skill_timeout_sec"],
        robot_state_freshness_sec=embodied.timeouts["robot_state_freshness_sec"],
        task_budget_sec=embodied.timeouts["task_budget_sec"],
        rpc_timeout_sec=embodied.timeouts["rpc_timeout_sec"],
    )
    semantic_mapping = load_semantic_mapping_config(robot_data.get("semantic_mapping", {}))
    perception_services = parse_perception_runtime_config(robot_data)

    return RobotConfig(
        name=name,
        type=type_,
        robot_type=robot_type,
        ros2_control=ros2_control,
        peripherals=peripherals,
        contract=contract,
        voice_asr=voice_asr,
        voice_tts=voice_tts,
        embodied=embodied,
        skill_gateway=skill_gateway,
        semantic_mapping=semantic_mapping,
        perception_services=perception_services,
        placement_execution=robot_data.get("placement_execution", {})
        if isinstance(robot_data.get("placement_execution", {}), dict)
        else {},
    )


def build_contract_from_robot_config_dict(robot_config: dict[str, Any]):
    """Build a runtime contract directly from the canonical dict loader output."""
    from robot_config.generators.contract import build_contract_from_robot_config_dict as _build

    return _build(robot_config)


def _robot_config_to_validation_dict(config: RobotConfig) -> dict[str, Any]:
    params = dict(config.ros2_control.params)
    return {"ros2_control": params}


def _validate_vlm_scene_sources(
    errors: list[str],
    section_path: str,
    scene_sources: dict[str, Any],
    required_reason: str,
) -> None:
    if not scene_sources.get("primary_camera_topic"):
        errors.append(f"{section_path}.scene_sources.primary_camera_topic is required when {required_reason}")
    if bool(scene_sources.get("require_depth", False)) and not (
        scene_sources.get("primary_aligned_depth_topic") or scene_sources.get("wrist_aligned_depth_topic")
    ):
        errors.append(f"{section_path}.scene_sources.require_depth=true requires at least one aligned depth topic")
    if bool(scene_sources.get("require_pointcloud", False)) and not (
        scene_sources.get("primary_pointcloud_topic") or scene_sources.get("wrist_pointcloud_topic")
    ):
        errors.append(f"{section_path}.scene_sources.require_pointcloud=true requires at least one pointcloud topic")


def _validate_vlm_api_config(
    errors: list[str],
    section_path: str,
    vlm_api: dict[str, Any],
    valid_providers: set[str],
    required_reason: str,
) -> None:
    provider = str(vlm_api.get("provider", "")).strip()
    if not provider:
        errors.append(f"{section_path}.vlm_api.provider is required when {required_reason}")
    elif provider not in valid_providers:
        errors.append(f"{section_path}.vlm_api.provider must be one of: " + ", ".join(sorted(valid_providers)))
    if not vlm_api.get("model"):
        errors.append(f"{section_path}.vlm_api.model is required when {required_reason}")
    if provider == "kimicode" and not str(vlm_api.get("api_key_env", "")).strip():
        errors.append(f"{section_path}.vlm_api.api_key_env is required when {required_reason}")
    if not vlm_api.get("base_url"):
        errors.append(f"{section_path}.vlm_api.base_url is required when {required_reason}")


def _validate_vlm_runtime_config(
    errors: list[str],
    section_path: str,
    config: dict[str, Any],
    timeout_policy: dict[str, Any],
    valid_providers: set[str],
    required_reason: str,
) -> None:
    _validate_vlm_scene_sources(errors, section_path, config.get("scene_sources", {}), required_reason)
    if float(timeout_policy.get("scene_freshness_sec", 0.0)) <= 0.0:
        errors.append("embodied.timeouts.scene_freshness_sec must be greater than zero")
    _validate_vlm_api_config(errors, section_path, config.get("vlm_api", {}), valid_providers, required_reason)
    if float(timeout_policy.get("model_idle_timeout_sec", 0.0)) <= 0.0:
        errors.append("embodied.timeouts.model_idle_timeout_sec must be greater than zero")


def validate_visual_games_consistency(
    entry: dict[str, Any],
    perception: dict[str, Any],
    *,
    entry_mode: str = "hermes",
) -> list[str]:
    """Check the visual-games <-> perception enable consistency rule.

    Any enabled ``embodied.entry.visual_games.<name>`` routes its trigger to
    ``perception_service``; if perception is disabled the request lands on a
    topic nobody consumes. This is the single source of truth for that rule,
    shared by both the typed :func:`validate_config` and the raw-dict launch
    entry :func:`validate_embodied_launch_dict`.
    """
    errors: list[str] = []
    games = entry.get("visual_games", {})
    if isinstance(games, dict):
        enabled_games = [
            name for name, policy in games.items() if isinstance(policy, dict) and policy.get("enabled", False)
        ]
        if enabled_games and not perception.get("enabled", False):
            errors.append(
                "embodied.entry.visual_games requires "
                "embodied.perception.enabled: true when any game is enabled "
                f"({enabled_games})"
            )
        if enabled_games and entry_mode == "hermes":
            errors.append(
                "embodied.entry.visual_games requires a voice entry mode, which is not supported "
                f"when any game is enabled ({enabled_games})"
            )
    return errors


def validate_embodied_launch_dict(config: dict[str, Any]) -> list[str]:
    """Validate the embodied consistency rules a launch consumer must honor.

    Launch files (e.g. ``embodied_pipeline.launch.py``) load a raw config dict
    via :func:`load_robot_config_dict` and apply ``with_perception`` / game
    overrides before generating nodes. They cannot cheaply build a typed
    :class:`RobotConfig`, so this is the canonical raw-dict gate they call after
    applying overrides. It intentionally covers only the launch-relevant
    game/perception consistency (the full typed validation needs a
    ``RobotConfig``) and reuses :func:`validate_visual_games_consistency` so the
    rule stays single-sourced.

    Returns a list of error strings (empty when the config is launchable).
    """
    embodied = config.get("embodied", {})
    if not isinstance(embodied, dict) or not embodied.get("enabled", False):
        return []
    entry = embodied.get("entry", {}) or {}
    perception = embodied.get("perception", {}) or {}
    entry_mode = str(embodied.get("entry_mode", "hermes")).lower()
    errors = []
    if entry_mode != "hermes":
        errors.append("embodied.entry_mode must be hermes")
    errors.extend(validate_visual_games_consistency(entry, perception, entry_mode=entry_mode))
    return errors


def validate_config(config: RobotConfig) -> list[str]:
    """Validate robot configuration.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    errors.extend(validate_placement_execution_config(getattr(config, "placement_execution", None)))

    semantic_mapping_dict = {
        "enabled": config.semantic_mapping.enabled,
        "camera": config.semantic_mapping.camera,
        "slam": config.semantic_mapping.slam,
        "perception": config.semantic_mapping.perception,
        "persistence": config.semantic_mapping.persistence,
        "filtering": config.semantic_mapping.filtering,
        "queue": config.semantic_mapping.queue,
        "lifecycle": config.semantic_mapping.lifecycle,
        "labels": config.semantic_mapping.labels,
        "label_refinement": config.semantic_mapping.label_refinement,
        "target_watch": config.semantic_mapping.target_watch,
        "interfaces": config.semantic_mapping.interfaces,
    }
    errors.extend(
        validate_semantic_mapping_config(
            {
                "semantic_mapping": semantic_mapping_dict,
                "perception_services": {
                    "services": [
                        {
                            "id": service.instance_id,
                            "enabled": service.enabled,
                            "required": service.required,
                            **(
                                {
                                    "bundle_path": str(service.bundle_path),
                                    "deployment": service.deployment,
                                    "adapter_class": service.adapter_class,
                                    "service_type": service.service_type,
                                    "endpoint": service.endpoint,
                                    "node_name": service.node_name,
                                    "runtime_options": dict(service.runtime_options),
                                }
                                if service.enabled
                                else {}
                            ),
                        }
                        for service in (config.perception_services.services if config.perception_services else ())
                    ]
                },
                "peripherals": [
                    {
                        "type": "camera" if isinstance(peripheral, CameraConfig) else peripheral.type,
                        "name": peripheral.name,
                        "driver": peripheral.driver,
                        "align_depth": peripheral.align_depth if isinstance(peripheral, CameraConfig) else False,
                        "transform": peripheral.transform if isinstance(peripheral, CameraConfig) else {},
                    }
                    for peripheral in config.peripherals
                ],
            }
        )
    )

    # Validate ros2_control config
    if not config.ros2_control.hardware_plugin:
        errors.append("ros2_control.hardware_plugin is required")

    try:
        calibration_paths = resolve_calibration_paths_from_config(_robot_config_to_validation_dict(config))
    except (TypeError, ValueError) as exc:
        errors.append(f"Invalid ros2_control calibration configuration: {exc}")
    else:
        for calib_file in calibration_paths:
            if calib_file and not Path(calib_file).exists():
                errors.append(f"Calibration file not found: {calib_file}")

    # Validate peripherals
    peripheral_names = set()
    for periph in config.peripherals:
        if periph.name in peripheral_names:
            if isinstance(periph, CameraConfig):
                errors.append(f"Duplicate camera name: {periph.name}")
            else:
                errors.append(f"Duplicate peripheral name: {periph.name}")
        peripheral_names.add(periph.name)

        if not isinstance(periph, CameraConfig):
            continue

        # Validate camera parameters
        if periph.width <= 0 or periph.height <= 0:
            errors.append(f"Invalid camera dimensions for {periph.name}: {periph.width}x{periph.height}")
        if periph.fps <= 0:
            errors.append(f"Invalid FPS for {periph.name}: {periph.fps}")

        # Validate calibration file if specified
        if periph.camera_info_url and periph.camera_info_url.startswith("file://"):
            calib_path = Path(periph.camera_info_url.replace("file://", ""))
            if not calib_path.exists():
                errors.append(f"Camera calibration file not found: {calib_path}")

    # Validate contract-peripheral references
    for obs in config.contract.observations:
        if obs.peripheral and obs.peripheral not in peripheral_names:
            errors.append(f"Observation '{obs.key}' references undefined peripheral: {obs.peripheral}")

    try:
        contract = config.to_contract()
    except ValueError as exc:
        errors.append(str(exc))
    else:
        errors.extend(validate_observation_transports(contract.observations))

    if config.voice_asr.enabled and not config.voice_asr.model_path and not config.voice_asr.auto_download_model:
        errors.append("voice_asr.model_path is required when voice_asr.enabled is true")

    if config.voice_tts.enabled:
        if not config.voice_tts.bundle_path:
            errors.append("voice_tts.bundle_path is required when voice_tts.enabled is true")
        if not config.voice_tts.deployment:
            errors.append("voice_tts.deployment is required when voice_tts.enabled is true")
        if not config.voice_tts.service_name.startswith("/"):
            errors.append("voice_tts.service_name must be an absolute ROS service name")
        if not config.voice_tts.playback_service_name.startswith("/"):
            errors.append("voice_tts.playback_service_name must be an absolute ROS service name")
        if config.voice_tts.playback_timeout_sec <= 0:
            errors.append("voice_tts.playback_timeout_sec must be positive")
        if not config.voice_tts.prompt_profile:
            errors.append("voice_tts.prompt_profile must be non-empty")
        positive_limits = {
            "segment_max_chars": config.voice_tts.segment_max_chars,
            "max_request_chars": config.voice_tts.max_request_chars,
            "max_prompt_audio_bytes": config.voice_tts.max_prompt_audio_bytes,
            "max_prompt_duration_sec": config.voice_tts.max_prompt_duration_sec,
            "max_segments": config.voice_tts.max_segments,
            "max_response_audio_bytes": config.voice_tts.max_response_audio_bytes,
        }
        for name, value in positive_limits.items():
            if value <= 0:
                errors.append(f"voice_tts.{name} must be positive")
        if config.voice_tts.segment_pause_ms < 0:
            errors.append("voice_tts.segment_pause_ms must be non-negative")

    if config.embodied.enabled:
        valid_directions = {"forward", "backward", "left", "right", "up", "down"}
        if config.embodied.entry_mode != "hermes":
            errors.append("embodied.entry_mode must be hermes")
        required_pose_names = {"home", "observe_table", "zero"}
        missing_pose_names = sorted(p for p in required_pose_names if p not in config.embodied.named_poses)
        if missing_pose_names:
            errors.append("embodied.named_poses is missing required pose(s): " + ", ".join(missing_pose_names))
        if config.embodied.default_place_name and config.embodied.default_place_name not in config.embodied.named_poses:
            errors.append(
                f"embodied.default_place_name references undefined pose: {config.embodied.default_place_name}"
            )

        if config.embodied.skill_templates:
            errors.append("embodied.skill_templates is removed; use embodied.skill_catalog_profile")
        if not config.embodied.skill_catalog_profile:
            errors.append("embodied.skill_catalog_profile is required when embodied is enabled")
        required_control_mode = config.skill_gateway.required_control_mode
        if not isinstance(required_control_mode, str) or not required_control_mode.strip():
            errors.append("skill_required_control_mode is required when embodied.skill_catalog_profile is configured")
        elif config.skill_gateway.control_modes and required_control_mode not in config.skill_gateway.control_modes:
            errors.append("skill_required_control_mode must be a control_modes member")

        for axis in ("x", "y", "z"):
            axis_limits = config.embodied.workspace.get(axis)
            if axis_limits is None:
                continue
            if not isinstance(axis_limits, list) or len(axis_limits) != 2:
                errors.append(f"embodied.safety.workspace.{axis} must be a [min, max] list")
                continue
            if axis_limits[0] >= axis_limits[1]:
                errors.append(f"embodied.safety.workspace.{axis} must satisfy min < max")
        max_radius_m = config.embodied.workspace.get("max_radius_m")
        if max_radius_m is not None and float(max_radius_m) <= 0.0:
            errors.append("embodied.safety.workspace.max_radius_m must be greater than zero")

        if config.embodied.relative_motion_step_m <= 0.0:
            errors.append("embodied.execution.relative_motion_step_m must be greater than zero")

        if config.embodied.relative_motion_reference_frame != "base":
            errors.append("embodied.execution.relative_motion_reference_frame currently must be 'base'")

        direction_mapping = config.embodied.relative_motion_direction_mapping
        if direction_mapping:
            missing_directions = valid_directions.difference(direction_mapping.keys())
            if missing_directions:
                errors.append(
                    "embodied.execution.relative_motion_direction_mapping is missing directions: "
                    + ", ".join(sorted(missing_directions))
                )
            for direction, vector in direction_mapping.items():
                if direction not in valid_directions:
                    errors.append(
                        "embodied.execution.relative_motion_direction_mapping contains unsupported "
                        f"direction: {direction}"
                    )
                    continue
                if not isinstance(vector, list) or len(vector) != 3:
                    errors.append(
                        f"embodied.execution.relative_motion_direction_mapping.{direction} must be a 3-element list"
                    )
                    continue
                try:
                    normalized = [float(v) for v in vector]
                except (TypeError, ValueError):
                    errors.append(
                        f"embodied.execution.relative_motion_direction_mapping.{direction} must contain numeric values"
                    )
                    continue
                if all(abs(v) < 1e-9 for v in normalized):
                    errors.append(
                        f"embodied.execution.relative_motion_direction_mapping.{direction} must not be a zero vector"
                    )

        try:
            timeout_policy = resolve_embodied_timeout_policy(
                {
                    "execution": {
                        "skill_timeout_sec": config.embodied.skill_timeout_sec,
                        "primitive_timeout_sec": config.embodied.primitive_timeout_sec,
                        "primitive_wait_sec": config.embodied.primitive_wait_sec,
                    },
                    "perception": config.embodied.perception or {},
                    "timeouts": config.embodied.timeouts or {},
                }
            )
        except ValueError as exc:
            errors.append(str(exc))
            timeout_policy = {}
        valid_vlm_api_providers = {"kimicode", "openai_compatible"}
        perception = config.embodied.perception or {}
        if perception.get("enabled", False):
            if not str(perception.get("request_topic", "")).strip():
                errors.append("embodied.perception.request_topic is required when perception is enabled")
            if not str(perception.get("result_topic", "")).strip():
                errors.append("embodied.perception.result_topic is required when perception is enabled")
            _validate_vlm_runtime_config(
                errors,
                "embodied.perception",
                perception,
                timeout_policy,
                valid_vlm_api_providers,
                "perception is enabled",
            )

        errors.extend(validate_visual_games_consistency(config.embodied.entry or {}, perception))

        if float(timeout_policy.get("task_budget_sec", 0.0)) <= 0.0:
            errors.append("embodied.timeouts.task_budget_sec must be greater than zero")
        if float(timeout_policy.get("rpc_timeout_sec", 0.0)) <= 0.0:
            errors.append("embodied.timeouts.rpc_timeout_sec must be greater than zero")
        if float(timeout_policy.get("gripper_settle_sec", 0.0)) <= 0.0:
            errors.append("embodied.timeouts.gripper_settle_sec must be greater than zero")

        conversation = perception.get("conversation", {})
        try:
            max_history_turns = int(conversation.get("max_history_turns", 0))
        except (TypeError, ValueError):
            errors.append("embodied.perception.conversation.max_history_turns must be an integer")
        else:
            if max_history_turns < 0:
                errors.append("embodied.perception.conversation.max_history_turns must be >= 0")

    return errors


def validate_config_file(config_path: str | Path) -> bool:
    """Validate a robot configuration file.

    Returns:
        True if valid, False otherwise
    """
    try:
        config = load_robot_config(config_path)
        errors = validate_config(config)

        if errors:
            logger.error(f"Configuration errors in {config_path}:")
            for error in errors:
                logger.error(f"  - {error}")
            return False

        logger.info(f"Configuration {config_path} is valid")
        return True

    except Exception as e:
        logger.error(f"Failed to validate {config_path}: {e}")
        return False
