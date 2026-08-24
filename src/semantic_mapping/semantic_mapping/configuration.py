"""Translate the robot_config semantic mapping SSOT into ROS node parameters."""

import json
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from inference_manifest import canonical_semantic_identity_json
from robot_config.launch_builders.perception_models import generate_perception_model_nodes
from robot_config.loader import load_robot_config_dict
from robot_config.perception_runtime_config import parse_perception_runtime_config

_CONSTRUCTION_ROLES = ("sam2_masks", "ram_plus_tags", "siglip2_image")


def load_semantic_mapping_robot_config(robot_config_name: str, config_path_override: str = "") -> dict:
    """Load one validated robot config for a standalone semantic mapping launch."""
    if config_path_override:
        config_path = Path(config_path_override)
    else:
        config_path = (
            Path(get_package_share_directory("robot_config")) / "config" / "robots" / f"{robot_config_name}.yaml"
        )
    config = load_robot_config_dict(config_path)
    if not config.get("semantic_mapping", {}).get("enabled", False):
        raise ValueError("robot.semantic_mapping.enabled must be true for standalone semantic mapping launch")
    return config


def semantic_mapping_parameters(config: dict, *, offline: bool = False) -> dict:
    """Flatten the validated SSOT sections into the mapping node parameter contract."""
    mapping = config["semantic_mapping"]
    camera = mapping["camera"]
    slam = mapping["slam"]
    perception = mapping["perception"]
    persistence = mapping["persistence"]
    filtering = mapping["filtering"]
    queue = mapping["queue"]
    lifecycle = mapping["lifecycle"]
    labels = mapping["labels"]
    label_refinement = mapping["label_refinement"]
    target_watch = mapping["target_watch"]
    interfaces = mapping["interfaces"]
    roles = perception["semantic_roles"]
    query_only = bool(mapping.get("query_only", False))
    mapping_backend = perception["mapping_backend"]
    runtime = parse_perception_runtime_config(config) if mapping_backend == "service" else None
    services = {} if runtime is None else {service.instance_id: service for service in runtime.enabled_services}

    def service(role: str):
        return services[roles[role]]

    def identity(role: str) -> str:
        semantic_identity = service(role).validated_manifest.manifest.model.semantic_identity
        return canonical_semantic_identity_json(semantic_identity)

    generation = (
        0 if runtime is None else runtime.configuration_generation({role: roles[role] for role in _CONSTRUCTION_ROLES})
    )

    parameters = {
        "rgb_topic": camera["rgb_topic"],
        "depth_topic": camera["depth_topic"],
        "camera_info_topic": camera["camera_info_topic"],
        "global_frame": slam["global_frame"],
        "geometry_map_id": slam["geometry_map_id"],
        "geometry_map_hash": slam["geometry_map_hash"],
        "localization_session_id": slam["localization_session_id"],
        "calibration_id": slam["calibration_id"],
        "urdf_hash": slam["urdf_hash"],
        "coordinate_convention": slam["coordinate_convention"],
        "database_path": persistence["database_path"],
        "configuration_generation": generation,
        "online_processing_enabled": not query_only,
        "sync_slop_sec": queue["sync_slop_sec"],
        "max_masks_per_frame": queue.get("max_masks_per_frame", 32),
        "max_masks_per_batch": queue["max_masks_per_batch"],
        "min_mask_pixels": filtering["min_mask_pixels"],
        "min_mask_area_ratio": filtering["min_mask_area_ratio"],
        "min_mask_valid_depth_ratio": filtering["min_mask_valid_depth_ratio"],
        "max_mask_overlap_ratio": filtering["max_mask_overlap_ratio"],
        "min_frame_valid_depth_ratio": filtering["min_frame_valid_depth_ratio"],
        "depth_trunc_m": filtering["depth_trunc_m"],
        "min_points": filtering["min_points"],
        "ground_filter_enabled": filtering.get("ground_filter_enabled", True),
        "ground_reference_frame": filtering.get("ground_reference_frame", "base_link"),
        "ground_height_offset_m": filtering.get("ground_height_offset_m", 0.0),
        "ground_max_bottom_clearance_m": filtering.get("ground_max_bottom_clearance_m", 0.15),
        "ground_max_object_height_m": filtering.get("ground_max_object_height_m", 0.75),
        "ground_max_footprint_m": filtering.get("ground_max_footprint_m", 1.2),
        "max_object_extent_m": filtering.get("max_object_extent_m", 0.65),
        "max_object_distance_m": filtering.get("max_object_distance_m", 2.5),
        "association_distance_m": lifecycle["association_distance_m"],
        "association_max_size_ratio": lifecycle["association_max_size_ratio"],
        "embedding_similarity_threshold": lifecycle["embedding_similarity_threshold"],
        "association_position_weight": lifecycle["association_position_weight"],
        "label_switch_confidence_margin": lifecycle["label_switch_confidence_margin"],
        "min_label_confidence": labels["min_confidence"],
        "max_label_candidates_per_mask": labels["max_candidates_per_mask"],
        "label_recurrence_count_ratio": labels.get("recurrence_count_ratio", 3.0),
        "label_high_confidence_override_margin": labels.get("high_confidence_override_margin", 0.08),
        "allowed_label_aliases_json": json.dumps(labels.get("allowed_labels", {}), sort_keys=True),
        "excluded_labels": labels["excluded_labels"],
        "label_refinement_enabled": label_refinement["enabled"],
        "label_refinement_model": label_refinement["model"],
        "label_refinement_model_identity": label_refinement["model_identity"],
        "label_refinement_prompt": label_refinement["prompt"],
        "label_refinement_min_confidence": label_refinement["min_confidence"],
        "label_refinement_trigger_below_confidence": label_refinement["trigger_below_confidence"],
        "label_refinement_min_observations": label_refinement["min_observations"],
    }
    if mapping_backend == "service":
        parameters.update(
            {
                "sam_service": service("sam2_masks").endpoint,
                "ram_plus_service": service("ram_plus_tags").endpoint,
                "siglip2_service": service("siglip2_image").endpoint,
                "sam_service_instance_id": roles["sam2_masks"],
                "ram_plus_service_instance_id": roles["ram_plus_tags"],
                "siglip2_service_instance_id": roles["siglip2_image"],
                "sam_model_identity": identity("sam2_masks"),
                "ram_plus_model_identity": identity("ram_plus_tags"),
                "siglip2_model_identity": identity("siglip2_image"),
            }
        )
    if offline:
        parameters.update(
            {
                "service_wait_sec": perception["service_wait_sec"],
                "artifact_output_dir": persistence["artifact_output_dir"],
            }
        )
        return parameters

    parameters.update(
        {
            "cloud_map_topic": slam["cloud_map_topic"],
            "active_map_hash_topic": slam["active_map_hash_topic"],
            "localization_ready_topic": slam["localization_ready_topic"],
            "authoritative_map_odom_topic": slam["authoritative_map_odom_topic"],
            "mapping_backend": mapping_backend,
            "encode_text_service": service("siglip2_text").endpoint if mapping_backend == "service" else "",
            "gdino_confirmation_service": (
                service("gdino_confirmation").endpoint
                if mapping_backend == "service" and "gdino_confirmation" in roles
                else ""
            ),
            "semantic_map_topic": interfaces["semantic_map_topic"],
            "object_cloud_topic": interfaces["object_cloud_topic"],
            "query_service": interfaces["query_service"],
            "target_service": interfaces["target_service"],
            "sync_queue_size": queue["sync_queue_size"],
            "tf_timeout_sec": queue["tf_timeout_sec"],
            "processing_interval_sec": queue["processing_interval_sec"],
            "frame_queue_capacity": queue["frame_capacity"],
            "frame_queue_policy": queue["policy"],
            "stale_after_sec": lifecycle["stale_after_sec"],
            "move_stability_m": lifecycle["move_stability_m"],
            "move_confirmations": lifecycle["move_confirmations"],
            "track_state_topic": target_watch["track_state_topic"],
            "track_state_frame": target_watch["track_state_frame"],
            "track_state_updates_enabled": target_watch["track_state_updates_enabled"],
            "track_state_max_age_sec": target_watch["track_state_max_age_sec"],
            "track_state_max_covariance_m2": target_watch["track_state_max_covariance_m2"],
            "track_state_confirmation_gap_sec": target_watch["track_state_confirmation_gap_sec"],
            "track_state_persist_interval_sec": target_watch["track_state_persist_interval_sec"],
            "footprint_ready_topic": target_watch["footprint_ready_topic"],
            "obstacle_map_ready_topic": target_watch["obstacle_map_ready_topic"],
            "reachability_ready_topic": target_watch["reachability_ready_topic"],
        }
    )
    return parameters


def semantic_perception_nodes(config: dict, *, offline: bool = False) -> list:
    """Build generic model hosts required by the selected semantic workflow."""
    if config["semantic_mapping"].get("query_only", False):
        return []
    perception = config["semantic_mapping"]["perception"]
    if perception["mapping_backend"] == "embedded":
        return []
    roles = perception["semantic_roles"]
    selected_roles = _CONSTRUCTION_ROLES if offline else tuple(roles)
    instance_ids = {roles[role] for role in selected_roles}
    generation = parse_perception_runtime_config(config).configuration_generation(
        {role: roles[role] for role in _CONSTRUCTION_ROLES}
    )
    return generate_perception_model_nodes(
        config,
        instance_ids=instance_ids,
        configuration_generation=generation,
        require_semantic_identity=True,
    )
