"""Workflow-specific readiness gates with structured evidence."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReadinessEvidence:
    gate: str
    satisfied: bool
    detail: str


@dataclass(frozen=True)
class WorkflowReadiness:
    workflow: str
    evidence: tuple[ReadinessEvidence, ...]

    @property
    def ready(self) -> bool:
        return all(item.satisfied for item in self.evidence)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(item.detail for item in self.evidence if not item.satisfied)

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons)


def _result(workflow: str, gates: tuple[tuple[str, bool, str], ...]) -> WorkflowReadiness:
    return WorkflowReadiness(workflow, tuple(ReadinessEvidence(*gate) for gate in gates))


def offline_map_construction(
    *,
    sam2_ready: bool,
    sam2_identity_compatible: bool,
    ram_plus_ready: bool,
    ram_plus_identity_compatible: bool,
    siglip2_image_ready: bool,
    siglip2_image_identity_compatible: bool,
    rgbd_input_ready: bool,
    timestamped_tf_ready: bool,
    localization_ready: bool,
) -> WorkflowReadiness:
    return _result(
        "offline_map_construction",
        (
            ("sam2_ready", sam2_ready, "SAM2 image inference is not ready"),
            ("sam2_identity_compatible", sam2_identity_compatible, "SAM2 semantic identity is incompatible"),
            ("ram_plus_ready", ram_plus_ready, "RAM++ image inference is not ready"),
            (
                "ram_plus_identity_compatible",
                ram_plus_identity_compatible,
                "RAM++ semantic identity is incompatible",
            ),
            ("siglip2_image_ready", siglip2_image_ready, "SigLIP2 image inference is not ready"),
            (
                "siglip2_image_identity_compatible",
                siglip2_image_identity_compatible,
                "SigLIP2 image semantic identity is incompatible",
            ),
            ("rgbd_input_ready", rgbd_input_ready, "aligned RGB-D and CameraInfo input is not ready"),
            ("timestamped_tf_ready", timestamped_tf_ready, "timestamped camera transform is not ready"),
            ("localization_ready", localization_ready, "global localization input is not ready"),
        ),
    )


def online_map_construction(
    *,
    active_map_identity_compatible: bool,
    authoritative_map_odom: bool,
    cloud_map_ready: bool,
    queue_write_allowed: bool,
    **offline_gates: bool,
) -> WorkflowReadiness:
    offline = offline_map_construction(**offline_gates)
    return WorkflowReadiness(
        "online_map_construction",
        offline.evidence
        + (
            ReadinessEvidence(
                "active_map_identity_compatible",
                active_map_identity_compatible,
                "active SLAM map identity is incompatible",
            ),
            ReadinessEvidence(
                "authoritative_map_odom",
                authoritative_map_odom,
                "authoritative map-to-odom contract is not ready",
            ),
            ReadinessEvidence("cloud_map_ready", cloud_map_ready, "cloud_map contract is not ready"),
            ReadinessEvidence("queue_write_allowed", queue_write_allowed, "mapping queue/write admission is closed"),
        ),
    )


def structured_query(*, database_readable: bool, database_compatible: bool) -> WorkflowReadiness:
    return _result(
        "structured_query",
        (
            ("database_readable", database_readable, "semantic database is not readable"),
            ("database_compatible", database_compatible, "semantic database is incompatible"),
        ),
    )


def text_query(
    *,
    database_readable: bool,
    database_compatible: bool,
    siglip2_text_ready: bool,
    embedding_space_compatible: bool,
) -> WorkflowReadiness:
    structured = structured_query(database_readable=database_readable, database_compatible=database_compatible)
    return WorkflowReadiness(
        "text_query",
        structured.evidence
        + (
            ReadinessEvidence("siglip2_text_ready", siglip2_text_ready, "SigLIP2 text inference is not ready"),
            ReadinessEvidence(
                "embedding_space_compatible",
                embedding_space_compatible,
                "SigLIP2 text and stored image embedding spaces differ",
            ),
        ),
    )


def navigation_staging(
    *,
    object_action_admissible: bool,
    active_map_identity_compatible: bool,
    localization_ready: bool,
    authoritative_map_odom: bool,
    timestamped_tf_ready: bool,
    footprint_ready: bool,
    obstacle_map_ready: bool,
    reachability_ready: bool,
) -> WorkflowReadiness:
    return _result(
        "navigation_staging",
        (
            (
                "object_action_admissible",
                object_action_admissible,
                "semantic object is not action-admissible",
            ),
            (
                "active_map_identity_compatible",
                active_map_identity_compatible,
                "active SLAM map identity is incompatible",
            ),
            ("localization_ready", localization_ready, "global localization is not ready"),
            (
                "authoritative_map_odom",
                authoritative_map_odom,
                "authoritative map-to-odom contract is not ready",
            ),
            ("timestamped_tf_ready", timestamped_tf_ready, "timestamped camera transform is not ready"),
            ("footprint_ready", footprint_ready, "robot footprint contract is not ready"),
            ("obstacle_map_ready", obstacle_map_ready, "obstacle map contract is not ready"),
            ("reachability_ready", reachability_ready, "Nav2 reachability checker is not ready"),
        ),
    )


def manipulation_confirmation(
    *,
    navigation: WorkflowReadiness,
    object_confirmation_admissible: bool,
    gdino_ready: bool,
    confirmation_sam2_ready: bool,
    confirmation_result_fresh: bool,
) -> WorkflowReadiness:
    if navigation.workflow != "navigation_staging":
        raise ValueError("manipulation confirmation requires navigation staging evidence")
    return WorkflowReadiness(
        "manipulation_confirmation",
        navigation.evidence
        + (
            ReadinessEvidence(
                "object_confirmation_admissible",
                object_confirmation_admissible,
                "semantic object is not admissible for fresh confirmation",
            ),
            ReadinessEvidence("gdino_ready", gdino_ready, "Grounding DINO confirmation inference is not ready"),
            ReadinessEvidence(
                "confirmation_sam2_ready", confirmation_sam2_ready, "confirmation SAM2 inference is not ready"
            ),
            ReadinessEvidence(
                "confirmation_result_fresh",
                confirmation_result_fresh,
                "fresh manipulation confirmation has not run",
            ),
        ),
    )


def read_only_diagnostics(*, database_diagnostic_open: bool) -> WorkflowReadiness:
    return _result(
        "read_only_diagnostics",
        (("database_diagnostic_open", database_diagnostic_open, "semantic database is not open for diagnostics"),),
    )
