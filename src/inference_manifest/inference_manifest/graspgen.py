"""The GraspGen execution contract shared by the exporter, the packager and the session.

GraspGen is not one compiled graph. ``model_utils.graspgen_export.export_onnx`` emits
eight subgraphs, ``perception_service.package_graspgen_ascend_bundle`` binds them into
eight manifest roles, and ``perception_service.graspgen_session`` drives those roles in a
fixed order while computing the PointNet++ sampling geometry on the host between them.
Each of the three used to carry its own copy of the role order and of the sampling
constants, so a reordered role or a different neighbourhood size would agree with itself
everywhere it was written down and only surface as a shape failure on the device.

They read this module instead. ``inference_manifest`` is the one package all three
already depend on, and the manifest is where the contract is finally written down, so the
definition belongs here rather than in any one of them.
"""

from __future__ import annotations

from typing import Any

from inference_manifest.models import HOST_SEMANTIC_PREFIX, INTERNAL_SEMANTIC_PREFIX

# Bumped whenever the role set, the role order, the sampling geometry or the binding
# semantics change in a way that makes previously exported subgraphs unusable.
# ``export_onnx`` stamps it into ``graspgen.onnx.json`` and the packager refuses to package
# an ONNX manifest that was produced against a different contract, so a stale export cannot
# be compiled into OMs that the session would then drive with the wrong geometry.
#
# 2: GraspGen became a perception model. The host-computed tensors moved to the ``host.``
#    namespace and the grasp poses stopped masquerading as a discriminator-head output.
GRASPGEN_CONTRACT_VERSION = 2

GRASPGEN_EXECUTION: tuple[str, ...] = (
    "generator_sa1",
    "generator_sa2",
    "generator_encoder_head",
    "discriminator_sa1",
    "discriminator_sa2",
    "discriminator_encoder_head",
    "denoiser",
    "discriminator_head",
)

# PointNet++ set abstraction as the GraspGen checkpoints were trained: two sampled stages
# (furthest point sampling, then a ball query into a fixed-size neighbourhood) feeding an
# encoder head that consumes everything stage two produced in a single group. The compiled
# OMs have these counts baked into their static input shapes, so the exporter must trace
# with them and the backend must group points with them.
GRASPGEN_NPOINTS: tuple[int, int] = (256, 64)
GRASPGEN_RADII: tuple[float, float] = (0.02, 0.04)
GRASPGEN_NSAMPLES: tuple[int, int] = (64, 128)
GRASPGEN_FPS_START_INDEX = 0
GRASPGEN_BALL_QUERY_ORDER = "input_index"

# The service contract: one object point cloud in, a batch of 4x4 grasp poses and their
# confidences out. These are the only GraspGen semantics a caller ever sees, and they are
# what the perception ``ModelDescriptor`` declares.
GRASPGEN_POINT_CLOUD_SEMANTIC = "observation.object_points"
GRASPGEN_POSE_SEMANTIC = "grasp.poses"
GRASPGEN_CONFIDENCE_SEMANTIC = "grasp.confidence"
GRASPGEN_GRASP_OUTPUTS = frozenset({GRASPGEN_POSE_SEMANTIC, GRASPGEN_CONFIDENCE_SEMANTIC})

# The encoder embeddings are the only tensors handed directly from one OM to another, so
# they keep the ``internal.`` namespace and travel by device pointer over a declared link.
GRASPGEN_GENERATOR_EMBEDDING = f"{INTERNAL_SEMANTIC_PREFIX}graspgen.generator_embedding"
GRASPGEN_DISCRIMINATOR_EMBEDDING = f"{INTERNAL_SEMANTIC_PREFIX}graspgen.discriminator_embedding"

# Everything else the roles consume or produce is computed on the host between them: the
# PointNet++ neighbourhoods built by furthest point sampling and ball query, the diffusion
# state the session iterates, the grasp transforms it integrates, and the head's raw
# logits. None of them has an in-graph producer and none crosses the service boundary,
# which is exactly what the ``host.`` namespace denotes.
_HOST = f"{HOST_SEMANTIC_PREFIX}graspgen."
GRASPGEN_GROUPED_POINTS = f"{_HOST}grouped_points"
GRASPGEN_STAGE2_FEATURES = f"{_HOST}stage2_features"
GRASPGEN_GLOBAL_FEATURES = f"{_HOST}global_features"
GRASPGEN_STAGE_FEATURES = f"{_HOST}features"
GRASPGEN_DIFFUSION_SAMPLE = f"{_HOST}diffusion_sample"
GRASPGEN_DIFFUSION_TIME = f"{_HOST}diffusion_time"
GRASPGEN_PREDICTED_NOISE = f"{_HOST}predicted_noise"
GRASPGEN_GRASP_RT = f"{_HOST}grasp_rt"
GRASPGEN_GRASP_LOGITS = f"{_HOST}grasp_logits"
GRASPGEN_GRASP_SCORES = f"{_HOST}grasp_scores"

GRASPGEN_HOST_SEMANTICS = frozenset(
    {
        GRASPGEN_GROUPED_POINTS,
        GRASPGEN_STAGE2_FEATURES,
        GRASPGEN_GLOBAL_FEATURES,
        GRASPGEN_STAGE_FEATURES,
        GRASPGEN_DIFFUSION_SAMPLE,
        GRASPGEN_DIFFUSION_TIME,
        GRASPGEN_PREDICTED_NOISE,
        GRASPGEN_GRASP_RT,
        GRASPGEN_GRASP_LOGITS,
        GRASPGEN_GRASP_SCORES,
    }
)


def graspgen_input_semantics(role: str) -> dict[str, str]:
    """Map a role's runtime input slots, in ABI index order, onto manifest semantics."""
    if role in {"generator_sa1", "discriminator_sa1"}:
        return {"grouped_features": GRASPGEN_GROUPED_POINTS}
    if role in {"generator_sa2", "discriminator_sa2"}:
        return {"grouped_features": GRASPGEN_STAGE2_FEATURES}
    if role in {"generator_encoder_head", "discriminator_encoder_head"}:
        return {"grouped_features": GRASPGEN_GLOBAL_FEATURES}
    if role == "denoiser":
        return {
            "object_embedding": GRASPGEN_GENERATOR_EMBEDDING,
            "sample": GRASPGEN_DIFFUSION_SAMPLE,
            "timestep": GRASPGEN_DIFFUSION_TIME,
        }
    if role == "discriminator_head":
        return {
            "object_embedding": GRASPGEN_DISCRIMINATOR_EMBEDDING,
            "grasp_rt": GRASPGEN_GRASP_RT,
        }
    raise ValueError(f"unknown graspgen role {role!r}")


def graspgen_output_semantics(role: str) -> dict[str, str]:
    """Map a role's runtime output slots, in ABI index order, onto manifest semantics."""
    if role in {"generator_sa1", "generator_sa2", "discriminator_sa1", "discriminator_sa2"}:
        return {"features": GRASPGEN_STAGE_FEATURES}
    if role == "generator_encoder_head":
        return {"object_embedding": GRASPGEN_GENERATOR_EMBEDDING}
    if role == "discriminator_encoder_head":
        return {"object_embedding": GRASPGEN_DISCRIMINATOR_EMBEDDING}
    if role == "denoiser":
        return {"predicted_noise": GRASPGEN_PREDICTED_NOISE}
    if role == "discriminator_head":
        # The head emits per-grasp logits and their sigmoid, both shaped [batch, 1]. The
        # published poses are integrated on the host from the denoiser samples and the
        # published confidences are the squeezed scores, so neither service output is an
        # output slot on any OM.
        return {"logits": GRASPGEN_GRASP_LOGITS, "confidence": GRASPGEN_GRASP_SCORES}
    raise ValueError(f"unknown graspgen role {role!r}")


def graspgen_geometry(*, include_head_stage: bool = False) -> dict[str, Any]:
    """Render the sampling geometry in the form a manifest carries it.

    ``include_head_stage`` appends the encoder head's null stage: it groups everything
    stage two produced, so it has no sampling count, radius or neighbourhood size. The
    ONNX manifest lists it to keep one entry per exported set-abstraction stage, while the
    unified manifest and the backend describe only the two sampled stages.
    """
    tail: list[Any] = [None] if include_head_stage else []
    return {
        "npoints": [*GRASPGEN_NPOINTS, *tail],
        "radii": [*GRASPGEN_RADII, *tail],
        "nsamples": [*GRASPGEN_NSAMPLES, *tail],
        "fps_start_index": GRASPGEN_FPS_START_INDEX,
        "ball_query_order": GRASPGEN_BALL_QUERY_ORDER,
    }
