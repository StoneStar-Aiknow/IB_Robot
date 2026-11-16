"""GraspGen inference backend: the in-process unified-runtime pipeline (ROS-free)."""

import logging
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from inference_service.runtime_composition import (
    require_runtime_dependencies,
)
from inference_service.unified_runtime import RegistrySet, RuntimeProviders

logger = logging.getLogger(__name__)

LOCAL_BACKEND_REQUIRES_CUDA = "GraspGen local backend requires CUDA for the manifest-selected Torch deployment."
_DEFAULT_MAX_INFERENCE_GRASPS_PER_BATCH = 1000
VALID_INFERENCE_BACKENDS = {"local_cuda", "ascend_local"}


def backend_latency_ms(result: object) -> float:
    """Return backend latency for either unified-runtime latency representation."""
    latency = getattr(result, "latency", None)
    backend_ms = getattr(latency, "backend_ms", None)
    if backend_ms is not None:
        return float(backend_ms)
    latency_ms = getattr(result, "latency_ms", None)
    if latency_ms is None:
        raise TypeError("GraspGen runtime result does not expose latency_ms")
    return float(latency_ms)


def load_grasp_metadata(config_path: Path) -> tuple[str, int]:
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    data = config.get("data") if isinstance(config, dict) else None
    if not isinstance(data, dict):
        raise ValueError(f"GraspGen config is missing a data mapping: {config_path}")
    gripper_name = str(data.get("gripper_name") or "").strip()
    if not gripper_name:
        raise ValueError(f"GraspGen config is missing data.gripper_name: {config_path}")
    point_count = int(data.get("num_points", 2048))
    if point_count <= 0:
        raise ValueError(f"GraspGen config data.num_points must be positive: {config_path}")
    return gripper_name, point_count


def cuda_status() -> str:
    import torch

    return (
        f"torch={torch.__version__}, "
        f"torch.version.cuda={torch.version.cuda}, "
        f"torch.cuda.is_available()={torch.cuda.is_available()}"
    )


def _find_workspace_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "src").is_dir() and (parent / ".git").exists():
            return parent
        if (parent / "models").is_dir():
            return parent
    return Path(".").resolve()


WORKSPACE_ROOT = _find_workspace_root()
DEFAULT_MODEL_DIR = WORKSPACE_ROOT / "models" / "grasp"


def graspgen_inference_batches(total_grasps: int, max_batch_size: int) -> list[int]:
    total = max(1, int(total_grasps))
    batch = max(1, int(max_batch_size))
    batches = []
    remaining = total
    while remaining > 0:
        current = min(batch, remaining)
        batches.append(current)
        remaining -= current
    return batches


class LocalPipelineBackend:
    """In-process GraspGen inference through the unified generic pipeline.

    GraspGen is a ``tensor_model/graspgen/generate_grasps`` model, so its semantics live in
    ``perception_service.GraspGenAdapter`` (point-cloud preparation and grasp
    decoding). The selected model session owns either Torch CUDA execution or the
    eight-role Ascend execution. This class only drives that pair in-process;
    it holds no GraspGen semantics of its own.
    """

    def __init__(
        self,
        manifest_path: str,
        *,
        deployment_name: str = "ascend_310p",
        device_id: int = 0,
        random_seed: int | None = None,
        registry_set: RegistrySet | None = None,
        providers: RuntimeProviders | None = None,
    ):
        if not manifest_path.strip():
            raise ValueError("local GraspGen manifest_path must not be empty")
        if device_id < 0:
            raise ValueError("local GraspGen device_id must be a non-negative integer")
        self._manifest_path = manifest_path.strip()
        self._deployment_name = deployment_name.strip() or "ascend_310p"
        self._device_id = int(device_id)
        self._random_seed = random_seed
        registry_set, providers = require_runtime_dependencies(
            registry_set,
            providers,
            owner=type(self).__name__,
        )
        self._registry_set = registry_set
        self._providers = providers
        self._session = None
        self._pipeline = None
        self._adapter = None
        self._sample_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        from inference_manifest import load_inference_manifest
        from inference_service.backends.types import RuntimeContext
        from inference_service.unified_runtime import (
            ModelRuntimeHandle,
            RuntimeAssembly,
        )
        from perception_service.graspgen_adapter import GraspGenAdapter

        bundle_root = Path(self._manifest_path).expanduser().resolve()
        manifest_root = bundle_root.parent if bundle_root.name == "inference_manifest.json" else bundle_root
        validated = load_inference_manifest(manifest_root, self._deployment_name)
        model = validated.manifest.model
        if (model.interface, model.model_type, model.operation) != (
            "tensor_model",
            "graspgen",
            "generate_grasps",
        ):
            raise ValueError(
                "local pipeline requires tensor_model/graspgen/generate_grasps, "
                f"got {model.interface}/{model.model_type}/{model.operation}"
            )
        adapter = GraspGenAdapter.from_bundle(validated.bundle_root, model.semantic_identity, model=model)
        adapter.validate_identity(model.semantic_identity)
        adapter.validate_deployment(self._deployment_name)

        runtime_options: dict[str, object] = {}
        if self._random_seed is not None and validated.deployment.backend == "ascend":
            runtime_options["random_seed"] = self._random_seed
        context = RuntimeContext(
            validated_manifest=validated,
            runtime_options=runtime_options,
            runtime_profile=validated.runtime_profile,
        )
        session = self._registry_set.session_builder_registry.create(
            context,
            allowed_deployments=adapter.identity.supported_deployments,
            backend_registry=self._registry_set.backend_registry,
            providers=self._providers,
            builder_options={"adapter": adapter},
        )
        pipeline = ModelRuntimeHandle(
            RuntimeAssembly(
                runtime_executor=session,
                session=session,
                execution_contract="request-direct",
                runtime_id=f"manipulation-graspgen-{validated.deployment.backend}-local",
                load_context=context,
            )
        )
        pipeline.load()
        self._adapter = adapter
        self._session = session
        self._pipeline = pipeline
        logger.info(
            "LocalPipelineBackend ready (manifest=%s, deployment=%s, backend=%s, point_count=%d)",
            manifest_root,
            self._deployment_name,
            validated.deployment.backend,
            adapter.config.point_count,
        )

    def sample(
        self,
        object_pc: np.ndarray,
        *,
        grasp_threshold: float,
        num_grasps: int,
        topk_num_grasps: int,
        min_grasps: int = 80,
        max_tries: int = 4,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        self._ensure_loaded()
        from inference_service.unified_runtime import ExecutionContext, ModelRequest
        from model_utils.graspgen_contract import GRASPGEN_CONFIDENCE_SEMANTIC, GRASPGEN_POSE_SEMANTIC

        points = np.asarray(object_pc, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError(f"object_pc must have shape [N,3] with N > 0, got {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("object_pc contains non-finite values")

        # The adapter owns cleaning, capping, centring and kappa scaling, and hands
        # back the centre so the decoded grasps return to the camera frame.
        inputs, object_center = self._adapter.prepare(points)
        batches = graspgen_inference_batches(num_grasps, _DEFAULT_MAX_INFERENCE_GRASPS_PER_BATCH)
        if len(batches) > 1:
            logger.info(
                "Running local GraspGen inference in %d batches: total=%d max_batch=%d",
                len(batches),
                num_grasps,
                _DEFAULT_MAX_INFERENCE_GRASPS_PER_BATCH,
            )

        all_poses = []
        all_confidence = []
        backend_seconds = 0.0
        started = time.perf_counter()
        # Keep a complete multi-batch request on one uninterrupted random stream.
        # The session owns a persistent Generator, so each infer call consumes
        # fresh diffusion noise without resetting the configured seed.
        with self._sample_lock:
            for index, batch_size in enumerate(batches, start=1):
                batch_poses = []
                batch_confidence = []
                target_count = min(min_grasps, batch_size)
                for attempt in range(1, max_tries + 1):
                    result = self._pipeline.execute(
                        ModelRequest(inputs),
                        ExecutionContext(f"graspgen_local_{uuid.uuid4().hex}"),
                    )
                    # Current GraspGen OM files have a static batch of 1000. For a
                    # partial final batch, retain only the requested number of raw
                    # samples so num_grasps keeps the same meaning as the CUDA path.
                    if batch_size < len(np.asarray(result.outputs[GRASPGEN_CONFIDENCE_SEMANTIC]).reshape(-1)):
                        result = replace(
                            result,
                            outputs={
                                GRASPGEN_POSE_SEMANTIC: np.asarray(result.outputs[GRASPGEN_POSE_SEMANTIC])[:batch_size],
                                GRASPGEN_CONFIDENCE_SEMANTIC: np.asarray(
                                    result.outputs[GRASPGEN_CONFIDENCE_SEMANTIC]
                                ).reshape(-1)[:batch_size],
                            },
                        )
                    candidates = self._adapter.postprocess(
                        result,
                        object_center=object_center,
                        max_grasps=0,
                        min_confidence=float(grasp_threshold),
                    )
                    if candidates:
                        batch_poses.append(np.stack([candidate.pose_matrix for candidate in candidates]))
                        batch_confidence.append(
                            np.asarray([candidate.confidence for candidate in candidates], dtype=np.float32)
                        )
                    backend_seconds += backend_latency_ms(result) / 1000.0
                    accepted = sum(len(values) for values in batch_poses)
                    logger.info(
                        "LocalPipelineBackend batch %d/%d attempt %d/%d kept %d grasps, total=%d",
                        index,
                        len(batches),
                        attempt,
                        max_tries,
                        len(candidates),
                        accepted,
                    )
                    if accepted >= target_count:
                        break
                all_poses.extend(batch_poses)
                all_confidence.extend(batch_confidence)

        inference_seconds = time.perf_counter() - started
        if not all_poses:
            return (
                np.zeros((0, 4, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                inference_seconds,
            )
        poses = np.concatenate(all_poses, axis=0)
        confidence = np.concatenate(all_confidence, axis=0)
        # Each batch arrives sorted on its own, so re-sort across batches to keep the
        # adapter's confidence-descending contract whether or not topk trims the tail.
        order = np.argsort(-confidence, kind="stable")
        if topk_num_grasps > 0:
            order = order[:topk_num_grasps]
        poses = poses[order]
        confidence = confidence[order]
        logger.info(
            "LocalPipelineBackend completed: batches=%d grasps=%d wall_s=%.3f backend_s=%.3f denoiser_steps=%d",
            len(batches),
            len(confidence),
            inference_seconds,
            backend_seconds,
            self._adapter.config.diffusion_steps,
        )
        return poses, confidence, inference_seconds

    def warmup(self, object_pc: np.ndarray) -> None:
        """Run inference without consuming the configured request random stream."""
        self._ensure_loaded()
        assert self._session is not None
        random_generator = getattr(self._session, "_random", None)  # noqa: SLF001 - paired session integration.
        random_state = None if random_generator is None else deepcopy(random_generator.bit_generator.state)
        try:
            self.sample(
                object_pc,
                grasp_threshold=1.0,
                num_grasps=1,
                topk_num_grasps=1,
            )
        finally:
            if random_generator is not None and random_state is not None:
                random_generator.bit_generator.state = random_state


# Compatibility name for callers that only select the Ascend deployment.
AscendLocalBackend = LocalPipelineBackend
