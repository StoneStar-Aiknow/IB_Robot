"""Typed perception plugins backed exclusively by shared model sessions."""

from __future__ import annotations

import importlib
import itertools
import json
from collections.abc import Callable

import numpy as np
from cv_bridge import CvBridge

from ibrobot_msgs.msg import Detection2D, DetectionArray
from inference_manifest import CompiledDeployment, TorchDeployment
from inference_service.backends import RuntimeContext
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.model_service_plugin import ModelServicePlugin, PluginRuntimeStatus
from inference_service.model_sessions import AscendOmModelSession, ModelSession, TorchModelSession

from .graspgen_adapter import GraspGenAdapter
from .graspgen_session import GraspGenAscendSession
from .model_contracts import (
    point_cloud_xyz,
    rank_detections,
    validate_detection_batch,
    validate_image_message,
    validate_mask_batch,
    validate_text_batch,
)
from .ram_plus_adapter import RAMPlusAdapter
from .semantic_model_adapters import (
    GroundingDINOAdapter,
    GroundingDINORawAdapter,
    SAM2Adapter,
    SAM2PromptAdapter,
    SigLIP2ImageAdapter,
    SigLIP2TextAdapter,
)


def _detection_array(bridge, header, records) -> DetectionArray:
    detections = []
    for record in records:
        detection = Detection2D()
        detection.header = header
        detection.confidence = float(getattr(record, "score", getattr(record, "confidence", 0.0)))
        detection.bbox = np.asarray(record.bbox_xyxy, dtype=float).tolist()
        if record.mask is not None:
            detection.mask = bridge.cv2_to_imgmsg((record.mask > 0).astype(np.uint8) * 255, encoding="mono8")
            detection.mask.header = header
        detection.label = str(getattr(record, "label", ""))
        detections.append(detection)
    return DetectionArray(header=header, detections=detections)


def _filter_masks(records, image_area: int, max_masks: int, min_pixels: int, min_area_ratio: float, max_overlap: float):
    candidates = [
        (index, record)
        for index, record in enumerate(records)
        if record.area >= min_pixels and record.area / image_area >= min_area_ratio
    ]
    candidates.sort(key=lambda item: (-item[1].score, -item[1].area, item[0]))
    accepted = []
    for _, record in candidates:
        mask = record.mask > 0
        if any(np.count_nonzero(mask & (other.mask > 0)) / max(1, record.area) > max_overlap for other in accepted):
            continue
        accepted.append(record)
        if max_masks and len(accepted) >= max_masks:
            break
    return accepted


def _load_torch_module(family: str, context: RuntimeContext):
    identity_path = context.validated_manifest.bundle_root / "assets" / "adapter.json"
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        module_name, function_name = identity["torch_module_loader"].split(":", 1)
        loader = getattr(importlib.import_module(module_name), function_name)
    except (AttributeError, ImportError, KeyError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"{family} Torch bundle has no loadable family module integration in {identity_path}: {exc}"
        ) from exc
    module = loader(context)
    if not callable(module):
        raise TypeError(f"{family} Torch family module loader did not return a callable")
    return module


def _ascend_device_id(family: str, adapter, deployment, options, allowed: set[str]) -> int:
    if deployment.backend != "ascend" or not adapter.compiled_abi_finalized:
        raise RuntimeError(f"{family} compiled adapter ABI is not finalized; deployment remains not-ready")
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(f"unknown Ascend runtime options: {unknown}")
    return int(options.get("device_id", 0))


def _new_session(family: str, adapter, validated, options) -> ModelSession:
    deployment = validated.deployment
    if isinstance(deployment, CompiledDeployment):
        device_id = _ascend_device_id(family, adapter, deployment, options, {"acl_config_path", "device_id"})
        return AscendOmModelSession(device_id=device_id)
    if isinstance(deployment, TorchDeployment):
        if options:
            raise ValueError(
                f"Torch family plugins do not select backend/device or accept runtime options: {sorted(options)}"
            )
        return TorchModelSession(lambda context: _load_torch_module(family, context))
    raise RuntimeError(f"{family} deployment type is unsupported and has no fallback")


def _new_graspgen_session(family: str, adapter, validated, options) -> ModelSession:
    """Build the host-orchestrated GraspGen session.

    GraspGen has no Torch deployment: the eight OM sub-graphs and the host math between
    them are one contract, so an un-compiled bundle has nothing to fall back to. The extra
    ``random_seed`` option makes the denoising loop reproducible in tests and replays.
    """
    deployment = validated.deployment
    if not isinstance(deployment, CompiledDeployment):
        raise RuntimeError(f"{family} requires a compiled Ascend deployment and has no Torch fallback")
    device_id = _ascend_device_id(family, adapter, deployment, options, {"acl_config_path", "device_id", "random_seed"})
    return GraspGenAscendSession(device_id=device_id, config=adapter.config)


class _SessionPlugin(ModelServicePlugin):
    service_type = ""
    family = ""
    adapter_class = None
    _session_factory: Callable = staticmethod(_new_session)

    def __init__(self, host, validated, options) -> None:
        model = validated.manifest.model
        if model.kind != "perception" or model.family != self.family:
            raise ValueError(f"plugin requires perception family {self.family!r}, got {model.kind!r}/{model.family!r}")
        if any(name in options for name in ("backend", "device", "model_backend", "fallback")):
            raise ValueError(
                "plugins accept only a validated named deployment; raw backend/device/fallback is forbidden"
            )
        self.host = host
        self.bridge = CvBridge()
        self.validated = validated
        self.adapter = self.adapter_class.from_bundle(
            validated.bundle_root, model.semantic_identity, model=model, deployment=validated.deployment
        )
        self.adapter.validate_identity(model.semantic_identity)
        self._closed = False
        self._requests = itertools.count(1)
        self.session = self._session_factory(self.family, self.adapter, validated, options)
        if not isinstance(self.session, ModelSession):
            raise TypeError("perception plugins require a ModelSession")
        try:
            self.session.load(RuntimeContext(validated_manifest=validated, runtime_options=options))
        except Exception:
            self.session.close()
            self._closed = True
            raise

    def _infer(self, inputs):
        return self.session.infer(
            NamedTensorRequest(
                request_id=f"{self.family}-{next(self._requests)}",
                inputs=inputs,
                metadata={"service_type": self.service_type},
            )
        )

    def image_rgb(self, image):
        validate_image_message(image)
        return np.asarray(self.bridge.imgmsg_to_cv2(image, desired_encoding="rgb8"), dtype=np.uint8)

    def runtime_status(self) -> PluginRuntimeStatus:
        health = self.session.health()
        state = health.state.value if hasattr(health.state, "value") else str(health.state)
        runtime_version = self.session.runtime_version
        ready = health.ready and bool(runtime_version)
        return PluginRuntimeStatus(
            state=state,
            ready=ready,
            failure_reason=""
            if ready
            else (
                "loaded model runtime did not expose a version"
                if health.ready
                else (health.message or health.reason_code or f"model runtime is {state}")
            ),
            metadata={"runtime_version": runtime_version} if runtime_version else {},
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.session.close()


class RAMPlusRecognizeTagsPlugin(_SessionPlugin):
    service_type = "ibrobot_msgs/srv/RecognizeTags"
    family = "ram_plus"
    adapter_class = RAMPlusAdapter

    def handle(self, request, response) -> str:
        result = self._infer(self.adapter.preprocess(self.image_rgb(request.image)))
        tags = self.adapter.postprocess(result, score_threshold=float(request.score_threshold))
        response.tags = [tag.label for tag in tags]
        response.scores = [tag.score for tag in tags]
        return f"recognized {len(tags)} tags"


class SAM2GenerateMasksPlugin(_SessionPlugin):
    service_type = "ibrobot_msgs/srv/GenerateMasks"
    family = "sam2"
    adapter_class = SAM2Adapter

    def handle(self, request, response) -> str:
        image = self.image_rgb(request.image)
        result = self._infer(self.adapter.preprocess(image))
        records = _filter_masks(
            self.adapter.postprocess(result, image_shape=image.shape[:2]),
            image.shape[0] * image.shape[1],
            int(request.max_masks),
            int(request.min_mask_pixels),
            max(0.0, float(request.min_mask_area_ratio)),
            max(0.0, float(request.max_overlap_ratio)),
        )
        response.detections = _detection_array(self.bridge, request.image.header, records)
        return f"generated {len(records)} masks"


class SigLIP2EncodeEmbeddingsPlugin(_SessionPlugin):
    service_type = "ibrobot_msgs/srv/EncodeEmbeddings"
    family = "siglip2"
    adapter_class = SigLIP2ImageAdapter

    def handle(self, request, response) -> str:
        from ibrobot_msgs.msg import MaskEmbedding

        validate_mask_batch(request.image, request.masks)
        if len(request.candidate_labels) > 16:
            raise ValueError("candidate-label batch exceeds limit 16")
        if any(not label.strip() for label in request.candidate_labels):
            raise ValueError("candidate labels must not be empty")
        image = self.image_rgb(request.image)
        masks = [self.bridge.imgmsg_to_cv2(mask, desired_encoding="mono8") for mask in request.masks]
        labels = list(request.candidate_labels)
        result = self._infer(self.adapter.preprocess((image, masks, labels)))
        records = self.adapter.postprocess(result, candidate_labels=labels)
        if len(records) != len(masks):
            raise RuntimeError("SigLIP2 returned a different embedding count than the mask batch")
        response.results = [
            MaskEmbedding(
                mask_index=record.mask_index,
                embedding=record.embedding.astype(float).tolist(),
                embedding_dim=len(record.embedding),
                matched_label=record.matched_label,
                matched_score=record.matched_score,
                success=True,
            )
            for record in records
        ]
        return f"encoded {len(records)} masks"


class SigLIP2EncodeTextPlugin(_SessionPlugin):
    service_type = "ibrobot_msgs/srv/EncodeText"
    family = "siglip2"
    adapter_class = SigLIP2TextAdapter

    def handle(self, request, response) -> str:
        from ibrobot_msgs.msg import TextEmbedding

        validate_text_batch(request.texts)
        result = self._infer(self.adapter.preprocess(request.texts))
        features = self.adapter.postprocess(result)
        if len(features) != len(request.texts):
            raise RuntimeError("SigLIP2 returned a different embedding count than the text batch")
        response.results = [
            TextEmbedding(
                text_index=index,
                embedding=embedding.astype(float).tolist(),
                embedding_dim=len(embedding),
                success=True,
            )
            for index, embedding in enumerate(features)
        ]
        return f"encoded {len(features)} texts"


class GroundingDetectPlugin(_SessionPlugin):
    service_type = "ibrobot_msgs/srv/GroundingDetect"
    family = "grounding_dino"
    adapter_class = GroundingDINOAdapter

    def handle(self, request, response) -> str:
        if not request.text_prompt.strip():
            raise ValueError("text prompt must not be empty")
        image = self.image_rgb(request.image)
        result = self._infer(
            self.adapter.preprocess(
                (image, request.text_prompt, float(request.box_threshold), float(request.text_threshold))
            )
        )
        records = rank_detections(
            [
                record
                for record in self.adapter.postprocess(
                    result,
                    image_shape=image.shape[:2],
                    labels=(request.text_prompt,),
                )
                if record.confidence >= (float(request.box_threshold) or 0.35)
            ]
        )
        response.detections = _detection_array(self.bridge, request.image.header, records)
        return f"confirmed {len(records)} detections"


class GroundingDINORawDetectPlugin(_SessionPlugin):
    """Text detection service for a compiled raw Grounding-DINO deployment."""

    service_type = "ibrobot_msgs/srv/GroundingDetect"
    family = "grounding_dino_raw"
    adapter_class = GroundingDINORawAdapter

    def handle(self, request, response) -> str:
        if not request.text_prompt.strip():
            raise ValueError("text prompt must not be empty")
        image = self.image_rgb(request.image)
        result = self._infer(
            self.adapter.preprocess(
                (image, request.text_prompt, float(request.box_threshold), float(request.text_threshold))
            )
        )
        records = rank_detections(
            self.adapter.postprocess(
                result,
                image_shape=image.shape[:2],
                prompt=request.text_prompt,
                box_threshold=float(request.box_threshold),
                text_threshold=float(request.text_threshold),
            )
        )
        response.detections = _detection_array(self.bridge, request.image.header, records)
        return f"detected {len(records)} boxes"


class SegmentDetectionsPlugin(_SessionPlugin):
    """Fill masks for detections using a manifest-bound SAM2 box-prompt service."""

    service_type = "ibrobot_msgs/srv/SegmentDetections"
    family = "sam2_prompt"
    adapter_class = SAM2PromptAdapter

    def handle(self, request, response) -> str:
        source = list(request.detections.detections)
        validate_detection_batch(source)
        if not source:
            response.detections = DetectionArray(header=request.image.header, detections=[])
            return "segmented 0 detections"
        image = self.image_rgb(request.image)
        masks = self._segment_in_manifest_batches(image, source)
        output = []
        for detection, mask in zip(source, masks, strict=True):
            record = Detection2D()
            record.header = request.image.header
            record.label = detection.label
            record.confidence = detection.confidence
            record.bbox = detection.bbox
            record.mask = self.bridge.cv2_to_imgmsg((mask > 0).astype(np.uint8) * 255, encoding="mono8")
            record.mask.header = request.image.header
            output.append(record)
        response.detections = DetectionArray(header=request.image.header, detections=output)
        return f"segmented {len(output)} detections"

    def _segment_in_manifest_batches(self, image, source) -> list[np.ndarray]:
        """Segment every detection, one compiled decoder batch at a time.

        The compiled SAM2 decoder accepts a fixed number of box prompts while
        `GroundingDetect` may legitimately confirm up to `MAX_DETECTIONS` objects, so
        the service slices the request into manifest-sized chunks and concatenates the
        masks back in the original detection order instead of truncating the tail.
        """
        batch_size = int(getattr(self.adapter, "batch_size", 0)) or len(source)
        masks: list[np.ndarray] = []
        for start in range(0, len(source), batch_size):
            chunk = source[start : start + batch_size]
            result = self._infer(self.adapter.preprocess((image, [detection.bbox for detection in chunk])))
            chunk_masks = self.adapter.postprocess(result, image_shape=image.shape[:2], count=len(chunk))
            if len(chunk_masks) != len(chunk):
                raise RuntimeError(
                    f"SAM2 box-prompt deployment returned {len(chunk_masks)} masks for {len(chunk)} detections"
                )
            masks.extend(chunk_masks)
        return masks


class GraspGenGenerateGraspsPlugin(_SessionPlugin):
    service_type = "ibrobot_msgs/srv/GenerateGrasps"
    family = "graspgen"
    adapter_class = GraspGenAdapter
    _session_factory: Callable = staticmethod(_new_graspgen_session)

    def handle(self, request, response) -> str:
        from ibrobot_msgs.msg import GraspCandidate, GraspCandidateArray

        inputs, object_center = self.adapter.prepare(point_cloud_xyz(request.object_points))
        result = self._infer(inputs)
        candidates = self.adapter.postprocess(
            result,
            object_center=object_center,
            max_grasps=int(request.max_grasps),
            min_confidence=float(request.min_confidence),
        )
        header = request.object_points.header
        # Width and collision fields stay at their defaults: GraspGen scores a pose, it
        # does not measure the gripper aperture or clear the scene. manipulation_execution
        # fills those in from its own geometry pass.
        response.grasps = GraspCandidateArray(
            header=header,
            grasps=[
                GraspCandidate(
                    header=header,
                    pose_matrix=candidate.pose_matrix.reshape(16).astype(float).tolist(),
                    confidence=candidate.confidence,
                )
                for candidate in candidates
            ],
        )
        return f"generated {len(candidates)} grasps"


__all__ = [
    "GraspGenGenerateGraspsPlugin",
    "GroundingDetectPlugin",
    "GroundingDINORawDetectPlugin",
    "RAMPlusRecognizeTagsPlugin",
    "SAM2GenerateMasksPlugin",
    "SegmentDetectionsPlugin",
    "SigLIP2EncodeEmbeddingsPlugin",
    "SigLIP2EncodeTextPlugin",
]
