"""Explicit-image ROS 2 service nodes for semantic perception models."""

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Header

from ibrobot_msgs.msg import Detection2D, DetectionArray, MaskEmbedding, ModelRuntimeInfo, TextEmbedding
from ibrobot_msgs.srv import EncodeEmbeddings, EncodeText, GenerateMasks, GroundingDetect, RecognizeTags

from .grounded_sam2_wrapper import GroundedSAM2Wrapper
from .model_contracts import (
    ModelManifest,
    sha256_file,
    validate_image_message,
    validate_mask_batch,
    validate_text_batch,
)
from .ram_plus_adapter import RAM_PLUS_PREPROCESSING
from .ram_plus_wrapper import RAMPlusWrapper
from .sam2_wrapper import SAM2Wrapper, SegmentationMask
from .semantic_model_adapters import GroundingDINOAdapter, SAM2Adapter, SigLIP2ImageAdapter
from .siglip2_wrapper import SigLIP2Wrapper


def _hash_path(path: str | Path) -> str:
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        identity_files = [path / name for name in ("config.json", "preprocessor_config.json")]
        values = [sha256_file(item) for item in identity_files if item.is_file()]
        return ":".join(values)
    return ""


def _runtime_info(manifest: ModelManifest | None, ready: bool, message: str) -> ModelRuntimeInfo:
    result = ModelRuntimeInfo()
    if manifest is not None:
        for key, value in asdict(manifest).items():
            setattr(result, key, value)
    result.ready = ready
    result.message = message
    return result


def _manifest(
    *,
    name: str,
    version: str,
    path: str | Path,
    backend: str,
    runtime_version: str,
    preprocessing: str,
    embedding_dim: int = 0,
    normalization: str = "",
) -> ModelManifest:
    return ModelManifest(
        model_name=name,
        model_version=version,
        weights_hash=_hash_path(path),
        config_hash="",
        backend=backend,
        runtime_version=runtime_version,
        preprocessing_hash=preprocessing,
        embedding_dim=embedding_dim,
        normalization=normalization,
    )


class _ModelServiceNode(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)
        self._bridge = CvBridge()
        self._wrapper = None
        self._manifest = None
        self._load_error = "model has not been loaded"
        self.declare_parameter("backend", "cuda")

    @property
    def _backend(self) -> str:
        return self.get_parameter("backend").get_parameter_value().string_value

    def _load(self, factory) -> None:
        try:
            self._wrapper = factory()
            self._load_error = ""
        except Exception as exc:  # Keep the service alive to report not-ready state.
            self._load_error = str(exc)
            self.get_logger().error(f"Model initialization failed: {exc}")

    def _image_rgb(self, image):
        validate_image_message(image)
        return np.asarray(self._bridge.imgmsg_to_cv2(image, desired_encoding="rgb8"), dtype=np.uint8)

    def _model_info(self) -> ModelRuntimeInfo:
        return _runtime_info(self._manifest, self._wrapper is not None, self._load_error)

    def destroy_node(self):
        close = getattr(self._wrapper, "close", None)
        if callable(close):
            close()
        return super().destroy_node()


def _detection_array(bridge: CvBridge, header: Header, records: list[SegmentationMask]) -> DetectionArray:
    detections = []
    for record in records:
        detection = Detection2D()
        detection.header = header
        detection.confidence = record.score
        detection.bbox = record.bbox_xyxy.astype(float).tolist()
        detection.mask = bridge.cv2_to_imgmsg((record.mask > 0).astype(np.uint8) * 255, encoding="mono8")
        detection.mask.header = header
        detections.append(detection)
    return DetectionArray(header=header, detections=detections)


def _filter_masks(
    records: list[SegmentationMask],
    image_area: int,
    max_masks: int,
    min_pixels: int,
    min_area_ratio: float,
    max_overlap_ratio: float,
) -> list[SegmentationMask]:
    candidates = [
        (index, record)
        for index, record in enumerate(records)
        if record.area >= min_pixels and record.area / image_area >= min_area_ratio
    ]
    candidates.sort(key=lambda item: (-item[1].score, -item[1].area, item[0]))
    accepted = []
    for _, record in candidates:
        mask = record.mask > 0
        if any(
            np.count_nonzero(mask & (other.mask > 0)) / max(1, record.area) > max_overlap_ratio for other in accepted
        ):
            continue
        accepted.append(record)
        if max_masks and len(accepted) >= max_masks:
            break
    return accepted


class SAM2ServiceNode(_ModelServiceNode):
    def __init__(self):
        super().__init__("sam2_service")
        self.declare_parameter("checkpoint", "sam2.1_hiera_tiny/assets/sam2.1_hiera_tiny.pt")
        self.declare_parameter("config", "configs/sam2.1/sam2.1_hiera_t.yaml")
        self.declare_parameter("model_dir", "")
        self.declare_parameter("points_per_batch", 64)
        checkpoint = self.get_parameter("checkpoint").value
        config = self.get_parameter("config").value
        model_dir = self.get_parameter("model_dir").value or None
        points_per_batch = self.get_parameter("points_per_batch").value
        self._load(
            lambda: SAM2Wrapper(
                backend=self._backend,
                checkpoint=checkpoint,
                config=config,
                model_dir=model_dir,
                points_per_batch=points_per_batch,
            )
        )
        if self._wrapper is not None:
            self._manifest = _manifest(
                name="sam2",
                version=Path(checkpoint).stem,
                path=self._wrapper.checkpoint_path,
                backend=self._backend,
                runtime_version=self._wrapper.runtime_version,
                preprocessing=SAM2Adapter.identity.preprocessing,
            )
        self.create_service(
            GenerateMasks,
            "~/generate_masks",
            self._callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def _callback(self, request, response):
        started = time.perf_counter()
        response.model = self._model_info()
        if self._wrapper is None:
            response.message = self._load_error
            return response
        try:
            image = self._image_rgb(request.image)
            records = self._wrapper.generate(image)
            records = _filter_masks(
                records,
                image.shape[0] * image.shape[1],
                int(request.max_masks),
                int(request.min_mask_pixels),
                max(0.0, float(request.min_mask_area_ratio)),
                max(0.0, float(request.max_overlap_ratio)),
            )
            response.detections = _detection_array(self._bridge, request.image.header, records)
            response.success = True
            response.message = f"generated {len(records)} masks"
        except Exception as exc:
            response.message = str(exc)
        response.inference_time_ms = (time.perf_counter() - started) * 1000.0
        return response


class RAMPlusServiceNode(_ModelServiceNode):
    def __init__(self):
        super().__init__("ram_plus_service")
        self.declare_parameter("checkpoint", "ram_plus_swin_large_14m/assets/ram_plus_swin_large_14m.pth")
        self.declare_parameter("model_dir", "")
        self.declare_parameter("text_encoder", "bert-base-uncased")
        checkpoint = self.get_parameter("checkpoint").value
        model_dir = self.get_parameter("model_dir").value or None
        text_encoder = self.get_parameter("text_encoder").value
        self._load(
            lambda: RAMPlusWrapper(
                backend=self._backend,
                checkpoint=checkpoint,
                model_dir=model_dir,
                text_encoder=text_encoder,
            )
        )
        if self._wrapper is not None:
            self._manifest = _manifest(
                name="ram_plus",
                version="swin_large_14m",
                path=self._wrapper.checkpoint_path,
                backend=self._backend,
                runtime_version=self._wrapper.runtime_version,
                preprocessing=RAM_PLUS_PREPROCESSING,
            )
        self.create_service(
            RecognizeTags,
            "~/recognize_tags",
            self._callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def _callback(self, request, response):
        started = time.perf_counter()
        response.model = self._model_info()
        if self._wrapper is None:
            response.message = self._load_error
            return response
        try:
            results = self._wrapper.recognize(self._image_rgb(request.image), float(request.score_threshold))
            response.tags = [result.label for result in results]
            response.scores = [result.score for result in results]
            response.success = True
            response.message = f"recognized {len(results)} tags"
        except Exception as exc:
            response.message = str(exc)
        response.inference_time_ms = (time.perf_counter() - started) * 1000.0
        return response


class SigLIP2ServiceNode(_ModelServiceNode):
    def __init__(self):
        super().__init__("siglip2_service")
        self.declare_parameter("model_path", "siglip2_so400m_patch14_384/assets/model")
        self.declare_parameter("model_dir", "")
        model_path = self.get_parameter("model_path").value
        model_dir = self.get_parameter("model_dir").value or None
        self._load(lambda: SigLIP2Wrapper(backend=self._backend, model_path=model_path, model_dir=model_dir))
        if self._wrapper is not None:
            self._manifest = _manifest(
                name="siglip2",
                version=Path(model_path).name,
                path=self._wrapper.model_path,
                backend=self._backend,
                runtime_version=self._wrapper.runtime_version,
                preprocessing=SigLIP2ImageAdapter.identity.preprocessing,
                embedding_dim=self._wrapper.embedding_dim,
                normalization="l2",
            )
        inference_group = MutuallyExclusiveCallbackGroup()
        self.create_service(
            EncodeEmbeddings,
            "~/encode_embeddings",
            self._callback,
            callback_group=inference_group,
        )
        self.create_service(
            EncodeText,
            "~/encode_text",
            self._text_callback,
            callback_group=inference_group,
        )

    def _callback(self, request, response):
        started = time.perf_counter()
        response.model = self._model_info()
        if self._wrapper is None:
            response.message = self._load_error
            return response
        try:
            validate_mask_batch(request.image, request.masks)
            image = self._image_rgb(request.image)
            masks = [self._bridge.imgmsg_to_cv2(mask, desired_encoding="mono8") for mask in request.masks]
            results = self._wrapper.encode(image, masks, list(request.candidate_labels))
            response.results = [
                MaskEmbedding(
                    mask_index=result.mask_index,
                    embedding=result.embedding.astype(float).tolist(),
                    embedding_dim=len(result.embedding),
                    matched_label=result.matched_label,
                    matched_score=result.matched_score,
                    success=True,
                )
                for result in results
            ]
            response.success = True
            response.message = f"encoded {len(results)} masks"
        except Exception as exc:
            response.message = str(exc)
        response.inference_time_ms = (time.perf_counter() - started) * 1000.0
        return response

    def _text_callback(self, request, response):
        started = time.perf_counter()
        response.model = self._model_info()
        if self._wrapper is None:
            response.message = self._load_error
            return response
        try:
            validate_text_batch(request.texts)
            features = self._wrapper.encode_text(list(request.texts))
            response.results = [
                TextEmbedding(
                    text_index=index,
                    embedding=embedding.astype(float).tolist(),
                    embedding_dim=len(embedding),
                    success=True,
                )
                for index, embedding in enumerate(features)
            ]
            response.success = True
            response.message = f"encoded {len(features)} texts"
        except Exception as exc:
            response.message = str(exc)
        response.inference_time_ms = (time.perf_counter() - started) * 1000.0
        return response


class GroundingConfirmationServiceNode(_ModelServiceNode):
    def __init__(self):
        super().__init__("grounding_confirmation_service")
        for name, default in (
            ("sam_checkpoint", "grounded_sam2_swint_ogc/assets/sam2.1_hiera_tiny.pt"),
            ("sam_config", "configs/sam2.1/sam2.1_hiera_t.yaml"),
            ("gdino_checkpoint", "grounded_sam2_swint_ogc/assets/groundingdino_swint_ogc.pth"),
            ("gdino_text_encoder", "grounded_sam2_swint_ogc/assets/bert-base-uncased"),
            ("model_dir", ""),
        ):
            self.declare_parameter(name, default)

        def load_wrapper():
            return GroundedSAM2Wrapper(
                device=self._backend,
                sam_checkpoint=self.get_parameter("sam_checkpoint").value,
                sam_config=self.get_parameter("sam_config").value,
                gdino_checkpoint=self.get_parameter("gdino_checkpoint").value,
                gdino_text_encoder=self.get_parameter("gdino_text_encoder").value,
                model_dir=self.get_parameter("model_dir").value or None,
            )

        self._load(load_wrapper)
        if self._wrapper is not None:
            self._manifest = _manifest(
                name="grounding_dino_sam2",
                version=Path(self.get_parameter("gdino_checkpoint").value).stem,
                path=Path(self.get_parameter("model_dir").value or ".") / self.get_parameter("gdino_checkpoint").value,
                backend=self._wrapper.device,
                runtime_version="torch",
                preprocessing=GroundingDINOAdapter.identity.preprocessing,
            )
        self.create_service(
            GroundingDetect,
            "~/grounding_detect",
            self._callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def _callback(self, request, response):
        started = time.perf_counter()
        response.model = self._model_info()
        if self._wrapper is None:
            response.message = self._load_error
            return response
        try:
            image_rgb = self._image_rgb(request.image)
            records = self._wrapper.detect_and_segment(
                image_rgb[:, :, ::-1],
                request.text_prompt,
                float(request.box_threshold) or 0.35,
                float(request.text_threshold) or 0.25,
            )
            masks = [
                SegmentationMask(
                    mask=record.mask,
                    bbox_xyxy=record.bbox_xyxy,
                    score=record.confidence,
                    stability_score=0.0,
                    area=int(np.count_nonzero(record.mask)),
                )
                for record in records
            ]
            response.detections = _detection_array(self._bridge, request.image.header, masks)
            for message, record in zip(response.detections.detections, records, strict=True):
                message.label = record.label
            response.success = True
            response.message = f"confirmed {len(records)} detections"
        except Exception as exc:
            response.message = str(exc)
        response.inference_time_ms = (time.perf_counter() - started) * 1000.0
        return response


def _main(node_type, args=None):
    rclpy.init(args=args)
    node = node_type()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main_sam2(args=None):
    _main(SAM2ServiceNode, args)


def main_ram_plus(args=None):
    _main(RAMPlusServiceNode, args)


def main_siglip2(args=None):
    _main(SigLIP2ServiceNode, args)


def main_grounding_confirmation(args=None):
    _main(GroundingConfirmationServiceNode, args)
