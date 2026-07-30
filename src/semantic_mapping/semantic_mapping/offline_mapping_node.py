"""ROS 2 node that validates and prepares an offline semantic mapping run."""

import json
import threading
import time
import uuid

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from ibrobot_msgs.srv import EncodeEmbeddings, GenerateMasks, RecognizeTags

from .artifact_export import SemanticArtifactExporter
from .association import SemanticObservation, SemanticTracker
from .database import MappingRunRecord, SemanticMapDatabase
from .frame_processor import MaskCandidate, filter_masks, prepare_frame
from .geometry import project_masked_depth, quaternion_matrix, transform_geometry
from .offline_bag import OfflineBagSource, OfflineTopicContract, RosbagReader, create_run_manifest
from .pipeline import SerializedCommitter
from .representative_view import OptionalCaptioner, RepresentativeViewStore
from .runtime_identity import MappingRunPinMismatch, SemanticIdentity
from .service_pipeline import ServiceFramePipeline


class OfflineMappingNode(Node):
    def __init__(self):
        super().__init__("offline_mapping")
        defaults = {
            "bag_path": "",
            "storage_id": "",
            "rgb_topic": "/camera/front/image_raw",
            "depth_topic": "/camera/front/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/front/camera_info",
            "global_frame": "map",
            "sync_slop_sec": 0.06,
            "geometry_map_id": "",
            "geometry_map_hash": "",
            "localization_session_id": "",
            "calibration_id": "",
            "urdf_hash": "",
            "coordinate_convention": "ros-rep-103-map-enu",
            "sam_model_identity": "",
            "ram_plus_model_identity": "",
            "siglip2_model_identity": "",
            "gdino_model_identity": "",
            "database_path": "~/.ros/ibrobot/offline_semantic_map.sqlite3",
            "sam_service": "/sam2_service/generate_masks",
            "ram_plus_service": "/ram_plus_service/recognize_tags",
            "siglip2_service": "/siglip2_service/encode_embeddings",
            "service_wait_sec": 1.0,
            "configuration_generation": 0,
            "sam_service_instance_id": "sam2",
            "ram_plus_service_instance_id": "ram_plus",
            "siglip2_service_instance_id": "siglip2_image",
            "max_masks": 8,
            "min_mask_pixels": 30,
            "min_mask_area_ratio": 0.0005,
            "min_mask_valid_depth_ratio": 0.2,
            "max_mask_overlap_ratio": 0.8,
            "min_frame_valid_depth_ratio": 0.05,
            "depth_trunc_m": 4.0,
            "min_points": 30,
            "association_distance_m": 0.45,
            "embedding_similarity_threshold": 0.72,
            "association_position_weight": 0.55,
            "caption_enabled": False,
            "caption_model_identity": "",
            "caption_prompt": "Describe this object concisely.",
            "artifact_output_dir": "~/.ros/ibrobot/semantic_artifacts",
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

        bag_path = self.get_parameter("bag_path").value
        if not bag_path:
            raise ValueError("bag_path is required")
        topics = OfflineTopicContract(
            rgb=self.get_parameter("rgb_topic").value,
            aligned_depth=self.get_parameter("depth_topic").value,
            camera_info=self.get_parameter("camera_info_topic").value,
        )
        semantic_identities = {
            "sam2": SemanticIdentity.from_json(self.get_parameter("sam_model_identity").value),
            "ram_plus": SemanticIdentity.from_json(self.get_parameter("ram_plus_model_identity").value),
            "siglip2_image": SemanticIdentity.from_json(self.get_parameter("siglip2_model_identity").value),
        }
        self.manifest = create_run_manifest(
            global_frame=self.get_parameter("global_frame").value,
            geometry_map_id=self.get_parameter("geometry_map_id").value,
            geometry_map_hash=self.get_parameter("geometry_map_hash").value,
            localization_session_id=self.get_parameter("localization_session_id").value,
            calibration_id=self.get_parameter("calibration_id").value,
            urdf_hash=self.get_parameter("urdf_hash").value,
            coordinate_convention=self.get_parameter("coordinate_convention").value,
            semantic_identities=semantic_identities,
            bag_path=bag_path,
            topics=topics,
        )
        self.source = OfflineBagSource(
            RosbagReader(bag_path, self.get_parameter("storage_id").value),
            topics,
            global_frame=self.manifest.global_frame,
            sync_slop_ns=int(float(self.get_parameter("sync_slop_sec").value) * 1e9),
        )
        self.source.validate_topics()
        self.tf_buffer = self.source.build_tf_buffer()
        self._frames = iter(self.source.frames(self.tf_buffer))
        self._bridge = CvBridge()
        self._tracker = SemanticTracker(
            association_distance_m=float(self.get_parameter("association_distance_m").value),
            embedding_similarity_threshold=float(self.get_parameter("embedding_similarity_threshold").value),
            position_weight=float(self.get_parameter("association_position_weight").value),
        )
        self._database = SemanticMapDatabase(self.get_parameter("database_path").value, self.manifest)
        now_ns = time.time_ns()
        self._run = MappingRunRecord(
            run_id=str(uuid.uuid4()),
            configuration_generation=int(self.get_parameter("configuration_generation").value),
            expected_service_instance_ids={
                "sam2": self.get_parameter("sam_service_instance_id").value,
                "ram_plus": self.get_parameter("ram_plus_service_instance_id").value,
                "siglip2_image": self.get_parameter("siglip2_service_instance_id").value,
            },
            required_semantic_identities=self.manifest.canonical_semantic_identities,
            status="active",
            started_ns=now_ns,
            updated_ns=now_ns,
        )
        self._run_pin = self._database.create_mapping_run(self._run)
        self._committer = SerializedCommitter()
        self._representative_views = RepresentativeViewStore()
        self._geometry_points: dict[str, list[np.ndarray]] = {}
        self._exporter = SemanticArtifactExporter(self.get_parameter("artifact_output_dir").value, self._database)
        self._exporter.export_manifest(self.manifest)
        self._state_lock = threading.Lock()
        self._in_flight = False
        self._finished = False
        self._run_admission_open = True
        service_group = ReentrantCallbackGroup()
        self._sam_client = self.create_client(
            GenerateMasks, self.get_parameter("sam_service").value, callback_group=service_group
        )
        self._ram_client = self.create_client(
            RecognizeTags, self.get_parameter("ram_plus_service").value, callback_group=service_group
        )
        self._siglip_client = self.create_client(
            EncodeEmbeddings, self.get_parameter("siglip2_service").value, callback_group=service_group
        )
        self._pipeline = ServiceFramePipeline(
            self._sam_client,
            self._ram_client,
            self._siglip_client,
            max_masks_per_batch=int(self.get_parameter("max_masks").value),
        )
        self._timer = self.create_timer(0.05, self._schedule_next, callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(f"Offline mapping input ready: {json.dumps(self.manifest.settings, sort_keys=True)}")

    def _services_ready(self) -> bool:
        return all(client.service_is_ready() for client in (self._sam_client, self._ram_client, self._siglip_client))

    def _schedule_next(self) -> None:
        with self._state_lock:
            if self._in_flight or self._finished:
                return
            if not self._services_ready():
                return
            try:
                frame = next(self._frames)
            except StopIteration:
                self._finished = True
                self._generate_captions()
                self._export_geometry()
                self.get_logger().info(
                    f"Offline mapping complete: objects={len(self._tracker.tracks)}, "
                    f"frames={self.source.diagnostics.synchronized_frames}"
                )
                return
            self._in_flight = True
        result = self._pipeline.process(
            frame.rgb,
            mask_options={
                "max_masks": int(self.get_parameter("max_masks").value),
                "min_mask_pixels": int(self.get_parameter("min_mask_pixels").value),
                "min_mask_area_ratio": float(self.get_parameter("min_mask_area_ratio").value),
                "max_overlap_ratio": float(self.get_parameter("max_mask_overlap_ratio").value),
            },
        )
        result.add_done_callback(lambda future, frame=frame: self._inference_done(frame, future))

    def _inference_done(self, messages, future) -> None:
        try:
            result = future.result()
            self._fuse_frame(messages, result)
        except Exception as exc:
            self.get_logger().error(f"Offline frame rejected: {exc}")
        finally:
            with self._state_lock:
                self._in_flight = False

    def _fuse_frame(self, messages, result) -> None:
        try:
            provenances = self._run_pin.validate_frame(result.model_diagnostics)
        except MappingRunPinMismatch as exc:
            self._stop_pinned_run(str(exc))
            raise
        deployment_provenance = {role: value.to_dict() for role, value in provenances.items()}
        image_bgr = self._bridge.imgmsg_to_cv2(messages.rgb, desired_encoding="bgr8")
        depth = self._bridge.imgmsg_to_cv2(messages.depth, desired_encoding="passthrough")
        intrinsics = np.asarray(messages.camera_info.k, dtype=np.float64).reshape(3, 3)
        transform = messages.transform.transform
        translation = np.asarray([transform.translation.x, transform.translation.y, transform.translation.z])
        rotation = quaternion_matrix(
            transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w
        )
        depth_scale = 1.0 if messages.depth.encoding in {"32FC1", "64FC1"} else 1000.0
        frame = prepare_frame(
            image_bgr=image_bgr,
            depth=depth,
            intrinsics=intrinsics,
            depth_scale=depth_scale,
            rgb_stamp_ns=messages.stamp_ns,
            depth_stamp_ns=int(messages.depth.header.stamp.sec) * 1_000_000_000
            + int(messages.depth.header.stamp.nanosec),
            info_stamp_ns=int(messages.camera_info.header.stamp.sec) * 1_000_000_000
            + int(messages.camera_info.header.stamp.nanosec),
            camera_frame=messages.rgb.header.frame_id or messages.camera_info.header.frame_id,
            translation=translation,
            rotation=rotation,
            max_stamp_skew_ns=int(float(self.get_parameter("sync_slop_sec").value) * 1e9),
            depth_trunc_m=float(self.get_parameter("depth_trunc_m").value),
            min_valid_depth_ratio=float(self.get_parameter("min_frame_valid_depth_ratio").value),
        )
        detections = list(result.masks.detections)
        accepted, _ = filter_masks(
            frame,
            [
                MaskCandidate(self._bridge.imgmsg_to_cv2(item.mask, desired_encoding="mono8"), item.confidence)
                for item in detections
            ],
            max_masks=int(self.get_parameter("max_masks").value),
            min_mask_pixels=int(self.get_parameter("min_mask_pixels").value),
            min_mask_area_ratio=float(self.get_parameter("min_mask_area_ratio").value),
            min_valid_depth_ratio=float(self.get_parameter("min_mask_valid_depth_ratio").value),
            max_overlap_ratio=float(self.get_parameter("max_mask_overlap_ratio").value),
            depth_trunc_m=float(self.get_parameter("depth_trunc_m").value),
        )
        embeddings = {int(item.mask_index): item for item in result.embeddings if item.success}
        matched_ids = set()
        for index in accepted:
            detection = detections[index]
            mask = self._bridge.imgmsg_to_cv2(detection.mask, desired_encoding="mono8")
            geometry = project_masked_depth(
                mask,
                frame.depth,
                frame.intrinsics,
                frame.depth_scale,
                float(self.get_parameter("depth_trunc_m").value),
                int(self.get_parameter("min_points").value),
            )
            if geometry is None:
                continue
            world = transform_geometry(geometry, frame.translation, frame.rotation)
            encoded = embeddings.get(index)
            label = encoded.matched_label if encoded is not None and encoded.matched_label else "unlabeled"
            confidence = float(encoded.matched_score) if encoded is not None else float(detection.confidence)
            observation = SemanticObservation(
                label=label,
                canonical_label=label.casefold(),
                confidence=confidence,
                position=world.centroid,
                size=world.size,
                point_count=world.points.shape[0],
                stamp_ns=messages.stamp_ns,
                embedding=None if encoded is None else np.asarray(encoded.embedding, dtype=np.float32),
                map_version=self.manifest.geometry_map_hash,
                session_id=self.manifest.localization_session_id,
                source_frame=frame.camera_frame,
                semantic_identities={
                    role: identity.to_dict() for role, identity in self.manifest.canonical_semantic_identities.items()
                },
                deployment_provenance=deployment_provenance,
                mapping_run_id=self._run.run_id,
                attributes={"source_frame": frame.camera_frame, "offline": True},
            )

            def commit(observation=observation):
                track = self._tracker.update(observation, excluded_object_ids=matched_ids)
                matched_ids.add(track.object_id)
                self._database.upsert(track, observation)
                return track

            track = self._committer.commit(commit)
            self._geometry_points.setdefault(track.object_id, []).append(world.points)
            self._representative_views.consider(
                self._representative_views.create(
                    track.object_id,
                    messages.stamp_ns,
                    confidence,
                    image_bgr,
                    mask,
                    np.asarray(detection.bbox, dtype=np.float32),
                )
            )

    def _generate_captions(self) -> None:
        if not self.get_parameter("caption_enabled").value:
            return
        try:
            from embodied_common.vlm_api_client import VLMClient

            captioner = OptionalCaptioner(VLMClient(), self.get_parameter("caption_model_identity").value)
        except Exception as exc:
            self.get_logger().error(f"Caption client unavailable; objects remain queryable: {exc}")
            return
        prompt = self.get_parameter("caption_prompt").value
        for object_id in sorted(self._tracker.tracks):
            view = self._representative_views.get(object_id)
            if view is not None:
                record = captioner.caption(object_id, view, prompt)
                self._database.upsert_caption(record)
                if not record.success:
                    self.get_logger().warn(f"Caption unavailable for {object_id}: {record.message}")

    def _export_geometry(self) -> None:
        for object_id in sorted(self._geometry_points):
            track = self._tracker.tracks[object_id]
            points = np.concatenate(self._geometry_points[object_id])
            self._exporter.export_geometry(object_id, track.object_version, points, track.last_seen_ns)

    def _stop_pinned_run(self, reason: str) -> None:
        with self._state_lock:
            self._finished = True
            self._run_admission_open = False
        now_ns = time.time_ns()
        self._database.update_mapping_run_status(
            self._run.run_id,
            "failed",
            now_ns,
            ended_ns=now_ns,
            reason=reason,
        )
        self.get_logger().error(f"Offline mapping run stopped: {reason}")

    def destroy_node(self):
        now_ns = time.time_ns()
        if self._run_admission_open:
            status = "completed" if self._finished else "paused"
            self._database.update_mapping_run_status(
                self._run.run_id,
                status,
                now_ns,
                ended_ns=now_ns if status == "completed" else None,
            )
        self._database.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = OfflineMappingNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
