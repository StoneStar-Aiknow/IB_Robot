"""ROS 2 node that validates and prepares an offline semantic mapping run."""

import itertools
import json
import pathlib
import threading
import time
import uuid

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time

from ibrobot_msgs.srv import EncodeEmbeddings, GenerateMasks, RecognizeTags

from .artifact_export import SemanticArtifactExporter
from .association import SemanticObservation, SemanticTracker
from .database import MappingRunRecord, SemanticMapDatabase
from .frame_processor import MaskCandidate, filter_masks, prepare_frame
from .geometry import (
    project_masked_depth,
    quaternion_matrix,
    select_geometry_mask_indices,
    transform_geometry,
)
from .label_refinement import CloudLabelRefiner, apply_refinement, record_refinement_rejection, should_refine_label
from .offline_bag import OfflineBagSource, OfflineTopicContract, RosbagReader, create_run_manifest, uniform_sample
from .pipeline import SerializedCommitter
from .representative_view import OptionalCaptioner, RepresentativeViewStore
from .runtime_identity import MappingRunPinMismatch, SemanticIdentity
from .service_pipeline import ServiceFramePipeline, parse_label_aliases, ram_mask_candidates, select_ram_label


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
            "max_masks_per_frame": 32,
            "max_masks_per_batch": 8,
            "min_mask_pixels": 30,
            "min_mask_area_ratio": 0.0005,
            "min_mask_valid_depth_ratio": 0.2,
            "max_mask_overlap_ratio": 0.8,
            "min_frame_valid_depth_ratio": 0.05,
            "depth_trunc_m": 4.0,
            "min_points": 30,
            "ground_filter_enabled": True,
            "ground_reference_frame": "base_link",
            "ground_height_offset_m": 0.0,
            "ground_max_bottom_clearance_m": 0.15,
            "ground_max_object_height_m": 0.75,
            "ground_max_footprint_m": 1.2,
            "max_object_extent_m": 0.65,
            "max_object_distance_m": 2.5,
            "association_distance_m": 0.45,
            "association_max_size_ratio": 4.0,
            "embedding_similarity_threshold": 0.72,
            "min_label_confidence": 0.2,
            "max_label_candidates_per_mask": 5,
            "association_position_weight": 0.55,
            "label_switch_confidence_margin": 0.05,
            "label_recurrence_count_ratio": 3.0,
            "label_high_confidence_override_margin": 0.08,
            "allowed_label_aliases_json": "",
            "caption_enabled": False,
            "caption_model_identity": "",
            "caption_prompt": "Describe this object concisely.",
            "label_refinement_enabled": False,
            "label_refinement_model": "",
            "label_refinement_model_identity": "",
            "label_refinement_prompt": "Identify the single physical object in this masked crop.",
            "label_refinement_min_confidence": 0.8,
            "label_refinement_trigger_below_confidence": 0.7,
            "label_refinement_min_observations": 1,
            "artifact_output_dir": "~/.ros/ibrobot/semantic_artifacts",
            "max_frames": 0,
            "start_frame": 0,
            "frame_sampling": "sequential",
            "diagnostics_output_dir": "",
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        self.declare_parameter("excluded_labels", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("actionable_labels", Parameter.Type.STRING_ARRAY)

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
        max_frames = int(self.get_parameter("max_frames").value)
        start_frame = int(self.get_parameter("start_frame").value)
        if start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        frame_sampling = str(self.get_parameter("frame_sampling").value)
        if frame_sampling == "uniform":
            available_frames = list(self.source.frames(self.tf_buffer))
            selected_frames = uniform_sample(available_frames, max_frames)
            self._frames = iter(selected_frames)
            self.get_logger().info(
                f"Uniform frame sampling selected {len(selected_frames)} of {len(available_frames)} usable frames"
            )
        elif frame_sampling == "sequential":
            self._frames = itertools.islice(self.source.frames(self.tf_buffer), start_frame, None)
        else:
            raise ValueError("frame_sampling must be 'sequential' or 'uniform'")
        self._bridge = CvBridge()
        self._label_aliases = parse_label_aliases(str(self.get_parameter("allowed_label_aliases_json").value))
        self._actionable_labels = {
            str(label).strip().casefold() for label in (self.get_parameter("actionable_labels").value or ())
        }
        self._tracker = SemanticTracker(
            association_distance_m=float(self.get_parameter("association_distance_m").value),
            embedding_similarity_threshold=float(self.get_parameter("embedding_similarity_threshold").value),
            position_weight=float(self.get_parameter("association_position_weight").value),
            max_size_ratio=float(self.get_parameter("association_max_size_ratio").value),
            label_switch_confidence_margin=float(self.get_parameter("label_switch_confidence_margin").value),
            label_recurrence_count_ratio=float(self.get_parameter("label_recurrence_count_ratio").value),
            label_high_confidence_override_margin=float(
                self.get_parameter("label_high_confidence_override_margin").value
            ),
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
        diagnostics_path = str(self.get_parameter("diagnostics_output_dir").value).strip()
        self._diagnostics_dir = pathlib.Path(diagnostics_path).expanduser() if diagnostics_path else None
        if self._diagnostics_dir is not None:
            for name in ("rgb", "depth", "sam2", "ram_plus", "ram_plus_local", "siglip2", "fusion", "frames"):
                (self._diagnostics_dir / name).mkdir(parents=True, exist_ok=True)
        self._max_frames = max_frames
        self._attempted_frames = 0
        self._successful_frames = 0
        self._rejected_frames = 0
        self._completion_started = False
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
            max_masks_per_batch=int(self.get_parameter("max_masks_per_batch").value),
            excluded_labels=self.get_parameter("excluded_labels").value or (),
            max_mask_candidates=int(self.get_parameter("max_label_candidates_per_mask").value),
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
                if self._max_frames and self._attempted_frames >= self._max_frames:
                    self._finalize_run()
                    return
                frame = next(self._frames)
                self._attempted_frames += 1
            except StopIteration:
                self._finalize_run()
                return
            self._in_flight = True
            if self._attempted_frames == 1 or self._attempted_frames % 10 == 0:
                target = self._max_frames or "all"
                self.get_logger().info(
                    f"Offline mapping progress: frame={self._attempted_frames}/{target}, "
                    f"successful={self._successful_frames}, rejected={self._rejected_frames}, "
                    f"tracks={len(self._tracker.tracks)}"
                )
        depth = self._bridge.imgmsg_to_cv2(frame.depth, desired_encoding="passthrough")
        intrinsics = np.asarray(frame.camera_info.k, dtype=np.float64).reshape(3, 3)
        transform = frame.transform.transform
        camera_translation = np.asarray([transform.translation.x, transform.translation.y, transform.translation.z])
        camera_rotation = quaternion_matrix(
            transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w
        )
        ground_transform = self.tf_buffer.lookup_transform(
            self.manifest.global_frame,
            str(self.get_parameter("ground_reference_frame").value),
            Time(nanoseconds=frame.stamp_ns),
        )
        ground_height = float(ground_transform.transform.translation.z) + float(
            self.get_parameter("ground_height_offset_m").value
        )
        base_xy = np.asarray([ground_transform.transform.translation.x, ground_transform.transform.translation.y])

        def mask_selector(detections):
            return select_geometry_mask_indices(
                detections,
                lambda mask: self._bridge.imgmsg_to_cv2(mask, desired_encoding="mono8"),
                depth,
                intrinsics,
                1.0 if frame.depth.encoding in {"32FC1", "64FC1"} else 1000.0,
                float(self.get_parameter("depth_trunc_m").value),
                int(self.get_parameter("min_points").value),
                camera_translation,
                camera_rotation,
                ground_height,
                base_xy,
                enabled=bool(self.get_parameter("ground_filter_enabled").value),
                max_bottom_clearance_m=float(self.get_parameter("ground_max_bottom_clearance_m").value),
                max_object_height_m=float(self.get_parameter("ground_max_object_height_m").value),
                max_footprint_m=float(self.get_parameter("ground_max_footprint_m").value),
                max_object_extent_m=float(self.get_parameter("max_object_extent_m").value),
                max_horizontal_distance_m=float(self.get_parameter("max_object_distance_m").value),
            )

        result = self._pipeline.process(
            frame.rgb,
            mask_options={
                "max_masks": int(self.get_parameter("max_masks_per_frame").value),
                "min_mask_pixels": int(self.get_parameter("min_mask_pixels").value),
                "min_mask_area_ratio": float(self.get_parameter("min_mask_area_ratio").value),
                "max_overlap_ratio": float(self.get_parameter("max_mask_overlap_ratio").value),
            },
            mask_selector=mask_selector,
        )
        result.add_done_callback(lambda future, frame=frame: self._inference_done(frame, future))

    def _inference_done(self, messages, future) -> None:
        try:
            result = future.result()
            self._fuse_frame(messages, result)
        except Exception as exc:
            self._rejected_frames += 1
            self.get_logger().error(f"Offline frame rejected: {exc}")
        else:
            self._successful_frames += 1
            if self._attempted_frames % 10 == 0 or (self._max_frames and self._attempted_frames == self._max_frames):
                self.get_logger().info(
                    f"Offline mapping fused frame {self._attempted_frames}/{self._max_frames or 'all'}: "
                    f"successful={self._successful_frames}, rejected={self._rejected_frames}, "
                    f"tracks={len(self._tracker.tracks)}"
                )
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
            max_masks=int(self.get_parameter("max_masks_per_frame").value),
            min_mask_pixels=int(self.get_parameter("min_mask_pixels").value),
            min_mask_area_ratio=float(self.get_parameter("min_mask_area_ratio").value),
            min_valid_depth_ratio=float(self.get_parameter("min_mask_valid_depth_ratio").value),
            max_overlap_ratio=float(self.get_parameter("max_mask_overlap_ratio").value),
            depth_trunc_m=float(self.get_parameter("depth_trunc_m").value),
        )
        embeddings = {int(item.mask_index): item for item in result.embeddings if item.success}
        matched_ids = set()
        ground_accepted = []
        track_assignments = {}
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
            ground_accepted.append(index)
            encoded = embeddings.get(index)
            min_confidence = float(self.get_parameter("min_label_confidence").value)
            excluded_labels = self.get_parameter("excluded_labels").value or ()
            label, confidence = select_ram_label(
                index,
                result.mask_tag_counts,
                result.mask_tags,
                result.mask_tag_scores,
                min_confidence,
                excluded_labels,
                self._label_aliases,
            )
            if not label:
                continue
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
                attributes={
                    "source_frame": frame.camera_frame,
                    "offline": True,
                    "semantic_actionable": label.casefold() in self._actionable_labels,
                },
                label_candidates=ram_mask_candidates(
                    index,
                    result.mask_tag_counts,
                    result.mask_tags,
                    result.mask_tag_scores,
                    excluded_labels,
                    self._label_aliases,
                ),
            )

            def commit(observation=observation):
                track = self._tracker.update(observation, excluded_object_ids=matched_ids)
                track.attributes["semantic_actionable"] = track.label.casefold() in self._actionable_labels
                matched_ids.add(track.object_id)
                self._database.upsert(track, observation)
                return track

            track = self._committer.commit(commit)
            track_assignments[index] = track.object_id
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
        self._export_diagnostics(
            messages, result, image_bgr, depth, detections, ground_accepted, embeddings, track_assignments
        )

    def _generate_label_refinements(self) -> None:
        if not self.get_parameter("label_refinement_enabled").value:
            return
        try:
            from embodied_common.vlm_api_client import VLMClient

            refiner = CloudLabelRefiner(
                VLMClient(),
                model=str(self.get_parameter("label_refinement_model").value),
                model_identity=str(self.get_parameter("label_refinement_model_identity").value),
                prompt=str(self.get_parameter("label_refinement_prompt").value),
                min_confidence=float(self.get_parameter("label_refinement_min_confidence").value),
                excluded_labels=self.get_parameter("excluded_labels").value or (),
            )
        except Exception as exc:
            self.get_logger().error(f"Cloud label refinement unavailable; RAM++ labels remain active: {exc}")
            return
        excluded = self.get_parameter("excluded_labels").value or ()
        min_observations = int(self.get_parameter("label_refinement_min_observations").value)
        for object_id in sorted(self._tracker.tracks):
            track = self._tracker.tracks[object_id]
            view = self._representative_views.get(object_id)
            if (
                view is None
                or track.observation_count < min_observations
                or not should_refine_label(
                    track.label,
                    track.confidence,
                    excluded,
                    float(self.get_parameter("label_refinement_trigger_below_confidence").value),
                    inconsistent=len(track.attributes.get("label_evidence", {})) > 1,
                )
            ):
                continue
            try:
                candidates = self._tracker.aggregated_label_candidates(track, excluded_labels=excluded)
                result = refiner.refine(view, candidates)
                apply_refinement(track, result)
                self._database.upsert(track)
            except Exception as exc:
                candidates = self._tracker.aggregated_label_candidates(track, excluded_labels=excluded)
                record_refinement_rejection(
                    track,
                    candidates=candidates,
                    model_identity=refiner.model_identity,
                    error=exc,
                )
                self._database.upsert(track)
                self.get_logger().warn(
                    f"Cloud label refinement rejected for {object_id}; "
                    f"RAM++={track.label!r}/{track.confidence:.3f}, candidates={list(candidates)}: {exc}"
                )

    def _finalize_run(self) -> None:
        if self._completion_started:
            return
        self._completion_started = True
        self._finished = True
        self.get_logger().info(
            f"Offline mapping inference complete: attempted={self._attempted_frames}, "
            f"successful={self._successful_frames}, rejected={self._rejected_frames}, "
            f"tracks={len(self._tracker.tracks)}"
        )
        self.get_logger().info("Offline mapping post-processing: refining labels")
        self._generate_label_refinements()
        self._generate_captions()
        self.get_logger().info("Offline mapping post-processing: exporting geometry")
        self._export_geometry()
        self.get_logger().info(
            f"Offline mapping complete: frames={self._attempted_frames}, "
            f"successful={self._successful_frames}, rejected={self._rejected_frames}, "
            f"objects={len(self._tracker.tracks)}"
        )
        self._timer.cancel()
        rclpy.shutdown()

    def _export_diagnostics(
        self, messages, result, image_bgr, depth, detections, accepted, embeddings, track_assignments
    ):
        if self._diagnostics_dir is None:
            return
        frame_id = f"{self._attempted_frames:04d}_{messages.stamp_ns}"
        root = self._diagnostics_dir
        cv2.imwrite(str(root / "rgb" / f"{frame_id}.jpg"), image_bgr)
        depth_m = depth.astype(np.float32)
        if messages.depth.encoding not in {"32FC1", "64FC1"}:
            depth_m /= 1000.0
        depth_visual = np.clip(depth_m / 4.0 * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(root / "depth" / f"{frame_id}.jpg"), cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET))
        sam = image_bgr.copy()
        local = image_bgr.copy()
        for index, detection in enumerate(detections):
            mask = self._bridge.imgmsg_to_cv2(detection.mask, desired_encoding="mono8") > 0
            color = (37 * (index + 3) % 255, 97 * (index + 5) % 255, 173 * (index + 7) % 255)
            sam[mask] = (0.55 * sam[mask] + 0.45 * np.asarray(color)).astype(np.uint8)
            local[mask] = (0.55 * local[mask] + 0.45 * np.asarray(color)).astype(np.uint8)
            x1, y1, x2, y2 = [int(value) for value in detection.bbox]
            cv2.rectangle(sam, (x1, y1), (x2, y2), color, 1)
            cv2.rectangle(local, (x1, y1), (x2, y2), color, 2 if index in accepted else 1)
            candidates = ram_mask_candidates(
                index,
                result.mask_tag_counts,
                result.mask_tags,
                result.mask_tag_scores,
                self.get_parameter("excluded_labels").value or (),
                self._label_aliases,
            )
            label = candidates[0][0] if candidates else "unassigned"
            score = candidates[0][1] if candidates else 0.0
            cv2.putText(
                sam,
                f"M{index}:{detection.confidence:.2f}",
                (x1, max(15, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
            cv2.putText(
                local,
                f"M{index} {label}:{score:.2f}",
                (x1 + 2, max(15, y1 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
            )
        cv2.imwrite(str(root / "sam2" / f"{frame_id}.jpg"), sam)
        cv2.imwrite(str(root / "ram_plus_local" / f"{frame_id}.jpg"), local)
        ram = np.zeros_like(image_bgr)
        labels = sorted(zip(result.tags, result.tag_scores, strict=True), key=lambda item: item[1], reverse=True)
        for row, (label, score) in enumerate(labels[:24]):
            cv2.putText(
                ram,
                f"{row + 1:02d} {label} {score:.3f}",
                (12, 24 + row * 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (240, 240, 240),
                1,
            )
        cv2.imwrite(str(root / "ram_plus" / f"{frame_id}.jpg"), ram)
        siglip = image_bgr.copy()
        for index, item in embeddings.items():
            if index >= len(detections):
                continue
            mask = self._bridge.imgmsg_to_cv2(detections[index].mask, desired_encoding="mono8") > 0
            siglip[mask] = (0.55 * siglip[mask] + np.asarray((0, 220, 80)) * 0.45).astype(np.uint8)
            x1, y1, _x2, _y2 = [int(value) for value in detections[index].bbox]
            cv2.putText(
                siglip,
                f"{item.matched_label}:{item.matched_score:.3f}",
                (x1, max(15, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 80),
                1,
            )
        cv2.imwrite(str(root / "siglip2" / f"{frame_id}.jpg"), siglip)
        (root / "frames" / f"{frame_id}.json").write_text(
            json.dumps(
                {
                    "stamp_ns": messages.stamp_ns,
                    "sam_detections": len(detections),
                    "accepted_masks": list(accepted),
                    "ram_tags": list(result.tags),
                    "ram_scores": list(result.tag_scores),
                    "ram_local_counts": list(result.mask_tag_counts),
                    "ram_local_tags": list(result.mask_tags),
                    "ram_local_scores": list(result.mask_tag_scores),
                    "masks": {
                        str(index): {
                            "bbox": [float(value) for value in detection.bbox],
                            "area": int(
                                np.count_nonzero(self._bridge.imgmsg_to_cv2(detection.mask, desired_encoding="mono8"))
                            ),
                            "siglip_label": embeddings[index].matched_label if index in embeddings else "",
                            "siglip_score": float(embeddings[index].matched_score) if index in embeddings else 0.0,
                            "object_id": track_assignments.get(index, ""),
                            "ram_candidates": [
                                {"label": label, "score": score}
                                for label, score in ram_mask_candidates(
                                    index,
                                    result.mask_tag_counts,
                                    result.mask_tags,
                                    result.mask_tag_scores,
                                    self.get_parameter("excluded_labels").value or (),
                                    self._label_aliases,
                                )
                            ],
                        }
                        for index, detection in enumerate(detections)
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
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
        self._timer.cancel()
        if rclpy.ok():
            rclpy.shutdown()

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
