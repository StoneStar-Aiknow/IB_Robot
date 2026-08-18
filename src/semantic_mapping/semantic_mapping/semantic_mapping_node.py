"""ROS 2 node for persistent open-vocabulary RGB-D semantic mapping."""

import json
import queue
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion, Vector3
from nav_msgs.msg import OccupancyGrid
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Bool, Header, String
from tf2_ros import Buffer, TransformException, TransformListener

from ibrobot_msgs.msg import SemanticMapMetadata, SemanticObject3D, SemanticObject3DArray, TrackState
from ibrobot_msgs.srv import (
    EncodeEmbeddings,
    EncodeText,
    GenerateMasks,
    GetSemanticObjects,
    GroundingDetect,
    RecognizeTags,
    ResolveSemanticTarget,
)

from .association import (
    LifecycleEvidence,
    SemanticObservation,
    SemanticTrack,
    SemanticTracker,
    has_manual_label,
    is_manually_actionable,
)
from .database import MappingRunRecord, SemanticMapDatabase, SemanticMapManifest
from .frame_processor import MaskCandidate, filter_masks, prepare_frame
from .geometry import (
    is_ground_object,
    project_masked_depth,
    quaternion_matrix,
    select_geometry_mask_indices,
    transform_geometry,
)
from .hf_grounded_sam2 import HFGroundedSAM2
from .label_refinement import CloudLabelRefiner, apply_refinement, record_refinement_rejection, should_refine_label
from .online_lifecycle import OnlineLifecycleCoordinator
from .pipeline import BoundedFrameQueue, SerializedCommitter
from .pointcloud import xyz_to_pointcloud2
from .query import ObjectQuery, query_tracks
from .representative_view import RepresentativeViewStore
from .runtime_identity import (
    MappingRunPinMismatch,
    RuntimeDiagnostic,
    SemanticIdentity,
    require_embedding_compatibility,
)
from .service_pipeline import (
    ServiceFramePipeline,
    canonicalize_label,
    parse_label_aliases,
    ram_mask_candidates,
    select_ram_label,
)
from .siglip_encoder import SigLIPEncoder
from .slam_readiness import evaluate_slam_readiness
from .target_resolution import resolve_target
from .workflow_readiness import manipulation_confirmation, navigation_staging, structured_query, text_query
from .write_policy import SemanticWritePolicy


def _depth_scale(encoding: str) -> float:
    return 1.0 if encoding in {"32FC1", "64FC1"} else 1000.0


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _time_message(stamp_ns: int):
    return Time(nanoseconds=stamp_ns).to_msg()


def validate_mapping_backend(backend: str) -> str:
    if backend not in {"embedded", "service"}:
        raise ValueError("mapping_backend must be 'embedded' or 'service'")
    return backend


class SemanticMappingNode(Node):
    def __init__(self):
        super().__init__("semantic_mapping")
        self._bridge = CvBridge()
        self._processing_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._committer = SerializedCommitter()
        self._write_policy = SemanticWritePolicy()
        self._representative_views = RepresentativeViewStore()
        self._refinement_queue = queue.Queue(maxsize=32)
        self._refinement_pending: set[str] = set()
        self._refinement_attempted_stamp: dict[str, int] = {}
        self._refinement_worker = None
        self._label_refiner = None
        self._refinement_shutdown = threading.Event()
        self._declare_parameters()

        self.global_frame = self.get_parameter("global_frame").value
        self.text_prompt = self.get_parameter("text_prompt").value
        self.min_points = int(self.get_parameter("min_points").value)
        self.depth_trunc_m = float(self.get_parameter("depth_trunc_m").value)
        self._label_aliases = parse_label_aliases(str(self.get_parameter("allowed_label_aliases_json").value))
        self.tf_timeout = Duration(seconds=float(self.get_parameter("tf_timeout_sec").value))
        self.mapping_backend = validate_mapping_backend(self.get_parameter("mapping_backend").value)
        self._last_validated_tf_stamp_ns = 0
        self._run_admission_open = True

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
            stale_after_sec=float(self.get_parameter("stale_after_sec").value),
        )
        self._lifecycle = OnlineLifecycleCoordinator(
            self._tracker,
            move_distance_m=float(self.get_parameter("association_distance_m").value),
            move_stability_m=float(self.get_parameter("move_stability_m").value),
            move_confirmations=int(self.get_parameter("move_confirmations").value),
            move_confirmation_max_gap_sec=float(self.get_parameter("track_state_confirmation_gap_sec").value),
        )
        self._track_state_persisted_ns: dict[str, int] = {}
        self._track_state_watermarks_ns: dict[str, int] = {}
        self._track_state_session_ids: dict[str, str] = {}
        self._retired_track_state_sessions: dict[str, set[str]] = {}
        database_path = str(Path(self.get_parameter("database_path").value).expanduser())
        semantic_identities = {}
        self._gdino_observation_identity = None
        if self.mapping_backend == "service":
            semantic_identities = {
                "sam2": SemanticIdentity.from_json(self.get_parameter("sam_model_identity").value),
                "ram_plus": SemanticIdentity.from_json(self.get_parameter("ram_plus_model_identity").value),
                "siglip2_image": SemanticIdentity.from_json(self.get_parameter("siglip2_model_identity").value),
            }
        elif self.get_parameter("gdino_model_identity").value:
            self._gdino_observation_identity = SemanticIdentity.from_json(
                self.get_parameter("gdino_model_identity").value
            )
        self._manifest = SemanticMapManifest(
            global_frame=self.global_frame,
            geometry_map_id=self.get_parameter("geometry_map_id").value,
            geometry_map_hash=self.get_parameter("geometry_map_hash").value,
            localization_session_id=self.get_parameter("localization_session_id").value,
            calibration_id=self.get_parameter("calibration_id").value,
            urdf_hash=self.get_parameter("urdf_hash").value,
            coordinate_convention=self.get_parameter("coordinate_convention").value,
            semantic_identities=semantic_identities,
            settings={"mapping_backend": self.mapping_backend},
        )
        self._database = SemanticMapDatabase(database_path, self._manifest)
        self._run = None
        self._run_pin = None
        for track in self._database.load():
            self._tracker.add_track(track)
        # Persisted mapping-session stamps protect the loaded map from replayed
        # observations, but they must not order a fresh tracking session: the
        # offline map was built on a different clock. TrackState freshness is
        # enforced by the node-clock age gate, and ordering by this run's
        # live perception stamps and per-session track-state watermarks.
        self._live_seen_ns: dict[str, int] = {}
        self._semantic_commit_watermark_ns = max(
            (track.last_seen_ns for track in self._tracker.tracks.values()), default=0
        )

        self._detector = None
        self._siglip = None
        self._service_pipeline = None
        self._gdino_confirmation_client = None
        if self.mapping_backend == "embedded":
            detector_backend = self.get_parameter("detector_backend").value
            self.get_logger().info(f"Loading Grounding DINO and SAM2 models with backend={detector_backend}")
            if detector_backend == "huggingface":
                self._detector = HFGroundedSAM2(
                    grounding_model_path=self.get_parameter("grounding_model_path").value,
                    sam_checkpoint=self.get_parameter("sam_checkpoint").value,
                    sam_config=self.get_parameter("sam_config").value,
                    device=self.get_parameter("device").value,
                )
            elif detector_backend == "native":
                from perception_service.grounded_sam2_wrapper import GroundedSAM2Wrapper

                self._detector = GroundedSAM2Wrapper(
                    device=self.get_parameter("device").value,
                    sam_checkpoint=self.get_parameter("sam_checkpoint").value,
                    sam_config=self.get_parameter("sam_config").value,
                    gdino_checkpoint=self.get_parameter("gdino_checkpoint").value,
                    gdino_text_encoder=self.get_parameter("gdino_text_encoder").value,
                    model_dir=self.get_parameter("model_dir").value or None,
                )
            else:
                raise ValueError("detector_backend must be 'huggingface' or 'native'")
            if self.get_parameter("siglip_enabled").value:
                self.get_logger().info("Loading SigLIP image encoder")
                self._siglip = SigLIPEncoder(
                    self.get_parameter("siglip_model_path").value,
                    device=self.get_parameter("device").value,
                )
        else:
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
            self._service_pipeline = ServiceFramePipeline(
                self._sam_client,
                self._ram_client,
                self._siglip_client,
                max_masks_per_batch=int(self.get_parameter("max_masks_per_batch").value),
                excluded_labels=self.get_parameter("excluded_labels").value or (),
                max_mask_candidates=int(self.get_parameter("max_label_candidates_per_mask").value),
            )
            gdino_endpoint = self.get_parameter("gdino_confirmation_service").value
            if gdino_endpoint:
                self._gdino_confirmation_client = self.create_client(
                    GroundingDetect, gdino_endpoint, callback_group=service_group
                )
            now_ns = time.time_ns()
            self._run = MappingRunRecord(
                run_id=str(uuid.uuid4()),
                configuration_generation=int(self.get_parameter("configuration_generation").value),
                expected_service_instance_ids={
                    "sam2": self.get_parameter("sam_service_instance_id").value,
                    "ram_plus": self.get_parameter("ram_plus_service_instance_id").value,
                    "siglip2_image": self.get_parameter("siglip2_service_instance_id").value,
                },
                required_semantic_identities=self._manifest.canonical_semantic_identities,
                status="active",
                started_ns=now_ns,
                updated_ns=now_ns,
            )
            self._run_pin = self._database.create_mapping_run(self._run)

        self._tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)

        state_qos = QoSProfile(depth=1)
        state_qos.reliability = ReliabilityPolicy.RELIABLE
        state_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._map_pub = self.create_publisher(
            SemanticObject3DArray, self.get_parameter("semantic_map_topic").value, state_qos
        )
        self._track_state_sub = self.create_subscription(
            TrackState,
            self.get_parameter("track_state_topic").value,
            self._track_state_callback,
            10,
        )
        self._cloud_pub = self.create_publisher(
            PointCloud2, self.get_parameter("object_cloud_topic").value, qos_profile_sensor_data
        )
        self._cloud_map_ready = False
        self._active_geometry_map_hash = ""
        self._localization_ready = False
        self._authoritative_map_odom = False
        self._footprint_ready = False
        self._obstacle_map_ready = False
        self._reachability_ready = False
        self._cloud_map_sub = self.create_subscription(
            PointCloud2,
            self.get_parameter("cloud_map_topic").value,
            self._cloud_map_callback,
            qos_profile_sensor_data,
        )
        self._active_map_hash_sub = self.create_subscription(
            String,
            self.get_parameter("active_map_hash_topic").value,
            self._active_map_hash_callback,
            state_qos,
        )
        self._readiness_subscriptions = [
            self.create_subscription(
                Bool,
                self.get_parameter(topic_parameter).value,
                lambda message, attribute=attribute: setattr(self, attribute, bool(message.data)),
                state_qos,
            )
            for topic_parameter, attribute in (
                ("localization_ready_topic", "_localization_ready"),
                ("authoritative_map_odom_topic", "_authoritative_map_odom"),
                ("footprint_ready_topic", "_footprint_ready"),
                ("obstacle_map_ready_topic", "_obstacle_map_ready"),
                ("reachability_ready_topic", "_reachability_ready"),
            )
        ]
        self._amcl_pose_received_ns = 0
        self._costmap_received_ns = 0
        if bool(self.get_parameter("auto_readiness_enabled").value):
            self._amcl_sub = self.create_subscription(
                PoseWithCovarianceStamped,
                str(self.get_parameter("amcl_pose_topic").value),
                self._amcl_pose_callback,
                10,
            )
            self._costmap_sub = self.create_subscription(
                OccupancyGrid,
                str(self.get_parameter("nav2_costmap_topic").value),
                self._costmap_auto_callback,
                1,
            )
            self._auto_readiness_timer = self.create_timer(
                1.0, self._auto_readiness_callback, callback_group=MutuallyExclusiveCallbackGroup()
            )
        self._query_service = self.create_service(
            GetSemanticObjects, self.get_parameter("query_service").value, self._query_callback
        )
        self._target_service = self.create_service(
            ResolveSemanticTarget,
            self.get_parameter("target_service").value,
            self._target_callback,
        )
        self._encode_text_client = self.create_client(
            EncodeText,
            self.get_parameter("encode_text_service").value,
            callback_group=ReentrantCallbackGroup(),
        )

        rgb_sub = message_filters.Subscriber(
            self, Image, self.get_parameter("rgb_topic").value, qos_profile=qos_profile_sensor_data
        )
        depth_sub = message_filters.Subscriber(
            self, Image, self.get_parameter("depth_topic").value, qos_profile=qos_profile_sensor_data
        )
        info_sub = message_filters.Subscriber(
            self, CameraInfo, self.get_parameter("camera_info_topic").value, qos_profile=qos_profile_sensor_data
        )
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, info_sub],
            queue_size=int(self.get_parameter("sync_queue_size").value),
            slop=float(self.get_parameter("sync_slop_sec").value),
        )
        self._synchronizer.registerCallback(self._synchronized_callback)
        self._frame_queue = BoundedFrameQueue(
            capacity=int(self.get_parameter("frame_queue_capacity").value),
            policy=self.get_parameter("frame_queue_policy").value,
        )
        self._queue_timer = self.create_timer(0.01, self._process_queued_frame)
        self._stale_timer = self.create_timer(1.0, self._stale_callback)
        self._publish_map()
        self._start_label_refinement()
        self.get_logger().info(f"Semantic mapping ready; database={database_path}, global_frame={self.global_frame}")

    def _declare_parameters(self) -> None:
        defaults = {
            "rgb_topic": "/camera/front/image_raw",
            "depth_topic": "/camera/front/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/front/camera_info",
            "global_frame": "camera_init",
            "text_prompt": "object",
            "semantic_map_topic": "/semantic_mapping/objects",
            "object_cloud_topic": "/semantic_mapping/object_cloud",
            "cloud_map_topic": "/cloud_map",
            "active_map_hash_topic": "/slam/active_geometry_map_hash",
            "localization_ready_topic": "/slam/localization_ready",
            "authoritative_map_odom_topic": "/slam/authoritative_map_odom_ready",
            "query_service": "/semantic_mapping/get_objects",
            "target_service": "/semantic_mapping/resolve_target",
            "track_state_topic": "/object_tracker/track_state",
            "track_state_frame": "odom",
            "track_state_updates_enabled": False,
            "track_state_max_age_sec": 1.0,
            "track_state_max_covariance_m2": 0.25,
            "track_state_confirmation_gap_sec": 1.0,
            "track_state_persist_interval_sec": 1.0,
            "encode_text_service": "/siglip2_service/encode_text",
            "gdino_confirmation_service": "",
            "robot_position_x": 0.0,
            "robot_position_y": 0.0,
            "robot_position_z": 0.0,
            "footprint_ready_topic": "/navigation/footprint_ready",
            "obstacle_map_ready_topic": "/navigation/obstacle_map_ready",
            "reachability_ready_topic": "/navigation/reachability_ready",
            "amcl_pose_topic": "/amcl_pose",
            "nav2_costmap_topic": "/global_costmap/costmap",
            "auto_readiness_enabled": True,
            "auto_readiness_timeout_sec": 5.0,
            "database_path": "~/.ros/ibrobot/semantic_map.sqlite3",
            "mapping_backend": "embedded",
            "sam_service": "/sam2_service/generate_masks",
            "ram_plus_service": "/ram_plus_service/recognize_tags",
            "siglip2_service": "/siglip2_service/encode_embeddings",
            "configuration_generation": 0,
            "sam_service_instance_id": "sam2",
            "ram_plus_service_instance_id": "ram_plus",
            "siglip2_service_instance_id": "siglip2_image",
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
            "sync_queue_size": 10,
            "sync_slop_sec": 0.06,
            "tf_timeout_sec": 0.15,
            "processing_interval_sec": 0.5,
            "frame_queue_capacity": 2,
            "frame_queue_policy": "drop_oldest",
            "robot_mode": "mapping",
            "base_stable": True,
            "scan_epoch": 0,
            "write_override": False,
            "box_threshold": 0.35,
            "text_threshold": 0.25,
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
            "min_frame_valid_depth_ratio": 0.05,
            "max_masks_per_frame": 32,
            "max_masks_per_batch": 8,
            "min_mask_pixels": 30,
            "min_mask_area_ratio": 0.0005,
            "min_mask_valid_depth_ratio": 0.2,
            "max_mask_overlap_ratio": 0.8,
            "association_distance_m": 0.45,
            "association_max_size_ratio": 4.0,
            "association_position_weight": 0.55,
            "embedding_similarity_threshold": 0.72,
            "label_switch_confidence_margin": 0.05,
            "label_recurrence_count_ratio": 3.0,
            "label_high_confidence_override_margin": 0.08,
            "allowed_label_aliases_json": "",
            "min_label_confidence": 0.2,
            "max_label_candidates_per_mask": 5,
            "stale_after_sec": 10.0,
            "move_stability_m": 0.1,
            "move_confirmations": 2,
            "device": "cuda",
            "detector_backend": "native",
            "grounding_model_path": "",
            "model_dir": "",
            "sam_checkpoint": "sam2.1_hiera_tiny/assets/sam2.1_hiera_tiny.pt",
            "sam_config": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "gdino_checkpoint": "grounded_sam2_swint_ogc/assets/groundingdino_swint_ogc.pth",
            "gdino_text_encoder": "grounded_sam2_swint_ogc/assets/bert-base-uncased",
            "siglip_enabled": True,
            "siglip_model_path": "siglip2_so400m_patch14_384/assets/model",
            "label_refinement_enabled": False,
            "label_refinement_model": "",
            "label_refinement_model_identity": "",
            "label_refinement_prompt": "Identify the single physical object in this masked crop.",
            "label_refinement_min_confidence": 0.8,
            "label_refinement_trigger_below_confidence": 0.7,
            "label_refinement_min_observations": 1,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.declare_parameter("excluded_labels", Parameter.Type.STRING_ARRAY)
        self._last_processed_ns = 0

    def _synchronized_callback(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        if self.mapping_backend == "service" and not self._run_admission_open:
            return
        stamp_ns = _stamp_ns(rgb_msg.header.stamp)
        minimum_interval_ns = int(float(self.get_parameter("processing_interval_sec").value) * 1e9)
        if stamp_ns - self._last_processed_ns < minimum_interval_ns:
            return
        self._frame_queue.put((rgb_msg, depth_msg, info_msg, stamp_ns, int(self.get_parameter("scan_epoch").value)))

    def _cloud_map_callback(self, message: PointCloud2) -> None:
        self._cloud_map_ready = bool(message.header.frame_id and message.width * message.height > 0)

    def _active_map_hash_callback(self, message: String) -> None:
        self._active_geometry_map_hash = message.data.strip()

    def _amcl_pose_callback(self, message: PoseWithCovarianceStamped) -> None:
        self._amcl_pose_received_ns = time.time_ns()

    def _costmap_auto_callback(self, message: OccupancyGrid) -> None:
        if message.width * message.height > 0:
            self._costmap_received_ns = time.time_ns()

    def _auto_readiness_callback(self) -> None:
        """Auto-derive SLAM and navigation readiness when no external publishers exist.

        External publishers (SLAM stack, nav2 bridge) take precedence — their Bool
        messages set the flags directly via the readiness subscriptions.  This timer
        is a fallback: when those topics have no publisher, it derives the flags from
        observable ROS signals (AMCL pose, TF availability, costmap publication).
        """
        now = time.time_ns()
        timeout_ns = int(float(self.get_parameter("auto_readiness_timeout_sec").value) * 1e9)
        # localization: AMCL publishing within timeout
        if (
            not self._localization_ready
            and self._amcl_pose_received_ns
            and now - self._amcl_pose_received_ns < timeout_ns
        ):
            self._localization_ready = True
        # map->odom TF available
        if not self._authoritative_map_odom:
            try:
                self._tf_buffer.can_transform(self.global_frame, "odom", Time())
                self._authoritative_map_odom = True
            except TransformException:
                pass
        # timestamped camera TF: map->base_link lookup works
        if self._last_validated_tf_stamp_ns == 0:
            try:
                self._tf_buffer.lookup_transform(self.global_frame, "base_link", Time(), timeout=Duration(seconds=0.1))
                self._last_validated_tf_stamp_ns = now
            except (TransformException, Exception):
                pass
        # footprint + obstacle from costmap
        if not self._footprint_ready and self._costmap_received_ns and now - self._costmap_received_ns < timeout_ns:
            self._footprint_ready = True
            self._obstacle_map_ready = True
        # reachability: implied when costmap is live
        if not self._reachability_ready and self._footprint_ready:
            self._reachability_ready = True

    def _process_queued_frame(self) -> None:
        if not self._processing_lock.acquire(blocking=False):
            return
        queued = self._frame_queue.get(timeout=0.0)
        if queued is None:
            self._processing_lock.release()
            return
        rgb_msg, depth_msg, info_msg, stamp_ns, frame_scan_epoch = queued.payload
        try:
            if self.mapping_backend == "service":
                self._start_service_frame(rgb_msg, depth_msg, info_msg, stamp_ns, frame_scan_epoch)
                return
            self._process_frame(rgb_msg, depth_msg, info_msg, stamp_ns, frame_scan_epoch=frame_scan_epoch)
            self._last_processed_ns = stamp_ns
        except Exception as exc:
            self.get_logger().error(f"Semantic frame processing failed: {exc}")
            if self.mapping_backend == "service":
                self._processing_lock.release()
        finally:
            if self.mapping_backend != "service":
                self._processing_lock.release()

    def _start_service_frame(self, rgb_msg, depth_msg, info_msg, stamp_ns, frame_scan_epoch):
        if not self._run_admission_open:
            raise RuntimeError("mapping run admission is closed")
        if not all(client.service_is_ready() for client in (self._sam_client, self._ram_client, self._siglip_client)):
            raise RuntimeError("service-backed semantic perception is not ready")
        depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        intrinsics = np.asarray(info_msg.k, dtype=np.float64).reshape(3, 3)
        camera_frame = rgb_msg.header.frame_id or info_msg.header.frame_id
        camera_transform = self._tf_buffer.lookup_transform(
            self.global_frame,
            camera_frame,
            Time.from_msg(rgb_msg.header.stamp),
            timeout=self.tf_timeout,
        )
        base_transform = self._tf_buffer.lookup_transform(
            self.global_frame,
            str(self.get_parameter("ground_reference_frame").value),
            Time.from_msg(rgb_msg.header.stamp),
            timeout=self.tf_timeout,
        )
        camera_translation = np.asarray(
            [
                camera_transform.transform.translation.x,
                camera_transform.transform.translation.y,
                camera_transform.transform.translation.z,
            ]
        )
        camera_rotation = quaternion_matrix(
            camera_transform.transform.rotation.x,
            camera_transform.transform.rotation.y,
            camera_transform.transform.rotation.z,
            camera_transform.transform.rotation.w,
        )
        ground_height = float(base_transform.transform.translation.z) + float(
            self.get_parameter("ground_height_offset_m").value
        )
        base_xy = np.asarray([base_transform.transform.translation.x, base_transform.transform.translation.y])

        def mask_selector(detections):
            return select_geometry_mask_indices(
                detections,
                lambda mask: self._bridge.imgmsg_to_cv2(mask, desired_encoding="mono8"),
                depth,
                intrinsics,
                _depth_scale(depth_msg.encoding),
                self.depth_trunc_m,
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

        result = self._service_pipeline.process(
            rgb_msg,
            mask_options={
                "max_masks": int(self.get_parameter("max_masks_per_frame").value),
                "min_mask_pixels": int(self.get_parameter("min_mask_pixels").value),
                "min_mask_area_ratio": float(self.get_parameter("min_mask_area_ratio").value),
                "max_overlap_ratio": float(self.get_parameter("max_mask_overlap_ratio").value),
            },
            mask_selector=mask_selector,
        )

        def completed(future):
            try:
                service_result = future.result()
                try:
                    provenances = self._run_pin.validate_frame(service_result.model_diagnostics)
                except MappingRunPinMismatch as exc:
                    self._stop_pinned_run(str(exc))
                    raise
                deployment_provenance = {role: value.to_dict() for role, value in provenances.items()}
                embeddings = {int(item.mask_index): item for item in service_result.embeddings if item.success}
                excluded_labels = self.get_parameter("excluded_labels").value or ()
                detections = []
                for index, message in enumerate(service_result.masks.detections):
                    encoded = embeddings.get(index)
                    candidates = ram_mask_candidates(
                        index,
                        service_result.mask_tag_counts,
                        service_result.mask_tags,
                        service_result.mask_tag_scores,
                        excluded_labels,
                        self._label_aliases,
                    )
                    label, confidence = select_ram_label(
                        index,
                        service_result.mask_tag_counts,
                        service_result.mask_tags,
                        service_result.mask_tag_scores,
                        float(self.get_parameter("min_label_confidence").value),
                        excluded_labels,
                        self._label_aliases,
                    )
                    if not label:
                        continue
                    detections.append(
                        SimpleNamespace(
                            label=label,
                            confidence=confidence,
                            bbox_xyxy=np.asarray(message.bbox, dtype=np.float32),
                            mask=self._bridge.imgmsg_to_cv2(message.mask, desired_encoding="mono8"),
                            embedding=(None if encoded is None else np.asarray(encoded.embedding, dtype=np.float32)),
                            label_candidates=candidates,
                        )
                    )
                self._process_frame(
                    rgb_msg,
                    depth_msg,
                    info_msg,
                    stamp_ns,
                    detections=detections,
                    frame_scan_epoch=frame_scan_epoch,
                    semantic_identities={
                        role: identity.to_dict()
                        for role, identity in self._manifest.canonical_semantic_identities.items()
                    },
                    deployment_provenance=deployment_provenance,
                    mapping_run_id=self._run.run_id,
                    geometry_prefiltered=True,
                )
                self._last_processed_ns = stamp_ns
            except Exception as exc:
                self.get_logger().error(f"Service-backed semantic frame failed: {exc}")
            finally:
                self._processing_lock.release()

        result.add_done_callback(completed)

    def _process_frame(
        self,
        rgb_msg: Image,
        depth_msg: Image,
        info_msg: CameraInfo,
        stamp_ns: int,
        *,
        detections=None,
        frame_scan_epoch=None,
        semantic_identities=None,
        deployment_provenance=None,
        mapping_run_id="",
        geometry_prefiltered=False,
    ) -> None:
        image_bgr = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        camera_frame = rgb_msg.header.frame_id or info_msg.header.frame_id
        intrinsics = np.asarray(info_msg.k, dtype=np.float64).reshape(3, 3)

        transform = self._tf_buffer.lookup_transform(
            self.global_frame,
            camera_frame,
            Time.from_msg(rgb_msg.header.stamp),
            timeout=self.tf_timeout,
        )
        self._last_validated_tf_stamp_ns = stamp_ns
        translation = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ]
        )
        quaternion = transform.transform.rotation
        rotation = quaternion_matrix(quaternion.x, quaternion.y, quaternion.z, quaternion.w)
        frame = prepare_frame(
            image_bgr=image_bgr,
            depth=depth,
            intrinsics=intrinsics,
            depth_scale=_depth_scale(depth_msg.encoding),
            rgb_stamp_ns=stamp_ns,
            depth_stamp_ns=_stamp_ns(depth_msg.header.stamp),
            info_stamp_ns=_stamp_ns(info_msg.header.stamp),
            camera_frame=camera_frame,
            translation=translation,
            rotation=rotation,
            max_stamp_skew_ns=int(float(self.get_parameter("sync_slop_sec").value) * 1e9),
            depth_trunc_m=self.depth_trunc_m,
            min_valid_depth_ratio=float(self.get_parameter("min_frame_valid_depth_ratio").value),
        )
        slam = evaluate_slam_readiness(
            expected_map_hash=self._manifest.geometry_map_hash,
            active_map_hash=self._active_geometry_map_hash,
            localization_ready=self._localization_ready,
            authoritative_map_odom=self._authoritative_map_odom,
            cloud_map_ready=self._cloud_map_ready,
            timestamped_tf_ready=self._last_validated_tf_stamp_ns == stamp_ns,
        )
        if not slam.ready:
            raise RuntimeError(slam.reason)

        if detections is None:
            detections = self._detector.detect_and_segment(
                image_bgr,
                self.text_prompt,
                box_threshold=float(self.get_parameter("box_threshold").value),
                text_threshold=float(self.get_parameter("text_threshold").value),
            )
            if self._gdino_observation_identity is not None:
                semantic_identities = {
                    **(semantic_identities or {}),
                    "grounding_dino": self._gdino_observation_identity.to_dict(),
                }
        if self._label_aliases:
            allowed_detections = []
            for detection in detections:
                canonical = canonicalize_label(detection.label, self._label_aliases)
                if not canonical:
                    continue
                detection.label = canonical
                allowed_detections.append(detection)
            detections = allowed_detections
        accepted_indices, diagnostics = filter_masks(
            frame,
            [MaskCandidate(detection.mask, detection.confidence) for detection in detections],
            max_masks=int(self.get_parameter("max_masks_per_frame").value),
            min_mask_pixels=int(self.get_parameter("min_mask_pixels").value),
            min_mask_area_ratio=float(self.get_parameter("min_mask_area_ratio").value),
            min_valid_depth_ratio=float(self.get_parameter("min_mask_valid_depth_ratio").value),
            max_overlap_ratio=float(self.get_parameter("max_mask_overlap_ratio").value),
            depth_trunc_m=self.depth_trunc_m,
        )
        # Unique-item ordering: largest masks claim their tracks first so a
        # whole-object mask is preferred over any part mask of the same item.
        detections = sorted(
            (detections[index] for index in accepted_indices),
            key=lambda detection: -int(np.count_nonzero(detection.mask)),
        )
        self.get_logger().debug(
            "Mask filtering: "
            f"input={diagnostics.input_count}, accepted={diagnostics.accepted_count}, "
            f"invalid={diagnostics.rejected_invalid}, small={diagnostics.rejected_too_small}, "
            f"depth={diagnostics.rejected_depth}, overlap={diagnostics.rejected_overlap}, "
            f"limit={diagnostics.rejected_limit}"
        )
        frame_clouds = []
        ground_transform = None
        ground_height = None
        if not geometry_prefiltered and bool(self.get_parameter("ground_filter_enabled").value):
            ground_transform = self._tf_buffer.lookup_transform(
                self.global_frame,
                str(self.get_parameter("ground_reference_frame").value),
                Time.from_msg(rgb_msg.header.stamp),
                timeout=self.tf_timeout,
            )
            ground_height = float(ground_transform.transform.translation.z) + float(
                self.get_parameter("ground_height_offset_m").value
            )
        matched_object_ids = set()
        frame_scan_epoch = int(self.get_parameter("scan_epoch").value) if frame_scan_epoch is None else frame_scan_epoch
        admission = self._write_policy.admit(
            mode=self.get_parameter("robot_mode").value,
            base_stable=bool(self.get_parameter("base_stable").value),
            frame_scan_epoch=frame_scan_epoch,
            active_scan_epoch=int(self.get_parameter("scan_epoch").value),
            override=bool(self.get_parameter("write_override").value),
        )
        if not admission.allowed:
            self.get_logger().debug(admission.reason)
            return
        for detection in detections:
            geometry = project_masked_depth(
                detection.mask,
                frame.depth,
                frame.intrinsics,
                depth_scale=frame.depth_scale,
                depth_trunc_m=self.depth_trunc_m,
                min_points=self.min_points,
            )
            if geometry is None:
                continue
            world_geometry = transform_geometry(geometry, frame.translation, frame.rotation)
            if ground_transform is not None and not is_ground_object(
                world_geometry,
                ground_height,
                max_bottom_clearance_m=float(self.get_parameter("ground_max_bottom_clearance_m").value),
                max_object_height_m=float(self.get_parameter("ground_max_object_height_m").value),
                max_footprint_m=float(self.get_parameter("ground_max_footprint_m").value),
                max_object_extent_m=float(self.get_parameter("max_object_extent_m").value),
                reference_position_xy=np.asarray(
                    [ground_transform.transform.translation.x, ground_transform.transform.translation.y]
                ),
                max_horizontal_distance_m=float(self.get_parameter("max_object_distance_m").value),
            ):
                continue
            embedding = None
            if hasattr(detection, "embedding"):
                embedding = detection.embedding
            elif self._siglip is not None:
                embedding = self._siglip.encode(image_bgr, detection.mask, detection.bbox_xyxy)
            observation = SemanticObservation(
                label=detection.label,
                confidence=detection.confidence,
                position=world_geometry.centroid,
                size=world_geometry.size,
                point_count=world_geometry.points.shape[0],
                stamp_ns=stamp_ns,
                embedding=embedding,
                attributes={
                    "source_frame": camera_frame,
                    "semantic_actionable": False,
                },
                canonical_label=detection.label.casefold(),
                map_version=self._manifest.geometry_map_hash,
                session_id=self._manifest.localization_session_id,
                source_frame=camera_frame,
                semantic_identities=semantic_identities or {},
                deployment_provenance=deployment_provenance or {},
                mapping_run_id=mapping_run_id,
                label_candidates=tuple(getattr(detection, "label_candidates", ())),
            )

            track = self._committer.commit(self._commit_semantic_observation, observation, matched_object_ids)
            if track is None:
                continue
            if bool(self.get_parameter("label_refinement_enabled").value):
                with self._state_lock:
                    self._representative_views.consider(
                        self._representative_views.create(
                            track.object_id,
                            stamp_ns,
                            detection.confidence,
                            image_bgr,
                            detection.mask,
                            detection.bbox_xyxy,
                        )
                    )
                self._enqueue_label_refinement(track)
            frame_clouds.append(world_geometry.points)

        if frame_clouds:
            self._cloud_pub.publish(
                xyz_to_pointcloud2(np.concatenate(frame_clouds), rgb_msg.header.stamp, self.global_frame)
            )
        self._publish_map(stamp_ns)

    def _start_label_refinement(self) -> None:
        if not bool(self.get_parameter("label_refinement_enabled").value):
            return
        try:
            from embodied_common.vlm_api_client import VLMClient

            self._label_refiner = CloudLabelRefiner(
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
        self._refinement_worker = threading.Thread(target=self._label_refinement_loop, daemon=True)
        self._refinement_worker.start()

    def _enqueue_label_refinement(self, track: SemanticTrack) -> None:
        if self._label_refiner is None or track.observation_count < int(
            self.get_parameter("label_refinement_min_observations").value
        ):
            return
        excluded = self.get_parameter("excluded_labels").value or ()
        with self._state_lock:
            if (
                not should_refine_label(
                    track.label,
                    track.confidence,
                    excluded,
                    float(self.get_parameter("label_refinement_trigger_below_confidence").value),
                    inconsistent=len(track.attributes.get("label_evidence", {})) > 1,
                )
                or track.object_id in self._refinement_pending
            ):
                return
            view = self._representative_views.get(track.object_id)
            if view is None or view.stamp_ns <= self._refinement_attempted_stamp.get(track.object_id, 0):
                return
            try:
                self._refinement_queue.put_nowait(track.object_id)
                self._refinement_pending.add(track.object_id)
            except queue.Full:
                self.get_logger().warn("Cloud label refinement queue is full; keeping RAM++ label")

    def _label_refinement_loop(self) -> None:
        while not self._refinement_shutdown.is_set():
            try:
                object_id = self._refinement_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                with self._state_lock:
                    view = self._representative_views.get(object_id)
                    track = self._tracker.tracks.get(object_id)
                    candidates = (
                        ()
                        if track is None
                        else self._tracker.aggregated_label_candidates(
                            track,
                            excluded_labels=self.get_parameter("excluded_labels").value or (),
                        )
                    )
                if view is None:
                    continue
                self._refinement_attempted_stamp[object_id] = view.stamp_ns
                result = self._label_refiner.refine(view, candidates)
                if self._refinement_shutdown.is_set():
                    continue

                def commit_refinement(object_id=object_id, result=result):
                    with self._state_lock:
                        track = self._tracker.tracks.get(object_id)
                        if track is None:
                            return
                        apply_refinement(track, result)
                        self._database.upsert(track)

                self._committer.commit(commit_refinement)
                self._publish_map()
            except Exception as exc:
                if self._refinement_shutdown.is_set():
                    continue

                def commit_rejection(object_id=object_id, candidates=candidates, error=exc):
                    with self._state_lock:
                        track = self._tracker.tracks.get(object_id)
                        if track is None:
                            return None
                        record_refinement_rejection(
                            track,
                            candidates=candidates,
                            model_identity=self._label_refiner.model_identity,
                            error=error,
                        )
                        self._database.upsert(track)
                        return track.label, track.confidence

                ram_result = self._committer.commit(commit_rejection)
                ram_summary = "missing track" if ram_result is None else f"{ram_result[0]!r}/{ram_result[1]:.3f}"
                self.get_logger().warn(
                    f"Cloud label refinement rejected for {object_id}; "
                    f"RAM++={ram_summary}, candidates={list(candidates)}: {exc}"
                )
            finally:
                with self._state_lock:
                    self._refinement_pending.discard(object_id)

    def _stale_callback(self) -> None:
        with self._state_lock:
            if self._tracker.mark_stale(self.get_clock().now().nanoseconds):
                for track in self._tracker.tracks.values():
                    self._database.upsert(track)
                self._publish_map()

    def _commit_semantic_observation(
        self, observation: SemanticObservation, matched_object_ids: set[str]
    ) -> SemanticTrack | None:
        with self._state_lock:
            if observation.stamp_ns < self._semantic_commit_watermark_ns:
                return None
            track = self._tracker.update(observation, excluded_object_ids=matched_object_ids)
            track.attributes["semantic_actionable"] = is_manually_actionable(track)
            matched_object_ids.add(track.object_id)
            self._database.upsert(track, observation)
            self._semantic_commit_watermark_ns = max(self._semantic_commit_watermark_ns, observation.stamp_ns)
            self._live_seen_ns[track.object_id] = max(self._live_seen_ns.get(track.object_id, 0), observation.stamp_ns)
            return track

    def _accept_track_state_order_locked(self, state: TrackState, stamp_ns: int) -> bool:
        track = self._tracker.tracks.get(state.object_id)
        if track is None:
            return False
        watermark_ns = self._track_state_watermarks_ns.get(state.object_id, 0)
        # Loaded maps carry the mapping session's last_seen stamps on a foreign
        # clock; only this run's live perception and track-state watermarks
        # order incoming track states for a fresh tracking session.
        live_seen_ns = self._live_seen_ns.get(state.object_id, 0)
        current_session = self._track_state_session_ids.get(state.object_id)
        retired_sessions = self._retired_track_state_sessions.setdefault(state.object_id, set())
        if state.session_id in retired_sessions:
            return False
        if stamp_ns <= max(live_seen_ns, watermark_ns):
            self._lifecycle.reset_move_candidate(state.object_id)
            return False
        if current_session is not None and state.session_id != current_session:
            retired_sessions.add(current_session)
            self._lifecycle.reset_move_candidate(state.object_id)
        self._track_state_session_ids[state.object_id] = state.session_id
        self._track_state_watermarks_ns[state.object_id] = stamp_ns
        self._semantic_commit_watermark_ns = max(self._semantic_commit_watermark_ns, stamp_ns)
        return True

    def _track_state_callback(self, state: TrackState) -> None:
        if not bool(self.get_parameter("track_state_updates_enabled").value):
            return

        def reset_candidate() -> None:
            if state.object_id:
                with self._state_lock:
                    self._lifecycle.reset_move_candidate(state.object_id)

        if not state.object_id or not state.session_id:
            reset_candidate()
            return
        stamp_ns = _stamp_ns(state.header.stamp)
        age_sec = (self.get_clock().now().nanoseconds - stamp_ns) / 1e9
        if age_sec < 0.0 or age_sec > float(self.get_parameter("track_state_max_age_sec").value):
            reset_candidate()
            return
        if state.lifecycle_state != TrackState.TRACKING:
            with self._state_lock:
                if SemanticMappingNode._accept_track_state_order_locked(self, state, stamp_ns):
                    self._lifecycle.reset_move_candidate(state.object_id)
            return
        if (
            state.motion_state not in {TrackState.MOVING, TrackState.STATIONARY}
            or not state.measured
            or state.prediction_only
            or not state.actionable
        ):
            reset_candidate()
            return
        covariance_values = [float(state.pose.covariance[index]) for index in (0, 7)]
        if any(not np.isfinite(value) or value < 0.0 for value in covariance_values) or max(covariance_values) > float(
            self.get_parameter("track_state_max_covariance_m2").value
        ):
            reset_candidate()
            return
        source_frame = state.header.frame_id
        if source_frame != str(self.get_parameter("track_state_frame").value):
            reset_candidate()
            return
        try:
            transform = self._tf_buffer.lookup_transform(
                self.global_frame,
                source_frame,
                Time.from_msg(state.header.stamp),
                timeout=self.tf_timeout,
            )
        except TransformException as error:
            reset_candidate()
            self.get_logger().warn(f"Tracked target transform unavailable for {state.object_id}: {error}")
            return
        quaternion = transform.transform.rotation
        rotation = quaternion_matrix(quaternion.x, quaternion.y, quaternion.z, quaternion.w)
        translation = np.asarray(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ],
            dtype=np.float64,
        )
        source_position = np.asarray([state.pose.pose.position.x, state.pose.pose.position.y, 0.0], dtype=np.float64)
        if not np.all(np.isfinite(source_position)):
            reset_candidate()
            return
        transformed_position = rotation @ source_position + translation
        if not np.all(np.isfinite(transformed_position)):
            reset_candidate()
            return

        slam = evaluate_slam_readiness(
            expected_map_hash=self._manifest.geometry_map_hash,
            active_map_hash=self._active_geometry_map_hash,
            localization_ready=self._localization_ready,
            authoritative_map_odom=self._authoritative_map_odom,
            cloud_map_ready=self._cloud_map_ready,
            timestamped_tf_ready=True,
        )
        if not self._run_admission_open or not slam.ready:
            reset_candidate()
            return

        def commit_tracked_position():
            with self._state_lock:
                track = self._tracker.tracks.get(state.object_id)
                if track is None:
                    return None
                if not SemanticMappingNode._accept_track_state_order_locked(self, state, stamp_ns):
                    return None
                global_position = np.asarray(
                    [transformed_position[0], transformed_position[1], track.position[2]], dtype=np.float64
                )
                if state.motion_state == TrackState.STATIONARY:
                    self._lifecycle.reset_move_candidate(state.object_id)
                    if np.linalg.norm(global_position - track.position) > self._lifecycle.move_stability_m:
                        return None
                    previous_seen_ns = track.last_seen_ns
                    state_changed = track.state != "observed"
                    if state_changed:
                        self._tracker.transition(
                            state.object_id,
                            "observed",
                            LifecycleEvidence(
                                identity_confirmed=True,
                                geometry_confirmed=True,
                                details={"identity_source": "track_state", "stationary": True},
                            ),
                        )
                    track.last_seen_ns = max(previous_seen_ns, stamp_ns)
                    persist_interval_ns = int(float(self.get_parameter("track_state_persist_interval_sec").value) * 1e9)
                    last_persisted_ns = self._track_state_persisted_ns.get(state.object_id)
                    if (
                        not state_changed
                        and last_persisted_ns is not None
                        and stamp_ns - last_persisted_ns < persist_interval_ns
                    ):
                        return None
                elif not self._lifecycle.observe_tracked_identity(
                    state.object_id,
                    global_position,
                    stamp_ns=stamp_ns,
                    session_id=state.session_id,
                ):
                    return None
                track.last_seen_ns = max(track.last_seen_ns, stamp_ns)
                track.attributes["tracked_position_update"] = {
                    "session_id": state.session_id,
                    "source_frame": source_frame,
                    "stamp_ns": stamp_ns,
                    "confidence": float(state.confidence),
                }
                self._database.upsert(track)
                self._track_state_persisted_ns[state.object_id] = stamp_ns
                return track

        try:
            updated_track = self._committer.commit(commit_tracked_position)
        except Exception as error:
            self.get_logger().error(f"Tracked target update failed for {state.object_id}: {error}")
            return
        if updated_track is not None:
            self._publish_map(stamp_ns)

    async def _query_callback(self, request: GetSemanticObjects.Request, response: GetSemanticObjects.Response):
        database = structured_query(database_readable=True, database_compatible=True)
        if not database.ready:
            response.message = database.reason
            response.metadata = self._metadata_message(False, database.reason)
            return response
        query_embedding = None
        if request.query_text:
            readiness = text_query(
                database_readable=True,
                database_compatible=True,
                siglip2_text_ready=self._encode_text_client.service_is_ready(),
                embedding_space_compatible=True,
            )
            if not readiness.ready:
                response.message = readiness.reason
                response.metadata = self._metadata_message(False, readiness.reason)
                return response
            encoded = await self._encode_text_client.call_async(EncodeText.Request(texts=[request.query_text]))
            if not encoded.success or len(encoded.results) != 1 or not encoded.results[0].success:
                response.message = encoded.message or "query text encoding failed"
                response.metadata = self._metadata_message(False, response.message)
                return response
            try:
                text_diagnostic = RuntimeDiagnostic.from_runtime_info(encoded.model)
                require_embedding_compatibility(
                    text_diagnostic.semantic_identity,
                    self._manifest.canonical_semantic_identities["siglip2_image"],
                )
            except ValueError as exc:
                readiness = text_query(
                    database_readable=True,
                    database_compatible=True,
                    siglip2_text_ready=True,
                    embedding_space_compatible=False,
                )
                response.message = f"{readiness.reason}: {exc}"
                response.metadata = self._metadata_message(False, response.message)
                return response
            query_embedding = np.asarray(encoded.results[0].embedding, dtype=np.float32)

        region_center = None
        if request.region_radius_m > 0.0:
            region_center = np.asarray(
                [request.region_center.x, request.region_center.y, request.region_center.z], dtype=np.float64
            )
        with self._state_lock:
            captions = {}
            labeled_tracks = [track for track in self._tracker.tracks.values() if has_manual_label(track)]
            if request.query_text:
                for track in labeled_tracks:
                    caption = self._database.get_caption(track.object_id)
                    if caption is not None and caption.success:
                        captions[track.object_id] = caption.caption
            ranked = query_tracks(
                labeled_tracks,
                ObjectQuery(
                    object_ids=frozenset(request.object_ids),
                    canonical_label=request.label,
                    states=frozenset(request.states),
                    include_inactive=request.include_inactive,
                    min_confidence=float(request.min_confidence),
                    max_age_ns=int(float(request.max_age_sec) * 1e9),
                    region_center=region_center,
                    region_radius_m=float(request.region_radius_m),
                    max_results=int(request.max_results),
                    query_text=request.query_text,
                ),
                now_ns=self.get_clock().now().nanoseconds,
                query_embedding=query_embedding,
                captions=captions,
            )
        tracks = [item.track for item in ranked]
        response.semantic_map = self._map_message(tracks)
        response.metadata = self._metadata_message(True, "")
        response.success = True
        response.message = f"Returned {len(tracks)} semantic objects"
        return response

    def _metadata_message(self, compatible: bool, reason: str) -> SemanticMapMetadata:
        return SemanticMapMetadata(
            schema_version=self._manifest.schema_version,
            map_version=self._manifest.geometry_map_hash,
            geometry_map_hash=self._manifest.geometry_map_hash,
            localization_session_id=self._manifest.localization_session_id,
            global_frame=self._manifest.global_frame,
            calibration_hash=self._manifest.calibration_id,
            urdf_hash=self._manifest.urdf_hash,
            coordinate_convention=self._manifest.coordinate_convention,
            compatible=compatible,
            readiness_reason=reason,
        )

    def _target_callback(self, request, response):
        with self._state_lock:
            track = self._tracker.tracks.get(request.object_id)
        if track is None:
            response.message = f"semantic object not found: {request.object_id}"
            response.metadata = self._metadata_message(False, response.message)
            return response
        active_map_hash = self._active_geometry_map_hash
        navigation = navigation_staging(
            object_action_admissible=is_manually_actionable(track) or track.active,
            active_map_identity_compatible=bool(
                not active_map_hash or active_map_hash == self._manifest.geometry_map_hash
            ),
            localization_ready=self._localization_ready,
            authoritative_map_odom=self._authoritative_map_odom,
            timestamped_tf_ready=self._last_validated_tf_stamp_ns > 0,
            footprint_ready=self._footprint_ready,
            obstacle_map_ready=self._obstacle_map_ready,
            reachability_ready=self._reachability_ready,
        )
        if not navigation.ready:
            response.message = navigation.reason
            response.metadata = self._metadata_message(False, navigation.reason)
            return response

        confirmation = manipulation_confirmation(
            navigation=navigation,
            object_confirmation_admissible=track.state == "observed",
            gdino_ready=False,
            confirmation_sam2_ready=False,
            confirmation_result_fresh=False,
        )
        if request.require_manipulation_ready and not confirmation.ready:
            response.message = confirmation.reason
            response.metadata = self._metadata_message(False, confirmation.reason)
            return response

        def checker(candidate):
            return True, ""

        robot_position = np.asarray(
            [
                self.get_parameter("robot_position_x").value,
                self.get_parameter("robot_position_y").value,
                self.get_parameter("robot_position_z").value,
            ],
            dtype=np.float64,
        )
        resolution = resolve_target(
            track,
            robot_position,
            float(request.stand_off_distance_m),
            checker,
            require_manipulation_ready=request.require_manipulation_ready,
        )
        response.object = self._track_message(track)
        response.metadata = self._metadata_message(resolution.ready, resolution.message if not resolution.ready else "")
        if not resolution.ready:
            response.message = resolution.message
            return response
        staging = resolution.staging
        response.staging_pose = PoseStamped()
        response.staging_pose.header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id=self._manifest.global_frame
        )
        response.staging_pose.pose.position.x = float(staging.position[0])
        response.staging_pose.pose.position.y = float(staging.position[1])
        response.staging_pose.pose.position.z = float(staging.position[2])
        response.staging_pose.pose.orientation = Quaternion(
            z=float(np.sin(staging.yaw / 2.0)), w=float(np.cos(staging.yaw / 2.0))
        )
        response.clearance_m = float(staging.clearance_m)
        response.navigation_ready = True
        response.manipulation_ready = confirmation.ready
        response.success = True
        response.message = resolution.message
        return response

    def _stop_pinned_run(self, reason: str) -> None:
        if self._run is None or not self._run_admission_open:
            return
        self._run_admission_open = False
        now_ns = time.time_ns()
        self._database.update_mapping_run_status(
            self._run.run_id,
            "failed",
            now_ns,
            ended_ns=now_ns,
            reason=reason,
        )
        self.get_logger().error(f"Mapping run stopped: {reason}")

    def _publish_map(self, stamp_ns: int | None = None) -> None:
        with self._state_lock:
            tracks = [track for track in self._tracker.tracks.values() if has_manual_label(track)]
        self._map_pub.publish(self._map_message(tracks, stamp_ns))

    def _map_message(self, tracks, stamp_ns: int | None = None) -> SemanticObject3DArray:
        stamp = _time_message(self.get_clock().now().nanoseconds if stamp_ns is None else stamp_ns)
        return SemanticObject3DArray(
            header=Header(stamp=stamp, frame_id=self.global_frame),
            objects=[self._track_message(track) for track in sorted(tracks, key=lambda item: item.object_id)],
        )

    def _track_message(self, track: SemanticTrack) -> SemanticObject3D:
        pose = PoseWithCovarianceStamped()
        pose.header = Header(stamp=_time_message(track.last_seen_ns), frame_id=self.global_frame)
        pose.pose.pose.position.x, pose.pose.pose.position.y, pose.pose.pose.position.z = track.position.tolist()
        pose.pose.pose.orientation.w = 1.0
        covariance = max(float(np.linalg.norm(track.size)) * 0.01, 1e-4)
        pose.pose.covariance[0] = covariance
        pose.pose.covariance[7] = covariance
        pose.pose.covariance[14] = covariance
        caption_record = self._database.get_caption(track.object_id)
        semantic_actionable = is_manually_actionable(track)
        action_ready = track.active and semantic_actionable
        return SemanticObject3D(
            object_id=track.object_id,
            object_version=track.object_version,
            label=track.label,
            query_labels=[track.canonical_label],
            confidence=track.confidence,
            pose=pose,
            size=Vector3(x=float(track.size[0]), y=float(track.size[1]), z=float(track.size[2])),
            first_seen=_time_message(track.first_seen_ns),
            last_seen=_time_message(track.last_seen_ns),
            observation_count=track.observation_count,
            point_count=track.point_count,
            active=track.active,
            state=track.state,
            action_ready=action_ready,
            readiness_reason=(
                ""
                if action_ready
                else (
                    f"object state {track.state} is not action-ready"
                    if not track.active
                    else f"label {track.canonical_label} is not manipulation-actionable"
                )
            ),
            map_version=track.map_version,
            observation_source_frame=str(track.attributes.get("source_frame", "")),
            model_versions_json=json.dumps(track.model_versions, ensure_ascii=True, sort_keys=True),
            caption=caption_record.caption if caption_record is not None and caption_record.success else "",
            attributes_json=json.dumps(track.attributes, ensure_ascii=True, sort_keys=True),
        )

    def destroy_node(self):
        self._refinement_shutdown.set()
        close_database = True
        if self._refinement_worker is not None:
            self._refinement_worker.join(timeout=5.0)
            if self._refinement_worker.is_alive():
                self.get_logger().warn("Cloud label refinement worker is still stopping; database remains open")
                close_database = False
        if self._run is not None and self._run_admission_open:
            now_ns = time.time_ns()
            self._database.update_mapping_run_status(self._run.run_id, "paused", now_ns)
        if close_database:
            self._database.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMappingNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, TransformException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
