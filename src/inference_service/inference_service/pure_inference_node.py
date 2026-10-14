#!/usr/bin/env python3
"""Cloud endpoint for one validated distributed inference pipeline."""

from __future__ import annotations

import json
import threading
import traceback
from dataclasses import dataclass

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from ibrobot_msgs.msg import (
    DistributedInferenceRequest,
    DistributedInferenceResult,
    InferencePipelineStatus,
    VideoStreamDescriptor,
    VideoStreamStatus,
)
from inference_manifest import load_inference_manifest, load_inference_manifest_metadata
from inference_service.compute_video_streams import ComputeVideoStreamManager
from inference_service.distributed import (
    CloudBackendRuntime,
    DistributedCloudService,
    build_pipeline_identity,
    structured_error_from_exception,
)
from inference_service.distributed.ros_protocol import (
    decode_failure_result,
    request_from_message,
    result_to_message,
    status_from_message,
    status_to_message,
    transport_failure_result,
    video_descriptor_from_message,
    video_status_from_message,
    video_status_to_message,
)
from inference_service.runtime_composition import (
    build_policy_runtime_dependencies,
    require_runtime_dependencies,
)
from inference_service.unified_runtime import RegistrySet, RuntimeProviders
from robot_config.contract_utils import contract_fingerprint, iter_specs

# Receiver status is a heartbeat plus diagnostics; lifecycle transitions are
# published immediately through a guard condition, so the periodic cadence
# only bounds how stale the counters can get. A faster timer takes the GIL
# away from the RTP receive/decode threads in the same process.
_VIDEO_STATUS_PERIOD_S = 0.25
_VIDEO_DIAGNOSTICS_PERIOD_S = 2.0


@dataclass(frozen=True)
class CloudNodeConfig:
    pipeline_id: str
    model_path: str
    deployment: str
    request_timeout: float
    runtime_options_json: str
    request_topic: str
    result_topic: str
    heartbeat_topic: str
    robot_config_path: str
    video_descriptor_topic: str
    video_status_topic: str


class PureInferenceNode(Node):
    """Own one cloud backend and its session-scoped request transport."""

    def __init__(
        self,
        config: CloudNodeConfig,
        *,
        node_name: str,
        registry_set: RegistrySet | None = None,
        providers: RuntimeProviders | None = None,
        runtime: CloudBackendRuntime | None = None,
        inference_admission=None,
        use_global_arguments: bool = True,
    ) -> None:
        super().__init__(node_name, use_global_arguments=use_global_arguments)
        registry_set, providers = require_runtime_dependencies(
            registry_set,
            providers,
            owner=type(self).__name__,
        )
        self._registry_set = registry_set
        self._providers = providers
        try:
            runtime_options = json.loads(config.runtime_options_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"runtime_options_json is invalid: {exc}") from exc
        if not isinstance(runtime_options, dict):
            raise RuntimeError("runtime_options_json must decode to an object")

        metadata = load_inference_manifest_metadata(config.model_path, config.deployment)
        identity = build_pipeline_identity(config.pipeline_id, metadata)
        from robot_config.loader import build_contract_from_robot_config_dict, load_robot_config_dict

        startup_error = None
        stream_manager = None
        owns_runtime = runtime is None
        require_target_observation = False
        try:
            if config.robot_config_path:
                from robot_config.runtime_target import RuntimeTarget, resolve_runtime_target

                robot_config = load_robot_config_dict(config.robot_config_path)
                require_target_observation = resolve_runtime_target(robot_config, "", None) is RuntimeTarget.BENCHMARK
                if require_target_observation and metadata.policy.n_obs_steps != 1:
                    raise ValueError("benchmark exact observations currently require n_obs_steps=1")
                contract = build_contract_from_robot_config_dict(robot_config)
                required_inputs = set(metadata.policy.input_features)
                observation_specs = tuple(
                    spec for spec in iter_specs(contract) if not spec.is_action and spec.key in required_inputs
                )
                stream_manager = ComputeVideoStreamManager(
                    pipeline_id=config.pipeline_id,
                    session_id="pending",
                    session_generation=1,
                    contract_fingerprint=contract_fingerprint(contract),
                    deployment_fingerprint=metadata.fingerprint,
                    observation_specs=observation_specs,
                    rate_hz=float(contract.rate_hz),
                    n_obs_steps=metadata.policy.n_obs_steps,
                    require_target_observation=require_target_observation,
                )
            validated = load_inference_manifest(config.model_path, config.deployment)
            runtime = runtime or CloudBackendRuntime(
                config.pipeline_id,
                validated,
                request_timeout=config.request_timeout,
                runtime_options=runtime_options,
                registry_set=self._registry_set,
                providers=self._providers,
            )
        except Exception as exc:
            if owns_runtime:
                runtime = None
            startup_error = structured_error_from_exception(exc, "startup")
            self.get_logger().error(f"distributed cloud backend failed to start: {exc}")
        self._config = config
        self._stream_manager = stream_manager
        self._service = DistributedCloudService(
            identity,
            runtime,
            startup_error=startup_error,
            stream_manager=stream_manager,
            owns_runtime=owns_runtime,
            inference_admission=inference_admission,
            require_target_observation=require_target_observation,
        )
        self._fingerprint = metadata.fingerprint
        self._result_pub = self.create_publisher(DistributedInferenceResult, config.result_topic, 10)
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        descriptor_qos = QoSProfile(
            depth=max(1, len(stream_manager.diagnostic_snapshots()) if stream_manager is not None else 1),
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            DistributedInferenceRequest,
            config.request_topic,
            self._request_callback,
            10,
            callback_group=ReentrantCallbackGroup(),
        )
        self.create_subscription(
            VideoStreamDescriptor,
            config.video_descriptor_topic,
            self._video_descriptor_callback,
            descriptor_qos,
            callback_group=ReentrantCallbackGroup(),
        )
        self.create_subscription(
            VideoStreamStatus,
            config.video_status_topic,
            self._video_status_callback,
            10,
            callback_group=ReentrantCallbackGroup(),
        )
        self._video_status_pub = self.create_publisher(VideoStreamStatus, config.video_status_topic, 10)
        self._video_status_lock = threading.Lock()
        self._video_status_signatures: dict[str, tuple[object, ...]] = {}
        self._video_status_trigger = self.create_guard_condition(
            self._publish_changed_video_status,
            callback_group=ReentrantCallbackGroup(),
        )
        if stream_manager is not None:
            stream_manager.set_status_listener(self._video_status_trigger.trigger)
        self._status_pub = self.create_publisher(InferencePipelineStatus, config.heartbeat_topic, status_qos)
        self.create_subscription(
            InferencePipelineStatus,
            config.heartbeat_topic,
            self._edge_status_callback,
            status_qos,
            callback_group=ReentrantCallbackGroup(),
        )
        # The heartbeat timer gets its own callback group. Timers created without one
        # land in the node's default MutuallyExclusiveCallbackGroup, so this 0.5s timer
        # would share a group with the video status timer below -- and a
        # MultiThreadedExecutor cannot run two callbacks from one exclusive group at
        # once, no matter how many threads it has.
        #
        # This mattered because the video status timer used to run at 0.02s (see the
        # comment on that timer): with two rtp cameras it published 100 messages a
        # second and was ready on nearly every executor pass, which starved the
        # heartbeat. Measured on the two-machine rtp setup (2026-09-21): same process,
        # same moment, the 0.02s timer delivered a steady 100.0 Hz while this one fell
        # from 2 Hz to ~1.3 Hz with gaps up to 2.0s on loopback and 3.0s across the
        # network. The edge side declares the session dead after 2.0s of silence
        # (distributed/session.py heartbeat_timeout), so every distributed rtp run lost
        # its session seconds after the handshake and the benchmark aborted before
        # sending a single frame. The dds path never showed this because
        # _publish_video_status returns immediately when there is no stream manager and
        # publishes nothing.
        #
        # The publish rate is fixed below, so the two timers no longer compete for
        # executor passes. Keep the separate group anyway: it is what makes the
        # heartbeat independent of whatever else lands in the default group, and it is
        # cheap.
        #
        # _publish_status only calls DistributedCloudService.status(), which already
        # runs under the service's own _lifecycle_lock and is already reached
        # concurrently from the reentrant subscription callbacks above, so moving it out
        # of the default group adds no new shared state.
        self._status_timer = self.create_timer(
            0.5, self._publish_status, callback_group=MutuallyExclusiveCallbackGroup()
        )
        # 0.25s, matching recording_node._publish_video_status -- the other cloud-side
        # node that receives rtp and republishes receiver status on this same topic.
        #
        # This was 0.25s until 41f57a5e ("benchmark: integrate native observation
        # streaming") changed it to 0.02s; that commit message gives no reason, and
        # nothing subscribing to this topic needs a faster rate. The edge publishes its
        # own sender-origin status every 0.25s (pipeline_policy_node), and the only
        # readers are recording_node, the two ci_smoke clients, and this node's own
        # subscription above (which the edge's sender status is what actually matters
        # for -- compute_video_streams.observe_status ignores receiver-origin messages).
        #
        # At 0.02s with two streams this publisher put 100 RELIABLE messages a second on
        # the wire and publish() itself blocked. py-spy on the cloud host (220 samples):
        # worker threads sat inside rclpy publish() in 76 samples, each call blocking
        # 1.6-2.2s, and the heartbeat blocked in its own publish() for 4.5s
        # straight. That is inside DDS, below the executor, so the callback group fix
        # alone could not reach it. Single-variable check the same day: with this
        # publisher idle the heartbeat held 2.000 Hz / max gap 0.500s; with it running at
        # 100 msg/s the heartbeat fell to 1.0-1.3 Hz / max gap 2.0-3.0s.
        self._video_status_timer = self.create_timer(_VIDEO_STATUS_PERIOD_S, self._publish_video_status)
        self._video_diagnostics_timer = self.create_timer(
            _VIDEO_DIAGNOSTICS_PERIOD_S, self._log_video_runtime_diagnostics
        )
        self._log_video_stream_diagnostics()
        self.get_logger().info(
            f"Distributed cloud pipeline loaded: id={config.pipeline_id}, deployment={config.deployment}, "
            f"backend={metadata.deployment.backend}, request={config.request_topic}, result={config.result_topic}"
        )

    def _log_video_stream_diagnostics(self) -> None:
        if self._stream_manager is None:
            return
        for snapshot in self._stream_manager.diagnostic_snapshots():
            self.get_logger().info(
                f"Video stream startup: observation={snapshot.observation_key}, "
                f"stream_id={snapshot.stream_id}, mode={snapshot.mode}, "
                f"encoder={snapshot.configured_encoder_backend}/{snapshot.selected_encoder_backend}, "
                f"decoder={snapshot.configured_decoder_backend}/{snapshot.selected_decoder_backend}, "
                f"endpoint={snapshot.endpoint[0]}:{snapshot.endpoint[1]}, "
                f"contract_fingerprint={snapshot.contract_fingerprint}, "
                f"deployment_fingerprint={snapshot.deployment_fingerprint}, security={snapshot.security}, "
                f"lifecycle={snapshot.lifecycle_state}, ready={snapshot.ready}"
            )

    def _edge_status_callback(self, message: InferencePipelineStatus) -> None:
        if message.role != InferencePipelineStatus.ROLE_EDGE:
            return
        try:
            status = status_from_message(message)
            response = self._service.observe_edge(status)
            self._status_pub.publish(status_to_message(response, stamp=self.get_clock().now().to_msg()))
        except Exception as exc:
            self.get_logger().error(f"invalid edge handshake status: {exc}\n{traceback.format_exc()}")

    def _publish_status(self) -> None:
        try:
            status = self._service.status()
            self._status_pub.publish(status_to_message(status, stamp=self.get_clock().now().to_msg()))
        except Exception as exc:
            self.get_logger().error(f"failed to publish cloud status: {exc}")

    def _video_descriptor_callback(self, message: VideoStreamDescriptor) -> None:
        if self._stream_manager is None:
            return
        try:
            self._stream_manager.observe_descriptor(video_descriptor_from_message(message))
        except Exception as exc:
            self.get_logger().error(f"invalid video stream descriptor: {exc}")

    def _video_status_callback(self, message: VideoStreamStatus) -> None:
        # The topic is bidirectional: skip this node's own receiver statuses
        # before paying for the full value conversion.
        if self._stream_manager is None or message.status_origin != "sender":
            return
        try:
            self._stream_manager.observe_status(
                video_status_from_message(message),
                receive_time_ns=self.get_clock().now().nanoseconds,
            )
        except Exception as exc:
            self.get_logger().error(f"invalid video stream status: {exc}")

    def _publish_video_status(self) -> None:
        self._publish_video_statuses(changed_only=False)

    def _publish_changed_video_status(self) -> None:
        self._publish_video_statuses(changed_only=True)

    def _publish_video_statuses(self, *, changed_only: bool) -> None:
        if self._stream_manager is None:
            return
        with self._video_status_lock:
            stamp = self.get_clock().now().to_msg()
            for status in self._stream_manager.statuses():
                signature = (
                    status.session_id,
                    status.session_generation,
                    status.lifecycle_state,
                    status.ready,
                    status.keyframe_ready,
                )
                if changed_only and self._video_status_signatures.get(status.observation_key) == signature:
                    continue
                self._video_status_signatures[status.observation_key] = signature
                self._video_status_pub.publish(video_status_to_message(status, stamp=stamp))

    def _log_video_runtime_diagnostics(self) -> None:
        if self._stream_manager is None:
            return
        # All streams go into one log line per diagnostic. rclpy keys its throttle
        # context on the call site -- CallerId is (function_name, file_path,
        # line_number, last_index), with no message content in it (rcutils_logger.py:54).
        # A throttled log inside a for loop therefore shares one context across every
        # stream: the first stream logs, resets the 2s timer, and the rest of the
        # streams are dropped. The video status timer fires faster than the 2s throttle
        # window (see _video_status_timer), so with two cameras the second camera's
        # line never appears at all. The sending end of this same link already logs one
        # aggregated line (pipeline_policy_node.py _publish_video_stream_control);
        # this matches it.
        decoder_lines = [
            f"observation={observation_key}, "
            f"input_frames={metrics.input_frames}, output_frames={metrics.output_frames}, "
            f"input_fps={metrics.input_frame_rate_hz:.2f}, output_fps={metrics.output_frame_rate_hz:.2f}, "
            f"backlog={metrics.decoder_backlog_depth}, output_age_ms={metrics.decoder_output_age_ns / 1e6:.1f}, "
            f"dropped_stale={metrics.dropped_stale_decoder_frames}, metadata_depth={metrics.metadata_fifo_depth}"
            for observation_key, metrics in self._stream_manager.decoder_diagnostics()
        ]
        if decoder_lines:
            self.get_logger().info("Decoder runtime: " + " | ".join(decoder_lines), throttle_duration_sec=2.0)
        # Left outside the aggregation above on purpose: this log has a single
        # call site and is not inside a loop, so its throttle context is not
        # shared with anything.
        alignment_delta_ns = self._stream_manager.state_alignment_delta_ns()
        if alignment_delta_ns is not None:
            self.get_logger().info(
                f"State alignment: delta_ms={alignment_delta_ns / 1e6:.1f}",
                throttle_duration_sec=2.0,
            )
        selection_lines = [
            f"observation={observation_key}, "
            f"capture_age_ms={capture_age_ms:.1f}, live_age_ms={live_age_ms:.1f}, "
            f"selected_capture_ns={capture_ns}"
            for observation_key, capture_ns, capture_age_ms, live_age_ms in self._stream_manager.selection_diagnostics()
        ]
        if selection_lines:
            self.get_logger().info("Video selection: " + " | ".join(selection_lines), throttle_duration_sec=2.0)

    def _request_callback(self, message: DistributedInferenceRequest) -> None:
        request = None
        try:
            request = request_from_message(message)
            self.get_logger().info(
                "[IBROBOT_INFERENCE][DISTRIBUTED_REQUEST] "
                f"request={request.request_id} tensor_keys={sorted(request.inputs)} "
                f"stream_keys={[item.observation_key for item in request.stream_references]}"
            )
            result = self._service.handle(request)
            if request.operation.name == "INFER":
                performance = dict(result.performance)
                self.get_logger().info(
                    "[IBROBOT_INFERENCE][INFERENCE_TIMING] "
                    f"request={request.request_id} "
                    f"request_start_mono_ns={performance.get('request_start_monotonic_ns', 0)} "
                    f"assembly_start_mono_ns={performance.get('stream_assembly_start_monotonic_ns', 0)} "
                    f"assembly_end_mono_ns={performance.get('stream_assembly_end_monotonic_ns', 0)} "
                    f"inference_start_mono_ns={performance.get('inference_start_monotonic_ns', 0)} "
                    f"inference_end_mono_ns={performance.get('inference_end_monotonic_ns', 0)} "
                    f"result_mono_ns={performance.get('result_monotonic_ns', 0)} "
                    f"backend_latency_ms={result.backend_latency_ms:.6f}"
                )
            if request.operation.name == "INFER" and self._stream_manager is not None:
                for status in self._stream_manager.statuses():
                    self.get_logger().info(
                        "[IBROBOT_INFERENCE][VIDEO_RECEIVER] "
                        f"request={request.request_id} observation={status.observation_key} "
                        f"stream={status.stream_id} ready={status.ready} "
                        f"received_packets={status.received_packets} decoded_frames={status.decoded_frames} "
                        f"lost_packets={status.lost_packets} dropped_packets={status.dropped_packets} "
                        f"queue_overflow={status.receiver_queue_overflow_drops} "
                        f"sequence_gaps={status.sequence_gap_events} reordered={status.reordered_packets} "
                        f"recovery_keyframes={status.recovery_keyframes} jitter_ns={status.jitter_ns} "
                        f"receive_mono_ns={status.receive_monotonic_ns} "
                        f"decode_start_mono_ns={status.decode_start_monotonic_ns} "
                        f"decode_end_mono_ns={status.decode_end_monotonic_ns} "
                        f"decoded_capture_ns={status.last_decoded_capture_timestamp_ns}"
                    )
        except Exception as exc:
            result = decode_failure_result(message, exc, self._fingerprint, self._config.pipeline_id)
            self.get_logger().error(f"distributed request decode failed: {exc}")
        try:
            result_message = result_to_message(result)
        except Exception as exc:
            result = transport_failure_result(
                message,
                exc,
                self._fingerprint,
                self._config.pipeline_id,
                code="encode_failed",
                stage="encode",
                operation=request.operation if request is not None else None,
            )
            result_message = result_to_message(result)
            self.get_logger().error(f"distributed result encode failed: {exc}")
        self._result_pub.publish(result_message)

    def destroy_node(self) -> None:
        try:
            self._service.close()
        except Exception as exc:
            self.get_logger().error(f"cloud pipeline shutdown failed: {exc}")
        super().destroy_node()


def _read_config() -> tuple[CloudNodeConfig, str]:
    reader = Node("_pure_inference_param_reader")
    defaults: dict[str, object] = {
        "pipeline_id": "policy",
        "model_path": "",
        "deployment": "cpu",
        "request_timeout": 5.0,
        "runtime_options_json": "{}",
        "node_name": "inference_policy_cloud",
        "request_topic": "/inference/policy/request",
        "result_topic": "/inference/policy/result",
        "heartbeat_topic": "/inference/policy/heartbeat",
        "robot_config_path": "",
        "video_descriptor_topic": "/inference/policy/video/descriptors",
        "video_status_topic": "/inference/policy/video/status",
    }
    for name, default in defaults.items():
        reader.declare_parameter(name, default)
    values = {name: reader.get_parameter(name).value for name in defaults}
    reader.destroy_node()
    node_name = str(values.pop("node_name"))
    return CloudNodeConfig(**values), node_name


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: PureInferenceNode | None = None
    dependencies = None
    try:
        config, node_name = _read_config()
        dependencies = build_policy_runtime_dependencies()
        node = PureInferenceNode(
            config,
            node_name=node_name,
            registry_set=dependencies.registry_set,
            providers=dependencies.providers,
        )
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if dependencies is not None:
            dependencies.providers.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
