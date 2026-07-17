#!/usr/bin/env python3
"""Cloud endpoint for one validated distributed inference pipeline."""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from ibrobot_msgs.msg import DistributedInferenceRequest, DistributedInferenceResult, InferencePipelineStatus
from inference_manifest import load_inference_manifest, load_inference_manifest_metadata
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
)


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


class PureInferenceNode(Node):
    """Own one cloud backend and its session-scoped request transport."""

    def __init__(self, config: CloudNodeConfig, *, node_name: str) -> None:
        super().__init__(node_name)
        try:
            runtime_options = json.loads(config.runtime_options_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"runtime_options_json is invalid: {exc}") from exc
        if not isinstance(runtime_options, dict):
            raise RuntimeError("runtime_options_json must decode to an object")

        metadata = load_inference_manifest_metadata(config.model_path, config.deployment)
        identity = build_pipeline_identity(config.pipeline_id, metadata)
        startup_error = None
        try:
            validated = load_inference_manifest(config.model_path, config.deployment)
            runtime = CloudBackendRuntime(
                config.pipeline_id,
                validated,
                request_timeout=config.request_timeout,
                runtime_options=runtime_options,
            )
        except Exception as exc:
            runtime = None
            startup_error = structured_error_from_exception(exc, "startup")
            self.get_logger().error(f"distributed cloud backend failed to start: {exc}")
        self._config = config
        self._service = DistributedCloudService(identity, runtime, startup_error=startup_error)
        self._fingerprint = metadata.fingerprint
        self._result_pub = self.create_publisher(DistributedInferenceResult, config.result_topic, 10)
        self.create_subscription(
            DistributedInferenceRequest,
            config.request_topic,
            self._request_callback,
            10,
            callback_group=ReentrantCallbackGroup(),
        )

        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status_pub = self.create_publisher(InferencePipelineStatus, config.heartbeat_topic, status_qos)
        self.create_subscription(
            InferencePipelineStatus,
            config.heartbeat_topic,
            self._edge_status_callback,
            status_qos,
            callback_group=ReentrantCallbackGroup(),
        )
        self._status_timer = self.create_timer(0.5, self._publish_status)
        self.get_logger().info(
            f"Distributed cloud pipeline loaded: id={config.pipeline_id}, deployment={config.deployment}, "
            f"backend={metadata.deployment.backend}, request={config.request_topic}, result={config.result_topic}"
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

    def _request_callback(self, message: DistributedInferenceRequest) -> None:
        request = None
        try:
            request = request_from_message(message)
            result = self._service.handle(request)
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
    try:
        config, node_name = _read_config()
        node = PureInferenceNode(config, node_name=node_name)
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
