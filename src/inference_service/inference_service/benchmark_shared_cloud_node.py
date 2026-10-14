"""Shared distributed cloud owner for parallel benchmark RTP lanes."""

from __future__ import annotations

import json
import logging
from contextlib import ExitStack

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from inference_manifest import load_inference_manifest
from inference_service.benchmark_shared_policy_node import (
    FIFOAdmission,
    SharedPolicySession,
    validate_shared_manifest,
)
from inference_service.distributed.runtime import CloudBackendRuntime
from inference_service.pure_inference_node import CloudNodeConfig, PureInferenceNode
from inference_service.runtime_composition import build_policy_runtime_dependencies


def _cleanup(callback):
    try:
        callback()
    except BaseException:
        # Python 3.10 ExitStack can lose earlier callback failures from the chain.
        logging.getLogger(__name__).exception("Shared cloud cleanup failed: %s", callback)
        raise


def _configs(raw: str) -> list[CloudNodeConfig]:
    values = json.loads(raw)
    if not isinstance(values, list) or not values:
        raise ValueError("cloud_configs_json must be a non-empty list")
    configs = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("each cloud config must be an object")
        configs.append(CloudNodeConfig(**value))
    return configs


class SharedCloudHost:
    def __init__(self, configs, dependencies, *, max_in_flight=1, queue_size=64):
        if not configs:
            raise ValueError("cloud configs must be non-empty")
        if type(max_in_flight) is not int or max_in_flight < 1:
            raise ValueError("max_in_flight must be a positive int")
        if type(queue_size) is not int or queue_size < 1:
            raise ValueError("queue_size must be a positive integer")
        first = configs[0]
        runtime_options = json.loads(first.runtime_options_json)
        if not isinstance(runtime_options, dict):
            raise ValueError("runtime_options_json must decode to an object")
        identity = (
            first.pipeline_id,
            first.model_path,
            first.deployment,
            runtime_options,
            first.request_timeout,
        )
        if any(
            (
                config.pipeline_id,
                config.model_path,
                config.deployment,
                json.loads(config.runtime_options_json),
                config.request_timeout,
            )
            != identity
            for config in configs
        ):
            raise ValueError(
                "all shared cloud lanes must use identical model, deployment, pipeline, runtime options, and timeout"
            )
        topics = set()
        for config in configs:
            for field in (
                "request_topic",
                "result_topic",
                "heartbeat_topic",
                "video_descriptor_topic",
                "video_status_topic",
            ):
                topic = getattr(config, field)
                if not isinstance(topic, str) or not topic.startswith("/") or "//" in topic or topic.endswith("/"):
                    raise ValueError(f"{field} must be an absolute ROS topic")
                if topic in topics:
                    raise ValueError("shared cloud lanes must use disjoint absolute endpoint topics")
                topics.add(topic)
        manifest = load_inference_manifest(first.model_path, first.deployment)
        validate_shared_manifest(manifest)
        self.admission = FIFOAdmission(queue_size, max_active=max_in_flight)
        self.runtime = None
        self.nodes = []
        try:
            self.runtime = CloudBackendRuntime(
                first.pipeline_id,
                manifest,
                request_timeout=first.request_timeout,
                runtime_options=runtime_options,
                registry_set=dependencies.registry_set,
                providers=dependencies.providers,
                model_session_factory=lambda context, options: SharedPolicySession(
                    context.device, max_in_flight=max_in_flight
                ),
                max_active_executions=max_in_flight,
            )
            for index, config in enumerate(configs):
                self.nodes.append(
                    PureInferenceNode(
                        config,
                        node_name=f"{config.pipeline_id}_{index}",
                        registry_set=dependencies.registry_set,
                        providers=dependencies.providers,
                        runtime=self.runtime,
                        inference_admission=self.admission,
                        use_global_arguments=False,
                    )
                )
        except BaseException:
            logging.getLogger(__name__).exception("Shared cloud initialization failed")
            self.close()
            raise

    def close(self):
        nodes, self.nodes = self.nodes, []
        runtime, self.runtime = self.runtime, None
        # Detach ownership first so failed cleanup is not repeated by another close.
        with ExitStack() as cleanup:
            if runtime is not None:
                cleanup.callback(_cleanup, runtime.close)
            for node in nodes:
                cleanup.callback(_cleanup, node.destroy_node)
            cleanup.callback(_cleanup, self.admission.close)


def main(args=None):
    rclpy.init(args=args)
    host = dependencies = executor = None
    try:
        reader = Node("benchmark_shared_cloud_parameters")
        try:
            reader.declare_parameter("cloud_configs_json", "")
            reader.declare_parameter("max_in_flight", 1)
            reader.declare_parameter("queue_size", 64)
            configs = _configs(reader.get_parameter("cloud_configs_json").value)
            max_in_flight = reader.get_parameter("max_in_flight").value
            queue_size = reader.get_parameter("queue_size").value
        finally:
            reader.destroy_node()
        dependencies = build_policy_runtime_dependencies()
        host = SharedCloudHost(configs, dependencies, max_in_flight=max_in_flight, queue_size=queue_size)
        executor = MultiThreadedExecutor(num_threads=max(8, 2 * len(host.nodes) + 2))
        for node in host.nodes:
            executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            with ExitStack() as cleanup:
                if dependencies is not None:
                    cleanup.callback(_cleanup, dependencies.providers.close)
                if host is not None:
                    cleanup.callback(_cleanup, host.close)
                if executor is not None:
                    cleanup.callback(_cleanup, executor.shutdown)
                if host is not None:
                    cleanup.callback(_cleanup, host.admission.close)
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
