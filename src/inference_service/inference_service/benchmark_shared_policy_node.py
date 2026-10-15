"""One shared policy weight owner with independent legacy LIBERO policy frontends.

This executable deliberately does not change the distributed or scheduled node.
Launch replaces the per-lane edge/cloud inference nodes, not their DispatchInfer
clients. Each frontend retains its own contract subscriptions and ROS endpoints.
The hosted policy is model-neutral: admission depends only on the structural
contract in ``validate_shared_manifest``, not on the model type.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from inference_manifest import load_inference_manifest
from inference_service.backends.types import BackendAdmissionEvidence
from inference_service.model_sessions.lerobot_torch import LeRobotTorchModelSession
from inference_service.pipeline import create_pipeline_manager
from inference_service.pipeline_policy_node import (
    DeadlineExceededError,
    PipelineBusyError,
    PipelineNodeConfig,
    PipelinePolicyNode,
    RequestCanceledError,
)
from inference_service.runtime_composition import build_policy_runtime_dependencies


def validate_shared_manifest(manifest):
    """Fail closed on contracts the shared owner cannot host, before allocating weights.

    These are structural requirements of the shared runtime, not model identity:
    the owner hosts LeRobot Torch policy sessions, executes one direct predict per
    request, and interleaves lanes against one instance, which is only correct for
    request-scoped state. Any Torch policy deployment that satisfies the contract
    is admitted, regardless of model type.
    """
    model = manifest.manifest.model
    contract = manifest.deployment.execution_contract
    if model.interface != "policy":
        raise ValueError("shared benchmark hosts policy pipelines only")
    if manifest.deployment.backend != "torch":
        raise ValueError("shared benchmark supports only Torch deployments")
    if contract.state_scope != "request":
        raise ValueError(
            "shared benchmark requires request-scoped state; interleaved lanes cannot share stream-scoped state"
        )
    if contract.execution_structure != "direct":
        raise ValueError("shared benchmark executes direct predictions only")
    config = json.loads((manifest.bundle_root / "config.json").read_text())
    if config.get("rtc_config") is not None or config.get("use_relative_actions", False):
        raise ValueError(
            "shared benchmark cannot host RTC or relative-action processors: their preprocess/postprocess "
            "pairs retain per-request state outside the serialized window, so interleaved lanes would mix them"
        )
    # The config flag is not the only place relative actions can hide: the
    # trained processor chain may carry the relative/absolute pair directly.
    # A fully disabled pair is inert (pass-through on both sides), but an
    # enabled one retains cross-request state and must fail closed here.
    pre_steps = json.loads((manifest.bundle_root / "policy_preprocessor.json").read_text())["steps"]
    post_steps = json.loads((manifest.bundle_root / "policy_postprocessor.json").read_text())["steps"]
    for step in pre_steps:
        if step.get("registry_name") == "relative_actions_processor" and bool(
            (step.get("config") or {}).get("enabled", False)
        ):
            raise ValueError(
                "shared benchmark cannot host an enabled relative_actions_processor in the preprocessor chain: "
                "its paired postprocessing state would mix interleaved lanes"
            )
    for step in post_steps:
        if step.get("registry_name") == "absolute_actions_processor" and bool(
            (step.get("config") or {}).get("enabled", False)
        ):
            raise ValueError(
                "shared benchmark cannot host an enabled absolute_actions_processor in the postprocessor chain: "
                "its paired preprocessing state would mix interleaved lanes"
            )


class SharedPolicySession(LeRobotTorchModelSession):
    """Request-only chunk API over one shared Torch policy; select_action and RTC remain unsupported.

    Processor calls are serialized because LeRobot steps retain the current
    transition and normalization caches. Model execution is outside that lock,
    with a synchronized worker-local CUDA stream when concurrent execution is
    explicitly requested. Concurrency safety is the model's own property:
    PI05 has measured five-lane RTP concurrency evidence (recorded in the PR
    review discussion); other policies run under the same path but carry no
    concurrency audit yet.
    """

    def __init__(self, device_name, *, max_in_flight=1):
        if type(max_in_flight) is not int or max_in_flight < 1:
            raise ValueError("shared policy max_in_flight must be a positive int")
        if max_in_flight > 1 and not str(device_name).startswith("cuda"):
            raise ValueError(
                "shared policy concurrency above max_in_flight=1 requires a CUDA device: "
                "the concurrent execution path drives worker-local CUDA streams"
            )
        super().__init__(device_name)
        self._max_in_flight = max_in_flight
        self._stream_local = threading.local()
        self._processor_lock = threading.Lock()
        self._profile_nvtx = os.environ.get("IBROBOT_PROFILE_SHARED_POLICY") == "1"

    def preprocess(self, inputs):
        with self._processor_lock:
            return super().preprocess(inputs)

    def postprocess(self, action):
        with self._processor_lock:
            return super().postprocess(action)

    def _load(self, context, rollback):
        validate_shared_manifest(context.validated_manifest)
        super()._load(context, rollback)
        if getattr(self.policy, "rtc_processor", None) is not None:
            raise ValueError("shared policy sessions do not support an RTC processor")
        # PI05 concurrency evidence: measured on real five-lane RTP runs and
        # recorded in the PR review discussion. Other policies run the same
        # path without a per-model concurrency audit. This
        # is not a claim that fatal CUDA-context failures are request-isolated.
        self._update_loaded_capabilities(
            stateful=False,
            resettable=False,
            thread_safe=self._max_in_flight > 1,
            max_in_flight_per_instance=self._max_in_flight,
            supports_multiple_instances=False,
            resource_domain=None,
            max_in_flight_per_resource_domain=None,
            admission_evidence=BackendAdmissionEvidence(
                overlapping_calls=True,
                output_isolation=True,
                failure_isolation=True,
                deterministic_cleanup=True,
            )
            if self._max_in_flight > 1
            else None,
        )

    def _execute(self, request, context):
        if request.metadata.get("action_method", "predict_action_chunk") != "predict_action_chunk":
            raise ValueError("shared policy sessions only support predict_action_chunk, never select_action")
        torch = self._torch
        # Correlate this model call with the frontend's request/lane log and
        # CUDA launch IDs. No profiling API is touched unless explicitly enabled.
        marker = torch.cuda.nvtx.range(f"shared_policy:{context.request_id}") if self._profile_nvtx else nullcontext()
        with marker:
            if self._max_in_flight == 1:
                return super()._execute(request, context)
            stream = getattr(self._stream_local, "stream", None)
            if stream is None:
                stream = self._stream_local.stream = torch.cuda.Stream(device=self._device)
            stream.wait_stream(torch.cuda.current_stream(self._device))
            with torch.cuda.stream(stream):
                try:
                    return super()._execute(request, context)
                finally:
                    # Inputs outlive queued kernels, including on error paths.
                    stream.synchronize()

    def _release(self):
        super()._release()
        self._stream_local = threading.local()


class FIFOAdmission:
    """Bounded FIFO admission with independently bounded active requests."""

    def __init__(self, capacity, max_active=1):
        if capacity < 1:
            raise ValueError("capacity must be at least one")
        if max_active < 1 or max_active > capacity:
            raise ValueError("max_active must be between one and capacity")
        self._condition = threading.Condition()
        self._queue = deque()
        self._active = 0
        self._closed = False
        self._capacity = capacity
        self._max_active = max_active

    @contextmanager
    def __call__(self, deadline, canceled):
        token = object()
        expires = time.monotonic() + max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds())
        with self._condition:
            if self._closed or len(self._queue) >= self._capacity:
                raise PipelineBusyError("shared inference admission is closed or full")
            self._queue.append(token)
            try:
                while True:
                    if canceled():
                        raise RequestCanceledError("shared inference canceled before model admission")
                    if self._closed:
                        raise PipelineBusyError("shared inference admission is closed")
                    remaining = expires - time.monotonic()
                    if remaining <= 0:
                        raise DeadlineExceededError("shared inference deadline expired in FIFO admission")
                    if self._active < self._max_active and self._queue[0] is token:
                        self._queue.popleft()
                        self._active += 1
                        break
                    self._condition.wait(min(remaining, 0.01))
            except BaseException:
                self._queue.remove(token)
                self._condition.notify_all()
                raise
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class LaneManagerView:
    """Non-owning lane identity/default prompt adapter; reset never touches weights."""

    def __init__(self, manager, shared_id, config):
        if getattr(manager.capabilities(shared_id), "stateful", True) is not False:
            raise ValueError("shared runtime must explicitly declare stateful=False after load")
        self._manager = manager
        self._shared_id = shared_id
        self._config = config

    def infer(self, pipeline_id, request):
        if pipeline_id != self._config.pipeline_id:
            raise ValueError("incorrect lane pipeline identity")
        request = replace(
            request,
            request_id=f"{self._config.action_server}:{request.request_id}",
            prompt=request.prompt or self._config.default_task or None,
        )
        return self._manager.infer(self._shared_id, request)

    def health(self, pipeline_id):
        return self._manager.health(self._shared_id)

    def reset(self, pipeline_id, deadline):
        # The frontend operation lock drains this lane only. Preserve the fresh
        # reset observation already delivered by ResetBenchmark/PreparePolicyEpisode.
        if datetime.now(timezone.utc) >= deadline:
            raise DeadlineExceededError("lane reset deadline expired")
        if not self.health(pipeline_id).ready:
            raise RuntimeError("shared runtime is not ready; restart the shared inference process")


def parse_lane_configs(raw):
    values = json.loads(raw)
    if not isinstance(values, list) or not values:
        raise ValueError("lane_configs_json must be a non-empty list of PipelineNodeConfig objects plus node_name")
    lanes = []
    names = set()
    endpoints = set()
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("each lane config must be an object")
        value = dict(value)
        name = value.pop("node_name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("each lane requires a unique non-empty node_name")
        names.add(name)
        config = PipelineNodeConfig(**value)
        if config.execution_mode != "monolithic" or config.scheduler_enabled:
            raise ValueError("shared lanes require legacy monolithic DDS frontends, not distributed/scheduled")
        if not math.isfinite(config.request_timeout) or config.request_timeout <= 0:
            raise ValueError("request_timeout must be finite and positive")
        for field in ("action_server", "reset_service", "health_topic", "action_topic"):
            endpoint = getattr(config, field)
            if not endpoint.startswith("/") or "//" in endpoint or endpoint.endswith("/") or endpoint in endpoints:
                raise ValueError(f"{field} must be a unique absolute ROS endpoint")
            endpoints.add(endpoint)
        if lanes:
            first = lanes[0][0]
            if (
                Path(config.model_path).resolve() != Path(first.model_path).resolve()
                or config.deployment != first.deployment
                or json.loads(config.runtime_options_json) != json.loads(first.runtime_options_json)
            ):
                raise ValueError("all lanes must use identical model_path, deployment, and runtime options")
        lanes.append((config, name))
    return lanes


class SharedPolicyHost:
    """Own exactly one manager/load and all non-owning frontends."""

    def __init__(self, lanes, dependencies, *, max_in_flight=1, queue_size=64):
        if not lanes:
            raise ValueError("lane configs must be non-empty")
        if type(max_in_flight) is not int or max_in_flight < 1:
            raise ValueError("max_in_flight must be a positive int")
        if type(queue_size) is not int or queue_size < 1:
            raise ValueError("queue_size must be a positive integer")
        self.nodes = []
        self.manager = None
        self.admission = FIFOAdmission(queue_size, max_active=max_in_flight)
        first = lanes[0][0]
        try:
            manifest = load_inference_manifest(first.model_path, first.deployment)
            validate_shared_manifest(manifest)
            self.manager = create_pipeline_manager(
                first.pipeline_id,
                manifest,
                runtime_options=json.loads(first.runtime_options_json),
                registry_set=dependencies.registry_set,
                providers=dependencies.providers,
                model_session_factory=lambda context, options: SharedPolicySession(
                    context.device, max_in_flight=max_in_flight
                ),
                max_active_executions=max_in_flight,
            )
            for config, name in lanes:
                node = PipelinePolicyNode(
                    config,
                    node_name=name,
                    registry_set=dependencies.registry_set,
                    providers=dependencies.providers,
                    shared_manager=LaneManagerView(self.manager, first.pipeline_id, config),
                    inference_admission=self.admission,
                    use_global_arguments=False,
                )
                self.nodes.append(node)
            topics = set()
            for node in self.nodes:
                lane_topics = {spec.topic for spec in node._obs_specs}
                if topics & lane_topics:
                    raise ValueError("lane robot contracts must use disjoint observation topics")
                topics.update(lane_topics)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.admission.close()
        for node in reversed(self.nodes):
            node.destroy_node()
        self.nodes.clear()
        if self.manager is not None:
            manager, self.manager = self.manager, None
            manager.close()


def main(args=None):
    rclpy.init(args=args)
    host = dependencies = executor = None
    try:
        reader = Node("benchmark_shared_policy_parameters")
        try:
            reader.declare_parameter("lane_configs_json", "")
            reader.declare_parameter("max_in_flight", 1)
            reader.declare_parameter("queue_size", 64)
            lanes = parse_lane_configs(reader.get_parameter("lane_configs_json").value)
            max_in_flight = reader.get_parameter("max_in_flight").value
            queue_size = reader.get_parameter("queue_size").value
        finally:
            reader.destroy_node()
        dependencies = build_policy_runtime_dependencies()
        host = SharedPolicyHost(lanes, dependencies, max_in_flight=max_in_flight, queue_size=queue_size)
        executor = MultiThreadedExecutor(num_threads=max(8, 2 * len(host.nodes) + 4))
        for node in host.nodes:
            executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if host is not None:
            host.admission.close()
        if executor is not None:
            executor.shutdown()
        if host is not None:
            host.close()
        if dependencies is not None:
            dependencies.providers.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
