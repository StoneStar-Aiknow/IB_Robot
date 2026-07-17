"""ROS 2 node exposing one validated unified inference pipeline."""

from __future__ import annotations

import json
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
import rclpy.action
import torch
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger

from ibrobot_msgs.action import DispatchInfer
from ibrobot_msgs.msg import (
    DistributedInferenceRequest,
    DistributedInferenceResult,
    InferencePipelineStatus,
    VariantsList,
)
from inference_manifest import ValidatedManifest, load_inference_manifest, load_inference_manifest_metadata
from inference_service.backends import InferenceRequest
from inference_service.distributed import (
    DistributedProtocolError,
    DistributedResult,
    EdgeProcessorRuntime,
    EdgeSession,
    Operation,
    StructuredError,
    build_pipeline_identity,
    structured_error_from_exception,
)
from inference_service.distributed.ros_protocol import (
    error_to_message,
    request_to_message,
    result_from_message,
    status_from_message,
    status_to_message,
)
from inference_service.pipeline import InferencePipelineManager, create_pipeline_manager
from robot_config.contract_utils import (
    SpecView,
    StreamBuffer,
    decode_value,
    feature_from_spec,
    iter_specs,
    qos_profile_from_dict,
    stamp_from_header_ns,
    zero_pad,
)
from robot_config.utils import (
    build_joint_conversion_table,
    build_joint_conversion_table_from_urdf,
    parse_bool,
    resolve_calibration_source_specs_from_config,
    resolve_gripper_joints_from_config,
    resolve_joint_names_from_config,
    resolve_lerobot_norm_mode,
)
from tensormsg.converter import TensorMsgConverter


@dataclass(frozen=True)
class PipelineNodeConfig:
    pipeline_id: str
    model_path: str
    deployment: str
    execution_mode: str
    request_timeout: float
    default_task: str
    runtime_options_json: str
    robot_config_path: str
    use_sim: bool
    action_server: str
    reset_service: str
    health_topic: str
    action_topic: str
    request_topic: str
    result_topic: str
    heartbeat_topic: str


@dataclass
class _SubState:
    spec: SpecView
    buffer: StreamBuffer


@dataclass
class _PendingOperation:
    event: threading.Event
    operation: Operation
    session_id: str = ""
    session_generation: int = 0
    deployment_fingerprint: str = ""
    result: DistributedResult | None = None
    error: StructuredError | None = None


class PipelinePolicyNode(Node):
    """Own exactly one pipeline process and its pipeline-scoped ROS interfaces."""

    def __init__(self, config: PipelineNodeConfig, *, node_name: str) -> None:
        super().__init__(node_name)
        if config.execution_mode not in {"monolithic", "distributed"}:
            raise RuntimeError(f"unsupported pipeline execution mode {config.execution_mode!r}")

        self._config = config
        self._manager: InferencePipelineManager | None = None
        self._edge_runtime: EdgeProcessorRuntime | None = None
        self._edge_session: EdgeSession | None = None
        manifest_loader = (
            load_inference_manifest if config.execution_mode == "monolithic" else load_inference_manifest_metadata
        )
        self._manifest: ValidatedManifest = manifest_loader(config.model_path, config.deployment)
        self._contract = None
        self._frequency = 10.0
        self._obs_specs: list[SpecView] = []
        self._state_specs: list[SpecView] = []
        self._obs_zero: dict[str, np.ndarray] = {}
        self._subs: dict[str, _SubState] = {}
        self._joint_rad_limits: list[tuple[float, float, float, float]] = []
        self._last_inference_time: float | None = None
        self._inference_count = 0
        self._last_error = ""
        self._remote_state = "unavailable"
        self._pending_lock = threading.RLock()
        self._pending: dict[str, _PendingOperation] = {}
        self._goal_request_ids: dict[int, str] = {}

        self._load_contract(config.robot_config_path)
        self._setup_observation_subscriptions()
        try:
            runtime_options = json.loads(config.runtime_options_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"pipeline {config.pipeline_id!r} runtime_options_json is invalid: {exc}") from exc
        if not isinstance(runtime_options, dict):
            raise RuntimeError(f"pipeline {config.pipeline_id!r} runtime_options_json must decode to an object")
        if config.execution_mode == "monolithic":
            self._manager = create_pipeline_manager(
                config.pipeline_id,
                self._manifest,
                request_timeout=config.request_timeout,
                default_task=config.default_task or None,
                runtime_options=runtime_options,
            )
        else:
            for name in ("request_topic", "result_topic", "heartbeat_topic"):
                if not getattr(config, name):
                    raise RuntimeError(f"distributed pipeline requires {name}")
            self._edge_runtime = EdgeProcessorRuntime(
                config.pipeline_id,
                self._manifest,
                default_task=config.default_task or None,
            )
            self._edge_runtime.load()
            self._edge_session = EdgeSession(build_pipeline_identity(config.pipeline_id, self._manifest))
            self._edge_session.start()

        self._action_pub = self.create_publisher(VariantsList, config.action_topic, 10)
        self._health_pub = self.create_publisher(DiagnosticStatus, config.health_topic, 10)
        if config.execution_mode == "distributed":
            status_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._request_pub = self.create_publisher(DistributedInferenceRequest, config.request_topic, 10)
            self.create_subscription(
                DistributedInferenceResult,
                config.result_topic,
                self._distributed_result_callback,
                10,
                callback_group=ReentrantCallbackGroup(),
            )
            self._status_pub = self.create_publisher(InferencePipelineStatus, config.heartbeat_topic, status_qos)
            self.create_subscription(
                InferencePipelineStatus,
                config.heartbeat_topic,
                self._cloud_status_callback,
                status_qos,
                callback_group=ReentrantCallbackGroup(),
            )
            self._status_timer = self.create_timer(0.5, self._publish_distributed_status)
            self._heartbeat_timer = self.create_timer(0.25, self._check_heartbeat)
        self._action_server = rclpy.action.ActionServer(
            self,
            DispatchInfer,
            config.action_server,
            execute_callback=self._dispatch_infer_callback,
            goal_callback=lambda _request: rclpy.action.GoalResponse.ACCEPT,
            cancel_callback=self._cancel_callback,
            callback_group=(
                MutuallyExclusiveCallbackGroup() if config.execution_mode == "monolithic" else ReentrantCallbackGroup()
            ),
        )
        self._reset_server = self.create_service(
            Trigger,
            config.reset_service,
            self._reset_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self._health_timer = self.create_timer(1.0, self._publish_health)
        self.get_logger().info(
            f"Unified pipeline started: id={config.pipeline_id}, mode={config.execution_mode}, "
            f"bundle={self._manifest.manifest.bundle.name}, "
            f"deployment={config.deployment}, backend={self._manifest.deployment.backend}, "
            f"action={config.action_server}, reset={config.reset_service}"
        )

    def _load_contract(self, robot_config_path: str) -> None:
        config_path = Path(robot_config_path)
        if not config_path.is_file():
            raise RuntimeError(f"robot config file not found: {robot_config_path}")

        from robot_config.loader import build_contract_from_robot_config_dict, load_robot_config_dict

        robot_config = load_robot_config_dict(config_path)
        self._contract = build_contract_from_robot_config_dict(robot_config)
        self._frequency = float(self._contract.rate_hz)

        required_inputs = set(self._manifest.policy.input_features)
        all_observations = [spec for spec in iter_specs(self._contract) if not spec.is_action]
        self._obs_specs = [spec for spec in all_observations if spec.key in required_inputs]
        missing_inputs = sorted(required_inputs - {spec.key for spec in self._obs_specs})
        if missing_inputs:
            raise RuntimeError(
                f"pipeline {self._config.pipeline_id!r} robot contract does not provide required observations: "
                f"{missing_inputs}"
            )
        self._state_specs = [spec for spec in self._obs_specs if spec.key == "observation.state"]
        self._topic_to_qos = {observation.topic: observation.qos for observation in self._contract.observations}

        calibration_sources = resolve_calibration_source_specs_from_config(robot_config)
        joint_names = resolve_joint_names_from_config(robot_config)
        gripper_joints = resolve_gripper_joints_from_config(robot_config) or ["6"]
        norm_mode = resolve_lerobot_norm_mode(robot_config)
        if parse_bool(self._config.use_sim, default=False) and joint_names:
            from robot_config.launch_builders.description import generate_robot_description

            description = generate_robot_description(robot_config, True)
            if description is None:
                raise RuntimeError("failed to generate robot_description for simulated joint conversion")
            robot_description, _ = description
            self._joint_rad_limits = build_joint_conversion_table_from_urdf(
                robot_description,
                joint_names,
                gripper_joints,
                norm_mode=norm_mode,
            )
        elif calibration_sources and joint_names:
            self._joint_rad_limits = build_joint_conversion_table(
                calibration_sources,
                joint_names,
                gripper_joints,
                norm_mode=norm_mode,
            )

        velocity_joints = robot_config.get("ros2_control", {}).get("velocity_joints", [])
        base_velocity_max = robot_config.get("ros2_control", {}).get("base_vel_max_rad", 0)
        if velocity_joints and base_velocity_max > 0:
            self._joint_rad_limits.extend(
                (-float(base_velocity_max), float(base_velocity_max), 200.0, -100.0) for _ in velocity_joints
            )

    def _setup_observation_subscriptions(self) -> None:
        from rosidl_runtime_py.utilities import get_message

        for spec in self._obs_specs:
            _, metadata, _ = feature_from_spec(spec, use_videos=False)
            key = self._subscription_key(spec)
            self._obs_zero[key] = zero_pad(metadata)
            message_type = get_message(spec.ros_type)
            qos = qos_profile_from_dict(self._topic_to_qos.get(spec.topic, {})) or QoSProfile(depth=10)
            self.create_subscription(
                message_type,
                spec.topic,
                lambda message, current_spec=spec: self._observation_callback(message, current_spec),
                qos,
                callback_group=ReentrantCallbackGroup(),
            )
            self._subs[key] = _SubState(
                spec=spec,
                buffer=StreamBuffer(
                    policy=getattr(spec, "resample_policy", "hold"),
                    step_ns=int(1e9 / self._frequency),
                    tol_ns=int(max(0, getattr(spec, "asof_tol_ms", 0)) * 1_000_000),
                ),
            )

    def _subscription_key(self, spec: SpecView) -> str:
        if spec.key == "observation.state" and len(self._state_specs) > 1:
            return f"{spec.key}_{spec.topic.replace('/', '_')}"
        return spec.key

    def _observation_callback(self, message: object, spec: SpecView) -> None:
        receive_time = self.get_clock().now().nanoseconds
        timestamp = stamp_from_header_ns(message) if spec.stamp_src == "header" else None
        value = decode_value(spec.ros_type, message, spec)
        if value is not None:
            self._subs[self._subscription_key(spec)].buffer.push(int(timestamp or receive_time), value)

    def _sample_observations(self, sample_time_ns: int) -> dict[str, Any]:
        observations: dict[str, Any] = {}
        if len(self._state_specs) > 1:
            state_parts = []
            for spec in self._state_specs:
                key = self._subscription_key(spec)
                value = self._subs[key].buffer.sample(sample_time_ns)
                state_parts.append(value if value is not None else self._obs_zero[key])
            observations["observation.state"] = np.concatenate(state_parts)

        for key, state in self._subs.items():
            if key.startswith("observation.state_") and len(self._state_specs) > 1:
                continue
            value = state.buffer.sample(sample_time_ns)
            observations[key] = value if value is not None else self._obs_zero[key]
        return observations

    def _rad_to_lerobot(self, state: np.ndarray) -> np.ndarray:
        if not self._joint_rad_limits:
            return np.ascontiguousarray(state, dtype=np.float32)
        converted = state.astype(np.float64).copy()
        for index, (minimum, maximum, span, offset) in enumerate(self._joint_rad_limits):
            if index < len(state):
                converted[index] = (state[index] - minimum) / (maximum - minimum) * span + offset
        return converted.astype(np.float32)

    def _lerobot_to_rad(self, action: object) -> np.ndarray:
        candidate = action
        detach = getattr(candidate, "detach", None)
        if callable(detach):
            candidate = detach()
        cpu = getattr(candidate, "cpu", None)
        if callable(cpu):
            candidate = cpu()
        converted = np.asarray(candidate).astype(np.float64).copy()
        if not self._joint_rad_limits:
            return converted
        for index, (minimum, maximum, span, offset) in enumerate(self._joint_rad_limits):
            if index < converted.shape[-1]:
                converted[..., index] = (converted[..., index] - offset) / span * (maximum - minimum) + minimum
        return converted.astype(np.float32)

    @staticmethod
    def _to_policy_inputs(observations: dict[str, Any]) -> dict[str, Any]:
        return {
            key: torch.from_numpy(np.ascontiguousarray(value)) if isinstance(value, np.ndarray) else value
            for key, value in observations.items()
        }

    def _dispatch_infer_callback(self, goal_handle: object) -> DispatchInfer.Result:
        request = goal_handle.request
        request_id = request.inference_id or f"{self._config.pipeline_id}-{time.monotonic_ns()}"
        self._goal_request_ids[id(goal_handle)] = request_id
        sample_time = request.obs_timestamp.sec * 1_000_000_000 + request.obs_timestamp.nanosec
        if sample_time <= 0:
            sample_time = self.get_clock().now().nanoseconds
        total_start = time.perf_counter()

        try:
            observations = self._sample_observations(sample_time)
            if "observation.state" in observations:
                observations["observation.state"] = self._rad_to_lerobot(observations["observation.state"])
            observations = self._to_policy_inputs(observations)
            deadline = self._goal_deadline(request.deadline)
            if self._config.execution_mode == "monolithic":
                result = self._require_manager().infer(
                    self._config.pipeline_id,
                    InferenceRequest(
                        request_id=request_id,
                        inputs=observations,
                        prompt=request.prompt if request.prompt else None,
                        deadline=deadline,
                    ),
                )
                raw_action = result.action
                chunk_size = result.actual_chunk_size
                backend_latency_ms = result.backend_latency_ms
                total_latency_ms = result.total_latency_ms
            else:
                edge_runtime = self._require_edge_runtime()
                canonical_inputs = edge_runtime.preprocess(
                    observations,
                    prompt=request.prompt if request.prompt else None,
                )
                distributed_result = self._round_trip(
                    Operation.INFER,
                    request_id,
                    inputs=dict(canonical_inputs),
                    prompt=request.prompt if request.prompt else None,
                    deadline=deadline or datetime.now(timezone.utc) + timedelta(seconds=self._config.request_timeout),
                    goal_handle=goal_handle,
                )
                raw_action = edge_runtime.postprocess(
                    distributed_result.action,
                    actual_chunk_size=distributed_result.actual_chunk_size,
                )
                chunk_size = distributed_result.actual_chunk_size
                backend_latency_ms = distributed_result.backend_latency_ms
                total_latency_ms = (time.perf_counter() - total_start) * 1000.0

            action = self._lerobot_to_rad(raw_action)
            action_message = TensorMsgConverter.to_variant({"action": action})
            self._action_pub.publish(action_message)

            response = DispatchInfer.Result()
            response.action_chunk = action_message
            response.chunk_size = chunk_size
            response.success = True
            response.message = "OK"
            response.inference_latency_ms = total_latency_ms
            response.pipeline_id = self._config.pipeline_id
            response.request_id = request_id
            response.deployment_fingerprint = self._manifest.fingerprint
            response.backend_latency_ms = backend_latency_ms
            response.error = error_to_message(None)
            goal_handle.succeed()
            self._last_inference_time = time.time()
            self._inference_count += 1
            self._last_error = ""
            return response
        except Exception as exc:
            self._last_error = str(exc)
            self.get_logger().error(f"pipeline inference failed: {exc}\n{traceback.format_exc()}")
            error = (
                exc.error
                if isinstance(exc, DistributedProtocolError)
                else structured_error_from_exception(exc, "pipeline")
            )
            response = DispatchInfer.Result()
            response.action_chunk = VariantsList()
            response.chunk_size = 0
            response.success = False
            response.message = str(exc)
            response.inference_latency_ms = (time.perf_counter() - total_start) * 1000.0
            response.pipeline_id = self._config.pipeline_id
            response.request_id = request_id
            response.deployment_fingerprint = self._manifest.fingerprint
            response.backend_latency_ms = 0.0
            response.error = error_to_message(error)
            if error.code == "request_canceled" and goal_handle.is_cancel_requested:
                goal_handle.canceled()
            else:
                goal_handle.abort()
            return response
        finally:
            self._goal_request_ids.pop(id(goal_handle), None)

    def _reset_callback(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        try:
            if self._config.execution_mode == "monolithic":
                self._require_manager().reset(self._config.pipeline_id)
            else:
                request_id = f"{self._config.pipeline_id}-reset-{time.monotonic_ns()}"
                self._round_trip(Operation.RESET, request_id)
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self._last_error = str(exc)
        else:
            response.success = True
            response.message = f"pipeline {self._config.pipeline_id} reset"
            self._last_error = ""
        return response

    def _publish_health(self) -> None:
        try:
            if self._config.execution_mode == "monolithic":
                diagnostics = self._require_manager().health(self._config.pipeline_id)
                backend_health = diagnostics.backend_health
                level = DiagnosticStatus.OK if diagnostics.ready else DiagnosticStatus.WARN
                message = backend_health.message or diagnostics.state.value
                values = {
                    **diagnostics.metadata,
                    "backend_state": backend_health.state.value,
                    "backend_reason": backend_health.reason_code or "",
                    "backend_failure_count": backend_health.failure_count,
                }
            else:
                session = self._require_edge_session()
                level = DiagnosticStatus.OK if session.ready else DiagnosticStatus.WARN
                message = self._remote_state if session.ready else session.state.value
                session_id, generation = session.session
                values = {
                    "pipeline_id": self._config.pipeline_id,
                    "bundle": self._manifest.manifest.bundle.name,
                    "deployment": self._config.deployment,
                    "deployment_fingerprint": self._manifest.fingerprint,
                    "backend": self._manifest.deployment.backend,
                    "state": session.state.value,
                    "remote_state": self._remote_state,
                    "session_id": session_id,
                    "session_generation": generation,
                }
            values.update(
                {
                    "inference_count": self._inference_count,
                    "last_inference_time": self._last_inference_time or "",
                }
            )
        except Exception as exc:
            level = DiagnosticStatus.ERROR
            message = str(exc)
            values = {"pipeline_id": self._config.pipeline_id, "state": "unavailable"}

        health = DiagnosticStatus()
        health.level = level
        health.name = f"inference/{self._config.pipeline_id}"
        health.message = self._last_error or message
        health.hardware_id = self._manifest.fingerprint
        health.values = [KeyValue(key=str(key), value=str(value)) for key, value in values.items()]
        self._health_pub.publish(health)

    def _cloud_status_callback(self, message: InferencePipelineStatus) -> None:
        if message.role != InferencePipelineStatus.ROLE_CLOUD:
            return
        try:
            status = status_from_message(message)
            update = self._require_edge_session().observe_cloud(status)
            self._remote_state = status.runtime_state
            self._complete_invalidated(update.invalidated_request_ids, update.error)
        except Exception as exc:
            self._last_error = f"invalid cloud status: {exc}"
            self.get_logger().error(self._last_error)

    def _publish_distributed_status(self) -> None:
        try:
            message = status_to_message(
                self._require_edge_session().local_status(),
                stamp=self.get_clock().now().to_msg(),
            )
            self._status_pub.publish(message)
        except Exception as exc:
            self._last_error = f"failed to publish distributed status: {exc}"

    def _check_heartbeat(self) -> None:
        update = self._require_edge_session().expire_heartbeat()
        self._complete_invalidated(update.invalidated_request_ids, update.error)

    def _distributed_result_callback(self, message: DistributedInferenceResult) -> None:
        try:
            result = result_from_message(message)
            update = self._require_edge_session().accept_result(result)
        except Exception as exc:
            with self._pending_lock:
                pending = self._pending.get(message.request_id)
                matches_pending = pending is not None and self._matches_pending_envelope(message, pending)
            if not matches_pending:
                self.get_logger().warning(
                    f"discarded malformed stale or mismatched response {message.request_id!r}: {exc}"
                )
                return
            error = StructuredError(
                code="decode_failed",
                message=str(exc) or type(exc).__name__,
                stage="decode",
            )
            with self._pending_lock:
                pending = self._pending.get(message.request_id)
                if pending is not None:
                    pending.error = error
                    pending.event.set()
            self.get_logger().error(f"invalid distributed result: {exc}")
            return
        if update.error is not None and update.error.code == "stale_response":
            self.get_logger().warning(update.error.message)
            return
        if update.error is not None and result.request_id not in self._pending:
            self.get_logger().warning(update.error.message)
            return
        with self._pending_lock:
            pending = self._pending.get(result.request_id)
            if pending is not None:
                pending.result = result
                pending.error = result.error
                pending.event.set()
            if update.canceled_request_id:
                canceled = self._pending.get(update.canceled_request_id)
                if canceled is not None:
                    canceled.error = StructuredError(
                        code="request_canceled",
                        message=f"distributed request {update.canceled_request_id!r} was canceled",
                        stage="cancel",
                    )
                    canceled.event.set()
        self._complete_invalidated(update.invalidated_request_ids, update.error)

    def _round_trip(
        self,
        operation: Operation,
        request_id: str,
        *,
        inputs: dict[str, object] | None = None,
        prompt: str | None = None,
        deadline: datetime | None = None,
        target_request_id: str = "",
        goal_handle: object | None = None,
    ) -> DistributedResult:
        pending = _PendingOperation(event=threading.Event(), operation=operation)
        with self._pending_lock:
            if request_id in self._pending:
                raise RuntimeError(f"duplicate pending distributed request {request_id!r}")
            self._pending[request_id] = pending

        cancel_sent = False
        try:

            def publish_request(current) -> None:
                with self._pending_lock:
                    pending.session_id = current.session_id
                    pending.session_generation = current.session_generation
                    pending.deployment_fingerprint = current.deployment_fingerprint
                self._request_pub.publish(request_to_message(current))

            request = self._require_edge_session().dispatch_request(
                operation,
                request_id,
                publish_request,
                inputs=inputs,
                prompt=prompt,
                deadline=deadline,
                target_request_id=target_request_id,
            )
            while not pending.event.wait(timeout=0.05):
                if request.deadline is not None and datetime.now(timezone.utc) >= request.deadline:
                    raise DistributedProtocolError(
                        StructuredError(
                            code="deadline_exceeded",
                            message=f"distributed request {request.request_id!r} timed out",
                            stage="transport",
                        )
                    )
                if goal_handle is not None and goal_handle.is_cancel_requested and not cancel_sent:
                    cancel_sent = True
                    self._route_cancel(request.request_id)
            if pending.error is not None:
                raise DistributedProtocolError(pending.error)
            if pending.result is None:
                raise RuntimeError(f"distributed request {request.request_id!r} completed without a result")
            if not pending.result.success:
                assert pending.result.error is not None
                raise DistributedProtocolError(pending.result.error)
            return pending.result
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            self._require_edge_session().abandon_request(request_id)

    def _matches_pending_envelope(
        self,
        message: DistributedInferenceResult,
        pending: _PendingOperation,
    ) -> bool:
        return (
            message.pipeline_id == self._config.pipeline_id
            and message.session_id == pending.session_id
            and message.session_generation == pending.session_generation
            and message.deployment_fingerprint == pending.deployment_fingerprint
            and message.operation in {int(pending.operation), int(Operation.UNKNOWN)}
        )

    def _route_cancel(self, target_request_id: str) -> None:
        session = self._require_edge_session()
        if not session.cancellation_supported:
            raise DistributedProtocolError(
                StructuredError(
                    code="unsupported_capability",
                    message="cloud backend does not support cancellation",
                    stage="cancel",
                )
            )
        request_id = f"{target_request_id}-cancel-{time.monotonic_ns()}"
        self._round_trip(
            Operation.CANCEL,
            request_id,
            target_request_id=target_request_id,
        )

    def _cancel_callback(self, _goal_handle: object):
        if self._config.execution_mode == "distributed" and self._require_edge_session().cancellation_supported:
            return rclpy.action.CancelResponse.ACCEPT
        return rclpy.action.CancelResponse.REJECT

    def _complete_invalidated(
        self,
        request_ids: tuple[str, ...],
        error: StructuredError | None,
    ) -> None:
        if not request_ids:
            return
        failure = error or StructuredError(
            code="session_invalidated",
            message="distributed session was invalidated",
            stage="transport",
            recoverable=True,
        )
        with self._pending_lock:
            for request_id in request_ids:
                pending = self._pending.get(request_id)
                if pending is None:
                    continue
                pending.error = failure
                pending.event.set()

    def _goal_deadline(self, deadline_message: object) -> datetime | None:
        seconds = int(deadline_message.sec)
        nanoseconds = int(deadline_message.nanosec)
        if seconds == 0 and nanoseconds == 0:
            return None
        return datetime.fromtimestamp(seconds + nanoseconds / 1_000_000_000, tz=timezone.utc)

    def _require_manager(self) -> InferencePipelineManager:
        if self._manager is None:
            raise RuntimeError(f"pipeline {self._config.pipeline_id!r} manager is closed")
        return self._manager

    def _require_edge_runtime(self) -> EdgeProcessorRuntime:
        if self._edge_runtime is None:
            raise RuntimeError(f"pipeline {self._config.pipeline_id!r} edge runtime is closed")
        return self._edge_runtime

    def _require_edge_session(self) -> EdgeSession:
        if self._edge_session is None:
            raise RuntimeError(f"pipeline {self._config.pipeline_id!r} distributed session is closed")
        return self._edge_session

    def destroy_node(self) -> None:
        manager = self._manager
        self._manager = None
        if manager is not None:
            try:
                manager.close()
            except Exception as exc:
                self.get_logger().error(f"pipeline shutdown failed: {exc}")
        edge_session = self._edge_session
        self._edge_session = None
        if edge_session is not None:
            pending = edge_session.close()
            self._complete_invalidated(pending, None)
        edge_runtime = self._edge_runtime
        self._edge_runtime = None
        if edge_runtime is not None:
            edge_runtime.close()
        super().destroy_node()


def _read_config() -> tuple[PipelineNodeConfig, str]:
    reader = Node("_pipeline_policy_param_reader")
    defaults: dict[str, object] = {
        "pipeline_id": "policy",
        "model_path": "",
        "deployment": "cpu",
        "execution_mode": "monolithic",
        "request_timeout": 5.0,
        "default_task": "",
        "runtime_options_json": "{}",
        "robot_config_path": "",
        "use_sim": False,
        "node_name": "inference_policy",
        "action_server": "/inference/policy/dispatch",
        "reset_service": "/inference/policy/reset",
        "health_topic": "/inference/policy/health",
        "action_topic": "/actions/policy",
        "request_topic": "",
        "result_topic": "",
        "heartbeat_topic": "",
    }
    for name, default in defaults.items():
        reader.declare_parameter(name, default)
    values = {name: reader.get_parameter(name).value for name in defaults}
    reader.destroy_node()
    node_name = str(values.pop("node_name"))
    return PipelineNodeConfig(**values), node_name


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: PipelinePolicyNode | None = None
    try:
        config, node_name = _read_config()
        node = PipelinePolicyNode(config, node_name=node_name)
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
