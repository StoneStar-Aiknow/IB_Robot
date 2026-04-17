#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Action Dispatcher Node.

Maintains a queue of actions and triggers inference when low.
Publishes actions to ros2_control via TopicExecutor at a fixed frequency.

Supports cross-frame temporal smoothing for action chunks.
"""

import time
import collections
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import numpy as np
import torch
import rclpy
import rclpy.action
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Int32, Bool
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty

from ibrobot_msgs.action import DispatchInfer
from ibrobot_msgs.msg import VariantsList
from robot_config.contract_utils import iter_specs
from tensormsg.converter import TensorMsgConverter

from .topic_executor import TopicExecutor
from .temporal_smoother import (
    TemporalSmoother,
    TemporalSmootherConfig,
    TemporalSmootherManager,
)


class ActionDispatcherNode(Node):
    """
    Simplified action dispatcher.
    - Queue: collections.deque (when smoothing disabled) or TemporalSmoother (when enabled)
    - Trigger: Simple watermark check
    - Execution: TopicExecutor (100Hz streaming)

    Cross-frame smoothing can be enabled via parameters to ensure smooth
    transitions between consecutive action chunks.
    """

    def __init__(self):
        super().__init__("action_dispatcher")
        self.get_logger().info("Initializing Action Dispatcher")

        # 1. Parameters
        self.declare_parameter("queue_size", 100)
        self.declare_parameter("watermark_threshold", 20)
        self.declare_parameter("control_frequency", 100.0)
        self.declare_parameter(
            "inference_action_server", "/act_inference_node/DispatchInfer"
        )
        self.declare_parameter("robot_config_path", "")
        self.declare_parameter("joint_state_topic", "/joint_states")

        # Temporal smoothing parameters
        self.declare_parameter("temporal_smoothing_enabled", False)
        self.declare_parameter("temporal_ensemble_coeff", 0.01)
        self.declare_parameter("chunk_size", 100)
        self.declare_parameter("smoothing_device", "")

        self._queue_limit = self.get_parameter("queue_size").value
        self._watermark = self.get_parameter("watermark_threshold").value
        self._control_hz = self.get_parameter("control_frequency").value
        self._server_name = self.get_parameter("inference_action_server").value

        # Smoothing config
        self._smoothing_enabled = self.get_parameter("temporal_smoothing_enabled").value
        self._temporal_ensemble_coeff = self.get_parameter(
            "temporal_ensemble_coeff"
        ).value
        self._chunk_size = self.get_parameter("chunk_size").value
        smoothing_device = self.get_parameter("smoothing_device").value
        if smoothing_device == "":
            smoothing_device = None

        # 2. State & Queue
        self._queue = collections.deque(maxlen=self._queue_limit)
        self._last_action: Optional[np.ndarray] = None
        self._inference_in_progress = False
        self._is_running = True

        # Track actions executed during inference for temporal alignment
        self._plan_length_at_inference_start: int = 0

        # 3. Initialize Temporal Smoother (if enabled)
        self._smoother: Optional[TemporalSmootherManager] = None
        if self._smoothing_enabled:
            self._smoother = TemporalSmootherManager(
                enabled=True,
                chunk_size=self._chunk_size,
                temporal_ensemble_coeff=self._temporal_ensemble_coeff,
                device=smoothing_device,
            )
            self.get_logger().info(
                f"Temporal smoothing ENABLED: coeff={self._temporal_ensemble_coeff}, "
                f"chunk_size={self._chunk_size}"
            )
        else:
            self.get_logger().info("Temporal smoothing DISABLED (using simple queue)")

        # 4. Load Contract (Essential for TopicExecutor mapping)
        robot_config_path = self.get_parameter("robot_config_path").value
        self._action_specs = []
        if robot_config_path:
            try:
                from robot_config.loader import load_robot_config

                robot_cfg = load_robot_config(robot_config_path)
                self._contract = robot_cfg.to_contract()
                self._action_specs = [
                    s for s in iter_specs(self._contract) if s.is_action
                ]
                self.get_logger().info(
                    f"Loaded {len(self._action_specs)} action specs from robot_config"
                )
            except Exception as e:
                self.get_logger().error(
                    f"Failed to load contract from {robot_config_path}: {e}"
                )
        else:
            self.get_logger().warn(
                "No robot_config_path provided! TopicExecutor will use defaults."
            )

        # 5. Executor (Topic-based)
        self._executor = TopicExecutor(self, {"action_specs": self._action_specs})
        if not self._executor.initialize():
            raise RuntimeError("Failed to initialize TopicExecutor")

        # 6. Communication
        self._infer_client = rclpy.action.ActionClient(
            self, DispatchInfer, self._server_name
        )

        # Subscriptions
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._joint_sub = self.create_subscription(
            JointState,
            self.get_parameter("joint_state_topic").value,
            self._joint_cb,
            qos,
        )

        # Publishers
        self._queue_size_pub = self.create_publisher(Int32, "~/queue_size", 10)
        self._smoothing_enabled_pub = self.create_publisher(
            Bool, "~/smoothing_enabled", 10
        )

        # 7. Timers
        self._cb_group = MutuallyExclusiveCallbackGroup()
        self._timer = self.create_timer(
            1.0 / self._control_hz, self._control_loop, callback_group=self._cb_group
        )

        # Services
        self._reset_srv = self.create_service(Empty, "~/reset", self._reset_cb)
        self._toggle_smoothing_srv = self.create_service(
            Empty, "~/toggle_smoothing", self._toggle_smoothing_cb
        )

        self.get_logger().info(
            f"Dispatcher ready. Hz: {self._control_hz}, Watermark: {self._watermark}, "
            f"Smoothing: {'ON' if self._smoothing_enabled else 'OFF'}"
        )
        self.get_logger().info(f"Waiting for inference server: {self._server_name}")

        # Periodic stats tracking
        self._dispatch_count = 0
        self._total_inference_latency_ms = 0.0
        self._last_stats_time = time.monotonic()
        self._stats_interval_s = 5.0
        self._hold_count = 0
        self._consecutive_failures = 0
        self._last_stats_dispatch_count = 0

    def _joint_cb(self, msg):
        """Optional: could use current state for safety or initialization."""
        pass

    def _get_plan_length(self) -> int:
        """Get current plan length (works for both modes)."""
        if self._smoother is not None:
            return self._smoother.plan_length
        return len(self._queue)

    def _control_loop(self):
        if not self._is_running:
            return

        q_size = self._get_plan_length()
        self._queue_size_pub.publish(Int32(data=q_size))
        self._smoothing_enabled_pub.publish(Bool(data=self._smoothing_enabled))

        # A. Trigger Inference if queue is low
        if q_size <= self._watermark and not self._inference_in_progress:
            self._request_inference()

        # B. Get Action
        action = None
        if q_size > 0:
            if self._smoother is not None:
                action_tensor = self._smoother.get_next_action()
                if isinstance(action_tensor, torch.Tensor):
                    action = action_tensor.detach().cpu().numpy()
                else:
                    action = action_tensor
            else:
                action = self._queue.popleft()
            self._last_action = action
        elif self._last_action is not None:
            action = self._last_action
            self._hold_count += 1

        # C. Execute
        if action is not None:
            if isinstance(action, torch.Tensor):
                action_np = action.detach().cpu().numpy()
            else:
                action_np = np.array(action)
            self._executor.execute(action_np)

        # D. Periodic stats (only when new inferences arrived)
        now = time.monotonic()
        if now - self._last_stats_time >= self._stats_interval_s:
            new_inferences = self._dispatch_count - self._last_stats_dispatch_count
            if new_inferences > 0:
                avg_lat = self._total_inference_latency_ms / self._dispatch_count
                self.get_logger().info(
                    f"[stats] inferences={self._dispatch_count}, "
                    f"avg_latency={avg_lat:.1f}ms, "
                    f"queue={q_size}, hold={self._hold_count}"
                )
            self._last_stats_dispatch_count = self._dispatch_count
            self._hold_count = 0
            self._last_stats_time = now

    def _request_inference(self):
        """Send async goal to inference service."""
        if not self._infer_client.wait_for_server(timeout_sec=0.1):
            return

        self._inference_in_progress = True
        self._plan_length_at_inference_start = self._get_plan_length()

        goal = DispatchInfer.Goal()
        goal.obs_timestamp = self.get_clock().now().to_msg()

        self.get_logger().debug(
            f"Requesting inference @ {goal.obs_timestamp.sec}, "
            f"plan_length_at_start: {self._plan_length_at_inference_start}"
        )

        send_goal_future = self._infer_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Inference goal REJECTED")
            self._inference_in_progress = False
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        self._inference_in_progress = False
        result = future.result().result

        if not result.success:
            self._consecutive_failures += 1
            if self._consecutive_failures == 1:
                self.get_logger().warn(f"Inference failed: {result.message}")
            else:
                self.get_logger().debug(
                    f"Inference failed (#{self._consecutive_failures}): {result.message}"
                )
            return

        if self._consecutive_failures > 0:
            self.get_logger().info(
                f"Inference recovered (after {self._consecutive_failures} failures)"
            )
            self._consecutive_failures = 0

        self._dispatch_count += 1
        self._total_inference_latency_ms += result.inference_latency_ms

        # Decode VariantsList to Numpy/Tensor
        batch = TensorMsgConverter.from_variant(result.action_chunk)
        if "action" in batch:
            action_chunk = batch["action"]

            # Convert to Numpy if it's a Torch Tensor
            if hasattr(action_chunk, "detach"):
                action_chunk_tensor = action_chunk
                action_chunk_np = action_chunk.detach().cpu().numpy()
            else:
                action_chunk_tensor = torch.from_numpy(action_chunk)
                action_chunk_np = action_chunk

            # Reshape to (N, action_dim)
            if action_chunk_np.ndim == 1:
                action_chunk_np = action_chunk_np.reshape(1, -1)
                action_chunk_tensor = action_chunk_tensor.reshape(1, -1)

            # Calculate actions executed during inference
            current_plan_length = self._get_plan_length()
            actions_executed = max(
                0, self._plan_length_at_inference_start - current_plan_length
            )

            if self._smoother is not None:
                # Use smoother for cross-frame smoothing
                new_length = self._smoother.update(
                    action_chunk_tensor, actions_executed
                )
                self.get_logger().debug(
                    f"Smoothed update: {len(action_chunk_np)} new, "
                    f"skipped {actions_executed}, plan={new_length}"
                )
            else:
                # Simple queue mode: align and replace
                relevant_actions = action_chunk_np[actions_executed:]
                self._queue.clear()
                self._queue.extend(relevant_actions)
                self.get_logger().debug(
                    f"Queue update: {len(relevant_actions)} actions "
                    f"(skipped {actions_executed}), total={len(self._queue)}"
                )

            if self._dispatch_count == 1:
                self.get_logger().info(
                    f"✓ First inference received: "
                    f"chunk={len(action_chunk_np)}, "
                    f"latency={result.inference_latency_ms:.1f}ms, "
                    f"queue={self._get_plan_length()}"
                )

    def _reset_cb(self, request, response):
        self.get_logger().info("Resetting dispatcher state")
        self._queue.clear()
        if self._smoother is not None:
            self._smoother.reset()
        self._inference_in_progress = False
        self._plan_length_at_inference_start = 0
        self._last_action = None
        return response

    def _toggle_smoothing_cb(self, request, response):
        """Toggle smoothing on/off at runtime (requires smoother to be initialized)."""
        if self._smoother is None:
            self.get_logger().warn("Cannot toggle smoothing: smoother not initialized")
            return response

        self._smoothing_enabled = not self._smoothing_enabled
        self._smoother._config.enabled = self._smoothing_enabled
        self._smoother._smoother.config.enabled = self._smoothing_enabled

        self.get_logger().info(
            f"Temporal smoothing {'ENABLED' if self._smoothing_enabled else 'DISABLED'}"
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ActionDispatcherNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
