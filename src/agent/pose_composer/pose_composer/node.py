"""ROS node: compose arm poses from descriptions and execute them on the X2 runtime.

    ~/compose  ibrobot_msgs/srv/ComposeArmPose      the Agent, while it prepares a plan
    ~/execute  ibrobot_msgs/action/ExecuteArmPose   the Gateway's pose_composer executor

Compose asks the language model for a description of the pose (spec), solves
and checks it against the measured arm pose, plans the path, and keeps the
result. Execute takes the composition prepared for the same text (or composes
again if there is none or it expired), re-plans the path from the arm's pose
at that moment, and sends it to the runtime's joint.trajectory action, which
validates it once more. A cancel is forwarded to the runtime, which holds the
arms where they are. The runtime mode is the Gateway's: execute only checks it.
"""

from __future__ import annotations

import json
import threading
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from embodied_common import pose_composer_contract as CONTRACT
from ibrobot_msgs.action import ExecuteArmPose
from ibrobot_msgs.srv import ComposeArmPose, GetRuntimeStatus
from pose_composer import model, spec
from pose_composer.composer import ComposeError, Composer, description_json

POINT_RATE_HZ = 25.0


class PoseComposerNode(Node):
    def __init__(self, **kwargs):
        super().__init__(CONTRACT.NODE_NAME, **kwargs)
        p = self.declare_parameter
        p("mjcf_path", "")
        p("runtime_profile", "")
        p("trajectory_action", "/arm_trajectory_controller/follow_joint_trajectory")
        p("runtime_status_service", "/runtime/get_status")
        p("joint_state_topic", "/joint_states")
        p("trajectory_modes_json", '["stream"]')
        p("llm_provider", "openai_compatible")
        p("llm_base_url", "")
        p("llm_api_key_env", "")
        p("llm_model", "")
        p("llm_timeout_sec", 60.0)
        p("llm_temperature", 0.3)
        p("llm_extra_body_json", "{}")
        param = lambda name: self.get_parameter(name).value  # noqa: E731
        model.configure(param("mjcf_path"), param("runtime_profile"))
        from embodied_common.vlm_api_client import VLMAPIClient

        client = VLMAPIClient(
            provider=param("llm_provider"),
            base_url=param("llm_base_url"),
            api_key_env=param("llm_api_key_env"),
            model=param("llm_model"),
            timeout_sec=float(param("llm_timeout_sec")),
            temperature=float(param("llm_temperature")),
            multimodal=False,
        )
        extra_body = json.loads(param("llm_extra_body_json") or "{}")

        def language_model(messages):
            reply = client.complete(messages, extra_body=extra_body or None)
            if reply.get("status") != "ok":
                raise RuntimeError(reply.get("error") or "no reply")
            return reply["content"]

        self._composer = Composer(language_model, ttl_sec=CONTRACT.PREPARED_TTL_SEC)
        self._trajectory_modes = set(json.loads(param("trajectory_modes_json")))
        group = ReentrantCallbackGroup()
        self._positions: dict[str, float] = {}
        self._joint_lock = threading.Lock()
        self.create_subscription(
            JointState, param("joint_state_topic"), self._on_joints,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT), callback_group=group,
        )  # fmt: skip
        self._traj = ActionClient(self, FollowJointTrajectory, param("trajectory_action"), callback_group=group)
        self._status = self.create_client(GetRuntimeStatus, param("runtime_status_service"), callback_group=group)
        # One composition or execution at a time: both read the arm pose and the history.
        self._busy = threading.Lock()
        self.create_service(ComposeArmPose, "~/compose", self._on_compose, callback_group=group)
        self._server = ActionServer(
            self, ExecuteArmPose, "~/execute", execute_callback=self._on_execute,
            goal_callback=lambda _goal: GoalResponse.ACCEPT,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT, callback_group=group,
        )  # fmt: skip
        self.get_logger().info(f"pose composer ready ({spec.PROMPT_VERSION}, model {param('llm_model')})")

    # --- inputs -------------------------------------------------------------

    def _on_joints(self, msg: JointState) -> None:
        with self._joint_lock:
            self._positions.update(dict(zip(msg.name, msg.position, strict=False)))

    def _measured(self) -> dict[str, float]:
        with self._joint_lock:
            if not all(n in self._positions for n in model.ARM_JOINTS):
                raise ComposeError("POSE_NO_JOINT_STATE", "no arm joint state yet")
            return {n: float(self._positions[n]) for n in model.ARM_JOINTS}

    # --- compose ------------------------------------------------------------

    def _on_compose(self, request, response):
        if not self._busy.acquire(blocking=False):
            response.success, response.error_code = False, "POSE_BUSY"
            response.message = "another pose is being composed or executed"
            return response
        try:
            start = self._measured()
            prepared = self._composer.compose(request.text, start)
            composition = prepared.composition
            response.kind = composition.kind
            response.user_message = composition.user_message
            response.composition_id = prepared.composition_id
            response.description_json = description_json(prepared)
            response.approximate = bool(composition.approx)
            if composition.kind in ("pose", "sequence"):
                plan = spec.plan(composition, start)
                if not plan.ok:
                    self._composer.take(prepared.text, prepared.composition_id)
                    raise ComposeError("POSE_NO_SAFE_PATH", "no collision-free path to the pose")
                response.duration_sec = float(plan.traj.times[-1])
                response.message = f"composed {composition.kind} ({len(composition.frames)} frame(s))"
            else:
                response.message = composition.reason or composition.kind
            response.success = True
        except ComposeError as exc:
            response.success, response.error_code, response.message = False, exc.code, exc.message
        finally:
            self._busy.release()
        return response

    # --- execute ------------------------------------------------------------

    def _on_execute(self, goal_handle):
        result = ExecuteArmPose.Result()
        if not self._busy.acquire(blocking=False):
            return self._fail(goal_handle, result, "POSE_BUSY", "another pose is being composed or executed")
        try:
            return self._execute_locked(goal_handle, result)
        except ComposeError as exc:
            return self._fail(goal_handle, result, exc.code, exc.message)
        finally:
            self._busy.release()

    def _feedback(self, goal_handle, state: str, progress: float) -> None:
        feedback = ExecuteArmPose.Feedback()
        feedback.state, feedback.progress = state, float(progress)
        goal_handle.publish_feedback(feedback)

    def _execute_locked(self, goal_handle, result):
        request = goal_handle.request
        mode = self._runtime_mode()
        if mode not in self._trajectory_modes:
            return self._fail(
                goal_handle, result, "POSE_RUNTIME_UNAVAILABLE",
                f"runtime mode is {mode!r}; arm poses run in {sorted(self._trajectory_modes)}",
            )  # fmt: skip
        prepared = self._composer.take(request.text, request.composition_id)
        result.used_prepared = prepared is not None
        start = self._measured()
        if prepared is None:
            self._feedback(goal_handle, "composing", 0.0)
            prepared = self._composer.compose(request.text, start)
            self._composer.take(prepared.text, prepared.composition_id)
        composition = prepared.composition
        if composition.kind not in ("pose", "sequence"):
            return self._fail(goal_handle, result, "POSE_NOT_EXECUTABLE", composition.user_message or composition.kind)
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.error_code, result.message = "POSE_CANCELLED", "cancelled before moving"
            return result
        self._feedback(goal_handle, "planning", 0.0)
        plan = spec.plan(composition, start)
        if not plan.ok:
            return self._fail(goal_handle, result, "POSE_NO_SAFE_PATH", "no collision-free path from the current pose")
        ok, code, message = self._run_trajectory(goal_handle, plan)
        note = "" if result.used_prepared else " (composed again: the prepared composition was missing or expired)"
        if code == "POSE_CANCELLED":
            goal_handle.canceled()
            result.error_code, result.message = code, message
            return result
        if not ok:
            return self._fail(goal_handle, result, code, message + note)
        self._composer.remember(prepared.text, composition.frames[-1])
        self.get_logger().info(
            f"pose {prepared.composition_id} reached ({'prepared with the plan' if result.used_prepared else 'composed at execution'}): "
            f"{prepared.text}"
        )
        goal_handle.succeed()
        result.success = True
        result.message = f"{composition.user_message or 'pose reached'}; {message}{note}"
        return result

    def _fail(self, goal_handle, result, code: str, message: str):
        goal_handle.abort()
        result.success, result.error_code, result.message = False, code, message
        return result

    def _runtime_mode(self) -> str:
        if not self._status.wait_for_service(timeout_sec=3.0):
            raise ComposeError("POSE_RUNTIME_UNAVAILABLE", "runtime status service unavailable")
        future = self._status.call_async(GetRuntimeStatus.Request())
        deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done() or future.result() is None:
            raise ComposeError("POSE_RUNTIME_UNAVAILABLE", "runtime status did not answer")
        return str(future.result().status.active_mode)

    def _run_trajectory(self, goal_handle, plan) -> tuple[bool, str, str]:
        if not self._traj.wait_for_server(timeout_sec=5.0):
            return False, "POSE_RUNTIME_UNAVAILABLE", "runtime trajectory action unavailable"
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(model.ARM_JOINTS)
        times, positions = plan.traj.times, plan.traj.positions
        step = max(1, int(round(len(times) / max(1.0, times[-1] * POINT_RATE_HZ))))
        for k in [*range(step, len(times) - 1, step), len(times) - 1]:
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in positions[k]]
            sec = float(times[k])
            point.time_from_start = Duration(sec=int(sec), nanosec=int((sec - int(sec)) * 1e9))
            goal.trajectory.points.append(point)
        future = self._traj.send_goal_async(goal)
        while not future.done():
            time.sleep(0.02)
        handle = future.result()
        if not handle.accepted:
            return False, "POSE_RUNTIME_REJECTED", "runtime rejected the trajectory"
        result_future = handle.get_result_async()
        started, total = time.monotonic(), float(times[-1])
        cancel_sent = False
        while not result_future.done():
            if goal_handle.is_cancel_requested and not cancel_sent:
                handle.cancel_goal_async()
                cancel_sent = True
            self._feedback(goal_handle, "moving", min((time.monotonic() - started) / total, 1.0))
            time.sleep(0.1)
        wrapped = result_future.result()
        outcome = wrapped.result
        if cancel_sent:
            return False, "POSE_CANCELLED", f"stopped; arms held where they were ({outcome.error_string})"
        if outcome.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            return False, "POSE_RUNTIME_REJECTED", outcome.error_string or "runtime did not complete the trajectory"
        return True, "", f"moved in {total:.1f}s"


def main(argv=None) -> int:
    rclpy.init(args=argv)
    node = PoseComposerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0
