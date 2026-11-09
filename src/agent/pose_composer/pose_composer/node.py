"""ROS node: compose arm poses from descriptions and execute them on the X2 runtime.

    ~/compose  ibrobot_msgs/srv/ComposeArmPose      the Agent, while it prepares a plan
    ~/lookup   ibrobot_msgs/srv/LookupArmPose       the Agent, before it plans: a confirmed remembered pose?
    ~/feedback ibrobot_msgs/srv/ArmPoseFeedback     the Agent: the user's verdict on the last pose
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
import math
import threading
import time
from contextlib import nullcontext

import numpy as np
import rclpy
from aimdk_grasp.fixed_pose_backend import START_TOLERANCE_RAD
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from embodied_common import pose_composer_contract as CONTRACT
from embodied_common.tracing import create_trace_logger, trace_scope, trace_stage
from ibrobot_msgs.action import ExecuteArmPose
from ibrobot_msgs.srv import ArmPoseFeedback, ComposeArmPose, GetRuntimeStatus, LookupArmPose
from pose_composer.errors import ComposeError
from pose_composer.execution_backend import backend_for
from pose_composer.memory import PoseMemory

POINT_RATE_HZ = 25.0
FIXED_START_TOLERANCE_RAD = START_TOLERANCE_RAD

_trace = create_trace_logger("ib_trace.pose_composer")


class PoseComposerNode(Node):
    def __init__(self, **kwargs):
        super().__init__(CONTRACT.NODE_NAME, **kwargs)
        p = self.declare_parameter
        p("fixed_library", "", descriptor=ParameterDescriptor(read_only=True))
        p("fixed_library_sha256", "", descriptor=ParameterDescriptor(read_only=True))
        p("mjcf_path", "")
        p("runtime_profile", "")
        p("arm_limits_json", "null")
        startup_only = ParameterDescriptor(read_only=True)
        p("lookup_enabled", True, descriptor=startup_only)
        p("task_poses_json", "{}", descriptor=startup_only)
        p("path_validation_json", "{}", descriptor=startup_only)
        p("display_completion_json", "{}", descriptor=startup_only)
        p("task_pose_simulation", False, descriptor=startup_only)
        # Runtime description, grasp_execution and Gateway status service for poses that open hands; "{}" disables.
        p("hand_restore_json", "{}", descriptor=startup_only)
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
        # The pose memory (shared by every conversation, kept across restarts); "" disables it.
        p("memory_path", "")
        p("memory_max_poses", 500)
        p("memory_candidate_ttl_days", 7.0)
        # Seed poses (pinned); "" loads none.
        p("memory_seeds_path", "")
        p("max_joint_delta_rad", 0.0)
        p("max_hand_displacement_m", 0.0)
        param = lambda name: self.get_parameter(name).value  # noqa: E731
        self._motion_limits = (float(param("max_joint_delta_rad")), float(param("max_hand_displacement_m")))
        if param("fixed_library"):
            import hashlib
            from pathlib import Path

            from pose_composer.fixed_backend import FixedBackend
            from pose_composer.fixed_library import FixedComposer, display_pose_policy

            if hashlib.sha256(Path(param("fixed_library")).read_bytes()).hexdigest() != param("fixed_library_sha256"):
                raise ValueError("fixed pose artifact changed after launch configuration binding")
            if not json.loads(param("hand_restore_json") or "{}"):
                raise ValueError("fixed poses require live hand/head checks")
            self._selected_backend = FixedBackend(
                param("runtime_profile"),
                json.loads(param("arm_limits_json")),
                param("fixed_library"),
                task_poses=json.loads(param("task_poses_json")),
            )
            self._composer = FixedComposer(self._selected_backend.library)
            self._display_pose_completion = display_pose_policy(
                json.loads(param("display_completion_json")), self._selected_backend.library
            )
            from aimdk_grasp.path_validation import ValidationGate

            self._validation_policy = json.loads(param("path_validation_json"))
            # Parse and bind any certificate before advertising an execution endpoint.
            gate = ValidationGate(self._selected_backend, self._validation_policy)
            self._validation_gate_factory = lambda: ValidationGate(
                self._selected_backend, self._validation_policy, certificate=gate.certificate
            )
            self.get_logger().info(f"fixed path validation policy: {json.dumps(gate.report, sort_keys=True)}")
        else:
            self._composer = self._generated_composer(param)

        from pose_composer.task_poses import TaskPoseRunner

        self._task_pose_runner = TaskPoseRunner(
            json.loads(param("task_poses_json")),
            simulation=param("task_pose_simulation"),
            backend=backend_for(self),
        )
        # On hardware, task poses are always sent raised by the predicted gravity
        # sag, so the measured arms land on the configured target. A simulated
        # arm follows its command exactly and gets none. Free poses keep their
        # compiled commands unchanged.
        self._task_pose_commands = None
        self._hand_config = json.loads(param("hand_restore_json") or "{}")
        if not param("task_pose_simulation"):
            from aimdk_grasp.sag_compensation import SagCompensation, deployed_hands

            self._task_pose_commands = SagCompensation(
                backend_for(self).model.ARM_JOINTS,
                json.loads(param("arm_limits_json") or "null"),
                installed_hands=deployed_hands(self._hand_config.get("grasp_execution")),
            )
            self.get_logger().info(f"task pose sag compensation: {self._task_pose_commands.describe()}")
        self._joint_received = {}
        self._hand_restorer_cache = None
        self._hand_lock = threading.Lock()
        self._trajectory_modes = set(json.loads(param("trajectory_modes_json")))
        group = self._callback_group = ReentrantCallbackGroup()
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
        if param("fixed_library"):
            # Construct immutable hand geometry and read-only status clients
            # before advertising actions. First motion must not pay this cold
            # startup cost; live-state checks still run for every trajectory.
            self._hand_restorer()
        self.create_service(ComposeArmPose, "~/compose", self._on_compose, callback_group=group)
        self.create_service(LookupArmPose, "~/lookup", self._on_lookup, callback_group=group)
        self.create_service(ArmPoseFeedback, "~/feedback", self._on_feedback, callback_group=group)
        self._server = ActionServer(
            self, ExecuteArmPose, "~/execute", execute_callback=self._on_execute,
            goal_callback=lambda _goal: GoalResponse.ACCEPT,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT, callback_group=group,
        )  # fmt: skip
        self.get_logger().info(f"pose executor ready (fixed library={bool(param('fixed_library'))})")

    def _generated_composer(self, param):
        from aimdk_robot.pose_backend.model import configure

        from pose_composer import spec
        from pose_composer.composer import Composer
        from pose_composer.seeds import fingerprint, load_seeds

        configure(
            param("mjcf_path"),
            param("runtime_profile"),
            json.loads(param("arm_limits_json")),
            task_poses=json.loads(param("task_poses_json")),
        )
        from embodied_common.vlm_api_client import VLMAPIClient

        client = (
            VLMAPIClient(
                provider=param("llm_provider"),
                base_url=param("llm_base_url"),
                api_key_env=param("llm_api_key_env"),
                model=param("llm_model"),
                timeout_sec=float(param("llm_timeout_sec")),
                temperature=float(param("llm_temperature")),
                multimodal=False,
            )
            if param("llm_model")
            else None
        )
        extra_body = json.loads(param("llm_extra_body_json") or "{}")

        def language_model(messages):
            if client is None:
                raise ComposeError("POSE_LANGUAGE_DISABLED", "this deployment only executes configured task poses")
            with trace_stage(_trace, "pose.model"):
                reply = client.complete(messages, extra_body=extra_body or None)
            if reply.get("status") != "ok":
                raise RuntimeError(reply.get("error") or "no reply")
            return reply["content"]

        memory = None
        if str(param("memory_path")).strip():
            memory = PoseMemory(
                param("memory_path"),
                fingerprint(spec.PROMPT_VERSION, str(param("llm_model")), str(param("mjcf_path"))),
                max_poses=int(param("memory_max_poses")),
                candidate_ttl_sec=float(param("memory_candidate_ttl_days")) * 86400.0,
            )
            seeded = load_seeds(memory, str(param("memory_seeds_path")).strip())
            stale = memory.mark_stale()
            counts = memory.revalidate(spec.resolve)
            memory.evict()
            self.get_logger().info(
                f"pose memory {memory.path}: {len(memory.list())} poses, seeds loaded {seeded}, "
                f"stale {stale} -> {counts}"
            )
        return Composer(language_model, ttl_sec=CONTRACT.PREPARED_TTL_SEC, memory=memory)

    def destroy_node(self):
        if self._composer.memory is not None:
            self._composer.memory.close()
        restorer = getattr(self, "_hand_restorer_cache", None)
        if restorer is not None:
            restorer.ports.close()
        return super().destroy_node()

    def _hand_restorer(self):
        """The hand restorer, built on first use from the runtime's description; raises when it cannot be.

        A deployment whose configuration lacks the pieces, or whose hands have no verified model, refuses
        in the runner (``HAND_CAPABILITY_MISSING``) rather than opening a hand without its checks.
        """
        with self._hand_lock:
            if self._hand_restorer_cache is None:
                config = self._hand_config
                if not config:
                    raise RuntimeError("hand_restore_json is not configured for this deployment")
                from aimdk_grasp.hand_restore_ros import create_hand_restorer

                self._hand_restorer_cache = create_hand_restorer(self, config, callback_group=self._callback_group)
                self.get_logger().info("hand restore ready")
            return self._hand_restorer_cache

    # --- inputs -------------------------------------------------------------

    def _on_joints(self, msg: JointState) -> None:
        with self._joint_lock:
            self._positions.update(dict(zip(msg.name, msg.position, strict=False)))
            self._joint_received.update({n: time.monotonic() for n in msg.name[: len(msg.position)]})

    def _measured(self) -> dict[str, float]:
        with self._joint_lock:
            if not all(n in self._positions for n in backend_for(self).model.ARM_JOINTS):
                raise ComposeError("POSE_NO_JOINT_STATE", "no arm joint state yet")
            return {n: float(self._positions[n]) for n in backend_for(self).model.ARM_JOINTS}

    # --- compose ------------------------------------------------------------

    def _on_compose(self, request, response):
        if not self._busy.acquire(blocking=False):
            response.success, response.error_code = False, "POSE_BUSY"
            response.message = "another pose is being composed or executed"
            return response
        try:
            with trace_scope(request.session_id), trace_stage(_trace, "pose.compose"):
                return self._compose_locked(request, response)
        finally:
            self._busy.release()

    def _compose_locked(self, request, response):
        try:
            start = self._measured()
            outside = backend_for(self).safety.limit_violations(
                start, limits=backend_for(self).model.load().hard_limits
            )
            if outside:
                self.get_logger().warning(f"measured arm pose outside declared runtime limits: {outside}")
            prepared = self._composer.compose(
                request.text, start, session_id=request.session_id, utterance=request.utterance
            )
            composition = prepared.composition
            response.from_memory = prepared.from_memory
            response.pose_id = prepared.pose_id
            self.get_logger().info(
                f"pose memory: {'hit ' + prepared.pose_id if prepared.from_memory else 'miss; generated'} "
                f"target={request.text!r}"
            )
            response.kind = composition.kind
            response.user_message = composition.user_message
            response.composition_id = prepared.composition_id
            response.description_json = json.dumps(prepared.description, ensure_ascii=False, sort_keys=True)
            response.approximate = bool(composition.approx)
            if composition.kind in ("pose", "sequence"):
                with trace_stage(_trace, "pose.plan"):
                    plan = backend_for(self).spec.plan(composition, start)
                if not plan.ok:
                    self._composer.take(prepared.text, prepared.composition_id)
                    self.get_logger().warning(f"pose path refused: {json.dumps(plan.report.summary())}")
                    raise ComposeError("POSE_NO_SAFE_PATH", "no collision-free path to the pose")
                reason = backend_for(self).check_displacement(plan, start, *self._motion_limits)
                if reason:
                    self._composer.take(prepared.text, prepared.composition_id)
                    raise ComposeError("POSE_MOTION_LIMIT", reason)
                response.duration_sec = float(plan.traj.times[-1])
                response.message = f"composed {composition.kind} ({len(composition.frames)} frame(s))"
                if getattr(backend_for(self), "library", None) is not None:
                    response.message += "; candidate only; measured path validation required at execution"
            else:
                response.message = composition.reason or composition.kind
            response.success = True
        except ComposeError as exc:
            response.success, response.error_code, response.message = False, exc.code, exc.message
        return response

    def _on_lookup(self, request, response):
        if not bool(self.get_parameter("lookup_enabled").value):
            response.hit, response.pose_id, response.state = False, "", ""
            response.user_message, response.catalog_json = "", "[]"
            return response
        if hasattr(self._composer, "lookup"):
            return self._composer.lookup(request, response)
        memory = self._composer.memory
        record = memory.lookup(request.text, confirmed_only=True) if memory is not None else None
        response.catalog_json = json.dumps(
            memory.planning_catalog() if memory is not None and record is None else [], ensure_ascii=False
        )
        if record is not None:
            response.hit, response.pose_id, response.state = True, record.pose_id, record.state
            response.user_message = f"好的，{request.text.strip()}"
        return response

    def _on_feedback(self, request, response):
        if request.verdict not in ("good", "bad", "next_task", "offer", "status"):
            response.success, response.message = False, f"unknown verdict {request.verdict!r}"
            return response
        pose_id, outcome = self._composer.feedback(request.session_id, request.verdict)
        response.success, response.pose_id, response.message = True, pose_id, outcome
        if pose_id:
            self.get_logger().info(f"pose memory: {request.verdict} on {pose_id} -> {outcome}")
        return response

    # --- execute ------------------------------------------------------------

    def _on_execute(self, goal_handle):
        result = ExecuteArmPose.Result()
        if not self._busy.acquire(blocking=False):
            return self._fail(goal_handle, result, "POSE_BUSY", "another pose is being composed or executed")
        self._path_report = None
        try:
            self._execution_gate = (
                self._validation_gate_factory() if hasattr(self, "_validation_gate_factory") else None
            )
            if self._execution_gate is not None:
                self._path_report = self._execution_gate.report
            if getattr(self, "_stop_unknown", False):
                return self._fail(goal_handle, result, "POSE_STOP_UNKNOWN", "previous motion stop is unconfirmed")
            with trace_scope(goal_handle.request.composition_id), trace_stage(_trace, "pose.execute"):
                return self._execute_locked(goal_handle, result)
        except ComposeError as exc:
            return self._fail(goal_handle, result, exc.code, exc.message)
        finally:
            if getattr(self, "_path_report", None) is not None:
                self.get_logger().info("pose path validation: " + json.dumps(self._path_report, sort_keys=True))
                policy = self._path_report["policy"]
                result.message += f"; path_validation=hand:{policy['hand']},body:{policy['body']}"
            if len(result.message) > 300:
                self.get_logger().info("pose result detail: " + result.message)
                result.message = result.message[:299] + "…"
            self._busy.release()

    def _preparation_message(self, message, report):
        """Keep the action result readable; full diagnostics stay in the node log."""
        self.get_logger().info("pose preparation: " + json.dumps(report, sort_keys=True))
        hands = ",".join(report.get("completed_hands", [])) or "none"
        return f"{message}; preparation={report.get('total_sec', 0):.3f}s; opened_hands={hands}"

    def _feedback(self, goal_handle, state: str, progress: float) -> None:
        feedback = ExecuteArmPose.Feedback()
        feedback.state, feedback.progress = state, float(progress)
        goal_handle.publish_feedback(feedback)

    def _execute_locked(self, goal_handle, result):
        request = goal_handle.request
        gate = getattr(self, "_execution_gate", None)
        with (
            gate.stage("runtime_state") if gate is not None else nullcontext(),
            trace_stage(_trace, "pose.runtime_mode"),
        ):
            mode = self._runtime_mode()
        if mode not in self._trajectory_modes:
            return self._fail(
                goal_handle, result, "POSE_RUNTIME_UNAVAILABLE",
                f"runtime mode is {mode!r}; arm poses run in {sorted(self._trajectory_modes)}",
            )  # fmt: skip
        if request.named_pose:
            if request.text or request.composition_id:
                return self._fail(
                    goal_handle, result, "TASK_POSE_BAD_REQUEST", "named_pose is exclusive with free text"
                )

            return self._execute_task_pose(goal_handle, result, request.named_pose, request.task_poses_digest, mode)
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
        description = getattr(prepared, "description", {}) or {}
        left = (description.get("left") or {}).get("task_pose")
        right = (description.get("right") or {}).get("task_pose")
        if composition.kind == "pose" and description.get("kind") == "pose" and left and left == right:
            return self._execute_task_pose(
                goal_handle, result, left, self._task_pose_runner.snapshot.get("digest", ""), mode
            )
        if getattr(backend_for(self), "library", None) is not None:
            return self._execute_fixed_target(goal_handle, result, prepared)
        self._feedback(goal_handle, "planning", 0.0)
        with trace_stage(_trace, "pose.plan"):
            plan = backend_for(self).spec.plan(composition, start)
        if not plan.ok:
            return self._fail(goal_handle, result, "POSE_NO_SAFE_PATH", "no collision-free path from the current pose")
        reason = backend_for(self).check_displacement(plan, start, *self._motion_limits)
        if reason:
            return self._fail(goal_handle, result, "POSE_MOTION_LIMIT", reason)
        with trace_stage(_trace, "pose.trajectory"):
            ok, code, message = self._run_trajectory(goal_handle, plan)
        note = "" if result.used_prepared else " (composed again: the prepared composition was missing or expired)"
        if code == "POSE_CANCELLED":
            goal_handle.canceled()
            result.error_code, result.message = code, message
            return result
        if not ok:
            return self._fail(goal_handle, result, code, message + note)
        pose_id = self._composer.succeeded(prepared)
        if pose_id:
            record = self._composer.memory.get(pose_id)
            self.get_logger().info(f"pose memory: {pose_id} is {record.state if record else 'gone'}")
        self.get_logger().info(
            f"pose {prepared.composition_id} reached ({'prepared with the plan' if result.used_prepared else 'composed at execution'}"
            f"{', from memory ' + prepared.pose_id if prepared.from_memory else ''}): "
            f"{prepared.text}"
        )
        goal_handle.succeed()
        result.success = True
        result.message = f"{composition.user_message or 'pose reached'}; {message}{note}"
        return result

    def _execute_task_pose(self, goal_handle, result, name, digest, mode):
        """One HOME preparation path for named requests and both-arm seed anchors."""
        request = goal_handle.request

        def read_state():
            with self._joint_lock:
                return (
                    {n: self._positions[n] for n in backend_for(self).model.ARM_JOINTS if n in self._positions},
                    dict(self._joint_received),
                )

        commands = getattr(self, "_task_pose_commands", None)
        shaping = {} if commands is None else {"commands": commands}
        try:
            skipped = self._task_pose_runner.execute(
                name,
                digest,
                read_state=read_state,
                move=lambda plan, deadline: self._run_trajectory(goal_handle, plan, deadline=deadline, **shaping),
                cancelled=lambda: goal_handle.is_cancel_requested,
                timeout_sec=request.timeout_sec,
                motion_limits=self._motion_limits,
                hands=self._hand_restorer,
                gateway_task_id=request.gateway_task_id,
                progress=lambda state, fraction: self._feedback(goal_handle, state, fraction),
            )
        except ComposeError as exc:
            message = self._preparation_message(str(exc), self._task_pose_runner.report)
            if exc.code == "POSE_CANCELLED":
                goal_handle.canceled()
                result.error_code, result.message = exc.code, message
                return result
            return self._fail(goal_handle, result, exc.code, message)
        if self._runtime_mode() != mode:
            return self._fail(
                goal_handle, result, "POSE_RUNTIME_UNAVAILABLE", "runtime mode changed during preparation"
            )
        goal_handle.succeed()
        result.success = True
        outcome = "skipped" if skipped else "executed"
        result.message = self._preparation_message(
            f"task pose {name} arrival checked once; continuous stability not verified; arm motion {outcome}",
            self._task_pose_runner.report,
        )
        return result

    def _execute_fixed_target(self, goal_handle, result, prepared):
        from pose_composer.task_poses import TaskPoseRunner

        backend = backend_for(self)
        display = getattr(self, "_display_pose_completion", {})
        display = display if prepared.pose_id in display.get("pose_ids", ()) else {}
        runner = TaskPoseRunner(
            {
                "schema_version": 1,
                "robot_type": "x2_ultra_omnihand",
                "joint_names": list(backend.ARM_JOINTS),
                "poses": {
                    prepared.pose_id: {
                        "joint_positions": prepared.composition.frames[0],
                        "hardware_approved": True,
                        "position_tolerance_rad": 0.1,
                        # The Gateway supplies the deployment's smaller cap.
                        # CPU geometry checks on the 310P can exceed 90 seconds.
                        "timeout_sec": 300.0,
                    }
                },
            },
            backend=backend,
            verify_arrival=not bool(display),
            clock=lambda: time.monotonic(),
            pause=lambda seconds: time.sleep(seconds),
        )

        def read_state():
            with self._joint_lock:
                return (
                    {n: self._positions[n] for n in backend.ARM_JOINTS if n in self._positions},
                    dict(self._joint_received),
                )

        try:
            runner.execute(
                prepared.pose_id,
                runner.snapshot["digest"],
                read_state=read_state,
                move=lambda plan, deadline: self._run_trajectory(
                    goal_handle, plan, deadline=deadline, verify_completion=not bool(display)
                ),
                cancelled=lambda: goal_handle.is_cancel_requested,
                timeout_sec=goal_handle.request.timeout_sec,
                motion_limits=self._motion_limits,
                gateway_task_id=goal_handle.request.gateway_task_id,
                progress=lambda state, fraction: self._feedback(goal_handle, state, fraction),
            )
        except ComposeError as exc:
            message = self._preparation_message(str(exc), runner.report)
            if exc.code == "POSE_CANCELLED":
                goal_handle.canceled()
                result.error_code, result.message = exc.code, message
                return result
            return self._fail(goal_handle, result, exc.code, message)
        goal_handle.succeed()
        result.success = True
        message = (
            f"display playback finished: {prepared.pose_id}; final pose not verified"
            if display
            else f"fixed pose {prepared.pose_id} arrival checked once; continuous stability not verified"
        )
        result.message = self._preparation_message(message, runner.report)
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
        status = future.result().status
        if status.lifecycle != "ACTIVE" or status.stop_latched or status.faults:
            raise ComposeError("POSE_RUNTIME_UNAVAILABLE", "runtime is not active or has a latched stop")
        return str(status.active_mode)

    def _run_trajectory(
        self, goal_handle, plan, *, deadline=None, commands=None, verify_completion=True
    ) -> tuple[bool, str, str]:
        def named_refusal():
            if deadline is None:
                return None
            if goal_handle.is_cancel_requested:
                return False, "POSE_CANCELLED", "task pose canceled before trajectory dispatch"
            if time.monotonic() >= deadline:
                return False, "TASK_POSE_TIMEOUT", "task-pose budget expired before trajectory dispatch"
            return None

        refusal = named_refusal()
        if refusal:
            return refusal
        wait_sec = 5.0 if deadline is None else min(5.0, max(0.0, deadline - time.monotonic()))
        if not self._traj.wait_for_server(timeout_sec=wait_sec):
            return named_refusal() or (False, "POSE_RUNTIME_UNAVAILABLE", "runtime trajectory action unavailable")
        refusal = named_refusal()
        if refusal:
            return refusal
        goal = FollowJointTrajectory.Goal()
        if not verify_completion:
            from control_msgs.msg import JointTolerance

            goal.goal_tolerance = [
                JointTolerance(name=name, position=-1.0, velocity=-1.0, acceleration=-1.0)
                for name in backend_for(self).ARM_JOINTS
            ]
        goal.trajectory.joint_names = list(backend_for(self).model.ARM_JOINTS)
        if getattr(backend_for(self), "library", None) is None:
            plan = backend_for(self).prepare_execution(plan, POINT_RATE_HZ)
        if not plan.ok:
            return False, "UNSAFE_POSE", str(plan.report.summary())
        times, positions = plan.traj.times, plan.traj.positions
        if getattr(backend_for(self), "library", None) is not None and not getattr(self, "_hand_config", None):
            return False, "HAND_CAPABILITY_MISSING", "fixed candidates require measured body and hand checks"
        hand_guard = None
        gate = None
        if getattr(self, "_hand_config", None):
            from aimdk_grasp.hand_restore import RestoreContext, RestoreError

            hand_deadline = deadline or (time.monotonic() + float(goal_handle.request.timeout_sec or 90.0))
            context = RestoreContext(
                goal_handle.request.gateway_task_id, hand_deadline, lambda: goal_handle.is_cancel_requested
            )
            try:
                restorer = self._hand_restorer()
                fixed = getattr(backend_for(self), "library", None) is not None
                gate = None
                if fixed:
                    from aimdk_grasp.path_validation import ValidationGate

                    gate = getattr(self, "_execution_gate", None) or ValidationGate(
                        backend_for(self), getattr(self, "_validation_policy", None)
                    )
                    self._path_report = gate.report
                    baseline = gate.check(restorer, context, plan, FIXED_START_TOLERANCE_RAD)
                else:
                    restorer.ports.prepare(context)
                    baseline = restorer.check_arm_path(context, backend_for(self).model.ARM_JOINTS, positions)
                arm_phase = "pre_dispatch"
                motion_started = None

                def hand_guard():
                    observed = restorer.guard_arm_motion(context, baseline, arm_phase)
                    if gate is not None and gate.monitor is not None:
                        gate.monitor.check(
                            observed.feedback, motion_started, gate.conditions(), feedback_age_s=observed.feedback_age_s
                        )

                hand_guard()
            except RestoreError as exc:
                return False, exc.code, str(exc)
            except (ValueError, RuntimeError) as exc:
                return False, "HAND_CAPABILITY_MISSING", str(exc)
        dispatch_started = time.monotonic()
        # ``commands`` raises what is sent by the predicted sag; every check here
        # and in the runtime keeps comparing the measured arm with the plan.
        sent = positions if commands is None else commands(positions)
        if commands is not None:
            raised = np.asarray(sent, dtype=float) - np.asarray(positions, dtype=float)
            worst = np.unravel_index(int(np.argmax(np.abs(raised))), raised.shape)
            self._task_pose_runner.report["sag_compensation"] = {
                "max_rad": float(abs(raised[worst])),
                "joint": backend_for(self).model.ARM_JOINTS[worst[1]],
                "final_rad": [float(v) for v in raised[-1]],
            }
        for k in range(1, len(times)):
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in sent[k]]
            sec = float(times[k])
            point.time_from_start = Duration(sec=int(sec), nanosec=int((sec - int(sec)) * 1e9))
            goal.trajectory.points.append(point)
        refusal = named_refusal()
        if refusal:
            return refusal
        if getattr(backend_for(self), "library", None) is not None:
            joints = backend_for(self).ARM_JOINTS
            with self._joint_lock:
                now = time.monotonic()
                if any(
                    n not in self._positions
                    or not math.isfinite(self._positions[n])
                    or not 0 <= now - self._joint_received.get(n, float("-inf")) <= 0.5
                    for n in joints
                ):
                    return False, "TASK_POSE_STALE_STATE", "fresh finite arm state required before dispatch"
                current = {n: self._positions[n] for n in joints}
            if any(abs(current[n] - positions[0][i]) > FIXED_START_TOLERANCE_RAD for i, n in enumerate(joints)):
                return False, "POSE_START_CHANGED", "arms moved while the fixed path was being checked; no motion sent"
        if hand_guard is not None:
            try:
                hand_guard()
            except RestoreError as exc:
                return False, exc.code, str(exc)
        arm_phase = "trajectory_sent"
        motion_started = time.monotonic()
        if gate is not None:
            gate.report["stages"]["dispatch_gate"] = {
                "status": "executed",
                "sec": motion_started - dispatch_started,
                "reason": "fresh start rechecked",
            }
            gate.report["stages"]["arm_motion"] = {
                "status": "executed",
                "sec": 0.0,
                "reason": "dispatch sent; terminal pending",
            }

        def stop_unknown(message):
            self._stop_unknown = True
            if gate is not None:
                gate.report["stages"]["arm_motion"] = {
                    "status": "failed",
                    "sec": time.monotonic() - motion_started,
                    "reason": message,
                    "stop_confirmed": False,
                }
            return False, "POSE_STOP_UNKNOWN", message

        try:
            future = self._traj.send_goal_async(goal)
        except Exception:
            return stop_unknown("trajectory dispatch raised without a terminal; further motion blocked")
        handshake_failure = None
        while not future.done():
            if hand_guard is not None and handshake_failure is None:
                try:
                    hand_guard()
                except RestoreError as exc:
                    handshake_failure = (False, exc.code, str(exc))
            if time.monotonic() - motion_started > 3.0:
                # The send may have reached the runtime. A missing acceptance
                # response is not evidence that nothing moved. Cancel a late
                # acceptance and latch this executor until reconciled.
                def cancel_late(reply):
                    try:
                        late = reply.result()
                        if late is not None and late.accepted:
                            late.cancel_goal_async()
                    except Exception:
                        pass  # Stop remains unknown; no new motion is accepted.

                future.add_done_callback(cancel_late)
                return stop_unknown("trajectory acceptance missing; further motion blocked")
            time.sleep(0.02)
        try:
            handle = future.result()
        except Exception:
            return stop_unknown("trajectory acceptance failed; further motion blocked")
        if handle is None:
            return stop_unknown("trajectory acceptance absent; further motion blocked")
        _trace.info("[mark] trace_id=- stage=pose.trajectory_accepted")
        if not handle.accepted:
            return False, "POSE_RUNTIME_REJECTED", "runtime rejected the trajectory"
        try:
            result_future = handle.get_result_async()
        except Exception:
            return stop_unknown("trajectory result channel failed; further motion blocked")
        started, total = time.monotonic(), float(times[-1])
        cancel_sent = False
        cancel_started = None
        hand_failure = handshake_failure
        while not result_future.done():
            if hand_guard is not None and hand_failure is None:
                try:
                    hand_guard()
                except RestoreError as exc:
                    hand_failure = (False, exc.code, str(exc))
            if (
                goal_handle.is_cancel_requested
                or (deadline is not None and time.monotonic() >= deadline)
                or hand_failure is not None
            ) and not cancel_sent:
                try:
                    handle.cancel_goal_async()
                except Exception:
                    return stop_unknown("trajectory cancellation failed; further motion blocked")
                cancel_sent = True
                cancel_started = time.monotonic()
            stop_wait = (
                gate.certificate.data["response_sec"]
                if hand_guard is not None and gate is not None and gate.monitor is not None
                else 3.0
            )
            if cancel_started is not None and time.monotonic() - cancel_started > stop_wait:
                return stop_unknown("runtime did not confirm cancellation; further motion blocked")
            self._feedback(goal_handle, "moving", min((time.monotonic() - started) / total, 1.0))
            time.sleep(0.1)
        if getattr(self, "_path_report", None) is not None:
            self._path_report["stages"]["arm_motion"] = {
                "status": "executed",
                "sec": time.monotonic() - started,
                "reason": "runtime terminal received",
            }
        try:
            wrapped = result_future.result()
        except Exception:
            return stop_unknown("trajectory terminal unavailable; further motion blocked")
        if wrapped is None or wrapped.result is None:
            return stop_unknown("trajectory terminal absent; further motion blocked")
        outcome = wrapped.result
        if hand_guard is not None and hand_failure is None:
            try:
                hand_guard()
            except RestoreError as exc:
                hand_failure = (False, exc.code, str(exc))
        if hand_failure is not None:
            return hand_failure
        if cancel_sent and not goal_handle.is_cancel_requested:
            return False, "TASK_POSE_TIMEOUT", "trajectory budget expired; cancellation confirmed"
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
