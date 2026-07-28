"""Pick action orchestration and shared runtime control helpers."""

from __future__ import annotations

import time

import rclpy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from ibrobot_msgs.action import PickObject, PrimitiveCommand
from manipulation_execution.pick_executor_models import FlowState, PickCancelled, PickFlowError


class PickFlowPhase:
    """Orchestrate the phase objects while keeping ROS lifecycle in the node."""

    def _check_cancel(self, goal_handle) -> None:
        if goal_handle.is_cancel_requested:
            raise PickCancelled()

    def _publish_feedback(
        self,
        goal_handle,
        state: FlowState,
        phase: str,
        detail: str,
    ) -> None:
        self._check_cancel(goal_handle)
        if not state.completed_phases or state.completed_phases[-1] != phase:
            state.completed_phases.append(phase)
        feedback = PickObject.Feedback()
        feedback.phase = phase
        feedback.progress = float(self._PHASE_PROGRESS.get(phase, 0.0))
        feedback.attempt = int(state.attempt)
        feedback.detail = detail
        goal_handle.publish_feedback(feedback)

    def _wait_future(self, future, goal_handle, deadline: float, timeout_sec: float, label: str):
        local_deadline = min(deadline, time.monotonic() + max(0.1, timeout_sec))
        while rclpy.ok() and not future.done():
            self._check_cancel(goal_handle)
            if time.monotonic() >= local_deadline:
                future.cancel()
                raise PickFlowError("RPC_TIMEOUT", f"{label} timed out")
            time.sleep(0.05)
        response = future.result()
        if response is None:
            raise PickFlowError("RPC_FAILED", f"{label} returned no response")
        return response

    def _wait_for_service(self, client, deadline: float, service_name: str, *, required: bool = True) -> bool:
        timeout = min(self._ready_timeout, self._remaining(deadline))
        ready = client.service_is_ready() or client.wait_for_service(timeout_sec=max(0.1, timeout))
        if not ready and required:
            raise PickFlowError("SERVICE_UNAVAILABLE", f"required service unavailable: {service_name}")
        return ready

    def _preflight(self, goal_handle, deadline: float, state: FlowState) -> None:
        self._publish_feedback(goal_handle, state, "preflight", "checking grasp services and safe primitive server")
        if not self._primitive_client.wait_for_server(timeout_sec=min(self._ready_timeout, self._remaining(deadline))):
            raise PickFlowError("PRIMITIVE_SERVER_UNAVAILABLE", self._primitive_action_name)
        self._wait_for_service(self._planner_client, deadline, self._planner_service)
        self._wait_for_service(self._detect_client, deadline, self._detect_service)
        self._wait_for_service(
            self._move_configuration_client,
            deadline,
            self._move_configuration_service,
        )
        verification_required = self._verification_policy == "required"
        self._wait_for_service(
            self._verifier_client,
            deadline,
            self._verifier_service,
            required=verification_required,
        )
        self._wait_for_service(self._ik_client, deadline, self._ik_service)
        self._wait_for_service(self._fk_client, deadline, self._fk_service)
        for index, (ik_client, fk_client) in enumerate(
            zip(self._ik_worker_clients, self._fk_worker_clients, strict=True)
        ):
            namespace = f"{self._ik_worker_prefix}_{index}"
            self._wait_for_service(ik_client, deadline, f"{namespace}/compute_ik")
            self._wait_for_service(fk_client, deadline, f"{namespace}/compute_fk")
        joint_state_deadline = min(deadline, time.monotonic() + self._ready_timeout)
        while self._snapshot_joint_state() is None:
            self._check_cancel(goal_handle)
            if time.monotonic() >= joint_state_deadline:
                raise PickFlowError(
                    "JOINT_STATE_UNAVAILABLE",
                    f"no current joint state received from {self._joint_state_topic}",
                )
            time.sleep(0.05)
        camera_frame = str(self._config.get("camera", {}).get("frame_id", "")).strip()
        if camera_frame:
            self._lookup_base_to_camera(camera_frame)

    def _run_primitive(
        self,
        goal_handle,
        deadline: float,
        task_id: str,
        primitive_name: str,
        *,
        pose: Pose | None = None,
        pose_name: str = "",
        velocity_scaling: float = 0.0,
        gripper_position: float = 0.0,
        joint_state: JointState | None = None,
        duration_sec: float = 0.0,
    ) -> None:
        primitive_goal = PrimitiveCommand.Goal()
        primitive_goal.task_id = task_id
        primitive_goal.primitive_name = primitive_name
        primitive_goal.pose_name = pose_name
        if pose is not None:
            primitive_goal.target_pose = pose
        primitive_goal.velocity_scaling = float(velocity_scaling)
        primitive_goal.gripper_position = float(gripper_position)
        if joint_state is not None:
            position_by_name = dict(zip(joint_state.name, joint_state.position, strict=False))
            missing = [name for name in self._arm_joint_names if name not in position_by_name]
            if missing:
                raise PickFlowError(
                    "IK_JOINTS_MISSING", f"IK result missing arm joints: {', '.join(missing)}", retryable=True
                )
            primitive_goal.joint_names = list(self._arm_joint_names)
            primitive_goal.joint_positions = [float(position_by_name[name]) for name in self._arm_joint_names]
            primitive_goal.primitive_duration_sec = float(duration_sec)
        primitive_goal.timeout_sec = float(self._remaining(deadline))

        send_future = self._primitive_client.send_goal_async(primitive_goal)
        primitive_handle = self._wait_future(send_future, goal_handle, deadline, self._rpc_timeout, primitive_name)
        if not primitive_handle.accepted:
            raise PickFlowError("PRIMITIVE_REJECTED", f"primitive rejected: {primitive_name}", retryable=True)

        result_future = primitive_handle.get_result_async()
        try:
            action_result = self._wait_future(
                result_future,
                goal_handle,
                deadline,
                self._remaining(deadline),
                primitive_name,
            )
        except PickCancelled:
            primitive_handle.cancel_goal_async()
            raise
        except PickFlowError:
            primitive_handle.cancel_goal_async()
            raise
        result = action_result.result
        if not result.success:
            raise PickFlowError(
                result.error_code or "PRIMITIVE_FAILED",
                result.message or f"primitive failed: {primitive_name}",
                retryable=primitive_name in {"move_to_named_pose", "move_to_pose"},
            )

    def _move_to_observe(self, goal_handle, deadline: float, state: FlowState, task_id: str) -> None:
        observe_pose = str(self._config.get("observe_pose", "observe_table"))
        if not observe_pose:
            return
        self._publish_feedback(goal_handle, state, "observe", f"moving to {observe_pose}")
        self._run_primitive(
            goal_handle,
            deadline,
            task_id,
            "move_to_named_pose",
            pose_name=observe_pose,
            velocity_scaling=float(self._config.get("observe_velocity_scaling", 0.05)),
        )
        self._sleep_with_cancel(goal_handle, deadline, float(self._config.get("observe_settle_sec", 0.6)))
        camera_frame = str(self._config.get("camera", {}).get("frame_id", "")).strip()
        if camera_frame:
            self._record_frame_diagnostic(state, "observe_settled_latest", camera_frame)

    def _sleep_with_cancel(self, goal_handle, deadline: float, duration_sec: float) -> None:
        end = min(deadline, time.monotonic() + max(0.0, duration_sec))
        while time.monotonic() < end:
            self._check_cancel(goal_handle)
            time.sleep(0.05)

    def _execute_pick(self, goal_handle):
        goal = goal_handle.request
        state = FlowState(completed_phases=[])
        timeout_sec = float(goal.timeout_sec or self._config.get("timeout_sec", 180.0))
        deadline = time.monotonic() + timeout_sec
        target_query = str(goal.target_query).strip()
        task_id = str(goal.task_id).strip() or f"pick-{int(time.time() * 1000)}"
        try:
            if len(target_query) > 200:
                raise PickFlowError("INVALID_TARGET", "target_query is too long")
            self._preflight(goal_handle, deadline, state)
            self._move_to_observe(goal_handle, deadline, state, task_id)
            stage_started = time.monotonic()
            grasp_header, candidates, scene = self._request_grasps(
                goal_handle,
                deadline,
                state,
                target_query,
            )
            self.get_logger().info(
                f"PIPELINE_TIMING stage=graspgen_request duration_s={time.monotonic() - stage_started:.3f}"
            )
            default_frame = str(grasp_header.frame_id)
            capture_transform = self._lookup_base_transform(default_frame, grasp_header.stamp)
            base_to_camera = self._transform_to_matrix(capture_transform)
            self._record_frame_diagnostic(
                state,
                "grasp_capture_stamp",
                default_frame,
                grasp_header.stamp,
                camera_transform=capture_transform,
            )
            self._record_frame_diagnostic(state, "post_plan_latest", default_frame)
            scene_base = self._scene_geometry_base(base_to_camera, scene)
            self._publish_feedback(goal_handle, state, "selecting", f"evaluating {len(candidates)} candidates")
            stage_started = time.monotonic()
            ranked = self._rank_candidates(
                default_frame,
                grasp_header.stamp,
                base_to_camera,
                candidates,
                scene,
                scene_base,
            )
            self.get_logger().info(
                f"PIPELINE_TIMING stage=candidate_geometry_ranking "
                f"duration_s={time.monotonic() - stage_started:.3f} candidates={len(ranked)}"
            )
            candidate_seed = self._snapshot_joint_state()
            if candidate_seed is None:
                raise PickFlowError(
                    "JOINT_STATE_UNAVAILABLE",
                    f"no current joint state received from {self._joint_state_topic}",
                )
            max_attempts = int(self._config.get("max_execution_attempts", 1))
            prepared_candidates, last_error = self._prepare_ranked_candidates(
                ranked,
                scene_base,
                candidate_seed,
                goal_handle,
                deadline,
            )

            if not prepared_candidates:
                if last_error is None:
                    raise PickFlowError("NO_EXECUTABLE_CANDIDATE", "no candidate could be prepared")
                raise PickFlowError(
                    "ALL_CANDIDATES_FAILED",
                    f"all {len(ranked)} ranked candidates failed preparation; last={last_error.code}: {last_error}",
                )

            prepared_scoring = self._config.get("prepared_candidate_scoring", {})
            if bool(prepared_scoring.get("enabled", False)):
                prepared_candidates.sort(
                    key=lambda item: (
                        -item.selection_score,
                        item.contact_z_error_m,
                        item.contact_residual_xy_m,
                        -item.ranked.score,
                        item.ranked.index,
                    )
                )
            else:
                prepared_candidates.sort(
                    key=lambda item: (
                        item.contact_z_error_m,
                        item.contact_residual_xy_m,
                    )
                )
            self._record_prepared_ranking(state, prepared_candidates)
            summary_parts = []
            for item in prepared_candidates[:10]:
                envelope = item.fixed_finger_envelope
                fixed_text = "n/a" if envelope is None else f"{envelope.fixed_gap_m:.4f}/{envelope.score:.3f}"
                base_side_text = (
                    "n/a"
                    if item.ranked.fixed_finger_base_side is None
                    else f"{item.ranked.fixed_finger_base_side.alignment_cos:.3f}"
                )
                fk_base_side_text = (
                    "n/a"
                    if item.fk_fixed_finger_base_side is None
                    else f"{item.fk_fixed_finger_base_side.alignment_cos:.3f}"
                )
                summary_parts.append(
                    f"{item.ranked.index}:score={item.selection_score:.3f}/fixed={fixed_text}/"
                    f"base_side={base_side_text}/fk_base_side={fk_base_side_text}/"
                    f"z={item.contact_z_error_m:.4f}/xy={item.contact_residual_xy_m:.4f}/"
                    f"approach_axis={item.approach_axis_error_deg}/closing_axis={item.closing_axis_error_deg:.1f}"
                )
            summary = ", ".join(summary_parts)
            self.get_logger().info(f"prepared candidate rank: {summary}")

            attempt_candidates = (
                prepared_candidates if max_attempts <= 0 else prepared_candidates[: max(1, max_attempts)]
            )
            for attempt, prepared in enumerate(attempt_candidates, start=1):
                state.attempt = attempt
                candidate = prepared.ranked
                try:
                    self._execute_candidate(
                        goal_handle,
                        deadline,
                        state,
                        task_id,
                        target_query,
                        prepared,
                        scene_base,
                    )
                    self._publish_feedback(goal_handle, state, "completed", "object grasped and verified")
                    result = self._result_from_state(state)
                    result.success = True
                    result.error_code = ""
                    result.message = (
                        f"picked {target_query!r}; candidate={candidate.index}; attempts={attempt}; "
                        f"verification_confidence={state.verification_confidence:.3f}"
                    )
                    goal_handle.succeed()
                    return result
                except PickCancelled:
                    raise
                except PickFlowError as exc:
                    last_error = exc
                    self.get_logger().warning(
                        f"pick candidate failed: attempt={attempt} candidate={candidate.index} "
                        f"code={exc.code} retryable={exc.retryable} message={exc}"
                    )
                    if not exc.retryable:
                        raise
            if last_error is None:
                raise PickFlowError("NO_EXECUTION_RESULT", "prepared candidates produced no execution result")
            raise PickFlowError(
                "ALL_CANDIDATES_FAILED",
                f"all {state.attempt} attempted candidates failed; last={last_error.code}: {last_error}",
            )
        except PickCancelled as exc:
            result = self._result_from_state(state)
            result.success = False
            result.error_code = exc.code
            result.message = str(exc)
            goal_handle.canceled()
            return result
        except PickFlowError as exc:
            result = self._result_from_state(state)
            result.success = False
            result.error_code = exc.code
            result.message = str(exc)
            goal_handle.abort()
            return result
        except Exception as exc:
            self.get_logger().exception("unexpected pick execution failure")
            result = self._result_from_state(state)
            result.success = False
            result.error_code = "INTERNAL_ERROR"
            result.message = str(exc)
            goal_handle.abort()
            return result
        finally:
            with self._goal_lock:
                self._goal_active = False
