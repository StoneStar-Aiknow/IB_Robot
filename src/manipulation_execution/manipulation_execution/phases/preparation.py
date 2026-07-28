"""IK/FK candidate preparation and target-gripper validation phase."""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rclpy.duration import Duration
from sensor_msgs.msg import JointState

from manipulation_execution.contact_compensation import ContactPrediction, compensate_contact_xy
from manipulation_execution.grasp_geometry import (
    CandidatePlan,
    FixedFingerBaseSide,
    canonicalize_joint5,
    fixed_finger_base_side_alignment,
    fixed_finger_envelope_score,
    grasp_axis_errors,
    joint5_closing_axis_correction,
    prepared_candidate_soft_score,
    xyz_within_workspace,
)
from manipulation_execution.pick_executor_models import (
    BaseSceneGeometry,
    FlowState,
    IKPayload,
    PickCancelled,
    PickFlowError,
    PreparedCandidate,
    RankedCandidate,
)
from manipulation_execution.so101_geometry import axis_error_deg, gripper_mesh_min_z, tabletop_clearance

JOINT5_ORIENTATION_MAX_CLOSING_ERROR_DEG = 20.0


class PreparationPhase:
    """Convert ranked source candidates into executable target configurations."""

    def _solve_ik(
        self,
        pose: Pose,
        goal_handle,
        deadline: float,
        seed: JointState | None = None,
        *,
        client=None,
    ) -> JointState:
        client = self._ik_client if client is None else client
        ik_config = self._config.get("ik", {})
        request = GetPositionIK.Request()
        request.ik_request.group_name = str(ik_config.get("group_name", "arm"))
        request.ik_request.ik_link_name = self._ee_frame
        request.ik_request.pose_stamped.header.frame_id = self._base_frame
        request.ik_request.pose_stamped.pose = pose
        request.ik_request.avoid_collisions = bool(ik_config.get("avoid_collisions", False))
        if seed is not None:
            request.ik_request.robot_state.joint_state = seed
        ik_timeout = float(ik_config.get("timeout_sec", 2.0))
        request.ik_request.timeout = Duration(seconds=ik_timeout).to_msg()
        future = client.call_async(request)
        response = self._wait_future(future, goal_handle, deadline, ik_timeout + 1.0, "IK")
        if int(response.error_code.val) != 1:
            raise PickFlowError("IK_FAILED", f"IK failed with code {response.error_code.val}", retryable=True)
        return response.solution.joint_state

    def _compute_fk(self, joint_state: JointState, goal_handle, deadline: float, *, client=None) -> Pose:
        client = self._fk_client if client is None else client
        request = GetPositionFK.Request()
        request.header.frame_id = self._base_frame
        request.fk_link_names = [self._ee_frame]
        request.robot_state.joint_state = joint_state
        future = client.call_async(request)
        response = self._wait_future(future, goal_handle, deadline, self._rpc_timeout, "FK")
        if int(response.error_code.val) != 1 or not response.pose_stamped:
            raise PickFlowError("FK_FAILED", f"FK failed with code {response.error_code.val}", retryable=True)
        return response.pose_stamped[0].pose

    def _orientation_guard(self) -> dict:
        target_gripper = self._config.get("target_gripper", {})
        guard = target_gripper.get("ik_orientation_guard", {})
        return guard if isinstance(guard, dict) else {}

    def _joint5_abs_max(self) -> float | None:
        value = self._orientation_guard().get("joint5_abs_max")
        if value is None:
            return None
        limit = float(value)
        if not math.isfinite(limit) or limit <= 0.0:
            raise PickFlowError("INVALID_GRASP_CONFIG", "ik_orientation_guard.joint5_abs_max must be positive")
        return limit

    def _validate_joint5(self, joint_state: JointState) -> float | None:
        limit = self._joint5_abs_max()
        if limit is None:
            return None
        value = self._joint_position(joint_state, "5")
        if value is None:
            raise PickFlowError("IK_JOINT5_MISSING", "IK result has no joint 5", retryable=True)
        if abs(value) > limit:
            raise PickFlowError(
                "IK_JOINT5_LIMIT",
                f"joint 5 absolute value {abs(value):.4f} exceeds {limit:.4f}",
                retryable=True,
            )
        return value

    def _apply_joint5_retry_if_needed(
        self,
        pose: Pose,
        solution: JointState,
        goal_handle,
        deadline: float,
        *,
        ik_client=None,
    ) -> tuple[JointState, float | None]:
        limit = self._joint5_abs_max()
        if limit is None:
            return solution, None
        original_joint5 = self._joint_position(solution, "5")
        if original_joint5 is None:
            return solution, None
        if abs(original_joint5) <= math.pi / 2.0:
            return solution, None

        bounded_joint5 = canonicalize_joint5(original_joint5)
        retry_seed = self._joint_state_with_joint5(solution, bounded_joint5)
        retry_solution = self._solve_ik(pose, goal_handle, deadline, retry_seed, client=ik_client)
        retry_joint5 = self._joint_position(retry_solution, "5")
        if retry_joint5 is None or abs(retry_joint5) > limit:
            raise PickFlowError(
                "IK_JOINT5_RETRY_FAILED",
                f"joint 5 retry did not enter limit {limit:.4f}: {retry_joint5}",
                retryable=True,
            )
        return retry_solution, original_joint5

    def _grasp_orientation_errors(
        self,
        target_quaternion: tuple[float, float, float, float],
        actual_quaternion: tuple[float, float, float, float],
    ):
        guard = self._orientation_guard()
        if not bool(guard.get("enabled", False)):
            return None
        target_gripper = self._config.get("target_gripper", {})
        return grasp_axis_errors(
            target_quaternion,
            actual_quaternion,
            guard.get("approach_axis_ee", [0.0, 0.0, 1.0]),
            target_gripper.get("closing_axis_ee", [1.0, 0.0, 0.0]),
            closing_axis_180_symmetric=bool(guard.get("closing_axis_180_symmetric", False)),
        )

    def _orientation_limits(self) -> tuple[float, float]:
        guard = self._orientation_guard()
        max_approach = float(guard.get("max_approach_error_deg", 25.0))
        max_closing = min(
            float(guard.get("max_closing_error_deg", JOINT5_ORIENTATION_MAX_CLOSING_ERROR_DEG)),
            JOINT5_ORIENTATION_MAX_CLOSING_ERROR_DEG,
        )
        if not all(math.isfinite(value) and 0.0 <= value <= 180.0 for value in (max_approach, max_closing)):
            raise PickFlowError(
                "INVALID_GRASP_CONFIG",
                "ik_orientation_guard axis error limits must be within [0, 180] degrees",
            )
        return max_approach, max_closing

    def _solve_grasp_ik_fk(
        self,
        pose: Pose,
        goal_handle,
        deadline: float,
        seed: JointState | None = None,
        *,
        validate_orientation: bool = True,
        ik_client=None,
        fk_client=None,
    ) -> IKPayload:
        joint_state = self._solve_ik(pose, goal_handle, deadline, seed, client=ik_client)
        joint_state, original_joint5 = self._apply_joint5_retry_if_needed(
            pose,
            joint_state,
            goal_handle,
            deadline,
            ik_client=ik_client,
        )
        self._validate_joint5(joint_state)
        fk_pose = self._compute_fk(joint_state, goal_handle, deadline, client=fk_client)
        ee_xyz, ee_quaternion = self._pose_components(fk_pose)
        _, target_quaternion = self._pose_components(pose)
        errors = self._grasp_orientation_errors(target_quaternion, ee_quaternion)
        approach_error = None if errors is None else errors.approach_deg
        closing_error = None if errors is None else errors.closing_deg
        if validate_orientation and errors is not None:
            max_approach, max_closing = self._orientation_limits()
            if errors.approach_deg > max_approach or errors.closing_deg > max_closing:
                raise PickFlowError(
                    "IK_ORIENTATION_REJECTED",
                    f"FK orientation error exceeds limits: approach={errors.approach_deg:.3f}/{max_approach:.3f} "
                    f"closing={errors.closing_deg:.3f}/{max_closing:.3f}",
                    retryable=True,
                )
        return IKPayload(
            joint_state=joint_state,
            ee_xyz=ee_xyz,
            ee_quaternion=ee_quaternion,
            joint5_retry_applied=original_joint5 is not None,
            original_joint5=original_joint5,
            approach_axis_error_deg=approach_error,
            closing_axis_error_deg=closing_error,
        )

    def _solve_orientation_consistent_grasp_ik_fk(
        self,
        pose: Pose,
        goal_handle,
        deadline: float,
        seed: JointState | None = None,
        *,
        ik_client=None,
        fk_client=None,
    ) -> IKPayload:
        guard = self._orientation_guard()
        if not bool(guard.get("enabled", False)):
            return self._solve_grasp_ik_fk(
                pose,
                goal_handle,
                deadline,
                seed,
                ik_client=ik_client,
                fk_client=fk_client,
            )

        _, target_quaternion = self._pose_components(pose)
        target_gripper = self._config.get("target_gripper", {})
        max_approach, max_closing = self._orientation_limits()
        current_seed = seed
        seen_joint5: set[float] = set()
        last_reason = "orientation correction was not attempted"
        for attempt in range(3):
            payload = self._solve_grasp_ik_fk(
                pose,
                goal_handle,
                deadline,
                current_seed,
                validate_orientation=False,
                ik_client=ik_client,
                fk_client=fk_client,
            )
            joint5 = self._joint_position(payload.joint_state, "5")
            approach_error = payload.approach_axis_error_deg
            closing_error = payload.closing_axis_error_deg
            if joint5 is None or approach_error is None or closing_error is None:
                last_reason = "orientation correction requires joint 5 and FK axis errors"
                break
            if approach_error <= max_approach and closing_error <= max_closing:
                return payload
            last_reason = (
                f"attempt={attempt} approach={approach_error:.3f}/{max_approach:.3f} "
                f"closing={closing_error:.3f}/{max_closing:.3f}"
            )
            try:
                correction = joint5_closing_axis_correction(
                    target_quaternion,
                    payload.ee_quaternion,
                    guard.get("approach_axis_ee", [0.0, 0.0, 1.0]),
                    target_gripper.get("closing_axis_ee", [1.0, 0.0, 0.0]),
                    closing_axis_180_symmetric=bool(guard.get("closing_axis_180_symmetric", False)),
                )
            except ValueError as exc:
                last_reason = str(exc)
                break
            corrected_joint5 = canonicalize_joint5(joint5 + correction)
            correction_key = round(corrected_joint5, 9)
            if correction_key in seen_joint5 or abs(corrected_joint5 - joint5) <= 1e-6:
                break
            seen_joint5.add(round(joint5, 9))
            current_seed = self._joint_state_with_joint5(payload.joint_state, corrected_joint5)
        raise PickFlowError(
            "IK_ORIENTATION_REJECTED",
            f"no orientation-consistent joint 5 branch: {last_reason}",
            retryable=True,
        )

    def _validate_joint5_branch_continuity(self, seed: JointState | None, solution: JointState) -> None:
        if seed is None or self._joint5_abs_max() is None:
            return
        seed_joint5 = self._joint_position(seed, "5")
        solution_joint5 = self._joint_position(solution, "5")
        if seed_joint5 is None or solution_joint5 is None:
            return
        delta = abs(solution_joint5 - seed_joint5)
        if delta > math.pi / 2.0:
            raise PickFlowError(
                "IK_JOINT5_BRANCH_CHANGED",
                f"joint 5 branch changed by {delta:.4f} rad ({seed_joint5:.4f} -> {solution_joint5:.4f})",
                retryable=True,
            )

    def _validate_fk_fixed_finger_base_side(
        self,
        candidate_index: int,
        plan: CandidatePlan,
        payload: IKPayload,
    ) -> FixedFingerBaseSide | None:
        target_gripper = self._config.get("target_gripper", {})
        base_side_config = target_gripper.get("fixed_finger_base_side", {})
        if not (
            bool(base_side_config.get("enabled", False))
            and str(target_gripper.get("type", "")) == "asymmetric_single_moving_jaw"
        ):
            return None
        if plan.target_width_min_base is None or plan.target_width_max_base is None:
            raise PickFlowError(
                "FK_FIXED_FINGER_BASE_SIDE_UNAVAILABLE",
                f"candidate {candidate_index}: target width extent is unavailable for final FK fixed-finger check",
                retryable=True,
            )
        try:
            alignment = fixed_finger_base_side_alignment(
                payload.ee_xyz,
                payload.ee_quaternion,
                target_gripper.get("fixed_finger_contact_ee", [-0.014, 0.0, -0.080]),
                plan.target_width_min_base,
                plan.target_width_max_base,
                base_side_config.get("reference_point_base", [0.0, 0.0, 0.0]),
            )
        except ValueError as exc:
            raise PickFlowError(
                "FK_FIXED_FINGER_BASE_SIDE_UNAVAILABLE",
                f"candidate {candidate_index}: final FK fixed-finger check failed: {exc}",
                retryable=True,
            ) from exc
        minimum_alignment = max(-1.0, min(1.0, float(base_side_config.get("min_alignment_cos", 0.0))))
        if alignment.alignment_cos < minimum_alignment:
            raise PickFlowError(
                "FK_FIXED_FINGER_BASE_SIDE_REJECTED",
                f"candidate {candidate_index}: final FK fixed finger is on the outer side "
                f"(alignment={alignment.alignment_cos:.3f} < {minimum_alignment:.3f}, "
                f"inward_offset={alignment.inward_offset_m:.4f}m)",
                retryable=True,
            )
        return alignment

    def _prepare_candidate(
        self,
        ranked: RankedCandidate,
        scene_base: BaseSceneGeometry,
        goal_handle,
        deadline: float,
        *,
        apply_compensation: bool = False,
        initial_seed: JointState | None = None,
        ik_client=None,
        fk_client=None,
    ) -> PreparedCandidate:
        plan = ranked.plan
        compensation = self._config.get("contact_compensation", {})
        payload: IKPayload
        contact_residual_xy = 0.0
        z_error = 0.0
        compensation_enabled = bool(compensation.get("enabled", True))
        if compensation_enabled and apply_compensation:

            def _predict(command_xyz, previous_payload):
                seed = previous_payload.joint_state if previous_payload is not None else initial_seed
                command_pose = self._pose(command_xyz, plan.quaternion)
                predicted = self._solve_grasp_ik_fk(
                    command_pose,
                    goal_handle,
                    deadline,
                    seed,
                    ik_client=ik_client,
                    fk_client=fk_client,
                )
                return ContactPrediction(
                    contact_base=self._contact_for_pose(
                        self._pose(predicted.ee_xyz, predicted.ee_quaternion),
                        plan.target_contact_ee,
                    ),
                    payload=predicted,
                )

            result = compensate_contact_xy(
                plan.grasp,
                plan.target_contact_base,
                _predict,
                tolerance_m=float(compensation.get("xy_tolerance_m", 0.003)),
                max_iterations=int(compensation.get("max_iterations", 6)),
                max_correction_m=float(compensation.get("max_correction_m", 0.03)),
            )
            if not result.converged:
                raise PickFlowError(
                    "CONTACT_COMPENSATION_FAILED",
                    f"candidate {ranked.index}: {result.reason}",
                    retryable=True,
                )
            z_error = float(plan.target_contact_base[2] - result.prediction.contact_base[2])
            selection_min_contact_z = float(self._config.get("candidate_selection", {}).get("min_contact_z", 0.0))
            if result.prediction.contact_base[2] < selection_min_contact_z:
                raise PickFlowError(
                    "IK_FK_PREDICTED_CONTACT_Z",
                    f"candidate {ranked.index}: predicted contact z {result.prediction.contact_base[2]:.4f} "
                    f"< min_contact_z {selection_min_contact_z:.4f}",
                    retryable=True,
                )
            correction_x = result.correction_x
            correction_y = result.correction_y
            plan = replace(
                plan,
                grasp=result.command_xyz,
                lift=(plan.lift[0] + correction_x, plan.lift[1] + correction_y, plan.lift[2]),
            )
            payload = result.prediction.payload
            contact_residual_xy = math.hypot(float(result.residual_x), float(result.residual_y))
        else:
            payload = self._solve_orientation_consistent_grasp_ik_fk(
                self._pose(plan.grasp, plan.quaternion),
                goal_handle,
                deadline,
                initial_seed,
                ik_client=ik_client,
                fk_client=fk_client,
            )
            predicted_contact = self._contact_for_pose(
                self._pose(payload.ee_xyz, payload.ee_quaternion),
                plan.target_contact_ee,
            )
            contact_residual_xy = math.hypot(
                plan.target_contact_base[0] - predicted_contact[0],
                plan.target_contact_base[1] - predicted_contact[1],
            )
            z_error = float(plan.target_contact_base[2] - predicted_contact[2])

            if compensation_enabled:
                max_correction = max(0.0, float(compensation.get("max_correction_m", 0.03)))
                residual_x = float(plan.target_contact_base[0] - predicted_contact[0])
                residual_y = float(plan.target_contact_base[1] - predicted_contact[1])
                if abs(residual_x) > max_correction or abs(residual_y) > max_correction:
                    raise PickFlowError(
                        "CONTACT_COMPENSATION_FAILED",
                        f"candidate {ranked.index}: contact residual x={residual_x:.4f} y={residual_y:.4f} "
                        f"exceeds {max_correction:.4f}",
                        retryable=True,
                    )

        if compensation_enabled and abs(z_error) > float(compensation.get("max_z_error_m", 0.015)):
            raise PickFlowError(
                "CONTACT_Z_ERROR",
                f"candidate {ranked.index}: contact z error {z_error:.4f}",
                retryable=True,
            )

        fk_fixed_finger_base_side = self._validate_fk_fixed_finger_base_side(ranked.index, plan, payload)

        check_orientation = bool(self._config.get("ik", {}).get("check_orientation", False))
        position_only_quaternion = (0.0, 0.0, 0.0, 1.0)
        for label, xyz in (("approach", plan.approach), ("lift", plan.lift)):
            allowed, reason = xyz_within_workspace(xyz, self._workspace)
            if not allowed:
                raise PickFlowError("WORKSPACE_REJECTED", f"candidate {ranked.index} {label}: {reason}", retryable=True)
            ik_quaternion = plan.quaternion if check_orientation else position_only_quaternion
            self._solve_ik(
                self._pose(xyz, ik_quaternion),
                goal_handle,
                deadline,
                initial_seed,
                client=ik_client,
            )

        closing_axis = self._config.get("target_gripper", {}).get("closing_axis_ee", [1.0, 0.0, 0.0])
        closing_error = payload.closing_axis_error_deg
        if closing_error is None:
            closing_error = axis_error_deg(plan.quaternion, payload.ee_quaternion, closing_axis)

        mesh_min_z = None
        actual_tabletop_clearance = None
        if self._mesh_directory is not None:
            try:
                mesh_min_z = gripper_mesh_min_z(
                    self._mesh_directory,
                    payload.ee_xyz,
                    payload.ee_quaternion,
                    float(ranked.candidate.target_width_m),
                )
                if bool(self._target_geometry.get("tabletop_filter", False)):
                    if scene_base.table_plane is None:
                        raise PickFlowError("TARGET_TABLETOP_UNAVAILABLE", "fitted table plane is unavailable")
                    actual_tabletop_clearance = tabletop_clearance(
                        self._mesh_directory,
                        payload.ee_xyz,
                        payload.ee_xyz,
                        payload.ee_quaternion,
                        float(ranked.candidate.target_width_m),
                        scene_base.table_plane,
                        sweep_steps=1,
                    )
            except PickFlowError:
                raise
            except Exception as exc:
                raise PickFlowError("TARGET_GEOMETRY_FAILED", str(exc), retryable=True) from exc
            minimum_clearance = float(self._target_geometry.get("tabletop_clearance_m", 0.0))
            if actual_tabletop_clearance is not None and actual_tabletop_clearance < minimum_clearance:
                raise PickFlowError(
                    "TARGET_TABLETOP_COLLISION",
                    f"candidate {ranked.index}: FK tabletop clearance {actual_tabletop_clearance:.4f}m "
                    f"< {minimum_clearance:.4f}m",
                    retryable=True,
                )

        prepared_scoring = self._config.get("prepared_candidate_scoring", {})
        envelope = None
        target_gripper = self._config.get("target_gripper", {})
        robust_gap_config = target_gripper.get("fixed_finger_robust_gap", {})
        if (
            (bool(prepared_scoring.get("enabled", False)) or bool(robust_gap_config.get("enabled", False)))
            and str(target_gripper.get("type", "")) == "asymmetric_single_moving_jaw"
            and plan.target_width_min_base is not None
            and plan.target_width_max_base is not None
        ):
            envelope = fixed_finger_envelope_score(
                plan.grasp,
                plan.quaternion,
                target_gripper.get("fixed_finger_contact_ee", [-0.014, 0.0, -0.080]),
                target_gripper.get("closing_axis_ee", [1.0, 0.0, 0.0]),
                plan.target_width_min_base,
                plan.target_width_max_base,
                plan.fixed_finger_target_gap_m,
                gap_sigma_m=float(prepared_scoring.get("fixed_finger_gap_sigma_m", 0.006)),
                reliable_max_opening_m=float(prepared_scoring.get("reliable_max_opening_m", 0.072)),
                moving_min_clearance_m=float(prepared_scoring.get("moving_finger_min_clearance_m", 0.003)),
                fixed_score_weight=float(prepared_scoring.get("fixed_finger_score_weight", 0.80)),
            )
        selection_score = prepared_candidate_soft_score(
            prepared_scoring,
            fixed_finger_envelope=None if envelope is None else envelope.score,
            contact_residual_xy_m=contact_residual_xy,
            contact_z_error_m=abs(z_error),
            confidence=float(ranked.candidate.confidence),
            centroid_distance_m=ranked.contact_distance_m,
        )

        return PreparedCandidate(
            ranked=ranked,
            plan=plan,
            final_joint_state=payload.joint_state,
            actual_ee_xyz=payload.ee_xyz,
            actual_ee_quaternion=payload.ee_quaternion,
            contact_residual_xy_m=contact_residual_xy,
            contact_z_error_m=abs(z_error),
            approach_axis_error_deg=payload.approach_axis_error_deg,
            closing_axis_error_deg=closing_error,
            tabletop_clearance_m=actual_tabletop_clearance,
            mesh_min_z=mesh_min_z,
            fixed_finger_envelope=envelope,
            fk_fixed_finger_base_side=fk_fixed_finger_base_side,
            selection_score=selection_score,
        )

    def _verify_ik_worker_pool(
        self,
        candidate: RankedCandidate,
        joint_seed: JointState,
        goal_handle,
        deadline: float,
    ) -> None:
        if not self._ik_worker_clients:
            return
        check_orientation = bool(self._config.get("ik", {}).get("check_orientation", False))
        quaternion = candidate.plan.quaternion if check_orientation else (0.0, 0.0, 0.0, 1.0)
        pose = self._pose(candidate.plan.approach, quaternion)
        primary = self._solve_ik(pose, goal_handle, deadline, joint_seed)
        worker = self._solve_ik(
            pose,
            goal_handle,
            deadline,
            joint_seed,
            client=self._ik_worker_clients[0],
        )
        primary_positions = dict(zip(primary.name, primary.position, strict=False))
        worker_positions = dict(zip(worker.name, worker.position, strict=False))
        common_names = sorted(primary_positions.keys() & worker_positions.keys())
        if not common_names:
            raise PickFlowError("IK_WORKER_MISMATCH", "primary and worker IK returned no common joints")
        max_delta = max(abs(float(primary_positions[name]) - float(worker_positions[name])) for name in common_names)
        if max_delta > 1e-8:
            raise PickFlowError(
                "IK_WORKER_MISMATCH",
                f"primary and worker IK differ by {max_delta:.12f} rad",
            )
        self.get_logger().info(
            f"IK worker verification passed: workers={self._ik_worker_count} max_joint_delta={max_delta:.12f}"
        )

    def _prepare_ranked_candidates(
        self,
        ranked: list[RankedCandidate],
        scene_base: BaseSceneGeometry,
        joint_seed: JointState,
        goal_handle,
        deadline: float,
    ) -> tuple[list[PreparedCandidate], PickFlowError | None]:
        started = time.monotonic()
        if not self._ik_worker_clients:
            results: list[PreparedCandidate | PickFlowError] = []
            for candidate in ranked:
                try:
                    results.append(
                        self._prepare_candidate(
                            candidate,
                            scene_base,
                            goal_handle,
                            deadline,
                            initial_seed=joint_seed,
                        )
                    )
                except PickFlowError as exc:
                    results.append(exc)
        else:
            self._verify_ik_worker_pool(ranked[0], joint_seed, goal_handle, deadline)
            worker_count = min(len(self._ik_worker_clients), len(ranked))
            partitions: list[list[tuple[int, RankedCandidate]]] = [[] for _ in range(worker_count)]
            for position, candidate in enumerate(ranked):
                partitions[position % worker_count].append((position, candidate))

            ordered_results: list[PreparedCandidate | PickFlowError | None] = [None] * len(ranked)

            def prepare_partition(worker_index: int, partition: list[tuple[int, RankedCandidate]]):
                partition_results = []
                for position, candidate in partition:
                    try:
                        result = self._prepare_candidate(
                            candidate,
                            scene_base,
                            goal_handle,
                            deadline,
                            initial_seed=joint_seed,
                            ik_client=self._ik_worker_clients[worker_index],
                            fk_client=self._fk_worker_clients[worker_index],
                        )
                    except PickFlowError as exc:
                        result = exc
                    partition_results.append((position, result))
                return partition_results

            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="pick-candidate-ik") as pool:
                jobs = [pool.submit(prepare_partition, index, partition) for index, partition in enumerate(partitions)]
                for job in jobs:
                    for position, result in job.result():
                        ordered_results[position] = result
            if any(result is None for result in ordered_results):
                raise PickFlowError("IK_WORKER_INCOMPLETE", "parallel IK worker pool returned incomplete results")
            results = [result for result in ordered_results if result is not None]
            self.get_logger().info(
                f"PIPELINE_TIMING stage=candidate_ik_fk duration_s={time.monotonic() - started:.3f} "
                f"workers={worker_count} candidates={len(ranked)}"
            )

        if not self._ik_worker_clients:
            self.get_logger().info(
                f"PIPELINE_TIMING stage=candidate_ik_fk duration_s={time.monotonic() - started:.3f} "
                f"workers=0 candidates={len(ranked)}"
            )

        prepared_candidates: list[PreparedCandidate] = []
        last_error: PickFlowError | None = None
        for candidate, result in zip(ranked, results, strict=True):
            if isinstance(result, PickCancelled):
                raise result
            if isinstance(result, PickFlowError):
                last_error = result
                self.get_logger().warning(
                    f"pick candidate preparation failed: candidate={candidate.index} "
                    f"code={result.code} retryable={result.retryable} message={result}"
                )
                if not result.retryable:
                    raise result
                continue
            prepared_candidates.append(result)
        return prepared_candidates, last_error

    def _record_prepared_ranking(self, state: FlowState, candidates: list[PreparedCandidate]) -> None:
        if not state.debug_output_dir:
            return
        records = []
        for rank, item in enumerate(candidates, start=1):
            envelope = item.fixed_finger_envelope
            records.append(
                {
                    "rank": rank,
                    "candidate_index": item.ranked.index,
                    "selection_score": item.selection_score,
                    "fixed_finger_envelope_score": None if envelope is None else envelope.score,
                    "fixed_finger_gap_score": None if envelope is None else envelope.fixed_score,
                    "fixed_finger_gap_m": None if envelope is None else envelope.fixed_gap_m,
                    "fixed_finger_target_gap_m": None if envelope is None else envelope.target_gap_m,
                    "moving_finger_gap_m": None if envelope is None else envelope.moving_gap_m,
                    "moving_finger_gap_score": None if envelope is None else envelope.moving_score,
                    "contact_residual_xy_m": item.contact_residual_xy_m,
                    "contact_z_error_m": item.contact_z_error_m,
                    "ik_fk_approach_axis_error_deg": item.approach_axis_error_deg,
                    "ik_fk_closing_axis_error_deg": item.closing_axis_error_deg,
                    "ik_grasp_joint5": self._joint_position(item.final_joint_state, "5"),
                    "centroid_distance_m": item.ranked.contact_distance_m,
                    "confidence": float(item.ranked.candidate.confidence),
                    "source_rank_score": item.ranked.score,
                    "target_width_m": float(item.ranked.candidate.target_width_m),
                    "target_width_quality": float(item.ranked.candidate.target_width_quality),
                    "target_width_min_offset_m": float(item.ranked.candidate.target_width_min_offset_m),
                    "target_width_max_offset_m": float(item.ranked.candidate.target_width_max_offset_m),
                    "fixed_finger_base_side_alignment_cos": (
                        None
                        if item.ranked.fixed_finger_base_side is None
                        else item.ranked.fixed_finger_base_side.alignment_cos
                    ),
                    "fixed_finger_inward_offset_m": (
                        None
                        if item.ranked.fixed_finger_base_side is None
                        else item.ranked.fixed_finger_base_side.inward_offset_m
                    ),
                    "fk_fixed_finger_base_side_alignment_cos": (
                        None if item.fk_fixed_finger_base_side is None else item.fk_fixed_finger_base_side.alignment_cos
                    ),
                    "fk_fixed_finger_inward_offset_m": (
                        None
                        if item.fk_fixed_finger_base_side is None
                        else item.fk_fixed_finger_base_side.inward_offset_m
                    ),
                }
            )
        output_path = Path(state.debug_output_dir) / "prepared_candidate_ranking.json"
        try:
            output_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            self.get_logger().warning(f"failed to write {output_path}: {exc}")
