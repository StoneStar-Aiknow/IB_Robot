"""Execute an exact configured joint target and check fresh arrival feedback."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager

from pose_composer.errors import ComposeError
from pose_composer.execution_backend import GeneratedBackend
from robot_runtime.task_poses import normalize_task_poses, pose_hand_sides, task_pose_target


class TaskPoseRunner:
    def __init__(self, snapshot, *, simulation=False, clock=time.monotonic, pause=time.sleep, backend=None):
        self.backend = backend or GeneratedBackend()
        self.snapshot = normalize_task_poses(snapshot) if snapshot else {}
        self.simulation = simulation
        self.clock, self.pause = clock, pause
        self.report = {}
        if self.snapshot and tuple(self.snapshot["joint_names"]) != tuple(self.backend.model.ARM_JOINTS):
            raise ValueError("task-pose joint order does not match X2 backend")
        for pose in self.snapshot.get("poses", {}).values():
            if not self.backend.safety.check_pose(pose["joint_positions"]).ok:
                raise ValueError("configured task pose violates X2 target safety")

    def execute(self, *args, **kwargs):
        """Keep timing and partial completion evidence on success, cancellation and failure."""
        self.report = {"phase_sec": {}, "stages": {}, "completed_hands": [], "skipped_hands": []}
        for name in (
            "preflight",
            "takeover",
            "hand_preflight_after_takeover",
            "planning",
            "moving",
            "arm_settling",
            "opening_left",
            "opening_right",
            "confirming",
        ):
            self.report["stages"][name] = {"status": "skipped", "sec": 0.0, "reason": "stage not reached"}
        started = self.clock()
        try:
            return self._execute(*args, **kwargs)
        finally:
            self.report["total_sec"] = self.clock() - started

    @contextmanager
    def _stage(self, name, progress):
        started = self.clock()
        item = self.report["stages"][name] = {"status": "executed", "sec": 0.0, "reason": ""}
        progress(name, 0.0)
        try:
            yield
            progress(name, 1.0)
        except Exception:
            item["status"] = "failed"
            raise
        finally:
            self.report["phase_sec"][name] = self.clock() - started
            item["sec"] = self.report["phase_sec"][name]

    def _execute(
        self,
        name,
        digest,
        *,
        read_state,
        move,
        cancelled,
        timeout_sec=0.0,
        motion_limits=(0.0, 0.0),
        hands=None,
        gateway_task_id="",
        progress=lambda *_: None,
    ):
        """Reach the arm target, then open the hands the pose declares, under one deadline and one cancel.

        ``hands`` builds the restorer (``aimdk_grasp.hand_restore.HandRestorer``) and is only called for a pose
        that declares hands; a pose without them takes exactly the path it always did.
        """
        try:
            pose = task_pose_target(self.snapshot, name, digest)
        except ValueError as exc:
            raise ComposeError("TASK_POSE_CONFIG_MISMATCH", str(exc)) from exc
        if not self.simulation and not pose["hardware_approved"]:
            raise ComposeError("TASK_POSE_NOT_APPROVED", "task pose is not approved for hardware")
        sides = pose_hand_sides(pose)
        for side in ("left", "right"):
            if side not in sides:
                self.report["stages"][f"opening_{side}"]["reason"] = "hand action not requested"
        if not sides:
            for name in ("preflight", "takeover", "hand_preflight_after_takeover", "confirming"):
                self.report["stages"][name]["reason"] = "hand action not requested"
        if sides and not self.simulation and not pose["unloaded_operation_approved"]:
            raise ComposeError(
                "TASK_POSE_HANDS_NOT_APPROVED", "opening hands needs the pose's explicit unloaded-operation approval"
            )
        if isinstance(timeout_sec, bool) or not math.isfinite(timeout_sec) or timeout_sec < 0:
            raise ComposeError("TASK_POSE_BAD_TIMEOUT", "invalid task-pose timeout")
        deadline = self.clock() + min(pose["timeout_sec"], timeout_sec or pose["timeout_sec"])
        target = pose["joint_positions"]
        observed_stamp = 0.0

        def check():
            nonlocal observed_stamp
            if cancelled():
                raise ComposeError("POSE_CANCELLED", "task-pose preparation canceled")
            if self.clock() >= deadline:
                raise ComposeError("TASK_POSE_TIMEOUT", "task pose did not settle within the budget")
            q, stamps = read_state()
            if (
                set(q) != set(target)
                or any(not math.isfinite(q[j]) for j in target)
                or any(
                    j not in stamps
                    or not math.isfinite(stamps[j])
                    or not 0 <= self.clock() - stamps[j] <= pose["max_state_age_sec"]
                    for j in target
                )
            ):
                raise ComposeError("TASK_POSE_STALE_STATE", "arm feedback is incomplete, non-finite or stale")
            observed_stamp = min(stamps[j] for j in target)
            return q

        def at_target(q):
            return (
                max(abs(q[j] - target[j]) for j in target) <= pose["position_tolerance_rad"]
                and self.backend.safety.check_pose(q).ok
            )

        start = check()
        skipped = at_target(start)
        restorer = None
        context = hand_seconds = None
        if sides:
            from aimdk_grasp.hand_restore import RestoreContext

            context = RestoreContext(gateway_task_id, deadline, cancelled, progress)
            with self._stage("preflight", progress):
                restorer = self._restorer(hands)
                hand_seconds = self._hand_call(
                    lambda: restorer.preflight(context, sides, arm_target=target, arm_must_move=not skipped)
                )
            if hand_seconds >= deadline - self.clock():
                raise ComposeError("TASK_POSE_TIMEOUT", "the hand phase alone exceeds the task-pose budget")
            with self._stage("takeover", progress):
                self._hand_call(lambda: restorer.begin(context))
            # Takeover may have taken time. Always plan from a new measured arm pose.
            start = check()
            skipped = at_target(start)
            if not skipped:
                with self._stage("hand_preflight_after_takeover", progress):
                    self._hand_call(lambda: restorer.preflight(context, sides, arm_target=target, arm_must_move=True))
        if skipped:
            for name in ("planning", "moving"):
                self.report["stages"][name]["reason"] = "arms already at target"
        if not skipped:
            with self._stage("planning", progress):
                planner = getattr(self.backend, "plan_for_execution", self.backend.plan_safe)
                if hasattr(self.backend, "guard"):
                    previous_guard, self.backend.guard = self.backend.guard, check
                    try:
                        plan = planner(start, [target], [0.0])
                    finally:
                        self.backend.guard = previous_guard
                else:
                    plan = planner(start, [target], [0.0])
            if not plan.ok:
                raise ComposeError("POSE_NO_SAFE_PATH", "no safe path to configured task pose")
            reason = self.backend.check_displacement(plan, start, *motion_limits)
            if reason:
                raise ComposeError("POSE_MOTION_LIMIT", reason)
            check()
            if plan.traj.times[-1] + pose["stable_sec"] + (hand_seconds or 0.0) >= deadline - self.clock():
                raise ComposeError("TASK_POSE_TIMEOUT", "planned preparation exceeds its budget")
            with self._stage("moving", progress):
                ok, code, message = move(plan, deadline)
            if not ok:
                raise ComposeError(code or "TASK_POSE_MOVE_FAILED", message)
        # Temporary commissioning policy shared by fixed free_pose and task poses:
        # do not wait for a continuous stability window after trajectory terminal.
        # This is still a measured arrival check, not unconditional success.
        with self._stage("arm_settling", progress):
            q = check()
            worst_joint = max(target, key=lambda joint: abs(q[joint] - target[joint]))
            error = abs(q[worst_joint] - target[worst_joint])
            self.report["arrival_check"] = {
                "policy": "single_sample",
                "stable_window_verified": False,
                "worst_joint": worst_joint,
                "max_error_rad": error,
                "tolerance_rad": pose["position_tolerance_rad"],
            }
            if error > pose["position_tolerance_rad"]:
                raise ComposeError(
                    "TASK_POSE_NOT_REACHED",
                    f"trajectory finished but arrival not confirmed: {worst_joint} error={error:.6f} rad "
                    f"exceeds {pose['position_tolerance_rad']:.6f} rad",
                )
            safe = self.backend.safety.check_pose(q).ok
            check()  # Observe cancellation/deadline and feedback age after the synchronous check.
            if not safe:
                raise ComposeError("TASK_POSE_UNSAFE_STATE", "measured arrival failed the pose safety check")
        if restorer is None:
            return skipped

        def arms_still_there():
            if not at_target(check()):
                raise ComposeError("TASK_POSE_NOT_REACHED", "the arms left the task pose while the hands were prepared")

        for side in sides:
            arms_still_there()
            with self._stage(f"opening_{side}", progress):
                try:
                    hand_skipped = self._hand_call(lambda side=side: restorer.open_hand(context, side))
                finally:  # which ways the hand was opened (aimdk_grasp.omnihand.OPEN_PATHS)
                    taken = getattr(restorer, "open_paths_taken", {}).get(side)
                    if taken:
                        self.report["stages"][f"opening_{side}"]["paths"] = list(taken)
            self.report["completed_hands"].append(side)
            if hand_skipped:
                self.report["skipped_hands"].append(side)
        with self._stage("confirming", progress):
            self._hand_call(lambda: restorer.confirm(context, sides, arms_still_there))
        return skipped

    @staticmethod
    def _restorer(hands):
        if hands is None:
            raise ComposeError("HAND_CAPABILITY_MISSING", "this deployment has no hand-restore capability")
        try:
            restorer = hands()
        except ComposeError:
            raise
        except Exception as exc:  # a missing model, bundle or interface: refuse, never approximate
            raise ComposeError("HAND_CAPABILITY_MISSING", f"hand restore is unavailable: {exc}") from exc
        if restorer is None:
            raise ComposeError("HAND_CAPABILITY_MISSING", "this deployment has no hand-restore capability")
        return restorer

    @staticmethod
    def _hand_call(call):
        from aimdk_grasp.hand_restore import RestoreError

        try:
            return call()
        except RestoreError as exc:
            raise ComposeError(exc.code, str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise ComposeError("HAND_EXECUTION_FAILED", str(exc)) from exc
