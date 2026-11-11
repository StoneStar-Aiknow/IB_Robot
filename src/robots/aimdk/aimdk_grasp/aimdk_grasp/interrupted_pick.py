"""X2-only interrupted pick custody. Stop means no further motion, never an undo.

A confirmed grasp may retrace its lift and release at the original contact,
after new geometry and support checks. An incomplete hand action requires
manual intervention. Old collision certificates never authorize replay.
"""

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from manipulation_execution.grasp_pipeline.contracts import Effect, EffectorAction, GraspFailure, MotionState, Move

PICK_INTERRUPTED = "PICK_INTERRUPTED"
PICK_MANUAL_CHECK = "PICK_MANUAL_CHECK"
POSE_TOLERANCE_RAD = 0.02


@dataclass(frozen=True)
class Interruption:
    nonce: str
    state: str
    reason: str


class InterruptedPick:
    def __init__(self, executor, effector, posture, validate, *, clock=time.monotonic, state_path=None):
        self.executor, self.effector = executor, effector
        self.posture, self.validate, self.clock = posture, validate, clock
        self._lock = threading.RLock()
        self._record = None
        self._state_path = Path(state_path) if state_path is not None else None
        self._home = None
        self._home_joints = ()
        if self._state_path is not None and self._state_path.exists():
            saved = json.loads(self._state_path.read_text())
            if saved["state"] != "clear":
                self._record = Interruption(saved["nonce"], "manual", PICK_MANUAL_CHECK)
                self._home_joints = tuple(saved.get("home_joints", ()))
                home = np.asarray(saved.get("home", []), float)
                if home.shape == (len(self._home_joints),) and len(home) and np.isfinite(home).all():
                    self._home = home
                selection = getattr(executor, "selection", None)
                if selection is not None:
                    from .joints import arm_joints

                    side = next((s for s in ("left", "right") if self._home_joints == arm_joints(s)), None)
                    if side is None or side not in executor.parts or side not in effector.parts:
                        raise GraspFailure(PICK_MANUAL_CHECK, "saved interruption arm is unavailable")
                    # Restore only adapter routing. No stream is taken over and
                    # no trajectory or hand target is sent during inspection.
                    selection.side = side
        self._moves = []
        self._active = None
        self._effect = False
        self._unsafe = False
        self._baseline = None
        self._recovering = False
        self.held_factory = None
        self._closed_after = None
        self._grip_snapshot = None
        self._plan_id = ""
        for part in getattr(executor, "parts", {"arm": executor}).values():
            part.record_paths = True

    @property
    def status(self):
        with self._lock:
            return self._record

    def _save(self):
        if self._state_path is None:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self._record) if self._record else {"state": "clear", "nonce": ""}
        data.update(home_joints=self._home_joints, home=None if self._home is None else self._home.tolist())
        temporary = self._state_path.with_suffix(".tmp")
        with temporary.open("w") as output:
            json.dump(data, output, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, self._state_path)
        fd = os.open(self._state_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def confirm_manual_reset(self, nonce, *, confirmed_empty, note):
        """Read-only verification after a person restores the empty hand and arm."""
        with self._lock:
            if (
                self._record is None
                or self._record.nonce != nonce
                or self._recovering
                or not confirmed_empty
                or not str(note).strip()
            ):
                raise GraspFailure(PICK_MANUAL_CHECK, "matching interruption and explicit operator check required")
            if self._home is None or not self.effector.released():
                raise GraspFailure(PICK_MANUAL_CHECK, "fresh empty hand and original home are required")
            here = self.executor.measured(self._home_joints)
            if np.max(np.abs(here - self._home)) > POSE_TOLERANCE_RAD:
                raise GraspFailure(PICK_MANUAL_CHECK, "manually return the arm to its original start first")
            previous = self._record
            self._record = None
            try:
                self._save()
            except Exception:
                self._record = previous
                raise

    def begin(self, plan):
        with self._lock:
            if self._record is not None:
                raise GraspFailure(PICK_MANUAL_CHECK, "an interrupted pick must be resolved before a new pick")
            self._moves, self._active = [], None
            self._effect = self._unsafe = False
            self._closed_after = self._grip_snapshot = None
            self._plan_id = str(plan.report.get("x2_grasp_head_plan_id", ""))
            self._baseline = None
            first = next(step.segment for step in plan.steps if isinstance(step, Move))
            self._home_joints, self._home = first.joint_names, first.start.copy()
            # Persist before moving. This temporary running gate is not exposed
            # while the action owns the robot, but survives an unexpected exit.
            self._record = Interruption(uuid.uuid4().hex, "manual", PICK_MANUAL_CHECK)
            self._save()
            self._record = None

    def moving(self, segment):
        self._active = segment

    def moved(self, segment, outcome):
        self._active = None
        if outcome.state is MotionState.SUCCEEDED:
            path = outcome.report.get("executed_path")
            if path is None:
                self._unsafe = True
                return
            from manipulation_execution.grasp_pipeline.contracts import ArmSegment, SegmentKind

            self._moves.append(
                ArmSegment(
                    segment.name,
                    SegmentKind.TRAJECTORY,
                    segment.joint_names,
                    np.asarray(path["positions"]),
                    np.asarray(path["times"]),
                )
            )
            return
        self._unsafe |= outcome.state is not MotionState.CANCELLED
        # No nearest-point guess on an ambiguous/self-crossing path. The stream
        # reports the actual executed trajectory and last-command coordinate.
        path = outcome.report.get("executed_path")
        progress = outcome.report.get("path_progress")
        if path is None or progress is None or outcome.stopped_at is None:
            self._unsafe = True
            return
        try:
            from manipulation_execution.grasp_pipeline.contracts import ArmSegment, SegmentKind

            walked = ArmSegment(
                segment.name,
                SegmentKind.TRAJECTORY,
                segment.joint_names,
                np.asarray(path["positions"]),
                np.asarray(path["times"]),
            )
            back = walked.walked_back(outcome.stopped_at, path_progress=progress)
            if back is not None:
                # Store the executed partial forward path, never the unexecuted tail.
                self._moves.append(back.reversed(segment.name))
        except (ValueError, TypeError, KeyError):
            self._unsafe = True

    def effect_started(self, action):
        if action in {EffectorAction.CLOSE, EffectorAction.RELEASE}:
            self._effect = True

    def effect_finished(self, action, outcome):
        if action is EffectorAction.CLOSE and outcome.succeeded and not outcome.halted and outcome.held is True:
            self._closed_after = len(self._moves)
            self._effect = False
            try:
                self._grip_snapshot = self.effector.snapshot()
            except Exception:
                self._unsafe = True

    def end(self, result):
        with self._lock:
            if result.succeeded or result.at_rest:
                self._record = None
                self._save()
                return
            manual = (
                self._effect
                or self._unsafe
                or self._active is not None
                or result.error_code != "CANCELED"
                or getattr(result, "recovery_state", "") == "UNKNOWN"
            )
            try:
                self._baseline = np.asarray(self.posture(), float).copy()
                manual |= not np.isfinite(self._baseline).all()
                if self._closed_after is None:
                    manual |= not bool(self.effector.released())
                else:
                    manual |= self.effector.holding() != "" or self._grip_snapshot is None
            except Exception:
                manual = True
            self._record = Interruption(
                uuid.uuid4().hex, "manual" if manual else "pending", PICK_MANUAL_CHECK if manual else PICK_INTERRUPTED
            )
            self._save()

    def recover(self, nonce, *, deadline, cancelled):
        with self._lock:
            record = self._record
            if record is None:
                raise GraspFailure(PICK_MANUAL_CHECK, "no matching interrupted pick")
            if nonce != record.nonce or self._recovering:
                raise GraspFailure(PICK_MANUAL_CHECK, "stale or concurrent recovery request")
            if record.state != "pending":
                raise GraspFailure(PICK_MANUAL_CHECK, "hand/object or executed path requires manual inspection")
            self._recovering = True
            # A crash during recovery cannot resume the original full reverse path.
            self._record = Interruption(record.nonce, "manual", PICK_MANUAL_CHECK)
        try:
            self._save()
            carrying = self._closed_after is not None
            if carrying:
                # Reattach the grip observer after the previous action's finish;
                # this explicitly sends no command to the hand.
                self.effector.restore(self._grip_snapshot, send=False)

            def check(*, unchanged=True):
                if cancelled() or self.clock() >= deadline:
                    raise GraspFailure("CANCELED", "recovery cancelled or timed out")
                now = np.asarray(self.posture(), float)
                if (
                    now.shape != self._baseline.shape
                    or not np.isfinite(now).all()
                    or (unchanged and np.max(np.abs(now - self._baseline)) > POSE_TOLERANCE_RAD)
                    or (self.effector.holding() != "" if carrying else not self.effector.released())
                ):
                    raise GraspFailure(PICK_MANUAL_CHECK, "robot or hand moved since the interruption")

            check()

            def replay(steps):
                for step in steps:
                    check(unchanged=False)
                    outcome = self.executor.execute_checked(
                        step.segment,
                        carrying=carrying,
                        halted=lambda: carrying and self.effector.holding() != "",
                        cancelled=lambda: cancelled() or self.clock() >= deadline,
                    )
                    if outcome.state is not MotionState.SUCCEEDED:
                        raise GraspFailure(PICK_MANUAL_CHECK, f"retreat failed: {outcome.reason}")

            if carrying:
                if self.held_factory is None or not self._plan_id:
                    raise GraspFailure(PICK_MANUAL_CHECK, "original object geometry and support are unavailable")
                held = self.held_factory(self._moves)
                lifted = tuple(Move(s.reversed()) for s in reversed(self._moves[self._closed_after :]))
                self.validate(
                    (*lifted, Effect(EffectorAction.RELEASE)),
                    nonce,
                    held=held,
                    plan_id=self._plan_id,
                    deadline=deadline,
                    cancelled=cancelled,
                )
                check()
                replay(lifted)
                check(unchanged=False)
                held = replace(held, joint_positions=tuple(self.executor.measured(held.joint_names)))
                self.validate(
                    (Effect(EffectorAction.RELEASE),),
                    nonce,
                    held=held,
                    plan_id=self._plan_id,
                    deadline=deadline,
                    cancelled=cancelled,
                )
                check(unchanged=False)
                # Release only at the recorded contact, never at the interruption
                # or at home. The just-validated opening contains no arm motion.
                contact = self._moves[self._closed_after - 1]
                if np.max(np.abs(self.executor.measured(contact.joint_names) - contact.end)) > POSE_TOLERANCE_RAD:
                    raise GraspFailure(PICK_MANUAL_CHECK, "arm did not reach the original release pose")
                supervision = self.executor.supervise()
                released = self.effector.act(
                    EffectorAction.RELEASE,
                    halted=lambda: False,
                    cancelled=lambda: cancelled() or self.clock() >= deadline or bool(supervision()),
                )
                if not released.succeeded or released.halted or not self.effector.released():
                    raise GraspFailure(PICK_MANUAL_CHECK, "release was not confirmed")
                carrying = False
                steps = tuple(Move(s.reversed()) for s in reversed(self._moves[: self._closed_after]))
            else:
                steps = tuple(Move(segment.reversed()) for segment in reversed(self._moves))
            # After release a new capture includes the deposited object. Only
            # the empty, open hand's remaining path is authorized by that scene.
            if steps:
                self.validate(steps, nonce, deadline=deadline, cancelled=cancelled)
            check(unchanged=self._closed_after is None)
            replay(steps)
            check(unchanged=False)
            if self._moves:
                home = self._moves[0]
                here = self.executor.measured(home.joint_names)
                if np.max(np.abs(here - home.start)) > POSE_TOLERANCE_RAD:
                    raise GraspFailure(PICK_MANUAL_CHECK, "retreat did not reach the original pick start")
            with self._lock:
                self._record = None
                self._save()
        except Exception:
            with self._lock:
                self._record = Interruption(record.nonce, "manual", PICK_MANUAL_CHECK)
            try:
                self._save()
            finally:
                self.executor.stop("interrupted pick recovery failed")
            raise
        finally:
            try:
                self.executor.finish()
            except Exception:
                with self._lock:
                    self._record = Interruption(record.nonce, "manual", PICK_MANUAL_CHECK)
                    self._save()
                raise
            finally:
                try:
                    finish_hand = getattr(self.effector, "finish", None)
                    if finish_hand is not None:
                        finish_hand()
                except Exception:
                    with self._lock:
                        self._record = Interruption(record.nonce, "manual", PICK_MANUAL_CHECK)
                        self._save()
                    raise
                finally:
                    with self._lock:
                        self._recovering = False
