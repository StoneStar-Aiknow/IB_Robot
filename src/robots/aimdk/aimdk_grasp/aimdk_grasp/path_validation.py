"""Measured-state transport gate for fixed candidates. Never used by the compiler."""

import time
from contextlib import contextmanager

import numpy as np

from robot_runtime.path_validation import normalize_path_validation, path_validation_identity

from .hand_restore import RestoreError
from .joints import HEAD


class ValidationGate:
    def __init__(self, backend, policy=None, *, certificate=None):
        self.backend = backend
        self.policy = normalize_path_validation(policy)
        self.identity = path_validation_identity({"mode": "fixed", "path_validation": self.policy})
        self.report = {"policy": dict(self.policy), **self.identity, "stages": {}}
        for name in (
            "runtime_state",
            "candidate_build",
            "ownership_prepare",
            "endpoint_body",
            "hand_path",
            "hand_start",
            "body_path",
            "body_start",
            "dispatch_gate",
            "arm_motion",
        ):
            self.report["stages"][name] = {"status": "skipped", "sec": 0.0, "reason": "stage not reached"}
        self.monitor = None
        if "prevalidated" in self.policy.values():
            from .path_certificate import Certificate

            self.certificate = certificate if certificate is not None else Certificate.load(self.policy, backend)
        else:
            self.certificate = None

    @contextmanager
    def stage(self, name, status="executed", reason=""):
        item = self.report["stages"][name] = {"status": status, "reason": reason, "sec": 0.0}
        started = time.monotonic()
        try:
            yield
        except Exception:
            item["status"] = "failed"
            raise
        finally:
            item["sec"] = max(0.0, time.monotonic() - started)

    def check(self, restorer, ctx, plan, start_tolerance_rad, *, entry_start=None, entry_tolerance_rad=None):
        """Check a fixed candidate; ``start_tolerance_rad`` is the admitted plan-start drift.

        By default the entry segment rows[0] -> rows[1] is checked with that drift
        as its box. With ``entry_start`` (the measured arm, ``ARM_JOINTS`` order)
        the entry is checked from where the arm is, entry_start -> rows[1], with
        only ``entry_tolerance_rad`` for what it may still drift before dispatch;
        the caller must then admit dispatch within that radius of ``entry_start``.
        Certificate matching keeps the plan-start reference and tolerance.
        """
        backend, joints = self.backend, self.backend.ARM_JOINTS
        self._checked = None  # a failed check leaves nothing for recheck_entry
        # Revalidate even when a caller bypassed candidate construction.
        with self.stage("candidate_build"):
            if not backend.prepare_execution(plan, check_geometry=False).ok:
                raise RestoreError("UNSAFE_POSE", "malformed, excessive-speed or out-of-limit fixed candidate")
        with self.stage("ownership_prepare"):
            restorer.ports.prepare(ctx)
            baseline = restorer.arm_baseline(ctx)
        rows = plan.traj.positions
        head = {n: baseline.feedback.positions[n] for n in HEAD}
        if entry_start is None:
            entry_rows, entry_tolerance = rows[:2], start_tolerance_rad
            self.report["entry_start"] = {"reference": "plan_start", "radius_rad": float(start_tolerance_rad)}
        else:
            start = np.asarray(entry_start, dtype=float)
            if (
                start.shape != (len(joints),)
                or not np.isfinite(start).all()
                or entry_tolerance_rad is None
                or not np.isfinite(entry_tolerance_rad)
                or entry_tolerance_rad < 0
            ):
                raise RestoreError("UNSAFE_POSE", "invalid measured entry start or tolerance")
            entry_rows = np.vstack([start, np.asarray(rows[1], dtype=float)])
            entry_tolerance = float(entry_tolerance_rad)
            self.report["entry_start"] = {
                "reference": "measured",
                "positions_rad": start.tolist(),
                "radius_rad": entry_tolerance,
                "plan_start_drift_rad": float(np.max(np.abs(start - np.asarray(rows[0], dtype=float)))),
            }

        def guard():
            return restorer.guard_arm_motion(ctx, baseline)

        with self.stage("endpoint_body"):
            for row in (rows[0], rows[-1]):
                q = dict(zip(joints, row, strict=True))
                if not backend.check_pose(q, head=head).ok:
                    raise RestoreError("UNSAFE_POSE", "fixed start/target body or gravity rejected")
        matches = {}
        for part in ("hand", "body"):
            mode = self.policy[part]
            match, reason = (None, "configured online")
            if mode == "prevalidated":
                match, reason = self.certificate.match(
                    part, baseline.feedback, plan, start_tolerance_rad, restorer.validation_conditions(ctx)
                )
            matches[part] = match
            status = "reused" if match else "skipped" if mode == "off" else "executed"
            with self.stage(
                f"{part}_path", status, "configured off; full path not verified" if mode == "off" else reason
            ):
                if status == "executed":
                    if part == "hand":
                        restorer.geometry.check_arm_path(baseline.feedback, joints, rows, guard=guard)
                    else:
                        backend.body_geometry.check_path(joints, rows, head=head, guard=guard)
        self._check_entry(restorer, ctx, baseline, matches, entry_rows, entry_tolerance)
        if any(matches.values()):
            self.monitor = self.certificate.monitor(plan, matches)
            self.conditions = lambda: restorer.validation_conditions(ctx)
        guard()
        self._checked = {
            "rows": np.array(rows, dtype=float, copy=True),
            "baseline": baseline,
            "matches": dict(matches),
        }
        return baseline

    def _check_entry(self, restorer, ctx, baseline, matches, entry_rows, entry_tolerance):
        backend, joints = self.backend, self.backend.ARM_JOINTS
        head = {n: baseline.feedback.positions[n] for n in HEAD}

        def guard():
            return restorer.guard_arm_motion(ctx, baseline)

        for part in ("hand", "body"):
            match = matches[part]
            with self.stage(
                f"{part}_start", "reused" if match else "executed", "certificate entry tube" if match else ""
            ):
                if match:
                    continue
                if part == "hand":
                    # Off checks the measured entry shape, not a future
                    # preservation-error tube along an unverified path.
                    options = {"finger_radius": 0.0} if self.policy[part] == "off" else {}
                    restorer.geometry.check_arm_path(
                        baseline.feedback,
                        joints,
                        entry_rows,
                        guard=guard,
                        start_tolerance_rad=entry_tolerance,
                        **options,
                    )
                else:
                    backend.body_geometry.check_path(
                        joints, entry_rows, head=head, guard=guard, start_tolerance_rad=entry_tolerance
                    )

    def recheck_entry(self, restorer, ctx, plan, entry_start, entry_tolerance_rad):
        """Check the entry again from a newer measured start; the rest of the last check stands.

        Only the entry segment depends on where the arm is: the full path, the
        endpoints and any certificate match do not. The fingers and head are
        those of the last check's baseline, which the guard keeps enforcing.
        Requires a successful :meth:`check` of this same trajectory.
        """
        checked = getattr(self, "_checked", None)
        if checked is None or not np.array_equal(np.asarray(plan.traj.positions, dtype=float), checked["rows"]):
            raise RestoreError("POSE_START_UNVALIDATED", "no successful check of this trajectory to recheck")
        joints = self.backend.ARM_JOINTS
        start = np.asarray(entry_start, dtype=float)
        if start.shape != (len(joints),) or not np.isfinite(start).all() or not entry_tolerance_rad >= 0:
            raise RestoreError("UNSAFE_POSE", "invalid measured entry start or tolerance")
        entry_rows = np.vstack([start, checked["rows"][1]])
        self._checked = None  # a failed recheck leaves nothing to dispatch against
        self._check_entry(
            restorer, ctx, checked["baseline"], checked["matches"], entry_rows, float(entry_tolerance_rad)
        )
        restorer.guard_arm_motion(ctx, checked["baseline"])
        self._checked = checked
        entry = self.report.setdefault("entry_start", {})
        entry.setdefault("rechecks", []).append(
            {
                "previous_positions_rad": entry.get("positions_rad"),
                "positions_rad": start.tolist(),
                "plan_start_drift_rad": float(np.max(np.abs(start - checked["rows"][0]))),
            }
        )
        entry["positions_rad"] = start.tolist()
