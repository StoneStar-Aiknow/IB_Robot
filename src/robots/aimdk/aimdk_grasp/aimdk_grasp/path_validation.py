"""Measured-state transport gate for fixed candidates. Never used by the compiler."""

import time
from contextlib import contextmanager

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

    def check(self, restorer, ctx, plan, start_tolerance_rad):
        backend, joints = self.backend, self.backend.ARM_JOINTS
        # Revalidate even when a caller bypassed candidate construction.
        with self.stage("candidate_build"):
            if not backend.prepare_execution(plan, check_geometry=False).ok:
                raise RestoreError("UNSAFE_POSE", "malformed, excessive-speed or out-of-limit fixed candidate")
        with self.stage("ownership_prepare"):
            restorer.ports.prepare(ctx)
            baseline = restorer.arm_baseline(ctx)
        rows = plan.traj.positions
        head = {n: baseline.feedback.positions[n] for n in HEAD}

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
            with self.stage(
                f"{part}_start", "reused" if match else "executed", "certificate entry tube" if match else ""
            ):
                if not match:
                    if part == "hand":
                        # Off checks the measured entry shape, not a future
                        # preservation-error tube along an unverified path.
                        options = {"finger_radius": 0.0} if mode == "off" else {}
                        restorer.geometry.check_arm_path(
                            baseline.feedback,
                            joints,
                            rows[:2],
                            guard=guard,
                            start_tolerance_rad=start_tolerance_rad,
                            **options,
                        )
                    else:
                        backend.body_geometry.check_path(
                            joints, rows[:2], head=head, guard=guard, start_tolerance_rad=start_tolerance_rad
                        )
        if any(matches.values()):
            self.monitor = self.certificate.monitor(plan, matches)
            self.conditions = lambda: restorer.validation_conditions(ctx)
        guard()
        return baseline
