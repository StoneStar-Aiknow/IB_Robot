"""X2 fixed-target CPU execution. No language model, IK, MuJoCo or CUDA imports.

Targets and body geometry are compiled offline; runtime checks the exported
convex body model, static gravity and articulated-hand covers. No detour is invented.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from aimdk_grasp.feedback import Feedback
from aimdk_grasp.hand import HandMapping
from aimdk_grasp.hand_restore import HandSweepCheck, RestoreError
from aimdk_grasp.hand_restore_geometry import DualHandCollision
from aimdk_grasp.joints import HAND_ALIASES, HEAD, LEFT_ARM, RIGHT_ARM
from aimdk_grasp.robot_model import config_directory, planning_model
from robot_runtime.pose_backend import Plan, Report, Trajectory, intersect_joint_limits
from robot_runtime.task_poses import normalize_task_poses, task_pose_arm_targets

JOINTS = LEFT_ARM + RIGHT_ARM
EFFORT = np.array([36, 36, 24, 24, 24, 4.8, 4.8] * 2)
MAX_SPEED = 0.5
START_TOLERANCE_RAD = 0.005
GRAVITY_SHARE = 0.30


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class FixedBackend:
    ARM_JOINTS = JOINTS

    def __init__(self, profile_path, limits=None, library=None, body_geometry=None, *, task_poses=None):
        profile = yaml.safe_load(Path(profile_path).read_text())
        if tuple(profile["arm_joints"]) != JOINTS:
            raise ValueError("fixed poses require the X2 dual-arm joint order")
        self.hard_limits = intersect_joint_limits(JOINTS, profile["vendor"]["arm"]["limits"], limits)
        self.safe_limits = self.hard_limits
        self.rest = profile["conformance"]["start_positions"]
        self.model = self.safety = self.spec = self
        self.guard = lambda: None
        opened = dict(zip(HAND_ALIASES, [0.129, -1.54, 0.255, 0, 0, 0, 0, 0, 0, 0], strict=True))
        self.geometry = DualHandCollision(
            {
                side: HandSweepCheck(HandMapping(planning_model(config_directory(side)), {"open": opened}, side=side))
                for side in ("left", "right")
            }
        )
        files = {}
        for side in ("left", "right"):
            directory = config_directory(side)
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix in {".urdf", ".yaml", ".yml", ".json"}:
                    files[f"{side}/{path.relative_to(directory)}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.fingerprint = digest({"profile": profile, "limits": self.hard_limits, "geometry": files})
        from .compiled_body_geometry import CompiledGeometry

        self.body_geometry = CompiledGeometry(body_geometry) if body_geometry is not None else None
        self.library = None
        self.task_poses = {}
        if library is not None:
            data = json.loads(Path(library).read_text()) if isinstance(library, str | Path) else library
            payload = {key: value for key, value in data.items() if key != "digest"}
            if data.get("schema_version") != 1 or data.get("digest") != digest(payload):
                raise ValueError("invalid compiled pose library digest/version")
            if data.get("robot_fingerprint") != self.fingerprint or data.get("joint_names") != list(JOINTS):
                raise ValueError("compiled poses do not match the runtime robot model/limits")
            if not data.get("poses"):
                raise ValueError("empty compiled pose library")
            if data.get("task_poses"):
                self.task_poses = normalize_task_poses(data["task_poses"])
                if tuple(self.task_poses["joint_names"]) != JOINTS:
                    raise ValueError("compiled task-pose anchors use the wrong joint order")
                if task_poses is not None:
                    current = normalize_task_poses(task_poses) if task_poses else {}
                    if current.get("digest") != self.task_poses["digest"]:
                        raise ValueError("compiled task-pose anchors differ from the runtime snapshot")
            self.body_geometry = CompiledGeometry(data["body_geometry"])
            for record in data["poses"].values():
                if not record.get("phrases") or not self.check_pose(record["target"]).ok:
                    raise ValueError("invalid compiled pose target/phrases")
                anchors, _ = task_pose_arm_targets(record.get("description") or {}, self.task_poses)
                if any(record["target"][n] != value for n, value in anchors.items()):
                    raise ValueError("compiled pose does not preserve its exact task-pose anchors")
            self.library = data

    def load(self):
        return self

    def limit_violations(self, q, limits=None):
        limits = self.hard_limits if limits is None else limits
        if set(q) != set(JOINTS):
            return ["complete dual-arm state required"]
        return [name for name, (lo, hi) in limits.items() if not np.isfinite(q[name]) or not lo <= q[name] <= hi]

    def observation(self, q):
        return Feedback(
            1.0,
            {**q, **dict.fromkeys(HEAD, 0.0)},
            self.geometry.checks["left"].mapping.preset("open"),
            right_hand=self.geometry.checks["right"].mapping.preset("open"),
        )

    def check_pose(self, q, *, head=None):
        violations = self.limit_violations(q)
        if violations:
            return Report(False, limit_violations=violations)
        if self.body_geometry is None:
            return Report(False, notes=["compiled body model required"])
        ok, label, torques = self.body_geometry.check({**q, **(head or {})})
        share = float(np.max(np.abs([torques[n] for n in JOINTS]) / EFFORT))
        return Report(ok and share <= GRAVITY_SHARE, gravity_share=share, notes=[label] if label else [])

    def plan_safe(self, start, frames, holds):
        """Offline validation, including the reference open-hand geometry."""
        return self.prepare_execution(self._joint_candidate(start, frames, holds))

    def plan_for_execution(self, start, frames, holds):
        """Build a bounded candidate; the transport gate checks measured geometry.

        The final executor must check body AND real fingers/head before sending.
        Avoid first checking an assumed open-hand/head-zero state that does not
        establish clearance for the live robot and would be checked again.
        """
        return self.prepare_execution(self._joint_candidate(start, frames, holds), check_geometry=False)

    def _joint_candidate(self, start, frames, holds):
        if len(frames) != 1 or len(holds) != 1 or holds[0] != 0:
            raise ValueError("fixed backend accepts one joint target without a hold")
        target = frames[0]
        for q in (start, target):
            if self.limit_violations(q):
                return Plan(Trajectory([], []), False, [], Report(False, notes=["invalid start/target"]))
        a, b = (np.array([q[n] for n in JOINTS]) for q in (start, target))
        duration = max(1.0, 1.875 * float(np.max(np.abs(b - a))) / MAX_SPEED)
        times = np.linspace(0, duration, max(2, int(np.ceil(duration * 25))) + 1)
        s = times[:, None] / duration
        rows = a + (b - a) * (10 * s**3 - 15 * s**4 + 6 * s**5)
        rows[0], rows[-1] = a, b
        return Plan(Trajectory(times, rows), True, [], Report(True))

    def prepare_execution(self, plan, rate_hz=25.0, *, check_geometry=True):
        if not plan.ok:
            return plan
        times, rows = np.asarray(plan.traj.times), np.asarray(plan.traj.positions)
        if (
            times.ndim != 1
            or len(times) < 2
            or rows.shape != (len(times), 14)
            or not np.isfinite(times).all()
            or not np.isfinite(rows).all()
            or times[0] != 0
            or np.any(np.diff(times) <= 0)
        ):
            return Plan(plan.traj, False, [], Report(False, notes=["malformed fixed trajectory"]))
        if np.max(np.abs(np.diff(rows, axis=0) / np.diff(times)[:, None])) > MAX_SPEED + 1e-6:
            return Plan(plan.traj, False, [], Report(False, notes=["trajectory exceeds speed limit"]))
        for row in rows:
            if self.limit_violations(dict(zip(JOINTS, row, strict=True))):
                return Plan(plan.traj, False, [], Report(False, notes=["trajectory exceeds joint limits"]))
        if not check_geometry:
            return Plan(plan.traj, True, [], Report(True, notes=["measured-state geometry required before dispatch"]))
        try:
            self.body_geometry.check_path(JOINTS, rows, guard=self.guard)
            q = dict(zip(JOINTS, rows[0], strict=True))
            self.geometry.check_arm_path(self.observation(q), JOINTS, rows, guard=self.guard)
        except (ValueError, RestoreError) as exc:
            return Plan(plan.traj, False, [], Report(False, notes=[str(exc)]))
        return Plan(plan.traj, True, [], Report(True))

    def plan(self, composition, start):
        return self.plan_for_execution(start, composition.frames, [0.0])

    def check_displacement(self, plan, start, max_joint=0.0, max_hand=0.0):
        if max_hand:
            return "Cartesian commissioning bounds are not supported by the fixed backend"
        delta = float(np.max(np.abs(np.asarray(plan.traj.positions) - [start[n] for n in JOINTS])))
        return f"joint displacement {delta:.4f} exceeds {max_joint}" if max_joint and delta > max_joint else ""
