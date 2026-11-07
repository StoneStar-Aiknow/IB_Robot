"""Offline interval enclosures and strict runtime matching for fixed joint lines.

An integrity digest is not an approval. Live standing/unloaded evidence must be
provided by the transport; the present X2 transport has no such observation and
therefore falls back online. No MuJoCo dependency is introduced here.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot_runtime.path_validation import policy_digest

from .fixed_pose_backend import MAX_SPEED
from .hand_restore import PATH_HEAD_HOLD_RAD, PATH_SHAPE_HOLD_RAD, RestoreError
from .joints import HEAD

ASSUMPTIONS = {"standing_lower_body": True, "unloaded": True}
METHOD = "fixed-line-interval-v1"


@dataclass(frozen=True)
class LiveConditions:
    """Verified transport evidence, never a deployment configuration assertion.

    Bounds describe the enforced speed and measured/guaranteed cancellation
    contract, not the speed in a single stationary sample. X2 currently has no
    producer for this record and therefore cannot activate physical reuse.
    """

    received_at: float
    standing_lower_body: bool
    unloaded: bool
    speed_bound_rad_s: float
    cancel_bound_sec: float


def algorithm_digest():
    directory = Path(__file__).parent
    return policy_digest(
        {
            p: hashlib.sha256((directory / p).read_bytes()).hexdigest()
            for p in (
                "path_certificate.py",
                "compiled_body_geometry.py",
                "fixed_pose_backend.py",
                "hand_restore.py",
                "hand_restore_geometry.py",
                "robot_model.py",
                "leg_clearance.py",
                "leg_obstacle.py",
                "joints.py",
                "planning_model.py",
                "hand.py",
            )
        }
    )


def binding(backend):
    return {
        "method": METHOD,
        "algorithm_digest": algorithm_digest(),
        "robot_fingerprint": backend.fingerprint,
        "library_digest": backend.library["digest"],
        "joint_names": list(backend.ARM_JOINTS),
        "max_speed_rad_s": MAX_SPEED,
        "rate_hz": 25.0,
    }


def vector(value, name, size=14, positive=False):
    a = np.asarray(value, float)
    if a.shape != (size,) or not np.isfinite(a).all() or (a <= 0 if positive else a < 0).any():
        raise ValueError(f"invalid certificate {name}")
    return a


def compile_certificate(backend, *, entry_rad, tracking_rad, response_sec, poll_sec=0.1, feedback_age_sec=0.1):
    """Refuse unless collision AND gravity enclosures cover the entire error tube.

    The tube includes entry error, tracking error and worst-case continued
    motion at MAX_SPEED through stale feedback, polling and cancellation.
    Static gravity is bounded between samples using global reach derivatives.
    Each component may fail independently; failed components have no proof.
    """
    entry = vector(entry_rad, "entry_rad", positive=True)
    tracking = vector(tracking_rad, "tracking_rad", positive=True)
    for value in (response_sec, poll_sec, feedback_age_sec):
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError("certificate response/poll/feedback bounds must be positive")
    if poll_sec < 0.1 or feedback_age_sec < 0.1:
        raise ValueError("certificate cannot assume faster monitoring than the executor provides")
    tube = entry + tracking + MAX_SPEED * (response_sec + poll_sec + feedback_age_sec)
    home = dict(zip(backend.ARM_JOINTS, [0.4, 0, 0, -1.2, 0, 0, 0] * 2, strict=True))
    up = backend.library["poses"]["arms_up"]["target"]
    result = {
        "schema_version": 1,
        "binding": binding(backend),
        "assumptions": ASSUMPTIONS,
        "entry_rad": entry.tolist(),
        "tracking_rad": tracking.tolist(),
        "tube_rad": tube.tolist(),
        "response_sec": response_sec,
        "poll_sec": poll_sec,
        "feedback_age_sec": feedback_age_sec,
        "finger_entry_rad": 0.005,
        "finger_radius_rad": 0.005 + PATH_SHAPE_HOLD_RAD,
        "head_entry_rad": 0.005,
        "head_radius_rad": 0.005 + PATH_HEAD_HOLD_RAD,
        "routes": [],
    }
    for a, b in ((home, up), (up, home)):
        plan = backend.plan_for_execution(a, [b], [0.0])
        if not plan.ok:
            raise ValueError("invalid certificate candidate")
        rows = np.asarray(plan.traj.positions)
        # Joint limits apply to the tube too, not just its center line.
        limits = np.asarray([backend.hard_limits[n] for n in backend.ARM_JOINTS])
        inside = np.all(rows.min(axis=0) - tube >= limits[:, 0]) and np.all(rows.max(axis=0) + tube <= limits[:, 1])
        route = {"start": a, "target": b, "duration_sec": float(plan.traj.times[-1]), "components": {}, "refused": {}}
        for part in ("hand", "body"):
            try:
                if not inside:
                    raise ValueError("error tube exceeds joint limits")
                if part == "hand":
                    backend.geometry.check_arm_path(
                        backend.observation(a),
                        backend.ARM_JOINTS,
                        rows,
                        uncertainty=tube,
                        finger_radius=result["finger_radius_rad"],
                        head_radius=result["head_radius_rad"],
                    )
                    proof = {"residual_clearance_lower_m": -1e-6}
                else:
                    upper = backend.body_geometry.check_path(
                        backend.ARM_JOINTS,
                        rows,
                        uncertainty=tube,
                        head_radius=result["head_radius_rad"],
                        bound_gravity=True,
                    )
                    proof = {"residual_clearance_lower_m": 0.0, "gravity_upper_nm": upper}
                route["components"][part] = {"method": METHOD, **proof}
            except (ValueError, RestoreError) as exc:
                route["refused"][part] = str(exc)
        result["routes"].append(route)
    result["digest"] = policy_digest(result)
    return result


class Certificate:
    @classmethod
    def load(cls, policy, backend):
        raw = Path(policy["certificate_file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != policy["certificate_sha256"]:
            raise ValueError("path validation artifact changed after launch binding")
        return cls(json.loads(raw), backend)

    def __init__(self, data, backend):
        keys = {
            "schema_version",
            "binding",
            "assumptions",
            "entry_rad",
            "tracking_rad",
            "tube_rad",
            "response_sec",
            "poll_sec",
            "feedback_age_sec",
            "finger_entry_rad",
            "finger_radius_rad",
            "head_entry_rad",
            "head_radius_rad",
            "routes",
            "digest",
        }
        if set(data) != keys:
            raise ValueError("unknown or missing certificate fields")
        if (
            data.get("schema_version") != 1
            or data.get("binding") != binding(backend)
            or data.get("digest") != policy_digest({k: v for k, v in data.items() if k != "digest"})
            or data.get("assumptions") != ASSUMPTIONS
        ):
            raise ValueError("certificate format/digest/model/algorithm mismatch")
        self.data, self.backend = data, backend
        if any(
            isinstance(data[n], bool)
            or not isinstance(data[n], int | float)
            or not np.isfinite(data[n])
            or data[n] <= 0
            for n in ("response_sec", "poll_sec", "feedback_age_sec")
        ):
            raise ValueError("invalid certificate response/poll/feedback bound")
        self.entry = vector(data["entry_rad"], "entry", positive=True)
        self.tracking = vector(data["tracking_rad"], "tracking", positive=True)
        expected = (
            self.entry
            + self.tracking
            + MAX_SPEED * sum(data[n] for n in ("response_sec", "poll_sec", "feedback_age_sec"))
        )
        if not np.allclose(vector(data["tube_rad"], "tube"), expected, atol=1e-12, rtol=0):
            raise ValueError("invalid certificate response budget")
        if data["poll_sec"] < 0.1 or data["feedback_age_sec"] < 0.1 or data["response_sec"] <= 0:
            raise ValueError("invalid certificate monitor timing")
        if (
            data["finger_entry_rad"] != 0.005
            or data["head_entry_rad"] != 0.005
            or data["finger_radius_rad"] != 0.005 + PATH_SHAPE_HOLD_RAD
            or data["head_radius_rad"] != 0.005 + PATH_HEAD_HOLD_RAD
        ):
            raise ValueError("invalid certificate shape bounds")
        # Explicitly constrain v1 to the two compiled directed routes.
        home = dict(zip(backend.ARM_JOINTS, [0.4, 0, 0, -1.2, 0, 0, 0] * 2, strict=True))
        up = backend.library["poses"]["arms_up"]["target"]
        routes = data["routes"]
        if len(routes) != 2 or [(r["start"], r["target"]) for r in routes] != [(home, up), (up, home)]:
            raise ValueError("invalid certificate routes")
        for route in routes:
            for part, proof in route["components"].items():
                if part not in ("hand", "body") or proof.get("method") != METHOD:
                    raise ValueError("unsupported certificate component")
                floor = -1e-6 if part == "hand" else 0.0
                if proof.get("residual_clearance_lower_m") != floor:
                    raise ValueError("invalid certificate clearance enclosure")
                if part == "body":
                    from .fixed_pose_backend import EFFORT

                    upper = vector([proof["gravity_upper_nm"][n] for n in backend.ARM_JOINTS], "gravity upper")
                    if (upper > EFFORT * 0.29).any():
                        raise ValueError("invalid certificate gravity enclosure")

    def match(self, part, feedback, plan, start_tolerance_rad, conditions=None):
        if not self.conditions_match(conditions):
            return None, "standing/unloaded/speed/stop live evidence unavailable; online fallback"
        joints = self.backend.ARM_JOINTS
        rows = np.asarray(plan.traj.positions)

        # The proof is for a line. Reconstruct the exact minimum-jerk candidate;
        # arbitrary waypoint or timing edits must never inherit it.
        def q(row):
            return dict(zip(joints, map(float, row), strict=True))

        candidate = self.backend.plan_for_execution(q(rows[0]), [q(rows[-1])], [0.0])
        if (
            not candidate.ok
            or rows.shape != np.shape(candidate.traj.positions)
            or not np.allclose(rows, candidate.traj.positions, atol=1e-12, rtol=0)
            or not np.allclose(plan.traj.times, candidate.traj.times, atol=1e-12, rtol=0)
        ):
            return None, "candidate differs from certified line/timing; online fallback"
        for route in self.data["routes"]:
            if part not in route["components"]:
                continue
            start, target = ([pose[n] for n in joints] for pose in (route["start"], route["target"]))
            # Include the mandatory final dispatch-start tolerance in the entry box.
            if np.any(np.abs(rows[0] - start) + start_tolerance_rad > self.entry) or not np.array_equal(
                rows[-1], target
            ):
                continue
            if np.any(np.abs(feedback.vector(joints) - start) > self.entry):
                continue
            if not self.shape_matches(feedback, entry=True):
                return None, "finger/head entry outside certificate; online fallback"
            return route, "fixed directed route and measured entry matched"
        return None, "no proven component/route/entry match; online fallback"

    def shape_matches(self, feedback, *, entry):
        reference = self.backend.observation(dict(zip(self.backend.ARM_JOINTS, [0.0] * 14, strict=True)))
        try:
            hands = {s: feedback.hand(s) for s in ("left", "right")}
        except ValueError:
            # Certificates are proven with both hands mounted; one hand absent falls back online.
            return False
        return (
            all(
                np.max(np.abs(hands[s] - reference.hand(s)))
                <= self.data["finger_entry_rad" if entry else "finger_radius_rad"]
                for s in ("left", "right")
            )
            and np.max(np.abs(feedback.vector(HEAD))) <= self.data["head_entry_rad" if entry else "head_radius_rad"]
        )

    def conditions_match(self, conditions):
        return (
            isinstance(conditions, LiveConditions)
            and conditions.standing_lower_body is True
            and conditions.unloaded is True
            and np.isfinite(conditions.received_at)
            and 0 <= time.monotonic() - conditions.received_at <= self.data["feedback_age_sec"]
            and np.isfinite(conditions.speed_bound_rad_s)
            and 0 < conditions.speed_bound_rad_s <= MAX_SPEED
            and np.isfinite(conditions.cancel_bound_sec)
            and 0 < conditions.cancel_bound_sec <= self.data["response_sec"]
        )

    def monitor(self, plan, matches):
        return CertificateMonitor(self, plan, matches)


class CertificateMonitor:
    def __init__(self, certificate, plan, matches):
        self.certificate, self.plan = certificate, plan
        self.matches = matches
        self.last = None

    def check(self, feedback, started, conditions=None, *, feedback_age_s=None):
        now, data = time.monotonic(), self.certificate.data
        if not self.certificate.conditions_match(conditions) or not self.certificate.shape_matches(
            feedback, entry=started is None
        ):
            raise RestoreError("POSE_CERTIFICATE_RANGE", "certificate live assumptions or hand/head bounds lost")
        if self.last is not None and started is not None and now - self.last > data["poll_sec"]:
            raise RestoreError("POSE_CERTIFICATE_RANGE", "certificate monitor polling budget exceeded")
        self.last = now
        rows, times = np.asarray(self.plan.traj.positions), self.plan.traj.times
        elapsed = max(0, now - started) if started is not None else 0
        desired = np.array([np.interp(elapsed, times, rows[:, i]) for i in range(14)])
        measured = feedback.vector(self.certificate.backend.ARM_JOINTS)
        if np.any(np.abs(measured - desired) > self.certificate.tracking):
            raise RestoreError("POSE_CERTIFICATE_RANGE", "arm tracking left certificate tube; cancel required")
        # Joint stamps use the ROS clock; polling uses monotonic time. The
        # transport computes stamp age in its own ROS clock domain.
        if (
            feedback_age_s is None
            or not np.isfinite(feedback_age_s)
            or not 0 <= feedback_age_s <= data["feedback_age_sec"]
        ):
            raise RestoreError("POSE_CERTIFICATE_RANGE", "feedback exceeded certificate age budget")
