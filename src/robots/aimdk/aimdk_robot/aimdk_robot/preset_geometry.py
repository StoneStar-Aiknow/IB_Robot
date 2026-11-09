"""Dependency-free numerical guard for explicitly compiled named hold geometry.

Compilation is a workstation task. This reader never loads a ROS package,
MuJoCo, a URDF, or a hand planner on the robot. An unreviewed candidate is a
useful preview artifact, but is deliberately not an executable certificate.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
HEAD_CLEARANCE_M = 0.20
MAX_PATH_POINTS = 4096
EASING_PEAK_FACTOR = math.pi / 2


def canonical_digest(value: dict) -> str:
    """Hash the payload, excluding its own digest field."""
    payload = {key: item for key, item in value.items() if key != "sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _finite(value, shape, label):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must have finite shape {shape}")
    return array


class NamedHoldGeometryGuard:
    """Validate a native 34-joint polyline against a frozen sphere-cover model.

    Distances use the complete compiled collision cover. Each interval is
    tested at its midpoint with a triangle-inequality motion allowance. The
    allowance is exported per sphere and per native joint, not inferred from
    the clearances at the endpoints. The artifact remains disabled until its
    model coverage, motion bounds and head envelope have been reviewed.
    """

    def __init__(self, record: dict):
        self.record = record
        self.invalid_reason = ""
        self.models = []
        self.lower = self.upper = self.speed = None
        try:
            if (
                not isinstance(record, dict)
                or type(record.get("schema_version")) is not int
                or record.get("schema_version") != SCHEMA_VERSION
            ):
                raise ValueError("unsupported preset geometry schema")
            if record.get("sha256") != canonical_digest(record):
                raise ValueError("preset geometry digest mismatch")
            if record.get("validated") is not True:
                raise ValueError("preset geometry is an unvalidated candidate")
            required = record.get("required_head_clearance_m")
            if (
                isinstance(required, bool)
                or not isinstance(required, int | float)
                or not math.isfinite(required)
                or required < HEAD_CLEARANCE_M
            ):
                raise ValueError("head clearance cannot be less than 0.20 m")
            self.required = float(required)
            self.name = record["name"]
            if not isinstance(self.name, str) or not self.name:
                raise ValueError("motion name is required")
            limits = _finite(record["native_limits_rad"], (34, 2), "native limits")
            if np.any(limits[:, 0] >= limits[:, 1]):
                raise ValueError("invalid native joint limits")
            self.lower, self.upper = limits.T.copy()
            self.speed = _finite(record["native_speed_radps"], (34,), "native speeds")
            if np.any(self.speed <= 0):
                raise ValueError("native speeds must be positive")
            self.uncertainty = _finite(record["native_uncertainty_rad"], (34,), "native uncertainty")
            if np.any(self.uncertainty < 0):
                raise ValueError("negative native uncertainty")
            self.head = _finite(record["head_envelope_spheres"], (len(record["head_envelope_spheres"]), 4), "head")
            if not len(self.head) or np.any(self.head[:, 3] <= 0):
                raise ValueError("head envelope must contain positive spheres")
            if record.get("head_envelope_covers_allowed_motion") is not True:
                raise ValueError("head motion envelope is not certified")
            if record.get("complete_articulated_hands") is not True:
                raise ValueError("complete articulated hand coverage is required")
            if record.get("continuous_motion_bounds_verified") is not True:
                raise ValueError("continuous motion bounds have not been verified")
            for model in record["models"]:
                self.models.append(self._load_model(model))
            if {model["side"] for model in self.models} != {"left", "right"} or len(self.models) != 2:
                raise ValueError("exactly two physical hand models are required")
            if not record.get("sources"):
                raise ValueError("model provenance is missing")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self.invalid_reason = str(exc)

    @staticmethod
    def _load_model(model):
        joints = model["joints"]
        output = {"side": model["side"], "joints": joints}
        for index, joint in enumerate(joints):
            if type(joint["parent"]) is not int or not 0 <= joint["parent"] <= index:
                raise ValueError("FK tree is not parent first")
            origin = _finite(joint["origin"], (4, 4), "joint origin")
            rotation = origin[:3, :3]
            if (
                not np.allclose(origin[3], [0, 0, 0, 1], atol=1e-9)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7)
            ):
                raise ValueError("joint origin is not a rigid transform")
            axis = _finite(joint["axis"], (3,), "joint axis")
            if not (np.isclose(np.linalg.norm(axis), 1) or np.array_equal(axis, np.zeros(3))):
                raise ValueError("joint axis is not unit length")
            if type(joint["source"]) is not int or not -1 <= joint["source"] < 34:
                raise ValueError("joint source outside native vector")
            coefficients = np.asarray(joint["coefficients"], float)
            if coefficients.ndim != 1 or not len(coefficients) or not np.isfinite(coefficients).all():
                raise ValueError("invalid joint polynomial")
        spheres = _finite(model["spheres"], (len(model["spheres"]), 4), "collision spheres")
        frames = np.asarray(model["sphere_frames"], int)
        if frames.shape != (len(spheres),) or np.any(frames < 0) or np.any(frames > len(joints)):
            raise ValueError("sphere frame outside tree")
        if not len(spheres) or np.any(spheres[:, 3] <= 0):
            raise ValueError("collision sphere radius must be positive")
        output["spheres"], output["frames"] = spheres, frames
        output["moving"] = np.asarray(model["head_check_indices"], int)
        output["hands"] = np.asarray(model["hand_indices"], int)
        if not len(output["moving"]) or not len(output["hands"]):
            raise ValueError("head or hand pair coverage is missing")
        for indices in (output["moving"], output["hands"]):
            if np.any(indices < 0) or np.any(indices >= len(spheres)):
                raise ValueError("collision index outside sphere array")
        output["bounds"] = _finite(model["motion_bounds_m_per_rad"], (len(spheres), 34), "motion bounds")
        if np.any(output["bounds"] < 0):
            raise ValueError("negative motion bound")
        pairs = np.asarray(model["self_pairs"], int)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs):
            raise ValueError("self/body collision pairs are missing")
        if np.any(pairs < 0) or np.any(pairs >= len(spheres)):
            raise ValueError("self/body pair outside sphere array")
        output["pairs"] = pairs
        return output

    @staticmethod
    def placed(model, native):
        """Place one compiled hand/body cover; return spheres and link frames."""
        world = [np.eye(4)]
        for joint in model["joints"]:
            source = joint["source"]
            value = 0.0 if source < 0 else native[source]
            angle = float(np.polynomial.polynomial.polyval(value, joint["coefficients"]))
            limits = joint.get("limits")
            if limits is not None and not limits[0] - 1e-9 <= angle <= limits[1] + 1e-9:
                raise ValueError(f"derived joint outside limits: {joint['name']}")
            axis = np.asarray(joint["axis"], float)
            x, y, z = axis
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], float)
            turn = np.eye(4)
            turn[:3, :3] += math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)
            world.append(world[joint["parent"]] @ np.asarray(joint["origin"]) @ turn)
        world = np.asarray(world)
        local = model["spheres"]
        frames = world[model["frames"]]
        output = local.copy()
        output[:, :3] = np.einsum("sij,sj->si", frames[:, :3, :3], local[:, :3]) + frames[:, :3, 3]
        return output, world

    def _check(self, native, radius):
        if np.any(native - radius < self.lower) or np.any(native + radius > self.upper):
            return False, "native joint interval exceeds limits"
        placed = []
        for model in self.models:
            spheres, _ = self.placed(model, native)
            allowances = model["bounds"] @ radius
            selected = model["moving"]
            moving = spheres[selected]
            gaps = (
                np.linalg.norm(moving[:, None, :3] - self.head[None, :, :3], axis=-1)
                - moving[:, None, 3]
                - self.head[None, :, 3]
                - allowances[selected, None]
            )
            if float(gaps.min()) < self.required:
                return False, f"{model['side']} head clearance below {self.required:.3f} m"
            a, b = model["pairs"].T
            gaps = (
                np.linalg.norm(spheres[a, :3] - spheres[b, :3], axis=1)
                - spheres[a, 3]
                - spheres[b, 3]
                - allowances[a]
                - allowances[b]
            )
            if float(gaps.min()) < 0.0:
                return False, f"{model['side']} articulated hand/body collision bound"
            selected = model["hands"]
            placed.append((spheres[selected], allowances[selected]))
        left, left_allowance = placed[0]
        right, right_allowance = placed[1]
        gap = (
            np.linalg.norm(left[:, None, :3] - right[None, :, :3], axis=-1)
            - left[:, None, 3]
            - right[None, :, 3]
            - left_allowance[:, None]
            - right_allowance[None, :]
        )
        hand_clearance = self.record.get("hand_clearance_m", 0.01)
        if (
            isinstance(hand_clearance, bool)
            or not isinstance(hand_clearance, int | float)
            or not math.isfinite(hand_clearance)
            or hand_clearance < 0.01
        ):
            return False, "invalid inter-hand clearance"
        if float(gap.min()) < float(hand_clearance):
            return False, "left/right hand clearance bound"
        return True, "verified compiled geometry interval"

    def validate_path(self, *, name, times, positions):
        """Check clearance and cosine segment peak speeds, failing closed."""
        if self.invalid_reason:
            return False, self.invalid_reason
        try:
            q = np.asarray(positions, float)
            t = np.asarray(times, float)
            if name != self.name:
                return False, "motion does not match geometry artifact"
            if q.ndim != 2 or q.shape[1] != 34 or not 1 <= len(q) <= MAX_PATH_POINTS:
                return False, "path must contain 1 to 4096 native 34-joint points"
            if t.shape != (len(q),) or not np.isfinite(t).all() or not np.isfinite(q).all():
                return False, "non-finite or mismatched path"
            if t[0] < 0 or (len(t) > 1 and np.any(np.diff(t) <= 0)):
                return False, "path times must be nonnegative and strictly increasing"
            if len(t) > 1 and np.any(
                EASING_PEAK_FACTOR * np.abs(np.diff(q, axis=0)) / np.diff(t)[:, None] > self.speed + 1e-9
            ):
                return False, "native joint speed exceeds limits"
            for point in q:
                ok, reason = self._check(point, self.uncertainty)
                if not ok:
                    return False, reason
            checks = len(q)
            for start, finish in zip(q[:-1], q[1:], strict=True):
                delta = np.abs(finish - start)
                steps = max(1, int(np.ceil(delta.max() / 0.005)))
                if steps > 1024:
                    return False, "path interval exceeds validation budget"
                checks += steps
                if checks > MAX_PATH_POINTS:
                    return False, "complete path exceeds validation budget"
                radius = self.uncertainty + delta / (2 * steps)
                for index in range(steps):
                    midpoint = start + ((index + 0.5) / steps) * (finish - start)
                    ok, reason = self._check(midpoint, radius)
                    if not ok:
                        return False, reason
            return True, "complete native path and continuous clearance bounds verified"
        except (KeyError, TypeError, ValueError, OverflowError, FloatingPointError) as exc:
            return False, f"invalid geometry path: {exc}"


def load_guard(path):
    """Load a guard, retaining rejection semantics for missing/invalid assets."""
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        guard = NamedHoldGeometryGuard({})
        guard.invalid_reason = f"geometry artifact unavailable: {exc}"
        return guard
    return NamedHoldGeometryGuard(record)
