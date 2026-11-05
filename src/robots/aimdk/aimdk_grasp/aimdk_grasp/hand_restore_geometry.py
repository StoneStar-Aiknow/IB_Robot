"""Articulated-hand checks for preparation, using the grasp models' verified sphere covers.

Both side models use torso_link. Their hand aliases are local to the model;
never treat the two sets of R_* links as one physical hand. The fixed lower
body is the same STAND_DEFAULT obstacle used by X2 grasp execution.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .hand_restore import PATH_HEAD_HOLD_RAD, PATH_SHAPE_HOLD_RAD, RestoreError
from .joints import DIP_FOLLOWERS, HEAD, LEFT_ARM, RIGHT_ARM
from .leg_clearance import LEG_MODEL_PADDING_M, load_leg_hulls

ARM_SAMPLE_RAD = 0.004
CHUNK = 64


class RelativeBounds:
    """Joint-arc bounds from the FK tree; common ancestors cancel in relative distances."""

    def __init__(self, model):
        self.model = model
        parents = {j["child"]: (i, j) for i, j in enumerate(model.joints)}
        self.levers = {}
        for link in model.links:
            balls = model.local[model.link_slices[link]]
            reach = float(np.max(np.linalg.norm(balls[:, :3], axis=1) + balls[:, 3]))
            current, chain = link, {}
            while current in parents:
                index, joint = parents[current]
                if joint["bounds"] is not None:
                    chain[index] = reach
                reach += float(np.linalg.norm(joint["transform"][:3, 3]))
                current = joint["parent"]
            self.levers[link] = chain

    def angles(self, feedback, arm_delta, finger_radius=None, head_radius=None):
        finger_radius = PATH_SHAPE_HOLD_RAD if finger_radius is None else finger_radius
        head_radius = PATH_HEAD_HOLD_RAD if head_radius is None else head_radius
        model = self.model
        uncertainty = dict(zip(LEFT_ARM + RIGHT_ARM, arm_delta, strict=True))
        uncertainty.update(dict.fromkeys(HEAD, head_radius))
        native = model.mapping.check(feedback.hand(model.side))
        alias = model.mapping.to_alias(native)
        lower = model.mapping.to_alias(np.maximum(native - finger_radius, model.mapping.lower))
        upper = model.mapping.to_alias(np.minimum(native + finger_radius, model.mapping.upper))
        uncertainty.update(
            {name: max(abs(lower[name] - value), abs(upper[name] - value)) for name, value in alias.items()}
        )
        for name, (master, coefficients) in DIP_FOLLOWERS.items():
            lo, hi = sorted((lower[master], upper[master]))
            roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(coefficients))
            points = [lo, hi, *(float(r.real) for r in roots if abs(r.imag) < 1e-12 and lo <= r.real <= hi)]
            base = np.polynomial.polynomial.polyval(alias[master], coefficients)
            uncertainty[name] = float(np.max(np.abs(np.polynomial.polynomial.polyval(points, coefficients) - base)))
        inputs = np.array([uncertainty.get(name, 0.0) for name in model.fk_inputs])
        return inputs[model.fk_source] * np.abs(model.fk_scale)

    def pair(self, a, b, angles):
        left, right = self.levers[a], self.levers[b]
        return sum(left[i] * angles[i] for i in left.keys() - right.keys()) + sum(
            right[i] * angles[i] for i in right.keys() - left.keys()
        )

    def world(self, link, angles):
        return sum(radius * angles[i] for i, radius in self.levers[link].items())


def sphere_box_clearance(spheres, dimensions, pose):
    """Conservative sphere-to-solid-box clearance, including negative values inside the box."""
    pose = np.asarray(pose, float)
    rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    local = (spheres[..., :3] - pose[:3]) @ rotation
    gap = np.abs(local) - np.asarray(dimensions, float) / 2
    outside = np.linalg.norm(np.maximum(gap, 0), axis=-1)
    inside = np.minimum(np.max(gap, axis=-1), 0)
    return outside + inside - spheres[..., 3]


@dataclass(frozen=True)
class SideSweep:
    geometry: object
    side: str

    def check(self, feedback, frames, *, left_arm=None, right_arm=None):
        hands = {side: feedback.hand(side) for side in ("left", "right")}
        hands[self.side] = np.asarray(frames, float)
        return self.geometry.check(feedback, hands, left_arm=left_arm, right_arm=right_arm)


class DualHandCollision:
    def __init__(self, checks):
        if set(checks) != {"left", "right"}:
            raise ValueError("both physical hand models are required")
        self.checks = dict(checks)
        self.indices = {}
        self.leg_hulls = {}
        self.bounds = {}
        self.arm_radius = 0.0
        for side, check in self.checks.items():
            if check.mapping.side != side or check.model.side != side or check.model.base != "torso_link":
                raise ValueError("hand model side or reference frame mismatch")
            self.indices[side] = [i for i, name in enumerate(check.model.sphere_links) if name.startswith("R_")]
            if not self.indices[side]:
                raise ValueError("hand model has no finger/palm collision spheres")
            self.arm_radius = max(self.arm_radius, check.model.arm_reach()[1])
            self.bounds[side] = RelativeBounds(check.model)
            self.leg_hulls.update(load_leg_hulls(side))

    def sweep(self, side):
        return SideSweep(self, side)

    def check(
        self,
        feedback,
        hands,
        *,
        left_arm=None,
        right_arm=None,
        body_margin=0.0,
        arm_delta=None,
        sample_offset=0,
        finger_radius=PATH_SHAPE_HOLD_RAD,
        head_radius=PATH_HEAD_HOLD_RAD,
    ):
        """Check every supplied frame: both hands, opposite arms, torso/head and fixed legs."""
        values = {side: np.asarray(hands[side], float) for side in self.checks}
        count = max((len(value) if value.ndim == 2 else 1) for value in values.values())
        for arm in (left_arm, right_arm):
            if arm is not None and np.asarray(arm).ndim == 2:
                count = max(count, len(arm))
        placed = {}
        hand_margins = {}
        sphere_margins = {}
        minimum = float("inf")
        changing = set()
        for names, arm in ((LEFT_ARM, left_arm), (RIGHT_ARM, right_arm)):
            if arm is not None and np.asarray(arm).ndim == 2:
                changing.update(name for name, span in zip(names, np.ptp(arm, axis=0), strict=True) if span != 0)
        for side, check in self.checks.items():
            frames = np.broadcast_to(values[side], (count, 10))
            spheres = check.place(feedback, frames, left_arm=left_arm, right_arm=right_arm)
            pair_margin = None
            pair_constant = None
            hand_margins[side] = body_margin
            link_margins = {name: body_margin for name in check.model.links if name.startswith("R_")}
            hold_margins = dict.fromkeys(link_margins, 0.0)
            if arm_delta is not None:
                bounds = self.bounds[side]
                angles = bounds.angles(feedback, arm_delta, finger_radius, head_radius)

                def pair_margin(a, b, bounds=bounds, angles=angles):
                    return bounds.pair(a, b, angles)

                link_margins = {link: bounds.world(link, angles) for link in link_margins}
                hold_angles = bounds.angles(feedback, np.zeros(14), finger_radius, head_radius)
                hold_margins = {link: bounds.world(link, hold_angles) for link in link_margins}
                hand_margins[side] = max(link_margins.values())
                # check_arm_path keeps fingers/head fixed. Only classify rigid
                # pairs when all provided hand values are constant in time.
                if all(value.ndim == 1 or not np.ptp(value, axis=0).any() for value in values.values()):
                    moving = {i for i, joint in enumerate(check.model.joints) if joint["name"] in changing}

                    def pair_constant(a, b, bounds=bounds, moving=moving):
                        return not ((bounds.levers[a].keys() ^ bounds.levers[b].keys()) & moving)

            minimum = min(
                minimum,
                check.check_spheres(
                    spheres,
                    body_margin=body_margin,
                    pair_margin=pair_margin,
                    pair_constant=pair_constant,
                ),
            )
            placed[side] = spheres[:, self.indices[side]]
            links = np.asarray(check.model.sphere_links)[self.indices[side]]
            sphere_margins[side] = np.array([link_margins[link] for link in links])
            required = LEG_MODEL_PADDING_M + np.broadcast_to(sphere_margins[side], placed[side].shape[:2])
            for name, hull in self.leg_hulls.items():
                clearance = hull.lower_bound(placed[side])
                close = clearance < required + 1e-9
                # A broad-phase overlap is not a collision. Refine only those
                # spheres for which bounding volumes cannot prove separation.
                if close.any():
                    clearance[close] = hull.clearance(placed[side][close])
                minimum = min(minimum, float(np.min(clearance - LEG_MODEL_PADDING_M)))
                slack = clearance - required
                if np.min(slack) < -1e-6:
                    frame, ball = np.unravel_index(np.argmin(slack), slack.shape)
                    link = links[ball]
                    raise RestoreError(
                        "HAND_SWEEP_COLLISION",
                        f"{side} hand/{link} vs {name}: clearance={clearance[frame, ball]:.6f} m, "
                        f"required={required[frame, ball]:.6f} m "
                        f"(model={LEG_MODEL_PADDING_M:.6f}, "
                        f"arm_bound={link_margins[link] - hold_margins[link]:.6f}, "
                        f"shape_hold={hold_margins[link]:.6f}); sample={sample_offset + frame}",
                    )
        left, right = placed["left"], placed["right"]
        lo = {side: (balls[..., :3] - balls[..., 3:]).min(axis=1) for side, balls in placed.items()}
        hi = {side: (balls[..., :3] + balls[..., 3:]).max(axis=1) for side, balls in placed.items()}
        gap = np.maximum(np.maximum(lo["left"] - hi["right"], lo["right"] - hi["left"]), 0)
        # The AABB separation is a lower bound; distant hands need no quadratic check.
        cross_margin = sum(hand_margins.values()) if arm_delta is not None else body_margin
        nearby = np.linalg.norm(gap, axis=1) <= cross_margin
        for frame in np.flatnonzero(nearby):
            a, b = left[frame], right[frame]
            for start in range(0, len(a), 64):
                block = a[start : start + 64]
                distance = np.linalg.norm(block[:, None, :3] - b[None, :, :3], axis=-1)
                clearance = distance - block[:, None, 3] - b[None, :, 3]
                required = (
                    sphere_margins["left"][start : start + len(block), None] + sphere_margins["right"][None, :]
                    if arm_delta is not None
                    else body_margin
                )
                minimum = min(minimum, float(np.min(clearance)))
                slack = clearance - required
                if np.min(slack) < -1e-6:
                    i, j = np.unravel_index(np.argmin(slack), slack.shape)
                    margin = required[i, j] if arm_delta is not None else required
                    raise RestoreError(
                        "HAND_SWEEP_COLLISION",
                        f"left/right hands, clearance={clearance[i, j]:.6f} m, required={margin:.6f} m; "
                        f"sample={sample_offset + frame}",
                    )
        return minimum

    def check_arm_path(
        self,
        feedback,
        joints,
        positions,
        *,
        guard=lambda: None,
        start_tolerance_rad=0.0,
        uncertainty=None,
        finger_radius=PATH_SHAPE_HOLD_RAD,
        head_radius=PATH_HEAD_HOLD_RAD,
    ):
        """Check the actual linear joint segments with the measured finger configuration.

        Subdivide at 0.004 rad/joint. Each joint contributes half a segment's
        angular change times its descendant geometry's reach. Shared ancestors
        cancel for pairs in the same FK tree. Finger/head preservation errors
        are propagated separately, including polynomial DIP coupling.
        """
        joints = tuple(joints)
        if not np.isfinite(start_tolerance_rad) or start_tolerance_rad < 0:
            raise ValueError("start tolerance must be finite and nonnegative")
        q = np.asarray(positions, float)
        expected = LEFT_ARM + RIGHT_ARM
        if (
            len(joints) != 14
            or set(joints) != set(expected)
            or q.ndim != 2
            or q.shape[1] != 14
            or not len(q)
            or not np.isfinite(q).all()
        ):
            raise ValueError("finite complete dual-arm trajectory required")
        uncertainty = np.zeros(14) if uncertainty is None else np.asarray(uncertainty, float)
        if uncertainty.shape != (14,) or not np.isfinite(uncertainty).all() or (uncertainty < 0).any():
            raise ValueError("invalid arm uncertainty")
        if not all(np.isfinite(v) and v >= 0 for v in (finger_radius, head_radius)):
            raise ValueError("invalid finger/head uncertainty")
        uncertainty = uncertainty[[joints.index(name) for name in expected]]
        q = q[:, [joints.index(name) for name in expected]]
        rows, margins = [q[0]], [np.zeros(14)]
        for start, end in zip(q[:-1], q[1:], strict=True):
            count = max(1, int(np.ceil(np.max(np.abs(end - start)) / ARM_SAMPLE_RAD)))
            delta = np.abs(end - start) / count
            margin = delta / 2
            margins[-1] = np.maximum(margins[-1], margin)
            rows.extend(np.linspace(start, end, count + 1)[1:])
            margins.extend([margin] * count)
        rows = np.asarray(rows)
        hands = {side: feedback.hand(side) for side in self.checks}
        for start in range(0, len(rows), CHUNK):
            guard()
            block = rows[start : start + CHUNK]
            try:
                self.check(
                    feedback,
                    hands,
                    left_arm=block[:, :7],
                    right_arm=block[:, 7:],
                    arm_delta=np.max(margins[start : start + CHUNK], axis=0) + start_tolerance_rad + uncertainty,
                    finger_radius=finger_radius,
                    head_radius=head_radius,
                    sample_offset=start,
                )
            except RestoreError as exc:
                raise RestoreError(exc.code, f"{exc}; start_tolerance_rad={start_tolerance_rad:.6f}") from exc
        guard()
