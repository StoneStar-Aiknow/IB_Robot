"""Open the OmniHands from their measured pose, inside a task another component already owns.

This is the supervised maintenance procedure of the X2 deployment (hold both
hands until the arm command board takes over at gains 40/2, open the left hand,
then the right, observe 2 s after the release) as a library. It reuses this
package's hand logic: the native/alias mapping and open preset
(:class:`aimdk_grasp.hand.HandMapping`), the planning model's sphere cover
(:class:`aimdk_grasp.robot_spheres.RobotSphereModel`) and the stamped feedback
(:class:`aimdk_grasp.feedback.Feedback`). Output goes only through
:class:`RestorePorts`, whose ROS implementation is the runtime's hand stream
(:mod:`aimdk_grasp.hand_restore_ros`); nothing here starts a process or talks
to the vendor.

What the caller owns, and this module does not: the runtime mode, the arms and
the task. Where the maintenance tool required an idle Gateway, this requires
the Gateway to report the *caller's own task* as the active one, so a task may
run it and nobody else's task can.

Fail closed, never around a check. A hand whose collision model is missing is
reported by :func:`build_hand_plans` and refused by :meth:`HandRestorer.preflight`
before any output; so is an arm plan that would run with a hand shape the arm
planner has no model for (:meth:`HandRestorer.arm_planning_gap`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np

from .feedback import Feedback
from .hand import OPEN_HAND_TOLERANCE_RAD, HandMapping, hand_links
from .joints import ABSENT_HAND_COMMAND, BODY_JOINTS, LEFT_ARM, RIGHT_ARM, arm_joints, other_side
from .joints import installed_hands as parse_installed_hands
from .omnihand import HAND_HZ, HAND_STEP_RAD

SIDES = ("left", "right")
#: Why ``build_hand_plans`` has no plan for a hand the configuration declares absent.
NOT_INSTALLED = "not installed (backend_options.installed_hands)"
#: Maximum age of body/hand and arm-command feedback (s).
#: Temporary commissioning override; the previous on-robot limit was 0.3 s.
FEEDBACK_MAX_AGE_S = 1.0
RUNTIME_STATUS_MAX_AGE_S = 2.0
GATEWAY_STATUS_MAX_AGE_S = 1.0
#: An arm or the head may not move this far (rad) while a hand is driven.
BODY_HOLD_RAD = 0.025
# Finger drift is covered by the same bound in articulated collision checks.
PATH_SHAPE_HOLD_RAD = 0.1
PATH_HEAD_HOLD_RAD = 0.005
#: Another controller changed an arm target if the command board moved this far (rad).
COMMAND_DRIFT_RAD = 0.005
#: A hand lagging the previous command by more than this (rad) is not tracking.
TRACKING_RAD = 0.2
#: Two seconds of observation after the hands are released.
OBSERVE_S = 2.0
#: Measured within tolerance for this long (s) counts as settled.
SETTLE_S = 0.2
#: One drive loop (all frames and the settle wait) may not take longer.
DRIVE_MAX_S = 15.0
STALL_S = 0.15
HOLD_FRAMES_S = 5.0
OUTPUT_READY_S = 3.0
#: The arm command board after the stream took over, and the vendor's own before.
TAKEOVER_STIFFNESS, TAKEOVER_DAMPING = 40.0, 2.0
VENDOR_STIFFNESS = np.array([30, 20, 20, 50, 50, 20, 20] * 2, dtype=float)
VENDOR_DAMPING = 1.0


class RestoreError(RuntimeError):
    """A refusal or failure with a stable code (``HAND_*``, ``TASK_POSE_TIMEOUT``, ``POSE_CANCELLED``)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class CapabilityMissing(RestoreError):
    def __init__(self, message: str):
        super().__init__("HAND_CAPABILITY_MISSING", message)


# -- measurements -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandBoard:
    """What the arm controller is commanding, with its gains; ``age_s`` since it was received."""

    names: tuple[str, ...]
    position: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    age_s: float


@dataclass(frozen=True)
class RuntimeState:
    ok: bool
    detail: str
    active_mode: str
    age_s: float


@dataclass(frozen=True)
class GatewayState:
    control_plane_ready: bool
    motion_authorized: bool
    busy: bool
    active_task_id: str
    age_s: float


@dataclass(frozen=True)
class Observation:
    feedback: Feedback
    feedback_age_s: float
    command: CommandBoard | None
    runtime: RuntimeState | None
    gateway: GatewayState | None


class RestorePorts(Protocol):
    """Everything a restore touches. ``monotonic`` is the clock deadlines are measured on."""

    def observe(self) -> Observation: ...

    def refusal(self) -> str | None:
        """Why the runtime would not take a hand command now, or None."""

    def publish(self, left: np.ndarray, right: np.ndarray) -> None: ...

    def release(self) -> None: ...

    def monotonic(self) -> float: ...

    def wait(self, seconds: float) -> None: ...


@dataclass
class RestoreContext:
    """One execution: the task that owns it, the shared deadline and cancel, and where progress goes."""

    task_id: str
    deadline: float
    cancelled: Callable[[], bool]
    progress: Callable[[str, float], None] = lambda *_: None


# -- pure helpers -----------------------------------------------------------------------------


def sweep_frames(start, target, step_rad: float = HAND_STEP_RAD) -> np.ndarray:
    """The measured start, then straight-line frames no more than ``step_rad`` per joint apart."""
    start, target = np.asarray(start, float), np.asarray(target, float)
    if start.shape != (10,) or target.shape != (10,) or not np.isfinite([start, target]).all():
        raise ValueError("Ten finite hand values required")
    if not np.isfinite(step_rad) or not 0 < step_rad <= HAND_STEP_RAD:
        raise ValueError(f"Hand step must be finite and in (0, {HAND_STEP_RAD}] rad")
    count = max(1, int(np.ceil(np.max(np.abs(target - start)) / step_rad)))
    return np.linspace(start, target, count + 1)


def gains_class(board: CommandBoard) -> str:
    """``ready`` (40/2: the stream owns the arms), ``vendor_default`` (needs the hands held first) or ``other``."""
    stiffness, damping = np.asarray(board.stiffness, float), np.asarray(board.damping, float)
    if stiffness.shape != (14,) or damping.shape != (14,):
        return "other"
    if np.allclose(stiffness, TAKEOVER_STIFFNESS) and np.allclose(damping, TAKEOVER_DAMPING):
        return "ready"
    if np.allclose(stiffness, VENDOR_STIFFNESS) and np.allclose(damping, VENDOR_DAMPING):
        return "vendor_default"
    return "other"


# -- the collision model of a moving hand -----------------------------------------------------


class HandSweepCheck:
    """Limits and hand-involving self-collision along a sweep, in the planning model that mounts the hand as ``R_*``.

    Each side's planning model carries its physical OmniHand as ``R_*``
    aliases, with the other arm and head as locked bodies. Only pairs with an ``R_*`` link are
    checked, so the hand-link set must be exactly the model's hand
    (:func:`aimdk_grasp.hand.hand_links`); anything else means the mapping is
    not what this check assumes, and it is refused rather than run on a guess.
    """

    def __init__(self, mapping: HandMapping, directory=None):
        from .robot_model import collision_robot_config, config_directory, planning_model
        from .robot_spheres import RobotSphereModel

        directory = config_directory(mapping.side) if directory is None else directory
        self.mapping = mapping
        self.model = RobotSphereModel(mapping, directory)
        mounted = {link for link in self.model.links if link.startswith("R_")}
        if mounted != set(hand_links(planning_model(directory))):
            raise CapabilityMissing("the collision model's R_ links are not exactly the mounted hand's links")
        kin = collision_robot_config(directory)["robot_cfg"]["kinematics"]
        self.ignored = {frozenset((a, b)) for a, others in kin["self_collision_ignore"].items() for b in others}

    def check(self, feedback: Feedback, frames, *, left_arm=None, right_arm=None) -> float:
        """Smallest hand clearance (m) along ``frames``; raises when a frame leaves limits or touches.

        ``left_arm`` / ``right_arm`` (7 joints) replace the measured arms, to check a hand sweep against
        the pose the arms are about to reach.
        """
        return self.check_spheres(self.place(feedback, frames, left_arm=left_arm, right_arm=right_arm))

    def place(self, feedback: Feedback, frames, *, left_arm=None, right_arm=None):
        """Place this side's verified spheres without dropping the other arm or head."""
        frames = np.asarray(frames, float)
        arms = {"left": left_arm, "right": right_arm}
        selected = arms[self.mapping.side]
        arm = feedback.vector(arm_joints(self.mapping.side)) if selected is None else np.asarray(selected, float)
        positions = np.broadcast_to(arm, (len(frames), 7)) if arm.ndim == 1 else arm
        spheres, bounded = self.model.spheres(
            feedback, positions, other=arms[other_side(self.mapping.side)], hand=frames
        )
        if not bounded.all():
            raise RestoreError("HAND_SWEEP_OUT_OF_LIMITS", "hand sweep or measured body exceeds planning-model limits")
        return spheres

    def check_spheres(self, spheres, *, body_margin=0.0, pair_margin=None, pair_constant=None) -> float:
        """Check one hand, returning a conservative clearance lower bound.

        AABB separation proves distant link pairs safe without allocating their
        full sphere-distance matrix. Every close frame still uses the exact test.
        """
        minimum = float("inf")
        groups = {name: spheres[:, self.model.link_slices[name]] for name in self.model.links}
        boxes = {
            name: ((balls[..., :3] - balls[..., 3:]).min(axis=1), (balls[..., :3] + balls[..., 3:]).max(axis=1))
            for name, balls in groups.items()
        }
        for i, a in enumerate(self.model.links):
            for b in self.model.links[i + 1 :]:
                if not (a.startswith("R_") or b.startswith("R_")) or frozenset((a, b)) in self.ignored:
                    continue
                margin = (
                    pair_margin(a, b)
                    if pair_margin is not None
                    else (0.0 if a.startswith("R_") and b.startswith("R_") else body_margin)
                )
                lo_a, hi_a = boxes[a]
                lo_b, hi_b = boxes[b]
                group_a, group_b = groups[a], groups[b]
                if pair_constant is not None and pair_constant(a, b):
                    # The changing joints are all common ancestors: the same
                    # rigid transform acts on both links, preserving distance.
                    lo_a, hi_a, lo_b, hi_b = lo_a[:1], hi_a[:1], lo_b[:1], hi_b[:1]
                    group_a, group_b = group_a[:1], group_b[:1]
                gap = np.maximum(np.maximum(lo_a - hi_b, lo_b - hi_a), 0.0)
                lower = np.linalg.norm(gap, axis=1)
                nearby = lower <= margin + 1e-6
                if not nearby.all():
                    minimum = min(minimum, float(lower[~nearby].min()))
                if not nearby.any():
                    continue
                sa, sb = group_a[nearby], group_b[nearby]
                # Bound narrow-phase memory independently of the outer FK batch.
                clearance = float("inf")
                for start in range(0, len(sa), 8):
                    left, right = sa[start : start + 8], sb[start : start + 8]
                    a_xyz, b_xyz = left[..., :3], right[..., :3]
                    squared = np.sum(a_xyz * a_xyz, axis=-1)[:, :, None] + np.sum(b_xyz * b_xyz, axis=-1)[:, None, :]
                    squared -= 2 * (a_xyz @ np.swapaxes(b_xyz, -1, -2))
                    distances = np.sqrt(np.maximum(squared, 0.0))
                    clearance = min(clearance, float(np.min(distances - left[:, :, None, 3] - right[:, None, :, 3])))
                minimum = min(minimum, clearance)
                if clearance < margin - 1e-6:
                    raise RestoreError(
                        "HAND_SWEEP_COLLISION",
                        f"insufficient hand clearance: {a}/{b}, clearance={clearance:.6f} m, required={margin:.6f} m",
                    )
        return minimum


@dataclass
class HandPlan:
    """One hand this deployment can open: its mapping, open pose and collision check."""

    side: str
    mapping: HandMapping
    open_pose: np.ndarray
    sweep: object  # HandSweepCheck, or anything with .check(feedback, frames, left_arm=, right_arm=)


def build_hand_plans(grasp_execution: Mapping) -> tuple[dict[str, HandPlan], dict[str, str]]:
    """The hands with a verified model, and for each other hand why it has none.

    The open preset belongs to the deployed GraspGen-X gripper description,
    as for the grasp itself. A hand needs an ``R_*``-mounted collision model
    and a native/alias mapping. Each physical side uses its own planning model,
    independently of the arm selected for a subsequent grasp.
    """
    from .backend import grasp_installed_hands, x2_robot

    installed = grasp_installed_hands(grasp_execution)
    plans: dict[str, HandPlan] = {}
    missing = {}
    for side in SIDES:
        if side not in installed:
            missing[side] = NOT_INSTALLED
            continue
        robot = x2_robot(
            {**grasp_execution, "backend_options": {**(grasp_execution.get("backend_options") or {}), "arm_side": side}}
        )
        try:
            plans[side] = HandPlan(side, robot.hand, robot.hand.preset("open"), HandSweepCheck(robot.hand))
        except RestoreError as exc:
            missing[side] = str(exc)
    if plans and set(plans) == set(installed):
        from .hand_restore_geometry import DualHandCollision

        geometry = DualHandCollision({side: plan.sweep for side, plan in plans.items()})
        for side, plan in plans.items():
            plan.sweep = geometry.sweep(side)
    else:
        # Even a one-hand request needs every other installed articulated hand as an obstacle.
        missing.update({side: "every installed hand model is required for combined collision checks" for side in plans})
        plans.clear()
    return plans, missing


# -- the restore ------------------------------------------------------------------------------


@dataclass
class HandRestorer:
    """Takeover, left-then-right opening and confirmation, under one context's deadline and cancel."""

    ports: RestorePorts
    plans: Mapping[str, HandPlan]
    missing: Mapping[str, str] = field(default_factory=dict)
    required_mode: str = "policy_stream"
    step_rad: float = HAND_STEP_RAD
    # The deployed arm planner has rigid dummy hands, not a verified finger envelope.
    # Only a backend with that coverage (or a Mock) may enable combined arm motion.
    arm_motion_covered: bool = False
    installed_hands: tuple[str, ...] = SIDES
    _baseline: dict | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.installed_hands = parse_installed_hands(self.installed_hands)

    @property
    def installed(self) -> tuple[str, ...]:
        """Explicit mounting configuration; missing feedback never changes it."""
        return self.installed_hands

    # -- shared guards --

    def _guard(self, ctx: RestoreContext) -> None:
        if ctx.cancelled():
            raise RestoreError("POSE_CANCELLED", "hand preparation canceled")
        if self.ports.monotonic() >= ctx.deadline:
            raise RestoreError("TASK_POSE_TIMEOUT", "hand preparation exhausted the task-pose budget")

    def _observe(self, ctx: RestoreContext) -> Observation:
        self._guard(ctx)
        obs = self.ports.observe()
        self._guard(ctx)
        if not obs.feedback.healthy or not 0 <= obs.feedback_age_s <= FEEDBACK_MAX_AGE_S:
            raise RestoreError(
                "HAND_FEEDBACK_STALE",
                f"body/hand feedback unhealthy, stale, or from a future stamp: "
                f"healthy={obs.feedback.healthy}, age={obs.feedback_age_s:.6f}s, max={FEEDBACK_MAX_AGE_S:.6f}s",
            )
        runtime = obs.runtime
        if runtime is None or not 0 <= runtime.age_s <= RUNTIME_STATUS_MAX_AGE_S:
            raise RestoreError("HAND_RUNTIME_NOT_READY", "runtime status missing or stale")
        if not runtime.ok:
            raise RestoreError("HAND_RUNTIME_NOT_READY", f"runtime stopped, faulted, or inactive: {runtime.detail}")
        if runtime.active_mode != self.required_mode:
            raise RestoreError(
                "HAND_RUNTIME_NOT_READY", f"runtime mode is {runtime.active_mode!r}, hands need {self.required_mode!r}"
            )
        gateway = obs.gateway
        if gateway is None or not 0 <= gateway.age_s <= GATEWAY_STATUS_MAX_AGE_S:
            age = None if gateway is None else gateway.age_s
            raise RestoreError(
                "HAND_GATEWAY_NOT_OWNER", f"Gateway status missing or stale: age={age}, max={GATEWAY_STATUS_MAX_AGE_S}s"
            )
        if not (gateway.control_plane_ready and gateway.motion_authorized):
            raise RestoreError("HAND_GATEWAY_NOT_OWNER", "Gateway control plane not ready or motion not authorized")
        if not ctx.task_id or not gateway.busy or gateway.active_task_id != ctx.task_id:
            raise RestoreError(
                "HAND_GATEWAY_NOT_OWNER",
                "the Gateway does not report this execution's own task as the active one",
            )
        if obs.command is None or not 0 <= obs.command.age_s <= FEEDBACK_MAX_AGE_S:
            raise RestoreError("HAND_FEEDBACK_STALE", "arm command feedback missing or stale")
        board = obs.command
        if tuple(board.names) != LEFT_ARM + RIGHT_ARM or any(
            np.asarray(values).shape != (14,) or not np.isfinite(values).all()
            for values in (board.position, board.stiffness, board.damping)
        ):
            raise RestoreError("HAND_FEEDBACK_STALE", "arm command board has invalid names or values")
        try:
            obs.feedback.vector(BODY_JOINTS)
        except (KeyError, ValueError) as exc:
            raise RestoreError("HAND_FEEDBACK_STALE", f"body feedback: {exc}") from None
        return obs

    def _hand(self, obs: Observation, side: str) -> np.ndarray:
        """A measured hand, limit-checked by its mapping when it has one."""
        try:
            native = obs.feedback.hand(side)
        except ValueError as exc:
            raise RestoreError("HAND_FEEDBACK_STALE", str(exc)) from None
        plan = self.plans.get(side)
        try:
            if plan is not None:
                return plan.mapping.check(native)
        except ValueError as exc:
            raise RestoreError("HAND_FEEDBACK_STALE", f"{side} hand feedback: {exc}") from None
        native = np.asarray(native, float)
        if native.shape != (10,) or not np.isfinite(native).all():
            raise RestoreError("HAND_FEEDBACK_STALE", f"{side} hand feedback is not ten finite values")
        return native.copy()

    # -- capability and preflight --

    def require_capabilities(self, sides: Sequence[str]) -> None:
        for side in sides:
            if side not in self.installed or side not in self.plans:
                raise CapabilityMissing(f"cannot open the {side} hand: {self.missing.get(side, 'no model')}")

    def arm_planning_gap(self, obs: Observation) -> str:
        """Why the arm planner cannot be trusted with the measured hands, or an empty string.

        The dual-arm planner's official MJCF has rigid dummy hands. The shared
        articulated checker supplements it with both measured finger shapes;
        check_arm_path must still accept the final trajectory before dispatch.
        Older backends without that checker require explicit open-hand coverage
        and both hands open; recognizing a preset alone does not prove coverage.
        """
        if self.geometry is not None:
            for side in self.installed:
                self._hand(obs, side)
            return ""
        reasons = []
        if not self.arm_motion_covered:
            reasons.append("rigid dummy hands do not establish collision coverage for articulated fingers")
        for side in self.installed:
            plan = self.plans.get(side)
            if plan is None:
                reasons.append(f"{side} hand shape cannot be verified ({self.missing.get(side, 'no model')})")
                continue
            error = float(np.max(np.abs(self._hand(obs, side) - plan.open_pose)))
            if error > OPEN_HAND_TOLERANCE_RAD:
                reasons.append(f"{side} hand is {error:.3f} rad from open")
        return "; ".join(reasons)

    @property
    def geometry(self):
        """The shared real geometry checker, absent for older/synthetic backends."""
        checks = [getattr(self.plans[side].sweep, "geometry", None) for side in self.installed if side in self.plans]
        complete = bool(checks) and len(checks) == len(self.installed) and checks[0] is not None
        return checks[0] if complete and all(check is checks[0] for check in checks) else None

    def arm_baseline(self, ctx):
        obs = self._observe(ctx)
        if self.geometry is None:
            raise CapabilityMissing("no combined articulated-hand path checker")
        return obs

    def validation_conditions(self, ctx):
        """Optional live standing/load evidence. Missing transport evidence cannot enable reuse."""
        self._guard(ctx)
        observe = getattr(self.ports, "validation_conditions", None)
        return observe() if observe is not None else None

    def check_arm_path(self, ctx, joints, positions, *, start_tolerance_rad=0.0):
        """Validate articulated hands along the exact trajectory before its dispatch."""
        obs = self._observe(ctx)
        if self.geometry is None:
            raise CapabilityMissing("no combined articulated-hand path checker")

        def guard():
            # Geometry can take seconds. Refresh live ownership and feedback at
            # each bounded chunk, without changing the original checked shape.
            self.guard_arm_motion(ctx, obs)

        self.geometry.check_arm_path(obs.feedback, joints, positions, guard=guard)
        if start_tolerance_rad:
            # Only the measured-start -> first command transition can differ.
            # Later command segments have the exact checked positions.
            self.geometry.check_arm_path(
                obs.feedback,
                joints,
                positions[:2],
                guard=guard,
                start_tolerance_rad=start_tolerance_rad,
            )
        # Computation may have taken time: do not dispatch using a different hand shape.
        guard()
        return obs

    def guard_arm_motion(self, ctx, initial, phase="path_check"):
        """Keep the measured finger/head shape used by the path check; never re-baseline during motion."""
        from .joints import HEAD

        obs = self._observe(ctx)
        for side in self.installed:
            current, baseline = self._hand(obs, side), self._hand(initial, side)
            index = int(np.argmax(np.abs(current - baseline)))
            delta = float(abs(current[index] - baseline[index]))
            if delta > PATH_SHAPE_HOLD_RAD:
                raise RestoreError(
                    "HAND_PRESERVED_MOVED",
                    f"phase={phase}; {side} hand joint_index={index}; initial={baseline[index]:.6f} rad; "
                    f"current={current[index]:.6f} rad; delta={delta:.6f} rad; limit={PATH_SHAPE_HOLD_RAD:.6f} rad",
                )
        if np.max(np.abs(obs.feedback.vector(HEAD) - initial.feedback.vector(HEAD))) > PATH_HEAD_HOLD_RAD:
            raise RestoreError("HAND_BODY_MOVED", "head changed during arm motion")
        return obs

    def _check_hand_sweep(self, ctx, plan, feedback, frames, initial, **arms):
        # Opening sweeps need the same live-state maintenance as arm sweeps.
        # All original samples are checked; only work is split into chunks.
        for start in range(0, len(frames), 32):
            self.guard_arm_motion(ctx, initial)
            plan.sweep.check(feedback, frames[start : start + 32], **arms)
        self.guard_arm_motion(ctx, initial)

    def preflight(
        self,
        ctx: RestoreContext,
        sides: Sequence[str],
        *,
        arm_target: Mapping[str, float],
        arm_must_move: bool,
    ) -> float:
        """Refuse before any output; returns the least time (s) the hand phase will take.

        Checks the capability of every hand to open, the caller's ownership of the Gateway task, runtime and
        feedback health, the arm gains, that the arm planner covers the hands when the arms must move, that
        every sweep is collision-free against the pose the arms will have, and that the runtime would accept
        a hand command.
        """
        self.require_capabilities(sides)
        if not sides or tuple(sides) != tuple(side for side in SIDES if side in sides):
            raise CapabilityMissing("hand preparation requires unique sides ordered left before right")
        prepare = getattr(self.ports, "prepare", None)
        if prepare is not None:
            prepare(ctx)
        obs = self._observe(ctx)
        if gains_class(obs.command) == "other":
            raise RestoreError(
                "HAND_ARM_GAINS_UNVERIFIED", "arm gains are neither 40/2 (stream owns the arms) nor the vendor default"
            )
        if arm_must_move:
            gap = self.arm_planning_gap(obs)
            if gap:
                raise RestoreError(
                    "HAND_SHAPE_NOT_COVERED",
                    f"the arm planner has no model for the current hands, so the arms are not moved: {gap}",
                )
        try:
            left_arm = [arm_target[name] for name in LEFT_ARM]
            right_arm = [arm_target[name] for name in RIGHT_ARM]
        except KeyError as exc:
            raise RestoreError("HAND_SWEEP_OUT_OF_LIMITS", f"arm target lacks {exc}") from None
        if "right" in sides and "left" not in sides and "left" in self.installed:
            self._require_left_open(obs)
        seconds = OBSERVE_S + (SETTLE_S if gains_class(obs.command) != "ready" else 0.0)
        projected = obs.feedback
        for side in sides:
            plan = self.plans[side]
            frames = sweep_frames(self._hand(obs, side), plan.open_pose, self.step_rad)
            self._check_hand_sweep(ctx, plan, projected, frames, obs, left_arm=left_arm, right_arm=right_arm)
            projected = replace(projected, **{f"{side}_hand": plan.open_pose})
            seconds += len(frames) / HAND_HZ + SETTLE_S
        for side in self.installed:
            self._hand(obs, side)
        self._wait_output_ready(ctx)
        return seconds

    def _wait_output_ready(self, ctx: RestoreContext) -> None:
        # The stream owns a status reader of its own; wait for its first heartbeat, with nothing buffered.
        until = self.ports.monotonic() + OUTPUT_READY_S
        while self.ports.refusal():
            if self.ports.monotonic() >= until:
                raise RestoreError("HAND_OUTPUT_REFUSED", f"runtime refuses hand output: {self.ports.refusal()}")
            self.ports.wait(0.02)
            self._observe(ctx)

    # -- the hand phase --

    def begin(self, ctx: RestoreContext) -> bool:
        """Hold the measured hands for takeover before arm motion, without commanding the arms.

        Returns whether a takeover was needed.
        """
        obs = self._observe(ctx)
        self._baseline = {side: self._hand(obs, side) for side in self.installed}
        kind = gains_class(obs.command)
        if kind == "ready":
            return False
        if kind != "vendor_default":
            raise RestoreError("HAND_ARM_GAINS_UNVERIFIED", "arm gains changed to an unverified setting")
        ctx.progress("takeover", 0.0)
        hold = dict(self._baseline)
        self._drive(ctx, "takeover", obs, [hold], hold, moving=self.installed)
        return True

    def _require_left_open(self, obs: Observation) -> None:
        self.require_capabilities(["left"])
        gap = float(np.max(np.abs(self._hand(obs, "left") - self.plans["left"].open_pose)))
        if gap > OPEN_HAND_TOLERANCE_RAD:
            raise RestoreError("HAND_PRESERVED_MOVED", "the left hand must be open before the right hand opens")

    def open_hand(self, ctx: RestoreContext, side: str) -> bool:
        """Open one hand along a collision-checked straight line, holding the other where it is."""
        self.require_capabilities([side])
        if self._baseline is None:
            raise RestoreError("HAND_NOT_SETTLED", "begin the hand phase before opening a hand")
        plan = self.plans[side]
        obs = self._observe(ctx)
        start = self._hand(obs, side)
        held = {s: self._hand(obs, s) for s in self.installed}
        # In particular, do not accept a new preserved pose after the other hand drifted.
        for other in self.installed:
            if other != side and np.max(np.abs(held[other] - self._baseline[other])) > PATH_SHAPE_HOLD_RAD:
                raise RestoreError("HAND_PRESERVED_MOVED", f"{other} hand moved between preparation stages")
        if side == "right" and "left" in self.installed:
            self._require_left_open(obs)
        frames = sweep_frames(start, plan.open_pose, self.step_rad)
        self._check_hand_sweep(ctx, plan, obs.feedback, frames, obs)
        ctx.progress(f"opening_{side}", 0.0)
        skipped = float(np.max(np.abs(start - plan.open_pose))) <= OPEN_HAND_TOLERANCE_RAD
        if skipped:
            self._confirm(ctx, (side,), lambda: None, SETTLE_S, obs)
            self._baseline[side] = start.copy()
            return True
        final = {**held, side: plan.open_pose}
        sequence = [{**held, side: frame} for frame in frames[1:]]
        self._drive(ctx, f"opening_{side}", obs, sequence, final, moving=(side,))
        self._baseline[side] = plan.open_pose.copy()
        return False

    def confirm(self, ctx: RestoreContext, sides: Sequence[str], arm_check: Callable[[], None]) -> None:
        """Observe the released hands and the arms together for :data:`OBSERVE_S`; any change is a failure."""
        ctx.progress("confirming", 0.0)
        obs = self._observe(ctx)
        self._confirm(ctx, sides, arm_check, OBSERVE_S, obs)

    def _confirm(self, ctx, sides, arm_check, duration, first):
        """Require new samples throughout the hold; cached feedback cannot prove stability."""
        obs = first
        body = obs.feedback.vector(BODY_JOINTS)
        baseline = self._baseline or {}
        until = self.ports.monotonic() + duration
        while True:
            obs = self._observe(ctx)
            for side in self.installed:
                pose = self._hand(obs, side)
                if side in sides:
                    if float(np.max(np.abs(pose - self.plans[side].open_pose))) > OPEN_HAND_TOLERANCE_RAD:
                        raise RestoreError("HAND_NOT_SETTLED", f"{side} hand is not open after the release")
                elif side in baseline and float(np.max(np.abs(pose - baseline[side]))) > PATH_SHAPE_HOLD_RAD:
                    raise RestoreError("HAND_PRESERVED_MOVED", f"{side} hand moved during the release")
            if gains_class(obs.command) != "ready":
                raise RestoreError("HAND_ARM_GAINS_UNVERIFIED", "arm gains 40/2 lost after release")
            if np.max(np.abs(obs.command.position - first.command.position)) > COMMAND_DRIFT_RAD:
                raise RestoreError("HAND_COMMAND_CHANGED", "another controller changed an arm target after release")
            if float(np.max(np.abs(obs.feedback.vector(BODY_JOINTS) - body))) > BODY_HOLD_RAD:
                raise RestoreError("HAND_BODY_MOVED", "an arm or the head moved during the confirmation")
            arm_check()
            if self.ports.monotonic() >= until and obs.feedback.received_at - first.feedback.received_at >= duration:
                return
            self.ports.wait(0.02)

    # -- the 50 Hz command loop --

    def _drive(
        self,
        ctx: RestoreContext,
        name: str,
        first: Observation,
        sequence: list[dict],
        final: Mapping[str, np.ndarray],
        *,
        moving: Sequence[str],
    ) -> None:
        """Publish ``sequence`` then hold ``final`` until the moving hands settle; always release afterwards."""
        ports = self.ports
        body = first.feedback.vector(BODY_JOINTS)
        board = first.command
        held = {side: self._hand(first, side) for side in self.installed}
        previous = {side: held[side].copy() for side in self.installed}
        began = ports.monotonic()
        settled_at = None
        last_tick = None
        try:
            for index, frame in enumerate([*sequence, *([dict(final)] * int(HOLD_FRAMES_S * HAND_HZ))]):
                tick = ports.monotonic()
                if tick - began > DRIVE_MAX_S:
                    raise RestoreError("HAND_NOT_SETTLED", f"{name} exceeded {DRIVE_MAX_S:g} seconds")
                if last_tick is not None and tick - last_tick > STALL_S:
                    raise RestoreError("HAND_STALLED", "hand control loop stalled")
                obs = self._observe(ctx)
                if float(np.max(np.abs(obs.feedback.vector(BODY_JOINTS) - body))) > BODY_HOLD_RAD:
                    raise RestoreError("HAND_BODY_MOVED", "an arm or the head moved during hand preparation")
                command = obs.command
                if name != "takeover" and gains_class(command) != "ready":
                    raise RestoreError("HAND_ARM_GAINS_UNVERIFIED", "arm gains 40/2 lost during hand preparation")
                if tuple(command.names) != tuple(board.names) or (
                    float(np.max(np.abs(np.asarray(command.position) - np.asarray(board.position)))) > COMMAND_DRIFT_RAD
                ):
                    raise RestoreError("HAND_COMMAND_CHANGED", "another controller changed an arm target")
                actual = {side: self._hand(obs, side) for side in self.installed}
                for side in self.installed:
                    if side in moving:
                        if float(np.max(np.abs(actual[side] - previous[side]))) > TRACKING_RAD:
                            raise RestoreError("HAND_NOT_TRACKING", f"{side} hand is not tracking the previous command")
                    elif float(np.max(np.abs(actual[side] - held[side]))) > PATH_SHAPE_HOLD_RAD:
                        raise RestoreError("HAND_PRESERVED_MOVED", f"preserved {side} hand moved during {name}")
                refusal = ports.refusal()
                if refusal:
                    raise RestoreError("HAND_OUTPUT_REFUSED", f"runtime refuses hand output: {refusal}")
                ports.publish(self._command(frame, "left"), self._command(frame, "right"))
                previous = {side: np.asarray(frame[side], float) for side in self.installed}
                last_tick = tick
                error = max(float(np.max(np.abs(actual[side] - final[side]))) for side in moving)
                reached = index >= len(sequence)
                if reached and error <= OPEN_HAND_TOLERANCE_RAD and gains_class(command) == "ready":
                    settled_at = obs.feedback.received_at if settled_at is None else settled_at
                    if obs.feedback.received_at - settled_at >= SETTLE_S:
                        return
                else:
                    settled_at = None
                if index % int(HAND_HZ / 2) == 0:  # twice a second is plenty for feedback
                    ctx.progress(name, min(1.0, (index + 1) / max(1, len(sequence))))
                ports.wait(max(0.0, tick + 1.0 / HAND_HZ - ports.monotonic()))
            raise RestoreError(
                "HAND_NOT_SETTLED",
                "takeover gains and hand hold did not settle" if name == "takeover" else f"{name} did not settle",
            )
        finally:
            ports.release()

    def _command(self, frame: Mapping[str, np.ndarray], side: str) -> np.ndarray:
        if side not in self.installed:
            return np.array(ABSENT_HAND_COMMAND)
        plan = self.plans.get(side)
        values = np.asarray(frame[side], float)
        if plan is not None:
            try:
                return plan.mapping.check(values)
            except ValueError as exc:
                raise RestoreError("HAND_SWEEP_OUT_OF_LIMITS", f"{side} hand command: {exc}") from None
        if values.shape != (10,) or not np.isfinite(values).all():
            raise RestoreError("HAND_SWEEP_OUT_OF_LIMITS", f"{side} hand command is not ten finite values")
        return values
