"""Restore state machine with a deterministic port; no hardware or ROS discovery.

Both Mock hands use the left mapping intentionally. These tests verify control
behavior, not a physical right-hand model or arm/finger collision coverage.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.feedback import Feedback
from aimdk_grasp.hand import HandMapping
from aimdk_grasp.hand_restore import (
    FEEDBACK_MAX_AGE_S,
    VENDOR_STIFFNESS,
    CommandBoard,
    GatewayState,
    HandPlan,
    HandRestorer,
    HandSweepCheck,
    Observation,
    RestoreContext,
    RestoreError,
    RuntimeState,
    build_hand_plans,
    sweep_frames,
)
from aimdk_grasp.joints import BODY_JOINTS, LEFT_ARM, RIGHT_ARM
from aimdk_grasp.robot_model import planning_model
from test_hand import PRESETS


class Ports:
    def __init__(self, mapping):
        self.now = 10.0
        self.hands = {side: mapping.preset("close") for side in ("left", "right")}
        self.body = dict.fromkeys(BODY_JOINTS, 0.0)
        self.board = CommandBoard(LEFT_ARM + RIGHT_ARM, np.zeros(14), np.full(14, 40.0), np.full(14, 2.0), 0.0)
        self.runtime = RuntimeState(True, "", "policy_stream", 0.0)
        self.gateway = GatewayState(True, True, True, "task", 0.0)
        self.sent, self.releases = [], 0
        self.phase, self.cancelled, self.frozen_at = "", False, None
        self.tick = lambda: None
        self.follow, self.output_refusal = True, None

    def observe(self):
        stamp = self.now if self.frozen_at is None else self.frozen_at
        left, right = (self.hands.get(side) for side in ("left", "right"))
        feedback = Feedback(
            stamp, dict(self.body), None if left is None else left.copy(), True, None if right is None else right.copy()
        )
        return Observation(feedback, self.now - stamp, self.board, self.runtime, self.gateway)

    def publish(self, left, right):
        self.sent.append((self.phase, left.copy(), right.copy()))
        if self.follow:
            self.hands = {
                side: values.copy() for side, values in (("left", left), ("right", right)) if side in self.hands
            }
        self.board = replace(self.board, stiffness=np.full(14, 40.0), damping=np.full(14, 2.0))

    def release(self):
        self.releases += 1

    def refusal(self):
        return self.output_refusal

    def monotonic(self):
        return self.now

    def wait(self, seconds):
        self.now += seconds
        self.tick()

    def progress(self, phase, _fraction):
        self.phase = phase


@pytest.fixture
def restore():
    mapping = HandMapping(planning_model(), PRESETS)
    ports = Ports(mapping)
    checks = []
    sweep = SimpleNamespace(check=lambda *_args, **_kw: checks.append(ports.phase))
    plans = {side: HandPlan(side, mapping, mapping.preset("open"), sweep) for side in ("left", "right")}
    restorer = HandRestorer(ports, plans, arm_motion_covered=True)
    ctx = RestoreContext("task", ports.now + 30, lambda: ports.cancelled, ports.progress)
    return restorer, ports, ctx, checks


def prepare(restorer, ports, ctx, sides=("left", "right")):
    restorer.preflight(ctx, sides, arm_target=ports.body, arm_must_move=False)
    restorer.begin(ctx)


@pytest.mark.parametrize("side", ["left", "right"])
def test_one_installed_hand_opens_without_reading_or_preserving_an_absent_hand(restore, side):
    restorer, ports, ctx, _ = restore
    other = "right" if side == "left" else "left"
    restorer.installed_hands = (side,)
    restorer.plans = {side: restorer.plans[side]}
    del ports.hands[other]
    prepare(restorer, ports, ctx, (side,))
    assert not restorer.open_hand(ctx, side)
    restorer.confirm(ctx, (side,), lambda: None)
    np.testing.assert_allclose(ports.hands[side], restorer.plans[side].open_pose)
    assert ports.sent
    for _, left, right in ports.sent:
        np.testing.assert_array_equal(left if other == "left" else right, np.zeros(10))
    with pytest.raises(RestoreError, match=f"cannot open the {other}"):
        restorer.preflight(ctx, (other,), arm_target=ports.body, arm_must_move=False)


@pytest.mark.parametrize("installed", [("left",), ("right",), ()])
def test_restore_models_follow_the_declared_mounts(monkeypatch, installed):
    from aimdk_grasp import backend
    from aimdk_grasp.robot_model import config_directory

    def robot(config):
        side = config["backend_options"]["arm_side"]
        assert side in installed
        return SimpleNamespace(hand=HandMapping(planning_model(config_directory(side)), PRESETS, side=side))

    monkeypatch.setattr(backend, "x2_robot", robot)
    plans, missing = build_hand_plans({"backend_options": {"installed_hands": list(installed)}})
    assert set(plans) == set(installed)
    restorer = HandRestorer(None, plans, missing, installed_hands=installed)
    assert (restorer.geometry is not None) is bool(installed)
    assert set(missing) == {"left", "right"} - set(installed)


@pytest.mark.parametrize(
    "fault,code",
    [(None, None), ("owner", "HAND_GATEWAY_NOT_OWNER"), ("stale", "HAND_FEEDBACK_STALE"), ("head", "HAND_BODY_MOVED")],
)
def test_long_arm_check_refreshes_owner_and_detects_live_changes_before_dispatch(restore, fault, code):
    from aimdk_grasp.joints import HEAD

    restorer, ports, ctx, _ = restore
    original_observe = ports.observe
    last_poll = [ports.now]

    def observe():
        observation = original_observe()
        # A request starts on observe; its result is consumed by a later poll.
        # A single poll after a long computation must not re-age the old result.
        age = ports.now - last_poll[0]
        last_poll[0] = ports.now
        return replace(observation, gateway=replace(observation.gateway, age_s=age))

    ports.observe = observe

    def long_check(_feedback, _joints, _positions, *, guard, **_kwargs):
        for i in range(12):
            ports.now += 0.2
            if i == 4:
                if fault == "owner":
                    ports.gateway = replace(ports.gateway, active_task_id="other")
                elif fault == "stale":
                    ports.frozen_at = ports.now - (FEEDBACK_MAX_AGE_S + 0.1)
                elif fault == "head":
                    ports.body[HEAD[0]] += 0.02
            guard()

    geometry = SimpleNamespace(check_arm_path=long_check)
    for plan in restorer.plans.values():
        plan.sweep.geometry = geometry
    joints = LEFT_ARM + RIGHT_ARM
    positions = np.array([[ports.body[name] for name in joints]] * 2)
    if code:
        with pytest.raises(RestoreError) as error:
            restorer.check_arm_path(ctx, joints, positions)
        assert error.value.code == code
        assert ports.now < 12.0
    else:
        restorer.check_arm_path(ctx, joints, positions, start_tolerance_rad=0.005)
        assert ports.now > 14.0  # Full path and measured-start transition both checked.
    assert not ports.sent


def test_takeover_then_left_right_and_observe_release(restore):
    restorer, ports, ctx, checks = restore
    ports.board = replace(ports.board, stiffness=VENDOR_STIFFNESS, damping=np.ones(14))
    left_start, right_start = ports.hands["left"].copy(), ports.hands["right"].copy()
    prepare(restorer, ports, ctx)
    assert ports.sent[0][0] == "takeover"
    np.testing.assert_allclose(ports.sent[0][1], left_start)
    np.testing.assert_allclose(ports.sent[0][2], right_start)
    assert not restorer.open_hand(ctx, "left")
    left_commands = [row for row in ports.sent if row[0] == "opening_left"]
    for _, _, right in left_commands:
        np.testing.assert_allclose(right, right_start)
    assert not restorer.open_hand(ctx, "right")
    for _, left, _ in [row for row in ports.sent if row[0] == "opening_right"]:
        np.testing.assert_allclose(left, restorer.plans["left"].open_pose)
    released_at, count = ports.now, len(ports.sent)
    restorer.confirm(ctx, ("left", "right"), lambda: None)
    assert ports.now - released_at >= 2 and len(ports.sent) == count
    assert ports.releases == 3 and len(checks) >= 4
    left = np.stack([left_start, *(row[1] for row in left_commands)])
    assert np.max(np.abs(np.diff(left, axis=0))) <= 0.01 + 1e-12


def test_already_open_uses_new_feedback_without_publishing(restore):
    restorer, ports, ctx, _ = restore
    ports.hands = {side: plan.open_pose.copy() for side, plan in restorer.plans.items()}
    prepare(restorer, ports, ctx)
    for side in ("left", "right"):
        assert restorer.open_hand(ctx, side)
    restorer.confirm(ctx, ("left", "right"), lambda: None)
    assert not ports.sent and ports.now >= 12.4


def test_long_hand_preflight_refreshes_owner_and_checks_every_sample(restore):
    restorer, ports, ctx, _ = restore
    restorer.step_rad = 0.001
    original_observe, last_poll = ports.observe, [ports.now]
    checked = []

    def observe():
        result = original_observe()
        age = ports.now - last_poll[0]
        last_poll[0] = ports.now
        return replace(result, gateway=replace(result.gateway, age_s=age))

    def slow_check(_feedback, frames, **_kwargs):
        ports.now += len(frames) * 0.004
        checked.extend(frames.copy())

    ports.observe = observe
    for plan in restorer.plans.values():
        plan.sweep.check = slow_check
    expected = np.concatenate(
        [
            sweep_frames(ports.hands[side], restorer.plans[side].open_pose, restorer.step_rad)
            for side in ("left", "right")
        ]
    )
    restorer.preflight(ctx, ("left", "right"), arm_target=ports.body, arm_must_move=False)
    np.testing.assert_allclose(np.asarray(checked), expected)
    assert ports.now > 11.0 and not ports.sent


@pytest.mark.parametrize(
    "fault,code",
    [
        ("owner", "HAND_GATEWAY_NOT_OWNER"),
        ("stale", "HAND_FEEDBACK_STALE"),
        ("stop", "HAND_RUNTIME_NOT_READY"),
        ("mode", "HAND_RUNTIME_NOT_READY"),
        ("gains", "HAND_ARM_GAINS_UNVERIFIED"),
        ("board", "HAND_FEEDBACK_STALE"),
        ("nan_age", "HAND_GATEWAY_NOT_OWNER"),
        ("capability", "HAND_CAPABILITY_MISSING"),
        ("right_only", "HAND_PRESERVED_MOVED"),
        ("collision", "HAND_SWEEP_COLLISION"),
    ],
)
def test_preflight_refuses_before_output(restore, fault, code):
    restorer, ports, ctx, _ = restore
    sides = ("left", "right")
    if fault == "owner":
        ports.gateway = replace(ports.gateway, active_task_id="another-task")
    elif fault == "nan_age":
        ports.gateway = replace(ports.gateway, age_s=float("nan"))
    elif fault == "stale":
        ports.frozen_at = ports.now - (FEEDBACK_MAX_AGE_S + 0.1)
    elif fault == "stop":
        ports.runtime = replace(ports.runtime, ok=False)
    elif fault == "mode":
        ports.runtime = replace(ports.runtime, active_mode="idle")
    elif fault == "gains":
        ports.board = replace(ports.board, stiffness=np.zeros(14))
    elif fault == "board":
        ports.board = replace(ports.board, position=np.full(14, float("nan")))
    elif fault == "capability":
        restorer.plans.pop("right")
    elif fault == "right_only":
        sides = ("right",)
    elif fault == "collision":

        def collision(*_args, **_kw):
            raise RestoreError("HAND_SWEEP_COLLISION", "mock collision")

        restorer.plans["left"].sweep = SimpleNamespace(check=collision)
    with pytest.raises(RestoreError) as error:
        prepare(restorer, ports, ctx, sides)
    assert error.value.code == code and not ports.sent


@pytest.mark.parametrize(
    "fault,code",
    [
        ("cancel", "POSE_CANCELLED"),
        ("timeout", "TASK_POSE_TIMEOUT"),
        ("stale", "HAND_FEEDBACK_STALE"),
        ("owner", "HAND_GATEWAY_NOT_OWNER"),
        ("mode", "HAND_RUNTIME_NOT_READY"),
        ("body", "HAND_BODY_MOVED"),
        ("board", "HAND_COMMAND_CHANGED"),
        ("gains", "HAND_ARM_GAINS_UNVERIFIED"),
        ("preserved", "HAND_PRESERVED_MOVED"),
        ("tracking", "HAND_NOT_TRACKING"),
    ],
)
def test_mid_open_failure_always_releases_and_never_opens_right(restore, fault, code):
    restorer, ports, ctx, _ = restore
    prepare(restorer, ports, ctx)

    def fail():
        if fault == "cancel":
            ports.cancelled = True
        elif fault == "timeout":
            ctx.deadline = ports.now
        elif fault == "stale":
            ports.frozen_at = ports.now - (FEEDBACK_MAX_AGE_S + 0.1)
        elif fault == "owner":
            ports.gateway = replace(ports.gateway, active_task_id="another")
        elif fault == "mode":
            ports.runtime = replace(ports.runtime, active_mode="idle")
        elif fault == "body":
            ports.body[LEFT_ARM[0]] += 0.1
        elif fault == "board":
            ports.board = replace(ports.board, position=np.ones(14))
        elif fault == "gains":
            ports.board = replace(ports.board, stiffness=np.zeros(14))
        elif fault == "preserved":
            ports.hands["right"][4] -= 0.11
        else:
            ports.hands["left"][4] -= 0.3

    ports.tick = fail
    with pytest.raises(RestoreError) as error:
        restorer.open_hand(ctx, "left")
    assert error.value.code == code
    assert ports.releases == 1 and {row[0] for row in ports.sent} == {"opening_left"}


@pytest.mark.parametrize(
    "fault,code",
    [
        ("rebound", "HAND_NOT_SETTLED"),
        ("gains", "HAND_ARM_GAINS_UNVERIFIED"),
        ("board", "HAND_COMMAND_CHANGED"),
        ("cached", "HAND_FEEDBACK_STALE"),
    ],
)
def test_release_observation_rejects_rebound_or_loss_of_control(restore, fault, code):
    restorer, ports, ctx, _ = restore
    prepare(restorer, ports, ctx)
    for side in ("left", "right"):
        restorer.open_hand(ctx, side)
    count = len(ports.sent)

    def fail():
        if fault == "rebound":
            ports.hands["left"][4] = 0.1
        elif fault == "gains":
            ports.board = replace(ports.board, damping=np.zeros(14))
        elif fault == "board":
            ports.board = replace(ports.board, position=np.ones(14))
        elif ports.frozen_at is None:
            ports.frozen_at = ports.now

    ports.tick = fail
    with pytest.raises(RestoreError) as error:
        restorer.confirm(ctx, ("left", "right"), lambda: None)
    assert error.value.code == code and len(ports.sent) == count


def test_arm_motion_requires_proven_coverage_even_with_open_hands(restore):
    restorer, ports, ctx, _ = restore
    ports.hands = {side: plan.open_pose.copy() for side, plan in restorer.plans.items()}
    restorer.arm_motion_covered = False
    with pytest.raises(RestoreError, match="dummy hands") as error:
        restorer.preflight(ctx, ("left", "right"), arm_target=ports.body, arm_must_move=True)
    assert error.value.code == "HAND_SHAPE_NOT_COVERED" and not ports.sent


@pytest.mark.parametrize("grasp_side", ["left", "right", "auto"])
def test_preparation_keeps_both_native_targets_for_every_grasp_side(monkeypatch, grasp_side):
    from aimdk_grasp import backend
    from aimdk_grasp.robot_model import config_directory

    def robot(config):
        side = config["backend_options"]["arm_side"]
        side = "left" if side == "auto" else side
        return SimpleNamespace(hand=HandMapping(planning_model(config_directory(side)), PRESETS, side=side))

    mapping = HandMapping(planning_model(), PRESETS)
    monkeypatch.setattr(backend, "x2_robot", robot)
    plans, missing = build_hand_plans({"backend_options": {"arm_side": grasp_side}})
    assert set(plans) == {"left", "right"} and not missing
    np.testing.assert_allclose(plans["left"].open_pose, [-0.129, 1.54, -0.255, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(plans["right"].open_pose, [0.129, -1.54, 0.255, 0, 0, 0, 0, 0, 0, 0])
    assert plans["left"].sweep.geometry is plans["right"].sweep.geometry
    # The real left sweep uses all mounted hand links and rejects model limit violations.
    sweep = HandSweepCheck(mapping)
    body = dict.fromkeys(BODY_JOINTS, 0.0)
    body[LEFT_ARM[-1]] = 2.0
    with pytest.raises(RestoreError) as error:
        sweep.check(Feedback(1.0, body, mapping.preset("open")), [mapping.preset("open")])
    assert error.value.code == "HAND_SWEEP_OUT_OF_LIMITS"


def test_sweep_bounds_each_step_and_rejects_nonfinite_input():
    frames = sweep_frames(np.zeros(10), np.ones(10))
    assert np.max(np.abs(np.diff(frames, axis=0))) <= 0.01 + 1e-12
    with pytest.raises(ValueError):
        sweep_frames(np.zeros(10), np.full(10, float("nan")))


@pytest.mark.parametrize("fault", ["finger", "head", "cancel"])
def test_arm_path_shape_guard_rejects_changes_and_cancellation(restore, fault):
    from aimdk_grasp.joints import HEAD

    restorer, ports, ctx, _ = restore
    initial = restorer._observe(ctx)
    if fault == "finger":
        ports.hands["left"][4] -= 0.11
    elif fault == "head":
        ports.body[HEAD[0]] += 0.02
    else:
        ports.cancelled = True
    with pytest.raises(RestoreError) as exc:
        restorer.guard_arm_motion(ctx, initial)
    assert (
        exc.value.code
        == {"finger": "HAND_PRESERVED_MOVED", "head": "HAND_BODY_MOVED", "cancel": "POSE_CANCELLED"}[fault]
    )


@pytest.mark.parametrize("side", ["left", "right"])
def test_arm_guard_allows_finger_drift_within_covered_bound_and_reports_excess(restore, side):
    restorer, ports, ctx, _ = restore
    initial = restorer._observe(ctx)
    ports.hands[side][4] -= 0.099
    restorer.guard_arm_motion(ctx, initial)
    ports.hands[side][4] -= 0.002
    with pytest.raises(RestoreError, match="phase=trajectory_sent.*joint_index=4.*delta=0.101000.*limit=0.100000"):
        restorer.guard_arm_motion(ctx, initial, "trajectory_sent")


@pytest.mark.parametrize("field", ["feedback", "command"])
@pytest.mark.parametrize("age_s", [0.300888, 0.999999, 1.0, 1.000001, -0.001, float("nan"), float("inf")])
def test_temporary_one_second_feedback_age_guard(restore, field, age_s):
    restorer, ports, ctx, _ = restore
    original_observe = ports.observe

    def observe():
        observation = original_observe()
        if field == "feedback":
            return replace(observation, feedback_age_s=age_s)
        return replace(observation, command=replace(observation.command, age_s=age_s))

    ports.observe = observe
    if 0 <= age_s <= 1.0:
        observation = restorer._observe(ctx)
        assert (observation.feedback_age_s if field == "feedback" else observation.command.age_s) == age_s
    else:
        with pytest.raises(RestoreError) as error:
            restorer._observe(ctx)
        assert error.value.code == "HAND_FEEDBACK_STALE"
    assert not ports.sent


def test_temporary_age_override_still_rejects_unhealthy_feedback(restore):
    restorer, ports, ctx, _ = restore
    original_observe = ports.observe

    def observe():
        observation = original_observe()
        return replace(observation, feedback=replace(observation.feedback, healthy=False), feedback_age_s=0.300888)

    ports.observe = observe
    with pytest.raises(RestoreError) as error:
        restorer._observe(ctx)
    assert error.value.code == "HAND_FEEDBACK_STALE"
    assert not ports.sent
