from pathlib import Path
from types import SimpleNamespace

import numpy as np

from manipulation_execution.imitation_retargeting.x2.kinematics import MjcfKinematics
from manipulation_execution.imitation_retargeting.x2.safety import RateLimiter, effective_limits_rad

MODEL_XML = (Path(__file__).resolve().parents[1] / "manipulation_execution"
             / "imitation_retargeting" / "x2" / "model" / "x2_ultra.xml")
ARM_JOINTS = [f"{side}_{j}_joint" for side in ("left", "right") for j in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_yaw", "wrist_pitch", "wrist_roll")]
DT = 0.02


def _limits(margin_deg):
    kin = MjcfKinematics(MODEL_XML)
    ranges = np.array([kin.joint_range(n) for n in ARM_JOINTS])
    return effective_limits_rad(ranges[:, 0], ranges[:, 1], margin_deg)


def _runtime_advance(self, goal):
    # Verbatim body of followme_player.py _advance (video2x2_runtime 0717710);
    # self._step_limit() inlined as its non-approaching branch.
    if self._prev is None:
        self._prev = goal.copy()
        return self._prev
    step = self.rated_rad_s * self.vel_scale * self.dt
    delta = goal - self._prev
    excess = np.abs(delta) - step
    hit = excess > 0
    if hit.any() and not (self._approaching or self._returning):
        # Only the ceiling counts as a trip worth reporting. The approach
        # ramp is *meant* to saturate; counting it would bury the signal.
        self._cap_trips += hit
        self._cap_worst_rad = np.maximum(self._cap_worst_rad,
                                         np.where(hit, excess, 0.0))
    pose = self._prev + np.clip(delta, -step, step)

    clamped = np.clip(pose, self.low_rad, self.high_rad)
    over = np.abs(pose - clamped)
    touched = over > 0
    if touched.any():
        self._limit_trips += touched
        self._limit_worst_rad = np.maximum(self._limit_worst_rad, over)
    self._prev = clamped
    return clamped


def _goals(low, high, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    span = high - low
    goals = np.empty((n, 14))
    goals[0] = rng.uniform(low, high)
    for i in range(1, n):
        if rng.random() < 0.1:   # big jump, possibly well outside the limits
            goals[i] = rng.uniform(low - 0.3 * span, high + 0.3 * span)
        else:
            goals[i] = goals[i - 1] + rng.normal(0.0, 0.05, 14)
    return goals


def test_intersection_table_deg():
    low, high = (np.degrees(x) for x in _limits(margin_deg=0.0))
    left = [(-176.5, 116.9), (-3.0, 171.5), (-145, 145), (-135, 0), (-146, 146), (-30, 30), (-40, 40)]
    right = list(left)
    right[1] = (-171.5, 3.0)   # wrist_roll is symmetric, so its sign flip is a no-op
    np.testing.assert_allclose(np.stack([low, high], axis=1), np.array(left + right, float), rtol=0, atol=0.05)


def test_step_matches_runtime_advance_bitwise():
    low, high = _limits(margin_deg=1.0)
    rated = np.random.default_rng(1).uniform(2.0, 10.0, 14)
    ours = RateLimiter(low, high, rated, vel_scale=1.0, dt=DT)
    ref = SimpleNamespace(_prev=None, low_rad=low, high_rad=high, rated_rad_s=rated, vel_scale=1.0,
                          dt=DT, _approaching=False, _returning=False,
                          _cap_trips=np.zeros(14, dtype=np.int64), _cap_worst_rad=np.zeros(14),
                          _limit_trips=np.zeros(14, dtype=np.int64), _limit_worst_rad=np.zeros(14))
    goals = _goals(low, high)
    goals[0] = high + 0.2   # first tick passes through unclamped
    for goal in goals:
        assert np.array_equal(ours.step(goal), _runtime_advance(ref, goal))
    assert ours.cap_trips.sum() > 0 and ours.limit_trips.sum() > 0
    assert np.array_equal(ours.cap_trips, ref._cap_trips)
    assert np.array_equal(ours.cap_worst_rad, ref._cap_worst_rad)
    assert np.array_equal(ours.limit_trips, ref._limit_trips)
    assert np.array_equal(ours.limit_worst_rad, ref._limit_worst_rad)


def test_rate_and_limit_bounds():
    low, high = _limits(margin_deg=1.0)
    rated = np.random.default_rng(2).uniform(2.0, 10.0, 14)
    vel_scale = 0.7
    limiter = RateLimiter(low, high, rated, vel_scale=vel_scale, dt=DT)
    goals = _goals(low, high, seed=3)
    prev = limiter.step(goals[0]).copy()
    for goal in goals[1:]:
        out = limiter.step(goal).copy()
        assert np.all(np.abs(out - prev) <= rated * vel_scale * DT + 1e-12)
        assert np.all((out >= low) & (out <= high))
        prev = out
