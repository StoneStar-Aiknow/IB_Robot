"""Behavior of the opt-in full-body hold, without creating a ROS graph."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid
from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")

from aimdk_robot import named_motion as motion  # noqa: E402
from aimdk_robot import projection  # noqa: E402


class Clock:
    def __init__(self):
        self.now = 10.0
        self.on_sleep = lambda: None

    def sleep(self, seconds):
        self.now += max(seconds, 1e-5)
        self.on_sleep()


class HoldNode:
    def __init__(self, clock):
        self.clock = clock
        self.actual = [0.0] * 34
        self.commands = []
        self.claims = []
        self.releases = []
        self.faults = []
        self.guard_ok = True
        self.release_ok = True
        self.action_callback_group = None
        self.runtime_state = SimpleNamespace(stop_epoch=1)
        self.vendor_config = {"mc_actions": {"policy_stream": "UPPERBODY"}}
        self.acks = []

    def create_publisher(self, *args, **kwargs):
        return SimpleNamespace(publish=lambda message: self.acks.append(json.loads(message.data)))

    def platform_action(self):
        return "UPPERBODY"

    def ensure_input_source(self):
        return True

    def create_subscription(self, *args, **kwargs):
        return object()

    def current_mode(self):
        return "policy_stream"

    def output_admitted(self):
        return True

    def measured_named_hold_sample(self, **kwargs):
        return list(self.actual), self.clock.now

    def validate_named_hold_path(self, name, times, positions):
        return self.guard_ok, "test geometry result"

    def claim_arm(self, generation):
        self.claims.append(generation)
        return True

    def release_arm_confirmed(self, generation, **kwargs):
        self.releases.append(generation)
        return self.release_ok

    def fail_named_hold(self, reason):
        self.faults.append(reason)

    def command_recorded(self, arms, hands, **kwargs):
        assert len(arms) == 14 and len(hands) == 20
        self.actual = list(arms) + list(hands)
        self.commands.append(list(self.actual))
        return True


class Goal:
    def __init__(self, node, cancel_after=None):
        self.node = node
        self.cancel_after = cancel_after
        self.goal_id = SimpleNamespace(uuid=list(uuid.uuid4().bytes))
        self.feedback = []

    @property
    def is_cancel_requested(self):
        if isinstance(self.cancel_after, str):
            return self.cancel_after in self.feedback
        return self.cancel_after is not None and len(self.node.commands) >= self.cancel_after

    def publish_feedback(self, value):
        self.feedback.append(value.state)


def server(monkeypatch, *, keep_lease=True, cancel_after=None):
    clock = Clock()
    monkeypatch.setattr(motion.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(motion.time, "sleep", clock.sleep)
    node = HoldNode(clock)
    value = object.__new__(motion.NamedMotionServer)
    value._node = node
    value._lock = threading.Lock()
    value._running = None
    value._returning = False
    value._return_motions = set()
    value._display_completion = {}
    value._trajectories = {
        "overhead_heart": SimpleNamespace(target="both", positions=((0.0,) * 14,) * 2, hands=((0.0,) * 20,) * 2)
    }
    value._arm_limits = {name: (-2.0, 2.0) for name in projection.VENDOR_ARM_ORDER}
    value._blend_speed = 0.5
    value._max_speed = 1.0
    value._blend_min_s = 0.02
    value._rate_hz = 50.0
    value._return_settle_s = 0.2
    value._return_tolerance = 0.03
    value._init_hold(
        {
            "hold_motions": ["overhead_heart"],
            "hold_hand_limits": [[-2.0, 2.0]] * 20,
        }
    )
    goal = Goal(node, cancel_after)
    key = (value._hold_goal_id(goal), "overhead_heart")
    value._hold_leases[key] = clock.now
    if keep_lease:
        clock.on_sleep = lambda: value._hold_leases.__setitem__(key, clock.now)
    return value, node, goal, key


def run(value, goal):
    return value._run_hold_pose("overhead_heart", "policy_stream", "both", "", goal, 1, 1)


def test_absent_hold_configuration_does_not_create_subscription():
    value = object.__new__(motion.NamedMotionServer)
    value._init_hold({})
    assert not value._hold_names
    assert value._hold_subscription is None


def test_geometry_rejection_never_claims_or_sends(monkeypatch):
    value, node, goal, _key = server(monkeypatch)
    node.guard_ok = False
    ok, _code, _reason = run(value, goal)
    assert not ok
    assert node.claims == node.commands == node.releases == []


def test_missing_uuid_lease_never_claims(monkeypatch):
    value, node, goal, key = server(monkeypatch)
    value._hold_leases.pop(key)
    assert not run(value, goal)[0]
    assert not node.claims


def test_cancel_keeps_all_twenty_hand_values_and_confirms_release(monkeypatch):
    value, node, goal, _key = server(monkeypatch, cancel_after="holding")
    node.actual[14:] = [0.1] * 20
    value._trajectories["overhead_heart"].hands = ((0.1,) * 20,) * 2
    with pytest.raises(motion._Cancelled):
        run(value, goal)
    assert len(node.commands[-1]) == 34
    assert node.commands[-1][14:] == pytest.approx([0.1] * 20)
    assert node.releases == [1]
    assert "holding" in goal.feedback


def test_lease_expiry_has_bounded_termination(monkeypatch):
    value, node, goal, _key = server(monkeypatch, keep_lease=False)
    with pytest.raises(motion._Superseded, match="lease expired"):
        run(value, goal)
    assert node.releases == [1]
    assert node.faults
    assert len(node.commands) < 100


def test_absolute_cap_applies_even_when_lease_is_renewed(monkeypatch):
    value, node, goal, _key = server(monkeypatch)
    value._hold_max_duration_s = 0.1
    with pytest.raises(motion._Superseded, match="absolute duration"):
        run(value, goal)
    assert node.releases == [1]


def test_release_failure_turns_cancel_into_failure(monkeypatch):
    value, node, goal, _key = server(monkeypatch, cancel_after="holding")
    node.release_ok = False
    with pytest.raises(motion._Superseded, match="release was not confirmed"):
        run(value, goal)
    assert node.faults


def test_valid_exit_is_executed_under_same_owner(monkeypatch):
    value, node, goal, key = server(monkeypatch)
    value._on_hold_lease(
        SimpleNamespace(
            data=json.dumps(
                {
                    "goal_id": key[0],
                    "motion": key[1],
                    "exit_target": [0.1] * 34,
                }
            )
        )
    )
    ok, code, message = run(value, goal)
    assert ok and code == motion.ExecuteNamedMotion.Result.NONE
    identity = hashlib.sha256(json.dumps([0.1] * 34, separators=(",", ":")).encode()).hexdigest()
    assert json.loads(message) == {"motion": "overhead_heart", "exit_target_sha256": identity}
    assert node.claims == [1] and node.releases == [1]
    assert node.commands[-1] == pytest.approx([0.1] * 34)
    assert "exiting" in goal.feedback


def test_uuid_lease_acknowledgement_precedes_any_goal_claim(monkeypatch):
    value, node, _goal, key = server(monkeypatch)
    value._on_hold_lease(SimpleNamespace(data=json.dumps({"goal_id": key[0], "motion": key[1]})))
    assert node.acks == [{"goal_id": key[0], "motion": key[1]}]
    assert node.claims == node.commands == []


def test_invalid_lease_never_gets_an_acknowledgement(monkeypatch):
    value, node, _goal, key = server(monkeypatch)
    for payload in (
        {"goal_id": key[0], "motion": "unknown"},
        {"goal_id": "not-a-uuid", "motion": key[1]},
        {"goal_id": key[0], "motion": key[1], "exit_target": [True] * 34},
    ):
        value._on_hold_lease(SimpleNamespace(data=json.dumps(payload)))
    assert node.acks == []


def test_arrival_wait_keeps_exiting_feedback_alive_while_tracking_converges(monkeypatch):
    value, node, goal, key = server(monkeypatch)
    value._return_settle_s = 2.0
    value._running = ("overhead_heart", 1)
    started = node.clock.now
    feedback = []
    goal.publish_feedback = lambda message: feedback.append((node.clock.now, message.state))
    node.measured_named_hold_sample = lambda **kwargs: (
        [0.1 if node.clock.now - started < 1.2 else 0.0] * 34,
        node.clock.now,
    )
    assert value._hold_wait_arrival(
        "overhead_heart", "policy_stream", [0.0] * 34, goal, key, 1, 1, started, state="exiting"
    )
    assert feedback[-1][0] - started >= 1.1
    assert all(state == "exiting" for _, state in feedback)
    assert max(b[0] - a[0] for a, b in zip(feedback, feedback[1:], strict=False)) <= 0.15
    assert node.faults == []


def test_unsafe_exit_keeps_the_admitted_pose_until_cancel(monkeypatch):
    value, node, goal, key = server(monkeypatch, cancel_after="exit_blocked")
    node.validate_named_hold_path = lambda name, times, positions: (
        positions[-1][0] < 0.05,
        "exit violates test head clearance",
    )
    value._on_hold_lease(
        SimpleNamespace(
            data=json.dumps(
                {
                    "goal_id": key[0],
                    "motion": key[1],
                    "exit_target": [0.1] * 34,
                }
            )
        )
    )
    with pytest.raises(motion._Cancelled):
        run(value, goal)
    assert node.commands and all(point == [0.0] * 34 for point in node.commands)
    assert node.releases == [1]


def test_unrelated_or_malformed_lease_cannot_change_exit(monkeypatch):
    value, _node, _goal, key = server(monkeypatch)
    for payload in (
        {"goal_id": key[0], "motion": "wave", "exit_target": [0.1] * 34},
        {"goal_id": key[0], "motion": key[1], "exit_target": [True] * 34},
        {"goal_id": "0" * 32, "motion": key[1]},
    ):
        value._on_hold_lease(SimpleNamespace(data=json.dumps(payload)))
    assert value._hold_exit_targets == {}


def test_cosine_bridge_bounds_peak_speed(monkeypatch):
    value, _node, _goal, _key = server(monkeypatch)
    duration = value._hold_duration([0.0] * 34, [1.0] * 34)
    assert duration >= math.pi - 1e-8


def test_cancel_selects_the_full_tracking_lag_target(monkeypatch):
    value, node, goal, _key = server(monkeypatch)
    value._hold_sent = motion.deque([(9.9, [0.1] * 34), (10.0, [0.2] * 34)])
    with pytest.raises(motion._Cancelled):
        value._hold_cancel("overhead_heart", [0.2] * 34, goal, 1, 1)
    assert node.commands[-1] == pytest.approx([0.1] * 34)


def test_unconfirmed_arrival_latches_fault_before_release(monkeypatch):
    value, node, goal, _key = server(monkeypatch)
    value._hold_wait_arrival = lambda *args: False
    assert not run(value, goal)[0]
    assert node.faults and node.releases == [1]
