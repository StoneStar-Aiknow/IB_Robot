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
        self.command_owners = []
        self.guard_paths = []
        self.events = []
        self.claims = []
        self.releases = []
        self.faults = []
        self.guard_ok = True
        self.claim_ok = True
        self.release_ok = True
        self.follows_commands = True
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
        self.guard_paths.append((name, list(times), [list(point) for point in positions]))
        return self.guard_ok, "test geometry result"

    def claim_arm(self, generation):
        self.claims.append(generation)
        self.events.append("claim")
        return self.claim_ok

    def release_arm_confirmed(self, generation, **kwargs):
        self.releases.append(generation)
        self.events.append("release")
        return self.release_ok

    def fail_named_hold(self, reason):
        self.faults.append(reason)
        self.events.append("fault")

    def command_recorded(self, arms, hands, **kwargs):
        assert len(arms) == 14 and len(hands) == 20
        point = list(arms) + list(hands)
        if self.follows_commands:
            self.actual = point
        self.commands.append(point)
        self.command_owners.append((kwargs["owner"], kwargs["epoch"]))
        self.events.append("command")
        return True


class Goal:
    def __init__(self, node, cancel_after=None):
        self.node = node
        self.cancel_after = cancel_after
        self.goal_id = SimpleNamespace(uuid=list(uuid.uuid4().bytes))
        self.feedback = []
        self.request = SimpleNamespace(name="overhead_heart", target="both", resume_token="", interrupt=False)

    @property
    def is_cancel_requested(self):
        if isinstance(self.cancel_after, str):
            return self.cancel_after in self.feedback
        return self.cancel_after is not None and len(self.node.commands) >= self.cancel_after

    def publish_feedback(self, value):
        self.feedback.append(value.state)

    def succeed(self):
        self.node.events.append("succeeded")

    def canceled(self):
        assert self.is_cancel_requested, "an executing goal cannot transition to CANCELED without ActionCancel"
        self.node.events.append("canceled")

    def abort(self):
        self.node.events.append("aborted")


def server(monkeypatch, *, keep_lease=True, cancel_after=None):
    clock = Clock()
    monkeypatch.setattr(motion.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(motion.time, "sleep", clock.sleep)
    node = HoldNode(clock)
    value = object.__new__(motion.NamedMotionServer)
    value._node = node
    value._lock = threading.Lock()
    value._running = None
    value._generation = 0
    value._returning = False
    value._return_motions = set()
    value._display_completion = {}
    value._trajectory_modes = ["policy_stream"]
    value._admission_failure = lambda: None
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


def request_target(value, key, field, target):
    value._on_hold_lease(SimpleNamespace(data=json.dumps({"goal_id": key[0], "motion": key[1], field: target})))


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


def test_pending_stop_uses_guarded_target_without_entering_template(monkeypatch):
    value, node, goal, key = server(monkeypatch)
    value._trajectories[key[1]].positions = ((1.0,) * 14,) * 2
    value._trajectories[key[1]].hands = ((1.0,) * 20,) * 2
    node.validate_named_hold_path = lambda name, times, positions: (
        max(positions[-1]) < 0.5,
        "template must not be validated or commanded for an already stopped goal",
    )
    request_target(value, key, "stop_target", [0.1] * 34)
    result = value._execute_admitted(goal)
    identity = hashlib.sha256(json.dumps([0.1] * 34, separators=(",", ":")).encode()).hexdigest()
    assert result.success
    assert json.loads(result.message) == {"motion": key[1], "stop_target_sha256": identity}
    assert node.commands[-1] == pytest.approx([0.1] * 34)
    assert all(max(point) <= 0.1 for point in node.commands)
    assert "entering" not in goal.feedback and "holding" not in goal.feedback
    assert "stopping" in goal.feedback
    assert node.events[-2:] == ["release", "succeeded"]
    assert node.claims == node.releases == [1]
    assert set(node.command_owners) == {(1, 1)}
    assert not node.faults
    assert value._hold_stop_targets == value._hold_exit_targets == value._hold_leases == {}


def test_pending_stop_still_requires_public_stream_drain(monkeypatch):
    value, node, goal, key = server(monkeypatch)
    request_target(value, key, "stop_target", [0.1] * 34)
    node.claim_ok = False
    ok, code, _message = run(value, goal)
    assert not ok and code == motion.ExecuteNamedMotion.Result.BUSY
    assert node.commands == node.releases == []


def test_invalid_pending_stop_latches_hold_without_claim_or_movement(monkeypatch):
    value, node, goal, key = server(monkeypatch)
    request_target(value, key, "stop_target", [3.0] * 34)
    result = value._execute_admitted(goal)
    assert not result.success
    assert node.claims == node.commands == node.releases == []
    assert node.faults and "succeeded" not in node.events


@pytest.mark.parametrize("phase", ["entering", "holding", "exiting"])
def test_stop_interrupts_each_active_bridge_from_fresh_actual_state(monkeypatch, phase):
    value, node, goal, key = server(monkeypatch)
    value._trajectories[key[1]].positions = ((0.3,) * 14,) * 2
    value._trajectories[key[1]].hands = ((0.3,) * 20,) * 2
    if phase == "exiting":
        request_target(value, key, "exit_target", [0.6] * 34)
    injected = []

    def on_sleep():
        value._hold_leases[key] = node.clock.now
        if phase in goal.feedback and not injected:
            injected.append(len(node.commands))
            # A lagging measurement differs from the commanded template/exit.
            node.actual = [0.025] * 34
            request_target(value, key, "stop_target", [-0.1] * 34)
            request_target(value, key, "exit_target", [0.8] * 34)
            request_target(value, key, "stop_target", [-0.2] * 34)

    node.clock.on_sleep = on_sleep
    assert run(value, goal)[0]
    assert injected and "stopping" in goal.feedback
    assert node.commands[-1] == pytest.approx([-0.1] * 34)
    assert any(points == [[0.025] * 34, [-0.1] * 34] for _name, _times, points in node.guard_paths)
    assert all(max(point) <= 0.3 for point in node.commands[injected[0] :])
    assert node.claims == node.releases == [1]
    assert set(node.command_owners) == {(1, 1)}
    assert node.faults == []


@pytest.mark.parametrize("phase", ["entering", "exiting"])
def test_stop_interrupts_measured_arrival_wait(monkeypatch, phase):
    value, node, goal, key = server(monkeypatch)
    if phase == "exiting":
        request_target(value, key, "exit_target", [0.1] * 34)
    original_wait = value._hold_wait_arrival
    original_sample = node.measured_named_hold_sample
    waiting = []

    def wait(*args, **kwargs):
        if kwargs.get("state", "entering") == phase:
            waiting.append(True)
            node.measured_named_hold_sample = lambda **_kwargs: ([0.2] * 34, node.clock.now)
        return original_wait(*args, **kwargs)

    def on_sleep():
        value._hold_leases[key] = node.clock.now
        if waiting and not value._hold_stop_targets:
            node.measured_named_hold_sample = original_sample
            request_target(value, key, "stop_target", [-0.1] * 34)

    value._hold_wait_arrival = wait
    node.clock.on_sleep = on_sleep
    assert run(value, goal)[0]
    assert waiting and "stopping" in goal.feedback
    assert node.commands[-1] == pytest.approx([-0.1] * 34)
    assert not node.faults


@pytest.mark.parametrize("rejection", ["limits", "geometry"])
def test_rejected_stop_does_not_move_and_latches_hold(monkeypatch, rejection):
    value, node, goal, key = server(monkeypatch)
    if rejection == "geometry":
        node.validate_named_hold_path = lambda name, times, positions: (
            max(positions[-1]) < 0.05,
            "stop violates test head clearance",
        )

    def on_sleep():
        value._hold_leases[key] = node.clock.now
        if "holding" in goal.feedback:
            request_target(value, key, "stop_target", [3.0 if rejection == "limits" else 0.1] * 34)

    node.clock.on_sleep = on_sleep
    result = value._execute_admitted(goal)
    assert not result.success
    assert node.commands and all(point == [0.0] * 34 for point in node.commands)
    assert "fault" in node.events
    assert node.events.index("fault") < node.events.index("release") < node.events.index("aborted")
    assert "succeeded" not in node.events and "canceled" not in node.events


def test_stop_requires_measured_arrival_before_success(monkeypatch):
    value, node, goal, key = server(monkeypatch)
    request_target(value, key, "stop_target", [0.1] * 34)
    node.follows_commands = False
    result = value._execute_admitted(goal)
    assert not result.success and "arrival unconfirmed" in result.message
    assert node.faults and node.releases == [1]
    assert "succeeded" not in node.events


def test_stop_lease_loss_still_latches_hold(monkeypatch):
    value, node, goal, key = server(monkeypatch, keep_lease=False)
    value._hold_lease_timeout_s = 0.05
    request_target(value, key, "stop_target", [0.2] * 34)
    result = value._execute_admitted(goal)
    assert not result.success and "lease expired" in result.message
    assert node.faults and node.releases == [1]
    assert "succeeded" not in node.events


@pytest.mark.parametrize("failure", ["feedback", "admission", "stop_epoch", "absolute_cap"])
def test_stopping_preserves_runtime_failure_guards(monkeypatch, failure):
    value, node, goal, key = server(monkeypatch)
    request_target(value, key, "stop_target", [0.2] * 34)

    def on_sleep():
        value._hold_leases[key] = node.clock.now
        if "stopping" in goal.feedback:
            if failure == "feedback":
                node.measured_named_hold_sample = lambda **kwargs: None
            elif failure == "admission":
                node.output_admitted = lambda: False
            elif failure == "stop_epoch":
                node.runtime_state.stop_epoch += 1
            else:
                value._hold_max_duration_s = 0.01

    node.clock.on_sleep = on_sleep
    result = value._execute_admitted(goal)
    assert not result.success
    assert node.faults and node.releases == [1]
    assert "succeeded" not in node.events and "canceled" not in node.events


def test_stop_release_failure_prevents_succeeded_transition(monkeypatch):
    value, node, goal, key = server(monkeypatch)
    request_target(value, key, "stop_target", [0.1] * 34)
    node.release_ok = False
    result = value._execute_admitted(goal)
    assert not result.success and "release was not confirmed" in result.message
    assert node.faults and node.releases == [1]
    assert "succeeded" not in node.events


def test_action_cancel_has_priority_over_pending_graceful_stop(monkeypatch):
    value, node, goal, key = server(monkeypatch, cancel_after=0)
    node.actual = [0.05] * 34
    request_target(value, key, "stop_target", [0.1] * 34)
    result = value._execute_admitted(goal)
    assert not result.success and result.error_code == motion.ExecuteNamedMotion.Result.CANCELLED
    assert node.commands == [[0.05] * 34]
    assert node.events[-2:] == ["release", "canceled"]
    assert "stopping" not in goal.feedback


def test_action_cancel_during_stopping_holds_full_tracking_lag_target(monkeypatch):
    value, node, goal, key = server(monkeypatch, cancel_after="stopping")
    node.actual = [0.05] * 34
    request_target(value, key, "stop_target", [0.1] * 34)
    result = value._execute_admitted(goal)
    assert not result.success and result.error_code == motion.ExecuteNamedMotion.Result.CANCELLED
    assert node.commands[-1] == pytest.approx([0.05] * 34)
    assert node.events[-2:] == ["release", "canceled"]
    assert not node.faults


def test_stop_target_is_uuid_bound_first_fixed_and_cleaned_with_lease(monkeypatch):
    value, node, _goal, key = server(monkeypatch)
    other = (uuid.uuid4().hex, key[1])
    request_target(value, other, "stop_target", [0.2] * 34)
    assert key not in value._hold_stop_targets
    request_target(value, key, "stop_target", [0.1] * 34)
    request_target(value, key, "exit_target", [0.3] * 34)
    request_target(value, key, "stop_target", [0.4] * 34)
    assert value._hold_stop_targets[key] == (0.1,) * 34
    assert key not in value._hold_exit_targets
    node.clock.now += 2.0
    value._on_hold_lease(SimpleNamespace(data=json.dumps({"goal_id": key[0], "motion": key[1]})))
    assert value._hold_stop_targets == {}


def test_invalid_or_conflicting_stop_lease_gets_no_acknowledgement(monkeypatch):
    value, node, _goal, key = server(monkeypatch)
    for payload in (
        {"goal_id": key[0], "motion": key[1], "exit_target": [0.0] * 34, "stop_target": [0.0] * 34},
        {"goal_id": key[0], "motion": key[1], "stop_target": [True] * 34},
        {"goal_id": key[0], "motion": key[1], "stop_target": [0.0] * 33},
        {"goal_id": key[0], "motion": key[1], "stop_target": [float("nan")] * 34},
        {"goal_id": key[0], "motion": key[1], "stop_target": None},
        {"goal_id": key[0], "motion": "unknown", "stop_target": [0.0] * 34},
    ):
        value._on_hold_lease(SimpleNamespace(data=json.dumps(payload)))
    assert node.acks == []
    assert value._hold_stop_targets == {}
