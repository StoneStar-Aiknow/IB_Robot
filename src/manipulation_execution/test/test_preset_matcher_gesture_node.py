"""preset_motion_matcher_node with the hand gesture input: off is today's node, on adds heart/holy/xsign.

Real rclpy node, callbacks called directly with a controlled clock; the state
publisher is swapped for a recorder. The pose matcher is the real one except
where a test needs the overhead heart active, then a stand-in returns it.
"""

import json
import os
import time
from types import SimpleNamespace

import pytest
import rclpy
from rclpy.parameter import Parameter
from std_msgs.msg import String

from manipulation_execution import preset_motion_matcher_node as module
from manipulation_execution.preset_motions.matcher import MatchResult
from manipulation_execution.preset_motions.match_signal import _POSE_REASONS, TEMPLATE_NAMES, decode_match_state



class Recorder:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(json.loads(msg.data))

    def take(self):
        out, self.msgs = self.msgs, []
        return out


class Clock:
    """The node's (ROS, monotonic) callback time. ROS starts at the wall clock: the message ``stamp``
    is the node clock's own now(), as for the overhead heart."""

    def __init__(self):
        self.ros, self.mono = time.time(), 50.0

    def advance(self, s):
        self.ros += s
        self.mono += s


@pytest.fixture
def make():
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"
    rclpy.init()
    nodes = []

    def _make(**params):
        node = module.PresetMotionMatcherNode(
            parameter_overrides=[Parameter(k, value=v) for k, v in params.items()]
        )
        nodes.append(node)
        clock = Clock()
        node._times = lambda: (clock.ros, clock.mono)
        node._match_state_pub = Recorder()
        node._events_pub = Recorder()
        node.clock = clock
        node.seq = 0
        return node

    yield _make
    for node in nodes:
        node.destroy_node()
    rclpy.shutdown()


def session(node, running=True):
    node.seq += 1
    state = {"seq": node.seq, "stamp": node.clock.ros, "holding": running, "releasing": False,
             "state": "MIRRORING" if running else "WAIT", "phase": "running" if running else "idle"}  # fmt: skip
    node._state(String(data=json.dumps(state)))


def actor(node, actor_id=3):
    msg = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=int(node.clock.ros), nanosec=int(node.clock.ros % 1 * 1e9))),
        actor_id=actor_id, actor_present=True, start_ready=True,
    )  # fmt: skip
    node._actor(msg)


def gesture(node, cls="holy", score=0.9, actor_id=3, dt=0.066):
    """One gesture run stamped now (camera time = ROS time here), then time moves on by dt."""
    stamp = int(round(node.clock.ros * 1e9))
    run = {"schema": "x2_hand_gesture/v1", "stamp_ns": stamp, "actor_id": actor_id, "cls": cls, "score": score,
           "crop": "both", "roi": [0, 0, 100, 100], "infer_ms": 6.0}  # fmt: skip
    node._gesture_msg(String(data=json.dumps(run)))
    node.clock.advance(dt)
    return stamp


def by_template(msgs):
    out = {}
    for m in msgs:
        out.setdefault(m["template_name"], []).append(m)
    return out


def test_off_has_no_gesture_input_and_publishes_overhead_heart_only(make):
    node = make()
    assert node._gesture is None and node.count_subscribers("/hri/hand/gesture") == 0
    session(node)
    actor(node)
    node._tick()
    msgs = node._match_state_pub.take()
    assert len(msgs) == 3 and {m["template_name"] for m in msgs} == {"overhead_heart"}
    assert not hasattr(node, "_gesture_msg") or node.count_subscribers("/hri/hand/gesture") == 0


def test_on_and_off_publish_the_same_overhead_heart(make):
    def run(node):
        session(node)
        actor(node)
        for _ in range(3):
            node._tick()
            node.clock.advance(0.05)
        return [m for m in node._match_state_pub.take() if m["template_name"] == "overhead_heart"]

    off, on = run(make()), run(make(gesture_enabled=True))
    strip = [{k: v for k, v in m.items() if k != "stamp"} for m in off]
    assert strip == [{k: v for k, v in m.items() if k != "stamp"} for m in on]


def test_on_subscribes_and_every_publish_is_one_message_per_template(make):
    node = make(gesture_enabled=True, gesture_topic="/t/gesture")
    assert node.count_subscribers("/t/gesture") == 1
    session(node)
    actor(node)
    node._match_state_pub.take()
    node._tick()
    node._tick()
    msgs = node._match_state_pub.take()
    assert [m["template_name"] for m in msgs] == list(TEMPLATE_NAMES) * 2
    for m in msgs:
        assert set(m) == {"schema_version", "stamp", "template_name", "session_epoch", "actor_id",
                          "source_stamp_ns", "sequence", "active", "input_valid", "reason"}  # fmt: skip
        assert m["schema_version"] == 1
    gestures = by_template(msgs)
    for name in ("heart", "holy", "xsign"):
        first, second = gestures[name]
        assert second["sequence"] == first["sequence"] + 1
        assert first["reason"] in _POSE_REASONS and not first["active"] and not first["input_valid"]
        assert first["actor_id"] == 3 and first["session_epoch"] == gestures["overhead_heart"][0]["session_epoch"]


def test_holy_votes_in_and_reads_as_a_valid_active_state(make):
    node = make(gesture_enabled=True)
    session(node)
    actor(node)
    stamps = [gesture(node) for _ in range(5)]
    node._match_state_pub.take()
    node._tick()
    holy = by_template(node._match_state_pub.take())["holy"][0]
    assert holy["active"] and holy["input_valid"] and holy["reason"] == "holding"
    assert holy["source_stamp_ns"] == stamps[-1] and holy["actor_id"] == 3
    signal = decode_match_state(
        holy, now_ros_s=node.clock.ros, source_after_ns=stamps[0] - 1, actor_id=3, templates=TEMPLATE_NAMES
    )
    assert signal.active and signal.valid and signal.template_name == "holy"
    # The default (overhead only, as mirror_session reads it) refuses it.
    assert decode_match_state(holy, now_ros_s=node.clock.ros, source_after_ns=0, actor_id=3).reason == (
        "template_invalid"
    )


def test_time_exit_on_the_tick(make):
    node = make(gesture_enabled=True)
    session(node)
    actor(node)
    for _ in range(5):
        gesture(node)
    assert by_template(node._match_state_pub.take())["holy"][-1]["active"]
    node.clock.advance(0.3)  # 0.366 s after the last holy run
    actor(node)  # the ActorState callback publishes too: the exit is reported there
    node._tick()
    first, second = by_template(node._match_state_pub.take())["holy"]
    assert not first["active"] and first["reason"] == "pose_exited"
    assert not second["active"] and second["reason"] == "confirming"  # old votes, no fresh run: no re-entry


def test_other_actor_and_actor_change(make):
    node = make(gesture_enabled=True)
    session(node)
    actor(node, 3)
    for _ in range(6):
        gesture(node, actor_id=8)
    node._tick()
    assert not by_template(node._match_state_pub.take())["holy"][-1]["active"]
    for _ in range(5):
        gesture(node)
    node._tick()
    assert by_template(node._match_state_pub.take())["holy"][-1]["active"]
    actor(node, 8)
    holy = by_template(node._match_state_pub.take())["holy"][-1]
    assert not holy["active"] and holy["actor_id"] == 8 and holy["reason"] == "pose_exited"


def test_nothing_counts_without_a_running_session(make):
    node = make(gesture_enabled=True)
    actor(node)
    for _ in range(6):
        gesture(node)
    node._tick()
    holy = by_template(node._match_state_pub.take())["holy"][-1]
    assert not holy["active"] and not holy["input_valid"]
    assert node._gesture.counted == 0


def test_session_end_drops_the_gesture(make):
    node = make(gesture_enabled=True)
    session(node)
    actor(node)
    for _ in range(5):
        gesture(node)
    session(node, running=False)
    holy = by_template(node._match_state_pub.take())["holy"][-1]
    assert not holy["active"] and not holy["input_valid"]


def test_one_template_at_a_time_first_keeps_it(make):
    node = make(gesture_enabled=True, gesture_vote_json=json.dumps({"min_frames": 2}))
    session(node)
    actor(node)
    for _ in range(3):
        gesture(node, "holy")
    for _ in range(5):
        gesture(node, "xsign")  # xsign is now the majority, holy still has a run within 0.3 s at first
    for msgs in [node._match_state_pub.take()]:
        groups = [msgs[i : i + 4] for i in range(0, len(msgs), 4)]
        for g in groups:
            assert [m["template_name"] for m in g] == list(TEMPLATE_NAMES)
            assert sum(m["active"] for m in g) <= 1
    node._tick()
    last = by_template(node._match_state_pub.take())
    assert last["xsign"][-1]["active"] and not last["holy"][-1]["active"]


def test_overhead_heart_beats_the_hand_heart(make):
    node = make(gesture_enabled=True)
    session(node)
    actor(node)
    for _ in range(5):
        gesture(node, "heart")
    node._tick()
    assert by_template(node._match_state_pub.take())["heart"][-1]["active"]

    epoch = node._epoch
    overhead = MatchResult("overhead_heart", 0.9, True, False, False, "holding", 3,
                           int(node.clock.ros * 1e9), 5, epoch, {})  # fmt: skip
    node._matcher = SimpleNamespace(
        set_session=lambda **_k: overhead, tick=lambda **_k: overhead, observe_actor=lambda *_a, **_k: None
    )
    gesture(node, "heart", dt=0.01)
    msgs = by_template(node._match_state_pub.take())
    assert msgs["overhead_heart"][-1]["active"]
    heart = msgs["heart"][-1]
    assert not heart["active"] and heart["reason"] == "confirming" and heart["input_valid"]


def test_bad_vote_parameters_fail_the_node(make):
    with pytest.raises((TypeError, ValueError)):
        make(gesture_enabled=True, gesture_vote_json=json.dumps({"window": 1.0}))
    with pytest.raises(ValueError):
        make(gesture_enabled=True, gesture_vote_json=json.dumps({"min_frames": 0}))


def test_status_has_a_gesture_section_only_when_on(make):
    for enabled in (False, True):
        node = make(gesture_enabled=enabled)
        node._status_pub = Recorder()
        node._status()
        assert ("gesture" in node._status_pub.msgs[0]) is enabled
