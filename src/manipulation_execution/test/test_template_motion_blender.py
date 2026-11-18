"""A blender with real label / port / CBF monitors and client, a fake clock and a scripted data plane. No ROS."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from manipulation_execution.mirror_session.core import MirrorSession, SessionConfig, SessionState
from manipulation_execution.mirror_session.core import Phase as SessionPhase
from manipulation_execution.mirror_session.hooks import (
    ActorSignal,
    CoverDetector,
    PearWindow,
    SessionHooks,
    SessionInputs,
    UpperBodyCommand,
    WrapUpContext,
    WrapUpPurpose,
    WrapUpStep,
)
from manipulation_execution.mirror_session.mock_playback import MockPlaybackHooks
from manipulation_execution.mirror_session.runtime import RuntimeObservation, WaitReason
from manipulation_execution.template_motion.blender import BlendConfig, Blender, Phase
from manipulation_execution.template_motion.hooks import TemplateBlendHooks
from manipulation_execution.template_motion.labels import LabelBook
from manipulation_execution.template_motion.port import CbfMonitor, PlayClient, PortMonitor
from manipulation_execution.template_motion.registry import check_segment

TICK = 0.02
MIRROR = (0.0, 0.2, 0.0, -0.4, 0.0, 0.0, 0.0, 0.0, -0.2, 0.0, -0.4, 0.0, 0.0, 0.0)
HAND_MIRROR = (0.5,) * 20


def pose(**changes):
    values = list(MIRROR)
    for key, value in changes.items():
        values[int(key[1:])] = value
    return tuple(values)


P_HEART = pose(j0=-0.6, j7=-0.6)  # 0.6 rad from the mirror
P_LEFT = pose(j0=-0.6, j1=0.6)
P_RIGHT = pose(j7=-0.6, j8=-0.6)


def segment(name, P, hand=0.2, **timing):
    T = tuple(p + (0.3 if j in (0, 7) else 0.0) for j, p in enumerate(P))
    raw = {
        "P": list(P),
        "T": list(T),
        "hand_P": [hand] * 20,
        "hand_T": [hand + 0.1] * 20,
        "t_in": 1.0,
        "t_out": 1.0,
        "t_out_stop": 0.5,
        "hold_max_s": 4.0,
        "start_tol_rad": 0.03,
        "hand_start_tol": 0.05,
        "report_sha256": "0" * 64,
        **timing,
    }
    return check_segment(name, raw)


SEGMENTS = {
    "heart": segment("heart", P_HEART),
    "xsign_left_front": segment("xsign_left_front", P_LEFT),
    "xsign_right_front": segment("xsign_right_front", P_RIGHT),
}
TEMPLATES = {"heart": ["heart"], "xsign": ["xsign_left_front", "xsign_right_front"]}


class Rig:
    """One ``step`` is one session tick: the monitors hear what is scripted, the blender steps, the gate sends.

    The gate sends exactly what the blender (or the mirror) asked for; ``gate_lag``
    keeps it from moving, as a limiter that never reaches. The hand output follows
    the blender's hand target at once. ``port`` is what the data plane publishes
    each tick (None: silent).
    """

    def __init__(self, config=None, templates=TEMPLATES, hand_travel_s=None):
        self.now = 100.0
        self.labels = LabelBook(templates, stale_s=0.5)
        self.port_monitor = PortMonitor()
        self.cbf_monitor = CbfMonitor()
        self.messages = []
        self.client = PlayClient(self.messages.append, self.port_monitor)
        self.logs = []
        self.blender = Blender(
            config or BlendConfig(),
            SEGMENTS,
            templates,
            self.labels,
            self.port_monitor,
            self.cbf_monitor,
            self.client,
            self.logs.append,
            hand_travel_s,
        )
        self.mirror = MIRROR
        self.active = {}  # template -> actor id, published every tick
        self.port = {"state": "IDLE"}
        self.cbf = {}
        self.actor = ActorSignal(True, 7, 0.0)
        self.source = "yolox"
        self.publish_enabled = True
        self.gate_lag = False
        self.last_sent = None
        self.hand = HAND_MIRROR
        self.out = []  # every arm the gate sent

    def publish_port(self, **message):
        self.port = message
        self._port_now()

    def _port_now(self):
        if self.port is not None:
            self.port_monitor.observe(json.dumps({"schema": "x2_template_status/v1", **self.port}), self.now)

    def inputs(self):
        return SimpleNamespace(
            publish_enabled=self.publish_enabled,
            actor=self.actor,
            window=PearWindow((SimpleNamespace(source=self.source),)),
            last_sent=self.last_sent,
        )

    def hear(self):
        """What the monitors receive this tick."""
        for template in self.labels._templates:
            actor = self.active.get(template)
            message = {"schema_version": 1, "template_name": template, "actor_id": actor if actor is not None else -1}
            message.update(active=actor is not None, input_valid=True)
            self.labels.observe(json.dumps(message), self.now)
        self._port_now()
        if self.cbf is not None:
            self.cbf_monitor.observe(json.dumps(self.cbf), self.now)

    def step(self, dt=TICK):
        self.now += dt
        self.hear()
        arm = self.blender.step(self.now, self.inputs(), self.mirror)
        sent = self.mirror if arm is None else arm
        if not (self.gate_lag and self.last_sent is not None):
            self.last_sent = UpperBodyCommand(tuple(sent), (0.0,) * 20)
        self.out.append(tuple(self.last_sent.arm))
        target = self.blender.hand_target(self.now)
        self.hand = HAND_MIRROR if target is None else tuple(target)
        self.blender.observe_hand_output(self.hand)
        self.client.pump(self.now)
        return arm

    def run(self, seconds, until=None):
        for _ in range(round(seconds / TICK)):
            self.step()
            if until is not None and until():
                return True
        return until is None

    def sent(self, op):
        return [json.loads(m) for m in self.messages if json.loads(m)["op"] == op]

    def started(self):
        starts = self.sent("start")
        return starts[-1]["seg_id"] if starts else None


def max_step(arms):
    return max(
        (max(abs(a - b) for a, b in zip(x, y, strict=False)) for x, y in zip(arms, arms[1:], strict=False)), default=0.0
    )


def max_jerkless(arms):
    """The largest second difference per tick, any joint."""
    return max(
        (
            max(abs(a - 2 * b + c) for a, b, c in zip(x, y, z, strict=False))
            for x, y, z in zip(arms, arms[1:], arms[2:], strict=False)
        ),
        default=0.0,
    )


# -- helpers ---------------------------------------------------------------------------


def to_wait_ack(rig, template="heart"):
    rig.active[template] = 7
    assert rig.run(4.0, lambda: rig.blender.phase is Phase.WAIT_ACK)
    return rig.started()


def to_playing(rig, state="ENTER_B", P=P_HEART):
    seg_id = to_wait_ack(rig)
    rig.publish_port(state=state, seg_id=seg_id, seg="heart", P_actual=list(P), hand_P=[0.2] * 20)
    rig.step()
    assert rig.blender.phase is Phase.PLAYING
    return seg_id


def exit_reason(rig):
    return rig.blender.snapshot()["last_exit"]


# -- the happy path ---------------------------------------------------------------------


def test_off_while_no_label_the_mirror_passes_through():
    rig = Rig()
    for _ in range(50):
        assert rig.step() is None
    assert rig.blender.phase is Phase.IDLE
    assert rig.messages == [] and rig.blender.hand_target(rig.now) is None


def test_a_label_blends_to_P_plays_and_blends_back_to_the_mirror_smoothly():
    rig = Rig()
    rig.step()
    rig.active["heart"] = 7
    rig.step()
    assert rig.blender.phase is Phase.A
    assert rig.logs[-1].startswith("template enter heart seg=heart actor=7 episode=1")
    assert "T_A=1.10" in rig.logs[-1]  # the default hand travel (1.0) + 0.1 beats 1.875 * 0.6 / 1.5
    assert rig.blender.hand_target(rig.now) == SEGMENTS["heart"].hand_P
    entered = rig.now
    assert rig.run(2.0, lambda: rig.blender.phase is Phase.WAIT_ACK)
    assert 1.1 - 1e-6 <= rig.now - entered <= 1.1 + 2 * TICK
    assert rig.last_sent.arm == P_HEART
    seg_id = rig.started()
    assert rig.sent("start") == [{"op": "start", "seg": "heart", "seg_id": seg_id, "schema": "x2_template_play/v1"}]
    rig.run(0.1)
    assert len(rig.sent("keepalive")) == 5  # one every tick
    rig.publish_port(state="ENTER_B", seg_id=seg_id, seg="heart", P_actual=list(P_HEART), hand_P=[0.2] * 20)
    rig.step()
    assert rig.blender.phase is Phase.PLAYING and rig.blender.played == 1
    rig.run(1.0)
    rig.publish_port(state="HOLD", seg_id=seg_id, seg="heart", P_actual=list(P_HEART), hand_P=[0.2] * 20)
    rig.run(1.0)
    assert rig.blender.phase is Phase.PLAYING and not rig.sent("release")
    del rig.active["heart"]
    rig.step()
    assert rig.blender.phase is Phase.WAIT_DONE
    assert rig.sent("release") == [{"op": "release", "seg_id": seg_id, "fast": False, "schema": "x2_template_play/v1"}]
    rig.publish_port(state="EXIT_B", seg_id=seg_id, seg="heart", P_actual=list(P_HEART), hand_P=[0.2] * 20)
    rig.run(1.0)
    rig.publish_port(state="DONE", seg_id=seg_id, seg="heart", reason="released", P_actual=list(P_HEART))
    rig.step()
    assert rig.blender.phase is Phase.C and exit_reason(rig) == "label_inactive"
    assert rig.blender.hand_target(rig.now) is None
    assert not rig.client.busy  # the keepalive ends with DONE
    assert rig.run(2.0, lambda: rig.blender.phase is Phase.COOLDOWN)
    assert rig.last_sent.arm == MIRROR
    assert rig.run(1.2, lambda: rig.blender.phase is Phase.IDLE)
    assert rig.step() is None
    assert max_step(rig.out) <= 0.03  # 1.5 rad/s
    assert max_jerkless(rig.out) <= 0.0025  # 6.25 rad/s², a min-jerk C of 0.6 rad in 0.8 s
    assert rig.blender.exits == {"label_inactive": 1}


def test_the_same_episode_is_not_entered_twice_a_new_one_is():
    rig = Rig()
    rig.active["heart"] = 7
    rig.run(0.5)
    rig.mirror = pose(j3=-0.2)  # the person keeps moving: A follows the mirror
    rig.actor = ActorSignal(True, 8, 0.0)  # another person: back to the mirror
    rig.step()
    assert exit_reason(rig) == "actor_changed"
    rig.actor = ActorSignal(True, 7, 0.0)
    rig.run(3.0)
    assert rig.blender.entries == 1  # still episode 1
    del rig.active["heart"]
    rig.step()
    rig.active["heart"] = 7
    rig.step()
    assert rig.blender.phase is Phase.A and rig.blender.entries == 2


# -- A ----------------------------------------------------------------------------------


def test_leaving_A_half_way_is_smooth_and_never_faster_than_c_peak_speed():
    rig = Rig()
    rig.active["heart"] = 7
    rig.run(0.56)
    assert rig.blender.phase is Phase.A
    del rig.active["heart"]
    rig.step()
    assert rig.blender.phase is Phase.C and exit_reason(rig) == "label_lost"
    assert rig.run(3.0, lambda: rig.blender.phase is Phase.COOLDOWN)
    assert rig.last_sent.arm == MIRROR
    assert max_step(rig.out) <= 1.5 * 1.03 * TICK
    assert max_jerkless(rig.out) <= 0.0025  # 6.25 rad/s², a min-jerk C of 0.6 rad in 0.8 s
    assert rig.messages == []


def test_a_gate_that_never_reaches_P_times_out_without_a_start():
    rig = Rig()
    rig.gate_lag = True
    rig.active["heart"] = 7
    rig.step()
    rig.step()
    assert rig.run(2.0, lambda: rig.blender.phase is Phase.C)
    assert exit_reason(rig) == "a_timeout" and not rig.messages
    assert rig.run(2.0, lambda: rig.blender.phase is Phase.COOLDOWN)


def test_a_hand_that_does_not_arrive_times_out_too():
    rig = Rig()
    rig.active["heart"] = 7
    rig.step()
    rig.blender.observe_hand_output = lambda values: None  # the hand output stays at the mirror's
    rig.blender._hand_out = HAND_MIRROR
    assert rig.run(2.0, lambda: rig.blender.phase is Phase.C)
    assert exit_reason(rig) == "a_timeout" and not rig.messages


def test_a_cbf_that_holds_during_A_goes_back_to_the_mirror():
    rig = Rig()
    rig.active["heart"] = 7
    rig.run(0.3)
    rig.cbf = {"cbf": {"stuck_s": 0.2, "releasing": False, "stuck_collision": False}}
    rig.step()
    assert rig.blender.phase is Phase.C and exit_reason(rig) == "a_cbf_hold"


def test_T_A_has_no_upper_bound_and_respects_the_hand_travel():
    far = pose(j0=1.8, j7=1.8)  # 2.4 rad from P: 1.875 * 2.4 / 1.5 = 3.0 s
    rig = Rig(hand_travel_s=lambda start, goal: 0.2)
    rig.mirror = far
    rig.active["heart"] = 7
    rig.step()
    assert "T_A=3.00" in rig.logs[-1]
    rig.run(3.2, lambda: rig.blender.phase is Phase.WAIT_ACK)
    assert rig.blender.phase is Phase.WAIT_ACK
    assert max_step(rig.out) <= 1.5 * TICK * 1.01
    rig2 = Rig(hand_travel_s=lambda start, goal: 1.4)
    rig2.step()  # the hand output is known from the first tick on
    rig2.active["heart"] = 7
    rig2.step()
    assert "T_A=1.50" in rig2.logs[-1]


# -- entry gates ------------------------------------------------------------------------


def _never_enters(rig, seconds=1.0):
    entries = rig.blender.entries
    rig.active["heart"] = 7
    rig.run(seconds)
    return rig.blender.entries == entries and rig.blender.phase is Phase.IDLE


def test_entry_needs_a_person_frame():
    rig = Rig()
    rig.source = "center_fallback"
    assert _never_enters(rig)
    rig.blender.step(
        rig.now, SimpleNamespace(publish_enabled=True, actor=rig.actor, window=PearWindow(()), last_sent=None), MIRROR
    )
    assert rig.blender.phase is Phase.IDLE  # a tick without frames keeps the last frame's source
    rig.source = "yolox"
    rig.step()
    assert rig.blender.phase is Phase.A


def test_entry_needs_publish_enabled_and_the_labels_actor():
    rig = Rig()
    rig.publish_enabled = False
    assert _never_enters(rig)
    rig = Rig()
    rig.actor = ActorSignal(True, 8, 0.0)
    assert _never_enters(rig)
    rig = Rig()
    rig.actor = ActorSignal(False, None, 0.0)
    assert _never_enters(rig)


def test_entry_needs_a_fresh_idle_port_and_a_quiet_cbf():
    rig = Rig()
    rig.port = None
    assert _never_enters(rig)
    rig = Rig()
    rig.port = {"state": "HOLD", "seg_id": 5}
    assert _never_enters(rig)
    rig = Rig()
    rig.cbf = {"cbf": {"stuck_s": 0.0, "releasing": True}}
    assert _never_enters(rig)
    rig = Rig()
    rig.cbf = None
    assert _never_enters(rig)
    rig = Rig()
    rig.cbf = {"mode": "x"}  # CBF off: no cbf section, never intervenes
    rig.port = {"state": "DONE", "seg_id": 3}
    rig.active["heart"] = 7
    rig.step()
    assert rig.blender.phase is Phase.A


def test_entry_needs_the_segment_in_the_ports_whitelist():
    rig = Rig()
    rig.port = {"state": "IDLE", "segments": ["xsign_left_front"]}
    assert _never_enters(rig)


def test_xsign_takes_the_closer_P():
    rig = Rig()
    rig.mirror = pose(j7=-0.5, j8=-0.5)
    rig.active["xsign"] = 7
    rig.step()
    assert "seg=xsign_right_front" in rig.logs[-1]
    rig = Rig()
    rig.mirror = pose(j0=-0.5, j1=0.5)
    rig.active["xsign"] = 7
    rig.step()
    assert "seg=xsign_left_front" in rig.logs[-1]


# -- the port ---------------------------------------------------------------------------


def test_no_ack_releases_marks_the_port_suspect_and_returns():
    rig = Rig()
    seg_id = to_wait_ack(rig)
    assert rig.run(0.5, lambda: rig.blender.phase is Phase.C)
    assert exit_reason(rig) == "no_ack"
    assert rig.sent("release") == [{"op": "release", "seg_id": seg_id, "fast": True, "schema": "x2_template_play/v1"}]
    assert rig.blender.snapshot()["port_suspect"]
    rig.run(2.5)
    del rig.active["heart"]
    rig.step()
    assert _never_enters(rig)  # not again this round
    rig.publish_port(state="STOPPED", seg_id=seg_id, reason="released")
    rig.blender.begin_round()
    rig.step()  # the client hears STOPPED after this tick's step
    assert rig.blender.phase is Phase.IDLE and not rig.client.busy
    rig.step()
    assert rig.blender.phase is Phase.A  # a new round trusts the port again; episode 2 was never entered


def test_rejected_goes_back_from_P_and_counts():
    rig = Rig()
    seg_id = to_wait_ack(rig)
    rig.publish_port(state="REJECTED", seg_id=seg_id, reason="busy")
    rig.step()
    assert rig.blender.phase is Phase.C and exit_reason(rig) == "rejected:busy"
    assert rig.blender.exits == {"rejected": 1}
    assert not rig.client.busy
    assert rig.run(3.0, lambda: rig.blender.phase is Phase.IDLE)
    assert rig.blender.entries == 1


def test_an_old_status_of_another_segment_is_not_an_ack():
    rig = Rig()
    seg_id = to_wait_ack(rig)
    rig.publish_port(state="HOLD", seg_id=seg_id + 100)
    rig.step()
    assert rig.blender.phase is Phase.WAIT_ACK


def test_a_silent_port_while_playing_releases_fast_and_ends_on_the_deadline():
    rig = Rig()
    seg_id = to_playing(rig)
    rig.port = None
    assert rig.run(0.6, lambda: rig.blender.phase is Phase.WAIT_DONE)
    assert rig.sent("release")[-1] == {"op": "release", "seg_id": seg_id, "fast": True, "schema": "x2_template_play/v1"}
    waited = rig.now
    assert rig.run(1.0, lambda: rig.blender.phase is Phase.C)
    assert exit_reason(rig) == "status_stale"  # what the log says, not done_timeout
    assert rig.now - waited == pytest.approx(0.5 + 0.3, abs=2 * TICK)  # t_out_stop + done_margin_s
    assert rig.client.busy and rig.sent("release")[-1]["fast"]  # one more fast release
    rig.run(0.5 + TICK)  # wind_down_s: then the keepalive ends, the port's own timeout stops it
    assert not rig.client.busy
    sent = len(rig.sent("keepalive"))
    assert rig.run(3.0, lambda: rig.blender.phase is Phase.IDLE)
    assert len(rig.sent("keepalive")) == sent
    del rig.active["heart"]
    rig.step()
    rig.active["heart"] = 7
    rig.port = {"state": "IDLE"}
    rig.step()
    assert rig.blender.phase is Phase.A  # nothing stuck: the next activation plays


def test_a_data_plane_restart_mid_playback_goes_back_to_the_mirror():
    rig = Rig()
    seg_id = to_playing(rig)
    keepalives = len(rig.sent("keepalive"))
    restarted = rig.now
    rig.port = None  # the restarted port: IDLE, no segment, every 100 ms (the label stays active)
    for k in range(60):
        if k % 5 == 0:
            rig.publish_port(state="IDLE", seg_id=None)
        rig.step()
        if rig.blender.phase is Phase.C:
            break
    assert rig.blender.phase is Phase.C and exit_reason(rig) == "port_lost"
    assert 0.3 <= rig.now - restarted <= 0.3 + 0.1 + 2 * TICK
    assert not rig.client.busy  # no keepalive to a port that forgot the segment
    assert rig.sent("release")[-1]["seg_id"] == seg_id and rig.sent("release")[-1]["fast"]
    assert rig.sent("keepalive")[keepalives:] and rig.sent("keepalive")[-1]["seg_id"] == seg_id
    sent = len(rig.messages)
    rig.port = {"state": "IDLE", "seg_id": None}
    assert rig.run(3.0, lambda: rig.blender.phase is Phase.IDLE)
    assert len(rig.messages) == sent and rig.last_sent.arm == MIRROR
    assert max_step(rig.out) <= 0.03
    assert "total=" in rig.logs[-1] and "reason=port_lost" in rig.logs[-1]
    del rig.active["heart"]
    rig.step()
    rig.active["heart"] = 7
    rig.step()
    assert rig.blender.phase is Phase.A  # the next activation plays


def test_one_refusal_of_another_start_mid_playback_is_not_a_lost_port():
    rig = Rig()
    seg_id = to_playing(rig, state="HOLD")
    ours = rig.port
    rig.publish_port(state="REJECTED", seg_id=seg_id + 100, seg="holy", reason="busy")
    rig.step()
    rig.port = ours
    rig.run(1.0)
    assert rig.blender.phase is Phase.PLAYING and rig.client.busy


def test_another_segments_playback_is_a_lost_port():
    rig = Rig()
    seg_id = to_playing(rig)
    rig.port = {"state": "HOLD", "seg_id": seg_id + 1, "seg": "holy"}
    assert rig.run(0.5, lambda: rig.blender.phase is Phase.C)
    assert exit_reason(rig) == "port_lost"


def test_a_playback_never_outlasts_t_in_hold_max_and_t_out():
    rig = Rig()
    seg_id = to_wait_ack(rig)
    started = rig.now
    rig.port = {"state": "ENTER_B", "seg_id": seg_id, "seg": "heart", "P_actual": list(P_HEART)}
    assert rig.run(9.0, lambda: rig.blender.phase is Phase.WAIT_DONE)  # the port plays ENTER_B for ever
    assert rig.now - started == pytest.approx(1.0 + 4.0 + 1.0 + 1.0, abs=2 * TICK)  # + play_margin_s
    assert rig.sent("release")[-1]["fast"]
    assert rig.run(1.0, lambda: rig.blender.phase is Phase.C)
    assert exit_reason(rig) == "play_timeout"  # what the log says, not done_timeout
    rig.run(0.5 + TICK)
    assert not rig.client.busy  # wound down although the port still says ENTER_B
    assert rig.run(3.0, lambda: rig.blender.phase is Phase.IDLE) and rig.last_sent.arm == MIRROR


def test_a_port_restart_while_waiting_for_done_is_a_lost_port():
    rig = Rig()
    to_playing(rig, state="HOLD")
    rig.run(0.9)
    del rig.active["heart"]
    rig.step()
    assert rig.blender.phase is Phase.WAIT_DONE
    waited = rig.now
    rig.port = {"state": "IDLE", "seg_id": None}
    assert rig.run(1.0, lambda: rig.blender.phase is Phase.C)
    assert exit_reason(rig) == "port_lost" and rig.now - waited < 0.4 and not rig.client.busy
    assert rig.sent("release")[-1]["fast"]


def test_the_data_planes_own_exit_is_waited_for():
    rig = Rig()
    seg_id = to_playing(rig, state="HOLD")
    rig.publish_port(state="EXIT_B", seg_id=seg_id, reason="hold_max", P_actual=list(P_HEART))
    rig.step()
    assert rig.blender.phase is Phase.WAIT_DONE
    rig.run(0.5)
    rig.publish_port(state="DONE", seg_id=seg_id, reason="hold_max")
    rig.step()
    assert rig.blender.phase is Phase.C and exit_reason(rig) == "hold_max"


def test_stopped_while_playing_goes_back_from_P_actual():
    rig = Rig()
    seg_id = to_playing(rig)
    rig.publish_port(state="STOPPED", seg_id=seg_id, reason="arm_off", P_actual=list(P_HEART))
    rig.step()
    assert rig.blender.phase is Phase.C and exit_reason(rig) == "stopped:arm_off"
    assert rig.blender.exits == {"stopped": 1}


def test_a_late_label_loss_waits_for_min_hold():
    rig = Rig()
    seg_id = to_playing(rig)
    del rig.active["heart"]
    rig.run(1.0)
    assert rig.blender.phase is Phase.PLAYING  # never in HOLD yet
    rig.publish_port(state="HOLD", seg_id=seg_id, P_actual=list(P_HEART))
    assert rig.run(1.0, lambda: rig.blender.phase is Phase.WAIT_DONE)
    assert rig.sent("release")[-1]["fast"] is False


# -- the round ends ---------------------------------------------------------------------


def wrap_until_done(rig, purpose=WrapUpPurpose.STOP, budget=2.5):
    steps, elapsed = [], 0.0
    target = UpperBodyCommand(pose(j3=-0.1), (0.0,) * 20)
    while elapsed <= budget:
        rig.now += TICK
        rig._port_now()
        context = WrapUpContext(purpose, rig.last_sent, None, target, elapsed, budget)
        step = rig.blender.wrap(context, lambda arm: UpperBodyCommand(tuple(arm), (0.0,) * 20))
        if step is None:
            return None, steps, elapsed
        steps.append(step)
        if step.command is not None:
            rig.last_sent = step.command
        rig.client.pump(rig.now)
        if step.done:
            return step, steps, elapsed
        elapsed += TICK
    raise AssertionError("wrap-up ran past its budget")


def test_a_round_that_ends_while_playing_releases_fast_and_returns_within_the_budget():
    rig = Rig()
    seg_id = to_playing(rig, state="HOLD")
    rig.blender.end_round(rig.now)
    assert rig.blender.phase is Phase.STOP_WAIT and exit_reason(rig) == "stop"
    assert rig.sent("release")[-1] == {"op": "release", "seg_id": seg_id, "fast": True, "schema": "x2_template_play/v1"}
    assert rig.blender.hand_target(rig.now) == (0.2,) * 20
    rig.publish_port(state="EXIT_B", seg_id=seg_id, reason="released", P_actual=list(P_HEART))
    last, steps, elapsed = wrap_until_done(rig)
    assert last.done and last.command.arm == pose(j3=-0.1)
    assert elapsed <= 2.5 - 0.3 + TICK
    arms = [s.command.arm for s in steps]
    assert arms[0] == P_HEART
    assert max_step(arms) <= 2.0 * TICK
    assert rig.blender.phase is Phase.IDLE and rig.blender.hand_target(rig.now) is None


def test_the_stop_wrap_ends_early_when_the_port_is_done():
    rig = Rig()
    seg_id = to_playing(rig)
    rig.blender.end_round(rig.now)
    rig.publish_port(state="DONE", seg_id=seg_id, reason="released", P_actual=list(P_HEART))
    last, steps, _ = wrap_until_done(rig)
    held = [s for s in steps if s.command.arm == P_HEART]
    assert len(held) <= 1  # no hold once DONE
    assert last.done


def test_the_hand_goes_back_on_its_own_when_no_wrap_up_runs():
    rig = Rig()
    to_playing(rig)
    rig.blender.end_round(rig.now)
    assert rig.blender.hand_target(rig.now + 0.6) is not None
    assert rig.blender.hand_target(rig.now + 0.71) is None  # t_out_stop + wrap_done_margin_s


def test_a_round_that_ends_in_A_hands_the_wrap_up_back():
    rig = Rig()
    rig.active["heart"] = 7
    rig.run(0.4)
    rig.blender.end_round(rig.now)
    assert rig.blender.phase is Phase.IDLE and exit_reason(rig) == "stop" and not rig.messages
    assert wrap_until_done(rig)[0] is None
    assert rig.blender.hand_target(rig.now) is None


def test_the_approach_wrap_up_is_never_the_blenders():
    rig = Rig()
    to_playing(rig)
    rig.blender.end_round(rig.now)
    assert wrap_until_done(rig, purpose=WrapUpPurpose.APPROACH)[0] is None


def test_disable_round_releases_fast_and_the_client_alone_keeps_it_alive_a_while():
    rig = Rig()
    seg_id = to_playing(rig)
    rig.blender.disable_round("ValueError", rig.now)
    assert rig.blender.phase is Phase.IDLE and rig.blender.exits == {"error": 1}
    assert rig.sent("release")[-1] == {"op": "release", "seg_id": seg_id, "fast": True, "schema": "x2_template_play/v1"}
    line = rig.logs[-1]
    assert line.startswith("template exit heart seg=heart phase=PLAYING reason=error:ValueError d=0.600")
    assert " total=" in line
    keepalives = len(rig.sent("keepalive"))
    assert rig.client.busy
    rig.run(0.9)  # the port still plays (ENTER_B): kept alive, no new entry
    assert len(rig.sent("keepalive")) >= keepalives + 40 and rig.client.busy and rig.blender.entries == 1
    rig.run(0.2)  # t_out_stop 0.5 + wind_down_s 0.5
    assert not rig.client.busy
    sent = len(rig.messages)
    rig.run(0.5)
    assert len(rig.messages) == sent and rig.blender.entries == 1


def test_after_an_error_the_keepalive_ends_on_the_terminal_status():
    rig = Rig()
    seg_id = to_playing(rig)
    rig.blender.disable_round("ValueError", rig.now)
    rig.run(0.2)
    rig.publish_port(state="DONE", seg_id=seg_id, seg="heart", reason="released")
    rig.step()
    assert not rig.client.busy


def test_disable_round_outside_a_playback_logs_nothing_and_sends_nothing():
    rig = Rig()
    rig.blender.disable_round("ValueError", rig.now)
    assert rig.messages == [] and rig.logs == [] and rig.blender.exits == {}


# -- config -----------------------------------------------------------------------------


def test_blend_config_names_bad_keys():
    with pytest.raises(ValueError, match="unknown"):
        BlendConfig.from_mapping({"a_peak": 1.0})
    with pytest.raises(ValueError, match="c_min_s"):
        BlendConfig.from_mapping({"c_min_s": 0})
    with pytest.raises(ValueError, match="arrive_fraction"):
        BlendConfig.from_mapping({"arrive_fraction": 1.5})
    with pytest.raises(ValueError, match="a_min_s"):
        BlendConfig.from_mapping({"a_min_s": float("nan")})
    assert BlendConfig.from_mapping({"cooldown_s": 0, "require_person_frame": False}).cooldown_s == 0.0
    assert BlendConfig.from_mapping(None) == BlendConfig()


def test_an_unknown_segment_is_refused():
    with pytest.raises(ValueError, match="unknown segments"):
        Rig(templates={"heart": ["nope"]})


# -- TemplateBlendHooks ------------------------------------------------------------------

RESET = UpperBodyCommand(pose(j3=-0.1), (0.0,) * 20)


class Inner(SessionHooks):
    """A mirror that always sends MIRROR; post_process passes through."""

    def __init__(self):
        self.cancel = False
        self.calls = []

    def retarget(self, inputs):
        return UpperBodyCommand(MIRROR, (0.3,) * 20)

    def post_process(self, command):
        return command

    def cancel_criteria(self, inputs, round_elapsed_s):
        return self.cancel

    def reset_pose(self, arm_at_takeover):
        return RESET

    def begin_round(self):
        self.calls.append("begin_round")

    def end_round(self):
        self.calls.append("end_round")

    def wrap_up(self, context):
        self.calls.append(("wrap_up", context.purpose))
        return WrapUpStep(RESET, True)


def hooked(rig, inner=None):
    inner = inner or Inner()
    return inner, TemplateBlendHooks(inner, rig.blender, rig.logs.append, clock=lambda: rig.now)


def test_off_the_inner_command_passes_as_the_same_object():
    rig = Rig()
    inner, hooks = hooked(rig)
    command = UpperBodyCommand(MIRROR, (0.3,) * 20)
    inner.retarget = lambda inputs: command
    hooks.begin_round()
    for _ in range(10):
        rig.now += TICK
        assert hooks.retarget(rig.inputs()) is command
    assert inner.calls == ["begin_round"]


def test_a_blender_error_never_reaches_the_session():
    rig = Rig()
    inner, hooks = hooked(rig)
    hooks.begin_round()
    rig.active["heart"] = 7
    rig.step()
    assert rig.blender.phase is Phase.A

    def broken(*args):
        raise ZeroDivisionError("boom")

    rig.blender.step = broken
    raw = UpperBodyCommand(MIRROR, (0.3,) * 20)
    inner.retarget = lambda inputs: raw
    assert hooks.retarget(rig.inputs()) is raw
    assert hooks.errors == 1 and rig.blender.snapshot()["round_disabled"]
    assert rig.blender.exits == {"error": 1}
    assert "templates off for this round" in rig.logs[-2]
    assert rig.logs[-1].startswith("template exit heart seg=heart phase=A reason=error:ZeroDivisionError d=0.600")
    rig.blender.wrap = broken
    context = WrapUpContext(WrapUpPurpose.STOP, None, None, RESET, 0.0, 2.5)
    assert hooks.wrap_up(context) == WrapUpStep(RESET, True)
    rig.blender.hand_target = broken
    assert hooks.hand_target(rig.now) is None
    assert hooks.snapshot()["errors"] == 3


def test_cancel_reason_is_the_inner_hooks():
    rig = Rig()
    inner, hooks = hooked(rig)
    inner.cancel_reason = "left the zone"
    assert hooks.cancel_reason == "left the zone"


def test_a_limited_blend_handed_back_by_the_mock_is_not_taken_for_the_mirror():
    rig = Rig()
    mock = MockPlaybackHooks(lambda playing, round_id: None, reset_pose=RESET.arm, max_step=[0.01] * 14)
    _, hooks = hooked(rig, mock)
    hooks.begin_round()
    mock.acknowledge(mock.snapshot()["round"], True, True)
    rig.last_sent = UpperBodyCommand(MIRROR, (0.0,) * 20)
    mock.offer(MIRROR)
    rig.active["heart"] = 7
    for _ in range(30):
        rig.now += TICK
        rig.hear()
        command = hooks.post_process(hooks.retarget(rig.inputs()))
        rig.last_sent = command
    assert rig.blender.phase is Phase.A
    assert hooks._mirror.arm == MIRROR  # never one of the blended commands


# -- in the real session core ------------------------------------------------------------

IDENTITY = {"id": "fake", "type": "fake", "runtime_name": "isolated", "runtime_version": "1"}


class Runtime:
    def __init__(self):
        self.status = RuntimeObservation("ACTIVE", "idle", False, (), "isolated", "1", IDENTITY, 0.0)

    def get_status(self, timeout_s):
        return self.status

    def set_mode(self, mode, timeout_s):
        self.status = replace(self.status, active_mode=mode)
        return True, mode

    def stop(self, policy, timeout_s):
        self.status = replace(self.status, stop_latched=True)
        return True, "latched"


class Cover(CoverDetector):
    def update(self, frame):
        return False


class Session:
    def __init__(self):
        self.rig = Rig()
        self.inner, self.hooks = hooked(self.rig)
        self.runtime = Runtime()
        self.sent = []

        def sleep(seconds):
            self.rig.now += seconds

        self.session = MirrorSession(
            SessionConfig("policy_stream", IDENTITY),
            self.runtime,
            self.sent.append,
            self.hooks,
            Cover(),
            lambda: self.rig.now,
            sleep,
        )
        assert self.session.startup().idle
        self.runtime.status = replace(self.runtime.status, active_mode="policy_stream")
        self.session.observe_arm_position((0.0,) * 14)
        assert self.session.request("start")[0]
        self.session.tick()
        assert self.session.operator_touch()[0]
        self.sent.clear()

    def tick(self, n=1):
        rig = self.rig
        for _ in range(n):
            rig.now += TICK
            rig.hear()
            self.session.observe_status(self.runtime.status)
            inputs = SessionInputs(
                window=PearWindow((SimpleNamespace(source=rig.source),)), actor=rig.actor, publish_enabled=True
            )
            self.session.tick(inputs)
            target = self.hooks.hand_target(rig.now)
            self.hooks.observe_hand_output(HAND_MIRROR if target is None else target)
            rig.client.pump(rig.now)

    def until(self, predicate, seconds=4.0):
        for _ in range(round(seconds / TICK)):
            self.tick()
            if predicate():
                return True
        return False


def test_in_the_session_a_cancel_mid_playback_returns_through_P_actual_within_the_budget():
    s = Session()
    rig = s.rig
    s.tick(5)
    assert all(command.arm == MIRROR for command in s.sent)
    rig.active["heart"] = 7
    assert s.until(lambda: rig.blender.phase is Phase.WAIT_ACK)
    seg_id = rig.started()
    rig.publish_port(state="HOLD", seg_id=seg_id, seg="heart", P_actual=list(P_HEART), hand_P=[0.2] * 20)
    s.tick(10)
    assert rig.blender.phase is Phase.PLAYING
    s.inner.cancel = True
    s.tick()
    assert s.session.phase == SessionPhase.WRAP_UP
    assert rig.blender.phase is Phase.STOP_WAIT
    assert rig.sent("release")[-1]["fast"] is True
    rig.publish_port(state="EXIT_B", seg_id=seg_id, reason="released", P_actual=list(P_HEART))
    wrap_started = rig.now
    assert s.until(lambda: s.session.phase != SessionPhase.WRAP_UP)
    assert rig.now - wrap_started <= 2.5
    assert s.session.state == SessionState.STANDBY
    assert s.sent[-1].arm == RESET.arm
    assert max_step([c.arm for c in s.sent]) <= 2.0 * TICK
    assert ("wrap_up", WrapUpPurpose.CANCEL) not in s.inner.calls  # the blender owned this wrap-up
    assert s.inner.calls.count("end_round") == 1


def test_in_the_session_a_blender_error_keeps_mirroring():
    s = Session()
    rig = s.rig
    rig.active["heart"] = 7
    s.tick(10)
    assert rig.blender.phase is Phase.A

    def broken(*args):
        raise KeyError("boom")

    rig.blender._tick_a = broken
    s.tick(5)
    assert s.session.state == SessionState.MIRRORING and s.session.phase == SessionPhase.RUNNING
    assert s.sent[-1].arm == MIRROR and s.hooks.errors == 1
    assert WaitReason.HOOK_ERROR.value not in str(s.session.snapshot())
