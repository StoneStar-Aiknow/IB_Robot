"""template_motion's pieces around the blender: whitelist, labels, port, config, hand overlay, launch key. No ROS."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from manipulation_execution.mirror_session.hand_overlay import (
    HAND_HIGH,
    HAND_LOW,
    HAND_NEUTRAL,
    HandOverlay,
    travel_time_s,
)
from manipulation_execution.template_motion.config import ARM_LIMITS, load_config, parse_config
from manipulation_execution.template_motion.labels import LabelBook
from manipulation_execution.template_motion.port import CbfMonitor, PlayClient, PortMonitor
from manipulation_execution.template_motion.registry import SCHEMA, check_segment, parse_segments

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "src" / "robot_config" / "config" / "motion_template"
HAND = (HAND_LOW, HAND_HIGH)


def entry(**changes):
    raw = {
        "P": [0.1] * 14,
        "T": [0.3] * 14,
        "hand_P": [0.2] * 20,
        "hand_T": [0.3] * 20,
        "t_in": 1.0,
        "t_out": 1.0,
        "t_out_stop": 0.5,
        "hold_max_s": 4.0,
        "start_tol_rad": 0.03,
        "hand_start_tol": 0.05,
        "report_sha256": "a" * 64,
    }
    raw.update(changes)
    return raw


# -- registry ---------------------------------------------------------------------------


def test_a_good_entry_loads():
    segment = check_segment("heart", entry())
    assert segment.P == (0.1,) * 14 and segment.t_out_stop == 0.5


@pytest.mark.parametrize(
    "changes, words",
    [
        ({"t_out_stop": 0.29}, "t_out_stop"),  # M0 revision 2: [0.3, 0.6]
        ({"t_out_stop": 0.61}, "t_out_stop"),
        ({"t_in": 0.2}, "t_in"),
        ({"hold_max_s": 0.0}, "hold_max_s"),
        ({"P": [0.1] * 13}, "P must list 14"),
        ({"hand_T": [float("nan")] * 20}, "hand_T must be finite"),
        ({"report_sha256": "A" * 64}, "report_sha256"),
        ({"T": [1.5] * 14, "t_out_stop": 0.5}, "exceeds 2 rad/s"),  # 1.875 * 1.4 / 0.5
    ],
)
def test_a_bad_entry_names_the_field(changes, words):
    with pytest.raises(ValueError, match=words):
        check_segment("x", entry(**changes))


def test_limits_are_checked_when_given():
    with pytest.raises(ValueError, match=r"P\[3\]"):
        check_segment("x", entry(), arm_limits=ARM_LIMITS)  # elbow 0.1 > 0 - 1 degree
    with pytest.raises(ValueError, match="hand range"):
        check_segment("x", entry(hand_P=[9.0] * 20), hand_limits=HAND)


def test_a_document_keeps_the_good_entries_and_names_the_bad():
    loaded, rejected = parse_segments({"schema": SCHEMA, "segments": {"a": entry(), "b": entry(t_in=5.0)}})
    assert list(loaded) == ["a"] and "t_in" in rejected["b"]
    with pytest.raises(ValueError):
        parse_segments({"schema": "other", "segments": {}})


def test_arm_limits_are_the_cbf_less_one_degree():
    assert len(ARM_LIMITS) == 14
    assert ARM_LIMITS[3] == pytest.approx((-2.3562 + np.radians(1.0), -np.radians(1.0)))


# -- labels -----------------------------------------------------------------------------


def label(template="heart", active=True, actor=7, valid=True, version=1):
    return json.dumps(
        {
            "schema_version": version,
            "template_name": template,
            "actor_id": actor,
            "active": active,
            "input_valid": valid,
        }
    )


def test_episodes_count_activations():
    book = LabelBook(["heart"], stale_s=0.5)
    assert book.current("heart", 0.0) is None
    book.observe(label(), 1.0)
    assert book.current("heart", 1.1).episode == 1
    book.observe(label(), 1.2)
    assert book.current("heart", 1.2).episode == 1
    book.observe(label(active=False), 1.3)
    assert book.current("heart", 1.3) is None
    book.observe(label(), 1.4)
    assert book.current("heart", 1.4).episode == 2
    assert book.current("heart", 1.95) is None  # stale
    book.observe(label(), 2.0)
    assert book.current("heart", 2.0).episode == 3  # stale counts as inactive
    assert book.current("heart", 2.0).actor_id == 7


def test_invalid_input_unknown_actor_and_bad_messages_are_inactive():
    book = LabelBook(["heart"])
    book.observe(label(valid=False), 1.0)
    assert book.current("heart", 1.0) is None
    book.observe(label(actor=-1), 1.0)
    assert book.current("heart", 1.0) is None
    for bad in ("{", label(version=2), label(template="holy"), json.dumps({"template_name": "heart"})):
        book.observe(bad, 1.0)
    assert book.malformed == 4


# -- port -------------------------------------------------------------------------------


def status(**fields):
    return json.dumps({"schema": "x2_template_status/v1", "state": "IDLE", **fields})


def test_port_monitor_keeps_the_newest_well_formed_status():
    port = PortMonitor()
    port.observe(status(state="HOLD", seg_id=3, P_actual=[0.1] * 14, hand_P=[0.2] * 19), 1.0)
    latest = port.latest()
    assert latest.state == "HOLD" and latest.seg_id == 3 and latest.serial == 1
    assert latest.P_actual == (0.1,) * 14 and latest.hand_P is None
    for bad in ("x", status(state="PLAY"), status(seg_id=-1), json.dumps({"state": "IDLE"})):
        port.observe(bad, 2.0)
    assert port.malformed == 4 and port.latest() is latest


def test_cbf_monitor():
    cbf = CbfMonitor()
    cbf.observe(json.dumps({"mode": "x"}), 1.0)
    assert not cbf.latest().intervening
    cbf.observe(json.dumps({"cbf": {"stuck_s": 0.0, "releasing": False, "stuck_collision": False}}), 1.0)
    assert not cbf.latest().intervening
    for section in ({"stuck_s": 0.1}, {"releasing": True}, {"stuck_collision": True}):
        cbf.observe(json.dumps({"cbf": section}), 2.0)
        assert cbf.latest().intervening
    cbf.observe("[", 3.0)
    assert cbf.malformed == 1 and cbf.latest().received_at == 2.0


class Client:
    def __init__(self, **kwargs):
        self.port = PortMonitor()
        self.messages = []
        self.client = PlayClient(self.messages.append, self.port, first_seg_id=41, **kwargs)

    def ops(self, op):
        return [m for m in map(json.loads, self.messages) if m["op"] == op]


def test_the_keepalive_runs_every_tick_until_a_terminal_status_also_after_a_release():
    c = Client()
    seg_id = c.client.start("heart", 0.0)
    assert seg_id == 41 and c.client.busy
    with pytest.raises(RuntimeError):
        c.client.start("heart", 0.0)
    now = 0.0
    for _ in range(50):  # 50 Hz with a little jitter: never a gap over 50 ms
        now += 0.0195
        c.client.pump(now)
    assert len(c.ops("keepalive")) == 50
    c.client.release(seg_id, fast=False)
    c.client.release(seg_id + 1, fast=True)  # not ours: nothing sent
    assert c.ops("release") == [{"op": "release", "seg_id": 41, "fast": False, "schema": "x2_template_play/v1"}]
    c.port.observe(status(state="EXIT_B", seg_id=41), now)
    now += 0.02
    c.client.pump(now)
    assert len(c.ops("keepalive")) == 51
    c.port.observe(status(state="DONE", seg_id=40), now)  # another segment's DONE
    now += 0.02
    c.client.pump(now)
    assert c.client.busy
    c.port.observe(status(state="DONE", seg_id=41), now)
    now += 0.02
    c.client.pump(now)
    assert not c.client.busy and len(c.ops("keepalive")) == 52
    assert c.client.start("heart", now) == 42


def test_the_keepalive_gives_up_without_any_status_and_abandon_stops_it():
    c = Client(give_up_s=1.0)
    c.client.start("heart", 0.0)
    for k in range(1, 60):
        c.client.pump(k * 0.02)
    assert not c.client.busy  # 1.0 s without a status
    c.client.start("heart", 2.0)
    c.client.abandon()
    c.client.pump(2.1)
    assert not c.client.busy


def test_only_the_segments_own_statuses_keep_the_keepalive_from_giving_up():
    c = Client(give_up_s=1.0)
    c.client.start("heart", 0.0)
    for k in range(1, 60):  # a restarted port: IDLE without a segment, every 100 ms
        if k % 5 == 0:
            c.port.observe(status(state="IDLE", seg_id=None), k * 0.02)
        c.client.pump(k * 0.02)
    assert not c.client.busy


def test_wind_down_releases_fast_and_ends_the_keepalive_in_time():
    c = Client()
    seg_id = c.client.start("heart", 0.0)
    c.client.wind_down(0.1, 0.5)
    assert c.ops("release") == [{"op": "release", "seg_id": seg_id, "fast": True, "schema": "x2_template_play/v1"}]
    now = 0.1
    while now < 0.58:
        now += 0.02
        c.port.observe(status(state="EXIT_B", seg_id=seg_id), now)  # the port plays on
        c.client.pump(now)
        assert c.client.busy
    c.client.pump(0.62)
    assert not c.client.busy
    c.client.wind_down(0.7, 0.5)  # nothing kept alive: nothing sent
    assert len(c.ops("release")) == 1


def test_a_failing_publisher_is_counted_not_raised():
    port = PortMonitor()

    def broken(message):
        raise OSError("gone")

    client = PlayClient(broken, port)
    client.start("heart", 0.0)
    client.pump(0.05)
    assert client.publish_errors == 2


# -- config -----------------------------------------------------------------------------


def shipped():
    return yaml.safe_load((CONFIG_DIR / "motion_template.yaml").read_text())


def test_the_shipped_config_is_off():
    assert parse_config(shipped(), str(CONFIG_DIR), HAND) is None
    assert load_config("", HAND) is None
    assert parse_config(None, str(CONFIG_DIR)) is None


def test_the_shipped_config_switched_on_loads_every_template():
    raw = shipped()
    raw["motion_template"]["enabled"] = True
    for entry_ in raw["motion_template"]["templates"].values():
        entry_["enabled"] = True
    setup = parse_config(raw, str(CONFIG_DIR), HAND)
    assert setup.notes == []
    assert setup.templates == {
        "overhead_heart": ("overhead_heart",),
        "heart": ("heart",),
        "holy": ("holy",),
        "xsign": ("xsign_left_front", "xsign_right_front"),
    }
    assert setup.keepalive_period_s == 0.02 and setup.blend.c_peak_accel == 6.0
    assert {"heart@50", "heart@75"} <= set(setup.segments)


def test_config_errors_and_left_out_templates(tmp_path):
    heart = json.loads((CONFIG_DIR / "segments.json").read_text())["segments"]["heart"]
    (tmp_path / "segments.json").write_text(
        json.dumps({"schema": SCHEMA, "segments": {"heart": heart, "bad": entry(t_in=9.0)}})
    )

    def config(**changes):
        raw = {
            "enabled": True,
            "segments_file": "segments.json",
            "label_topic": "/l",
            "play_topic": "/p",
            "status_topic": "/s",
            "dataplane_status_topic": "/d",
            "templates": {"heart": {"enabled": True, "segments": ["heart"]}},
        }
        raw.update(changes)
        return {"motion_template": raw}

    with pytest.raises(ValueError, match="unknown"):
        parse_config(config(extra=1), str(tmp_path))
    with pytest.raises(ValueError, match="at most 0.05"):
        parse_config(config(keepalive_period_s=0.06), str(tmp_path))
    with pytest.raises(ValueError, match="absolute"):
        parse_config(config(play_topic="p"), str(tmp_path))
    with pytest.raises(ValueError, match="a_min_s"):
        parse_config(config(blend={"a_min_s": -1}), str(tmp_path))
    setup = parse_config(
        config(
            templates={"heart": {"enabled": True, "segments": ["heart"]}, "x": {"enabled": True, "segments": ["bad"]}}
        ),
        str(tmp_path),
    )
    assert setup.templates == {"heart": ("heart",)}
    assert any("bad" in note and "t_in" in note for note in setup.notes)
    assert any(note.startswith("template x left out") for note in setup.notes)


def test_make_segments_reproduces_the_shipped_whitelist(tmp_path):
    poses = Path("/mnt/oee/X2-wt/motion-template-sim/poses")
    if not poses.is_dir():
        pytest.skip("the pose panel's files are not on this machine")
    out = tmp_path / "segments.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/x2_motion_template/make_segments.py"),
            "--poses",
            str(poses),
            "--out",
            str(out),
            "--scale",
            "0.5",
            "0.75",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text()) == json.loads((CONFIG_DIR / "segments.json").read_text())


# -- hand overlay -----------------------------------------------------------------------


def test_a_template_target_replaces_the_mirror_and_neutral_goals():
    overlay = HandOverlay(50.0, neutral=list(HAND_NEUTRAL))
    overlay.apply(False, 0.0)
    assert "template" not in overlay.counts  # templates off: the status is the baseline's
    goal = list((HAND_LOW + HAND_HIGH) / 2.0)
    out = None
    for k in range(200):
        out = overlay.apply(False, k * 0.02, target=goal)
    assert out == pytest.approx(goal)
    assert overlay.counts["template"] == 200
    out = overlay.apply(False, 4.0, target=[0.0] * 19)  # malformed: ignored, neutral again
    assert overlay.counts["template"] == 200


def test_a_template_target_holds_also_while_the_hand_feedback_is_lost():
    overlay = HandOverlay(50.0, neutral=list(HAND_NEUTRAL))
    goal = list((HAND_LOW + HAND_HIGH) / 2.0)
    overlay.observe_state([0.1] * 10, [0.1] * 10, 0.0)
    out = None
    for k in range(300):  # the state stops after t = 0
        out = overlay.apply(True, k * 0.02, target=goal)
    assert overlay.snapshot(6.0)["state_lost"]
    assert out == pytest.approx(goal)  # not drifting to neutral


def test_travel_time_is_the_slowest_channel():
    start = np.zeros(20)
    assert travel_time_s(start, start) == 0.0
    goal = start.copy()
    goal[4] = 0.5
    assert travel_time_s(start, goal) > 0.0
    assert np.isfinite(travel_time_s(HAND_LOW, HAND_HIGH))


# -- launch -----------------------------------------------------------------------------


def test_the_launch_key_reaches_the_node_and_its_absence_changes_nothing(monkeypatch):
    pytest.importorskip("launch_ros")
    from test_mirror_session_launch_builder import _config, _node_parameters

    from manipulation_execution.launch_builders import mirror_session as builder

    monkeypatch.setattr(builder, "resolve_ros_path", lambda path: f"/resolved/{path}")
    without = _node_parameters(builder.generate_mirror_session_nodes(_config())[0])
    assert "motion_template_config" not in without
    with_key = _node_parameters(builder.generate_mirror_session_nodes(_config(motion_template_config="t.yaml"))[0])
    assert with_key.pop("motion_template_config") == "/resolved/t.yaml"
    assert with_key == without
