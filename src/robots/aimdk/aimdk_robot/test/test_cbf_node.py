"""The data plane node with an enforcing CBF, against the vendor mock: what it reports and logs.

The filter's decisions are covered by aimdk_dataplane's C++ tests. These cover
the node around it: the status fields the runtime reads, the INFO line, the
CBF_RELEASE log when the right hand is swapped for one the geometry was not
built for mid-mirror and comes back, and an empty right flange checked as a
phantom hand instead of released, its thumb at the compact pose and its start
and end logged once even when its type flickers; and the safety epoch the
template layer checks, which the control heartbeat does not move; and the
tick timing figures in the status, and their absence when switched off; and
a log output that stops taking lines, which holds up neither the commands nor
the status.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest
import rclpy
from ament_index_python.packages import get_package_prefix
from std_msgs.msg import Float64MultiArray, String

from aimdk_robot import dataplane

from test_runtime_bridge import OMNIHAND_PROFILE, Harness, ros_context  # noqa: F401  (the module's ROS context)

# The candidate mapping is an identity here; the mock's geometry is not the robot's.
# A phantom right thumb's default pose with it: model roll upper, abad lower end.
COMPACT_RIGHT_THUMB = [0.69813170, -1.464235482612666]
CBF = {
    "enabled": True,
    "mode": "enforce",
    "protected_mirror_mode": "policy_stream",
    "deadline_us": 10000.0,
    "thumb_mapping": {
        "verified": True,
        "command_scale": [1, 1, 1, 1],
        "command_offset": [0, 0, 0, 0],
        "feedback_scale": [1, 1, 1, 1],
        "feedback_offset": [0, 0, 0, 0],
    },
}
NODE_PARAMETERS = ("tick_timing",)


class Logged:
    """The node's log lines, read from its stderr pipe as they come; the reading can pause."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._lock = threading.Lock()
        self._reading = threading.Event()
        self._reading.set()
        self.read_fd, self.write_fd = os.pipe()  # the write end is the node's stderr

    def follow(self) -> None:
        def read():
            pending = b""
            while True:
                self._reading.wait()
                data = os.read(self.read_fd, 65536)
                if not data:
                    break
                *complete, pending = (pending + data).split(b"\n")
                with self._lock:
                    self.lines.extend(line.decode(errors="replace").rstrip() for line in complete)
            os.close(self.read_fd)

        threading.Thread(target=read, daemon=True).start()

    def close(self) -> None:
        """After the node is gone: the reader sees the end of the pipe."""
        os.close(self.write_fd)

    def stall(self) -> None:
        """Stop reading and fill the pipe: the node's next write to stderr blocks."""
        self._reading.clear()
        time.sleep(0.1)  # a read in progress takes one more chunk, then waits
        # A description of its own, so non-blocking here leaves the node's writes blocking.
        writer = os.open(f"/proc/self/fd/{self.write_fd}", os.O_WRONLY | os.O_NONBLOCK)
        try:
            for size in (4096, 1):  # 4096 is all or nothing on a pipe; then the last bytes
                try:
                    while True:
                        os.write(writer, b"\n" * size)
                except BlockingIOError:
                    pass
        finally:
            os.close(writer)
        assert self.full()

    def full(self) -> bool:
        writer = os.open(f"/proc/self/fd/{self.write_fd}", os.O_WRONLY | os.O_NONBLOCK)
        try:
            os.write(writer, b"\n")
            return False
        except BlockingIOError:
            return True
        finally:
            os.close(writer)

    def resume(self) -> None:
        self._reading.set()

    def matching(self, text: str) -> list[str]:
        with self._lock:
            return [line[line.index(text) :] for line in self.lines if text in line]


@pytest.fixture
def node(monkeypatch, request):
    logged = Logged()
    # Indirect parametrization overrides keys of the CBF section, and the node's own parameters.
    overrides = dict(getattr(request, "param", {}))
    node_parameters = {key: overrides.pop(key) for key in NODE_PARAMETERS if key in overrides}
    cbf_config = {**CBF, **overrides}

    def start_process(profile):
        executable = Path(get_package_prefix("aimdk_dataplane")) / "lib" / "aimdk_dataplane" / "aimdk_dataplane"
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="aimdk_dataplane_", delete=False) as parameters:
            json.dump(
                {
                    "aimdk_dataplane": {
                        "ros__parameters": {
                            "plan_json": json.dumps(dataplane.plan(profile)),
                            "cbf_safety_json": json.dumps(cbf_config),
                            **node_parameters,
                        }
                    }
                },
                parameters,
            )
        process = subprocess.Popen(
            [str(executable), "--ros-args", "--params-file", parameters.name],
            stdout=subprocess.DEVNULL,
            stderr=logged.write_fd,
        )
        logged.follow()
        return process

    monkeypatch.setattr(dataplane, "start_process", start_process)
    harness = Harness(profile=OMNIHAND_PROFILE, hand_type="dexterous", initial_action="UPPERBODY_REMOTE_SPLIT")
    try:
        yield harness, logged
    finally:
        logged.resume()
        harness.shutdown()
        logged.close()


def stream(harness, stop: threading.Event, topic="/aimdk/arm/commands", data=None) -> threading.Thread:
    """The arm (or other) target a mirror session would send, at 50 Hz until ``stop``."""
    publisher = harness.node.create_publisher(Float64MultiArray, topic, 10)
    command = Float64MultiArray()
    command.data = [0.0] * 14 if data is None else [float(value) for value in data]

    def run():
        while not stop.is_set():
            publisher.publish(command)
            time.sleep(0.02)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=(\S+)", line))


def test_a_hand_swapped_mid_mirror_is_released_logged_and_recovers(node):
    harness, logged = node
    harness.wait_active()
    cbf = harness.wait_for(lambda: harness.runtime._dataplane.cbf or None, "a status with the CBF report")
    assert cbf["mode"] == "enforce" and cbf["mirror_mode"] == "policy_stream"
    assert cbf["hand_configuration_matches"] is True and "max_after_check_us" in cbf

    assert harness.set_mode("policy_stream").success
    stop = threading.Event()
    thread = stream(harness, stop)
    try:
        # A verified frame carries its distance into the next INFO line.
        verified = harness.wait_for(
            lambda: [
                f
                for f in map(fields, logged.matching("CBF status:"))
                if f["mirror_active"] == "true" and f["min_distance_m"] != "na"
            ],
            "an INFO line with a distance",
        )[-1]
        assert int(verified["distance_frames"]) > 0
        float(verified["min_distance_m"])
        assert not logged.matching("CBF_RELEASE")

        # The right hand is swapped for one the geometry was not built for: a
        # non-verdict hold, released after hold_release_after_s, logged each second.
        harness.mock.set_parameters([rclpy.parameter.Parameter("right_hand_type", value="lite_s")])
        start = harness.wait_for(lambda: logged.matching("CBF_RELEASE start"), "the release to start", timeout_s=5.0)[0]
        assert float(fields(start)["held_s"]) >= 0.2
        harness.wait_for(lambda: len(logged.matching("CBF_RELEASE ongoing")) >= 2, "ongoing lines", timeout_s=5.0)
        released = harness.wait_for(
            lambda: [f for f in map(fields, logged.matching("CBF status:")) if int(f["released"]) > 0],
            "the INFO line to count released frames",
        )[-1]
        assert int(released["no_distance_frames"]) > 0

        # Wait for it: a status that has not reported the CBF yet reads as an empty section.
        harness.wait_for(
            lambda: harness.runtime._dataplane.cbf.get("hand_configuration_matches") is False,
            "a status reporting the mismatch",
        )
        # The hand returns: the episode ends and verified frames carry distances again.
        harness.mock.set_parameters([rclpy.parameter.Parameter("right_hand_type", value="dexterous")])
        end = harness.wait_for(lambda: logged.matching("CBF_RELEASE end"), "the release to end", timeout_s=5.0)[0]
        assert fields(end)["episode"] == fields(start)["episode"]
        ended_at = len(logged.matching("CBF status:"))
        harness.wait_for(
            lambda: [f for f in map(fields, logged.matching("CBF status:")[ended_at:]) if f["min_distance_m"] != "na"],
            "a distance after the release",
        )
        assert len(logged.matching("CBF_RELEASE start")) == 1
        # Another hand type is no empty flange.
        assert not logged.matching("CBF_HAND_ABSENT")
    finally:
        stop.set()
        thread.join(timeout=1.0)


def cbf_report(harness, predicate, what):
    """The data plane's CBF report once ``predicate`` holds; a timeout names the last report."""
    try:
        return harness.wait_for(
            lambda: (lambda cbf: cbf if cbf and predicate(cbf) else None)(harness.runtime._dataplane.cbf), what
        )
    except AssertionError as error:
        raise AssertionError(f"{error}; last CBF report: {harness.runtime._dataplane.cbf}") from None


def test_an_empty_right_flange_is_a_phantom_hand_not_a_release(node):
    """Two hands configured, the right flange empty: mirroring runs on the left hand's reading."""
    harness, logged = node
    harness.wait_active()
    harness.mock.set_parameters([rclpy.parameter.Parameter("right_hand_type", value="none")])
    cbf = cbf_report(harness, lambda cbf: cbf.get("phantom_hands") == [False, True], "the right hand as a phantom")
    assert cbf["hands_installed"] == [True, True]
    assert cbf["hand_configuration_matches"] is True
    # Logged once when it starts, so a real hand that drops out shows in the field.
    harness.wait_for(lambda: logged.matching("CBF_HAND_ABSENT side=right start"), "the phantom start line")
    assert not logged.matching("CBF_HAND_ABSENT side=left")

    response = harness.set_mode("policy_stream")
    assert response.success, response.message
    stop = threading.Event()
    thread = stream(harness, stop)
    try:
        verified = harness.wait_for(
            lambda: [
                f
                for f in map(fields, logged.matching("CBF status:"))
                if f["mirror_active"] == "true" and f["min_distance_m"] != "na"
            ],
            "an INFO line with a distance",
        )[-1]
        assert int(verified["distance_frames"]) > 0
        feedback_frames = harness.runtime._dataplane.cbf["thumb_feedback_frames"]

        # Well past hold_release_after_s: nothing was held for the empty flange,
        # and the missing right thumb is no feedback problem.
        time.sleep(1.5)
        cbf = harness.runtime._dataplane.cbf
        assert cbf["hand_configuration_matches"] is True
        assert cbf["released"] == 0 and cbf["release_episodes"] == 0
        assert cbf["thumb_feedback_frames"] == feedback_frames
        assert cbf["thumb_feedback_issue"] == ""
        assert not logged.matching("CBF_RELEASE")
        assert harness.status().active_mode == "policy_stream"

        # A left thumb reading inside the model band is used as it is; the
        # phantom right thumb is the compact pose, as its target, its current
        # angle and what is published, whatever the right command. The mock
        # stops echoing commands so the reading differs from the command.
        assert cbf["thumb_target_urdf"][2:] == pytest.approx(COMPACT_RIGHT_THUMB, abs=1e-9)
        assert cbf["thumb_current_urdf"][2:] == pytest.approx(COMPACT_RIGHT_THUMB, abs=1e-9)
        assert list(harness.mock.last_upper_body.hand_pos)[10:12] == pytest.approx(COMPACT_RIGHT_THUMB, abs=1e-6)
        harness.mock.set_parameters([rclpy.parameter.Parameter("follow_commands", value=False)])
        with harness.mock._lock:
            harness.mock._positions["L_thumb_roll_joint"] = 0.5
            harness.mock._positions["L_thumb_abad_joint"] = -0.6
        hand = [0.0] * 20
        hand[0], hand[1], hand[10], hand[11] = 0.9, -0.2, 0.3, -0.4
        stream(harness, stop, "/aimdk/hand/joint_commands", hand)
        cbf = cbf_report(
            harness,
            lambda cbf: cbf.get("thumb_target_urdf")
            and cbf["thumb_target_urdf"][:2] == pytest.approx([0.9, -0.2], abs=1e-9)
            and cbf["thumb_current_urdf"][:2] == pytest.approx([0.5, -0.6], abs=1e-9),
            "the left thumb command and reading in the CBF report",
        )
        assert cbf["released"] == 0
        assert cbf["thumb_target_urdf"][2:] == pytest.approx(COMPACT_RIGHT_THUMB, abs=1e-9)
        assert cbf["thumb_current_urdf"][2:] == pytest.approx(COMPACT_RIGHT_THUMB, abs=1e-9)
        assert cbf["thumb_feedback_issue"] == ""
        hand_pos = harness.wait_for(
            lambda: (lambda pos: pos if pos[:2] == pytest.approx([0.9, -0.2], abs=1e-6) else None)(
                list(harness.mock.last_upper_body.hand_pos)
            ),
            "the left thumb command published",
        )
        assert hand_pos[10:12] == pytest.approx(COMPACT_RIGHT_THUMB, abs=1e-6)
    finally:
        stop.set()
        thread.join(timeout=1.0)

    # A type flickering between 0 and 1 (each state 0.2 s, the hand state at
    # 50 Hz) stays one episode: no end until reported for 0.5 s, no new start.
    for _ in range(3):
        harness.mock.set_parameters([rclpy.parameter.Parameter("right_hand_type", value="dexterous")])
        time.sleep(0.2)
        harness.mock.set_parameters([rclpy.parameter.Parameter("right_hand_type", value="none")])
        time.sleep(0.2)
    assert len(logged.matching("CBF_HAND_ABSENT side=right start")) == 1
    assert not logged.matching("CBF_HAND_ABSENT side=right end")

    # The hand is fitted: the phantom ends, logged once, and the start was logged once.
    harness.mock.set_parameters([rclpy.parameter.Parameter("right_hand_type", value="dexterous")])
    cbf_report(harness, lambda cbf: cbf.get("phantom_hands") == [False, False], "no phantom hand")
    harness.wait_for(lambda: logged.matching("CBF_HAND_ABSENT side=right end"), "the phantom end line")
    assert len(logged.matching("CBF_HAND_ABSENT side=right start")) == 1
    assert len(logged.matching("CBF_HAND_ABSENT side=right end")) == 1
    assert not logged.matching("CBF_HAND_ABSENT side=left")


def test_the_status_carries_a_safety_epoch_the_heartbeat_does_not_move(node):
    """safety_epoch, top level: steady through mirroring while control_seq climbs, new on revocation."""
    harness, _logged = node
    harness.wait_active()
    seen: list[dict] = []
    lock = threading.Lock()

    def on_status(message):
        data = json.loads(message.data)
        with lock:
            seen.append(data)

    harness.node.create_subscription(String, dataplane.STATUS_TOPIC, on_status, 20)

    def since(start):
        with lock:
            return list(seen[start:])

    harness.wait_for(lambda: since(0), "a status")
    assert all(isinstance(status["safety_epoch"], int) and status["safety_epoch"] >= 1 for status in since(0))

    assert harness.set_mode("policy_stream").success
    stop = threading.Event()
    thread = stream(harness, stop)
    try:
        harness.wait_for(
            lambda: [s for s in since(0) if (s.get("cbf") or {}).get("mirror_active")], "mirroring in the status"
        )
        with lock:
            start = len(seen)
        time.sleep(1.5)  # 15 heartbeats of the runtime's control
        steady = [s for s in since(start) if (s.get("cbf") or {}).get("mirror_active")]
        assert len(steady) >= 20
        assert steady[-1]["control_seq"] - steady[0]["control_seq"] >= 5
        assert {s["safety_epoch"] for s in steady} == {steady[0]["safety_epoch"]}
        mirroring_epoch = steady[0]["safety_epoch"]
    finally:
        stop.set()
        thread.join(timeout=1.0)

    # Leaving the mode revokes the stream: the epoch moves and does not come back.
    with lock:
        start = len(seen)
    assert harness.set_mode("stream").success
    after = harness.wait_for(
        lambda: [s for s in since(start) if s["safety_epoch"] != mirroring_epoch], "a new safety epoch"
    )
    assert after[0]["safety_epoch"] > mirroring_epoch


class Statuses:
    """The data plane's status messages as they come, parsed."""

    def __init__(self, harness) -> None:
        self.seen: list[dict] = []
        self._lock = threading.Lock()
        harness.node.create_subscription(String, dataplane.STATUS_TOPIC, self._on_status, 20)

    def _on_status(self, message) -> None:
        data = json.loads(message.data)
        with self._lock:
            self.seen.append(data)

    def since(self, start: int) -> list[dict]:
        with self._lock:
            return list(self.seen[start:])

    def count(self) -> int:
        with self._lock:
            return len(self.seen)


def mirroring(statuses: list[dict]) -> list[dict]:
    return [s for s in statuses if (s.get("cbf") or {}).get("mirror_active")]


def test_the_status_accounts_for_every_command_tick(node):
    """tick_timing: counters in every status, histograms once a second, each count once."""
    harness, logged = node
    harness.wait_active()
    statuses = Statuses(harness)
    assert harness.set_mode("policy_stream").success
    stop = threading.Event()
    thread = stream(harness, stop)
    try:
        harness.wait_for(lambda: mirroring(statuses.since(0)), "mirroring in the status")
        start = statuses.count()
        time.sleep(2.0)
        steady = mirroring(statuses.since(start))
    finally:
        stop.set()
        thread.join(timeout=1.0)

    assert len(steady) >= 30
    timing = [s["tick_timing"] for s in steady]
    invocations = [t["invocations"] for t in timing]
    assert invocations == sorted(invocations)
    # The command timer at 50 Hz, 2 s: counted whether or not the CBF ran.
    assert invocations[-1] - invocations[0] >= 60
    assert timing[-1]["period_ms"] == pytest.approx(20.0)
    assert timing[-1]["paths"]["cbf"] > 0
    assert timing[-1]["outcomes"]["verified"] > 0
    # Each CBF frame with an upper body is published; a withheld one is not.
    outcomes = timing[-1]["outcomes"]
    assert timing[-1]["publishes"] >= timing[-1]["paths"]["cbf"] - outcomes.get("withheld", 0)
    assert timing[-1]["clock_reads"] > 0 and timing[-1]["clock_ns"] > 0
    assert set(timing[-1]["window_max_ms"]) >= {"tick", "filter", "status", "late", "bookkeeping", "tick_runqueue"}
    assert timing[-1]["scheduler"] is True
    assert isinstance(timing[-1]["tick_nice"], int) and timing[-1]["tick_tid"] > 0

    # Histograms about once a second, not in every 20 Hz status.
    with_histograms = [t for t in timing if "histograms" in t]
    assert 1 <= len(with_histograms) <= 4
    for report in with_histograms:
        histograms = report["histograms"]
        assert len(histograms["edges_ms"]) + 1 == len(histograms["tick"]["counts"])
        # Taken under one lock: every tick counted once, and each solve once.
        assert histograms["tick"]["n"] == report["invocations"]
        assert sum(histograms["tick"]["counts"]) == histograms["tick"]["n"]
        solved = report["paths"].get("cbf", 0) + report["paths"].get("discarded", 0)
        assert histograms["filter"]["n"] == solved
        assert histograms["late"]["n"] == report["invocations"] - 1
        # The thread's clocks for every tick: on a CPU, and waiting for one.
        assert histograms["tick_cpu"]["n"] == report["invocations"]
        assert histograms["tick_runqueue"]["n"] == report["invocations"]
        assert histograms["filter_cpu"]["n"] == solved
        assert histograms["status"]["n"] > 0
        # Every CBF frame once by the measured distances, under the same lock.
        by = report["outcome_distances"]
        assert set(by) - {"edges_m"} == set(report["outcomes"]) - {"held_collision", "released_collision"}
        for outcome, counts in by.items():
            if outcome != "edges_m":
                assert sum(counts["min"]) + counts["no_reading"] == report["outcomes"][outcome]
                assert sum(counts["cross_min"]) == sum(counts["min"])

    # The measured state of a verified frame, by pair names.
    readings = [s["cbf"]["input_distance"] for s in steady if s["cbf"]["input_distance"]]
    assert readings
    left, right = readings[-1]["cross_min_pair"].split("/")
    assert {left.split("_")[0], right.split("_")[0]} == {"left", "right"}
    assert readings[-1]["min_distance_m"] <= readings[-1]["cross_min_distance_m"]

    line = harness.wait_for(lambda: logged.matching("CBF status:"), "an INFO line")[-1]
    counters = fields(line)
    for name in ("missed_ticks", "node_deadlines", "deadlines", "publish_overruns"):
        assert counters[name].isdigit(), (name, line)


@pytest.mark.parametrize("node", [{"tick_timing": False}], indirect=True)
def test_without_tick_timing_the_status_has_no_figures(node):
    harness, logged = node
    harness.wait_active()
    statuses = Statuses(harness)
    assert harness.set_mode("policy_stream").success
    stop = threading.Event()
    thread = stream(harness, stop)
    try:
        harness.wait_for(lambda: len(mirroring(statuses.since(0))) >= 20, "mirroring in the status")
    finally:
        stop.set()
        thread.join(timeout=1.0)
    assert not [s for s in statuses.since(0) if "tick_timing" in s]
    line = harness.wait_for(lambda: logged.matching("CBF status:"), "an INFO line")[-1]
    assert fields(line)["missed_ticks"] == "na"


@pytest.mark.parametrize("node", [{"hands_installed": [True, False]}], indirect=True)
def test_a_right_hand_on_a_left_only_configuration_is_refused(node):
    """The reverse of an empty flange: a hand where the geometry has no collision body."""
    harness, _logged = node
    harness.wait_active()
    cbf = cbf_report(harness, lambda cbf: "hand_configuration_matches" in cbf, "a status with the CBF report")
    assert cbf["hands_installed"] == [True, False]
    assert cbf["hand_configuration_matches"] is False
    assert cbf["phantom_hands"] == [False, False]
    refused = harness.set_mode("policy_stream")
    assert not refused.success
    assert "right hand expected 0 (none), platform reports 1 (nimble_hands)" in refused.message
    assert harness.status().active_mode != "policy_stream"

    # Control: the right side reports none.
    harness.mock.set_parameters([rclpy.parameter.Parameter("right_hand_type", value="none")])
    cbf_report(harness, lambda cbf: cbf.get("hand_configuration_matches") is True, "matching hands")
    response = harness.set_mode("policy_stream")
    assert response.success, response.message


def test_a_log_output_that_stops_holds_up_neither_the_commands_nor_the_status(node):
    """Log lines are printed on a thread of their own. With the node's stderr pipe full every
    write to it blocks; the command ticks and the status go on, and the lines come out after."""
    harness, logged = node
    harness.wait_active()
    statuses = Statuses(harness)
    assert harness.set_mode("policy_stream").success
    stop = threading.Event()
    thread = stream(harness, stop)
    try:
        harness.wait_for(lambda: mirroring(statuses.since(0)), "mirroring in the status")
        harness.wait_for(lambda: logged.matching("CBF status:"), "an INFO line")
        logged.stall()
        lines = len(logged.matching("CBF status:"))
        time.sleep(2.5)  # past the next INFO line (every 2 s): it is written into the full pipe
        start, commands = statuses.count(), harness.mock.upper_body_count
        time.sleep(1.0)  # measured after it
        during = mirroring(statuses.since(start))
        sent = harness.mock.upper_body_count - commands
        assert logged.full(), "nothing may drain the pipe meanwhile"
        logged.resume()
        # The line made while the pipe was full was kept, not lost.
        harness.wait_for(lambda: len(logged.matching("CBF status:")) > lines, "the INFO line held back")
    finally:
        logged.resume()
        stop.set()
        thread.join(timeout=1.0)

    # 1 s: 20 statuses at 20 Hz, 50 commands at 50 Hz.
    assert len(during) >= 14, len(during)
    assert sent >= 35, sent
    assert during[-1]["cbf"]["calls"] - during[0]["cbf"]["calls"] >= 25
