"""The data plane node with an enforcing CBF, against the vendor mock: what it reports and logs.

The filter's decisions are covered by aimdk_dataplane's C++ tests. These cover
the node around it: the status fields the runtime reads, the INFO line, the
CBF_RELEASE log when the right hand is swapped for one the geometry was not
built for mid-mirror and comes back, and an empty right flange checked as a
phantom hand instead of released, its thumb at the compact pose and its start
and end logged once even when its type flickers.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest
import rclpy
from ament_index_python.packages import get_package_prefix
from std_msgs.msg import Float64MultiArray

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


class Logged:
    """The node's log lines, read from its stderr as they come."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._lock = threading.Lock()

    def follow(self, stream) -> None:
        def read():
            for line in stream:
                with self._lock:
                    self.lines.append(line.rstrip())

        threading.Thread(target=read, daemon=True).start()

    def matching(self, text: str) -> list[str]:
        with self._lock:
            return [line[line.index(text) :] for line in self.lines if text in line]


@pytest.fixture
def node(monkeypatch, request):
    logged = Logged()
    # Indirect parametrization overrides keys of the CBF section.
    cbf_config = {**CBF, **getattr(request, "param", {})}

    def start_process(profile):
        executable = Path(get_package_prefix("aimdk_dataplane")) / "lib" / "aimdk_dataplane" / "aimdk_dataplane"
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="aimdk_dataplane_", delete=False) as parameters:
            json.dump(
                {
                    "aimdk_dataplane": {
                        "ros__parameters": {
                            "plan_json": json.dumps(dataplane.plan(profile)),
                            "cbf_safety_json": json.dumps(cbf_config),
                        }
                    }
                },
                parameters,
            )
        process = subprocess.Popen(
            [str(executable), "--ros-args", "--params-file", parameters.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        logged.follow(process.stderr)
        return process

    monkeypatch.setattr(dataplane, "start_process", start_process)
    harness = Harness(profile=OMNIHAND_PROFILE, hand_type="dexterous", initial_action="UPPERBODY_REMOTE_SPLIT")
    try:
        yield harness, logged
    finally:
        harness.shutdown()


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
