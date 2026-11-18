"""The data plane node with an enforcing CBF, against the vendor mock: what it reports and logs.

The filter's decisions are covered by aimdk_dataplane's C++ tests. These cover
the node around it: the status fields the runtime reads, the INFO line, and the
CBF_RELEASE log when the right hand drops out mid-mirror and comes back.
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
def node(monkeypatch):
    logged = Logged()

    def start_process(profile):
        executable = Path(get_package_prefix("aimdk_dataplane")) / "lib" / "aimdk_dataplane" / "aimdk_dataplane"
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="aimdk_dataplane_", delete=False) as parameters:
            json.dump(
                {
                    "aimdk_dataplane": {
                        "ros__parameters": {
                            "plan_json": json.dumps(dataplane.plan(profile)),
                            "cbf_safety_json": json.dumps(CBF),
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


def stream(harness, stop: threading.Event) -> threading.Thread:
    """The arm target a mirror session would send, at 50 Hz until ``stop``."""
    publisher = harness.node.create_publisher(Float64MultiArray, "/aimdk/arm/commands", 10)
    command = Float64MultiArray()
    command.data = [0.0] * 14

    def run():
        while not stop.is_set():
            publisher.publish(command)
            time.sleep(0.02)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=(\S+)", line))


def test_a_hand_dropping_out_mid_mirror_is_released_logged_and_recovers(node):
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

        # The right hand drops out: a non-verdict hold, released after hold_release_after_s, logged each second.
        harness.mock.set_parameters([rclpy.parameter.Parameter("right_hand_type", value="none")])
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
    finally:
        stop.set()
        thread.join(timeout=1.0)
