"""The audio.speech readiness contract the launcher depends on.

``robot.launch.py`` turns a logical binding into ``--require-interfaces``, and
``wait_for_runtime`` only admits a consumer when the runtime's own
``RuntimeStatus`` reports that publish interface as ``ready``. These tests build
the status the way ``AimdkRuntimeNode`` builds it — the real profile description
plus a real ``InterfaceMonitor`` — so a regression that leaves ``states`` empty
fails here instead of as a 60-second launch timeout on the robot.

The first two tests drive the monitor through its public ``observe`` entry point
and are deterministic. The last one proves the same path over a real ROS graph:
the runtime's availability heartbeat must actually reach a subscriber.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from aimdk_robot.runtime_node import OBSERVED_INTERFACES  # noqa: E402
from builtin_interfaces.msg import Time  # noqa: E402

from ibrobot_msgs.msg import SpeechAudioChunk  # noqa: E402
from robot_runtime.contract import LIFECYCLE_ACTIVE  # noqa: E402
from robot_runtime.interface_description import build_description  # noqa: E402
from robot_runtime.interface_monitor import InterfaceMonitor  # noqa: E402
from robot_runtime.modes import ModeModel, ModeModelConfig  # noqa: E402
from robot_runtime.profile import load_profile  # noqa: E402
from robot_runtime.state import RuntimeState  # noqa: E402
from robot_runtime.wait_for_runtime import validate_runtime_snapshot  # noqa: E402

PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "x2_ultra.yaml"
SPEECH = "audio.speech"


class FakeNode:
    """Records subscriptions; the monitor is driven by observe() in these tests."""

    def __init__(self):
        self.subscriptions = []

    def create_subscription(self, msg_type, topic, callback, qos, **kwargs):
        self.subscriptions.append({"type": msg_type, "topic": topic, "qos": qos, "kwargs": kwargs})
        return self.subscriptions[-1]


@pytest.fixture(scope="module")
def profile():
    return load_profile(PROFILE)


def _description_and_observed(profile):
    """The published description, and the filtered copy the runtime's monitor gets."""
    description = build_description(profile)
    observed = {k: v for k, v in description["interfaces"].items() if k in OBSERVED_INTERFACES}
    return description, observed


def _status(profile, description, monitor):
    """The RuntimeStatus message AimdkRuntimeNode publishes for this description."""
    state = RuntimeState(
        name=str(profile["runtime"]["name"]),
        version=str(profile["runtime"]["version"]),
        capabilities=profile["capabilities"],
        modes=ModeModel(ModeModelConfig.from_profile(profile)),
        interface_description=description,
        interface_states=monitor.states,
    )
    state.set_lifecycle(LIFECYCLE_ACTIVE)
    return state.to_msg(Time(sec=1, nanosec=0))


def _reported_state(status):
    return json.loads(status.interface_description_json)["states"][SPEECH]["state"]


def _heartbeat():
    """What _publish_speech_heartbeat sends: availability, with no PCM."""
    chunk = SpeechAudioChunk()
    chunk.header.stamp = Time(sec=1, nanosec=0)
    chunk.header.frame_id = "enhanced_audio"
    chunk.source_id = "test-source"
    chunk.sequence = 1
    chunk.state = 0
    return chunk


def test_observed_interfaces_stay_a_narrow_allowlist():
    """Observing every declared stream would re-subscribe the sensor streams."""
    assert OBSERVED_INTERFACES == (SPEECH,)


def test_unknown_then_ready_through_the_real_runtime_status(profile):
    description, observed = _description_and_observed(profile)
    node = FakeNode()
    monitor = InterfaceMonitor(node, {"interfaces": observed})
    # The monitor must watch only the speech endpoint, not the whole profile.
    assert [entry["topic"] for entry in node.subscriptions] == [description["interfaces"][SPEECH]["endpoint"]]

    kwargs = {
        "runtime_name": str(profile["runtime"]["name"]),
        "required": ("interaction.audio_capture",),
        "interfaces": (SPEECH,),
    }

    # Silence before the first heartbeat is the normal idle state: the vendor
    # stream is wake-word gated. Reporting it as readiness is exactly the bug
    # that produced "interface 'audio.speech' is unknown: no sample".
    assert _reported_state(_status(profile, description, monitor)) == "unknown"
    with pytest.raises(ValueError, match=f"{SPEECH}.*unknown"):
        validate_runtime_snapshot(_status(profile, description, monitor), **kwargs)

    monitor.observe(SPEECH, description["interfaces"][SPEECH], _heartbeat())
    status = _status(profile, description, monitor)
    assert _reported_state(status) == "ready"
    validate_runtime_snapshot(status, **kwargs)


def test_a_dead_bridge_stops_admitting_consumers(profile, monkeypatch):
    description, observed = _description_and_observed(profile)
    monitor = InterfaceMonitor(FakeNode(), {"interfaces": observed})
    now = [0.0]
    monkeypatch.setattr("robot_runtime.interface_monitor.time.monotonic", lambda: now[0])
    monitor.observe(SPEECH, description["interfaces"][SPEECH], _heartbeat())

    kwargs = {"runtime_name": str(profile["runtime"]["name"]), "interfaces": (SPEECH,)}
    validate_runtime_snapshot(_status(profile, description, monitor), **kwargs)

    # Holding the last "ready" forever would admit every later consumer against
    # a silent runtime.
    now[0] += 2.5
    status = _status(profile, description, monitor)
    assert _reported_state(status) == "stale"
    with pytest.raises(ValueError, match=f"{SPEECH}.*stale"):
        validate_runtime_snapshot(status, **kwargs)


def test_heartbeat_reaches_a_real_subscriber(profile):
    """The heartbeat is the readiness signal, so it has to survive the transport."""
    rclpy = pytest.importorskip("rclpy")
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.executors import SingleThreadedExecutor

    description, observed = _description_and_observed(profile)
    endpoint = description["interfaces"][SPEECH]["endpoint"]
    rclpy.init()
    node = rclpy.create_node("speech_readiness_probe")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        monitor = InterfaceMonitor(node, {"interfaces": observed}, callback_group=MutuallyExclusiveCallbackGroup())
        publisher = node.create_publisher(SpeechAudioChunk, endpoint, 10)
        deadline = time.monotonic() + 10.0
        while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
        assert publisher.get_subscription_count() == 1, "the readiness monitor never subscribed"

        for _ in range(200):
            publisher.publish(_heartbeat())
            executor.spin_once(timeout_sec=0.02)
            if monitor.states()[SPEECH]["state"] == "ready":
                break
        assert monitor.states()[SPEECH]["state"] == "ready"
    finally:
        executor.remove_node(node)
        node.destroy_node()
        executor.shutdown(timeout_sec=0)
        if rclpy.ok():
            rclpy.shutdown()
