"""The data plane node with the template playback port on, against the vendor mock.

The player itself (whitelist, gate, trajectory, status rules) is covered by
aimdk_dataplane's test_template_play.cpp. These cover the node around it with
an enforcing CBF: a whole round on the vendor stream, the runtime-level stops
and the stream after them, the refusals, and the port switched off or
misconfigured. The interface is /mnt/oee/X2-wt/template-play-M0-interface.md.

What the mock cannot reach, and what stands in for it:

- step() giving no point while playing (STOPPED internal_error). No whitelisted
  segment produces a non-finite point, and test_template_play.cpp has no case
  for it either. The node takes the same fall-through as a runtime-level stop
  found in the playing tick: stop, then the ordinary path in that same tick.
- A runtime-level stop found inside the playing tick, other than a control
  lapse. Every other exit (a mode switch, a stop, a named claim, a clear)
  changes the safety epoch, and on_control stops the player before the next
  tick, which is then the ordinary path's. The no-pause check below holds
  either way: it is taken on the tick right after the exit, which is the
  playing tick when the epoch is only read there.
- The in-tick named_arm stop. A claim the data plane applies changes the epoch
  (stopped in on_control); one it refuses changes nothing and playback goes on.
- on_control's refusal of a start still waiting for its tick: the wait is at
  most one tick and cannot be lined up from outside. The tick refuses the same
  start with the same reason, not_admitted, and that path is tested.
- A refusal landing between a DONE and its publication: both happen in one
  command tick, microseconds apart. The test shows a refusal and the DONE
  going out together; test_template_play.cpp's
  ARefusalDoesNotHideADoneNotYetPublished covers the overwrite. The STOPPED a
  clear causes waits up to a tick for publication, which a stream of refusals
  does reach.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest
from aimdk_msgs.msg import UpperBodyCommandArray
from ament_index_python.packages import get_package_prefix
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray, String

from aimdk_robot import dataplane
from aimdk_robot.vendor_mock_node import ARM_JOINTS, DEXTEROUS_HAND_PARTS
from ibrobot_msgs.srv import StopRuntime

from test_cbf_node import CBF, Logged
from test_runtime_bridge import OMNIHAND_PROFILE, Harness, _vendor_qos, ros_context  # noqa: F401  (the module's ROS context)

COMMAND_TOPIC = "/mirror_session/template_play"
STATUS_TOPIC = "/aimdk_dataplane/template_status"
TERMINAL = {"DONE", "REJECTED", "STOPPED"}
PERIOD_S = 0.02  # the data plane's command tick
# A tick missed shows as a gap of two; ticks that all went out stay near one.
NO_PAUSE_S = 1.5 * PERIOD_S
# The CBF counters come with the data plane's status: a reading can be this many ticks old.
STATUS_LAG_TICKS = math.ceil(1.0 / (dataplane.STATUS_RATE_HZ * PERIOD_S))

# The CBF model's natural standing pose (cbf/generated_constants.hpp). With
# these hands it is slightly inside a pair, so the CBF accepts it as it stands
# (penetration recovery) and the gate admits it.
STAND = [
    0.392598, -0.025215, -0.013518, -1.12909, 0.006807, 0.011635, 0.0,
    0.39068, 0.028282, 0.022147, -1.147114, 0.006999, 0.000191, 0.00515,
]  # fmt: skip
# The left thumb roll's command band starts at 2 degrees; every other slot holds 0.
HAND = [0.03490658] + [0.0] * 19
# Both elbows bend 0.2 rad: a pose the CBF passes unmodified, so it can take over
# there. A finger moves on each hand.
TARGET = list(STAND)
TARGET[3] -= 0.2
TARGET[10] -= 0.2
HAND_TARGET = list(HAND)
HAND_TARGET[4] = 0.5
HAND_TARGET[14] = 0.5
SEGMENT = {
    "P": STAND,
    "T": TARGET,
    "hand_P": HAND,
    "hand_T": HAND_TARGET,
    "t_in": 0.4,
    "t_out": 0.4,
    "t_out_stop": 0.3,
    "hold_max_s": 8.0,
    "start_tol_rad": 0.03,
    "hand_start_tol": 0.05,
    "report_sha256": "0" * 64,
}


def segments_file(directory: Path) -> Path:
    path = directory / "segments.json"
    path.write_text(json.dumps({"schema": "x2_template_segments/v1", "segments": {"probe": SEGMENT}}), encoding="utf-8")
    return path


def enabled(directory: Path, **settings) -> str:
    return json.dumps({"enabled": True, "segments_file": str(segments_file(directory)), **settings})


@pytest.fixture
def start(monkeypatch):
    """Run the bridge, the mock and a data plane given ``template_play_json`` (None: not set at all)."""
    running: list[Harness] = []

    def run(template_play: str | None) -> tuple[Harness, Logged]:
        logged = Logged()

        def start_process(profile):
            executable = Path(get_package_prefix("aimdk_dataplane")) / "lib" / "aimdk_dataplane" / "aimdk_dataplane"
            parameters = {"plan_json": json.dumps(dataplane.plan(profile)), "cbf_safety_json": json.dumps(CBF)}
            if template_play is not None:
                parameters["template_play_json"] = template_play
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="aimdk_dataplane_", delete=False) as file:
                json.dump({"aimdk_dataplane": {"ros__parameters": parameters}}, file)
            process = subprocess.Popen(
                [str(executable), "--ros-args", "--params-file", file.name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            logged.follow(process.stderr)
            return process

        monkeypatch.setattr(dataplane, "start_process", start_process)
        harness = Harness(profile=OMNIHAND_PROFILE, hand_type="dexterous", initial_action="UPPERBODY_REMOTE_SPLIT")
        running.append(harness)
        # The arm stands where the stream will ask it to, the hands where the segment starts.
        harness.mock.set_positions(dict(zip(ARM_JOINTS, STAND, strict=True)))
        harness.mock.set_positions({f"L_{DEXTEROUS_HAND_PARTS[0]}_joint": HAND[0]})
        return harness, logged

    yield run
    for harness in running:
        # A data plane signalled before it spins does not exit on SIGTERM (an existing issue, not the port's).
        harness.dataplane.terminate()
        try:
            harness.dataplane.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            harness.dataplane.kill()
            harness.dataplane.wait(timeout=5.0)
        harness.shutdown()
        # Humble's MultiThreadedExecutor leaves its worker threads behind; under the
        # address-space limit a run of harnesses would run out of threads.
        harness.executor._executor.shutdown(wait=True)


class Session:
    """The mirror session's side: the public arm stream, the port's commands and keepalives, and what comes back."""

    def __init__(self, harness: Harness) -> None:
        self.harness = harness
        self._lock = threading.Lock()
        self.statuses: list[tuple[float, dict]] = []
        # (received, stamped by the data plane, arm, hand) per upper-body command.
        self.upper: list[tuple[float, float, list[float], list[float]]] = []
        self.alive: int | None = None  # The seg_id kept alive, until its DONE, REJECTED or STOPPED.
        self.period_s = PERIOD_S  # of the public stream
        self.streaming = threading.Event()
        self.streaming.set()
        self._closed = threading.Event()
        node = harness.node
        reliable = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
        node.create_subscription(String, STATUS_TOPIC, self._on_status, reliable)
        node.create_subscription(
            UpperBodyCommandArray, "/mc/upper_body_command", self._on_upper, _vendor_qos("upper_body_command")
        )
        self.commands = node.create_publisher(
            String, COMMAND_TOPIC, QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )
        self._arm = node.create_publisher(Float64MultiArray, "/aimdk/arm/commands", 10)
        self._threads = [
            threading.Thread(target=self._stream, daemon=True),
            threading.Thread(target=self._keepalive, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def close(self) -> None:
        self._closed.set()
        for thread in self._threads:
            thread.join(timeout=1.0)

    def _on_status(self, message) -> None:
        status = json.loads(message.data)
        with self._lock:
            self.statuses.append((time.monotonic(), status))
            if status["seg_id"] == self.alive and status["state"] in TERMINAL:
                self.alive = None

    def _on_upper(self, message) -> None:
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        with self._lock:
            self.upper.append((time.monotonic(), stamp, list(message.arm_pos), list(message.hand_pos)))

    def _stream(self) -> None:
        command = Float64MultiArray()
        command.data = list(STAND)  # P_actual, as the session sends it while a segment plays
        while not self._closed.is_set():
            if self.streaming.is_set():
                self._arm.publish(command)
            time.sleep(self.period_s)

    def _keepalive(self) -> None:
        while not self._closed.is_set():
            seg_id = self.alive
            if seg_id is not None:
                self.send(op="keepalive", seg_id=seg_id)
            time.sleep(0.05)

    def send(self, **fields) -> None:
        message = String()
        message.data = json.dumps({"schema": "x2_template_play/v1", **fields})
        self.commands.publish(message)

    def start(self, seg_id: int, segment: str = "probe") -> None:
        if segment == "probe":
            self.alive = seg_id
        self.send(op="start", seg=segment, seg_id=seg_id)

    def release(self, seg_id: int, fast: bool = False) -> None:
        self.send(op="release", seg_id=seg_id, fast=fast)

    def of(self, seg_id: int) -> list[tuple[float, dict]]:
        with self._lock:
            return [(at, status) for at, status in self.statuses if status["seg_id"] == seg_id]

    def states(self, seg_id: int) -> list[str]:
        """The states reported for ``seg_id``, repeats folded."""
        out: list[str] = []
        for _, status in self.of(seg_id):
            if not out or out[-1] != status["state"]:
                out.append(status["state"])
        return out

    def wait_state(self, seg_id: int, state: str, timeout_s: float = 5.0) -> tuple[float, dict]:
        return self.harness.wait_for(
            lambda: next(((at, s) for at, s in self.of(seg_id) if s["state"] == state), None),
            f"seg_id {seg_id} to report {state}",
            timeout_s=timeout_s,
        )

    def upper_between(self, begin: float, end: float = float("inf")):
        with self._lock:
            return [sample for sample in self.upper if begin <= sample[0] <= end]

    def right_after_a_tick(self) -> None:
        """Return as soon as a fresh command is seen: the next tick is a whole period away."""
        with self._lock:
            seen = len(self.upper)
        self.harness.wait_for(lambda: len(self.upper) > seen, "an upper-body command", timeout_s=2.0)


def mirroring(harness: Harness, port: bool = True) -> Session:
    """The runtime in the protected mirror mode, the session streaming, the CBF passing frames unmodified."""
    harness.wait_active()
    # The runtime refuses the mode until the data plane has reported hands that match.
    harness.wait_for(lambda: cbf(harness).get("hand_configuration_matches") is True, "the hands reported")
    assert harness.set_mode("policy_stream").success
    session = Session(harness)
    if port:
        harness.wait_for(lambda: session.commands.get_subscription_count() > 0, "the data plane on the command topic")
    harness.wait_for(
        lambda: cbf(harness).get("mirror_active") is True and cbf(harness).get("accepted", 0) > 10,
        "the CBF to accept the stream",
    )
    return session


def cbf(harness: Harness) -> dict:
    return dict(harness.runtime._dataplane.cbf)


def counter(harness: Harness, name: str) -> int:
    return int(cbf(harness).get(name, 0))


def max_gap(samples) -> float:
    """The longest time between two commands, on the data plane's own stamps."""
    stamps = [sample[1] for sample in samples]
    assert len(stamps) > 10, f"only {len(stamps)} commands to measure"
    return max(later - earlier for earlier, later in zip(stamps, stamps[1:], strict=False))


def largest_step(samples) -> float:
    return max(far(later[2], earlier[2]) for earlier, later in zip(samples, samples[1:], strict=False))


def far(arm: list[float], reference: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(arm, reference, strict=True))


def holding(session: Session, seg_id: int = 1) -> float:
    """Start ``seg_id``, wait until it holds at the target; returns when the hold was reported."""
    session.start(seg_id)
    at, _ = session.wait_state(seg_id, "HOLD")
    session.harness.wait_for(
        lambda: far(session.upper_between(at)[-1][2], TARGET) < 1e-6 if session.upper_between(at) else False,
        "the arm at the target",
    )
    return at


class Exit:
    """What went out around one exit: the longest gap, the ticks missed, and the CBF's withheld frames."""

    def __init__(self, session: Session, exit) -> None:
        harness = session.harness
        # Right after a tick, the stream fast enough to be fresh again by the next one.
        session.period_s = 0.005
        time.sleep(0.3)
        session.right_after_a_tick()
        withheld = counter(harness, "withheld")
        self.played = counter(harness, "played")  # as the exit came
        begin = time.monotonic()
        exit()
        time.sleep(0.6)
        self.withheld = counter(harness, "withheld") - withheld
        end = time.monotonic()
        session.period_s = PERIOD_S
        samples = session.upper_between(begin - 0.1, end)
        self.gap = max_gap(samples)
        stamps = [sample[1] for sample in samples]
        self.missed = sum(
            round((later - earlier) / PERIOD_S) - 1 for earlier, later in zip(stamps, stamps[1:], strict=False)
        )

    def __repr__(self) -> str:
        return f"gap {self.gap:.3f} s, {self.missed} tick(s) missed, {self.withheld} withheld by the CBF"


def flood(session: Session, first: int, stop: threading.Event) -> threading.Thread:
    """A start every 5 ms, each refused, until ``stop``: a refusal lands in every tick."""

    def run():
        seg_id = first
        while not stop.is_set():
            session.start(seg_id, segment="nosuch")
            seg_id += 1
            time.sleep(0.005)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


# --- 0: a whole round -------------------------------------------------------


def test_a_full_round_plays_the_template_and_hands_back_to_the_cbf(start, tmp_path):
    harness, logged = start(enabled(tmp_path))
    session = mirroring(harness)
    try:
        assert logged.matching("template playback enabled: 1 segment(s)")
        assert harness.node.count_publishers(STATUS_TOPIC) == 1  # the count the switched-off cases read as 0
        assert counter(harness, "played") == 0
        before = time.monotonic()
        time.sleep(1.0)
        session.start(1)
        entered, enter = session.wait_state(1, "ENTER_B")
        # The start point is what the data plane published on the admitting tick: the stream's pose.
        assert far(enter["P_actual"], STAND) < SEGMENT["start_tol_rad"] and far(enter["hand_P"], HAND) < 1e-9
        held, _ = session.wait_state(1, "HOLD")
        harness.wait_for(lambda: far(session.upper_between(held)[-1][2], TARGET) < 1e-6, "the arm at the target")
        # The stream asked for STAND throughout; the elbows went to the target, so the player published them.
        elbows = [arm[3] for _, _, arm, _ in session.upper_between(entered, held)]
        assert elbows == sorted(elbows, reverse=True) and elbows[-1] < STAND[3] - 0.1
        assert far(session.upper_between(held)[-1][3], HAND_TARGET) < 1e-9  # both hands, as the segment says
        assert counter(harness, "played") > 0

        session.release(1)
        done_at, done = session.wait_state(1, "DONE")
        assert done["reason"] == "release"
        assert session.states(1) == ["ENTER_B", "HOLD", "EXIT_B", "DONE"]
        counters = done["counters"]
        assert (counters["started"], counters["done"], counters["stopped"], counters["clamped"]) == (1, 1, 0, 0)
        # Back on the CBF path: it accepts frames again from the start point, the player publishes no more.
        accepted = counter(harness, "accepted")
        harness.wait_for(lambda: counter(harness, "accepted") > accepted + 10, "the CBF to accept frames again")
        assert far(session.upper_between(done_at)[-1][2], enter["P_actual"]) < 1e-3
        played = counter(harness, "played")
        time.sleep(0.3)
        assert counter(harness, "played") == played
        # Not one tick without a command, before, through and after the round.
        assert max_gap(session.upper_between(before, entered)) < NO_PAUSE_S
        assert max_gap(session.upper_between(entered, time.monotonic())) < NO_PAUSE_S
    finally:
        session.close()


# --- 1: a stop goes on with the ordinary path, no pause ---------------------


def test_a_runtime_stop_hands_the_stream_back_to_the_cbf_without_a_pause(start, tmp_path):
    """A clear (any command reset) mid-hold: STOPPED epoch_changed, and the CBF path publishes from the next tick.

    A clear can cost the plain mirror one tick too: when the first stream frame
    after it is not in by the next tick, the CBF withholds that tick (here about
    1 in 5 clears). So a tick may go missing only as a frame the CBF withheld,
    never as one nobody published, which is what a stop that does not hand the
    tick on would leave. The same clear in the plain mirror is the reference.
    internal_error leaves through the same fall-through.
    """
    harness, _ = start(enabled(tmp_path))
    session = mirroring(harness)
    try:
        plain = Exit(session, harness.runtime._clear_commands)
        assert plain.missed <= plain.withheld and plain.gap < 2.5 * PERIOD_S, f"plain mirror: {plain}"
        holding(session)
        stopped_at = time.monotonic()
        stop = Exit(session, harness.runtime._clear_commands)
        _, stopped = session.wait_state(1, "STOPPED")
        assert stopped["reason"] == "epoch_changed" and stopped["counters"]["stopped"] == 1
        assert stop.missed <= stop.withheld and stop.gap < 2.5 * PERIOD_S, (
            f"after the stop: {stop} (plain mirror: {plain})"
        )
        # The CBF path took over where the player left the arm: no jump, it publishes, the player does not.
        assert largest_step(session.upper_between(stopped_at)) < 0.05
        accepted = counter(harness, "accepted")
        harness.wait_for(lambda: counter(harness, "accepted") > accepted + 10, "the CBF to accept frames again")
        assert counter(harness, "played") <= stop.played + STATUS_LAG_TICKS + 1
        settled = counter(harness, "played")
        time.sleep(0.3)
        assert counter(harness, "played") == settled
    finally:
        session.close()


def test_a_lapsed_control_word_stops_playback_and_the_stream_goes_on(start, tmp_path):
    """The lapse tick publishes the safe command before the playing branch, so this
    shows the stream goes on; it would not show a stop that skips the fall-through
    (the clear above does).
    """
    harness, _ = start(enabled(tmp_path))
    session = mirroring(harness)
    try:
        holding(session)
        lapsed_at = time.monotonic()
        harness.executor.remove_node(harness.runtime)  # the runtime falls silent; the word lapses after 0.5 s
        _, stopped = session.wait_state(1, "STOPPED")
        assert stopped["reason"] == "control_lapsed"
        time.sleep(0.5)
        assert max_gap(session.upper_between(lapsed_at)) < NO_PAUSE_S
    finally:
        session.close()


# --- 2: runtime-level exits and refusals -------------------------------------


def test_a_mode_switch_stops_playback(start, tmp_path):
    harness, _ = start(enabled(tmp_path))
    session = mirroring(harness)
    try:
        holding(session)
        played = counter(harness, "played")
        stopped_at = time.monotonic()
        assert harness.set_mode("stream").success
        _, stopped = session.wait_state(1, "STOPPED")
        assert stopped["reason"] == "epoch_changed"
        # Stream mode has no CBF: the stream's own target goes out, the player's no more.
        harness.wait_for(lambda: far(session.upper[-1][2], STAND) < 1e-9, "the stream's target")
        time.sleep(0.5)
        assert counter(harness, "played") <= played + STATUS_LAG_TICKS + 1
        assert max_gap(session.upper_between(stopped_at)) < NO_PAUSE_S + PERIOD_S  # a mode switch clears
    finally:
        session.close()


def test_a_runtime_stop_request_stops_playback(start, tmp_path):
    """A stop closes admission: after it the data plane publishes no upper-body command, port or not."""
    harness, _ = start(enabled(tmp_path))
    session = mirroring(harness)
    try:
        holding(session)
        request = StopRuntime.Request()
        request.policy = "HOLD"
        assert harness.call(StopRuntime, "/runtime/stop", request).success
        _, stopped = session.wait_state(1, "STOPPED")
        assert stopped["reason"] == "epoch_changed"
        assert session.states(1)[-2:] == ["HOLD", "STOPPED"]
        settled = counter(harness, "played")
        time.sleep(0.3)
        assert counter(harness, "played") == settled
    finally:
        session.close()


def test_a_named_takeover_stops_playback(start, tmp_path):
    """The claim is applied only once the public stream is quiet; it changes the epoch, so on_control stops."""
    harness, _ = start(enabled(tmp_path))
    session = mirroring(harness)
    try:
        holding(session)
        session.streaming.clear()
        time.sleep(0.7)
        assert session.states(1)[-1] == "HOLD"  # a quiet stream alone does not stop playback
        assert harness.runtime.claim_arm(7)
        _, stopped = session.wait_state(1, "STOPPED")
        assert stopped["reason"] == "epoch_changed"
        harness.runtime.release_arm(7)
    finally:
        session.close()


def test_a_named_claim_refused_against_the_live_stream_leaves_playback_alone(start, tmp_path):
    harness, _ = start(enabled(tmp_path))
    session = mirroring(harness)
    try:
        holding(session)
        assert not harness.runtime.claim_arm(7)
        time.sleep(0.3)
        session.release(1)
        _, done = session.wait_state(1, "DONE")
        assert session.states(1) == ["ENTER_B", "HOLD", "EXIT_B", "DONE"]
        assert done["counters"]["stopped"] == 0
    finally:
        session.close()


def test_a_start_outside_the_mirror_mode_is_refused_not_admitted(start, tmp_path):
    harness, _ = start(enabled(tmp_path))
    session = mirroring(harness)
    try:
        assert harness.set_mode("stream").success
        time.sleep(0.3)
        session.start(1)
        _, rejected = session.wait_state(1, "REJECTED", timeout_s=2.0)
        assert rejected["reason"] == "not_admitted"
        assert session.states(1) == ["REJECTED"]
    finally:
        session.close()


# --- 3: a DONE or STOPPED is not lost to a refusal in the same tick ----------


def test_a_done_goes_out_with_the_refusals_of_its_tick(start, tmp_path):
    harness, _ = start(enabled(tmp_path))
    session = mirroring(harness)
    stop = threading.Event()
    try:
        holding(session)
        flood(session, 1000, stop)
        time.sleep(0.2)
        session.release(1)
        _, done = session.wait_state(1, "DONE")
        time.sleep(0.1)
        stop.set()
        busy = [status for _, status in session.statuses if (status["seg_id"] or 0) >= 1000]
        assert busy and all(status["state"] == "REJECTED" for status in busy)
        # A refusal went out in the DONE's own publication (same stamp), ahead of it.
        assert any(status["at"] == done["at"] for status in busy)
        assert done["counters"]["done"] == 1
    finally:
        stop.set()
        session.close()


def test_a_stopped_is_not_overwritten_by_a_refusal_before_it_goes_out(start, tmp_path):
    harness, _ = start(enabled(tmp_path))
    session = mirroring(harness)
    stop = threading.Event()
    try:
        holding(session)
        flood(session, 1000, stop)
        time.sleep(0.2)
        harness.runtime._clear_commands()
        _, stopped = session.wait_state(1, "STOPPED", timeout_s=2.0)
        time.sleep(0.2)
        stop.set()
        assert stopped["reason"] == "epoch_changed"
        assert any(status["state"] == "REJECTED" and status["at"] == stopped["at"] for _, status in session.statuses)
    finally:
        stop.set()
        session.close()


# --- 4, 5: off, or misconfigured ---------------------------------------------


def assert_off_and_mirroring(harness: Harness, session: Session, logged: Logged) -> None:
    """No port anywhere, and the mirror runs as without one."""
    assert harness.dataplane.poll() is None, "the data plane exited"
    assert harness.node.count_publishers(STATUS_TOPIC) == 0
    assert harness.node.count_subscribers(COMMAND_TOPIC) == 0
    assert "played" not in cbf(harness)
    assert not logged.matching("template playback enabled")
    accepted = counter(harness, "accepted")
    begin = time.monotonic()
    time.sleep(1.0)
    assert counter(harness, "accepted") > accepted + 30
    assert max_gap(session.upper_between(begin)) < NO_PAUSE_S
    assert session.statuses == []


@pytest.mark.parametrize("template_play", [None, "", json.dumps({"enabled": False})], ids=["unset", "empty", "false"])
def test_switched_off_there_is_no_port(start, template_play):
    harness, logged = start(template_play)
    session = mirroring(harness, port=False)
    try:
        assert_off_and_mirroring(harness, session, logged)
        assert not logged.matching("template")
    finally:
        session.close()


@pytest.mark.parametrize(
    ("settings", "logged_as"),
    [
        (
            lambda path: json.dumps({"enabled": "true", "segments_file": str(path)}),
            "template playback stays off, enabled is not true or false",
        ),
        (lambda path: json.dumps([1]), "template playback stays off, template_play_json is not an object"),
        (lambda path: "{not json", "template playback stays off, template_play_json is malformed"),
        (lambda path: json.dumps({"enabled": True, "segments_file": 7}), "template playback stays off"),
        (
            lambda path: json.dumps({"enabled": True, "segments_file": str(path), "status_topic": 7}),
            "template playback stays off",
        ),
        (
            lambda path: json.dumps({"enabled": True, "segments_file": str(path), "command_topic": "not a topic"}),
            "template playback stays off",
        ),
    ],
    ids=["enabled_string", "not_an_object", "not_json", "segments_file_number", "status_topic_number", "bad_topic"],
)
def test_malformed_settings_leave_the_port_off_and_the_node_running(start, tmp_path, settings, logged_as):
    harness, logged = start(settings(segments_file(tmp_path)))
    session = mirroring(harness, port=False)
    try:
        assert_off_and_mirroring(harness, session, logged)
        assert len(logged.matching(logged_as)) == 1, logged.matching("template")
    finally:
        session.close()
