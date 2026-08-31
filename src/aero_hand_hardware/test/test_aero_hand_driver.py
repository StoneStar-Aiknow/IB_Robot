import math
import struct
import threading
import time
from types import SimpleNamespace

import pytest
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from std_msgs.msg import Bool, Float64MultiArray

from aero_hand_hardware.aero_hand_driver import AeroHandDriver
from aero_hand_hardware.aero_hand_node import DEFAULT_RIGHT_JOINT_NAMES, AeroHandNode, _advance_deadline


def test_default_right_joint_names_match_public_compact_contract():
    assert DEFAULT_RIGHT_JOINT_NAMES == [
        "right_thumb_cmc_abd",
        "right_thumb_cmc_flex",
        "right_thumb_mcp_ip",
        "right_index",
        "right_middle",
        "right_ring",
        "right_pinky",
    ]


def test_node_accepts_double_array_limit_overrides_with_optional_safe_pose():
    lower_limits = "[0.0,0.0,0.0,0.0,0.0,0.0,0.0]"
    upper_limits = "[1.0,1.0,1.0,1.0,1.0,1.0,1.0]"
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "mock:=true",
            "-p",
            f"command_lower_limits:={lower_limits}",
            "-p",
            f"command_upper_limits:={upper_limits}",
        ]
    )
    node = None
    try:
        node = AeroHandNode()

        assert node.command_limits == [(0.0, 1.0)] * 7
        assert node.safe_pose_rad == []
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_mock_driver_round_trips_compact_degree_positions():
    driver = AeroHandDriver(mock=True)
    assert driver.connect() is True

    command = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    driver.set_joint_positions(command)

    assert driver.get_joint_positions() == command


@pytest.mark.parametrize(
    "command",
    [
        [0.0] * 6,
        [0.0] * 8,
        [0.0, 0.0, 0.0, math.nan, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, math.inf, 0.0, 0.0, 0.0],
    ],
)
def test_driver_rejects_invalid_commands(command):
    driver = AeroHandDriver(mock=True)
    driver.connect()

    with pytest.raises(ValueError):
        driver.set_joint_positions(command)


def test_driver_estop_clears_queued_motion_and_only_allows_safe_pose():
    driver = AeroHandDriver(mock=True)
    driver.connect()
    driver.set_joint_positions([10.0] * 7)

    driver.set_emergency_stop(True)

    assert driver._pending_command_deg is None
    with pytest.raises(RuntimeError, match="E-stop"):
        driver.set_joint_positions([20.0] * 7)
    driver.set_joint_positions([0.0] * 7, blocking=True, allow_during_estop=True)
    assert driver.get_joint_positions() == [0.0] * 7

    driver.set_emergency_stop(False)
    with pytest.raises(RuntimeError, match="not active"):
        driver.set_joint_positions([0.0] * 7, blocking=True, allow_during_estop=True)
    driver.set_joint_positions([30.0] * 7)
    assert driver.get_joint_positions() == [30.0] * 7


def test_mock_driver_reconnect_clears_disconnect_estop_latch():
    driver = AeroHandDriver(mock=True)
    driver.connect()
    driver.disconnect()

    driver.connect()
    driver.set_joint_positions([15.0] * 7)

    assert driver.get_joint_positions() == [15.0] * 7


def test_driver_estop_waits_for_the_serial_write_barrier():
    driver = AeroHandDriver(mock=False)
    driver._connected = True
    driver._hand = object()
    driver._port_lock.acquire()
    finished = threading.Event()

    thread = threading.Thread(target=lambda: (driver.set_emergency_stop(True), finished.set()))
    thread.start()
    assert not finished.wait(0.05)

    driver._port_lock.release()
    thread.join(timeout=1.0)
    assert finished.is_set()
    assert driver._estop_active is True


def test_state_deadline_preserves_20hz_average_on_50hz_timer():
    deadline = 0.0
    read_times = []
    for tick in range(1, 251):
        now = tick / 50.0
        if now < deadline:
            continue
        read_times.append(now)
        deadline = _advance_deadline(deadline, now, 1.0 / 20.0)

    assert len(read_times) == pytest.approx(100, abs=1)
    assert read_times[-1] - read_times[0] == pytest.approx(5.0, abs=0.06)


def test_stale_command_is_dropped_after_first_warning():
    warnings = []
    node = SimpleNamespace(
        _command_lock=threading.Lock(),
        _latest_command_rad=[0.0] * 7,
        _latest_command_time=time.monotonic() - 1.0,
        _estop_active=False,
        _safe_pose_pending=False,
        safe_pose_rad=[],
        _next_state_read=math.inf,
        command_timeout=0.25,
        driver=SimpleNamespace(set_joint_positions=lambda _positions: pytest.fail("stale command was sent")),
        _warn_rate_limited=lambda key, message: warnings.append((key, message)),
    )

    AeroHandNode.control_callback(node)
    AeroHandNode.control_callback(node)

    assert node._latest_command_rad is None
    assert len(warnings) == 1
    assert warnings[0][0] == "stale_command"


def test_estop_hold_drops_cached_command_immediately():
    estop_states = []
    node = SimpleNamespace(
        _command_lock=threading.Lock(),
        _latest_command_rad=[0.5] * 7,
        _latest_command_time=time.monotonic(),
        _estop_active=False,
        _safe_pose_pending=False,
        estop_behavior="hold",
        safe_pose_rad=[],
        _next_state_read=math.inf,
        command_timeout=0.25,
        driver=SimpleNamespace(
            set_emergency_stop=estop_states.append,
            set_joint_positions=lambda _positions: pytest.fail("E-stop hold sent a command"),
        ),
    )

    AeroHandNode.estop_callback(node, Bool(data=True))
    AeroHandNode.control_callback(node)

    assert node._estop_active is True
    assert node._latest_command_rad is None
    assert estop_states == [True]


def test_estop_safe_pose_is_sent_once_from_control_timer():
    commands = []
    safe_pose = [0.1 * index for index in range(7)]
    node = SimpleNamespace(
        _command_lock=threading.Lock(),
        _latest_command_rad=[0.5] * 7,
        _latest_command_time=time.monotonic(),
        _estop_active=False,
        _safe_pose_pending=False,
        estop_behavior="safe_pose",
        safe_pose_rad=safe_pose,
        _next_state_read=math.inf,
        command_timeout=0.25,
        driver=SimpleNamespace(
            set_emergency_stop=lambda _active: None,
            set_joint_positions=lambda positions, *, blocking=False, allow_during_estop=False: commands.append(
                (positions, blocking, allow_during_estop)
            ),
        ),
        _warn_rate_limited=lambda *_args: None,
    )

    AeroHandNode.estop_callback(node, Bool(data=True))
    assert commands == []
    AeroHandNode.control_callback(node)
    AeroHandNode.control_callback(node)

    assert commands == [([math.degrees(value) for value in safe_pose], True, True)]


def test_estop_safe_pose_retries_until_the_blocking_write_succeeds():
    """A failed safe-pose write must stay pending, not vanish into the command queue."""
    attempts = []
    safe_pose = [0.1 * index for index in range(7)]

    def flaky_write(positions, *, blocking=False, allow_during_estop=False):
        assert blocking is True
        assert allow_during_estop is True
        attempts.append(positions)
        if len(attempts) == 1:
            raise TimeoutError("serial write timed out")

    node = SimpleNamespace(
        _command_lock=threading.Lock(),
        _latest_command_rad=None,
        _latest_command_time=0.0,
        _estop_active=False,
        _safe_pose_pending=False,
        estop_behavior="safe_pose",
        safe_pose_rad=safe_pose,
        _next_state_read=math.inf,
        command_timeout=0.25,
        driver=SimpleNamespace(set_emergency_stop=lambda _active: None, set_joint_positions=flaky_write),
        _warn_rate_limited=lambda *_args: None,
    )

    AeroHandNode.estop_callback(node, Bool(data=True))
    AeroHandNode.control_callback(node)
    assert node._safe_pose_pending is True

    AeroHandNode.control_callback(node)
    assert node._safe_pose_pending is False

    AeroHandNode.control_callback(node)
    assert len(attempts) == 2


def test_estop_keeps_joint_state_telemetry_active():
    published = []
    node = SimpleNamespace(
        _command_lock=threading.Lock(),
        _latest_command_rad=None,
        _latest_command_time=0.0,
        _estop_active=True,
        _safe_pose_pending=False,
        safe_pose_rad=[],
        _next_state_read=0.0,
        state_frequency=20.0,
        command_timeout=0.25,
        joint_names=list(DEFAULT_RIGHT_JOINT_NAMES),
        driver=SimpleNamespace(get_joint_positions=lambda: [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        joint_state_pub=SimpleNamespace(publish=published.append),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=TimeMsg)),
        _warn_rate_limited=lambda *_args: None,
    )

    AeroHandNode.control_callback(node)

    assert len(published) == 1
    assert published[0].name == DEFAULT_RIGHT_JOINT_NAMES
    assert published[0].position == pytest.approx([math.radians(value) for value in range(0, 61, 10)])


def test_commands_are_ignored_until_estop_is_released():
    estop_states = []
    node = SimpleNamespace(
        _command_lock=threading.Lock(),
        _latest_command_rad=None,
        _latest_command_time=0.0,
        _estop_active=True,
        _safe_pose_pending=False,
        estop_behavior="hold",
        command_limits=[(0.0, 1.0)] * 7,
        driver=SimpleNamespace(set_emergency_stop=estop_states.append),
        _warn_rate_limited=lambda *_args: None,
    )
    command = Float64MultiArray(data=[0.2] * 7)

    AeroHandNode.command_callback(node, command)
    assert node._latest_command_rad is None
    AeroHandNode.estop_callback(node, Bool(data=False))
    AeroHandNode.command_callback(node, command)
    assert node._latest_command_rad == [0.2] * 7
    assert estop_states == [False]


def test_command_callback_clamps_to_hardware_boundary_limits():
    warnings = []
    node = SimpleNamespace(
        _command_lock=threading.Lock(),
        _latest_command_rad=None,
        _latest_command_time=0.0,
        _estop_active=False,
        command_limits=[(0.0, 1.0)] * 7,
        _warn_rate_limited=lambda key, message: warnings.append((key, message)),
    )

    AeroHandNode.command_callback(node, Float64MultiArray(data=[-0.5, 0.2, 2.0, 0.4, 0.5, 0.6, 0.7]))

    assert node._latest_command_rad == [0.0, 0.2, 1.0, 0.4, 0.5, 0.6, 0.7]
    assert warnings == [("limited_command", "Clamped Aero Hand command to configured joint limits")]


class _SlowHand:
    """SDK stand-in whose readback blocks far longer than one control period.

    Emulates the real half-duplex protocol: ``_send_data`` issues a request and
    the reply only becomes readable after ``read_delay``. Writes are
    fire-and-forget, matching CTRL_POS which expects no ACK.
    """

    actuation_lower_limits = [0.0] * 7
    actuation_upper_limits = [100.0] * 7

    class _Model:
        def hand_joints(self, actuations_rad):
            return list(actuations_rad)

    class _Serial:
        def __init__(self, hand):
            self._hand = hand

        def reset_input_buffer(self):
            self._hand._reply_ready_at = None

        def close(self):
            self._hand.closed = True

        @property
        def in_waiting(self):
            hand = self._hand
            if hand.read_returns is None or hand._reply_ready_at is None:
                return 0
            return 16 if time.monotonic() >= hand._reply_ready_at else 0

        def read(self, count):
            hand = self._hand
            hand._reply_ready_at = None
            values = [
                int(
                    (value - hand.actuation_lower_limits[index])
                    / (hand.actuation_upper_limits[index] - hand.actuation_lower_limits[index])
                    * 65535
                )
                for index, value in enumerate(hand.read_returns)
            ]
            return struct.pack("<2B7H", 0x22, 0x00, *values)

    def __init__(self, read_delay: float = 0.16, read_returns=None):
        self.read_delay = read_delay
        self.read_returns = read_returns
        self.written = []
        self.write_times = []
        self.read_count = 0
        self.closed = False
        self._reply_ready_at = None
        self.ser = self._Serial(self)
        self.actuations_to_joints_model = self._Model()

    def _send_data(self, header, payload=None):
        self.read_count += 1
        self._reply_ready_at = time.monotonic() + self.read_delay

    def set_joint_positions(self, positions):
        self.written.append(list(positions))
        self.write_times.append(time.monotonic())

    def get_joint_positions_compact(self):
        self.read_count += 1
        time.sleep(self.read_delay)
        return self.read_returns


def _threaded_driver(hand, **kwargs):
    driver = AeroHandDriver(port="/dev/null", **kwargs)
    driver._hand = hand
    driver._connected = True
    driver._stop_event.clear()
    driver._io_thread = threading.Thread(target=driver._io_loop, daemon=True)
    driver._io_thread.start()
    return driver


def test_commands_reach_the_sdk_at_full_rate_despite_a_slow_readback():
    """A slow readback must not gate how fast commands actually reach the hand.

    Enqueue latency is not the metric that matters; this asserts on the interval
    between real SDK writes, which a serialized write-then-blocking-read loop
    would stretch to the read delay.
    """
    hand = _SlowHand(read_delay=0.16, read_returns=[10.0] * 7)
    driver = _threaded_driver(hand, command_frequency=50.0, state_frequency=20.0)
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            driver.set_joint_positions([1.0] * 7)
            time.sleep(0.005)
    finally:
        driver.disconnect()

    assert len(hand.write_times) >= 35, f"only {len(hand.write_times)} writes reached the SDK in 1 s"
    gaps = [second - first for first, second in zip(hand.write_times, hand.write_times[1:], strict=False)]
    assert max(gaps) < 0.060, f"worst write gap was {max(gaps) * 1000:.1f} ms"


def test_state_readback_still_succeeds_while_commands_stream():
    """The non-blocking reply poll must still deliver fresh state."""
    hand = _SlowHand(read_delay=0.16, read_returns=[10.0] * 7)
    driver = _threaded_driver(hand, command_frequency=50.0, state_frequency=20.0)
    state = None
    try:
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            driver.set_joint_positions([1.0] * 7)
            try:
                state = driver.get_joint_positions()
                break
            except TimeoutError:
                time.sleep(0.01)
    finally:
        driver.disconnect()

    assert state is not None, "no state was ever collected while commanding"
    assert state == pytest.approx([10.0] * 7, abs=0.05)


def test_set_joint_positions_does_not_block_on_a_slow_readback():
    """The regression this driver change exists to prevent."""
    hand = _SlowHand(read_delay=0.16, read_returns=[0.0] * 7)
    driver = _threaded_driver(hand, command_frequency=50.0, state_frequency=20.0)
    try:
        time.sleep(0.05)  # let the I/O thread enter a blocking read
        worst = 0.0
        for _ in range(20):
            start = time.perf_counter()
            driver.set_joint_positions([1.0] * 7)
            worst = max(worst, time.perf_counter() - start)
            time.sleep(0.02)
    finally:
        driver.disconnect()

    assert worst < 0.005, f"set_joint_positions blocked for {worst * 1000:.1f} ms"


def test_get_joint_positions_does_not_block_on_a_slow_readback():
    hand = _SlowHand(read_delay=0.16, read_returns=[5.0] * 7)
    driver = _threaded_driver(hand, command_frequency=50.0, state_frequency=20.0)
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                driver.get_joint_positions()
                break
            except TimeoutError:
                time.sleep(0.01)
        else:
            pytest.fail("driver never cached a readback")

        worst = 0.0
        for _ in range(20):
            start = time.perf_counter()
            driver.get_joint_positions()
            worst = max(worst, time.perf_counter() - start)
            time.sleep(0.01)
    finally:
        driver.disconnect()

    assert worst < 0.005, f"get_joint_positions blocked for {worst * 1000:.1f} ms"


def test_unanswered_reads_are_abandoned_without_stalling_writes():
    """A hand that never replies must not wedge the loop awaiting a reply."""
    hand = _SlowHand(read_delay=0.0, read_returns=None)
    driver = _threaded_driver(
        hand,
        command_frequency=50.0,
        state_frequency=20.0,
        state_timeout=0.05,
        read_reply_timeout=0.05,
    )
    try:
        deadline = time.monotonic() + 0.6
        while time.monotonic() < deadline:
            driver.set_joint_positions([2.0] * 7)
            time.sleep(0.005)
        with pytest.raises(TimeoutError):
            driver.get_joint_positions()
    finally:
        driver.disconnect()

    assert driver.read_failure_count > 0
    assert len(hand.write_times) >= 20, f"only {len(hand.write_times)} writes despite dead readback"


def test_queued_commands_coalesce_to_the_latest_value():
    """Backlogged commands must not replay stale poses onto the hand."""
    hand = _SlowHand(read_delay=0.0, read_returns=[0.0] * 7)
    driver = _threaded_driver(hand, command_frequency=20.0, state_frequency=1.0)
    try:
        for value in range(10):
            driver.set_joint_positions([float(value)] * 7)
        time.sleep(0.2)
    finally:
        driver.disconnect()

    assert hand.written, "no command reached the hand"
    assert hand.written[-1] == [9.0] * 7
    assert len(hand.written) < 10


def test_stale_cached_state_is_reported_instead_of_served():
    hand = _SlowHand(read_delay=0.0, read_returns=None)
    driver = _threaded_driver(
        hand,
        command_frequency=50.0,
        state_frequency=20.0,
        state_timeout=0.05,
        read_reply_timeout=0.05,
    )
    try:
        time.sleep(0.2)
        with pytest.raises(TimeoutError):
            driver.get_joint_positions()
        assert driver.read_failure_count > 0
    finally:
        driver.disconnect()


def test_disconnect_stops_the_io_thread():
    hand = _SlowHand(read_delay=0.0, read_returns=[0.0] * 7)
    driver = _threaded_driver(hand, command_frequency=50.0, state_frequency=20.0)
    thread = driver._io_thread
    driver.disconnect()

    assert thread is not None
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert driver.is_connected is False
