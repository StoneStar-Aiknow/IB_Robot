"""Strict real-hardware and deterministic mock driver for Aero Hand."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Sequence

from .aero_sdk_transport import AeroSdkTransport

JOINT_COUNT = 7


def _validated_positions(values: Sequence[float], *, label: str) -> list[float]:
    if len(values) != JOINT_COUNT:
        raise ValueError(f"{label} must contain exactly {JOINT_COUNT} positions")
    positions = [float(value) for value in values]
    if not all(math.isfinite(value) for value in positions):
        raise ValueError(f"{label} must contain only finite positions")
    return positions


class AeroHandDriver:
    """Own the Aero SDK object and expose compact seven-joint degrees.

    Real hardware I/O runs on a dedicated thread that exclusively owns the serial
    port. ``set_joint_positions`` and ``get_joint_positions`` only touch in-memory
    caches, so a slow or unanswered SDK readback can never stall the ROS control
    timer. The SDK's ``read`` uses a 10 ms inter-byte timeout, which means a
    partially answered frame can block far longer than one control period.

    Mock mode stays fully synchronous: it performs no I/O, so a thread would add
    nondeterminism without removing any blocking.
    """

    def __init__(
        self,
        *,
        port: str | None = None,
        baudrate: int = 921600,
        mock: bool = False,
        command_frequency: float = 50.0,
        state_frequency: float = 20.0,
        state_timeout: float = 0.5,
        read_reply_timeout: float = 0.3,
        logger=None,
    ):
        self.port = port or None
        self.baudrate = int(baudrate)
        self.mock = bool(mock)
        self.logger = logger or logging.getLogger(__name__)
        if command_frequency <= 0.0 or state_frequency <= 0.0:
            raise ValueError("command_frequency and state_frequency must be positive")
        if state_timeout <= 0.0:
            raise ValueError("state_timeout must be positive")
        if read_reply_timeout <= 0.0:
            raise ValueError("read_reply_timeout must be positive")
        self.command_frequency = float(command_frequency)
        self.state_frequency = float(state_frequency)
        self.state_timeout = float(state_timeout)
        self.read_reply_timeout = float(read_reply_timeout)
        self._hand = None
        self._transport: AeroSdkTransport | None = None
        self._connected = False
        self._mock_positions_deg = [0.0] * JOINT_COUNT
        self._mock_command_count = 0

        self._io_lock = threading.Lock()
        self._port_lock = threading.Lock()
        self._io_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pending_command_deg: list[float] | None = None
        self._estop_active = False
        self._state_deg: list[float] | None = None
        self._state_time = 0.0
        self._read_failures = 0
        self._write_failures = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def read_failure_count(self) -> int:
        """Cumulative failed readbacks, for diagnostics."""
        with self._io_lock:
            return self._read_failures

    @property
    def write_failure_count(self) -> int:
        """Cumulative failed command writes, for diagnostics."""
        with self._io_lock:
            return self._write_failures

    def connect(self) -> bool:
        if self._connected:
            return True
        if self.mock:
            with self._io_lock:
                self._pending_command_deg = None
                self._estop_active = False
                self._connected = True
            self.logger.info("Aero Hand mock driver connected")
            return True

        try:
            from aero_open_sdk.aero_hand import AeroHand

            self._hand = AeroHand(port=self.port, baudrate=self.baudrate)
            self._transport = AeroSdkTransport(self._hand)
        except Exception as exc:
            self._hand = None
            self._transport = None
            self._connected = False
            raise ConnectionError(f"Failed to connect Aero Hand on {self.port or 'auto-detect'}: {exc}") from exc

        self._connected = True
        self._stop_event.clear()
        with self._io_lock:
            self._pending_command_deg = None
            self._estop_active = False
        self._io_thread = threading.Thread(target=self._io_loop, name="aero_hand_io", daemon=True)
        self._io_thread.start()
        self.logger.info(f"Aero Hand connected on {self.port or 'auto-detect'}")
        return True

    def set_joint_positions(
        self,
        positions_deg: Sequence[float],
        *,
        blocking: bool = False,
        allow_during_estop: bool = False,
    ) -> None:
        """Queue a command. Returns immediately; the I/O thread performs the write.

        Args:
            positions_deg: Seven joint positions in degrees.
            blocking: Bypass the queue and write directly. Only use for E-stop safe
                      poses where you need synchronous error reporting. Blocks until
                      the I/O thread completes one cycle or the port lock is acquired.
            allow_during_estop: Permit the configured E-stop safe pose to bypass the
                                driver latch. Normal motion must leave this disabled.
        """
        positions = _validated_positions(positions_deg, label="Aero Hand command")
        if not self._connected:
            raise RuntimeError("Aero Hand driver is not connected")
        if self.mock:
            with self._io_lock:
                if allow_during_estop and not self._estop_active:
                    raise RuntimeError("Aero Hand E-stop safe pose rejected because E-stop is not active")
                if self._estop_active and not allow_during_estop:
                    raise RuntimeError("Aero Hand command rejected while E-stop is active")
                self._mock_positions_deg = positions
                self._mock_command_count += 1
            if self._mock_command_count % 50 == 0:
                formatted = ", ".join(f"{value:.1f}" for value in positions)
                self.logger.info(f"Aero Hand mock command (degrees): [{formatted}]")
            return
        if self._hand is None:
            raise RuntimeError("Aero Hand SDK object is unavailable")
        if blocking:
            with self._port_lock:
                with self._io_lock:
                    if allow_during_estop and not self._estop_active:
                        raise RuntimeError("Aero Hand E-stop safe pose rejected because E-stop is not active")
                    if self._estop_active and not allow_during_estop:
                        raise RuntimeError("Aero Hand command rejected while E-stop is active")
                self._hand.set_joint_positions(positions)
            return
        with self._io_lock:
            if self._estop_active and not allow_during_estop:
                raise RuntimeError("Aero Hand command rejected while E-stop is active")
            self._pending_command_deg = positions

    def set_emergency_stop(self, active: bool) -> None:
        """Latch command writes and establish a barrier against queued motion.

        When this method returns with ``active=True``, any write already holding
        the serial lock has completed and no queued normal command can start.
        """
        active = bool(active)
        if self.mock or not self._connected:
            with self._io_lock:
                self._estop_active = active
                if active:
                    self._pending_command_deg = None
            return

        with self._port_lock, self._io_lock:
            self._estop_active = active
            if active:
                self._pending_command_deg = None

    def get_joint_positions(self) -> list[float]:
        """Return the most recent readback. Returns immediately; never touches serial."""
        if not self._connected:
            raise RuntimeError("Aero Hand driver is not connected")
        if self.mock:
            return list(self._mock_positions_deg)
        if self._hand is None:
            raise RuntimeError("Aero Hand SDK object is unavailable")
        with self._io_lock:
            state = self._state_deg
            age = time.monotonic() - self._state_time
        if state is None:
            raise TimeoutError("Aero Hand joint-state is not available yet")
        if age > self.state_timeout:
            raise TimeoutError(f"Aero Hand joint-state is stale ({age:.3f}s)")
        return list(state)

    def _io_loop(self) -> None:
        """Serialize every SDK call onto one thread that solely owns the port.

        Writes must never wait on a readback. The Aero protocol is half-duplex
        request/response, but ``CTRL_POS`` is fire-and-forget (no ACK), so only
        reads occupy the port waiting for bytes. Instead of calling the SDK's
        blocking ``get_joint_positions_compact()``, the state request is issued
        and its reply is polled non-blockingly across later cycles. A hand that
        answers slowly (or never) therefore costs zero write cadence.
        """
        command_period = 1.0 / self.command_frequency
        state_period = 1.0 / self.state_frequency
        next_cycle = time.monotonic()
        next_read = time.monotonic()
        request_deadline = 0.0
        awaiting_reply = False

        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_cycle:
                self._stop_event.wait(min(command_period, next_cycle - now))
                continue
            while next_cycle <= now:
                next_cycle += command_period

            with self._io_lock:
                command = self._pending_command_deg
                self._pending_command_deg = None
                if self._estop_active:
                    command = None
            if command is not None:
                try:
                    with self._port_lock:
                        with self._io_lock:
                            if self._estop_active:
                                command = None
                        if command is not None:
                            self._hand.set_joint_positions(command)
                except Exception as exc:
                    with self._io_lock:
                        self._write_failures += 1
                    self.logger.warning(f"Aero Hand command write failed: {exc}")

            if self._stop_event.is_set():
                continue

            now = time.monotonic()
            if awaiting_reply:
                if self._collect_state_reply():
                    awaiting_reply = False
                elif now >= request_deadline:
                    awaiting_reply = False
                    with self._io_lock:
                        self._read_failures += 1
                    self.logger.debug("Aero Hand joint-state read timed out")
                continue

            if now < next_read:
                continue
            while next_read <= now:
                next_read += state_period
            if self._request_state():
                awaiting_reply = True
                request_deadline = time.monotonic() + self.read_reply_timeout

    def _request_state(self) -> bool:
        """Send a position request without waiting for the reply."""
        try:
            with self._port_lock:
                self._transport_for_hand().request_position()
        except Exception as exc:
            with self._io_lock:
                self._read_failures += 1
            self.logger.debug(f"Aero Hand state request failed: {exc}")
            return False
        return True

    def _collect_state_reply(self) -> bool:
        """Consume a complete reply if one has already arrived. Never blocks."""
        try:
            with self._port_lock:
                positions_deg = self._transport_for_hand().poll_position()
        except Exception as exc:
            with self._io_lock:
                self._read_failures += 1
            self.logger.debug(f"Aero Hand state read failed: {exc}")
            return True

        if positions_deg is None:
            return False

        try:
            validated = _validated_positions(
                positions_deg,
                label="Aero Hand joint-state readback",
            )
        except Exception as exc:
            with self._io_lock:
                self._read_failures += 1
            self.logger.debug(f"Aero Hand state decode failed: {exc}")
            return True

        with self._io_lock:
            self._state_deg = validated
            self._state_time = time.monotonic()
        return True

    def _transport_for_hand(self) -> AeroSdkTransport:
        if self._hand is None:
            raise RuntimeError("Aero Hand SDK object is unavailable")
        if self._transport is None or self._transport.hand is not self._hand:
            self._transport = AeroSdkTransport(self._hand)
        return self._transport

    def disconnect(self) -> None:
        self._stop_event.set()
        with self._io_lock:
            self._estop_active = True
            self._pending_command_deg = None
        thread = self._io_thread
        self._io_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        hand = self._hand
        self._hand = None
        transport = self._transport
        self._transport = None
        self._connected = False
        if hand is None:
            return
        if transport is not None:
            transport.close()
        else:
            AeroSdkTransport(hand).close()
