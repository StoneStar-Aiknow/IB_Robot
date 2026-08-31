"""mHandPro glove source and teleoperation device implementations."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from .mhandpro_sdk import CM_PPOSE, CS_SUCCEEDED
from .mhandpro_worker_client import MHandProWorkerClient


@dataclass(frozen=True)
class GloveFrame:
    positions: list[list[float]]
    sequence: int
    timestamp: float
    quaternions: list[list[float]] | None = None
    virtual_positions: list[list[float]] | None = None
    sensor_states: list[int] | None = None
    side: str = "right"
    sdk_frame_index: int = 0
    device_power: float = 0.0
    frequency: int = 0
    gyroscope: list[list[float]] | None = None
    accelerations: list[list[float]] | None = None
    velocities: list[list[float]] | None = None


class GloveSource(Protocol):
    @property
    def is_connected(self) -> bool: ...

    def is_side_connected(self, side: str) -> bool: ...

    @property
    def sdk_version(self) -> str: ...

    def connect(self) -> bool: ...

    def latest_frame(self, side: str) -> GloveFrame | None: ...

    def calibrate_p_pose(self, timeout: float) -> tuple[int, float]: ...

    def disconnect(self) -> None: ...


class RealMHandProSource:
    """Read frames from an isolated process that owns the vendor SDK."""

    def __init__(self, lib_path: str, side: str, *, startup_timeout: float = 30.0):
        if startup_timeout <= 0.0:
            raise ValueError("mHandPro startup_timeout must be positive")
        self._lib_path = lib_path
        self._side = side
        self._startup_timeout = float(startup_timeout)
        self._sdk: MHandProWorkerClient | None = None
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._sdk is not None and self._sdk.is_connected

    def is_side_connected(self, side: str) -> bool:
        return self.is_connected and side == self._side and self._sdk is not None and self._sdk.is_side_connected(side)

    @property
    def sdk_version(self) -> str:
        return self._sdk.sdk_version if self._sdk is not None else "unknown"

    @property
    def sdk(self) -> MHandProWorkerClient:
        if self._sdk is None:
            raise RuntimeError("mHandPro source is not connected")
        return self._sdk

    def connect(self) -> bool:
        sdk = MHandProWorkerClient(self._lib_path, self._side, startup_timeout=self._startup_timeout)
        sdk.connect()
        self._sdk = sdk
        self._is_connected = True
        return True

    def latest_frame(self, side: str) -> GloveFrame | None:
        if side != self._side or self._sdk is None:
            return None
        frame = self._sdk.latest_frame(side)
        if frame is None:
            return None
        return _glove_frame_from_worker(frame, side)

    def calibrate_p_pose(self, timeout: float) -> tuple[int, float]:
        return self.sdk.start_calibration(CM_PPOSE, timeout=timeout)

    def disconnect(self) -> None:
        sdk, self._sdk = self._sdk, None
        self._is_connected = False
        if sdk is not None:
            sdk.disconnect()


class SharedRealMHandProSource:
    """Own one vendor worker and expose every requested glove side."""

    def __init__(
        self,
        lib_path: str,
        sides,
        *,
        startup_timeout: float = 30.0,
        failure_policy: str = "require_all",
    ):
        self._sides = tuple(dict.fromkeys(str(side) for side in sides))
        if not self._sides or not set(self._sides).issubset({"left", "right"}):
            raise ValueError("mHandPro sides must contain left and/or right")
        if startup_timeout <= 0.0:
            raise ValueError("mHandPro startup_timeout must be positive")
        if failure_policy not in ("require_all", "allow_available"):
            raise ValueError("mHandPro failure_policy must be require_all or allow_available")
        self._lib_path = str(lib_path)
        self._startup_timeout = float(startup_timeout)
        self._failure_policy = failure_policy
        self._sdk_lock = threading.Lock()
        self._sdk: MHandProWorkerClient | None = None

    @property
    def is_connected(self) -> bool:
        with self._sdk_lock:
            sdk = self._sdk
        return sdk is not None and sdk.is_connected

    def is_side_connected(self, side: str) -> bool:
        with self._sdk_lock:
            sdk = self._sdk
        return side in self._sides and sdk is not None and sdk.is_side_connected(side)

    @property
    def sdk_version(self) -> str:
        with self._sdk_lock:
            sdk = self._sdk
        return sdk.sdk_version if sdk is not None else "unknown"

    def connect(self) -> bool:
        worker_side = "both" if len(self._sides) == 2 else self._sides[0]
        sdk = MHandProWorkerClient(
            self._lib_path,
            worker_side,
            startup_timeout=self._startup_timeout,
            failure_policy=self._failure_policy,
        )
        with self._sdk_lock:
            if self._sdk is not None:
                raise RuntimeError("mHandPro source is already connecting or connected")
            self._sdk = sdk
        try:
            sdk.connect()
        except Exception:
            with self._sdk_lock:
                if self._sdk is sdk:
                    self._sdk = None
            sdk.disconnect()
            raise
        return True

    def latest_frame(self, side: str) -> GloveFrame | None:
        with self._sdk_lock:
            sdk = self._sdk
        if side not in self._sides or sdk is None:
            return None
        frame = sdk.latest_frame(side)
        if frame is None:
            return None
        return _glove_frame_from_worker(frame, side)

    def calibrate_p_pose(self, timeout: float) -> tuple[int, float]:
        with self._sdk_lock:
            sdk = self._sdk
        if sdk is None:
            raise ConnectionError("mHandPro source is not connected")
        return sdk.start_calibration(CM_PPOSE, timeout=timeout)

    def disconnect(self) -> None:
        with self._sdk_lock:
            sdk, self._sdk = self._sdk, None
        if sdk is not None:
            sdk.disconnect()


class ReplayGloveSource:
    """Deterministic threaded source that replays geometric hand poses."""

    _POSE_SEQUENCE = ("open", "fist", "open", "thumb_abd", "open", "thumb_opp")

    def __init__(self, side: str, rate_hz: float = 50.0, segment_seconds: float = 0.7):
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        if rate_hz <= 0.0 or segment_seconds <= 0.0:
            raise ValueError("Replay rate and segment duration must be positive")
        self._side = side
        self._period = 1.0 / rate_hz
        self._segment_seconds = segment_seconds
        self._lock = threading.Lock()
        self._frame: GloveFrame | None = None
        self._sequence = 0
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_connected(self) -> bool:
        return self._running.is_set()

    def is_side_connected(self, side: str) -> bool:
        return side == self._side and self.is_connected

    @property
    def sdk_version(self) -> str:
        return "replay-v1"

    def connect(self) -> bool:
        if self._running.is_set():
            return True
        self._running.set()
        self._publish_phase(0.0)
        self._thread = threading.Thread(target=self._run, name=f"mhandpro-replay-{self._side}", daemon=True)
        self._thread.start()
        return True

    def latest_frame(self, side: str) -> GloveFrame | None:
        if side != self._side:
            return None
        with self._lock:
            return self._frame

    def calibrate_p_pose(self, timeout: float) -> tuple[int, float]:
        del timeout
        return CS_SUCCEEDED, 1.0

    def disconnect(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        started = time.monotonic()
        while self._running.is_set():
            self._publish_phase((time.monotonic() - started) / self._segment_seconds)
            time.sleep(self._period)

    def _publish_phase(self, phase: float) -> None:
        segment = int(math.floor(phase))
        blend = phase - segment
        first_name = self._POSE_SEQUENCE[segment % len(self._POSE_SEQUENCE)]
        second_name = self._POSE_SEQUENCE[(segment + 1) % len(self._POSE_SEQUENCE)]
        first = replay_pose(first_name, self._side)
        second = replay_pose(second_name, self._side)
        first_virtual = _replay_virtual_fingertips(first, thumb_curled=first_name == "fist")
        second_virtual = _replay_virtual_fingertips(second, thumb_curled=second_name == "fist")
        positions = [
            [a + (b - a) * blend for a, b in zip(first_point, second_point, strict=True)]
            for first_point, second_point in zip(first, second, strict=True)
        ]
        virtual_positions = [
            [a + (b - a) * blend for a, b in zip(first_point, second_point, strict=True)]
            for first_point, second_point in zip(first_virtual, second_virtual, strict=True)
        ]
        with self._lock:
            self._sequence += 1
            self._frame = GloveFrame(
                positions,
                self._sequence,
                time.monotonic(),
                quaternions=[[1.0, 0.0, 0.0, 0.0] for _ in range(20)],
                virtual_positions=virtual_positions,
                sensor_states=[0] * 20,
                side=self._side,
            )


def replay_pose(name: str, side: str = "right") -> list[list[float]]:
    """Create a stable 20-node hand skeleton for one named replay pose."""
    if name not in ("open", "fist", "thumb_abd", "thumb_opp"):
        raise ValueError(f"Unknown replay pose: {name}")
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    fist = 1.0 if name == "fist" else 0.0
    thumb_abd = 1.15 if name == "thumb_abd" else 0.45
    thumb_opp = 0.65 if name == "thumb_opp" else 0.04
    thumb_flex = 0.9 * fist
    positions: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(20)]

    thumb_root = [0.58, 0.38, 0.0]
    thumb_direction = [
        math.sin(thumb_abd) * math.cos(thumb_opp),
        math.cos(thumb_abd) * math.cos(thumb_opp),
        math.sin(thumb_opp),
    ]
    thumb_tip_direction = _bend_direction(thumb_direction, thumb_flex)
    positions[1] = thumb_root
    positions[2] = _add(thumb_root, _scale(thumb_direction, 0.42))
    positions[3] = _add(positions[2], _scale(thumb_tip_direction, 0.32))

    roots = ([0.52, 0.82, 0.0], [0.18, 1.0, 0.0], [-0.22, 0.96, 0.0], [-0.52, 0.78, 0.0])
    lengths = ((0.55, 0.38, 0.27), (0.62, 0.43, 0.3), (0.57, 0.4, 0.28), (0.48, 0.33, 0.24))
    for root_index, root, segment_lengths in zip((4, 8, 12, 16), roots, lengths, strict=True):
        base_direction = _unit(root)
        directions = (
            _bend_direction(base_direction, fist * math.radians(65.0)),
            _bend_direction(base_direction, fist * math.radians(125.0)),
            _bend_direction(base_direction, fist * math.radians(165.0)),
        )
        positions[root_index] = list(root)
        for offset, (length, direction) in enumerate(zip(segment_lengths, directions, strict=True), start=1):
            positions[root_index + offset] = _add(positions[root_index + offset - 1], _scale(direction, length))

    if side == "left":
        return [[-x, y, z] for x, y, z in positions]
    return positions


def _replay_virtual_fingertips(positions, *, thumb_curled: bool) -> list[list[float]]:
    """Complete deterministic replay poses with the five SDK fingertip nodes."""
    fingertips = []
    for finger_index, (previous, terminal) in enumerate(((2, 3), (6, 7), (10, 11), (14, 15), (18, 19))):
        direction = [positions[terminal][axis] - positions[previous][axis] for axis in range(3)]
        length = math.sqrt(sum(value * value for value in direction))
        if length < 1e-9:
            raise ValueError("Replay hand nodes overlap")
        direction = [value / length for value in direction]
        if finger_index == 0 and thumb_curled:
            reference = [1.0, 0.0, 0.0] if abs(direction[0]) < 0.9 else [0.0, 1.0, 0.0]
            direction = [
                direction[1] * reference[2] - direction[2] * reference[1],
                direction[2] * reference[0] - direction[0] * reference[2],
                direction[0] * reference[1] - direction[1] * reference[0],
            ]
            direction_length = math.sqrt(sum(value * value for value in direction))
            direction = [value / direction_length for value in direction]
        extension = 0.75 * length
        fingertips.append([positions[terminal][axis] + extension * direction[axis] for axis in range(3)])
    return fingertips


def _bend_direction(direction, angle):
    horizontal = _unit([direction[0], direction[1], 0.0])
    return [horizontal[0] * math.cos(angle), horizontal[1] * math.cos(angle), -math.sin(angle)]


def _unit(vector):
    length = math.sqrt(sum(value * value for value in vector))
    return [value / length for value in vector]


def _scale(vector, factor):
    return [value * factor for value in vector]


def _add(first, second):
    return [a + b for a, b in zip(first, second, strict=True)]


def _glove_frame_from_worker(frame: dict, side: str) -> GloveFrame:
    return GloveFrame(
        positions=frame["positions"],
        sequence=int(frame["sequence"]),
        timestamp=float(frame["timestamp"]),
        quaternions=frame.get("quaternions"),
        virtual_positions=frame.get("virtual_positions"),
        sensor_states=frame.get("sensor_states"),
        side=side,
        sdk_frame_index=int(frame.get("sdk_frame_index", 0)),
        device_power=float(frame.get("device_power", 0.0)),
        frequency=int(frame.get("frequency", 0)),
        gyroscope=frame.get("gyroscope"),
        accelerations=frame.get("accelerations"),
        velocities=frame.get("velocities"),
    )
