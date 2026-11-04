"""Unit tests for the ManagedFrameIngress wire keepalive.

The keepalive exists because a frozen control loop (no steps -> no new frames)
starves the cloud-side video receivers: their readiness and freshness gates
decay without recent packets, which deadlocks in-flight inference assembly.
These tests pin the trigger conditions without constructing a ROS node.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from observation_transport.managed_frame_ingress import (
    _WIRE_KEEPALIVE_IDLE_S,
    _WIRE_KEEPALIVE_MIN_INTERVAL_S,
    ManagedFrameIngress,
)


class _FakeLogger:
    def __init__(self) -> None:
        self.infos: list[str] = []

    def info(self, message: str, *args, **kwargs) -> None:
        self.infos.append(message)

    def debug(self, message: str, *args, **kwargs) -> None:
        pass

    def warning(self, message: str, *args, **kwargs) -> None:
        pass

    def error(self, message: str, *args, **kwargs) -> None:
        pass


def _fake_node() -> tuple[SimpleNamespace, _FakeLogger]:
    logger = _FakeLogger()
    return SimpleNamespace(get_logger=lambda: logger), logger


class _FakeIngress:
    def __init__(self, streams: tuple[SimpleNamespace, ...], session_generation: int = 7) -> None:
        self._streams = list(streams)
        self.session_generation = session_generation
        self.keepalive_requests: list[tuple[str, int]] = []

    def snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(session_generation=self.session_generation, streams=tuple(self._streams))

    def request_idle_keepalive(self, observation_key: str, *, session_generation: int) -> bool:
        self.keepalive_requests.append((observation_key, session_generation))
        return True


def _stream(observation_key: str = "image", *, encoded_frames: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        observation_key=observation_key,
        worker_alive=True,
        worker_failed=False,
        encoded_frames=encoded_frames,
    )


def _make_managed(ingress: _FakeIngress) -> tuple[ManagedFrameIngress, _FakeLogger]:
    managed = ManagedFrameIngress.__new__(ManagedFrameIngress)
    node, logger = _fake_node()
    managed._node = node
    managed._lock = threading.RLock()
    managed._condition = threading.Condition(managed._lock)
    managed._closed = False
    managed._ingress = ingress
    managed._last_diagnostic_signatures = {}
    managed._receiver_statuses = {}
    managed._recovery_attempts = {}
    managed._last_recovery_monotonic_ns = {}
    managed._wire_last_activity = {}
    managed._wire_keepalive_monotonic_ns = {}
    managed._wire_keepalive_counts = {}
    managed._wire_keepalive_announced = set()
    return managed, logger


def _force_idle(managed: ManagedFrameIngress, key: str, encoded_frames: int) -> None:
    """Seed the activity tracker as if the wire has been idle past the budget."""
    managed._wire_last_activity[key] = (encoded_frames, time.monotonic() - _WIRE_KEEPALIVE_IDLE_S * 4)


def test_wire_keepalive_fires_after_idle_and_announces_once():
    ingress = _FakeIngress((_stream(),))
    managed, logger = _make_managed(ingress)
    managed._wire_keepalive_tick()  # first sighting only records activity
    assert ingress.keepalive_requests == []
    _force_idle(managed, "image", 10)
    managed._wire_keepalive_tick()
    assert ingress.keepalive_requests == [("image", 7)]
    assert managed._wire_keepalive_counts["image"] == 1
    assert len(logger.infos) == 1
    assert "WIRE_KEEPALIVE" in logger.infos[0]
    # A second tick inside the min interval must not spam another recovery.
    managed._wire_keepalive_tick()
    assert ingress.keepalive_requests == [("image", 7)]
    assert len(logger.infos) == 1


def test_wire_keepalive_repeats_while_frozen_but_resets_on_progress():
    stream = _stream()
    ingress = _FakeIngress((stream,))
    managed, _logger = _make_managed(ingress)
    _force_idle(managed, "image", 10)
    managed._wire_keepalive_tick()
    assert len(ingress.keepalive_requests) == 1
    # Simulate the recovery re-encode landing: progress resets the idle streak.
    stream.encoded_frames = 11
    managed._wire_keepalive_tick()
    assert len(ingress.keepalive_requests) == 1
    # Frozen again past the idle budget -> keepalive fires once more.
    # Backdate the min-interval stamp to simulate real time passing.
    managed._wire_keepalive_monotonic_ns["image"] -= int(_WIRE_KEEPALIVE_MIN_INTERVAL_S * 4 * 1e9)
    _force_idle(managed, "image", 11)
    managed._wire_keepalive_tick()
    assert len(ingress.keepalive_requests) == 2
    assert managed._wire_keepalive_counts["image"] == 2


def _unhealthy_stream(*, alive: bool, failed: bool) -> SimpleNamespace:
    return SimpleNamespace(
        observation_key="image",
        worker_alive=alive,
        worker_failed=failed,
        encoded_frames=10,
    )


@pytest.mark.parametrize(
    ("stream", "session_generation"),
    [
        (_stream(), 0),
        (_unhealthy_stream(alive=False, failed=False), 7),
        (_unhealthy_stream(alive=True, failed=True), 7),
    ],
    ids=["inactive_session", "dead_worker", "failed_worker"],
)
def test_wire_keepalive_never_fires_outside_a_healthy_active_session(stream, session_generation):
    ingress = _FakeIngress((stream,), session_generation=session_generation)
    managed, _logger = _make_managed(ingress)
    _force_idle(managed, stream.observation_key, stream.encoded_frames)
    managed._wire_keepalive_tick()
    assert ingress.keepalive_requests == []


def test_wire_keepalive_swallows_recovery_errors():
    class _FailingIngress(_FakeIngress):
        def request_idle_keepalive(self, observation_key: str, *, session_generation: int) -> bool:
            raise RuntimeError("stale_session")

    ingress = _FailingIngress((_stream(),))
    managed, _logger = _make_managed(ingress)
    _force_idle(managed, "image", 10)
    managed._wire_keepalive_tick()  # must not raise
    assert managed._wire_keepalive_counts["image"] == 1
