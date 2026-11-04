"""Unit tests for the environment-node observation heartbeat.

The heartbeat re-publishes the latest observation batch while an episode sits idle.
It exists to break a class of benchmark deadlock observed on the RoboTwin edge lane:

- the environment publishes one observation per reset/step transaction;
- the policy asks for the frame that belongs to the last step timestamp;
- if that frame is lost on the wire (RTP/UDP), the policy retries forever while the
  environment waits for an action it will never receive;
- the run only ends when the stall watchdog kills it, discarding the whole cell.

The tests below pin the behaviour that matters: the heartbeat publishes only while the
episode is idle, it never publishes without an active batch, and it never raises into the
executor (a heartbeat must not fault an episode).
"""

from __future__ import annotations

import time

import pytest
from rclpy.time import Time

from benchmark_runtime.environment_node import HEARTBEAT_TAG, BenchmarkEnvironmentNode
from benchmark_runtime.observation_router import DeliveryContext

_COMMITTED = DeliveryContext(timestamp_sec=1_700_000_000, timestamp_nanosec=123_456_789)


class _FakeRouter:
    """Records republish calls; can be told to fail."""

    def __init__(self) -> None:
        self.republished: list[DeliveryContext] = []
        self.republish_error: Exception | None = None
        self.last_keys = None

    def republish(self, context: DeliveryContext, keys=None):
        self.last_keys = keys
        if self.republish_error is not None:
            raise self.republish_error
        self.republished.append(context)
        return None


class _FakeLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def info(self, message: str) -> None:
        self.infos.append(message)


class _StubNode(BenchmarkEnvironmentNode):
    """Node-shaped object without starting rclpy or the adapter."""

    def __init__(self, *, period_s: float = 0.25, batch: object | None = object()) -> None:
        # Deliberately bypass BenchmarkEnvironmentNode.__init__ / rclpy.Node.__init__:
        # the heartbeat is pure attribute + router logic and must stay testable without
        # spinning an executor.
        self._obs_heartbeat_period_s = period_s
        self._obs_heartbeat_batch = batch
        self._obs_heartbeat_context = _COMMITTED if batch is not None else None
        self._obs_heartbeat_last_commit = 0.0
        self._obs_heartbeat_warn_at = 0.0
        self._obs_heartbeat_keys = frozenset({"observation.images.cam_high"})
        self._observation_router = _FakeRouter()
        self._closed = False
        self._production_ready = True
        self._logger = _FakeLogger()
        self._now_ns = 1_700_000_000_000_000_000

    def get_clock(self):  # noqa: ANN201 - rclpy Clock surface
        node = self

        class _Clock:
            def now(self) -> Time:
                node._now_ns += 250_000_000
                return Time(nanoseconds=node._now_ns)

        return _Clock()

    def get_logger(self) -> _FakeLogger:
        return self._logger


def test_heartbeat_publishes_when_episode_is_idle() -> None:
    node = _StubNode()
    node._publish_observation_heartbeat()

    assert len(node._observation_router.republished) == 1
    assert node._observation_router.last_keys == node._obs_heartbeat_keys
    assert node._obs_heartbeat_last_commit > 0.0


def test_heartbeat_reuses_the_committed_delivery_context() -> None:
    """The re-send must answer the tick the policy is waiting on, not "now".

    Stamping the heartbeat with the current time moves the frame *past* that tick,
    which is what the cloud reports as newer_than_request.
    """
    node = _StubNode()

    node._publish_observation_heartbeat()
    node._obs_heartbeat_last_commit = 0.0  # the idle period elapsed
    node._publish_observation_heartbeat()

    assert [ctx for ctx in node._observation_router.republished] == [_COMMITTED, _COMMITTED]


def test_heartbeat_context_is_stable_across_ticks_while_the_clock_advances() -> None:
    node = _StubNode()

    node._publish_observation_heartbeat()
    first = node._observation_router.republished[0]
    node._now_ns += 60_000_000_000  # one minute of wall clock
    node._obs_heartbeat_last_commit = 0.0
    node._publish_observation_heartbeat()

    assert node._observation_router.republished[1] == first
    assert first.timestamp_sec != node.get_clock().now().nanoseconds // 1_000_000_000


def test_heartbeat_is_a_noop_without_a_committed_context() -> None:
    node = _StubNode()
    node._obs_heartbeat_context = None

    node._publish_observation_heartbeat()

    assert node._observation_router.republished == []


def test_heartbeat_skips_immediately_after_a_real_commit() -> None:
    node = _StubNode()
    node._obs_heartbeat_last_commit = time.monotonic()

    node._publish_observation_heartbeat()

    assert node._observation_router.republished == []


def test_heartbeat_is_a_noop_without_an_observation_batch() -> None:
    node = _StubNode(batch=None)

    node._publish_observation_heartbeat()

    assert node._observation_router.republished == []


def test_heartbeat_is_a_noop_when_disabled_or_closed() -> None:
    disabled = _StubNode(period_s=0.0)
    disabled._publish_observation_heartbeat()
    assert disabled._observation_router.republished == []

    closed = _StubNode()
    closed._closed = True
    closed._publish_observation_heartbeat()
    assert closed._observation_router.republished == []


def test_heartbeat_never_raises_into_the_executor() -> None:
    node = _StubNode()
    node._observation_router.republish_error = RuntimeError("router exploded")

    node._publish_observation_heartbeat()  # must not raise

    assert node._observation_router.republished == []
    assert any(HEARTBEAT_TAG in message for message in node._logger.warnings)


def test_heartbeat_failure_keeps_last_commit_untouched() -> None:
    node = _StubNode()
    node._observation_router.republish_error = RuntimeError("sink down")

    node._publish_observation_heartbeat()

    assert node._observation_router.republished == []
    assert node._obs_heartbeat_last_commit == 0.0
    assert node._logger.warnings


def test_heartbeat_warning_is_throttled() -> None:
    node = _StubNode()
    node._observation_router.republish_error = RuntimeError("boom")

    for _ in range(5):
        node._publish_observation_heartbeat()

    assert len(node._logger.warnings) == 1


def test_remember_batch_tracks_commit_time_and_context() -> None:
    node = _StubNode()
    batch = object()
    context = DeliveryContext(timestamp_sec=42, timestamp_nanosec=7)

    node._remember_observation_batch(batch, context)

    assert node._obs_heartbeat_batch is batch
    assert node._obs_heartbeat_context is context
    assert node._obs_heartbeat_last_commit > 0.0


def test_remember_batch_is_a_noop_when_heartbeat_disabled() -> None:
    node = _StubNode(period_s=0.0)
    original = node._obs_heartbeat_batch

    node._remember_observation_batch(object(), _COMMITTED)

    assert node._obs_heartbeat_batch is original


if __name__ == "__main__":  # pragma: no cover - manual execution
    raise SystemExit(pytest.main([__file__, "-q"]))
