"""Unit tests for ObservationRouter.republish against the real router.

The observation heartbeat leans on republish() to re-send only a subset of
routes with the original delivery context while leaving the transaction
identity and the sequence counters untouched. The heartbeat node tests use a
fake router on purpose (they pin the node's own gating), so this file pins
the router seam itself: a regression that makes republish() raise or advance
state would be silently swallowed by the node's exception handler, the
heartbeat would go quiet, and the original stall deadlock would return while
the fake-router tests stay green.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark_runtime.io_descriptor import ObservationBatch, ObservationSample
from benchmark_runtime.observation_router import (
    DeliveryContext,
    ObservationCommitError,
    ObservationRouter,
    ObservationSink,
    PreparedRoute,
)

IMG = "observation.images.top"
STATE = "observation.state"
T0 = 1_700_000_123_456_789_012
T1 = T0 + 10_000_000_000

RESET_CTX = DeliveryContext(timestamp_sec=T0 // 1_000_000_000, timestamp_nanosec=T0 % 1_000_000_000)
STEP_CTX = DeliveryContext(timestamp_sec=T1 // 1_000_000_000, timestamp_nanosec=T1 % 1_000_000_000)


class RecordingSink(ObservationSink):
    """Minimal real sink: records the (stamp, value) it commits, can fail on demand."""

    def __init__(self, key: str) -> None:
        self._key = key
        self.published: list[tuple[int, np.ndarray]] = []
        self.fail_next_commit = False

    @property
    def key(self) -> str:
        return self._key

    @property
    def transport(self) -> str:
        return "dds"

    def prepare(self, batch: ObservationBatch, context: DeliveryContext) -> PreparedRoute:
        stamp_ns = context.timestamp_sec * 1_000_000_000 + context.timestamp_nanosec
        return PreparedRoute(key=self._key, transport=self.transport, payload=(stamp_ns, batch[self._key]))

    def commit(self, prepared: PreparedRoute) -> None:
        if self.fail_next_commit:
            self.fail_next_commit = False
            raise RuntimeError("commit exploded")
        self.published.append(prepared.payload)


def _make_batch(sequence_id: int) -> ObservationBatch:
    return ObservationBatch(
        samples=(
            ObservationSample(
                key=IMG, value=np.zeros((4, 4, 3), dtype=np.uint8), capture_timestamp_ns=T0, clock_domain="env"
            ),
            ObservationSample(
                key=STATE, value=np.zeros(7, dtype=np.float32), capture_timestamp_ns=T0, clock_domain="env"
            ),
        ),
        episode_transaction_id="tx-1",
        sequence_id=sequence_id,
    )


def _router() -> tuple[ObservationRouter, RecordingSink, RecordingSink]:
    img_sink = RecordingSink(IMG)
    state_sink = RecordingSink(STATE)
    return ObservationRouter((img_sink, state_sink)), img_sink, state_sink


def test_republish_re_sends_only_requested_routes_with_the_original_stamp():
    router, img_sink, state_sink = _router()
    assert router.prepare(_make_batch(0), RESET_CTX).commit().status == "committed"

    receipt = router.republish(RESET_CTX, keys=frozenset({IMG}))

    assert receipt.status == "committed"
    assert receipt.committed_keys == (IMG,)
    assert len(img_sink.published) == 2, "the requested image route is re-sent"
    assert len(state_sink.published) == 1, "routes outside keys must not be re-sent"
    assert img_sink.published[1][0] == T0, "the re-send carries the committed stamp, not now()"


def test_republish_without_keys_re_sends_every_route():
    router, img_sink, state_sink = _router()
    assert router.prepare(_make_batch(0), RESET_CTX).commit().status == "committed"

    receipt = router.republish(STEP_CTX)

    assert receipt.status == "committed"
    assert receipt.committed_keys == (IMG, STATE)
    assert len(img_sink.published) == 2 and len(state_sink.published) == 2


def test_republish_leaves_identity_and_counters_intact():
    router, img_sink, _state_sink = _router()
    reset_receipt = router.prepare(_make_batch(0), RESET_CTX).commit()
    assert reset_receipt.status == "committed"

    router.republish(RESET_CTX, keys=frozenset({IMG}))

    assert router._last_committed_sequence == 0, "sequence counter must not advance"
    assert router._active_transaction_id == "tx-1", "transaction identity must not change"
    assert router.last_receipt is reset_receipt, "the authoritative receipt must not be overwritten"

    # The property the heartbeat actually depends on: the next real step batch
    # still carries the sequence the router expects and validates cleanly.
    step_receipt = router.prepare(_make_batch(1), STEP_CTX).commit()
    assert step_receipt.status == "committed"
    assert step_receipt.committed_keys == (IMG, STATE)


def test_republish_without_a_committed_batch_is_rejected_and_does_not_poison():
    router, _img_sink, _state_sink = _router()

    with pytest.raises(Exception, match="no committed observation batch"):
        router.republish(RESET_CTX, keys=frozenset({IMG}))

    # The failed republish must not leave the router unable to serve a real reset.
    assert router.prepare(_make_batch(0), RESET_CTX).commit().status == "committed"


def test_republish_failure_does_not_poison_the_next_transaction():
    router, img_sink, _state_sink = _router()
    assert router.prepare(_make_batch(0), RESET_CTX).commit().status == "committed"
    img_sink.fail_next_commit = True  # exactly the republish's commit explodes

    with pytest.raises(ObservationCommitError):
        router.republish(RESET_CTX, keys=frozenset({IMG}))

    # republish=True must skip poisoning: a transport hiccup on the heartbeat
    # can never invalidate the episode transaction.
    assert router._poisoned_transaction_id is None
    assert router._last_committed_sequence == 0
    assert router.prepare(_make_batch(1), STEP_CTX).commit().status == "committed"


if __name__ == "__main__":  # pragma: no cover - manual execution
    raise SystemExit(pytest.main([__file__, "-q"]))
