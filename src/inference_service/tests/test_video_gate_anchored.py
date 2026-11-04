"""Unit tests for the request-anchored RTP video freshness gate.

The gate must measure whether the wire's latest capture COVERS the request's
sample time, not whether it is fresh against the wall clock: a world paused
waiting for inference keeps its last frame current for the request it is
still answering.  Wall-clock aging misreads that legal pause as a dead feed
and deadlocks assembly (observed as 572s of video_send_stale refusals with
the wire keepalive actively feeding receivers on the 32k L10 cell).
"""

from __future__ import annotations

from types import SimpleNamespace

from inference_service.pipeline_policy_node import PipelinePolicyNode


def _gate(sent_ns: int, sample_time_ns: int, *, max_age_ns: int = 1_000_000_000):
    node = SimpleNamespace(
        _video_stream_manager=SimpleNamespace(latest_sent_capture_ns=lambda key: sent_ns),
    )
    state = SimpleNamespace(
        spec=SimpleNamespace(key="observation.images.image", topic="/benchmark/libero/observation/images/image"),
        max_age_ns=max_age_ns,
    )
    return PipelinePolicyNode._rtp_video_send_issue(node, state, sample_time_ns)


def test_frozen_world_last_frame_covers_its_request():
    # The world froze at T with the last frame captured at T; the request
    # still asks for T.  The anchored signature takes no wall clock at all:
    # pre-fix, the gate aged the capture against wall-now and refused this
    # exact scenario for 572s on the 32k L10 cell while the wire keepalive
    # was actively feeding receivers.
    frozen_capture = 1_000_000_000_000
    assert _gate(frozen_capture, frozen_capture) is None


def test_wire_ahead_of_request_passes():
    assert _gate(2_000_000_000_000, 1_000_000_000_000) is None


def test_feed_lagging_advancing_sample_time_fails_closed():
    # State keeps advancing (the request sample time moves ahead) while the
    # video wire stalls: coverage debt outlives the budget -> stale.
    issue = _gate(1_000_000_000_000, 1_000_000_000_000 + 2_500_000_000)
    assert issue is not None
    assert issue["reason"] == "video_send_stale"
    assert issue["age_ms"] == 2500.0


def test_never_sent_fails_closed():
    issue = _gate(0, 1_000_000_000_000)
    assert issue is not None
    assert issue["reason"] == "video_not_sent"
