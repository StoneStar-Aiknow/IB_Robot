"""Unit tests for achieved-wire-bitrate accounting.

The configured ``bitrate_bps`` is only the encoder's rate-control target.
These tests pin the counters that make the achieved bitrate measurable:
``StreamMetrics.sent_payload_bytes`` accumulates full datagram octets at the
send site, the frame snapshot carries them plus the configured target, and
the worker diagnostics report a windowed ``wire_kbps`` next to the target.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

from observation_transport.frame_ingress import FrameStreamSnapshot
from observation_transport.managed_frame_ingress import ManagedFrameIngress
from observation_transport.rtp_sender import H264RtpSender, StreamMetrics
from observation_transport.video_codec import EncodedPacket


class _FakeDatagramSink:
    def __init__(self) -> None:
        self.datagrams: list[bytes] = []

    def sendto(self, datagram: bytes, endpoint: object) -> None:
        self.datagrams.append(datagram)

    def close(self) -> None:
        return None


def test_sender_counts_full_datagram_octets():
    sink = _FakeDatagramSink()
    sender = H264RtpSender(
        stream_id="s1",
        endpoint=("127.0.0.1", 56000),
        ssrc=7,
        queue_capacity=4,
        datagram_sender=sink,
        initial_sequence=100,
        background_delivery=False,
    )
    sender.enqueue(EncodedPacket(payload=b"x" * 600, rtp_timestamp=256, capture_timestamp_ns=1_000, keyframe=True))
    sender.enqueue(EncodedPacket(payload=b"y" * 600, rtp_timestamp=512, capture_timestamp_ns=2_000))
    # send_pending() drains exactly one queued access unit per call.
    assert sender.send_pending() is True
    assert sender.send_pending() is True
    assert sender.send_pending() is False
    metrics = sender.status.metrics
    assert metrics.sent_frames == 2
    assert metrics.sent_payload_bytes == sum(len(d) for d in sink.datagrams) > 0
    sender.close()


def test_metrics_counter_default_and_accumulation():
    base = StreamMetrics()
    assert base.sent_payload_bytes == 0
    stepped = replace(base, sent_payload_bytes=base.sent_payload_bytes + 1234)
    assert stepped.sent_payload_bytes == 1234


def _snapshot(key: str, sent_bytes: int, frames: int = 50) -> FrameStreamSnapshot:
    return FrameStreamSnapshot(
        key,
        f"stream-{key}",
        True,
        accepted_frames=frames,
        encoded_frames=frames,
        sent_frames=frames,
        sent_payload_bytes=sent_bytes,
        configured_bitrate_bps=128_000,
    )


def _managed_ingress() -> tuple[ManagedFrameIngress, list[str]]:
    ingress = object.__new__(ManagedFrameIngress)
    ingress._lock = threading.Lock()
    ingress._condition = threading.Condition(ingress._lock)
    ingress._closed = False
    ingress._last_diagnostic_signatures = {}
    ingress._receiver_statuses = {}
    ingress._recovery_attempts = {}
    ingress._last_recovery_monotonic_ns = {}
    ingress._wire_last_activity = {}
    ingress._wire_keepalive_monotonic_ns = {}
    ingress._wire_keepalive_counts = {}
    ingress._wire_keepalive_announced = set()
    ingress._wire_diag_sent_bytes = {}
    ingress._wire_diag_monotonic_ns = {}
    logs: list[str] = []

    class _Logger:
        def info(self, message: str, *args: object, **kwargs: object) -> None:
            logs.append(message)

        def error(self, message: str, *args: object, **kwargs: object) -> None:
            logs.append(message)

    ingress._node = SimpleNamespace(
        get_logger=lambda: _Logger(),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: None)),
    )
    ingress._config = SimpleNamespace(pipeline_id="p", contract_fingerprint="c", deployment_fingerprint="d")
    return ingress, logs


def _set_snapshot(ingress: ManagedFrameIngress, streams: list[FrameStreamSnapshot]) -> None:
    ingress._ingress = SimpleNamespace(snapshot=lambda: SimpleNamespace(streams=streams))


def test_diagnostics_reports_windowed_kbps():
    ingress, logs = _managed_ingress()
    _set_snapshot(ingress, [_snapshot("k", 0)])
    ingress._publish_worker_diagnostics()
    assert any("wire_kbps=n/a" in line and "configured_kbps=128" in line for line in logs)

    # Second report: 125000 bytes over a 1s window -> 1000.0 kbps.  Advance
    # the frame signature (50 -> 75 crosses the //25 boundary) so the report
    # emits, and backdate the bookkeeping timestamp for a deterministic window.
    ingress._wire_diag_monotonic_ns["k"] = time.monotonic_ns() - 1_000_000_000
    _set_snapshot(ingress, [_snapshot("k", 125_000, frames=75)])
    ingress._publish_worker_diagnostics()
    kbps_lines = [line for line in logs if "wire_kbps=" in line and "n/a" not in line]
    assert kbps_lines and "wire_kbps=1000.0" in kbps_lines[-1]


def test_diagnostics_signature_ignores_byte_growth():
    # Bytes grow continuously; the emission signature must stay frame-based so
    # diagnostics keep their cadence instead of firing on every report.
    ingress, logs = _managed_ingress()
    _set_snapshot(ingress, [_snapshot("k", 0)])
    ingress._publish_worker_diagnostics()
    _set_snapshot(ingress, [_snapshot("k", 999_999)])
    ingress._publish_worker_diagnostics()
    worker_lines = [line for line in logs if "FRAME_INGRESS_WORKER" in line]
    assert len(worker_lines) == 1


def test_session_rebind_clears_bitrate_bookkeeping():
    # A cloud heartbeat announcing a new session rebinds the ingress and must
    # reset the achieved-bitrate window, so the next report does not read a
    # byte baseline from the previous session.
    from ibrobot_msgs.msg import InferencePipelineStatus

    ingress, _logs = _managed_ingress()
    ingress._wire_diag_sent_bytes["k"] = 123
    ingress._wire_diag_monotonic_ns["k"] = 456
    bound: list[str] = []

    ingress._ingress = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(session_generation=2, streams=()),
        bind_session=lambda session: (bound.append(session.session_id), True)[1],
        descriptors=lambda: (),
        statuses=lambda: (),
    )
    ingress._heartbeat_callback(
        SimpleNamespace(
            role=InferencePipelineStatus.ROLE_CLOUD,
            pipeline_id="p",
            deployment_fingerprint="d",
            ready=True,
            session_id="session-b",
            session_generation=2,
        )
    )

    assert bound == ["session-b"], "the new cloud session rebinds the ingress"
    assert "k" not in ingress._wire_diag_sent_bytes
    assert "k" not in ingress._wire_diag_monotonic_ns
