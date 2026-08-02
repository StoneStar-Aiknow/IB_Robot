"""Tests for RTP Annex-B episode recording integrity."""

from __future__ import annotations

import json

from inference_service.h264_stream_recorder import H264StreamRecorder
from inference_service.video_recording_coordinator import VideoRecordingCoordinator


def _write_frame(
    recorder: H264StreamRecorder,
    frame_index: int,
    *,
    lost_packets: int = 0,
    dropped: str | None = None,
) -> None:
    recorder.write_access_unit(
        payload=b"\x65payload",
        capture_timestamp_ns=None if dropped else 1_000_000_000 + frame_index,
        rtp_timestamp=90_000 + frame_index,
        frame_index=frame_index,
        keyframe=frame_index == 0,
        lost_packets=lost_packets,
        session_generation=1,
        dropped=dropped,
    )


def test_strict_mode_discards_files_on_rtp_sequence_gap(tmp_path):
    recorder = H264StreamRecorder(integrity_mode="strict")
    recorder.start_episode(tmp_path, "observation.images.top")
    _write_frame(recorder, 0)
    _write_frame(recorder, 1, lost_packets=2)

    assert recorder.stop_episode() is False
    assert not (tmp_path / "observation.images.top.h264").exists()
    assert not (tmp_path / "observation.images.top.h264.json").exists()


def test_strict_mode_discards_files_on_timestamp_mapping_failure(tmp_path):
    recorder = H264StreamRecorder(integrity_mode="strict")
    recorder.start_episode(tmp_path, "observation.images.top")
    _write_frame(recorder, 0)
    _write_frame(recorder, 1, dropped="timestamp_unmapped")

    assert recorder.stop_episode() is False
    assert list(tmp_path.iterdir()) == []


def test_tolerant_mode_preserves_gap_metadata_without_dropped_payload(tmp_path):
    recorder = H264StreamRecorder(integrity_mode="tolerant")
    recorder.start_episode(tmp_path, "observation.images.top")
    _write_frame(recorder, 0)
    _write_frame(recorder, 1, lost_packets=3)
    _write_frame(recorder, 2, dropped="timestamp_unmapped")

    assert recorder.stop_episode() is True
    stream = (tmp_path / "observation.images.top.h264").read_bytes()
    entries = [json.loads(line) for line in (tmp_path / "observation.images.top.h264.json").read_text().splitlines()]

    assert stream.count(b"\x00\x00\x00\x01") == 2
    assert entries[1]["lost_packets"] == 3
    assert entries[2]["dropped"] == "timestamp_unmapped"
    assert entries[2]["capture_timestamp_ns"] is None


def test_coordinator_strict_failure_discards_every_stream(tmp_path):
    coordinator = VideoRecordingCoordinator()
    clean = H264StreamRecorder(integrity_mode="strict")
    damaged = H264StreamRecorder(integrity_mode="strict")
    coordinator.register_recorder("observation.images.top", clean)
    coordinator.register_recorder("observation.images.wrist", damaged)
    coordinator.start_episode(tmp_path)
    _write_frame(clean, 0)
    _write_frame(damaged, 0, lost_packets=1)

    assert coordinator.stop_episode() is False
    assert list(tmp_path.glob("*.h264")) == []
    assert list(tmp_path.glob("*.h264.json")) == []
