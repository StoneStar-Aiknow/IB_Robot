"""The replayer preserves packet content and spacing, and can inject a silent stall."""

import gzip
import json
import socket
import threading
import time

import pytest

from robot_teleop.vr_replay import RecordedPacket, load_recording, replay


def _write_recording(path):
    records = [
        {"event": "connected", "host_mono_ns": 1},
        {"host_mono_ns": 1_000_000_000, "seq": 1, "raw": '{"a":1}'},
        {"host_mono_ns": 1_050_000_000, "seq": 2, "raw": '{"a":2}'},
        {"event": "disconnected", "host_mono_ns": 2_000_000_000},
    ]
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write("\n".join(json.dumps(record) for record in records) + "\n")


def _serve(arrivals):
    server = socket.create_server(("127.0.0.1", 0))

    def run():
        connection, _ = server.accept()
        with connection, connection.makefile("rb") as stream:
            for line in stream:
                arrivals.append((time.monotonic(), line))
        server.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return server.getsockname()[1], thread


def test_load_keeps_packets_and_skips_connection_events(tmp_path):
    path = tmp_path / "session.ndjson.gz"
    _write_recording(path)
    packets = load_recording(path)
    assert [packet.raw for packet in packets] == [b'{"a":1}\n', b'{"a":2}\n']
    assert packets[1].offset_s == pytest.approx(0.05)


def test_load_rejects_recording_without_packets(tmp_path):
    path = tmp_path / "empty.ndjson"
    path.write_text('{"event":"connected"}\n')
    with pytest.raises(ValueError, match="no packets"):
        load_recording(path)


def test_replay_preserves_order_and_scales_spacing():
    arrivals = []
    port, thread = _serve(arrivals)
    packets = [RecordedPacket(0.0, b"1\n"), RecordedPacket(0.4, b"2\n")]
    assert replay(packets, "127.0.0.1", port, speed=2.0) == 2
    thread.join(5.0)
    assert [line for _, line in arrivals] == [b"1\n", b"2\n"]
    assert arrivals[1][0] - arrivals[0][0] == pytest.approx(0.2, abs=0.08)


def test_replay_stall_keeps_connection_silent_then_resumes():
    arrivals = []
    port, thread = _serve(arrivals)
    packets = [RecordedPacket(0.0, b"1\n"), RecordedPacket(0.1, b"2\n")]
    assert replay(packets, "127.0.0.1", port, stall_at_s=0.1, stall_for_s=0.3) == 2
    thread.join(5.0)
    assert arrivals[1][0] - arrivals[0][0] >= 0.35


@pytest.mark.parametrize("kwargs", [{"speed": 0.0}, {"stall_for_s": -1.0}])
def test_replay_rejects_invalid_timing(kwargs):
    with pytest.raises(ValueError):
        replay([RecordedPacket(0.0, b"1\n")], "127.0.0.1", 1, **kwargs)
