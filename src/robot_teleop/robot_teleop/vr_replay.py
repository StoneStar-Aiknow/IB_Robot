"""Replay a recorded Quest NDJSON session into ``vr_teleop`` over its TCP socket.

Recording format (one JSON object per line): packets carry ``host_mono_ns`` and
``raw`` (the line the Quest app sent); connection events carry ``event`` and no
``raw``. Replaying through the socket keeps the device-parsing stage under test.
"""

from __future__ import annotations

import gzip
import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecordedPacket:
    offset_s: float
    raw: bytes


def load_recording(path: str | Path) -> list[RecordedPacket]:
    """Load packets from ``.ndjson`` or ``.ndjson.gz``, offsets relative to the first packet."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    packets: list[RecordedPacket] = []
    first_ns: int | None = None
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if "raw" not in record:
                continue
            stamp_ns = int(record["host_mono_ns"])
            if first_ns is None:
                first_ns = stamp_ns
            packets.append(RecordedPacket((stamp_ns - first_ns) / 1e9, record["raw"].encode("utf-8") + b"\n"))
    if not packets:
        raise ValueError(f"recording has no packets: {path}")
    return packets


def replay(
    packets: list[RecordedPacket],
    host: str,
    port: int,
    speed: float = 1.0,
    stall_at_s: float | None = None,
    stall_for_s: float = 0.0,
    connect_timeout_s: float = 10.0,
) -> int:
    """Send ``packets`` with their recorded spacing and return how many were sent.

    ``stall_at_s`` holds the connection open but silent for ``stall_for_s``, which
    reproduces a stream loss without tearing down TCP.
    """
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    if stall_for_s < 0.0:
        raise ValueError("stall_for_s must not be negative")
    sent = 0
    stalled = False
    with socket.create_connection((host, port), timeout=connect_timeout_s) as connection:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        start = time.monotonic()
        for packet in packets:
            if stall_at_s is not None and not stalled and packet.offset_s >= stall_at_s:
                stalled = True
                time.sleep(stall_for_s)
                start += stall_for_s
            delay = start + packet.offset_s / speed - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            connection.sendall(packet.raw)
            sent += 1
    return sent
