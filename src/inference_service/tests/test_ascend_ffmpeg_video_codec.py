from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inference_service.ascend_ffmpeg_video_codec import (
    AscendFfmpegH264Decoder,
    AscendFfmpegH264Encoder,
    build_ascend_child_environment,
    probe_ascend_codec_diagnostic,
    register_ascend_backend,
    resolve_ascend_ffmpeg,
)
from inference_service.video_codec import (
    CodecLifecycleState,
    EncodedPacket,
    VideoCodecError,
    VideoCodecRegistry,
    VideoFrame,
)
from inference_service.video_rtp import RtpPacket
from tensormsg.converter import hwc_uint8_to_nv12


class _QueuePipe:
    def __init__(self) -> None:
        self._chunks: deque[bytes] = deque()
        self._closed = False
        self._condition = threading.Condition()
        self.read_sizes: list[int] = []

    def put(self, payload: bytes) -> None:
        with self._condition:
            self._chunks.append(payload)
            self._condition.notify_all()

    def read(self, _size: int = -1) -> bytes:
        with self._condition:
            self.read_sizes.append(_size)
            while not self._chunks and not self._closed:
                self._condition.wait()
            return self._chunks.popleft() if self._chunks else b""

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class _InputPipe(io.BytesIO):
    def __init__(self, on_write=None) -> None:
        super().__init__()
        self._on_write = on_write

    def write(self, payload: bytes) -> int:
        result = super().write(payload)
        if self._on_write is not None:
            self._on_write(payload)
        return result


class _FakeProcess:
    def __init__(self, *, stdin=None, stdout=None, stderr=None) -> None:
        self.stdin = stdin or _InputPipe()
        self.stdout = stdout
        self.stderr = stderr or io.BytesIO()
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        del timeout
        self.returncode = 0
        if self.stdout is not None:
            self.stdout.close()
        return 0

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9


class _FakeSocket:
    def __init__(self) -> None:
        self.datagrams: deque[bytes] = deque()
        self.bound = None
        self.timeout = None
        self.closed = False

    def bind(self, address):
        self.bound = address

    def getsockname(self):
        return ("127.0.0.1", 24000)

    def settimeout(self, timeout):
        self.timeout = timeout

    def recvfrom(self, _size):
        if self.closed:
            raise OSError("closed")
        if not self.datagrams:
            raise TimeoutError("empty")
        return self.datagrams.popleft(), ("127.0.0.1", 10000)

    def sendto(self, data, _endpoint):
        self.datagrams.append(data)
        return len(data)

    def close(self):
        self.closed = True


def test_module_import_has_no_pyav_or_ascend_dependency():
    script = (
        "import sys; "
        "import inference_service.ascend_ffmpeg_video_codec; "
        "assert 'av' not in sys.modules; "
        "assert not any(name.startswith('acl') for name in sys.modules)"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)

    result = subprocess.run(
        [sys.executable, "-c", script], env=environment, capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


def test_ffmpeg_resolution_order_and_child_only_library_path(tmp_path: Path):
    explicit = _executable(tmp_path / "explicit" / "ffmpeg")
    prefix = tmp_path / "private"
    prefixed = _executable(prefix / "bin" / "ffmpeg")
    (prefix / "lib").mkdir()
    environment = {
        "IBROBOT_ASCEND_FFMPEG": str(explicit),
        "IBROBOT_ASCEND_FFMPEG_PREFIX": str(prefix),
        "LD_LIBRARY_PATH": "/inherited",
    }

    assert resolve_ascend_ffmpeg(environment) == str(explicit)
    explicit.unlink()
    assert resolve_ascend_ffmpeg(environment) == str(prefixed)
    child = build_ascend_child_environment(str(prefixed), environment)

    assert child["LD_LIBRARY_PATH"] == f"{prefix / 'lib'}:/inherited"
    assert environment["LD_LIBRARY_PATH"] == "/inherited"


def test_child_environment_can_isolate_ascend_toolkit_variables(tmp_path: Path):
    ffmpeg = _executable(tmp_path / "private" / "bin" / "ffmpeg")
    environment = {
        "IBROBOT_ASCEND_FFMPEG_ISOLATE_ENV": "1",
        "ASCEND_HOME_PATH": "/toolkit",
        "ASCEND_OPP_PATH": "/opp",
        "TOOLCHAIN_HOME": "/toolchain",
    }

    child = build_ascend_child_environment(str(ffmpeg), environment)

    assert "ASCEND_HOME_PATH" not in child
    assert "ASCEND_OPP_PATH" not in child
    assert "TOOLCHAIN_HOME" not in child


@pytest.mark.parametrize(
    ("kind", "direction", "listing"),
    [
        ("encoder", "-encoders", " V..... h264_ascend Ascend H264"),
        ("decoder", "-decoders", " V..... h264_ascend Ascend H264"),
    ],
)
def test_probe_accepts_any_ffmpeg_version_with_h264_ascend(tmp_path: Path, kind, direction, listing):
    ffmpeg = _executable(tmp_path / "ffmpeg")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        output = "ffmpeg version 4.4.4-private\n" if "-version" in command else listing
        return SimpleNamespace(returncode=0, stdout=output)

    diagnostic = probe_ascend_codec_diagnostic(kind, environ={"IBROBOT_ASCEND_FFMPEG": str(ffmpeg)}, run=run)

    assert diagnostic.available is True
    assert diagnostic.code == "available"
    assert calls[1][0][-1] == direction
    assert all(call[1]["timeout"] == 2.0 for call in calls)


def test_probe_accepts_ffmpeg_611(tmp_path: Path):
    ffmpeg = _executable(tmp_path / "ffmpeg")
    outputs = deque(["ffmpeg version 6.1.1\n", " V..... h264_ascend Ascend H264"])

    diagnostic = probe_ascend_codec_diagnostic(
        "decoder",
        environ={"IBROBOT_ASCEND_FFMPEG": str(ffmpeg)},
        run=lambda _command, **_kwargs: SimpleNamespace(returncode=0, stdout=outputs.popleft()),
    )

    assert diagnostic.available is True


def test_probe_returns_structured_direction_failure(tmp_path: Path):
    ffmpeg = _executable(tmp_path / "ffmpeg")
    outputs = deque(["ffmpeg version 5.0\n", " V..... libx264"])

    def run(_command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=outputs.popleft())

    missing_codec = probe_ascend_codec_diagnostic("encoder", environ={"IBROBOT_ASCEND_FFMPEG": str(ffmpeg)}, run=run)

    assert (missing_codec.available, missing_codec.code) == (False, "codec_direction_missing")


def test_backend_registration_is_optional_and_named_ascend(monkeypatch):
    registry = VideoCodecRegistry()
    monkeypatch.setattr(
        "inference_service.ascend_ffmpeg_video_codec.probe_ascend_codec_diagnostic",
        lambda kind: SimpleNamespace(available=True, kind=kind),
    )
    register_ascend_backend(registry)

    resolved = registry.resolve("ascend", "encoder")

    assert resolved.name == "ascend"
    assert resolved.capabilities.hardware_accelerated is True


def test_encoder_command_environment_rtp_and_timestamp_pairing(tmp_path: Path):
    ffmpeg, prefix, environment = _private_ffmpeg(tmp_path)
    fake_socket = _FakeSocket()
    fake_socket.datagrams.append(RtpPacket(96, True, 7, 1234, 99, b"\x65encoded").to_bytes())
    spawned = []

    def process_factory(command, **kwargs):
        spawned.append((command, kwargs))
        return _FakeProcess()

    encoder = AscendFfmpegH264Encoder(
        width=4,
        height=2,
        frame_rate_hz=20,
        bitrate_bps=100_000,
        gop_frames=10,
        device_id=2,
        channel_id=3,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=process_factory,
        socket_factory=lambda *_args: fake_socket,
    )
    frame = VideoFrame(np.zeros((2, 4, 3), dtype=np.uint8), 987_654, 987_700, 4, 2, "rgb24")

    packets = encoder.encode(frame)

    command, kwargs = spawned[0]
    assert _option(command, "-c:v") == "h264_ascend"
    assert _option(command, "-device_id") == "2"
    assert _option(command, "-channel_id") == "3"
    assert _option(command, "-pix_fmt") == "nv12"
    assert command[-1].startswith("rtp://127.0.0.1:24000?")
    assert kwargs["env"]["LD_LIBRARY_PATH"].startswith(str(prefix / "lib"))
    assert packets[0].capture_timestamp_ns == 987_654
    assert packets[0].rtp_timestamp == 1234
    assert packets[0].keyframe is True
    assert packets[0].payload == b"\x00\x00\x00\x01\x65encoded"
    assert encoder.metrics.input_frames == encoder.metrics.output_frames == 1
    encoder.close()
    assert fake_socket.closed is True


def test_encoder_writes_raw_nv12_and_reset_recreates_process_and_socket(tmp_path: Path):
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    sockets = [_FakeSocket(), _FakeSocket()]
    for item in sockets:
        item.datagrams.append(RtpPacket(96, True, 1, 90, 1, b"\x41x").to_bytes())
    processes = []

    def process_factory(_command, **_kwargs):
        process = _FakeProcess()
        processes.append(process)
        return process

    encoder = AscendFfmpegH264Encoder(
        width=4,
        height=2,
        frame_rate_hz=10,
        bitrate_bps=1000,
        gop_frames=2,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=process_factory,
        socket_factory=lambda *_args: sockets[len(processes)],
    )
    image = np.full((2, 4, 3), 30, dtype=np.uint8)
    encoder.encode(VideoFrame(image, 20, 20, 4, 2, "rgb24"))

    assert processes[0].stdin.getvalue() == hwc_uint8_to_nv12(image).tobytes()
    encoder.reset()
    assert len(processes) == 2
    assert sockets[0].closed is True
    assert encoder.state is CodecLifecycleState.RUNNING
    encoder.close()


def test_decoder_command_fixed_nv12_output_and_fifo_timestamp_pairing(tmp_path: Path):
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    process = _FakeProcess()
    spawned = []
    sockets = [_FakeSocket(), _FakeSocket(), _FakeSocket()]
    output_socket = sockets[1]

    def process_factory(command, **kwargs):
        spawned.append((command, kwargs))
        return process

    decoder = AscendFfmpegH264Decoder(
        width=4,
        height=2,
        frame_rate_hz=20,
        device_id=4,
        channel_id=5,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=process_factory,
        socket_factory=lambda *_args: sockets.pop(0),
    )
    nv12 = hwc_uint8_to_nv12(np.full((2, 4, 3), 80, dtype=np.uint8)).tobytes()
    output_socket.datagrams.append(nv12)

    frames = decoder.decode(EncodedPacket(b"\x00\x00\x00\x01\x65x", 999, 123_456, keyframe=True))

    command = spawned[0][0]
    assert command[-1].startswith("udp://127.0.0.1:24000?")
    assert _option(command, "-device_id") == "4"
    assert _option(command, "-channel_id") == "5"
    assert _option(command, "-fflags") == "nobuffer"
    assert _option(command, "-analyzeduration") == "0"
    assert _option(command, "-probesize") == "32"
    assert _option(command, "-i").startswith("udp://127.0.0.1:24000?")
    assert frames[0].capture_timestamp_ns == 123_456
    assert frames[0].pixel_format == "rgb24"
    assert frames[0].data.shape == (2, 4, 3)
    assert frames[0].keyframe is True
    assert decoder.metrics.output_frames == 1
    decoder.close()


def test_decoder_waits_for_asynchronous_ffmpeg_output(tmp_path: Path):
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    process = _FakeProcess()
    sockets = [_FakeSocket(), _FakeSocket(), _FakeSocket()]
    output_socket = sockets[1]
    decoder = AscendFfmpegH264Decoder(
        width=4,
        height=2,
        frame_rate_hz=20,
        io_timeout_s=0.5,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=lambda *_args, **_kwargs: process,
        socket_factory=lambda *_args: sockets.pop(0),
    )
    nv12 = hwc_uint8_to_nv12(np.full((2, 4, 3), 80, dtype=np.uint8)).tobytes()

    producer = threading.Thread(target=lambda: (time.sleep(0.01), output_socket.datagrams.append(nv12)))
    producer.start()
    frames = decoder.decode(EncodedPacket(b"\x00\x00\x00\x01\x65x", 999, 123_456, keyframe=True))
    producer.join()

    assert len(frames) == 1
    assert frames[0].capture_timestamp_ns == 123_456
    decoder.close()


def test_decoder_output_wait_is_bounded_by_frame_rate(tmp_path: Path):
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    sockets = [_FakeSocket(), _FakeSocket(), _FakeSocket()]
    decoder = AscendFfmpegH264Decoder(
        width=4,
        height=2,
        frame_rate_hz=20,
        io_timeout_s=0.5,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=lambda *_args, **_kwargs: _FakeProcess(stdout=_QueuePipe()),
        socket_factory=lambda *_args: sockets.pop(0),
    )

    started = time.monotonic()
    frames = decoder.decode(EncodedPacket(b"\x00\x00\x00\x01\x65x", 999, 123_456, keyframe=True))
    elapsed = time.monotonic() - started

    assert frames == []
    assert elapsed < 0.1
    decoder.close()


def test_decoder_reader_reassembles_output_datagrams(tmp_path: Path):
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    sockets = [_FakeSocket(), _FakeSocket(), _FakeSocket()]
    output_socket = sockets[1]
    decoder = AscendFfmpegH264Decoder(
        width=4,
        height=2,
        frame_rate_hz=20,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=lambda *_args, **_kwargs: _FakeProcess(),
        socket_factory=lambda *_args: sockets.pop(0),
    )
    nv12 = hwc_uint8_to_nv12(np.full((2, 4, 3), 80, dtype=np.uint8)).tobytes()
    output_socket.datagrams.extend((nv12[:5], nv12[5:]))

    frames = decoder.decode(EncodedPacket(b"\x00\x00\x00\x01\x65x", 999, 123_456, keyframe=True))

    assert len(frames) == 1
    decoder.close()


def test_decoder_reset_recreates_process_and_close_is_idempotent(tmp_path: Path):
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    processes = []

    def process_factory(_command, **_kwargs):
        process = _FakeProcess(stdout=_QueuePipe())
        processes.append(process)
        return process

    decoder = AscendFfmpegH264Decoder(
        width=4,
        height=2,
        frame_rate_hz=20,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=process_factory,
    )

    decoder.reset()
    decoder.close(timeout_s=0.1)
    decoder.close(timeout_s=0.1)

    assert len(processes) == 2
    assert decoder.state is CodecLifecycleState.CLOSED
    with pytest.raises(VideoCodecError, match="closed"):
        decoder.reset()


@pytest.mark.parametrize(
    "options, message",
    [
        ({"width": 3}, "even dimensions"),
        ({"frame_rate_hz": 29.97}, "integral frame rate"),
        ({"color_range": "full"}, "limited color range"),
        ({"device_id": -1}, "cannot be negative"),
    ],
)
def test_encoder_rejects_unsupported_configuration(tmp_path: Path, options, message):
    defaults = {
        "width": 4,
        "height": 2,
        "frame_rate_hz": 20,
        "bitrate_bps": 1000,
        "gop_frames": 2,
        "ffmpeg_path": str(_executable(tmp_path / "ffmpeg")),
    }
    defaults.update(options)

    with pytest.raises(ValueError, match=message):
        AscendFfmpegH264Encoder(**defaults)


def test_process_failure_sets_failed_state_metrics_and_stderr_tail(tmp_path: Path):
    ffmpeg = _executable(tmp_path / "ffmpeg")
    fake_socket = _FakeSocket()
    process = _FakeProcess(stderr=io.BytesIO(b"ascend channel failed\n"))
    encoder = AscendFfmpegH264Encoder(
        width=4,
        height=2,
        frame_rate_hz=20,
        bitrate_bps=1000,
        gop_frames=2,
        ffmpeg_path=str(ffmpeg),
        process_factory=lambda *_args, **_kwargs: process,
        socket_factory=lambda *_args: fake_socket,
    )
    process.returncode = 1

    with pytest.raises(VideoCodecError) as error:
        encoder.encode(VideoFrame(np.zeros((2, 4, 3), dtype=np.uint8), 1, 1, 4, 2, "rgb24"))

    assert error.value.code == "process_exited"
    assert encoder.state is CodecLifecycleState.FAILED
    assert encoder.metrics.errors == 1
    encoder.close()


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="ascii")
    path.chmod(0o755)
    return path


def _private_ffmpeg(tmp_path: Path):
    prefix = tmp_path / "private"
    ffmpeg = _executable(prefix / "bin" / "ffmpeg")
    (prefix / "lib").mkdir()
    return ffmpeg, prefix, {"IBROBOT_ASCEND_FFMPEG_PREFIX": str(prefix), "LD_LIBRARY_PATH": "/system"}


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]
