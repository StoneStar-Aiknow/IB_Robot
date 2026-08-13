"""Optional Ascend H.264 codec backed by a private FFmpeg process with h264_ascend support."""

from __future__ import annotations

import os
import re
import socket
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode

import numpy as np

from inference_service.video_codec import (
    CodecCapabilities,
    CodecLifecycleState,
    CodecMetrics,
    EncodedPacket,
    VideoCodecError,
    VideoCodecRegistry,
    VideoDecoder,
    VideoEncoder,
    VideoFrame,
)
from inference_service.video_rtp import H264Depacketizer, RtpPacket
from tensormsg.converter import hwc_uint8_to_nv12, nv12_to_hwc_uint8

_BACKEND = "ascend"
_PRIVATE_FFMPEG_PATHS = (
    "/home/HwHiAiUser/ffmpeg-ascend-cann83/install/bin/ffmpeg",
    "/usr/local/Ascend/ffmpeg/bin/ffmpeg",
    "/usr/local/Ascend/ascend-toolkit/latest/tools/ffmpeg/bin/ffmpeg",
    "/opt/ascend/ffmpeg/bin/ffmpeg",
)
_STDERR_TAIL_BYTES = 8192


@dataclass(frozen=True, slots=True)
class AscendCodecProbeDiagnostic:
    """Machine-readable result of one FFmpeg/backend direction probe."""

    kind: Literal["encoder", "decoder"]
    available: bool
    code: str
    reason: str
    ffmpeg_path: str | None = None
    version: str = ""
    codec: str = "h264_ascend"


def resolve_ascend_ffmpeg(environ: Mapping[str, str] | None = None) -> str | None:
    """Resolve only the explicitly configured or known private FFmpeg binary."""
    environment = os.environ if environ is None else environ
    candidates: list[str] = []
    explicit = environment.get("IBROBOT_ASCEND_FFMPEG", "").strip()
    if explicit:
        candidates.append(explicit)
    prefix = environment.get("IBROBOT_ASCEND_FFMPEG_PREFIX", "").strip()
    if prefix:
        candidates.append(str(Path(prefix) / "bin" / "ffmpeg"))
    candidates.extend(_PRIVATE_FFMPEG_PATHS)
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def build_ascend_child_environment(
    ffmpeg_path: str,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build process-local library search paths without mutating this process."""
    child = dict(os.environ if environ is None else environ)
    if child.get("IBROBOT_ASCEND_FFMPEG_ISOLATE_ENV", "").strip() == "1":
        for name in tuple(child):
            if name.startswith("ASCEND_") or name == "TOOLCHAIN_HOME":
                child.pop(name)
    configured_prefix = child.get("IBROBOT_ASCEND_FFMPEG_PREFIX", "").strip()
    prefix = Path(configured_prefix).expanduser() if configured_prefix else Path(ffmpeg_path).resolve().parent.parent
    library_candidates = (
        prefix / "lib",
        prefix / "lib64",
        Path("/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64"),
        Path("/usr/local/Ascend/driver/lib64"),
    )
    private_dirs = [str(path) for path in library_candidates if path.is_dir()]
    inherited = child.get("LD_LIBRARY_PATH", "")
    if inherited:
        private_dirs.append(inherited)
    if private_dirs:
        child["LD_LIBRARY_PATH"] = os.pathsep.join(private_dirs)
    return child


def probe_ascend_codec_diagnostic(
    kind: Literal["encoder", "decoder"],
    *,
    environ: Mapping[str, str] | None = None,
    run: Callable[..., Any] = subprocess.run,
    timeout_s: float = 2.0,
) -> AscendCodecProbeDiagnostic:
    if kind not in {"encoder", "decoder"}:
        raise ValueError(f"unsupported codec kind {kind!r}")
    ffmpeg = resolve_ascend_ffmpeg(environ)
    if ffmpeg is None:
        return AscendCodecProbeDiagnostic(kind, False, "ffmpeg_not_found", "private Ascend FFmpeg was not found")
    child_env = build_ascend_child_environment(ffmpeg, environ)
    try:
        version_result = run(
            [ffmpeg, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
            env=child_env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return AscendCodecProbeDiagnostic(kind, False, "version_probe_failed", str(exc), ffmpeg)
    version_output = version_result.stdout or ""
    version = version_output.splitlines()[0] if version_output else ""
    if version_result.returncode != 0:
        return AscendCodecProbeDiagnostic(
            kind, False, "version_probe_failed", version or "FFmpeg exited nonzero", ffmpeg
        )
    direction_flag = "-encoders" if kind == "encoder" else "-decoders"
    try:
        codec_result = run(
            [ffmpeg, "-hide_banner", direction_flag],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
            env=child_env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return AscendCodecProbeDiagnostic(kind, False, "direction_probe_failed", str(exc), ffmpeg, version)
    output = codec_result.stdout or ""
    if codec_result.returncode != 0:
        return AscendCodecProbeDiagnostic(
            kind, False, "direction_probe_failed", output.strip() or "FFmpeg exited nonzero", ffmpeg, version
        )
    if not re.search(r"(?m)^\s*[A-Z.]{6}\s+h264_ascend(?:\s|$)", output):
        return AscendCodecProbeDiagnostic(
            kind,
            False,
            "codec_direction_missing",
            f"h264_ascend is not listed by {direction_flag}",
            ffmpeg,
            version,
        )
    return AscendCodecProbeDiagnostic(kind, True, "available", "h264_ascend is available", ffmpeg, version)


def probe_ascend_codec(kind: str) -> CodecCapabilities | None:
    if kind not in {"encoder", "decoder"}:
        raise ValueError(f"unsupported codec kind {kind!r}")
    diagnostic = probe_ascend_codec_diagnostic(kind)
    if not diagnostic.available:
        raise VideoCodecError(
            diagnostic.code,
            diagnostic.reason,
            backend=_BACKEND,
            details={
                "kind": diagnostic.kind,
                "ffmpeg_path": diagnostic.ffmpeg_path or "",
                "version": diagnostic.version,
                "codec": diagnostic.codec,
            },
        )
    return CodecCapabilities(pixel_formats=("rgb24", "bgr24"), hardware_accelerated=True)


def register_ascend_backend(registry: VideoCodecRegistry, *, priority: int = 100) -> None:
    registry.register(
        _BACKEND,
        priority=priority,
        probe=probe_ascend_codec,
        encoder_factory=AscendFfmpegH264Encoder,
        decoder_factory=AscendFfmpegH264Decoder,
    )


class _StderrTail:
    def __init__(self, pipe: Any, limit: int = _STDERR_TAIL_BYTES) -> None:
        self._pipe = pipe
        self._limit = limit
        self._data = bytearray()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._drain, name="ascend-ffmpeg-stderr", daemon=True)
        self._thread.start()

    @property
    def text(self) -> str:
        with self._lock:
            return bytes(self._data).decode("utf-8", errors="replace")

    def join(self, timeout_s: float) -> None:
        self._thread.join(max(0.0, timeout_s))

    def _drain(self) -> None:
        try:
            while True:
                chunk = self._pipe.read(1024)
                if not chunk:
                    return
                if isinstance(chunk, str):
                    chunk = chunk.encode()
                with self._lock:
                    self._data.extend(chunk)
                    del self._data[: max(0, len(self._data) - self._limit)]
        except (OSError, ValueError):
            return


class _FixedFrameReader:
    def __init__(self, pipe: Any, frame_bytes: int) -> None:
        self._pipe = pipe
        self._frame_bytes = frame_bytes
        self._frames: deque[bytes] = deque()
        self._buffer = bytearray()
        self._error: Exception | None = None
        self._eof = False
        self._condition = threading.Condition()
        self._thread = threading.Thread(target=self._read, name="ascend-ffmpeg-stdout", daemon=True)
        self._thread.start()

    def get(self, timeout_s: float) -> bytes | None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while not self._frames and self._error is None and not self._eof:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if self._frames:
                return self._frames.popleft()
            if self._error is not None:
                raise self._error
            return None

    def drain(self) -> list[bytes]:
        with self._condition:
            if self._error is not None:
                raise self._error
            frames = list(self._frames)
            self._frames.clear()
            return frames

    def join(self, timeout_s: float) -> None:
        self._thread.join(max(0.0, timeout_s))

    def _read(self) -> None:
        try:
            while True:
                chunk = self._pipe.read(min(64 * 1024, max(4096, self._frame_bytes - len(self._buffer))))
                if not chunk:
                    with self._condition:
                        self._eof = True
                        self._condition.notify_all()
                    return
                with self._condition:
                    self._buffer.extend(chunk)
                    while len(self._buffer) >= self._frame_bytes:
                        self._frames.append(bytes(self._buffer[: self._frame_bytes]))
                        del self._buffer[: self._frame_bytes]
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()


class _DatagramPipe:
    def __init__(self, udp_socket: Any) -> None:
        self._socket = udp_socket

    def read(self, _size: int) -> bytes:
        while True:
            try:
                return self._socket.recvfrom(65535)[0]
            except TimeoutError:
                continue
            except OSError:
                return b""

    def close(self) -> None:
        self._socket.close()


class _AscendProcessCodec:
    def __init__(
        self, *, ffmpeg_path: str | None, process_factory: Callable[..., Any], environ: Mapping[str, str] | None
    ) -> None:
        self._ffmpeg = ffmpeg_path or resolve_ascend_ffmpeg(environ)
        if self._ffmpeg is None:
            raise VideoCodecError("ffmpeg_not_found", "private Ascend FFmpeg was not found", backend=_BACKEND)
        self._process_factory = process_factory
        self._environ = environ
        self._process: Any = None
        self._stderr: _StderrTail | None = None
        self._state = CodecLifecycleState.CREATED
        self._metrics = CodecMetrics()

    @property
    def state(self) -> CodecLifecycleState:
        return self._state

    @property
    def metrics(self) -> CodecMetrics:
        return self._metrics

    def _spawn(self, command: list[str], **streams: Any) -> Any:
        stdin = streams.pop("stdin", subprocess.PIPE)
        try:
            process = self._process_factory(
                command,
                stdin=stdin,
                stderr=subprocess.PIPE,
                bufsize=0,
                env=build_ascend_child_environment(self._ffmpeg, self._environ),
                **streams,
            )
        except OSError as exc:
            self._state = CodecLifecycleState.FAILED
            self._metrics = replace(self._metrics, errors=self._metrics.errors + 1)
            raise VideoCodecError("process_start_failed", str(exc), backend=_BACKEND) from exc
        if (stdin is subprocess.PIPE and process.stdin is None) or process.stderr is None:
            raise VideoCodecError("process_start_failed", "FFmpeg pipes were not created", backend=_BACKEND)
        self._process = process
        self._stderr = _StderrTail(process.stderr)
        self._state = CodecLifecycleState.RUNNING
        return process

    def _write(self, payload: bytes) -> None:
        self._require_running()
        if self._process.poll() is not None:
            self._fail("process_exited", "FFmpeg exited before accepting input")
        failure: list[Exception] = []

        def write() -> None:
            try:
                written = self._process.stdin.write(payload)
                if written is not None and written != len(payload):
                    raise OSError(f"short FFmpeg stdin write: {written}/{len(payload)}")
                flush = getattr(self._process.stdin, "flush", None)
                if flush is not None:
                    flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                failure.append(exc)

        writer = threading.Thread(target=write, name="ascend-ffmpeg-stdin", daemon=True)
        writer.start()
        writer.join(self._io_timeout_s)
        if writer.is_alive():
            self._fail("process_write_timeout", "timed out writing to FFmpeg stdin")
        if failure:
            self._fail("process_write_failed", str(failure[0]), cause=failure[0])

    def _fail(self, code: str, message: str, *, recoverable: bool = True, cause: Exception | None = None) -> None:
        self._state = CodecLifecycleState.FAILED
        self._metrics = replace(self._metrics, errors=self._metrics.errors + 1)
        error = VideoCodecError(
            code,
            message,
            backend=_BACKEND,
            recoverable=recoverable,
            details={"stderr_tail": self._stderr.text if self._stderr is not None else ""},
        )
        if cause is None:
            raise error
        raise error from cause

    def _require_running(self) -> None:
        if self._state is not CodecLifecycleState.RUNNING:
            raise VideoCodecError("invalid_state", f"codec is {self._state.value}", backend=_BACKEND)

    def _require_not_closed(self) -> None:
        if self._state is CodecLifecycleState.CLOSED:
            raise VideoCodecError("invalid_state", "codec is closed", backend=_BACKEND)

    def _stop_process(self, timeout_s: float) -> None:
        if timeout_s < 0:
            raise ValueError("timeout_s cannot be negative")
        process = self._process
        if process is None:
            return
        deadline = time.monotonic() + timeout_s
        with suppress(AttributeError, OSError, ValueError):
            process.stdin.close()
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))
        if self._stderr is not None:
            self._stderr.join(max(0.0, deadline - time.monotonic()))
        self._process = None


class AscendFfmpegH264Encoder(_AscendProcessCodec, VideoEncoder):
    def __init__(
        self,
        *,
        width: int,
        height: int,
        frame_rate_hz: float,
        bitrate_bps: int,
        gop_frames: int,
        input_pixel_format: str = "rgb24",
        profile: str = "main",
        color_space: str = "bt709",
        color_range: str = "limited",
        device_id: int = 0,
        channel_id: int = 0,
        io_timeout_s: float = 1.0,
        drain_timeout_s: float = 0.05,
        ffmpeg_path: str | None = None,
        process_factory: Callable[..., Any] = subprocess.Popen,
        socket_factory: Callable[..., Any] = socket.socket,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        _validate_common(width, height, frame_rate_hz, color_space, color_range, device_id, channel_id, io_timeout_s)
        if bitrate_bps <= 0 or gop_frames <= 0:
            raise ValueError("bitrate and GOP must be positive")
        if input_pixel_format not in {"rgb24", "bgr24"}:
            raise ValueError("Ascend encoder input_pixel_format must be rgb24 or bgr24")
        if profile not in {"baseline", "main", "high"}:
            raise ValueError("Ascend encoder profile must be baseline, main, or high")
        if drain_timeout_s <= 0:
            raise ValueError("drain_timeout_s must be positive")
        super().__init__(ffmpeg_path=ffmpeg_path, process_factory=process_factory, environ=environ)
        self._width = int(width)
        self._height = int(height)
        self._fps = int(frame_rate_hz)
        self._bitrate = int(bitrate_bps)
        self._gop = int(gop_frames)
        self._input_pixel_format = input_pixel_format
        self._profile = profile
        self._device_id = int(device_id)
        self._channel_id = int(channel_id)
        self._io_timeout_s = float(io_timeout_s)
        self._drain_timeout_s = float(drain_timeout_s)
        self._socket_factory = socket_factory
        self._socket: Any = None
        self._depacketizer = H264Depacketizer()
        self._timestamps: deque[int] = deque()
        self._start()

    def _start(self) -> None:
        udp_socket = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.bind(("127.0.0.1", 0))
        udp_socket.settimeout(self._io_timeout_s)
        host, port = udp_socket.getsockname()[:2]
        query = urlencode({"pkt_size": 1200})
        command = [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "nv12",
            "-video_size",
            f"{self._width}x{self._height}",
            "-framerate",
            str(self._fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "h264_ascend",
            "-device_id",
            str(self._device_id),
            "-channel_id",
            str(self._channel_id),
            "-profile",
            str({"baseline": 0, "main": 1, "high": 2}[self._profile]),
            "-rc_mode",
            "0",
            "-gop",
            str(self._gop),
            "-frame_rate",
            str(self._fps),
            "-max_bit_rate",
            str(max(2, round(self._bitrate / 1000))),
            "-f",
            "rtp",
            f"rtp://{host}:{port}?{query}",
        ]
        try:
            self._socket = udp_socket
            self._spawn(command, stdout=subprocess.DEVNULL)
        except Exception:
            udp_socket.close()
            self._socket = None
            raise

    def encode(self, frame: VideoFrame) -> list[EncodedPacket]:
        self._require_running()
        if (frame.width, frame.height) != (self._width, self._height):
            raise VideoCodecError("invalid_frame", "frame dimensions do not match encoder", backend=_BACKEND)
        if frame.pixel_format != self._input_pixel_format:
            raise VideoCodecError("invalid_pixel_format", f"expected {self._input_pixel_format}", backend=_BACKEND)
        if frame.color_space.lower() != "bt709" or frame.color_range.lower() != "limited":
            raise VideoCodecError("unsupported_color", "Ascend H.264 requires limited-range BT.709", backend=_BACKEND)
        array = np.asarray(frame.data)
        if array.shape != (self._height, self._width, 3) or array.dtype != np.uint8:
            raise VideoCodecError("invalid_frame", "expected a uint8 HWC color frame", backend=_BACKEND)
        try:
            nv12 = hwc_uint8_to_nv12(
                array,
                encoding="rgb8" if self._input_pixel_format == "rgb24" else "bgr8",
                color_range="limited",
            )
            self._write(nv12.tobytes())
            self._timestamps.append(frame.capture_timestamp_ns)
            packets = self._drain_access_units(self._drain_timeout_s)
        except VideoCodecError:
            raise
        except Exception as exc:
            self._fail("encode_failed", str(exc), cause=exc)
        self._metrics = replace(
            self._metrics,
            input_frames=self._metrics.input_frames + 1,
            output_frames=self._metrics.output_frames + len(packets),
            output_packets=self._metrics.output_packets + len(packets),
        )
        return packets

    def _drain_access_units(self, timeout_s: float) -> list[EncodedPacket]:
        """Collect encoded access units within *timeout_s*.

        The Ascend DVPP hardware encoder has an internal pipeline delay:
        output for frame N typically arrives only after frame N+1 has been
        submitted.  This method uses a short non-blocking drain so that the
        caller can continue feeding frames without deadlocking.  When no
        output is ready (e.g. the very first frame), an empty list is
        returned and the delayed output surfaces on the next call.
        """
        packets: list[EncodedPacket] = []
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._socket.settimeout(remaining)
            try:
                datagram = self._socket.recvfrom(65535)[0]
                access_unit, _lost = self._depacketizer.push(RtpPacket.from_bytes(datagram))
            except TimeoutError:
                break
            except (OSError, ValueError) as exc:
                self._fail("invalid_rtp", str(exc), cause=exc)
            if access_unit is not None:
                if not self._timestamps:
                    self._fail("timestamp_underflow", "encoded output has no input timestamp")
                capture_timestamp_ns = self._timestamps.popleft()
                packets.append(
                    EncodedPacket(
                        access_unit.payload,
                        access_unit.timestamp,
                        capture_timestamp_ns,
                        keyframe=access_unit.keyframe,
                    )
                )
        return packets

    def reset(self) -> None:
        self._require_not_closed()
        with suppress(Exception):
            self._drain_access_units(self._drain_timeout_s)
        self._stop_process(self._io_timeout_s)
        if self._socket is not None:
            self._socket.close()
        self._timestamps.clear()
        self._depacketizer.reset()
        self._state = CodecLifecycleState.CREATED
        self._start()

    def close(self, timeout_s: float = 1.0) -> None:
        if self._state is CodecLifecycleState.CLOSED:
            return
        self._stop_process(timeout_s)
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._timestamps.clear()
        self._state = CodecLifecycleState.CLOSED


class AscendFfmpegH264Decoder(_AscendProcessCodec, VideoDecoder):
    def __init__(
        self,
        *,
        width: int,
        height: int,
        frame_rate_hz: float,
        output_pixel_format: str = "rgb24",
        color_space: str = "bt709",
        color_range: str = "limited",
        device_id: int = 0,
        channel_id: int = 0,
        io_timeout_s: float = 1.0,
        ffmpeg_path: str | None = None,
        process_factory: Callable[..., Any] = subprocess.Popen,
        socket_factory: Callable[..., Any] = socket.socket,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        _validate_common(width, height, frame_rate_hz, color_space, color_range, device_id, channel_id, io_timeout_s)
        if output_pixel_format not in {"rgb24", "bgr24"}:
            raise ValueError("Ascend decoder output_pixel_format must be rgb24 or bgr24")
        super().__init__(ffmpeg_path=ffmpeg_path, process_factory=process_factory, environ=environ)
        self._width = int(width)
        self._height = int(height)
        self._fps = int(frame_rate_hz)
        self._output_pixel_format = output_pixel_format
        self._device_id = int(device_id)
        self._channel_id = int(channel_id)
        self._io_timeout_s = float(io_timeout_s)
        self._output_wait_s = min(self._io_timeout_s, 0.5 / frame_rate_hz)
        self._socket_factory = socket_factory
        self._socket: Any = None
        self._output_pipe: _DatagramPipe | None = None
        self._input_endpoint: tuple[str, int] | None = None
        self._frame_metadata: deque[tuple[int, bool]] = deque()
        self._reader: _FixedFrameReader | None = None
        self._start()

    def _start(self) -> None:
        udp_socket = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.bind(("127.0.0.1", 0))
        output_socket = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        output_socket.bind(("127.0.0.1", 0))
        output_socket.settimeout(0.1)
        output_host, output_port = output_socket.getsockname()[:2]
        output_pipe = _DatagramPipe(output_socket)
        reservation = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        reservation.bind(("127.0.0.1", 0))
        host, port = reservation.getsockname()[:2]
        reservation.close()
        self._input_endpoint = (host, port)
        command = [
            self._ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-fflags",
            "nobuffer",
            "-analyzeduration",
            "0",
            "-probesize",
            "32",
            "-f",
            "h264",
            "-c:v",
            "h264_ascend",
            "-device_id",
            str(self._device_id),
            "-channel_id",
            str(self._channel_id),
            "-i",
            f"udp://{host}:{port}?fifo_size=1048576&overrun_nonfatal=1",
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "nv12",
            "-vsync",
            "0",
            f"udp://{output_host}:{output_port}?pkt_size=60000",
        ]
        try:
            self._socket = udp_socket
            self._output_pipe = output_pipe
            self._spawn(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        except Exception:
            udp_socket.close()
            output_pipe.close()
            self._socket = None
            self._output_pipe = None
            raise
        self._reader = _FixedFrameReader(output_pipe, self._width * self._height * 3 // 2)

    def decode(self, packet: EncodedPacket) -> list[VideoFrame]:
        self._require_running()
        self._frame_metadata.append((packet.capture_timestamp_ns, packet.keyframe))
        try:
            assert self._socket is not None
            assert self._input_endpoint is not None
            for offset in range(0, len(packet.payload), 1200):
                self._socket.sendto(packet.payload[offset : offset + 1200], self._input_endpoint)
            assert self._reader is not None
            frames = self._reader.drain()
            if not frames:
                first_frame = self._reader.get(self._output_wait_s)
                if first_frame is not None:
                    frames = [first_frame, *self._reader.drain()]
        except VideoCodecError:
            raise
        except Exception as exc:
            self._fail("decode_failed", str(exc), cause=exc)
        decoded = [self._convert_frame(nv12) for nv12 in frames]
        self._metrics = replace(
            self._metrics,
            input_frames=self._metrics.input_frames + 1,
            output_frames=self._metrics.output_frames + len(decoded),
            output_packets=self._metrics.output_packets + 1,
        )
        return decoded

    def _convert_frame(self, nv12: bytes) -> VideoFrame:
        if not self._frame_metadata:
            self._fail("timestamp_underflow", "decoded output has no input timestamp")
        capture_timestamp_ns, keyframe = self._frame_metadata.popleft()
        image = nv12_to_hwc_uint8(
            nv12,
            width=self._width,
            height=self._height,
            output_encoding="rgb8" if self._output_pixel_format == "rgb24" else "bgr8",
            color_range="limited",
        )
        return VideoFrame(
            image,
            capture_timestamp_ns,
            capture_timestamp_ns,
            self._width,
            self._height,
            self._output_pixel_format,
            color_space="bt709",
            color_range="limited",
            keyframe=keyframe,
        )

    def reset(self) -> None:
        self._require_not_closed()
        self._stop_process(self._io_timeout_s)
        if self._output_pipe is not None:
            self._output_pipe.close()
            self._output_pipe = None
        if self._reader is not None:
            self._reader.join(self._io_timeout_s)
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._input_endpoint = None
        self._frame_metadata.clear()
        self._state = CodecLifecycleState.CREATED
        self._start()

    def close(self, timeout_s: float = 1.0) -> None:
        if self._state is CodecLifecycleState.CLOSED:
            return
        deadline = time.monotonic() + max(0.0, timeout_s)
        self._stop_process(timeout_s)
        if self._output_pipe is not None:
            self._output_pipe.close()
            self._output_pipe = None
        if self._reader is not None:
            self._reader.join(max(0.0, deadline - time.monotonic()))
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._input_endpoint = None
        self._frame_metadata.clear()
        self._state = CodecLifecycleState.CLOSED


def _validate_common(
    width: int,
    height: int,
    frame_rate_hz: float,
    color_space: str,
    color_range: str,
    device_id: int,
    channel_id: int,
    io_timeout_s: float,
) -> None:
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("Ascend H.264 requires positive even dimensions")
    if frame_rate_hz <= 0 or not float(frame_rate_hz).is_integer():
        raise ValueError("Ascend H.264 requires a positive integral frame rate")
    if color_space.lower() != "bt709":
        raise ValueError("Ascend H.264 supports BT.709 color space only")
    if color_range.lower() != "limited":
        raise ValueError("Ascend H.264 supports limited color range only")
    if device_id < 0 or channel_id < 0:
        raise ValueError("device_id and channel_id cannot be negative")
    if io_timeout_s <= 0:
        raise ValueError("io_timeout_s must be positive")
