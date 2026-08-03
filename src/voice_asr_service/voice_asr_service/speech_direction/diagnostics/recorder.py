"""高通量离线维测记录器。"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .gray_events import GrayEventTracker
from .manifest import SessionManifest
from .writers import ClosedVolume, RollingAudioWriter, RollingMetricsWriter


@dataclass(frozen=True)
class FrameMetrics:
    """单帧维测指标，时间轴统一使用会话整数采样位置。"""

    session_sample: int
    sample_count: int
    segment_seq: int
    vad_probability: float
    is_speech: bool
    is_gray: bool
    rms: float
    frame_doa_degree: int | None
    segment_doa_degree: int | None
    inference_elapsed_ms: float


@dataclass(frozen=True)
class DiagnosticsPacket:
    """一次后台写入所需的音频块和可选帧指标。"""

    raw_start_sample: int
    raw6ch: np.ndarray
    enh_start_sample: int | None
    enh4ch: np.ndarray | None
    metrics: FrameMetrics | None


@dataclass(frozen=True)
class RecorderStatus:
    """记录器对调用方公开的不可变状态快照。"""

    enabled: bool
    state: str
    disabled_reason: str | None
    disabled_at_sample: int | None
    dropped_count: int


DisabledCallback = Callable[[RecorderStatus], None]


class DiagnosticsRecorder:
    """通过有界队列把维测写盘与实时调用线程隔离。"""

    def __init__(
        self,
        session_dir: str | Path,
        *,
        session_id: str,
        sample_rate: int,
        rollover_seconds: int,
        save_raw6ch: bool,
        save_enh4ch: bool,
        save_frame_metrics: bool,
        save_gray_events: bool,
        queue_size: int = 128,
        drop_when_full: bool = True,
        on_disabled: DisabledCallback | None = None,
        gray_merge_gap_samples: int = 0,
        stop_timeout_seconds: float = 2.0,
    ) -> None:
        self._session_dir = Path(session_dir)
        self._session_id = session_id
        self._sample_rate = sample_rate
        self._rollover_seconds = rollover_seconds
        self._rollover_samples = sample_rate * rollover_seconds
        self._save_raw6ch = save_raw6ch
        self._save_enh4ch = save_enh4ch
        self._save_frame_metrics = save_frame_metrics
        self._save_gray_events = save_gray_events
        # 参数仅保留配置接口兼容；实时维测旁路无论取值都禁止阻塞生产线程。
        del drop_when_full
        self._on_disabled = on_disabled
        self._gray_merge_gap_samples = gray_merge_gap_samples
        self._stop_timeout_seconds = max(0.0, float(stop_timeout_seconds))
        self._queue: queue.Queue[DiagnosticsPacket] = queue.Queue(maxsize=queue_size)
        # 同一 Condition 定义 start/enqueue/stop 的线性化边界，STOP 发布后绝不再接收包。
        self._condition = threading.Condition(threading.RLock())
        self._status = RecorderStatus(True, "not_started", None, None, 0)
        self._manifest: SessionManifest | None = None
        self._writers: list[object] = []
        self._raw_writer: RollingAudioWriter | None = None
        self._enh_writer: RollingAudioWriter | None = None
        self._metrics_writer: RollingMetricsWriter | None = None
        self._gray_tracker: GrayEventTracker | None = None
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._cleanup_in_progress = False
        self._cleanup_done = False
        self._cleanup_owner_id: int | None = None
        self._stop_deadline: float | None = None

    @property
    def status(self) -> RecorderStatus:
        """返回线程安全的不可变状态快照。"""
        with self._condition:
            return self._status

    def start(self) -> None:
        """创建 Manifest、按独立开关创建 writer，并启动后台线程。"""
        # 初始化全程持有生命周期锁，并发 start 只能由一个调用者取得所有权。
        with self._condition:
            if self._status.state != "not_started" or self._stop_requested:
                return
            self._status = replace(self._status, state="starting")
            try:
                self._initialize_resources()
                thread = threading.Thread(
                    target=self._run,
                    name="DiagnosticsWriter",
                    daemon=True,
                )
                self._thread = thread
                thread.start()
            except BaseException as error:
                self._cleanup_in_progress = True
                self._cleanup_owner_id = threading.get_ident()
                self._disable(error, None)
                self._finish_cleanup()
                return
            self._status = replace(self._status, state="recording")
            self._condition.notify_all()

    def enqueue(self, packet: DiagnosticsPacket) -> bool:
        """非阻塞入队；队列满时返回 False，并只累计丢包数。"""
        with self._condition:
            if self._status.state != "recording" or self._stop_requested:
                return False
            try:
                self._queue.put_nowait(packet)
                return True
            except queue.Full:
                # 实时路径绝不能因维测磁盘吞吐不足而无限等待。
                self._status = replace(self._status, dropped_count=self._status.dropped_count + 1)
                return False

    def stop(self) -> None:
        """有限等待排空既有包，尽力关闭所有 writer，并保持并发幂等。"""
        current = threading.current_thread()
        current_id = threading.get_ident()
        with self._condition:
            if self._cleanup_done:
                return
            self._stop_requested = True
            if self._stop_deadline is None:
                self._stop_deadline = time.monotonic() + self._stop_timeout_seconds
            deadline = self._stop_deadline
            if self._status.state == "not_started":
                self._status = replace(self._status, enabled=False, state="completed")
                self._cleanup_done = True
                self._condition.notify_all()
                return
            # callback 运行在 worker 时只发布 STOP；由 worker finally 在安全点清理。
            if self._thread is current:
                return
            if self._cleanup_in_progress:
                if self._cleanup_owner_id == current_id:
                    return
                while not self._cleanup_done:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return
                    self._condition.wait(timeout=remaining)
                return
            self._cleanup_in_progress = True
            self._cleanup_owner_id = current_id
            thread = self._thread

        remaining = max(0.0, deadline - time.monotonic())
        if thread is not None and thread.is_alive():
            thread.join(timeout=remaining)
        if thread is not None and thread.is_alive():
            self._disable(TimeoutError("diagnostics stop timeout"), None)
            # 超时兜底：调用线程同步刷 manifest 终态，不依赖可能被进程退出强杀的
            # daemon worker。只读 jsonl 行数 + 写 manifest，不碰 writer，避免与仍在
            # write 的 worker 并发；worker 若之后能跑完 _finish_cleanup 会覆盖成更全的版本。
            self._flush_manifest_terminal()
            # 不与仍在 write 的 worker 并发 close；worker 退出后会接管最终清理。
            with self._condition:
                self._cleanup_in_progress = False
                self._cleanup_owner_id = None
                self._condition.notify_all()
            return
        if thread is not None:
            # join 返回说明 worker 的 finally（含清理）已结束，不能重复 close。
            with self._condition:
                while not self._cleanup_done:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return
                    self._condition.wait(timeout=remaining)
            return
        self._finish_cleanup()

    def _initialize_resources(self) -> None:
        manifest = SessionManifest(
            self._session_dir,
            session_id=self._session_id,
            sample_rate=self._sample_rate,
            rollover_seconds=self._rollover_seconds,
            save_raw6ch=self._save_raw6ch,
            save_enh4ch=self._save_enh4ch,
            save_frame_metrics=self._save_frame_metrics,
            save_gray_events=self._save_gray_events,
        )
        self._manifest = manifest
        full_dir = self._session_dir / "audio" / "full"
        if self._save_raw6ch:
            self._raw_writer = RollingAudioWriter(
                full_dir,
                stream="raw6ch",
                sample_rate=self._sample_rate,
                channels=6,
                rollover_samples=self._rollover_samples,
                on_closed=self._on_volume_closed,
            )
            self._writers.append(self._raw_writer)
        if self._save_enh4ch:
            self._enh_writer = RollingAudioWriter(
                full_dir,
                stream="enh4ch",
                sample_rate=self._sample_rate,
                channels=4,
                rollover_samples=self._rollover_samples,
                on_closed=self._on_volume_closed,
            )
            self._writers.append(self._enh_writer)
        if self._save_frame_metrics:
            self._metrics_writer = RollingMetricsWriter(
                self._session_dir / "metrics",
                sample_rate=self._sample_rate,
                rollover_samples=self._rollover_samples,
                on_closed=self._on_volume_closed,
            )
            self._writers.append(self._metrics_writer)
        if self._save_gray_events:
            self._gray_tracker = GrayEventTracker(
                self._session_dir,
                sample_rate=self._sample_rate,
                merge_gap_samples=self._gray_merge_gap_samples,
            )
            self._writers.append(self._gray_tracker)

    def _run(self) -> None:
        """后台顺序消费；STOP 后排空已接受包，首个故障后不再调用 writer。"""
        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.05)
                except queue.Empty:
                    with self._condition:
                        if self._stop_requested:
                            return
                    continue
                try:
                    if self.status.state != "diagnostics_disabled":
                        try:
                            self._write_packet(item)
                        except BaseException as error:
                            self._disable(error, self._packet_sample(item))
                finally:
                    self._queue.task_done()
                with self._condition:
                    if self._stop_requested and self._queue.empty():
                        return
        finally:
            with self._condition:
                should_cleanup = self._stop_requested and not self._cleanup_done
                if should_cleanup:
                    self._cleanup_in_progress = True
                    self._cleanup_owner_id = threading.get_ident()
            if should_cleanup:
                self._finish_cleanup()

    def _write_packet(self, packet: DiagnosticsPacket) -> None:
        """按 raw、enh、metrics、gray 四条独立开关路径顺序写入一个包。"""
        # 四路共享同一 packet，但任一路异常都交给 worker 统一永久禁用，避免部分失败后续写。
        if self._raw_writer is not None:
            self._raw_writer.write(
                start_sample=packet.raw_start_sample,
                samples=packet.raw6ch,
            )
        if self._enh_writer is not None and packet.enh4ch is not None:
            if packet.enh_start_sample is None:
                raise ValueError("enh4ch 存在时 enh_start_sample 不能为空")
            self._enh_writer.write(
                start_sample=packet.enh_start_sample,
                samples=packet.enh4ch,
            )
        metrics = packet.metrics
        if metrics is not None and self._metrics_writer is not None:
            self._metrics_writer.write(asdict(metrics))
        if metrics is not None and self._gray_tracker is not None:
            self._gray_tracker.observe(
                session_sample=metrics.session_sample,
                sample_count=metrics.sample_count,
                is_gray=metrics.is_gray,
                vad_probability=metrics.vad_probability,
                rms=metrics.rms,
                frame_doa_degree=metrics.frame_doa_degree,
            )

    def _finish_cleanup(self) -> None:
        """仅在 worker 停止后执行一次最终清理，并发布最终 Manifest 计数。"""
        try:
            for writer in self._writers:
                try:
                    writer.close()
                except BaseException as error:
                    self._disable(error, None)

            manifest = self._manifest
            if manifest is None:
                return
            try:
                gray_count = self._gray_event_count() if self._gray_tracker is not None else 0
            except BaseException as error:
                gray_count = 0
                self._disable(error, None)

            try:
                manifest.set_gray_event_count(gray_count)
                status = self.status
                if status.state == "diagnostics_disabled":
                    manifest.mark_disabled(
                        reason=status.disabled_reason or "diagnostics disabled",
                        disabled_session_sample=status.disabled_at_sample,
                        dropped_count=status.dropped_count,
                    )
                else:
                    manifest.complete(dropped_count=status.dropped_count)
            except BaseException as error:
                # complete 可能在成功提交后才抛异常；公开 snapshot 是终态判据。
                if manifest.snapshot()["status"] == "completed":
                    with self._condition:
                        self._status = replace(
                            self._status,
                            enabled=False,
                            state="completed",
                            disabled_reason=None,
                            disabled_at_sample=None,
                        )
                else:
                    self._disable(error, None)
                    with contextlib.suppress(BaseException):
                        status = self.status
                        manifest.set_gray_event_count(gray_count)
                        manifest.mark_disabled(
                            reason=status.disabled_reason or str(error),
                            disabled_session_sample=status.disabled_at_sample,
                            dropped_count=status.dropped_count,
                        )
        except BaseException as error:
            # 清理路径的任何意外异常都只能禁用维测，不能遗留永远等待的 waiter。
            self._disable(error, None)
        finally:
            with self._condition:
                if self._status.state != "diagnostics_disabled":
                    self._status = replace(self._status, enabled=False, state="completed")
                self._cleanup_done = True
                self._cleanup_in_progress = False
                self._cleanup_owner_id = None
                self._condition.notify_all()

    def _flush_manifest_terminal(self) -> None:
        """stop() 超时兜底：调用线程同步把 manifest 刷成终态。

        不关闭 writer、不排空队列，避免与仍在 write 的 worker 并发写文件。
        只读 gray_events.jsonl 行数并写 manifest；worker 若后续跑完
        _finish_cleanup 会用更全的版本（含已关闭的 volume）再次覆盖。
        """
        manifest = self._manifest
        if manifest is None:
            return
        with self._condition:
            if self._cleanup_done:
                # worker 已完成清理，manifest 已是终态，无需重复刷。
                return
            try:
                gray_count = self._gray_event_count() if self._gray_tracker is not None else 0
            except BaseException:
                gray_count = 0
            status = self.status
        # set_gray_event_count / mark_disabled / complete 内部各自 _commit，幂等可重入
        try:
            manifest.set_gray_event_count(gray_count)
            if status.state == "diagnostics_disabled":
                manifest.mark_disabled(
                    reason=status.disabled_reason or "diagnostics stop timeout",
                    disabled_session_sample=status.disabled_at_sample,
                    dropped_count=status.dropped_count,
                )
            else:
                manifest.complete(dropped_count=status.dropped_count)
        except BaseException:
            # 尽力而为，刷盘失败不应阻塞进程退出路径。
            pass

    def _on_volume_closed(self, volume: ClosedVolume) -> None:
        """把 writer 的文件名映射到 Manifest 所需的会话相对路径。"""
        manifest = self._manifest
        if manifest is None:
            raise RuntimeError("Manifest 尚未创建")
        directory = "metrics" if volume.stream == "frame_metrics" else "audio/full"
        manifest.add_closed_volume(
            volume.stream,
            path=f"{directory}/{volume.path}",
            index=volume.index,
            start_sample=volume.start_sample,
            end_sample=volume.end_sample,
            frame_count=volume.frame_count,
            wav_format=volume.wav_format,
        )

    def _gray_event_count(self) -> int:
        path = self._session_dir / "events" / "gray_events.jsonl"
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as event_file:
            return sum(1 for line in event_file if line.strip())

    @staticmethod
    def _packet_sample(packet: DiagnosticsPacket) -> int:
        if packet.metrics is not None:
            return packet.metrics.session_sample
        return packet.raw_start_sample

    def _disable(self, error: BaseException, sample: int | None) -> None:
        """只通知一次永久停用；最终 Manifest 由安全清理路径统一持久化。"""
        reason = str(error) or type(error).__name__
        with self._condition:
            if self._status.state == "diagnostics_disabled":
                return
            disabled = RecorderStatus(
                enabled=False,
                state="diagnostics_disabled",
                disabled_reason=reason,
                disabled_at_sample=sample,
                dropped_count=self._status.dropped_count,
            )
            self._status = disabled
            self._condition.notify_all()
        if self._on_disabled is not None:
            with contextlib.suppress(BaseException):
                self._on_disabled(disabled)


__all__ = [
    "DiagnosticsPacket",
    "DiagnosticsRecorder",
    "FrameMetrics",
    "RecorderStatus",
]
