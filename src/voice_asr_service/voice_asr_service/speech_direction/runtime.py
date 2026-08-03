"""音频采集 + worker 线程生命周期管理。

speech_direction 链路所需的采集与 worker 管理:多通道环形缓冲、PyAudio blocking
采集、worker 线程生命周期与共享状态。

设计要点:
    1. MultiChannelRingBuffer(多消费者):PyAudio 采集线程写入 6ch,worker 独立 reader。
    2. AudioCapture:专用线程 blocking read,兼容 Ubuntu 22.04 的 PyAudio 0.2.11。
    3. SpeechDirectionRuntime:启动采集 + worker 线程,管理共享状态,优雅停止。
    4. feed_audio:离线模式外部灌入 6ch 音频(无设备时),pipeline 与实时同路径。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pyaudio

    _HAS_PYAUDIO = True
except ImportError:
    pyaudio = None
    _HAS_PYAUDIO = False


class MultiChannelRingBuffer:
    """多消费者环形缓冲区(采用绝对计数法)。

    线程安全地缓存采集到的多通道音频,供多个消费者各自独立按帧读取。
    写入接口吃原始 bytes(PyAudio input stream 返回的 interleaved int16 bytes),
    内部解码为 float32 并存成 (channels, capacity) 的环形数组。

    关键设计(绝对计数法):write_pos / reader_pos 用**单调递增的绝对计数**,
    只在访问 _buf 时对 capacity 取模。这样:
      - available = write_pos - reader_pos(无需取模,天然正确)
      - 缓冲满时丢弃最旧,把落后 reader 推到 write_pos - capacity

    读策略:
        - "block"  :不足时阻塞等待,直到凑够 n 帧(实时处理线程)
        - "latest" :不足时返回当前全部可读帧(预取/刷新场景)
    """

    def __init__(self, capacity_frames: int, channels: int = 6, dtype=np.float32):
        """
        Args:
            capacity_frames: 缓冲容量(帧数,1 帧 = channels 个采样点)
            channels: 通道数
            dtype: 存储精度(float32 归一化到 [-1, 1])
        """
        self.channels = channels
        self.capacity = int(capacity_frames)
        self.dtype = dtype
        # 环形存储:(channels, capacity)
        self._buf = np.zeros((channels, self.capacity), dtype=dtype)
        self._write_pos = 0  # 已写入的绝对帧数(单调递增,访问 _buf 时 % capacity)
        self._reader_pos: dict[int, int] = {}
        self._next_reader_id = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def write(self, data) -> None:
        """写入音频数据。

        data: interleaved int16 bytes,或 (n_frames, channels)/(channels, n_frames) 数组。
        单次写入超过容量时,只保留最后 capacity 帧(实时系统优先保新)。
        """
        with self._cond:
            frames = self._decode(data)  # (n_frames, channels) float32
            n = frames.shape[0]
            if n == 0:
                return
            # 单次写入超过容量:只保留最后 capacity 帧
            if n > self.capacity:
                frames = frames[-self.capacity :]
                n = self.capacity
            start = self._write_pos % self.capacity
            end = start + n
            if end <= self.capacity:
                # 不回绕:一次写入(frames.T 形状 (channels, n))
                self._buf[:, start:end] = frames.T
            else:
                # 回绕:分两段写
                first = self.capacity - start
                self._buf[:, start:] = frames[:first].T
                second = n - first
                self._buf[:, :second] = frames[first:].T

            self._write_pos += n

            # 容量溢出:若写入量超过容量,丢弃最旧数据,把落后 reader 拉到合法下界
            low = self._write_pos - self.capacity
            for rid in list(self._reader_pos):
                if self._reader_pos[rid] < low:
                    self._reader_pos[rid] = low

            self._cond.notify_all()

    def _decode(self, data) -> np.ndarray:
        """bytes/int16/float → (n_frames, channels) float32,范围 [-1, 1]。"""
        if isinstance(data, bytes | bytearray):
            arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            n_total = arr.shape[0]
            if n_total % self.channels != 0:
                # 不完整帧:丢弃尾部不足一帧的样本,避免 reshape 报错
                arr = arr[: (n_total // self.channels) * self.channels]
            return arr.reshape(-1, self.channels)  # (n_frames, channels)
        # 已是 numpy 数组(或可转换为数组)
        # int16 整数数组需要除 32768 归一化到 [-1, 1](与 bytes 路径一致);
        # float 数组假定已在 [-1, 1] 区间,不再重复归一化,避免改变幅度。
        arr_raw = np.asarray(data)
        if arr_raw.dtype == np.int16:
            arr = arr_raw.astype(np.float32) / 32768.0
        else:
            arr = arr_raw.astype(self.dtype)
        if arr.ndim == 1:
            # 单通道数据,扩展为 (n, 1)
            return arr.reshape(-1, 1)
        if arr.ndim == 2:
            # (channels, n_frames) → 转置为 (n_frames, channels)
            if arr.shape[0] == self.channels and arr.shape[1] != self.channels:
                return arr.T
            return arr
        raise ValueError(f"不支持的数据维度: {arr.shape}")

    def register(self, start_latest: bool = True) -> int:
        """注册一个 reader,返回 reader_id。

        Args:
            start_latest: True=从最新写入位置开始读(丢弃历史);
                       False=从最早可读位置开始读(尽量不丢)
        """
        with self._cond:
            rid = self._next_reader_id
            self._next_reader_id += 1
            if start_latest:
                self._reader_pos[rid] = self._write_pos
            else:
                low = min(self._reader_pos.values()) if self._reader_pos else self._write_pos
                self._reader_pos[rid] = max(low, self._write_pos - self.capacity)
            return rid

    def read(self, reader_id: int, n_frames: int, read_mode: str = "block", timeout: float = 1.0):
        """读取 n_frames 帧。

        Returns:
            (channels, n_frames) float32;不足且 block 超时返回 None
        """
        deadline = time.monotonic() + timeout if read_mode == "block" else 0.0
        with self._cond:
            while True:
                avail = self._write_pos - self._reader_pos.get(reader_id, 0)
                if avail >= n_frames:
                    return self._read_n(reader_id, n_frames)
                if read_mode == "latest":
                    return self._read_n(reader_id, avail) if avail > 0 else None
                # block:等待新数据
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(timeout=remaining)

    def _read_n(self, reader_id: int, n: int) -> np.ndarray | None:
        """从 reader_pos 读 n 帧(可能回绕),返回 (channels, n)。"""
        if n <= 0:
            return None
        if n > self.capacity:
            n = self.capacity
        rpos = self._reader_pos.get(reader_id, 0)
        out = np.zeros((self.channels, n), dtype=self.dtype)
        start = rpos % self.capacity
        end = start + n
        if end <= self.capacity:
            out[:] = self._buf[:, start:end]
        else:
            # 回绕:分两段读
            first = self.capacity - start
            out[:, :first] = self._buf[:, start:]
            second = n - first
            out[:, first:] = self._buf[:, :second]
        self._reader_pos[reader_id] = rpos + n
        return out

    def total_written(self) -> int:
        """累计写入帧数(供诊断)。"""
        with self._lock:
            return self._write_pos


class AudioCapture:
    """ReSpeaker 多通道 blocking 采集器。

    设备查找内联在 _find_device(优先设备号,回退按名称搜索)。
    blocking read 在专用 Python 线程中执行，避免 PyAudio 0.2.11 callback
    C bridge 将异常异步注入 ROS executor。
    """

    def __init__(
        self,
        config,
        ring_buffer: MultiChannelRingBuffer,
        on_fatal_error: Callable[[str], None] | None = None,
    ):
        if not _HAS_PYAUDIO:
            raise ImportError("未安装 pyaudio。请先安装系统依赖 portaudio19-dev,再 `pip install pyaudio`。")

        self.config = config
        self.ring_buffer = ring_buffer
        self.sample_rate = config.audio.sample_rate
        self.channels = config.audio.channels
        self.chunk_size = config.audio.chunk_size
        self.device_index_cfg = config.audio.device_index
        self.device_name = config.audio.device_name
        self._on_fatal_error = on_fatal_error

        self._pa = None
        self._stream = None
        self._capture_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._frames_captured = 0
        self._overflow_count = 0
        self._running = False
        self._cleanup_pending = False

    def start(self):
        """启动 blocking 采集流和专用读取线程。"""
        with self._lock:
            if self._cleanup_pending:
                raise RuntimeError("音频采集存在未清理资源，请先调用 stop()")
            if self._running:
                return
            if self._stream is not None or self._capture_thread is not None or self._pa is not None:
                raise RuntimeError("音频采集存在未清理资源，请先调用 stop()")

            pa = pyaudio.PyAudio()
            self._pa = pa
            stream = None
            try:
                device_index = self._find_device()
                logger.info(
                    "* capturing: %dch @ %dHz, blocking mode",
                    self.channels,
                    self.sample_rate,
                )
                stream = pa.open(
                    rate=self.sample_rate,
                    format=pyaudio.get_format_from_width(2),
                    channels=self.channels,
                    input=True,
                    output=False,
                    input_device_index=device_index,
                    frames_per_buffer=self.chunk_size,
                )
                stream.start_stream()
            except Exception:
                # 启动事务失败时完整回滚；清理异常只记录，不能覆盖原始启动异常。
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        logger.warning("回滚音频流失败", exc_info=True)
                try:
                    pa.terminate()
                except Exception:
                    logger.warning("回滚 PyAudio 失败", exc_info=True)
                self._pa = None
                raise

            self._stream = stream
            self._stop_event.clear()
            # 必须在线程启动前置为运行态，避免快速读取线程观察到旧状态。
            self._running = True
            try:
                thread = threading.Thread(
                    target=self._capture_loop,
                    name="speech-direction-audio-capture",
                    daemon=True,
                )
                self._capture_thread = thread
                thread.start()
            except Exception:
                # Python 线程未成功启动，回滚已启动的 PortAudio 生命周期。
                self._stop_event.set()
                self._stream = None
                self._capture_thread = None
                self._pa = None
                self._running = False
                try:
                    stream.stop_stream()
                except Exception:
                    logger.warning("回滚音频流 stop 失败", exc_info=True)
                try:
                    stream.close()
                except Exception:
                    logger.warning("回滚音频流 close 失败", exc_info=True)
                try:
                    pa.terminate()
                except Exception:
                    logger.warning("回滚 PyAudio 失败", exc_info=True)
                raise

    def _capture_loop(self) -> None:
        """持续读取多通道 PCM；异常只通过受控回调上报。"""
        try:
            while not self._stop_event.is_set():
                with self._lock:
                    stream = self._stream
                if stream is None:
                    return

                # blocking read 不持有生命周期锁，确保 stop() 能关闭流解除阻塞。
                data = stream.read(
                    self.chunk_size,
                    exception_on_overflow=False,
                )
                if self._stop_event.is_set():
                    return
                self.ring_buffer.write(data)
                self._frames_captured += len(data) // (2 * self.channels)
        except Exception as exc:
            # stop_stream/close 用于解除阻塞，主动停止产生的异常不属于设备故障。
            if not self._stop_event.is_set() and self._on_fatal_error is not None:
                try:
                    self._on_fatal_error(f"音频采集读取失败: {exc}")
                except Exception:
                    # 外部 fatal 处理器不得击穿采集线程边界。
                    logger.exception("音频采集 fatal 回调异常")
        finally:
            with self._lock:
                self._running = False

    def _find_device(self) -> int:
        """查找 ReSpeaker 输入设备索引(优先设备号,回退按名称搜索)。"""
        p = self._pa
        # 第一步:优先用设备号直接匹配
        if (
            self.device_index_cfg is not None
            and self.device_index_cfg >= 0
            and self.device_index_cfg < p.get_device_count()
        ):
            info = p.get_device_info_by_index(self.device_index_cfg)
            if self.device_name in info.get("name", "") and info.get("maxInputChannels", 0) >= self.channels:
                logger.info("use input device [%d]: %s", self.device_index_cfg, info.get("name"))
                return self.device_index_cfg

        # 第二步:设备号无效,回退为按设备名搜索
        logger.info("设备号 %s 无效或不是 ReSpeaker,回退为按设备名搜索...", self.device_index_cfg)
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if self.device_name in info.get("name", "") and info.get("maxInputChannels", 0) >= self.channels:
                logger.info("use input device [%d]: %s", i, info.get("name"))
                return i

        raise RuntimeError("未找到 ReSpeaker 输入设备,请检查 USB 连接。")

    def stop(self):
        """幂等停止采集线程并释放 PyAudio 资源。"""
        with self._lock:
            stream = self._stream
            thread = self._capture_thread
            pa = self._pa
            if stream is None and thread is None and pa is None:
                return
            self._stop_event.set()
            self._stream = None
            self._capture_thread = None
            self._pa = None

        cleanup_error = None
        # 关闭 stream 用于解除另一个线程中的 blocking read；禁止持锁等待线程。
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception as exc:
                cleanup_error = exc
                logger.warning("停止音频流失败", exc_info=True)
            try:
                stream.close()
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
                logger.warning("关闭音频流失败", exc_info=True)

        thread_alive = False
        if thread is not None and thread is not threading.current_thread():
            try:
                thread.join(timeout=2.0)
                thread_alive = thread.is_alive()
                if thread_alive:
                    logger.warning("音频采集线程在 2 秒内未退出")
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
                logger.warning("等待音频采集线程退出失败", exc_info=True)

        if thread_alive:
            # 未确认线程退出前不能 terminate PortAudio，也不能丢失生命周期句柄。
            with self._lock:
                self._stream = stream
                self._capture_thread = thread
                self._pa = pa
                self._running = True
                self._cleanup_pending = True
            if cleanup_error is not None:
                raise cleanup_error
            raise RuntimeError("音频采集线程在 2 秒内未退出，资源保留供后续 stop() 重试")

        if pa is not None:
            try:
                pa.terminate()
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
                logger.warning("终止 PyAudio 失败", exc_info=True)

        with self._lock:
            self._running = False
            self._cleanup_pending = False
        if cleanup_error is not None:
            raise cleanup_error

    def cleanup_pending(self) -> bool:
        """是否仍持有待重试回收的 PortAudio 生命周期资源。"""
        with self._lock:
            return self._cleanup_pending

    def is_running(self) -> bool:
        return self._running

    def stats(self) -> dict:
        """返回采集统计信息。"""
        return {
            "frames_captured": self._frames_captured,
            "overflow_count": self._overflow_count,
            "running": self._running,
        }


class SpeechDirectionRuntime:
    """speech_direction 运行时:采集 + worker 线程生命周期 + 共享状态。

    node.py 持有本类,通过 get_speech_direction() 轮询段级方向。
    """

    def __init__(
        self,
        config,
        pipeline,
        *,
        enable_capture: bool = True,
        on_fatal_error: Callable[[str], None] | None = None,
    ):
        """
        Args:
            config: SpeechDirectionConfig
            pipeline: SpeechDirectionPipeline 实例
            enable_capture: 是否启用采集(True=实时,False=离线灌数据)
            on_fatal_error: 不可恢复错误回调；每个 runtime 最多调用一次
        """
        self.config = config
        self.pipeline = pipeline
        self.vad_state = pipeline.vad_state
        self.doa_state = pipeline.doa_state
        self.max_age_ms = config.speech_direction_max_age_ms
        self._on_fatal_error = on_fatal_error
        self._fatal_error_reported = False
        self._fatal_error_lock = threading.Lock()

        # RingBuffer(4 秒缓冲,够 DOA overlap)
        self.ring_buffer = MultiChannelRingBuffer(
            capacity_frames=int(config.audio.sample_rate) * 4,
            channels=int(config.audio.channels),
        )

        # 实时采集是 device 模式的硬要求；构造失败必须传播给节点进入降级。
        self.capture: AudioCapture | None = None
        if enable_capture:
            try:
                self.capture = AudioCapture(
                    config,
                    self.ring_buffer,
                    on_fatal_error=self._report_fatal_error,
                )
            except Exception as e:
                reason = f"音频采集初始化失败: {e}"
                logger.error(reason, exc_info=True)
                self._report_fatal_error(reason)

        self._running = False
        self._threads: list[threading.Thread] = []
        self._reader_id: int | None = None

    def _report_fatal_error(self, reason: str) -> None:
        """线程安全地向节点报告首个不可恢复错误。"""
        with self._fatal_error_lock:
            if self._fatal_error_reported:
                return
            self._fatal_error_reported = True
        if self._on_fatal_error is not None:
            self._on_fatal_error(reason)

    def start(self) -> None:
        """启动采集与 worker 线程。"""
        if self._running:
            return
        self._running = True
        logger.info("SpeechDirection runtime 启动中...")

        # 采集先行
        if self.capture is not None:
            try:
                self.capture.start()
                logger.info(
                    "已连接 ReSpeaker 采集: channels=%d, sample_rate=%d",
                    self.config.audio.channels,
                    self.config.audio.sample_rate,
                )
            except Exception as e:
                reason = f"音频采集启动失败: {e}"
                logger.error(reason, exc_info=True)
                self.capture = None
                self._report_fatal_error(reason)
        else:
            logger.info("无采集设备,离线模式(等待外部 feed_audio 灌数据)")

        # 核心 worker 线程；启动失败时回滚已启动 capture，保留原始启动错误。
        try:
            self._start_thread(self._worker_loop, "SpeechDirectionWorker")
        except Exception as e:
            reason = f"SpeechDirectionWorker 启动失败: {e}"
            logger.error(reason, exc_info=True)
            self._running = False
            capture = self.capture
            self._threads = []
            if capture is not None:
                try:
                    capture.stop()
                except Exception:
                    logger.warning("回滚音频采集失败", exc_info=True)
                finally:
                    if not getattr(capture, "cleanup_pending", lambda: False)():
                        self.capture = None
            self._report_fatal_error(reason)

    def _start_thread(self, target: Callable, name: str) -> None:
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def _worker_loop(self):
        """阻塞主循环:每 hop 读 6ch → pipeline 处理。"""
        self._reader_id = self.ring_buffer.register(start_latest=True)
        hop_size = self.pipeline.hop_size
        logger.info(
            "SpeechDirectionWorker 启动: frame=%d hop=%d sr=%d",
            self.pipeline.frame_size,
            hop_size,
            self.pipeline.sr,
        )
        try:
            while self._running:
                # 从 RingBuffer 读 hop_size 帧 6ch(block 模式等够)
                data = self.ring_buffer.read(self._reader_id, hop_size, read_mode="block", timeout=1.0)
                if data is None or data.shape[1] == 0:
                    continue
                # data shape: (6, hop_size)
                try:
                    self.pipeline.process_block(data)
                except Exception as e:
                    reason = f"pipeline 处理异常: {e}"
                    logger.error(reason, exc_info=True)
                    self._running = False
                    self._report_fatal_error(reason)
        except Exception as e:
            reason = f"SpeechDirectionWorker 异常退出: {e}"
            logger.error(reason, exc_info=True)
            self._report_fatal_error(reason)
        finally:
            # 只有 stop() 主动清除 running 才属于正常退出。
            if self._running:
                self._running = False
                self._report_fatal_error("SpeechDirectionWorker 非预期退出")
            logger.info("SpeechDirectionWorker 停止")

    def stop(self) -> None:
        """幂等停止线程并释放资源，即使单项清理失败也完成其余收尾。"""
        if not self._running and self.capture is None and not self._threads:
            return
        logger.info("正在停止 SpeechDirection runtime...")
        self._running = False
        cleanup_error = None
        capture = self.capture
        if capture is not None:
            try:
                capture.stop()
            except Exception as exc:
                cleanup_error = exc
                logger.warning("停止音频采集失败", exc_info=True)
            finally:
                # 只有确认没有 pending 资源时才解除 runtime 对 capture 的可达引用。
                if not getattr(capture, "cleanup_pending", lambda: False)():
                    self.capture = None
        current_thread = threading.current_thread()
        threads, self._threads = self._threads, []
        for thread in threads:
            # fatal 回调可能从 worker 间接触发销毁，禁止线程 join 自身。
            if thread is not current_thread:
                try:
                    thread.join(timeout=2.0)
                except Exception as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
                    logger.warning("等待 SpeechDirection worker 退出失败", exc_info=True)
        logger.info("SpeechDirection runtime 已停止")
        if cleanup_error is not None:
            raise cleanup_error

    def feed_audio(self, data) -> None:
        """离线模式外部灌入 6ch 音频(无采集设备时使用)。

        data: 与 AudioCapture 写入格式一致的 6ch interleaved int16 PCM bytes,
              也接受 (n, 6) int16/float 数组。
        """
        if not self._running:
            raise RuntimeError("runtime 未启动,请先调用 start()")
        self.ring_buffer.write(data)

    def get_speech_direction(self, max_age_ms: int | None = None):
        """当前讲话方向:有段级角度且未过期 → 返回 angle,否则 None。

        语义(段级 DOA 透传):
          - 段级 DOA 输出 = 已过完整灰区验证 = 已验证可信方向,直接透传,不再二次过滤
          - angle 非 None 且 age_ms < max_age → 返回 angle
          - 其他(冷启动/方向过期)→ angle=None

        返回字段:
          - wall_clock_ts: 来源墙钟时间戳(doa_state._wall_clock_ts, time.time 域),
            用于 age_ms 计算;与 ROS 时钟(get_clock().now)不同域,见 node.py 注释
          - age_ms: now - ts,方向真实年龄
          - type: 方向类型(mid_long_seg/seg_end),供 node 按类型构造 stamp
        """
        if max_age_ms is None:
            max_age_ms = self.max_age_ms

        full = self.doa_state.get_full()
        angle = full.get("angle")
        ts = full.get("wall_clock_ts", 0.0)
        seq_id = full.get("seq_id", 0)
        meta = full.get("meta") or {}
        # 方向类型(mid_long_seg/seg_end),供 node 按类型构造 stamp:
        # 长语音中间方向用发布时刻(age≈0),段末方向用来源真实 age
        direction_type = meta.get("type")
        now = time.time()
        age_ms = max(0.0, (now - ts) * 1000.0)

        out_angle = None
        if angle is not None and age_ms < max_age_ms:
            out_angle = float(angle)

        _, is_speech = self.vad_state.get()
        return {
            "wall_clock_ts": ts,
            "angle": out_angle,
            "is_speech": bool(is_speech),
            "age_ms": age_ms,
            "seq_id": int(seq_id),
            "type": direction_type,
        }


__all__ = [
    "MultiChannelRingBuffer",
    "AudioCapture",
    "SpeechDirectionRuntime",
]
