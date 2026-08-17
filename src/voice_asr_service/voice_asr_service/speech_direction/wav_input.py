"""离线 WAV 输入(切块写 ringbuf)。

读多通道 WAV,按时间戳切块写入 runtime 的 RingBuffer,使 pipeline 与实时采集同路径
(pipeline 不感知输入源)。用途:算法调试、离线 E2E 测试、回归验证。

只支持 WAV(soundfile 解码),不做 MP3/pydub 通用化(speech_direction 离线测试用真实多通道 WAV,
通用格式解码是 ASR 的事,不在这里复用)。
"""

from __future__ import annotations

import logging
import math
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)


def _decode_wav(wav_path: str) -> tuple[np.ndarray, int]:
    """用 soundfile 解码多通道 WAV,返回 (audio (n, channels) float32, sample_rate)。

    Raises:
        RuntimeError: 文件不存在 / soundfile 未安装 / 不是原始 6 通道 16 kHz ReSpeaker WAV
    """
    import os

    if not os.path.exists(wav_path):
        raise RuntimeError(f"WAV 文件不存在: {wav_path}")
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile 未安装,请 pip install soundfile(speech_direction 离线 WAV 解析依赖)") from exc

    audio, sample_rate = sf.read(wav_path, dtype="float32")
    if audio.ndim == 1:
        raise RuntimeError(f"WAV 是单声道,speech_direction 需多通道(≥4ch): {wav_path}")
    return audio, sample_rate


class WavInput:
    """离线 WAV 回放器:读多通道 WAV,切块喂给 runtime。"""

    def __init__(
        self,
        runtime,
        wav_path: str,
        *,
        replay_rate: float = 1.0,
        chunk_samples: int = 160,
    ):
        """
        Args:
            runtime: SpeechDirectionRuntime 实例(必须已 start)
            wav_path: ReSpeaker 原始 6 通道 16 kHz WAV 路径。
                文件保留设备输入的 ch0~ch5；pipeline 后续再按 channel_indices
                从中选择 ch1~ch4 作为四麦算法输入。“4 通道”仅指内部增强/DOA
                处理通道数，不表示 WavInput 接受 4 通道 WAV。
            replay_rate: 回放倍率(1.0=实时,>1 加速,<1 慢放)
            chunk_samples: 每块采样数(默认 160 = 10ms @ 16kHz,对齐采集 chunk_size)
        """
        self.runtime = runtime
        self.wav_path = wav_path
        self.replay_rate = float(replay_rate)
        self.chunk_samples = int(chunk_samples)

        audio, sample_rate = _decode_wav(wav_path)
        self.audio = audio  # (n, channels)
        self.sample_rate = sample_rate
        self.channels = audio.shape[1]

        # 转 interleaved int16 bytes(与 PyAudio callback 写入格式一致)
        clipped = np.clip(audio, -1.0, 1.0)
        self.audio_bytes = (clipped * 32767).astype(np.int16).tobytes()

        # 采样间隔(实时回放节拍)
        self._interval_sec = chunk_samples / self.sample_rate / max(self.replay_rate, 0.01)
        self._thread: threading.Thread | None = None
        self._running = False

    def _eof_padding_samples(self) -> int:
        """计算 EOF 后需追加的静音样本数，复用现有 pipeline 自然结算段末。

        兼容两条链路:
        - legacy 链路 (SpeechDirectionPipeline): 有 enh_block_size 滑窗,
          需用静音把滑窗中心之后的样本推出,再补 seg_end_gap_s 让状态机吐末段方向。
        - stateful 链路 (StreamingSpeechDirectionPipeline): 无滑窗,增量处理,
          只需补 seg_end_gap_s 对应静音让 TemporalSpeechGate 输出末段方向。
        """
        pipeline = self.runtime.pipeline
        hop_size = int(pipeline.hop_size)
        sample_rate = int(pipeline.sr)
        # 兼容两条链路的段末间隔参数名:
        # legacy 用 seg_end_gap_s (秒), streaming 用 segment_end_gap_samples (样本)
        params = pipeline.params
        if hasattr(params, "segment_end_gap_samples"):
            end_gap_samples = int(params.segment_end_gap_samples)
        else:
            end_gap_samples = math.ceil(float(params.seg_end_gap_s) * sample_rate)
        end_gap_hops = math.ceil(end_gap_samples / hop_size)

        # stateful 链路无 enh_block_size,跳过滑窗尾部填充
        if not hasattr(pipeline, "enh_block_size"):
            return end_gap_hops * hop_size

        # legacy 链路: FullSubNet 从增强滑窗中间取一个 hop;窗口中心之后的输入仍需用静音推出
        enh_block_size = int(pipeline.enh_block_size)
        center_start = (enh_block_size - hop_size) // 2
        enhancement_tail = max(0, enh_block_size - (center_start + hop_size))
        enhancement_tail_hops = math.ceil(enhancement_tail / hop_size)
        return (enhancement_tail_hops + end_gap_hops) * hop_size

    def _feed_eof_silence(self) -> None:
        """按完整 hop 追加 6ch 静音，让现有 VAD/状态机输出最后一个段末方向。"""
        padding_samples = self._eof_padding_samples()
        if padding_samples <= 0:
            return
        hop_size = int(self.runtime.pipeline.hop_size)
        silence_hop = np.zeros((hop_size, self.channels), dtype=np.int16).tobytes()
        for _ in range(padding_samples // hop_size):
            if not self._running:
                break
            self.runtime.feed_audio(silence_hop)
            time.sleep(hop_size / self.sample_rate / max(self.replay_rate, 0.01))

    def start(self) -> None:
        """启动回放线程(后台,非阻塞)。"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._replay_loop, name="WavInput", daemon=True)
        self._thread.start()
        logger.info(
            "WavInput 启动: %s (%dch @ %dHz, %.2fs, rate=%.2f)",
            self.wav_path,
            self.channels,
            self.sample_rate,
            len(self.audio) / self.sample_rate,
            self.replay_rate,
        )

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _replay_loop(self) -> None:
        """按 chunk 切块,以 replay_rate 节拍喂给 runtime。"""
        # int16 interleaved:每 chunk = chunk_samples * channels * 2 bytes
        bytes_per_sample = 2 * self.channels
        chunk_bytes = self.chunk_samples * bytes_per_sample
        offset = 0
        total = len(self.audio_bytes)

        try:
            while self._running and offset < total:
                chunk = self.audio_bytes[offset : offset + chunk_bytes]
                if not chunk:
                    break
                self.runtime.feed_audio(chunk)
                offset += len(chunk)
                # 实时回放节拍(replay_rate>1 加速,<1 慢放)
                time.sleep(self._interval_sec)
            # 正常播放到 EOF 后追加参数化静音，推出增强滑窗尾部并满足段末间隔。
            if self._running and offset >= total:
                self._feed_eof_silence()
        except Exception as e:
            logger.error("WavInput 回放异常: %s", e, exc_info=True)
        finally:
            self._running = False
            logger.info("WavInput 回放结束")


__all__ = ["WavInput"]
