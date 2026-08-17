"""增强 + 人声门控 + DOA 处理链 + 段级触发状态机。

算法逻辑与回归基线逐字对齐。

数据流(单 hop):
    RingBuffer.read(hop=2048 6ch)
      → enh_buf4 滑动缓冲(8192) → FullSubNet 增强 4ch → 取中段 hop
          → enh_ch1 → SpeechGate(4×512 子帧 VAD + RMS 灰区判据)
          → 灰区帧 enh4 → STFT(4096/Hann) → SRP scores(72角)
      → enh4 入全程滚动缓冲(上限 10 分钟)
      → 状态机(逐帧累积 scores+子帧RMS)
          段结束: 段时长<0.10s 或 max(子帧RMS)<0.005 → 丢弃
                  否则能量加权累积 argmax → DoaState.update(angle/墙钟ts/seq_id)
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .diagnostics import DiagnosticsPacket, FrameMetrics
from .doa.srp_phat import StftSrpPhat
from .enhancement.fullsubnet import FullSubNetEnhancer
from .speech_gate import SpeechGate

logger = logging.getLogger(__name__)


# ============================ 共享状态 ============================


class VadState:
    """VAD 共享状态(线程安全)。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._vad_prob = 0.0
        self._is_speech = False
        self._rms = 0.0
        self._is_gray = False
        self._wall_clock_ts = 0.0

    def update(self, vad_prob: float, is_speech: bool, rms: float, is_gray: bool, wall_clock_ts: float):
        with self._lock:
            self._vad_prob = float(vad_prob)
            self._is_speech = bool(is_speech)
            self._rms = float(rms)
            self._is_gray = bool(is_gray)
            self._wall_clock_ts = float(wall_clock_ts)

    def get(self) -> tuple[float, bool]:
        """返回 (vad_prob, is_speech)。"""
        with self._lock:
            return self._vad_prob, self._is_speech


class DoaState:
    """DOA 共享状态(线程安全)。段级输出时更新,无人声时不清空。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._angle: float | None = None
        self._wall_clock_ts = 0.0
        self._seq_id = 0  # 段序号,每段递增,用于消费者去重
        self._meta: dict = {}

    def update(self, angle: float, wall_clock_ts: float, seq_id: int, meta: dict | None = None):
        with self._lock:
            self._angle = float(angle)
            self._wall_clock_ts = float(wall_clock_ts)
            self._seq_id = int(seq_id)
            self._meta = dict(meta) if meta else {}

    def get_full(self) -> dict:
        with self._lock:
            return {
                "angle": self._angle,
                "wall_clock_ts": self._wall_clock_ts,
                "seq_id": self._seq_id,
                "meta": dict(self._meta),
            }


@dataclass
class PipelineParams:
    """处理链参数(参数与算法基线对齐)。"""

    sample_rate: int = 16000
    frame_size: int = 4096  # STFT/SRP 帧长
    hop_size: int = 2048  # hop = frame/2,128ms
    input_channels: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    # 段级状态机参数(灰区帧判定阈值 vad_threshold/rms_threshold 由 SpeechGate 持有,不在此)
    seg_end_gap_s: float = 0.15  # 非灰区持续多久判段结束
    min_seg_dur_s: float = 0.10  # 最短段时长
    min_accum_frames: int = 3
    max_accum_dur_s: float = 1.0  # 长段中间输出阈值
    seg_max_rms_threshold: float = 0.005  # 段内子帧 max RMS 门限
    # FullSubNet 增强上下文
    enh_block_size: int = 8192  # 滑动块大小(单 hop 增强有 STFT 边界效应,需 ≥8192 取中段)
    enh_full_max_seconds: int = 600  # 增强 4ch 全程滚动缓冲上限(10 分钟)


@dataclass
class HopResult:
    """单 hop 处理结果(供调试/维测读取)。"""

    enh4: np.ndarray | None = None  # (hop, 4) 增强后 4ch(取中段)
    enh_ch1: np.ndarray | None = None  # (hop,) 增强 ch1
    is_gray_hop: bool = False
    scores_frame: np.ndarray | None = None  # (n_angles,) 单帧 SRP scores
    frame_doa: int | None = None  # 单帧 argmax
    seg_output: int | None = None  # 段级触发输出 DOA(本 hop 触发时才有)
    seg_seq: int = 0
    raw_start_sample: int = 0
    enh_start_sample: int | None = None
    session_sample: int = 0
    hop_t: float = 0.0  # 仅由 session_sample / sample_rate 推导


class SpeechDirectionPipeline:
    """增强 + 人声门控 + DOA 处理链 + 段级触发状态机。

    线程安全性:process_block 由单一 worker 线程调用,VadState/DoaState 自身线程安全。
    """

    def __init__(
        self,
        fullnet: FullSubNetEnhancer,
        speech_gate: SpeechGate,
        srp: StftSrpPhat,
        params: PipelineParams,
        vad_state: VadState,
        doa_state: DoaState,
        diagnostics=None,
    ):
        """
        Args:
            diagnostics: 可选, DiagnosticsRecorder 实例;None 时关闭维测。
                         提供时,每个 hop 入队维测包(原始/降噪/灰区音频 + VAD/RMS/DOA 时序)。
        """
        self.fullnet = fullnet
        self.speech_gate = speech_gate
        self.srp = srp
        self.params = params
        self.vad_state = vad_state
        self.doa_state = doa_state
        self.diagnostics = diagnostics  # 可选维测记录器

        self.sr = params.sample_rate
        self.frame_size = params.frame_size
        self.hop_size = params.hop_size
        self.input_channels = params.input_channels

        # FullSubNet 增强上下文:原始 4ch 滑动缓冲
        self.enh_block_size = params.enh_block_size
        self._enh_buf4 = np.zeros((0, 4), dtype=np.float32)

        # 增强 4ch 全程滚动缓冲(上限 N 秒,供段结束截断保存)
        self.enh_full_max_samples = int(self.sr * params.enh_full_max_seconds)
        self._enh_full4_blocks: list[np.ndarray] = []
        self._enh_full4_total = 0
        self._enh_full4_dropped = 0

        # BlockFramer:DOA 的 STFT 用增强 4ch,保留上一块 hop overlap 拼 frame_size
        self._prev_block = np.zeros((self.hop_size, 4), dtype=np.float32)

        # 段级触发状态机(IDLE / ACCUMULATING)
        self._state = "IDLE"
        self._seg_scores: list[np.ndarray] = []  # 段内累积的 hop scores
        self._seg_rms: list[float] = []  # 段内累积的 hop RMS
        self._seg_sub_rms: list[float] = []  # 段内累积的子帧 RMS(段结束 max RMS 判定)
        self._seg_start_sample = 0
        self._seg_last_gray_sample = 0
        self._seg_seq = 0  # 段序号(每段递增,维测曲线用)
        # 输出序号:每次 doa_state.update(中间输出 / 段末输出)前 +1。
        # 中间方向与段末方向各自独立 seq_id,node.py 用 seq_id 去重时不会误吞段末。
        self._output_seq = 0
        # 段级输出历史:(output_seq, angle, hop_t, type) 列表,供回归测试/维测读取
        self._seg_history: list[tuple[int, int, float, str]] = []

        # 已处理样本计数是会话唯一时钟，任何秒数都只能由它换算。
        self._samples_processed = 0
        self._diagnostics_error_reported = False
        # close 契约：best-effort terminal。legacy 对照链路非并发，无推理竞态，
        # 故不设禁新推理标志；仅靠 _cleanup_complete 挡重入，避免重复释放已成功资源。
        self._cleanup_complete = False

    def process_block(
        self,
        data: np.ndarray,
        *,
        capture_start_sample: int | None = None,
    ) -> HopResult:
        """处理一个 hop 块(6, hop_size)。

        Args:
            data: (6, hop_size) float32,6 通道音频
            capture_start_sample: 原始采集绝对起点；None 时兼容旧连续调用。

        Returns:
            HopResult
        """
        t0 = time.perf_counter()
        if capture_start_sample is None:
            capture_start_sample = self._samples_processed
        raw_start_sample = int(capture_start_sample)

        # 1. 取 4ch(input_channels=[1,2,3,4],data 行索引即通道号)
        mic4 = data[np.array(self.input_channels), :].T.astype(np.float32)  # (hop, 4)

        # 2. 累积原始 4ch 到滑动缓冲,达 enh_block_size 后增强取中间 hop 段
        #    (FullNet 单 hop 增强有 STFT 边界效应致 RMS 偏低,需 ≥8192 上下文取中段)
        self._enh_buf4 = np.concatenate([self._enh_buf4, mic4], axis=0)
        if self._enh_buf4.shape[0] < self.enh_block_size:
            # raw 已由采集侧连续记录；增强窗口不足时只推进兼容时轴，不制造空维测包。
            self._samples_processed = raw_start_sample + mic4.shape[0]
            return HopResult(
                raw_start_sample=raw_start_sample,
                enh_start_sample=None,
                session_sample=raw_start_sample,
                hop_t=raw_start_sample / self.sr,
            )
        # 只保留最新 enh_block_size 样本(滑动)
        self._enh_buf4 = self._enh_buf4[-self.enh_block_size :]
        # 增强 4ch(整块),取中间 hop_size 段作为本 hop 输出
        # crm 当前未使用,enhance_4ch 仍返回该能力供未来维测在 DiagnosticsPacket 中携带
        stage_start = time.perf_counter()
        enh4_full, _ = self.fullnet.enhance_4ch(self._enh_buf4)
        fullsubnet_elapsed_ms = (time.perf_counter() - stage_start) * 1000
        center_start = (self.enh_block_size - self.hop_size) // 2
        enh4 = enh4_full[center_start : center_start + self.hop_size].copy()  # (hop, 4)
        enh_ch1 = enh4[:, 0].copy()  # (hop,) 增强 ch1

        # 增强 4ch 追加到全程滚动缓冲(上限 10 分钟)
        self._enh_full4_blocks.append(enh4.copy())
        self._enh_full4_total += enh4.shape[0]
        while self._enh_full4_total > self.enh_full_max_samples and len(self._enh_full4_blocks) > 1:
            dropped = self._enh_full4_blocks.pop(0)
            self._enh_full4_total -= dropped.shape[0]
            self._enh_full4_dropped += dropped.shape[0]

        # 3. 人声门控:Silero 4×512 子帧 VAD + RMS 灰区判据
        stage_start = time.perf_counter()
        gate = self.speech_gate.process_hop(enh_ch1)
        silero_vad_elapsed_ms = (time.perf_counter() - stage_start) * 1000
        vad_prob = gate.vad_prob
        is_gray_hop = gate.is_gray_hop
        rms = gate.rms
        is_speech = gate.is_speech

        # 4. 灰区 hop 才做 SRP(4ch 已增强,直接 STFT + scores)
        scores_frame = None
        frame_doa = None
        srp_elapsed_ms = 0.0
        if is_gray_hop:
            stage_start = time.perf_counter()
            # BlockFramer:上一块 overlap + 当前块 → 4096 帧
            block4 = np.concatenate([self._prev_block, enh4], axis=0)  # (4096, 4)
            self._prev_block = enh4.copy()
            # STFT 单帧(4096 帧,U=1)
            spec4 = self.srp.stft_4ch(block4)  # (n_freq, 1, 4)
            # 单帧 scores
            scores_frame = self.srp.compute_all_scores(spec4)[0]  # (n_angles,)
            frame_doa = int(self.srp.angles[int(np.argmax(scores_frame))])
            srp_elapsed_ms = (time.perf_counter() - stage_start) * 1000
        else:
            # 非灰区帧:prev_block 用增强 enh4 维持 overlap 连续性
            self._prev_block = enh4.copy()

        # 灰区 hop 时累积子帧 RMS(段结束 max RMS 判定用)。新段初始化必须先完成，
        # 因此把当前 hop 的 sub_rms 交给状态机在初始化后追加。
        self._samples_processed = raw_start_sample + mic4.shape[0]
        # 增强真实起点 = 当前采集块终点回看增强窗口，再加实际中段切片偏移。
        window_start_sample = self._samples_processed - self.enh_block_size
        enh_start_sample = window_start_sample + center_start
        session_sample = enh_start_sample
        hop_t = session_sample / self.sr

        # 5. 段级触发状态机
        seg_output = self._state_machine(is_gray_hop, scores_frame, rms, session_sample, gate.sub_rms)

        # 6. 更新 VAD 共享状态(用音频时间,便于 age_ms 计算)
        self.vad_state.update(vad_prob, is_speech, rms, is_gray_hop, hop_t)

        # 7. 维测旁路共享同一整数采样位置，丢包或异常都不得反向影响算法。
        elapsed_ms = (time.perf_counter() - t0) * 1000
        measured_stages_ms = fullsubnet_elapsed_ms + silero_vad_elapsed_ms + srp_elapsed_ms
        # other 是入口至指标构造前的剩余同步开销，不包含诊断入队和后台写盘。
        other_elapsed_ms = max(elapsed_ms - measured_stages_ms, 0.0)
        metrics = FrameMetrics(
            session_sample=session_sample,
            sample_count=self.hop_size,
            segment_seq=self._seg_seq,
            vad_probability=float(vad_prob),
            is_speech=bool(is_speech),
            is_gray=bool(is_gray_hop),
            rms=float(rms),
            frame_doa_degree=frame_doa,
            segment_doa_degree=seg_output,
            inference_elapsed_ms=elapsed_ms,
            fullsubnet_elapsed_ms=fullsubnet_elapsed_ms,
            silero_vad_elapsed_ms=silero_vad_elapsed_ms,
            srp_elapsed_ms=srp_elapsed_ms,
            other_elapsed_ms=other_elapsed_ms,
        )
        self._enqueue_diagnostics(
            raw_start_sample=raw_start_sample,
            raw6ch=None,
            enh_start_sample=enh_start_sample,
            enh4ch=enh4,
            metrics=metrics,
        )

        if is_gray_hop:
            logger.debug(
                "[Pipeline] vad=%.2f rms=%.4f gray_hop=%s frame_doa=%s elapsed=%.1fms",
                vad_prob,
                rms,
                is_gray_hop,
                frame_doa,
                elapsed_ms,
            )

        return HopResult(
            enh4=enh4,
            enh_ch1=enh_ch1,
            is_gray_hop=is_gray_hop,
            scores_frame=scores_frame,
            frame_doa=frame_doa,
            seg_output=seg_output,
            seg_seq=self._seg_seq,
            raw_start_sample=raw_start_sample,
            enh_start_sample=enh_start_sample,
            session_sample=session_sample,
            hop_t=hop_t,
        )

    def _enqueue_diagnostics(
        self,
        *,
        raw_start_sample: int,
        raw6ch: np.ndarray | None,
        enh_start_sample: int | None,
        enh4ch: np.ndarray | None,
        metrics: FrameMetrics | None,
    ) -> None:
        """尽力投递维测包；记录器故障只报告一次并永久退出实时路径。"""
        if self.diagnostics is None or self._diagnostics_error_reported:
            return
        packet = DiagnosticsPacket(
            raw_start_sample=raw_start_sample,
            raw6ch=raw6ch.copy() if raw6ch is not None else None,
            enh_start_sample=enh_start_sample,
            enh4ch=enh4ch.copy() if enh4ch is not None else None,
            metrics=metrics,
        )
        try:
            self.diagnostics.enqueue(packet)
        except Exception:
            self._diagnostics_error_reported = True
            logger.exception("[Pipeline] 维测入队异常，已停用 pipeline 维测旁路")

    # ------------------------------------------------------------------ 状态机
    def _state_machine(
        self,
        is_gray: bool,
        scores_frame,
        rms: float,
        session_sample: int,
        sub_rms=(),
    ):
        """hop 粒度段级触发状态机:一段灰区出一次 DOA,长段中间输出。

        灰区切段逻辑的流式化实现:
          - 离线基线:逐 512 帧 mask,连续灰区成段,段间 ≤merge_gap 合并,段时长 ≥min_dur 保留。
          - 流式 hop 等价:灰区 hop 入段;非灰 hop 累计间隔,超 merge_gap 则段结束;
            段间 ≤merge_gap 的非灰间隔合并(保持 ACCUMULATING);段时长 ≥min_seg_dur 输出,否则丢弃。
        """
        p = self.params
        seg_output = None
        # 状态转移只比较整数样本，避免长会话浮点秒精度改变边界判定。
        end_gap_samples = math.ceil(p.seg_end_gap_s * self.sr)
        min_seg_samples = math.ceil(p.min_seg_dur_s * self.sr)
        max_accum_samples = math.ceil(p.max_accum_dur_s * self.sr)
        hop_t = session_sample / self.sr

        if is_gray:
            # 灰区 hop
            if self._state == "IDLE":
                # 进入新段
                self._state = "ACCUMULATING"
                self._seg_seq += 1
                self._seg_scores = []
                self._seg_rms = []
                self._seg_sub_rms = []
                self._seg_start_sample = session_sample
                self._seg_last_gray_sample = session_sample
                logger.debug("[Gray] 段 %d 开始 @%.3fs", self._seg_seq, hop_t)

            self._seg_sub_rms.extend(sub_rms)
            # 入缓冲
            if scores_frame is not None:
                self._seg_scores.append(scores_frame)
                self._seg_rms.append(rms)
            # 更新段内最后灰区 hop 位置
            self._seg_last_gray_sample = session_sample

            # 长段覆盖采用半开区间 [seg_start, current_hop_start + hop_size)。
            accum_samples = self._seg_last_gray_sample - self._seg_start_sample + self.hop_size
            if accum_samples >= max_accum_samples and len(self._seg_scores) >= 1:
                accum_dur = accum_samples / self.sr
                angle, _ = self._segment_doa()
                if angle is not None:
                    # 中间方向用独立 _output_seq(与段末不同),node.py 去重不会误吞段末
                    self._output_seq += 1
                    self._seg_history.append((self._output_seq, angle, hop_t, "mid_long_seg"))
                    self.doa_state.update(
                        angle,
                        time.time(),
                        self._output_seq,
                        meta={
                            "type": "mid_long_seg",
                            "seg_seq": self._seg_seq,
                            "accum_dur": round(accum_dur, 3),
                        },
                    )
                    seg_output = angle
                    logger.info(
                        "[Gray] 段 %d 中间输出 DOA=%d° (累积%.2fs %d帧)",
                        self._seg_seq,
                        angle,
                        accum_dur,
                        len(self._seg_scores),
                    )
                # 清缓冲继续累积(仍在 ACCUMULATING)
                self._seg_scores = []
                self._seg_rms = []
                self._seg_sub_rms = []
                # 当前 hop 已被本窗口消费；下一窗口从当前半开区间终点开始。
                self._seg_start_sample = session_sample + self.hop_size

        else:
            # 非灰区 hop
            if self._state == "ACCUMULATING":
                # 非灰间隔直接用整数样本差。
                gap_samples = session_sample - self._seg_last_gray_sample
                # 非灰持续 >= merge_gap → 段结束
                if gap_samples >= end_gap_samples:
                    seg_samples = self._seg_last_gray_sample - self._seg_start_sample + self.hop_size
                    seg_dur = seg_samples / self.sr
                    if len(self._seg_scores) >= p.min_accum_frames and seg_samples >= min_seg_samples:
                        # 段内子帧 max RMS 判定:< 门限丢弃(卡"人耳听不到"的弱碎段)
                        seg_max_rms = float(max(self._seg_sub_rms)) if self._seg_sub_rms else 0.0
                        if seg_max_rms < p.seg_max_rms_threshold:
                            logger.info(
                                "[Gray] 段 %d 丢弃 (max RMS %.4f < %.2f, 段%.2fs)",
                                self._seg_seq,
                                seg_max_rms,
                                p.seg_max_rms_threshold,
                                seg_dur,
                            )
                        else:
                            angle, _ = self._segment_doa()
                            if angle is not None:
                                # 段末方向用独立 _output_seq(与中间方向不同)
                                self._output_seq += 1
                                self._seg_history.append((self._output_seq, angle, hop_t, "seg_end"))
                                self.doa_state.update(
                                    angle,
                                    time.time(),
                                    self._output_seq,
                                    meta={
                                        "type": "seg_end",
                                        "seg_seq": self._seg_seq,
                                        "seg_dur": round(seg_dur, 3),
                                    },
                                )
                                seg_output = angle
                                logger.info(
                                    "[Gray] 段 %d 结束 DOA=%d° (段%.2fs %d帧 maxRMS=%.3f)",
                                    self._seg_seq,
                                    angle,
                                    seg_dur,
                                    len(self._seg_scores),
                                    seg_max_rms,
                                )
                    else:
                        logger.debug(
                            "[Gray] 段 %d 丢弃 (时长%.2fs<%.2fs 或帧数<%d)",
                            self._seg_seq,
                            seg_dur,
                            p.min_seg_dur_s,
                            p.min_accum_frames,
                        )
                    # 清缓冲回 IDLE
                    self._state = "IDLE"
                    self._seg_scores = []
                    self._seg_rms = []
                    self._seg_sub_rms = []
                # else: 非灰持续 < merge_gap,保持 ACCUMULATING(音素内停顿)

        return seg_output

    def _segment_doa(self):
        """对段内累积的 hop scores 按帧 RMS 加权累积 → argmax → DOA。

        能量加权累积逻辑与算法基线等价:acc = Σ(w[k]*scores[k])/Σw,
        w[k] = rms[k]/max(rms)。
        """
        if not self._seg_scores:
            return None, None
        scores_arr = np.stack(self._seg_scores, axis=0)  # (n_hop_frames, n_angles)
        rms_arr = np.asarray(self._seg_rms, dtype=np.float32)
        rms_arr = np.maximum(rms_arr, 1e-8)
        w = rms_arr / (rms_arr.max() + 1e-8)  # 归一化 [0,1]
        acc = (scores_arr * w[:, None]).sum(0) / (w.sum() + 1e-8)  # (n_angles,)
        angle = int(self.srp.angles[int(np.argmax(acc))])
        return angle, acc

    def _reset_temporal_context(self) -> None:
        """清除只能跨连续样本复用的状态，不回退会话级输出序号。"""
        self._state = "IDLE"
        self._seg_scores = []
        self._seg_rms = []
        self._seg_sub_rms = []
        self._seg_start_sample = 0
        self._seg_last_gray_sample = 0
        self._enh_buf4 = np.zeros((0, 4), dtype=np.float32)
        self._enh_full4_blocks = []
        self._enh_full4_total = 0
        self._enh_full4_dropped = 0
        self._prev_block = np.zeros((self.hop_size, 4), dtype=np.float32)
        self.speech_gate.reset()

    def reset_for_gap(self, *, next_capture_sample: int, dropped_samples: int) -> None:
        """采集缺口后丢弃未完成段，并从新的绝对采样位置冷启动。"""
        if dropped_samples < 0:
            raise ValueError("dropped_samples 不能为负数")
        self._reset_temporal_context()
        self._samples_processed = int(next_capture_sample)

    def reset(self):
        """重置状态机与缓冲(冷启动 / 长停顿后复位)。"""
        self._reset_temporal_context()
        self._seg_seq = 0
        self._output_seq = 0
        self._seg_history = []

    def close(self) -> None:
        """best-effort terminal 关闭：尽力释放 legacy 增强器与 Silero，一个失败仍继续关闭另一个，末尾汇总异常。

        重入只在清理已尝试完毕后才直接返回；属 best-effort terminal，非可重试。
        """
        if self._cleanup_complete:
            # 清理已尝试完毕（含失败项），重入直接返回，避免重复释放已成功资源。
            return
        errors = []
        try:
            for component in (self.speech_gate, self.fullnet):
                close = getattr(component, "close", None)
                if close is None:
                    continue
                try:
                    close()
                except Exception as exc:
                    errors.append(str(exc) or type(exc).__name__)
        finally:
            self._cleanup_complete = True
        if errors:
            raise RuntimeError("; ".join(errors))

    def get_segment_history(self) -> list[tuple[int, int, float, str]]:
        """返回段级输出历史:(output_seq, angle, hop_t, type) 列表。

        type ∈ {"mid_long_seg", "seg_end"}。供回归测试/维测读取所有段级 DOA。
        注意:中间方向(mid_long_seg)和段末方向(seg_end)各有独立 output_seq,都计入历史。
        """
        return list(self._seg_history)


__all__ = ["VadState", "DoaState", "PipelineParams", "HopResult", "SpeechDirectionPipeline"]
