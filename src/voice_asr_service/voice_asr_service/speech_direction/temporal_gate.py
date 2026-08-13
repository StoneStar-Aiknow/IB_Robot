"""无 Top-2 的时间制 Silero VAD 门控。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalGateResult:
    """一个非重叠512样本帧的门控结果。"""

    vad_prob: float
    rms: float
    is_speech: bool
    is_gray: bool
    gate_state: str
    frame_start_sample: int
    frame_end_sample: int


class TemporalSpeechGate:
    """IDLE/CANDIDATE/ACTIVE 时间制门控，全部边界使用整数sample。"""

    def __init__(
        self,
        silero_engine,
        *,
        vad_threshold: float = 0.65,
        rms_threshold: float = 0.002,
        candidate_window_samples: int = 1024,
        exit_gap_samples: int = 2400,
    ):
        self.silero = silero_engine
        self.vad_threshold = float(vad_threshold)
        self.rms_threshold = float(rms_threshold)
        self.candidate_window_samples = int(candidate_window_samples)
        self.exit_gap_samples = int(exit_gap_samples)
        self.reset()

    def reset(self) -> None:
        """重置门控和 Silero 隐藏状态；gap后不得跨不连续音频复用。"""
        self._state = "IDLE"
        self._candidate_first_end: int | None = None
        self._non_gray_start: int | None = None
        self.silero.reset_state()

    def close(self) -> None:
        """释放 Silero 后端资源；具体 ONNX/raw ACL 差异由引擎内部处理。"""
        close = getattr(self.silero, "close", None)
        if close is not None:
            close()

    def process_frame(self, audio512: np.ndarray, *, frame_start_sample: int) -> TemporalGateResult:
        value = np.asarray(audio512, dtype=np.float32)
        if value.shape != (512,):
            raise ValueError(f"Silero时间制门控输入必须为(512,)，得到{value.shape}")
        frame_end = int(frame_start_sample) + 512
        prob = float(self.silero.inference(value))
        rms = float(np.sqrt(np.mean(value.astype(np.float64) ** 2)) + 1e-12)
        hit = prob > self.vad_threshold and rms > self.rms_threshold

        # 候选首帧不进入段累积；第二次命中后由pipeline回填候选期SRP帧。
        is_gray = False
        if self._state == "IDLE":
            if hit:
                self._state = "CANDIDATE"
                self._candidate_first_end = frame_end
        elif self._state == "CANDIDATE":
            # 状态不变量：CANDIDATE 态必须有候选首帧。用显式 RuntimeError 而非 assert，
            # 避免 python -O 剥离后退化成难以诊断的 TypeError（frame_end - None）。
            if self._candidate_first_end is None:
                raise RuntimeError(
                    f"TemporalSpeechGate 状态机损坏: CANDIDATE 态 _candidate_first_end 为 None,"
                    f" frame_start={frame_start_sample}"
                )
            if hit and frame_end - self._candidate_first_end <= self.candidate_window_samples:
                self._state = "ACTIVE"
                self._candidate_first_end = None
                self._non_gray_start = None
                is_gray = True
            elif hit:
                # 超窗命中成为新的候选首帧，禁止旧候选跨窗激活。
                self._candidate_first_end = frame_end
            elif frame_end - self._candidate_first_end > self.candidate_window_samples:
                self._state = "IDLE"
                self._candidate_first_end = None
        elif self._state == "ACTIVE":
            if hit:
                self._non_gray_start = None
                is_gray = True
            else:
                if self._non_gray_start is None:
                    self._non_gray_start = frame_start_sample
                if frame_end - self._non_gray_start >= self.exit_gap_samples:
                    self._state = "IDLE"
                    self._non_gray_start = None
        else:
            raise RuntimeError(f"未知时间制门控状态: {self._state}")

        return TemporalGateResult(
            vad_prob=prob,
            rms=rms,
            is_speech=self._state == "ACTIVE",
            is_gray=is_gray,
            gate_state=self._state,
            frame_start_sample=int(frame_start_sample),
            frame_end_sample=frame_end,
        )


__all__ = ["TemporalSpeechGate", "TemporalGateResult"]
