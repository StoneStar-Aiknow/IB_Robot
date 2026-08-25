"""FullSubNet cumulative 双帧 stateful 公共流式增强器。

每次接收连续 512 samples × 4ch，内部仍按 512/256 STFT 生成两个频谱帧；
FB/SB LSTM 由注入的执行器按 T=2 顺序推进，Host 统一负责两级 cumulative
Laplace norm、邻频展开、look-ahead=2 对齐和归一化 overlap-add。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np

from .fullsubnet_stateful_executor import (
    BATCH,
    INPUT_SAMPLES,
    LOOK_AHEAD,
    N_FFT,
    NUM_FREQS,
    SB_BATCH,
    SB_FEATURES,
    STATEFUL_FULLSUBNET_CONTRACT,
    STFT_HOP,
    StatefulFullSubNetExecutor,
)

logger = logging.getLogger(__name__)

HOP = STFT_HOP
EPS = 1.1920929e-7


def _decompress_cirm(mask: np.ndarray, k: float = 10.0, limit: float = 9.9) -> np.ndarray:
    """还原压缩域 cRM，公式与 FullSubNet 上游一致。"""
    value = np.clip(mask, -limit, limit)
    return -k * np.log((k - value) / (k + value) + 1e-8)


class StatefulFullSubNetEnhancer:
    """4ch cumulative stateful 公共增强器，512 samples 调用一次。"""

    def __init__(
        self,
        executor: StatefulFullSubNetExecutor,
        *,
        manifest_path: str = "",
        timing_enabled: bool = False,
        initialize_backend: bool = True,
    ):
        if manifest_path:
            self._verify_manifest(manifest_path)
        self._executor = executor
        self.backend = executor.backend
        self.input_samples = INPUT_SAMPLES
        self.timing_enabled = bool(timing_enabled)
        self._initialize_backend = bool(initialize_backend)
        self._lock = threading.RLock()
        self._closed = False  # 关闭一旦开始即禁新推理（process/reset 见此即拒绝）
        self._cleanup_complete = False  # 清理已尝试完毕（含失败项），重入据此返回
        self._timing_sample_counter = 0
        self.last_timing_ms: dict[str, float] = {}
        # periodic Hann 与当前 FullSubNet/历史 B4 验证保持一致。
        self._window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
        self._window_square = self._window * self._window
        self._reset_host_state()
        if self._initialize_backend:
            self._executor.reset()
        logger.info("FullSubNet stateful 公共增强器已加载: backend=%s", self.backend)

    @staticmethod
    def _verify_manifest(path: str) -> None:
        """用 STATEFUL_FULLSUBNET_CONTRACT 校验 checkpoint manifest（SSOT）。

        FullSubNet 训练仓的 checkpoint manifest（cum_fullsubnet_best_model_218epochs.
        manifest.json）是平铺结构：norm_type/checkpoint/sha256 等，没有嵌套的
        algorithm_contract 字段。这里校验 manifest 实际声明的契约字段
        （norm_type、checkpoint）与 Python 常量逐项一致；其余字段（sha256 是
        checkpoint 内容指纹、epoch/best_score 是训练元数据）不在算法契约范围。
        """
        with open(path, encoding="utf-8") as file:
            manifest = json.load(file)
        # 兼容两种 manifest 结构：标准 inference_manifest 的嵌套 algorithm_contract，
        # 与 FullSubNet 训练仓的平铺 checkpoint manifest。
        contract = manifest.get("algorithm_contract")
        if isinstance(contract, dict):
            fields = contract
        else:
            fields = manifest
        expected = asdict(STATEFUL_FULLSUBNET_CONTRACT)
        # 只校验 manifest 实际声明的契约字段；Python 多出的实现细节字段
        # （batch/num_freqs/sb_features）不强制 manifest 声明。
        for key, want in expected.items():
            if key not in fields:
                continue
            actual = fields[key]
            # checkpoint 字段语义化比对：manifest 用文件名（带 .tar 后缀），
            # contract 用逻辑名（无后缀）；剥离后缀后比对，避免文件名变化误报。
            if key == "checkpoint" and isinstance(actual, str) and isinstance(want, str):
                if actual.removesuffix(".tar").removesuffix(".ckpt") != want:
                    raise ValueError(f"stateful FullSubNet manifest {key}={actual!r},期望（逻辑名）={want!r}: {path}")
                continue
            if actual != want:
                raise ValueError(
                    f"stateful FullSubNet manifest {key}={actual!r},"
                    f"期望={want!r}（与 STATEFUL_FULLSUBNET_CONTRACT 不一致）: {path}"
                )

    def _reset_host_state(self) -> None:
        """Clear STFT, normalization, and overlap-add state without touching the backend."""

        self._fb_sum = np.zeros(BATCH, np.float64)
        self._fb_count = np.zeros(BATCH, np.float64)
        self._sb_sum = np.zeros((BATCH, NUM_FREQS), np.float64)
        self._sb_count = np.zeros((BATCH, NUM_FREQS), np.float64)
        self._stft_tail = np.zeros((HOP, BATCH), np.float32)
        self._raw_spectrum: deque[np.ndarray] = deque()
        self._ola_numerator = np.zeros((HOP, BATCH), np.float64)
        self._ola_denominator = np.zeros(HOP, np.float64)
        self._output_frame_index = 0
        self._timing_sample_counter = 0
        self.last_timing_ms = {}

    def reset(self) -> None:
        """Clear host state and reset recurrent backend state in place."""

        with self._lock:
            if self._closed:
                return
            self._reset_host_state()
            self._executor.reset()

    def _stft_two_frames(self, audio4: np.ndarray) -> np.ndarray:
        """连续512样本生成两个512/256分析帧，返回[B,T=2,F]。"""
        spectra = np.empty((BATCH, 2, NUM_FREQS), np.complex64)
        for step in range(2):
            chunk = audio4[step * HOP : (step + 1) * HOP]
            frame = np.concatenate([self._stft_tail, chunk], axis=0)
            self._stft_tail = chunk.copy()
            spectra[:, step, :] = np.fft.rfft(frame * self._window[:, None], axis=0).T.astype(np.complex64)
        return spectra

    def _normalize_fb(self, magnitude: np.ndarray) -> np.ndarray:
        normalized = np.empty_like(magnitude, dtype=np.float32)
        for step in range(2):
            self._fb_sum += magnitude[:, step, :].sum(axis=1, dtype=np.float64)
            self._fb_count += NUM_FREQS
            mean = self._fb_sum / self._fb_count
            normalized[:, step, :] = magnitude[:, step, :] / (mean[:, None] + EPS)
        return np.ascontiguousarray(normalized)

    def _build_and_normalize_sb(self, magnitude: np.ndarray, fb_output: np.ndarray) -> np.ndarray:
        # 频率轴 reflect pad；每个中心频点保留左右各15个 noisy 特征。
        padded = np.pad(magnitude, ((0, 0), (0, 0), (15, 15)), mode="reflect")
        # 滑窗视图替代 257 次切片+stack；只优化 Host 特征组织，不改变数值顺序。
        noisy = np.lib.stride_tricks.sliding_window_view(padded, 31, axis=2)  # [B,T,F,31]
        noisy = np.ascontiguousarray(noisy.transpose(0, 2, 1, 3))  # [B,F,T,31]
        features = np.concatenate([noisy, fb_output.transpose(0, 2, 1)[..., None]], axis=-1).transpose(
            0, 1, 3, 2
        )  # [B,F,32,T]
        normalized = np.empty_like(features, dtype=np.float32)
        for step in range(2):
            current = features[:, :, :, step]
            self._sb_sum += current.sum(axis=2, dtype=np.float64)
            self._sb_count += SB_FEATURES
            mean = self._sb_sum / self._sb_count
            normalized[:, :, :, step] = current / (mean[:, :, None] + EPS)
        return np.ascontiguousarray(normalized.transpose(0, 1, 3, 2).reshape(SB_BATCH, 2, SB_FEATURES))

    def _synthesis_hop(self, enhanced_spectrum: np.ndarray) -> np.ndarray:
        """单增强谱帧做加窗OLA，输出对应的256 samples×4ch。"""
        frame = np.fft.irfft(enhanced_spectrum, n=N_FFT, axis=0).real
        numerator = frame * self._window[:, None]
        current_num = self._ola_numerator + numerator[:HOP]
        current_den = self._ola_denominator + self._window_square[:HOP]
        output = np.divide(
            current_num,
            current_den[:, None],
            out=np.zeros_like(current_num),
            where=current_den[:, None] > 1e-8,
        )
        self._ola_numerator = numerator[HOP:].astype(np.float64, copy=True)
        self._ola_denominator = self._window_square[HOP:].astype(np.float64, copy=True)
        return output.astype(np.float32)

    def process_4ch(self, audio4: np.ndarray) -> np.ndarray | None:
        """处理连续[512,4]，look-ahead预热时返回None，之后返回[512,4]。"""
        with self._lock:
            if self._closed:
                raise RuntimeError("stateful FullSubNet 增强器已关闭")
            scope = getattr(self._executor, "execution_scope", None)
            with scope() if callable(scope) else nullcontext():
                return self._process_4ch_locked(audio4)

    def _process_4ch_locked(self, audio4: np.ndarray) -> np.ndarray | None:
        """锁内推进全部 Host 与 executor 状态，禁止 reset/close 交叉。"""
        value = np.asarray(audio4, dtype=np.float32)
        if value.shape != (INPUT_SAMPLES, BATCH):
            raise ValueError(f"stateful FullSubNet 输入必须为(512,4)，得到{value.shape}")
        if not np.isfinite(value).all():
            raise ValueError("stateful FullSubNet 输入包含NaN/Inf")

        started = time.perf_counter() if self.timing_enabled else 0.0
        stage = started
        raw = self._stft_two_frames(value)
        magnitude = np.abs(raw).astype(np.float32)
        stft_ms = (time.perf_counter() - stage) * 1000.0 if self.timing_enabled else 0.0
        stage = time.perf_counter() if self.timing_enabled else 0.0
        fb_input = self._normalize_fb(magnitude)
        fb_prepare_ms = (time.perf_counter() - stage) * 1000.0 if self.timing_enabled else 0.0
        fb_output = self._executor.run_fb(fb_input)
        fb_infer_ms = (
            float(getattr(self._executor, "last_timing_ms", {}).get("fb_infer_ms", 0.0)) if self.timing_enabled else 0.0
        )
        fb_output = np.asarray(fb_output, dtype=np.float32)
        stage = time.perf_counter() if self.timing_enabled else 0.0
        sb_input = self._build_and_normalize_sb(magnitude, fb_output)
        sb_prepare_ms = (time.perf_counter() - stage) * 1000.0 if self.timing_enabled else 0.0
        masks = self._executor.run_sb(sb_input)
        sb_infer_ms = (
            float(getattr(self._executor, "last_timing_ms", {}).get("sb_infer_ms", 0.0)) if self.timing_enabled else 0.0
        )
        masks = np.asarray(masks, dtype=np.float32).reshape(BATCH, NUM_FREQS, 2, 2)

        stage = time.perf_counter() if self.timing_enabled else 0.0
        output_hops = []
        for step in range(2):
            self._raw_spectrum.append(raw[:, step, :].T.copy())  # [F,B]
            if len(self._raw_spectrum) <= LOOK_AHEAD:
                continue
            aligned_raw = self._raw_spectrum.popleft()
            mask = masks[:, :, step, :]
            crm = (_decompress_cirm(mask[:, :, 0]) + 1j * _decompress_cirm(mask[:, :, 1])).T.astype(
                np.complex64
            )  # [F,B]
            output_hops.append(self._synthesis_hop(aligned_raw * crm))
            self._output_frame_index += 1
        if not output_hops:
            return None
        if len(output_hops) != 2:
            raise RuntimeError(f"stateful FullSubNet 通道/帧不同步: {len(output_hops)}")
        result = np.concatenate(output_hops, axis=0)
        postprocess_ms = (time.perf_counter() - stage) * 1000.0 if self.timing_enabled else 0.0
        if result.shape != (INPUT_SAMPLES, BATCH) or not np.isfinite(result).all():
            raise RuntimeError(f"stateful FullSubNet 输出错误: {result.shape}")
        if self.timing_enabled:
            self._timing_sample_counter += 1
            self.last_timing_ms = {
                "stft_ms": stft_ms,
                "fb_prepare_ms": fb_prepare_ms,
                "fb_infer_ms": fb_infer_ms,
                "sb_prepare_ms": sb_prepare_ms,
                "sb_infer_ms": sb_infer_ms,
                "postprocess_ms": postprocess_ms,
                "fullsubnet_total_ms": (time.perf_counter() - started) * 1000.0,
            }
            if self._timing_sample_counter % 32 == 0:
                logger.info(
                    "[FullSubNetTiming] n=%d %s",
                    self._timing_sample_counter,
                    " ".join(f"{key}={timing:.3f}ms" for key, timing in self.last_timing_ms.items()),
                )
        return result

    def close(self, *, close_executor: bool = True) -> None:
        """best-effort terminal 关闭：释放后端 executor 资源。

        关闭一旦开始即禁新推理；重入只在清理已尝试完毕后才直接返回，
        属 best-effort terminal，非可重试。
        """
        with self._lock:
            if self._cleanup_complete:
                # 清理已尝试完毕（含失败项），重入直接返回，避免重复释放已成功资源。
                return
            self._closed = True  # 禁新推理
            self._clear_host_state()
            try:
                if close_executor:
                    self._executor.close()
            finally:
                # A host-only close leaves backend ownership to the assembly;
                # the later backend-resource close must still be able to run.
                if close_executor:
                    self._cleanup_complete = True

    def close_host(self) -> None:
        """Invalidate host state while leaving the backend executor open."""

        with self._lock:
            if self._cleanup_complete:
                return
            self._closed = True
            self._clear_host_state()

    def _clear_host_state(self) -> None:
        self._fb_sum = np.zeros(0, np.float64)
        self._fb_count = np.zeros(0, np.float64)
        self._sb_sum = np.zeros((0, 0), np.float64)
        self._sb_count = np.zeros((0, 0), np.float64)
        self._stft_tail = np.zeros((0, BATCH), np.float32)
        self._raw_spectrum.clear()
        self._ola_numerator = np.zeros((0, BATCH), np.float64)
        self._ola_denominator = np.zeros(0, np.float64)


__all__ = ["StatefulFullSubNetEnhancer"]
