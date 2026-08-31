"""FullSubNet cumulative T=2 的 Torch/CUDA stateful 执行器。"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .fullsubnet_stateful_executor import (
    FB_FRAME_SHAPE,
    FB_OUTPUT_SHAPE,
    FB_STATE_SHAPE,
    NUM_FREQS,
    SB_FRAME_SHAPE,
    SB_HIDDEN,
    SB_OUTPUT_SHAPE,
    SB_STATE_SHAPE,
)

logger = logging.getLogger(__name__)

_MODEL_KW = {
    "num_freqs": NUM_FREQS,
    "look_ahead": 2,
    "sequence_model": "LSTM",
    "fb_num_neighbors": 0,
    "sb_num_neighbors": 15,
    "fb_output_activate_function": "ReLU",
    "sb_output_activate_function": False,
    "fb_model_hidden_size": 512,
    "sb_model_hidden_size": SB_HIDDEN,
    "norm_type": "cumulative_laplace_norm",
    "num_groups_in_drop_band": 2,
    "weight_init": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_model_class():
    """Load the upstream FullSubNet Model class from the installed wheel package."""
    from fullsubnet.model import Model

    return Model


class StatefulFullBandT2(nn.Module):
    """FB 两帧 stateful 网络，结构与导出 OM 的 wrapper 保持一致。"""

    def __init__(self, model):
        super().__init__()
        self.lstm = model.fb_model.sequence_model
        self.fc = model.fb_model.fc_output_layer
        self.activation = model.fb_model.activate_function

    def forward(self, frame, hidden, cell):
        output, (next_hidden, next_cell) = self.lstm(frame, (hidden, cell))
        return self.activation(self.fc(output)), next_hidden, next_cell


class StatefulSubBandT2(nn.Module):
    """SB 两帧 stateful 网络，batch 固定为 4*257。"""

    def __init__(self, model):
        super().__init__()
        self.lstm = model.sb_model.sequence_model
        self.fc = model.sb_model.fc_output_layer

    def forward(self, frame, hidden, cell):
        output, (next_hidden, next_cell) = self.lstm(frame, (hidden, cell))
        return self.fc(output), next_hidden, next_cell


class StatefulTorchFullSubNetExecutor:
    """让 FB/SB 模型与 h/c 常驻 Torch 设备，仅业务输入输出经过 Host。"""

    backend = "stateful_torch_cuda"

    def __init__(
        self,
        checkpoint_path: str,
        manifest_path: str = "",
        *,
        device: str = "cuda",
        timing_enabled: bool = False,
    ):
        if device not in {"cuda", "cpu"}:
            raise ValueError("stateful Torch executor 的 device 只能为 cuda 或 cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("stateful Torch/CUDA 后端要求 CUDA 可用，禁止静默回退 CPU")
        self.device = torch.device(device)
        self.backend = "stateful_torch_cuda" if device == "cuda" else "stateful_torch_cpu"
        self._lock = threading.Lock()
        self._closed = False  # 关闭一旦开始即禁新推理
        self._cleanup_complete = False  # 清理已尝试完毕（含失败项），重入据此返回
        self._poisoned = False
        self._timing_enabled = bool(timing_enabled)
        self._timing_ms = {"fb_infer_ms": 0.0, "sb_infer_ms": 0.0}

        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"FullSubNet cumulative checkpoint 不存在: {checkpoint}")
        manifest = None
        if manifest_path:
            manifest_file = Path(manifest_path)
            if not manifest_file.is_file():
                raise FileNotFoundError(f"FullSubNet cumulative manifest 不存在: {manifest_file}")
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if manifest.get("norm_type") != "cumulative_laplace_norm":
                raise ValueError(f"FullSubNet cumulative manifest norm_type 错误: {manifest_file}")
            expected_sha = manifest.get("sha256") or manifest.get("checkpoint_sha256")
            if expected_sha and _sha256(checkpoint) != expected_sha:
                raise ValueError(f"FullSubNet cumulative checkpoint SHA-256 错误: {checkpoint}")

        Model = _load_model_class()
        model = Model(**_MODEL_KW)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        result = model.load_state_dict(payload["model"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                "FullSubNet cumulative checkpoint 参数不匹配: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        model.eval()
        self._fb_model = StatefulFullBandT2(model).eval().to(self.device)
        self._sb_model = StatefulSubBandT2(model).eval().to(self.device)
        self._reset_states()
        logger.info("FullSubNet stateful Torch 已加载: checkpoint=%s device=%s", checkpoint, self.device)

    def _reset_states(self) -> None:
        self._fb_hidden = torch.zeros(FB_STATE_SHAPE, dtype=torch.float32, device=self.device)
        self._fb_cell = torch.zeros_like(self._fb_hidden)
        self._sb_hidden = torch.zeros(SB_STATE_SHAPE, dtype=torch.float32, device=self.device)
        self._sb_cell = torch.zeros_like(self._sb_hidden)

    @staticmethod
    def _validate(value: np.ndarray, expected: tuple[int, ...], label: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float32)
        if result.shape != expected:
            raise ValueError(f"{label} 输入必须为{expected}，得到{result.shape}")
        if not np.isfinite(result).all():
            raise ValueError(f"{label} 输入包含 NaN/Inf")
        return np.ascontiguousarray(result)

    @property
    def last_timing_ms(self):
        with self._lock:
            return dict(self._timing_ms)

    def run_fb(self, frame: np.ndarray) -> np.ndarray:
        value = self._validate(frame, FB_FRAME_SHAPE, "FullSubNet FB")
        started = time.perf_counter() if self._timing_enabled else 0.0
        with self._lock:
            self._ensure_runnable()
            try:
                with torch.inference_mode():
                    output, self._fb_hidden, self._fb_cell = self._fb_model(
                        torch.from_numpy(value).to(self.device), self._fb_hidden, self._fb_cell
                    )
                result = output.detach().cpu().numpy().astype(np.float32, copy=False)
                if result.shape != FB_OUTPUT_SHAPE:
                    raise RuntimeError(f"FullSubNet FB 输出必须为{FB_OUTPUT_SHAPE}，得到{result.shape}")
                if self._timing_enabled:
                    self._timing_ms["fb_infer_ms"] = (time.perf_counter() - started) * 1000.0
                return result
            except Exception:
                self._poisoned = True
                raise

    def run_sb(self, frame: np.ndarray) -> np.ndarray:
        value = self._validate(frame, SB_FRAME_SHAPE, "FullSubNet SB")
        started = time.perf_counter() if self._timing_enabled else 0.0
        with self._lock:
            self._ensure_runnable()
            try:
                with torch.inference_mode():
                    output, self._sb_hidden, self._sb_cell = self._sb_model(
                        torch.from_numpy(value).to(self.device), self._sb_hidden, self._sb_cell
                    )
                result = output.detach().cpu().numpy().astype(np.float32, copy=False)
                if result.shape != SB_OUTPUT_SHAPE:
                    raise RuntimeError(f"FullSubNet SB 输出必须为{SB_OUTPUT_SHAPE}，得到{result.shape}")
                if self._timing_enabled:
                    self._timing_ms["sb_infer_ms"] = (time.perf_counter() - started) * 1000.0
                return result
            except Exception:
                # FB 已推进而 SB 失败时状态已不一致，必须整层 reset 后才能继续。
                self._poisoned = True
                raise

    def _ensure_runnable(self) -> None:
        if self._closed:
            raise RuntimeError("FullSubNet stateful Torch executor 已关闭")
        if self._poisoned:
            raise RuntimeError("FullSubNet stateful Torch 状态不确定，必须先 reset")

    def reset(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._reset_states()
            self._poisoned = False
            self._timing_ms = {"fb_infer_ms": 0.0, "sb_infer_ms": 0.0}

    def close(self) -> None:
        """best-effort terminal 关闭：释放模型与状态引用，单项失败不阻断其余清理。

        关闭一旦开始即禁新推理；重入只在清理已尝试完毕后才直接返回，
        属 best-effort terminal，非可重试。
        """
        with getattr(self, "_lock", threading.Lock()):
            if getattr(self, "_cleanup_complete", False):
                # 清理已尝试完毕（含失败项），重入直接返回。
                return
            self._closed = True  # 禁新推理
            self._fb_model = None
            self._sb_model = None
            for name in ("_fb_hidden", "_fb_cell", "_sb_hidden", "_sb_cell"):
                setattr(self, name, None)
            self._cleanup_complete = True


__all__ = [
    "StatefulFullBandT2",
    "StatefulSubBandT2",
    "StatefulTorchFullSubNetExecutor",
]
