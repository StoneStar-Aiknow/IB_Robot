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

from inference_service.backends.errors import BackendLifecycleError
from inference_service.model_sessions.torch_completion import (
    drain_torch_device,
    drain_torch_device_before_release,
    resolve_torch_device,
    torch_execution_scope,
)
from inference_service.unified_runtime.errors import CancellationRequested, DeadlineExceeded
from inference_service.unified_runtime.lifecycle import check_lifecycle, lifecycle_lock

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
        cleanup=None,
    ):
        if device not in {"cuda", "cpu"}:
            raise ValueError("stateful Torch executor 的 device 只能为 cuda 或 cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("stateful Torch/CUDA 后端要求 CUDA 可用，禁止静默回退 CPU")
        self.device = resolve_torch_device(torch, device)
        self.backend = "stateful_torch_cuda" if device == "cuda" else "stateful_torch_cpu"
        self._lock = threading.Lock()
        self._closed = False  # 关闭一旦开始即禁新推理
        self._cleanup_complete = False  # 仅在设备工作完成且引用释放后标记成功
        self._poisoned = False
        self._release_drain_failures = 0  # 连续 drain 失败计数，达到上限后放弃等待
        self._timing_enabled = bool(timing_enabled)
        self._timing_ms = {"fb_infer_ms": 0.0, "sb_infer_ms": 0.0}

        self._fb_model = self._sb_model = None
        if cleanup is not None:
            cleanup.defer(self.close)
        try:
            self._load_models(checkpoint_path, manifest_path)
        except Exception as exc:
            from inference_service.unified_runtime.errors import clear_exception_frames

            clear_exception_frames(exc)
            if cleanup is None:
                try:
                    self.close()
                except Exception as cleanup_error:
                    exc.cleanup_owner = self
                    exc.close_pending = True
                    exc.rollback_errors = (str(cleanup_error),)
            raise

    def _load_models(self, checkpoint_path, manifest_path):
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
                with torch_execution_scope(torch, self.device.type, self.device, stateful=True), torch.inference_mode():
                    output, self._fb_hidden, self._fb_cell = self._fb_model(
                        torch.from_numpy(value).to(self.device), self._fb_hidden, self._fb_cell
                    )
                    result = output.detach().cpu().numpy().astype(np.float32, copy=False)
                    if result.shape != FB_OUTPUT_SHAPE:
                        from inference_service.unified_runtime import OutcomeEvidence

                        error = RuntimeError(f"FullSubNet FB 输出必须为{FB_OUTPUT_SHAPE}，得到{result.shape}")
                        error.evidence = OutcomeEvidence.completed("fullsubnet.fb.outputs", state_mutated=True)
                        raise error
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
                with torch_execution_scope(torch, self.device.type, self.device, stateful=True), torch.inference_mode():
                    output, self._sb_hidden, self._sb_cell = self._sb_model(
                        torch.from_numpy(value).to(self.device), self._sb_hidden, self._sb_cell
                    )
                    result = output.detach().cpu().numpy().astype(np.float32, copy=False)
                    if result.shape != SB_OUTPUT_SHAPE:
                        from inference_service.unified_runtime import OutcomeEvidence

                        error = RuntimeError(f"FullSubNet SB 输出必须为{SB_OUTPUT_SHAPE}，得到{result.shape}")
                        error.evidence = OutcomeEvidence.completed("fullsubnet.sb.outputs", state_mutated=True)
                        raise error
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
        with lifecycle_lock(self._lock):
            if self._closed:
                return
            check_lifecycle()
            if self._poisoned:
                drain_torch_device(torch, self.device.type, self.device)
            with torch_execution_scope(torch, self.device.type, self.device, complete_on_success=True, stateful=True):
                self._reset_states()
            self._poisoned = False
            self._timing_ms = {"fb_infer_ms": 0.0, "sb_infer_ms": 0.0}

    def close(self) -> None:
        """Fence device work before releasing references; retain failed cleanup."""
        self._closed = True
        try:
            with lifecycle_lock(self._lock):
                if self._cleanup_complete:
                    return
                # 设备卡在 sticky error 时永远不会报告完成；连续 drain 失败达到
                # 上限后放弃等待并释放引用，让 close 可以终止（见
                # drain_torch_device_before_release）。周期/取消异常不计入失败次数，
                # 换新预算的关闭仍可重试。
                try:
                    check_lifecycle()
                    drain_torch_device_before_release(
                        torch,
                        self.device.type,
                        self.device,
                        failed_attempts=self._release_drain_failures,
                        owner="FullSubNet stateful Torch close",
                    )
                except (DeadlineExceeded, CancellationRequested) as exc:
                    raise BackendLifecycleError(
                        f"FullSubNet device completion pending: {exc}", close_pending=True
                    ) from exc
                except BackendLifecycleError:
                    self._release_drain_failures += 1
                    raise
                self._fb_model = None
                self._sb_model = None
                for name in ("_fb_hidden", "_fb_cell", "_sb_hidden", "_sb_cell"):
                    setattr(self, name, None)
                self._cleanup_complete = True
        except Exception as exc:
            raise BackendLifecycleError(f"FullSubNet device completion pending: {exc}", close_pending=True) from exc


__all__ = [
    "StatefulFullBandT2",
    "StatefulSubBandT2",
    "StatefulTorchFullSubNetExecutor",
]
