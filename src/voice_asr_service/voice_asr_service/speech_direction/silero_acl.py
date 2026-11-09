"""Silero VAD OM 的 Ascend ACL 状态常驻执行器。"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from inference_service.backends.ascend.acl_runtime import AclRuntimeManager
from inference_service.backends.ascend.model import AclDeviceBuffer, AclModel
from inference_service.backends.errors import BackendLifecycleError
from inference_service.unified_runtime.lifecycle import lifecycle_lock
from voice_asr_service.speech_direction.contract import SILERO_AUDIO_SHAPE, speech_direction_bindings

# Silero VAD OM（silero_vad_v6_310p_mixed16，openvino_16k 变体）固定静态 ABI。
# 该 OM 已把采样率折叠为常量，仅保留音频和 LSTM state 两入两出。
# ABI 契约（semantic + shape）由 inference_manifest.speech_direction 统一提供，
# 不再在本文件硬编码 _LooseBindings 私有类。
_AUDIO_SHAPE = SILERO_AUDIO_SHAPE


class SileroVadAclRunner:
    """Silero OM、dataset 和 state 双 bank 全生命周期常驻的 Ascend ACL runner。"""

    def __init__(
        self,
        model_path: str,
        *,
        device_id: int = 0,
        runtime_manager: AclRuntimeManager | None = None,
        cleanup=None,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Silero VAD Ascend ACL OM 不存在: {path}")
        self._lock = threading.Lock()
        self._closed = False  # 关闭一旦开始即禁新推理（reset/inference 见此标志即拒绝）
        self._cleanup_complete = False  # 仅在所有资源释放成功后置位
        self._poisoned = False
        self._bank = 0
        if runtime_manager is None:
            raise RuntimeError("SileroVadAclRunner requires an explicitly injected ACL runtime provider")
        self._lease = runtime_manager.acquire(device_id)
        self._model: AclModel | None = None
        self._states: tuple[AclDeviceBuffer, AclDeviceBuffer] = ()
        if cleanup is not None:
            cleanup.defer(self.close)
        try:
            self._model = AclModel(
                self._lease,
                "silero_vad",
                path,
                speech_direction_bindings("silero_vad"),
            )
            self._model.load_descriptor()
            state_size = self._model.input_descriptors[1].size
            self._states = (
                self._model.allocate_device_buffer(state_size),
                self._model.allocate_device_buffer(state_size),
            )
            self._model.prepare_dataset_banks(
                (
                    {1: self._states[0]},
                    {1: self._states[1]},
                ),
                (
                    {1: self._states[1]},
                    {1: self._states[0]},
                ),
                host_output_indices={0},
            )
            self.reset()
        except Exception as exc:
            if cleanup is not None:
                exc.cleanup_owner = cleanup
                exc.close_pending = True
            else:
                try:
                    self.close()
                except Exception as cleanup_error:
                    exc.cleanup_owner = self
                    exc.close_pending = True
                    exc.rollback_errors = (str(cleanup_error),)
            raise

    def infer(self, audio: np.ndarray) -> float:
        """执行一帧并在 Device 内推进 state，只将概率标量复制回 Host。"""
        value = np.asarray(audio, dtype=np.float32)
        if value.shape != _AUDIO_SHAPE:
            raise ValueError(f"Silero VAD Ascend ACL 输入必须为{_AUDIO_SHAPE}，得到{value.shape}")
        with self._lock:
            if self._closed:
                raise RuntimeError("Silero VAD Ascend ACL runner 已关闭")
            if self._poisoned:
                raise RuntimeError("Silero VAD Ascend ACL 状态不确定，必须先 reset")
            try:
                probability = self._model.execute_bank(self._bank, {0: np.ascontiguousarray(value)})[0]
                self._bank = 1 - self._bank
            except Exception:
                self._poisoned = True
                raise
        return float(np.asarray(probability).reshape(-1)[0])

    def reset(self) -> None:
        with self._lock:
            if self._closed:
                return
            model = self._model
            for state in self._states:
                model.zero_device_buffer(state)
            self._bank = 0
            self._poisoned = False

    def close(self) -> None:
        """保留释放失败的模型及 lease，后续 close 从未完成处重试。"""
        self._closed = True
        try:
            with lifecycle_lock(self._lock):
                if self._cleanup_complete:
                    return
                if self._model is not None:
                    self._model.close()
                    self._model = None
                    self._states = ()
                self._lease.close()
                self._cleanup_complete = True
        except Exception as exc:
            raise BackendLifecycleError(str(exc), close_pending=True) from exc


__all__ = ["SileroVadAclRunner"]
