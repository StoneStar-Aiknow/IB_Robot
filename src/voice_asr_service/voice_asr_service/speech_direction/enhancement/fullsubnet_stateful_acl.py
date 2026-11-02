"""FullSubNet FB/SB Ascend ACL 状态常驻执行器。"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from inference_service.backends.ascend.acl_runtime import AclRuntimeManager
from inference_service.backends.ascend.model import AclDeviceBuffer, AclModel
from inference_service.backends.errors import BackendLifecycleError
from inference_service.unified_runtime.lifecycle import lifecycle_lock
from voice_asr_service.speech_direction.contract import speech_direction_bindings

logger = logging.getLogger(__name__)


class _StatefulAclStage:
    """一个OM的双bank hidden/cell Device回环。"""

    def __init__(self, model: AclModel):
        self.model = model
        self.bank = 0
        self._states: tuple[tuple[AclDeviceBuffer, AclDeviceBuffer], ...] = ()
        self._prepare()

    def _prepare(self) -> None:
        descriptors = self.model.input_descriptors
        if len(descriptors) != 3 or len(self.model.output_descriptors) != 3:
            raise ValueError(f"{self.model.role}必须有3个输入和3个输出")
        # 两组h/c互为下一轮输入和本轮输出，运行期不做state memcpy。
        state0 = (
            self.model.allocate_device_buffer(descriptors[1].size),
            self.model.allocate_device_buffer(descriptors[2].size),
        )
        state1 = (
            self.model.allocate_device_buffer(descriptors[1].size),
            self.model.allocate_device_buffer(descriptors[2].size),
        )
        self._states = (state0, state1)
        input_banks = []
        output_banks = []
        for bank in range(2):
            current = self._states[bank]
            following = self._states[1 - bank]
            input_banks.append({1: current[0], 2: current[1]})
            output_banks.append({1: following[0], 2: following[1]})
        self.model.prepare_dataset_banks(tuple(input_banks), tuple(output_banks), host_output_indices={0})
        self.reset()

    def run(self, frame: np.ndarray) -> np.ndarray:
        value = np.asarray(frame, dtype=np.float32)
        expected = self.model.input_descriptors[0].shape
        if expected is not None and value.shape != expected:
            raise ValueError(f"{self.model.role}输入必须为{expected}，得到{value.shape}")
        output = self.model.execute_bank(self.bank, {0: np.ascontiguousarray(value)})[0]
        self.bank = 1 - self.bank
        return np.asarray(output, dtype=np.float32)

    def reset(self) -> None:
        for state in self._states:
            for buffer in state:
                self.model.zero_device_buffer(buffer)
        self.bank = 0


class StatefulAclFullSubNetRunner:
    """FB/SB OM、dataset和状态buffer全生命周期常驻的 Ascend ACL executor。"""

    backend = "ascend"

    def __init__(
        self,
        fb_om_path: str,
        sb_om_path: str,
        *,
        device_id: int = 0,
        runtime_manager: AclRuntimeManager | None = None,
        cleanup=None,
        timing_enabled: bool = False,
    ):
        for path in (fb_om_path, sb_om_path):
            if not Path(path).is_file():
                raise FileNotFoundError(f"FullSubNet Ascend ACL OM不存在: {path}")
        self._lock = threading.Lock()
        self._closed = False  # 关闭一旦开始即禁新推理
        self._cleanup_complete = False  # 仅在所有资源释放成功后置位
        self._poisoned = False
        self._timing_enabled = bool(timing_enabled)
        self._timing_ms = {"fb_infer_ms": 0.0, "sb_infer_ms": 0.0}
        if runtime_manager is None:
            raise RuntimeError("StatefulAclFullSubNetRunner requires an explicitly injected ACL runtime provider")
        self._lease = runtime_manager.acquire(device_id)
        self._fb_model: AclModel | None = None
        self._sb_model: AclModel | None = None
        if cleanup is not None:
            cleanup.defer(self.close)
        try:
            self._fb_model = AclModel(
                self._lease, "fullsubnet_fb", Path(fb_om_path), speech_direction_bindings("fullsubnet_fb")
            )
            self._fb_model.load_descriptor()
            self._fb = _StatefulAclStage(self._fb_model)
            self._sb_model = AclModel(
                self._lease, "fullsubnet_sb", Path(sb_om_path), speech_direction_bindings("fullsubnet_sb")
            )
            self._sb_model.load_descriptor()
            self._sb = _StatefulAclStage(self._sb_model)
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
        logger.info("FullSubNet Ascend ACL状态常驻已加载: fb=%s sb=%s", fb_om_path, sb_om_path)

    @property
    def last_timing_ms(self):
        with self._lock:
            return dict(self._timing_ms)

    def run_fb(self, frame: np.ndarray) -> np.ndarray:
        started = time.perf_counter() if self._timing_enabled else 0.0
        result = self._run(self._fb, frame)
        if self._timing_enabled:
            with self._lock:
                self._timing_ms["fb_infer_ms"] = (time.perf_counter() - started) * 1000.0
        return result

    def run_sb(self, frame: np.ndarray) -> np.ndarray:
        started = time.perf_counter() if self._timing_enabled else 0.0
        result = self._run(self._sb, frame)
        if self._timing_enabled:
            with self._lock:
                self._timing_ms["sb_infer_ms"] = (time.perf_counter() - started) * 1000.0
        return result

    def _run(self, stage: _StatefulAclStage, frame: np.ndarray) -> np.ndarray:
        with self._lock:
            if self._closed:
                raise RuntimeError("FullSubNet Ascend ACL runner已关闭")
            if self._poisoned:
                raise RuntimeError("FullSubNet Ascend ACL状态不确定，必须先reset")
            try:
                return stage.run(frame)
            except Exception:
                # ACL execute失败后输出state可能只写入了一部分，禁止继续推进。
                self._poisoned = True
                raise

    def reset(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._fb.reset()
            self._sb.reset()
            self._poisoned = False
            self._timing_ms = {"fb_infer_ms": 0.0, "sb_infer_ms": 0.0}

    def close(self) -> None:
        """按依赖顺序关闭，失败后保留模型与 lease 供重试。"""
        self._closed = True
        try:
            with lifecycle_lock(self._lock):
                if self._cleanup_complete:
                    return
                for name in ("_sb_model", "_fb_model"):
                    model = getattr(self, name)
                    if model is not None:
                        model.close()
                        setattr(self, name, None)
                self._lease.close()
                self._cleanup_complete = True
        except Exception as exc:
            raise BackendLifecycleError(str(exc), close_pending=True) from exc


__all__ = ["StatefulAclFullSubNetRunner"]
