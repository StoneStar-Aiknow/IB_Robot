"""Cumulative stateful FullSubNet 的固定算法契约与执行器接口。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

N_FFT = 512
STFT_HOP = 256
INPUT_SAMPLES = 512
TIME_STEPS = 2
NUM_FREQS = 257
BATCH = 4
FB_HIDDEN = 512
SB_HIDDEN = 384
NUM_LAYERS = 2
SB_FEATURES = 32
SB_BATCH = BATCH * NUM_FREQS
LOOK_AHEAD = 2
NORM_TYPE = "cumulative_laplace_norm"
HOST_DTYPE = np.dtype(np.float32)

# FullSubNet cumulative 218epochs checkpoint 的逻辑模型身份；与 manifest 的
# algorithm_contract.family / checkpoint 字段对齐，SSOT，散落处不再各自硬编码。
FAMILY = "fullsubnet_cumulative_stateful"
CHECKPOINT = "cum_fullsubnet_best_model_218epochs"

# SRP-PHAT DOA 的帧/步长，与 manifest 的 srp_* 字段对齐。
SRP_FRAME_SAMPLES = 4096
SRP_HOP_SAMPLES = 512

FB_FRAME_SHAPE = (BATCH, TIME_STEPS, NUM_FREQS)
FB_STATE_SHAPE = (NUM_LAYERS, BATCH, FB_HIDDEN)
FB_OUTPUT_SHAPE = FB_FRAME_SHAPE
SB_FRAME_SHAPE = (SB_BATCH, TIME_STEPS, SB_FEATURES)
SB_STATE_SHAPE = (NUM_LAYERS, SB_BATCH, SB_HIDDEN)
SB_OUTPUT_SHAPE = (SB_BATCH, TIME_STEPS, 2)


@dataclass(frozen=True)
class StatefulFullSubNetContract:
    """两个平台必须一致的算法和静态 ABI，部署配置不得改写这些值。

    作为 manifest algorithm_contract 的唯一来源（SSOT）：打包脚本据此写 manifest，
    校验脚本与 _verify_manifest 据此逐项比对，避免 Python/JSON/shell 三处各写一份。
    """

    family: str = FAMILY
    checkpoint: str = CHECKPOINT
    n_fft: int = N_FFT
    stft_hop: int = STFT_HOP
    input_samples: int = INPUT_SAMPLES
    time_steps: int = TIME_STEPS
    batch: int = BATCH
    num_freqs: int = NUM_FREQS
    sb_features: int = SB_FEATURES
    look_ahead: int = LOOK_AHEAD
    norm_type: str = NORM_TYPE
    srp_frame_samples: int = SRP_FRAME_SAMPLES
    srp_hop_samples: int = SRP_HOP_SAMPLES
    host_dtype: str = "float32"


STATEFUL_FULLSUBNET_CONTRACT = StatefulFullSubNetContract()


@runtime_checkable
class StatefulFullSubNetExecutor(Protocol):
    """仅隔离神经网络执行；STFT、归一化和 OLA 仍由公共 Host 层负责。"""

    backend: str

    @property
    def last_timing_ms(self) -> Mapping[str, float]:
        """最近一次 FB/SB 执行耗时快照；后端实现返回副本。"""

    def run_fb(self, frame: np.ndarray) -> np.ndarray:
        """执行 FB T=2 网络，输入输出均使用固定 Host float32 ABI。"""

    def run_sb(self, frame: np.ndarray) -> np.ndarray:
        """执行 SB T=2 网络，输入输出均使用固定 Host float32 ABI。"""

    def reset(self) -> None:
        """清空 FB/SB recurrent state，保留模型与执行资源。"""

    def close(self) -> None:
        """best-effort terminal 释放后端模型、状态和设备资源；重入直接返回，非可重试。"""


__all__ = [
    "BATCH",
    "CHECKPOINT",
    "FAMILY",
    "FB_FRAME_SHAPE",
    "FB_HIDDEN",
    "FB_OUTPUT_SHAPE",
    "FB_STATE_SHAPE",
    "HOST_DTYPE",
    "INPUT_SAMPLES",
    "LOOK_AHEAD",
    "NORM_TYPE",
    "N_FFT",
    "NUM_FREQS",
    "NUM_LAYERS",
    "SB_BATCH",
    "SB_FEATURES",
    "SB_FRAME_SHAPE",
    "SB_HIDDEN",
    "SB_OUTPUT_SHAPE",
    "SB_STATE_SHAPE",
    "SRP_FRAME_SAMPLES",
    "SRP_HOP_SAMPLES",
    "STATEFUL_FULLSUBNET_CONTRACT",
    "STFT_HOP",
    "StatefulFullSubNetContract",
    "StatefulFullSubNetExecutor",
    "TIME_STEPS",
]
