"""
ACTWrapper.py

加载 act om 模型并推理
"""

import numpy as np
import torch
from torch import Tensor
import torch_npu
torch.npu.set_compile_mode(jit_compile=False)

from .OMmodel import OMmodel
from typing import Any


def logger(msg: str):
    print(f"[ACTWrapper]: {msg}")


class ACTWrapper:
    def __init__(self, model_path: str, config: Any):
        self.om_model = OMmodel(model_path)
        chunk_size = config.chunk_size
        action_dim = config.output_features["action"].shape
        self.input_features = config.input_features
        self.output_shape = [1, chunk_size, *action_dim]
        logger(f"Loaded ACT OM model from {model_path}, output shape: {self.output_shape}")

    def predict(self, batch: dict[str, Tensor]) -> tuple:
        input_arr = []

        for f in self.input_features:
            if f not in batch:
                logger(f"WARN: key: {f} not in batch, skipping")
                continue
            input_arr.append(batch[f].cpu().numpy())

        output = self.om_model.forward(input_arr)[0]

        o_tensor = torch.from_numpy(np.array(output, dtype=np.float32)).reshape(*self.output_shape)
        return (o_tensor,)
