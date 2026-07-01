"""SmolVLA RKNN multi-module pipeline orchestrator for the RK3588 NPU backend.

Mirrors the HMM 3-module split (vision -> prefill -> action) but uses
``RKNNLite`` instead of ``tcim_lite.runtime``. Key differences from HMM:

- No device-pointer KV-cache sharing (RKNNLite does not expose
  ``get_dev_output`` / ``set_dev_input``); the prefill outputs 32 separate
  KV tensors which the host flattens into a single ``past_kv_tensor`` before
  feeding the action (expert) module.
- The action module uses the original expert export (4-input flattened KV
  contract: ``past_kv_tensor + prefix_pad_masks + time + noise``), not the
  HMM-style 35-input contract.
- NHWC layout conversion is applied only to the vision image input.
- The flow-matching denoise loop (10 steps) runs on the host CPU, calling
  the action NPU module each step.
- All 3 RKNN modules stay resident in NPU memory. Load order matters:
  prefill must load before vision due to IOMMU address-space fragmentation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

_TARGET_DTYPE = np.float16


def _logger(msg: str) -> None:
    print(f"[SmolVLARKNNModel]: {msg}")


def _to_nhwc(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4:
        return np.ascontiguousarray(np.transpose(arr, (0, 2, 3, 1)))
    return arr


class SmolVLARKNNModel:
    """Orchestrator for the 3-module RKNN SmolVLA pipeline (vision/prefill/action).

    Args:
        vision_path / prefill_path / action_path: paths to ``.rknn`` modules.
        embedding_path: path to ``token_embedding.pt`` (SmolVLM2 text embedding).
        config: policy config view exposing chunk_size / max_action_dim / num_steps.
    """

    def __init__(
        self,
        vision_path: str,
        prefill_path: str,
        action_path: str,
        embedding_path: str,
        config: Any,
    ) -> None:
        self.config = config
        self.chunk_size = getattr(config, "chunk_size", 50)
        self.max_action_dim = getattr(config, "max_action_dim", 32)
        self.num_steps = int(getattr(config, "num_steps", None) or 10)
        self.min_period = getattr(config, "min_period", 0.004)
        self.max_period = getattr(config, "max_period", 4.0)

        self.num_layers = getattr(config, "num_layers", 16) or 16
        self.prefix_seq_len = getattr(config, "prefix_length", 177) or 177
        self.prefix_hidden_size = getattr(config, "prefix_hidden_size", 960) or 960

        _logger(f"Loading prefill RKNN from {prefill_path}")
        self.prefill_rknn = self._load_rknn(prefill_path)
        _logger(f"Loading vision RKNN from {vision_path}")
        self.vision_rknn = self._load_rknn(vision_path)
        _logger(f"Loading action RKNN from {action_path}")
        self.action_rknn = self._load_rknn(action_path)

        weight = torch.load(embedding_path, map_location="cpu", weights_only=True)
        w = weight["weight"] if "weight" in weight else next(iter(weight.values()))
        self.embed_tokens = torch.nn.Embedding(
            num_embeddings=w.shape[0],
            embedding_dim=w.shape[1],
        )
        self.embed_tokens.load_state_dict({"weight": w} if "weight" not in weight else weight)

        _logger(
            f"SmolVLARKNNModel ready (prefix_seq_len={self.prefix_seq_len}, "
            f"num_layers={self.num_layers}, hidden={self.prefix_hidden_size}, steps={self.num_steps})"
        )

    def _load_rknn(self, path: str) -> Any:
        from rknnlite.api import RKNNLite

        rknn = RKNNLite()
        ret = rknn.load_rknn(path)
        if ret != 0:
            raise RuntimeError(f"RKNN load_rknn failed for {path}: ret={ret}")
        ret = rknn.init_runtime(target=None, core_mask=RKNNLite.NPU_CORE_ALL)
        if ret != 0:
            raise RuntimeError(f"RKNN init_runtime failed for {path}: ret={ret}")
        return rknn

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != _TARGET_DTYPE:
            image = image.astype(_TARGET_DTYPE)
        image_nhwc = _to_nhwc(np.ascontiguousarray(image))
        outputs = self.vision_rknn.inference(inputs=[image_nhwc])
        if not outputs:
            raise RuntimeError("Vision RKNN returned no outputs")
        return np.asarray(outputs[0])

    def embed_prefix(
        self,
        images: list[np.ndarray],
        img_masks: list[np.ndarray],
        tokens: Tensor,
        masks: Tensor,
        state: Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        embs: list[np.ndarray] = []
        pad_masks: list[np.ndarray] = []
        for img, img_mask in zip(images, img_masks, strict=False):
            img_emb = self.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]
            img_emb = img_emb * math.sqrt(img_emb.shape[-1])
            embs.append(img_emb)
            pad_masks.append(np.broadcast_to(img_mask[:, None], (bsize, num_img_embs)).copy())
        lang_emb = self.embed_tokens(tokens.long())
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb.detach().cpu().numpy().astype(_TARGET_DTYPE))
        pad_masks.append(masks.detach().cpu().numpy())
        prefix_embs = np.concatenate(embs, axis=1)
        prefix_pad_masks = np.concatenate(pad_masks, axis=1)
        if prefix_embs.shape[1] < self.prefix_seq_len:
            pad_len = self.prefix_seq_len - prefix_embs.shape[1]
            prefix_embs = np.concatenate(
                [prefix_embs, np.zeros((prefix_embs.shape[0], pad_len, prefix_embs.shape[2]), dtype=_TARGET_DTYPE)],
                axis=1,
            )
            prefix_pad_masks = np.concatenate(
                [prefix_pad_masks, np.zeros((prefix_pad_masks.shape[0], pad_len), dtype=bool)], axis=1
            )
        return prefix_embs[:, : self.prefix_seq_len], prefix_pad_masks[:, : self.prefix_seq_len]

    def prefill(self, prefix_embs: np.ndarray, prefix_pad_masks: np.ndarray) -> list[np.ndarray]:
        attention_mask = np.ones((prefix_embs.shape[0], prefix_embs.shape[1], prefix_embs.shape[1]), dtype=np.int64)
        position_ids = np.cumsum(prefix_pad_masks.astype(np.int64), axis=1) - 1
        position_ids = np.where(prefix_pad_masks, position_ids, 0).astype(np.int64)
        outputs = self.prefill_rknn.inference(
            inputs=[
                np.ascontiguousarray(prefix_embs.astype(_TARGET_DTYPE)),
                np.ascontiguousarray(attention_mask),
                np.ascontiguousarray(position_ids),
            ]
        )
        if not outputs:
            raise RuntimeError("Prefill RKNN returned no outputs")
        return list(outputs)

    def _flatten_kv(self, flat_cache: list[np.ndarray]) -> np.ndarray:
        stacked = np.stack(flat_cache, axis=0)
        return stacked.reshape(self.num_layers, 2, 1, *stacked.shape[1:]).astype(_TARGET_DTYPE)

    def action(
        self,
        past_kv_tensor: np.ndarray,
        prefix_pad_masks: np.ndarray,
        time: np.ndarray,
        noise: np.ndarray,
    ) -> np.ndarray:
        outputs = self.action_rknn.inference(
            inputs=[
                np.ascontiguousarray(past_kv_tensor.astype(_TARGET_DTYPE)),
                np.ascontiguousarray(prefix_pad_masks),
                np.ascontiguousarray(time.astype(np.float32)),
                np.ascontiguousarray(noise.astype(_TARGET_DTYPE)),
            ]
        )
        if not outputs:
            raise RuntimeError("Action RKNN returned no outputs")
        return np.asarray(outputs[0])

    def forward(
        self,
        images: list[np.ndarray],
        tokens: Tensor,
        masks: Tensor,
        prefix_att_2d_masks_4d: np.ndarray | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        del prefix_att_2d_masks_4d
        tokens_t = tokens.to("cpu")
        masks_t = masks.to("cpu")
        img_masks = [np.ones(img.shape[0], dtype=bool) for img in images]

        prefix_embs, prefix_pad_masks = self.embed_prefix(images, img_masks, tokens_t, masks_t)
        flat_cache = self.prefill(prefix_embs, prefix_pad_masks)
        past_kv_tensor = self._flatten_kv(flat_cache)

        x_t = noise.to("cpu") if noise is not None else torch.randn(1, self.chunk_size, self.max_action_dim)
        dt = -1.0 / self.num_steps
        for step in range(self.num_steps):
            time = np.array([1.0 - step / self.num_steps], dtype=np.float32)
            v_t = self.action(past_kv_tensor, prefix_pad_masks, time, x_t.numpy())
            v_t_tensor = torch.from_numpy(np.asarray(v_t)).float()
            x_t = x_t + dt * v_t_tensor.squeeze(0)

        actions = x_t.squeeze(0)
        return actions.to("cpu")

    def close(self) -> None:
        for attr in ("vision_rknn", "prefill_rknn", "action_rknn"):
            mod = getattr(self, attr, None)
            if mod is not None:
                mod.release()
                setattr(self, attr, None)
