"""SmolVLA HMM multi-module pipeline orchestrator for the Houmo (LQ50/M50 xh2) backend.

Mirrors ``houmo-examples/vla/smolvla`` over ``tcim_lite.runtime``: the main
inference chain is ``vision -> prefill -> action`` (3 compiled ``.hmm`` modules)
plus the CPU-side token embedding layer, with the flow-matching denoise loop
orchestrated on the host. The KV-cache produced by the prefill module is handed
to the action module via device-pointer sharing
(``action.set_input(name, prefill.get_dev_input(name))``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

# tcim_lite emits fp16 weights; keep host-side math in fp16 to match.
_TARGET_DTYPE = torch.float16


def _logger(msg: str) -> None:
    print(f"[SmolVLAHMMModel]: {msg}")


def _create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device,
) -> torch.Tensor:
    """SmolVLA flow-matching timestep embedding (sine-cosine), openpi-compatible."""
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float64, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 2.0 * math.pi / period
    sin_input = time[:, None].double() * scaling_factor[None, :]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)


class SmolVLAHMMModel:
    """Orchestrator for the 3-module Houmo SmolVLA HMM pipeline (vision/prefill/action).

    Args:
        vision_path / prefill_path / action_path: paths to the compiled ``.hmm``
            modules (resolved from ``config.hmm.json`` by the runtime session).
        embedding_path: path to ``token_embedding.pt`` (SmolVLM2 text embedding weights).
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
        import tcim_lite

        self._tcim = tcim_lite
        self.config = config
        self.chunk_size = getattr(config, "chunk_size", 50)
        self.max_action_dim = getattr(config, "max_action_dim", 32)
        # SmolVLA's denoise loop uses ``num_steps`` (flow-matching steps), not
        # PI05's ``num_inference_steps``; fall back to config if absent.
        self.num_steps = int(getattr(config, "num_steps", None) or 10)
        self.min_period = getattr(config, "min_period", 0.004)
        self.max_period = getattr(config, "max_period", 4.0)

        _logger(f"Loading vision from {vision_path}")
        self.vision_model = tcim_lite.runtime.load(vision_path)
        _logger(f"Loading prefill from {prefill_path}")
        self.prefill_model = tcim_lite.runtime.load(prefill_path)
        _logger(f"Loading action from {action_path}")
        self.action_model = tcim_lite.runtime.load(action_path)

        # Number of VLM layers = half of the cache outputs (key + value per layer).
        past_outputs = [self.prefill_model.get_output_name(idx) for idx in range(self.prefill_model.get_num_outputs())]
        cache_names = [n for n in past_outputs if n.startswith("past_key_") or n.startswith("past_value_")]
        self.num_layers = len(cache_names) // 2

        # Hand the prefill KV-cache outputs to the action module by device-pointer
        # sharing: action reads the same device memory prefill wrote, no host copy.
        # The export names align (past_key_<i> / past_value_<i>) across prefill/action.
        action_input_names = {
            self.action_model.get_input_name(idx) for idx in range(self.action_model.get_num_inputs())
        }
        # Device-pointer sharing of the KV cache: action reads the same device memory
        # prefill writes, no host copy. Correct tcim_lite API is
        # ``set_dev_input(name, prefill.get_dev_output(name))`` — get_dev_output returns
        # the pre-allocated device-memory handle (filled when prefill.run() executes);
        # set_dev_input binds it as the action module's input. NB: get_dev_input targets
        # input tensors, not outputs, so the earlier set_input/get_dev_input form failed
        # with Status.UNINITIALIZED.
        for name in cache_names:
            if name in action_input_names:
                self.action_model.set_dev_input(name, self.prefill_model.get_dev_output(name))

        # CPU-side token embedding (SmolVLM2 text_model embedding), dumped at export.
        weight = torch.load(embedding_path, map_location="cpu", weights_only=True)
        w = weight["weight"] if "weight" in weight else next(iter(weight.values()))
        self.embed_tokens = torch.nn.Embedding(
            num_embeddings=w.shape[0],
            embedding_dim=w.shape[1],
        )
        self.embed_tokens.load_state_dict({"weight": w} if "weight" not in weight else weight)

        self.prefix_seq_len = self._probe_prefix_seq_len()
        # VLM hidden size, read from the prefill prefix_embs input shape.
        prefix_embs_info = self.prefill_model.get_input_info("prefix_embs")
        self.prefix_hidden_size = int(prefix_embs_info.shape[-1])
        _logger(
            f"SmolVLAHMMModel ready (prefix_seq_len={self.prefix_seq_len}, "
            f"num_layers={self.num_layers}, hidden={self.prefix_hidden_size}, steps={self.num_steps})"
        )

    def _probe_prefix_seq_len(self) -> int:
        """Read the prefix sequence length from the prefill ``prefix_embs`` input."""
        for idx in range(self.prefill_model.get_num_inputs()):
            name = self.prefill_model.get_input_name(idx)
            if name == "prefix_embs":
                shape = self.prefill_model.get_input_info(name).shape
                return int(shape[1])
        raise ValueError("Cannot derive prefix_seq_len: prefix_embs input not found in prefill module")

    def _run_module(self, module: Any, inputs: dict[str, np.ndarray], output_name: str) -> np.ndarray:
        for name, data in inputs.items():
            module.set_input(name, data)
        module.run()
        module.sync()
        out = module.get_output(output_name)
        to_numpy = getattr(out, "numpy", None)
        if callable(to_numpy):
            out = to_numpy()
        return np.asarray(out)

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != _TARGET_DTYPE:
            image = image.astype(np.float16)
        image_np = np.ascontiguousarray(image)
        return self._run_module(self.vision_model, {"pixel_values": image_np}, "image_embeddings")

    def embed_prefix(
        self,
        images: list[np.ndarray],
        img_masks: list[np.ndarray],
        tokens: Tensor,
        masks: Tensor,
        state: Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the prefix embeddings (vision + language, optionally state).

        Returns ``(prefix_embs, prefix_pad_masks)`` as numpy arrays. The runtime
        contract mirrors ``modeling_smolvla.embed_prefix`` minus the image-special
        tokens (this SmolVLA export sets ``add_image_special_tokens=False``).
        """
        embs: list[np.ndarray] = []
        pad_masks: list[np.ndarray] = []
        for img, img_mask in zip(images, img_masks, strict=False):
            img_emb = self.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]
            # normalize image embeddings (sqrt of hidden dim), like LeRobot SmolVLA
            img_emb = img_emb * math.sqrt(img_emb.shape[-1])
            embs.append(img_emb)
            pad_masks.append(np.broadcast_to(img_mask[:, None], (bsize, num_img_embs)).copy())
        lang_emb = self.embed_tokens(tokens.long())
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb.detach().cpu().numpy())
        pad_masks.append(masks.detach().cpu().numpy())
        prefix_embs = np.concatenate(embs, axis=1)
        prefix_pad_masks = np.concatenate(pad_masks, axis=1)
        # pad / truncate to the static prefix_seq_len the prefill module was built for
        if prefix_embs.shape[1] < self.prefix_seq_len:
            pad_len = self.prefix_seq_len - prefix_embs.shape[1]
            prefix_embs = np.concatenate(
                [prefix_embs, np.zeros((prefix_embs.shape[0], pad_len, prefix_embs.shape[2]), dtype=np.float16)],
                axis=1,
            )
            prefix_pad_masks = np.concatenate(
                [prefix_pad_masks, np.zeros((prefix_pad_masks.shape[0], pad_len), dtype=bool)], axis=1
            )
        return prefix_embs[:, : self.prefix_seq_len], prefix_pad_masks[:, : self.prefix_seq_len]

    def prefill(self, prefix_embs: np.ndarray, prefix_pad_masks: np.ndarray) -> None:
        attention_mask = np.ones((prefix_embs.shape[0], prefix_embs.shape[1], prefix_embs.shape[1]), dtype=np.int64)
        position_ids = np.cumsum(prefix_pad_masks.astype(np.int64), axis=1) - 1
        position_ids = np.where(prefix_pad_masks, position_ids, 0).astype(np.int64)
        self._run_module(
            self.prefill_model,
            {
                "prefix_embs": prefix_embs.astype(np.float16),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            "past_key_0",
        )

    def action(self, x_t: Tensor, timestep: Tensor, prefix_pad_masks: np.ndarray) -> np.ndarray:
        return self._run_module(
            self.action_model,
            {
                "x_t": x_t.detach().cpu().numpy().astype(np.float16),
                "timestep": timestep.detach().cpu().numpy().astype(np.float32),
                "prefix_pad_masks": np.ascontiguousarray(prefix_pad_masks),
            },
            "v_t",
        )

    def forward(
        self,
        images: list[np.ndarray],
        tokens: Tensor,
        masks: Tensor,
        prefix_att_2d_masks_4d: np.ndarray | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Run the full SmolVLA flow-matching pipeline.

        Args:
            images: list of per-camera ``[B, C, H, W]`` float32 arrays.
            tokens: language token ids ``[B, seq_len]`` int64.
            masks: language attention masks ``[B, seq_len]`` bool.
            prefix_att_2d_masks_4d: unused (SmolVLA builds attention internally).
            noise: optional initial noise ``[B, chunk_size, max_action_dim]``.

        Returns:
            Action tensor ``[chunk_size, action_dim]`` on CPU (float32).
        """
        del prefix_att_2d_masks_4d
        tokens_t = tokens.to("cpu")
        masks_t = masks.to("cpu")
        img_masks = [np.ones(img.shape[0], dtype=bool) for img in images]

        prefix_embs, prefix_pad_masks = self.embed_prefix(images, img_masks, tokens_t, masks_t)
        self.prefill(prefix_embs, prefix_pad_masks)

        x_t = noise.to("cpu") if noise is not None else torch.randn(1, self.chunk_size, self.max_action_dim)
        dt = -1.0 / self.num_steps
        for step in range(self.num_steps):
            # SmolVLA flow-matching: time goes from 1 -> 0 across denoise steps.
            time = torch.tensor([1.0 - step / self.num_steps], dtype=torch.float32)
            v_t = self.action(x_t, time, prefix_pad_masks)
            v_t_tensor = torch.from_numpy(np.asarray(v_t)).float()
            # v_t may carry padding to max_action_dim; slice to real action dim
            x_t = x_t + dt * v_t_tensor.squeeze(0)

        actions = x_t.squeeze(0)
        return actions.to("cpu")

    def close(self) -> None:
        for attr in ("vision_model", "prefill_model", "action_model"):
            mod = getattr(self, attr, None)
            if mod is not None and hasattr(mod, "release"):
                mod.release()
