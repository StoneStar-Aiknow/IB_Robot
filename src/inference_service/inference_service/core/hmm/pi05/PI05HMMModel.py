"""PI05 HMM multi-module pipeline orchestrator for the Houmo (LQ50/M50 xh2) backend.

Mirrors the verified ``houmo-examples/pi05/demo.py`` pipeline over
``tcim_lite.runtime``: 6 compiled ``.hmm`` modules (vision / prefill / decode /
time_mlp / action_in_proj / action_out_proj) plus the CPU-side embedding layer,
with the denoising loop orchestrated on the host. KV-cache produced by the
prefill module is handed to the decode module via device-pointer sharing
(``decode.set_input(name, prefill.get_dev_input(name))``).

This is the Houmo counterpart of ``ascend_om/pi05/PI05OMModel.py``: the OM
backend fuses the loop into 2 NPU models, while the Houmo toolchain splits
PI05 into 6 modules and runs the denoise loop on the host CPU.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

# tcim_lite emits fp16 weights; keep all host-side math in fp16 to match.
_TARGET_DTYPE = torch.float16


def _logger(msg: str) -> None:
    print(f"[PI05HMMModel]: {msg}")


def _get_safe_dtype(target_dtype: Any, device_type: str) -> torch.dtype:
    # mps does not support float64; fall back to float32 there.
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    return target_dtype


def _create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions.

    Exact copy of openpi ``create_sinusoidal_pos_embedding``.
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
    dtype = _get_safe_dtype(torch.float64, getattr(device, "type", "cpu"))
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 2.0 * math.pi / period
    sin_input = time[:, None] * scaling_factor[None, :]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)


def _make_att_2d_masks(pad_masks: Tensor, att_masks: list[int]) -> Tensor:
    cumsum = pad_masks.to(dtype=torch.int).cumsum(dim=-1) - 1
    att_2d_masks = cumsum[:, None, :]
    pad_2d_masks = pad_masks[:, None, :]
    return torch.where(pad_2d_masks.bool() & att_2d_masks.bool(), 0.0, float("-inf"))


def build_prefix_att_2d_masks_4d_np(num_cameras: int, lang_masks: np.ndarray, prefix_seq_len: int) -> np.ndarray:
    """Build the ``(B, 1, S, S)`` fp32 additive prefix attention mask.

    Equivalent to ``prefix_mask_utils.build_prefix_att_2d_masks_4d_np``. For PI05 every prefix
    token attends to itself and all earlier valid prefix tokens.
    """
    bsize = lang_masks.shape[0]
    # PI05 uses 256 image tokens per camera (SigLIP @ 224x224 -> 16x16 = 256 patches).
    image_tokens_total = 256 * num_cameras
    img_pad = np.ones((bsize, image_tokens_total), dtype=bool)
    pad_masks = np.concatenate([img_pad, lang_masks.astype(bool)], axis=1)
    if pad_masks.shape[1] < prefix_seq_len:
        pad = np.zeros((bsize, prefix_seq_len - pad_masks.shape[1]), dtype=bool)
        pad_masks = np.concatenate([pad_masks, pad], axis=1)
    pad_masks = pad_masks[:, :prefix_seq_len]
    cumsum = np.cumsum(pad_masks.astype(np.int32), axis=1) - 1
    att_2d = np.where(
        pad_masks[:, None, :] & (cumsum[:, None, :] >= 0),
        0.0,
        float("-inf"),
    ).astype(np.float32)
    # The compiled prefill expects a static [B, 1, S, 2048] mask: the trailing
    # KV-cache dimension is padded to 2048 (the prefill module's static seq axis)
    # with -inf so prefix tokens never attend beyond the valid prefix length.
    att_2d_4d = att_2d[:, None, :, :]
    kv_dim = 2048
    if att_2d_4d.shape[-1] < kv_dim:
        pad_cols = np.full(
            (bsize, 1, att_2d_4d.shape[-2], kv_dim - att_2d_4d.shape[-1]),
            float("-inf"),
            dtype=np.float32,
        )
        att_2d_4d = np.concatenate([att_2d_4d, pad_cols], axis=-1)
    return att_2d_4d


class PI05HMMModel:
    """Orchestrator for the 6-module Houmo PI05 HMM pipeline.

    Args:
        vision_path / prefill_path / decode_path / time_mlp_path /
        action_in_proj_path / action_out_proj_path: paths to the compiled ``.hmm``
            modules (resolved from ``config.hmm.json`` by the runtime session).
        embedding_path: path to ``embedding.pt`` (Gemma token embedding weights).
        config: policy config view exposing chunk_size / max_action_dim /
            num_inference_steps.
    """

    def __init__(
        self,
        vision_path: str,
        prefill_path: str,
        decode_path: str,
        time_mlp_path: str,
        action_in_proj_path: str,
        action_out_proj_path: str,
        embedding_path: str,
        config: Any,
    ) -> None:
        import tcim_lite

        self._tcim = tcim_lite
        self.config = config
        self.chunk_size = config.chunk_size
        self.max_action_dim = config.max_action_dim
        self.num_inference_steps = config.num_inference_steps
        self.image_resolution = (224, 224)
        self.min_period = 0.0005
        self.max_period = 1.0
        self.valid_kvcache_length = 0

        _logger(f"Loading vision from {vision_path}")
        self.vision_model = tcim_lite.runtime.load(vision_path)
        _logger(f"Loading prefill from {prefill_path}")
        self.prefill_model = tcim_lite.runtime.load(prefill_path)
        _logger(f"Loading decode from {decode_path}")
        self.decode_model = tcim_lite.runtime.load(decode_path)

        # Share the prefill/decode KV-cache device memory by pointer: prefill and
        # decode both expose the cache as same-named inputs
        # (``model_layers_X_self_attn_kcache_input``) backed by pre-allocated device
        # memory. Bind decode's cache inputs to prefill's via ``set_dev_input`` +
        # ``get_dev_input`` (input->input sharing of the pre-allocated buffer, filled
        # when prefill.run() executes). NB: ``set_input`` would stage a host copy and
        # break the zero-copy handoff.
        decode_input_names = {
            self.decode_model.get_input_name(idx) for idx in range(self.decode_model.get_num_inputs())
        }
        for idx in range(self.prefill_model.get_num_inputs()):
            name = self.prefill_model.get_input_name(idx)
            if "model_layer" in name and "cache" in name and name in decode_input_names:
                self.decode_model.set_dev_input(name, self.prefill_model.get_dev_input(name))

        self.time_mlp_model = tcim_lite.runtime.load(time_mlp_path)
        self.action_in_proj_model = tcim_lite.runtime.load(action_in_proj_path)
        self.action_out_proj_model = tcim_lite.runtime.load(action_out_proj_path)

        # Probe the compiled modules' I/O shapes to size the host-side Linear layers.
        action_in_proj_in_shape = self.action_in_proj_model.get_input_info("action_in").shape
        action_in_proj_out_shape = self.action_in_proj_model.get_output_info("action_in_proj_out").shape
        self.action_in_proj_in_features = int(action_in_proj_in_shape[-1])
        self.action_in_proj_out_features = int(action_in_proj_out_shape[-1])

        action_out_proj_out_shape = self.action_out_proj_model.get_output_info("action_out_proj_out").shape
        self.action_out_proj_out_features = int(action_out_proj_out_shape[-1])

        # Host-side projection heads (action_in_proj / action_out_proj equivalents).
        # NOTE: these mirror houmo-examples and are initialized with random weights;
        # real deployment must load the trained projection weights.
        self.action_proj = torch.nn.Linear(self.max_action_dim, self.action_in_proj_in_features)
        self.action_out_fc = torch.nn.Linear(self.action_out_proj_out_features, self.max_action_dim).half()

        # CPU-side token embedding (Gemma), weights dumped by the export pipeline.
        weight = torch.load(embedding_path, map_location="cpu", weights_only=True)
        self.embed_tokens = torch.nn.Embedding(
            num_embeddings=weight["weight"].shape[0],
            embedding_dim=weight["weight"].shape[1],
        )
        self.embed_tokens.load_state_dict(weight)

        self.prefix_seq_len = self._probe_prefix_seq_len()
        _logger(
            f"PI05HMMModel ready (prefix_seq_len={self.prefix_seq_len}, "
            f"action_in_proj {self.max_action_dim}->{self.action_in_proj_in_features}, "
            f"action_out_proj {self.action_out_proj_out_features}->{self.max_action_dim})"
        )

    def _probe_prefix_seq_len(self) -> int:
        """Read the prefix sequence length from the prefill attention_mask slot."""
        for idx in range(self.prefill_model.get_num_inputs()):
            name = self.prefill_model.get_input_name(idx)
            if name == "attention_mask":
                shape = self.prefill_model.get_input_info(name).shape
                return int(shape[-1])
        raise ValueError("Cannot derive num_image_tokens: prefix_seq_len not found in prefill inputs")

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
        out = self._run_module(self.vision_model, {"pixel_values": image_np}, "image_features")
        return out

    def prefill(self, attention_mask: Tensor, inputs_embeds: Tensor) -> None:
        if attention_mask.dtype != _TARGET_DTYPE:
            attention_mask = attention_mask.half()
        if inputs_embeds.dtype != _TARGET_DTYPE:
            inputs_embeds = inputs_embeds.half()

        seq_len = inputs_embeds.shape[1]
        pad_seq_len = 968 - seq_len
        if pad_seq_len > 0:
            pad_emb = torch.zeros(1, pad_seq_len, inputs_embeds.shape[2], dtype=inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, pad_emb], dim=1)

        # attention_mask is 4D [B, 1, S, S] from build_prefix_att_2d_masks_4d_np;
        # pad along the sequence axis (dim=2) keeping the 4D layout so torch.cat
        # matches dimensions. The trailing dim is also padded to 2048 below.
        mask_seq_len = attention_mask.size(2)
        mask_pad_seq_len = 968 - mask_seq_len
        if mask_pad_seq_len > 0:
            mask_last_dim = attention_mask.size(-1)
            pad_mask = torch.full((1, 1, mask_pad_seq_len, mask_last_dim), float("-inf"), dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=2)

        valid_length = int(seq_len)
        current_length = 0
        self._run_module(
            self.prefill_model,
            {
                "input_1": inputs_embeds.detach().cpu().numpy(),
                "valid_length": np.array([0], dtype=np.int32),
                "current_length": np.array([current_length], dtype=np.int32),
                "attention_mask": attention_mask.detach().cpu().numpy(),
            },
            "last_hidden_state",
        )
        self.valid_kvcache_length = valid_length

    def decode(self, attention_mask: Tensor, inputs_embeds: Tensor, cond: Tensor) -> np.ndarray:
        if attention_mask.dtype != _TARGET_DTYPE:
            attention_mask = attention_mask.half()
        if inputs_embeds.dtype != _TARGET_DTYPE:
            inputs_embeds = inputs_embeds.half()

        seq_len = inputs_embeds.shape[1]
        action_seq_len = 50
        pad_seq_len = action_seq_len - seq_len
        if pad_seq_len > 0:
            pad_emb = torch.zeros(1, pad_seq_len, inputs_embeds.shape[2], dtype=inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, pad_emb], dim=1)

        # attention_mask is 4D [B, 1, S, S]; pad along dim=2 keeping 4D layout.
        mask_seq_len = attention_mask.size(2)
        mask_pad_seq_len = action_seq_len - mask_seq_len
        if mask_pad_seq_len > 0:
            mask_last_dim = attention_mask.size(-1)
            pad_mask = torch.full((1, 1, mask_pad_seq_len, mask_last_dim), float("-inf"), dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=2)

        valid_length = self.valid_kvcache_length
        current_length = 0
        out = self._run_module(
            self.decode_model,
            {
                "input_1": inputs_embeds.detach().cpu().numpy(),
                "valid_length": np.array([valid_length], dtype=np.int32),
                "current_length": np.array([current_length], dtype=np.int32),
                "cond": cond.detach().cpu().numpy(),
                "attention_mask": attention_mask.detach().cpu().numpy(),
            },
            "last_hidden_state",
        )
        return out

    def action_in_proj(self, noisy_actions: Tensor) -> np.ndarray:
        out = self._run_module(
            self.action_in_proj_model,
            {"action_in": noisy_actions.detach().cpu().numpy()},
            "action_in_proj_out",
        )
        return out

    def action_out_proj(self, suffix_out: Tensor) -> np.ndarray:
        out = self._run_module(
            self.action_out_proj_model,
            {"action_out": suffix_out.detach().cpu().numpy()},
            "action_out_proj_out",
        )
        return out

    def time_mlp(self, time_emb: Tensor) -> np.ndarray:
        out = self._run_module(
            self.time_mlp_model,
            {"time_emb": time_emb.detach().cpu().numpy()},
            "time_mlp_out",
        )
        return out

    def embed_prefix(
        self,
        images: list[np.ndarray],
        img_masks: list[np.ndarray],
        tokens: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, Tensor, list[int]]:
        embs: list[np.ndarray] = []
        pad_masks: list[np.ndarray] = []
        att_masks: list[int] = []
        for img, img_mask in zip(images, img_masks, strict=False):
            img_emb = self.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]
            pad_masks.append(np.broadcast_to(img_mask[:, None], (bsize, num_img_embs)))
            embs.append(img_emb)
            att_masks += [0] * num_img_embs
        lang_emb = self.embed_tokens(tokens.long())
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb.detach().cpu().numpy())
        pad_masks.append(masks.detach().cpu().numpy())
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs
        prefix_embs = torch.from_numpy(np.concatenate(embs, axis=1))
        prefix_pad_masks = torch.from_numpy(np.concatenate(pad_masks, axis=1))
        return prefix_embs, prefix_pad_masks, att_masks

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        bsize = noisy_actions.shape[0]
        action_pad_len = self.chunk_size - noisy_actions.shape[1]
        pad_actions = torch.zeros(bsize, action_pad_len, noisy_actions.shape[2])
        padded_actions = torch.cat([noisy_actions, pad_actions], dim=1)
        action_emb = self.action_proj(padded_actions.half())
        time_emb = _create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj_out_features, self.min_period, self.max_period, noisy_actions.device
        )
        time_emb_actual_dim = time_emb.shape[-1]
        pad_time = torch.zeros(bsize, self.action_in_proj_out_features - time_emb_actual_dim)
        adarms_cond = torch.cat([time_emb, pad_time], dim=-1)
        suffix_pad_mask = torch.cat(
            [torch.ones(bsize, padded_actions.shape[1]), torch.zeros(bsize, self.action_in_proj_out_features)],
            dim=1,
        )
        return action_emb, suffix_pad_mask, adarms_cond

    def denoise_step(
        self,
        prefix_pad_masks: Tensor,
        x_t: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        action_emb, _, adarms_cond = self.embed_suffix(x_t, timestep)
        suffix_embs = torch.from_numpy(action_emb) if isinstance(action_emb, np.ndarray) else action_emb
        bsize = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = suffix_embs.shape[1]
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_att_2d = suffix_embs.new_zeros((bsize, suffix_len, suffix_len))
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
        full_att_2d_4d = full_att_2d[:, None, :, :]
        pad_len = 2048 - full_att_2d_4d.shape[-1]
        if pad_len > 0:
            full_att_2d_4d = torch.cat(
                [full_att_2d_4d, torch.full((bsize, 1, suffix_len, pad_len), float("-inf"))], dim=-1
            )
        suffix_out = torch.from_numpy(self.decode(full_att_2d_4d, suffix_embs, adarms_cond))
        v_t = self.action_out_fc(suffix_out.half())
        return v_t

    def forward(
        self,
        images: list[np.ndarray],
        tokens: Tensor,
        masks: Tensor,
        prefix_att_2d_masks_4d: np.ndarray,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Run the full PI05 pipeline and return a float32 action tensor.

        Args:
            images: list of per-camera ``[B, C, H, W]`` float32 arrays.
            tokens: language token ids ``[B, seq_len]`` int64.
            masks: language attention masks ``[B, seq_len]`` bool.
            prefix_att_2d_masks_4d: ``(B, 1, S, S)`` additive prefix mask
                (built by the caller via ``build_prefix_att_2d_masks_4d_np``).
            noise: optional initial noise ``[B, chunk_size, max_action_dim]``.

        Returns:
            Action tensor ``[chunk_size, action_dim]`` on CPU (float32).
        """
        # embed_prefix / embed_image operate on numpy arrays (set_input feeds
        # tcim_lite which expects ndarrays; astype/broadcast_to are numpy ops).
        image_arrays = [np.asarray(img) for img in images]
        tokens_t = torch.as_tensor(tokens).to("cpu")
        masks_t = torch.as_tensor(masks).to("cpu")
        img_masks = [np.ones(img.shape[0], dtype=bool) for img in image_arrays]
        noise_t = torch.as_tensor(noise).to("cpu") if noise is not None else None

        prefix_embs, prefix_pad_masks, _ = self.embed_prefix(image_arrays, img_masks, tokens_t, masks_t)

        prefix_att_2d_masks_4d_t = torch.from_numpy(np.asarray(prefix_att_2d_masks_4d))
        dt = (1.0 / self.num_inference_steps) * torch.ones(1, dtype=torch.float32)
        x_t = noise_t if noise_t is not None else torch.zeros(1, self.chunk_size, self.max_action_dim)

        self.prefill(prefix_att_2d_masks_4d_t, prefix_embs)

        for step in range(self.num_inference_steps):
            time = (1.0 - step / self.num_inference_steps) * torch.ones(1)
            time_tensor = time
            v_t = self.denoise_step(prefix_pad_masks, x_t, time_tensor)
            pad_tensor = [v_t] if not isinstance(v_t, list) else v_t
            x_t = x_t + dt[0] * pad_tensor[0].squeeze(0).float()

        actions = x_t.squeeze(0)
        return actions.to("cpu")

    def close(self) -> None:
        for attr in (
            "vision_model",
            "prefill_model",
            "decode_model",
            "time_mlp_model",
            "action_in_proj_model",
            "action_out_proj_model",
        ):
            mod = getattr(self, attr, None)
            if mod is not None and hasattr(mod, "release"):
                mod.release()
