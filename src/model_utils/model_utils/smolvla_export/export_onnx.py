"""Export SmolVLA policy to ONNX (split: VLM prefix + Expert single-denoise-step).

Workflow mirrors pi05_export:
  Stage 1 — VLM ONNX: images + lang_tokens + lang_masks + state
              → (past_kv_tensor, prefix_pad_masks)
  Stage 2 — Expert ONNX: past_kv_tensor + prefix_pad_masks + time + noise
              → actions  (single Euler step, loop runs on host)

Usage:
  source .shrc_local
  python -m model_utils.smolvla_export.export_onnx \
      --policy-path models/smolvla/pretrained_model \
      --output-dir models/smolvla/onnx
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


def flatten_kv(past_key_values):
    if isinstance(past_key_values, dict):
        past_key_values = [past_key_values[i] for i in sorted(past_key_values.keys())]
    if not isinstance(past_key_values, list | tuple):
        raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")
    keys = torch.stack([kv["key_states"] for kv in past_key_values], dim=0)
    values = torch.stack([kv["value_states"] for kv in past_key_values], dim=0)
    return torch.stack([keys, values], dim=1)


def unflatten_kv(flat_tensor):
    num_layers = flat_tensor.shape[0]
    keys = flat_tensor[:, 0]
    values = flat_tensor[:, 1]
    return [{"key_states": keys[i], "value_states": values[i]} for i in range(num_layers)]


def create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device="cpu"):
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time must be 1-D")
    dtype = torch.float32
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    cos_value = torch.sin(sin_input + math.pi / 2)
    return torch.cat([torch.sin(sin_input), cos_value], dim=1)


def make_att_2d_masks(pad_masks, att_masks):
    att_masks = att_masks.to(dtype=torch.int)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_masks = att_masks.to(dtype=torch.bool)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def resize_with_pad_onnx(img, width, height, pad_value=-1.0):
    """ONNX-traceable resize_with_pad for [B, C, H, W] float32 images."""
    batch_size, channels, cur_height, cur_width = img.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    img = F.interpolate(img, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
    img = img.clamp(-1.0, 1.0)
    pad_h0 = (height - resized_height) // 2
    pad_h1 = height - resized_height - pad_h0
    pad_w0 = (width - resized_width) // 2
    pad_w1 = width - resized_width - pad_w0
    return F.pad(img, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=pad_value)


class SmolVLAVLMWrapper(nn.Module):
    """Wraps the VLM prefix encoding: images + lang + state → KV cache + pad masks."""

    def __init__(self, policy):
        super().__init__()
        self.vla = policy.model
        self.config = policy.config

    def forward(self, img_top, img_wrist, lang_tokens, lang_masks, state):
        images = []
        img_masks_list = []
        bsz = img_top.shape[0]
        device = img_top.device

        for img in [img_top, img_wrist]:
            if img.dtype != torch.float32:
                img = img.to(torch.float32)
            img = img * 2.0 - 1.0
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad_onnx(
                    img,
                    self.config.resize_imgs_with_padding[0],
                    self.config.resize_imgs_with_padding[1],
                    pad_value=0.0,
                )
            images.append(img)
            img_masks_list.append(torch.ones(bsz, dtype=torch.bool, device=device))

        lang_tokens = lang_tokens.to(torch.long)
        lang_masks = lang_masks.to(torch.bool)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.vla.embed_prefix(
            images, img_masks_list, lang_tokens, lang_masks, state=state
        )

        prefix_pad_masks_int = prefix_pad_masks.to(dtype=torch.int)
        prefix_position_ids = torch.cumsum(prefix_pad_masks_int, dim=1) - 1
        prefix_pad_masks_int = prefix_pad_masks_int.to(dtype=torch.bool)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks_int, prefix_att_masks)

        _, past_key_values = self.vla.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        past_kv_tensor = flatten_kv(past_key_values)
        return past_kv_tensor, prefix_pad_masks


class SmolVLAExpertWrapper(nn.Module):
    """Wraps a single denoising step: KV cache + time + noise → actions."""

    def __init__(self, policy):
        super().__init__()
        self.vla = policy.model
        self.config = policy.config

    def forward(self, past_kv_tensor, prefix_pad_masks, time, noise):
        past_key_values = unflatten_kv(past_kv_tensor)
        prefix_pad_masks = prefix_pad_masks.to(torch.bool)

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.vla.embed_suffix(noise, time)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks.to(torch.int), dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks.to(torch.int), dim=1) - 1

        outputs_embeds, _ = self.vla.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.vla.action_out_proj(suffix_out)
        return v_t


def load_policy(policy_path: str, device: str = "cpu"):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from safetensors.torch import load_file

    config = PreTrainedConfig.from_pretrained(policy_path, local_files_only=True)
    policy = SmolVLAPolicy(config)

    safetensors_path = Path(policy_path) / "model.safetensors"
    if safetensors_path.exists():
        state_dict = load_file(str(safetensors_path))
        remapped = {}
        for k, v in state_dict.items():
            if not k.startswith("model."):
                remapped[f"model.{k}"] = v
            else:
                remapped[k] = v
        missing, unexpected = policy.load_state_dict(remapped, strict=False)
        LOGGER.info(f"Loaded weights from {safetensors_path} (missing={len(missing)}, unexpected={len(unexpected)})")

    policy = policy.to(device)
    policy.eval()
    return policy


def build_vlm_dummy_inputs(config, device, seed=42):
    torch.manual_seed(seed)
    bsz = 1
    img_h, img_w = 480, 640
    for key, feat in config.input_features.items():
        if "images" in key:
            img_h, img_w = feat.shape[1], feat.shape[2]
            break
    lang_len = config.tokenizer_max_length
    state_dim = config.max_state_dim

    img_top = torch.randn(bsz, 3, img_h, img_w, device=device, dtype=torch.float32).clamp(0, 1)
    img_wrist = torch.randn(bsz, 3, img_h, img_w, device=device, dtype=torch.float32).clamp(0, 1)
    lang_tokens = torch.randint(0, 1000, (bsz, lang_len), device=device, dtype=torch.long)
    lang_masks = torch.ones(bsz, lang_len, device=device, dtype=torch.bool)
    state = torch.randn(bsz, 6, device=device, dtype=torch.float32)
    state = F.pad(state, (0, state_dim - 6))

    return img_top, img_wrist, lang_tokens, lang_masks, state


def build_expert_dummy_inputs(vlm_wrapper, vlm_inputs, config, device, seed=42):
    torch.manual_seed(seed)
    bsz = 1
    with torch.no_grad():
        past_kv, prefix_masks = vlm_wrapper(*vlm_inputs)

    time = torch.tensor([1.0], device=device, dtype=torch.float32)
    noise = torch.randn(bsz, config.chunk_size, config.max_action_dim, device=device, dtype=torch.float32)
    return past_kv, prefix_masks, time, noise


def pad_vector(vector, new_dim):
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def export_vlm_onnx(wrapper, dummy_inputs, output_path, opset=17, dynamo=False):
    LOGGER.info(f"Exporting VLM ONNX to {output_path} (dynamo={dynamo})")
    wrapper.eval()

    input_names = ["img_top", "img_wrist", "lang_tokens", "lang_masks", "state"]
    output_names = ["past_kv_tensor", "prefix_pad_masks"]

    if dynamo:
        export_options = torch.onnx.ExportOptions()
        export_options.dynamic_shapes = False
        onnx_program = torch.onnx.dynamo_export(
            wrapper,
            *dummy_inputs,
            export_options=export_options,
        )
        onnx_program.save(str(output_path))
    else:
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(output_path),
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,
            verbose=False,
            dynamo=False,
        )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    LOGGER.info(f"VLM ONNX exported: {output_path} ({size_mb:.1f} MB)")


def export_expert_onnx(wrapper, dummy_inputs, output_path, opset=17, dynamo=False):
    LOGGER.info(f"Exporting Expert ONNX to {output_path} (dynamo={dynamo})")
    wrapper.eval()

    input_names = ["past_kv_tensor", "prefix_pad_masks", "time", "noise"]
    output_names = ["velocity"]

    if dynamo:
        export_options = torch.onnx.ExportOptions()
        export_options.dynamic_shapes = False
        onnx_program = torch.onnx.dynamo_export(
            wrapper,
            *dummy_inputs,
            export_options=export_options,
        )
        onnx_program.save(str(output_path))
    else:
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(output_path),
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,
            verbose=False,
            dynamo=False,
        )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    LOGGER.info(f"Expert ONNX exported: {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export SmolVLA to ONNX (VLM + Expert split)")
    parser.add_argument("--policy-path", type=str, required=True, help="Path to SmolVLA pretrained_model dir")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: <policy_path>/onnx)")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for export")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--dynamo", action="store_true", help="Use torch.onnx.dynamo_export (opset 18)")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM export if already exists")
    parser.add_argument("--skip-expert", action="store_true", help="Skip Expert export if already exists")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    policy_path = Path(args.policy_path).expanduser().resolve()
    output_dir = Path(args.output_dir or str(policy_path / "onnx")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vlm_onnx = output_dir / "smolvla_vlm.onnx"
    expert_onnx = output_dir / "smolvla_expert.onnx"
    runtime_dir = output_dir / "runtime_save"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Loading SmolVLA policy from {policy_path}")
    policy = load_policy(str(policy_path), device=args.device)
    config = policy.config

    LOGGER.info(
        f"Policy config: chunk_size={config.chunk_size}, max_action_dim={config.max_action_dim}, "
        f"num_steps={config.num_steps}, tokenizer_max_length={config.tokenizer_max_length}"
    )

    if args.dtype == "fp16":
        LOGGER.info("Converting model to float16 for export")
        try:
            policy.model = policy.model.half()
        except Exception as e:
            LOGGER.warning(f"Failed to convert to fp16, keeping fp32: {e}")
    else:
        # bfloat16 不能直接导出 ONNX，先转 float32
        LOGGER.info("Converting model to float32 for export")
        policy = policy.float()

    device = torch.device(args.device)

    if not (args.skip_vlm and vlm_onnx.exists()):
        LOGGER.info("=== Stage 1: VLM ONNX Export ===")
        vlm_wrapper = SmolVLAVLMWrapper(policy).to(device)
        vlm_wrapper.eval()

        dummy_inputs = build_vlm_dummy_inputs(config, device, args.seed)
        LOGGER.info(f"VLM inputs: {[t.shape for t in dummy_inputs]}")

        with torch.no_grad():
            past_kv, prefix_masks = vlm_wrapper(*dummy_inputs)
        LOGGER.info(f"VLM output shapes: past_kv={past_kv.shape}, prefix_masks={prefix_masks.shape}")

        torch.save(past_kv, runtime_dir / "past_kv_tensor.pth")
        torch.save(prefix_masks, runtime_dir / "prefix_pad_masks.pth")
        LOGGER.info(f"Saved runtime tensors to {runtime_dir}")

        export_vlm_onnx(vlm_wrapper, dummy_inputs, vlm_onnx, opset=args.opset, dynamo=args.dynamo)

    if not (args.skip_expert and expert_onnx.exists()):
        LOGGER.info("=== Stage 2: Expert ONNX Export ===")
        expert_wrapper = SmolVLAExpertWrapper(policy).to(device)
        expert_wrapper.eval()

        past_kv = torch.load(runtime_dir / "past_kv_tensor.pth", map_location=device)
        prefix_masks = torch.load(runtime_dir / "prefix_pad_masks.pth", map_location=device)

        torch.manual_seed(args.seed)
        bsz = 1
        time_input = torch.tensor([1.0], device=device, dtype=torch.float32)
        noise_input = torch.randn(
            bsz,
            config.chunk_size,
            config.max_action_dim,
            device=device,
            dtype=torch.float32,
        )

        expert_inputs = (past_kv, prefix_masks, time_input, noise_input)
        LOGGER.info(f"Expert inputs: {[t.shape for t in expert_inputs]}")

        with torch.no_grad():
            v_t = expert_wrapper(*expert_inputs)
        LOGGER.info(f"Expert output shape: {v_t.shape}")

        export_expert_onnx(expert_wrapper, expert_inputs, expert_onnx, opset=args.opset, dynamo=args.dynamo)

    LOGGER.info("=== Export complete ===")
    LOGGER.info(f"VLM ONNX:    {vlm_onnx}")
    LOGGER.info(f"Expert ONNX: {expert_onnx}")


if __name__ == "__main__":
    main()
