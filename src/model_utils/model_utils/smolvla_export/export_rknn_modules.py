#!/usr/bin/env python3
"""Export SmolVLA as 3 segmented ONNX modules for RKNN conversion.

Mirrors the HMM module split (vision → prefill → action) but outputs standard
ONNX instead of HMONNX, so each module can be converted via rknn-toolkit2.

Modules:
  vision:  pixel_values [1,3,512,512] → image_embeddings [1,N,960]
  prefill: prefix_embs [1,177,960] + attention_mask + position_ids → 32 KV cache tensors
  action:  x_t [1,50,32] + timestep + prefix_pad_masks + 32 KV tensors → v_t [1,50,32]

Also saves token_embedding.pt (CPU-side text embedding) and meta_info.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnxsim import simplify

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LEROBOT_SRC = REPO_ROOT / "libs" / "lerobot" / "src"


def ensure_lerobot_importable(lerobot_src: str | Path | None = None) -> Path:
    lerobot_src_path = Path(lerobot_src) if lerobot_src is not None else DEFAULT_LEROBOT_SRC
    lerobot_src_path = lerobot_src_path.resolve()
    if not lerobot_src_path.exists():
        raise FileNotFoundError(f"LeRobot src path not found: {lerobot_src_path}")
    if str(lerobot_src_path) not in sys.path:
        sys.path.insert(0, str(lerobot_src_path))
    return lerobot_src_path


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_hf_cached_model_path(model_id_or_path: str) -> str:
    model_path = Path(model_id_or_path)
    if model_path.exists():
        return str(model_path.resolve())
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_cache_dir = cache_root / f"models--{model_id_or_path.replace('/', '--')}"
    snapshots_dir = repo_cache_dir / "snapshots"
    refs_main = repo_cache_dir / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = snapshots_dir / revision
        if snapshot_dir.exists():
            return str(snapshot_dir.resolve())
    if snapshots_dir.exists():
        candidates = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()])
        if candidates:
            return str(candidates[-1].resolve())
    return model_id_or_path


def configure_hf_offline_env():
    os.environ.setdefault("HUGGINGFACE_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


@torch.no_grad()
def load_smolvla_policy(model_path: str, device: torch.device, lerobot_src: str | Path | None = None):
    ensure_lerobot_importable(lerobot_src)
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    configure_hf_offline_env()
    config = PreTrainedConfig.from_pretrained(model_path)
    if hasattr(config, "vlm_model_name") and isinstance(config.vlm_model_name, str):
        config.vlm_model_name = resolve_hf_cached_model_path(config.vlm_model_name)
    if hasattr(config, "device"):
        config.device = str(device)
    policy = SmolVLAPolicy.from_pretrained(model_path, config=config, strict=False)
    policy.eval()
    policy.to(device)
    return policy


# ── Vision module ──────────────────────────────────────────────────────────


class SmolVLAVisionPart(nn.Module):
    """Vision tower + connector with static patch_attention_mask (NonZero fix)."""

    def __init__(self, vlm_with_expert: nn.Module, image_height: int = 512, image_width: int = 512):
        super().__init__()
        vlm_model = vlm_with_expert.get_vlm_model()
        self.vision_model = vlm_model.vision_model
        self.connector = vlm_model.connector
        patch_size = self.vision_model.config.patch_size
        nb_h = image_height // patch_size
        nb_w = image_width // patch_size
        self.register_buffer(
            "patch_attention_mask",
            torch.ones(1, nb_h, nb_w, dtype=torch.bool),
            persistent=False,
        )
        self.vision_model.eval()
        self.connector.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.contiguous()
        pixel_values = pixel_values * 2.0 - 1.0
        vision_dtype = next(self.vision_model.parameters()).dtype
        pixel_values = pixel_values.to(dtype=vision_dtype)
        image_hidden_states = self.vision_model(
            pixel_values=pixel_values,
            patch_attention_mask=self.patch_attention_mask,
        ).last_hidden_state
        image_hidden_states = self.connector(image_hidden_states)
        return image_hidden_states


# ── Prefill module ─────────────────────────────────────────────────────────


def flatten_cache_dict(past_key_values: dict, num_layers: int) -> tuple[torch.Tensor, ...]:
    flat: list[torch.Tensor] = []
    for i in range(num_layers):
        c = past_key_values[i]
        flat.append(c["key_states"])
        flat.append(c["value_states"])
    return tuple(flat)


def build_prefill_input_names() -> list[str]:
    return ["prefix_embs", "attention_mask", "position_ids"]


def build_cache_output_names(num_layers: int) -> list[str]:
    names: list[str] = []
    for i in range(num_layers):
        names.append(f"past_key_{i}")
        names.append(f"past_value_{i}")
    return names


class SmolVLAPrefillPart(nn.Module):
    def __init__(self, vlm_with_expert: nn.Module):
        super().__init__()
        self.vlm_with_expert = vlm_with_expert
        self.num_layers = vlm_with_expert.num_vlm_layers

    def forward(self, prefix_embs, attention_mask, position_ids) -> tuple[torch.Tensor, ...]:
        prefix_dtype = self.vlm_with_expert.get_vlm_model().text_model.layers[0].self_attn.q_proj.weight.dtype
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=attention_mask.to(device=prefix_embs.device, dtype=torch.bool),
            position_ids=position_ids.to(device=prefix_embs.device, dtype=torch.long),
            past_key_values=None,
            inputs_embeds=[prefix_embs.to(dtype=prefix_dtype), None],
            use_cache=True,
            fill_kv_cache=True,
        )
        return flatten_cache_dict(past_key_values, self.num_layers)


# ── Action module ──────────────────────────────────────────────────────────


def build_cache_input_names(num_layers: int) -> list[str]:
    names: list[str] = []
    for i in range(num_layers):
        names.append(f"past_key_{i}")
        names.append(f"past_value_{i}")
    return names


class SmolVLAActionPart(nn.Module):
    def __init__(self, flow_model: nn.Module):
        super().__init__()
        self.flow_model = flow_model
        self.num_layers = flow_model.vlm_with_expert.num_vlm_layers

    def forward(self, x_t, timestep, prefix_pad_masks, *flat_cache) -> torch.Tensor:
        past_key_values: dict[int, dict[str, torch.Tensor]] = {}
        for i in range(self.num_layers):
            past_key_values[i] = {
                "key_states": flat_cache[i * 2],
                "value_states": flat_cache[i * 2 + 1],
            }
        x_t = x_t.to(dtype=self.flow_model.action_in_proj.weight.dtype)
        timestep = timestep.to(device=x_t.device, dtype=torch.float32)
        prefix_pad_masks = prefix_pad_masks.to(device=x_t.device, dtype=torch.bool)
        return self.flow_model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=timestep,
        )


# ── ONNX export helpers ────────────────────────────────────────────────────


def export_onnx(model, inputs, onnx_file, input_names, output_names, opset=17):
    print(f"  Exporting ONNX → {onnx_file.name} ...")
    torch.onnx.export(
        model,
        inputs,
        str(onnx_file),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        verbose=False,
        do_constant_folding=True,
    )


def simplify_onnx(onnx_file, simplified_file, input_shapes):
    print(f"  Simplifying → {simplified_file.name} ...")
    onnx_model = onnx.load(str(onnx_file))
    try:
        import onnx_graphsurgeon as gs

        graph = gs.import_onnx(onnx_model)
        graph.cleanup().toposort()
        onnx_model = gs.export_onnx(graph)
        print("    graphsurgeon toposort applied")
    except Exception as exc:
        print(f"    Warning: graphsurgeon toposort failed: {exc}")
    try:
        model_simplified, check = simplify(onnx_model, test_input_shapes=input_shapes)
        if not check:
            print("    Warning: onnxsim check failed, saving anyway.")
    except Exception as exc:
        print(f"    Warning: onnxsim failed ({exc}); keeping toposorted ONNX.")
        model_simplified = onnx_model
    onnx.save(model_simplified, str(simplified_file))


# ── Main export ────────────────────────────────────────────────────────────


def export_all(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    export_dtype = torch.float16 if device.type == "cuda" else torch.float32

    policy = load_smolvla_policy(args.model_path, device, args.lerobot_src)
    flow_model = policy.model
    vlm_with_expert = flow_model.vlm_with_expert
    config = flow_model.config

    image_h, image_w = args.image_height, args.image_width
    prefix_length = args.prefix_length
    chunk_size = config.chunk_size
    max_action_dim = config.max_action_dim
    num_layers = vlm_with_expert.num_vlm_layers
    text_config = vlm_with_expert.config.text_config
    prefix_hidden_size = text_config.hidden_size
    num_kv_heads = text_config.num_key_value_heads
    head_dim = text_config.head_dim

    print(f"Device: {device}, dtype: {export_dtype}")
    print(f"Image: {image_h}x{image_w}, prefix_length: {prefix_length}")
    print(f"num_layers: {num_layers}, kv_heads: {num_kv_heads}, head_dim: {head_dim}")
    print(f"chunk_size: {chunk_size}, max_action_dim: {max_action_dim}")

    out_dir = Path(args.output_dir)
    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    # ── Vision ──
    print("\n=== Vision module ===")
    vision_model = SmolVLAVisionPart(vlm_with_expert, image_h, image_w)
    vision_model.eval()
    vision_model.to(device=device, dtype=export_dtype)
    dummy_pixel = torch.rand(1, 3, image_h, image_w, dtype=export_dtype, device=device)

    with torch.no_grad():
        image_embeddings = vision_model(dummy_pixel)
    print(f"  image_embeddings shape: {image_embeddings.shape}")
    n_image_tokens = image_embeddings.shape[1]

    vision_onnx = onnx_dir / "smolvla_vision.onnx"
    vision_simp = onnx_dir / "smolvla_vision_simp.onnx"
    export_onnx(vision_model, dummy_pixel, vision_onnx, ["pixel_values"], ["image_embeddings"])
    simplify_onnx(vision_onnx, vision_simp, {"pixel_values": [1, 3, image_h, image_w]})

    # ── Token embedding (CPU) ──
    print("\n=== Token embedding (CPU) ===")
    token_embedding = vlm_with_expert.get_vlm_model().text_model.get_input_embeddings()
    emb_file = out_dir / "token_embedding.pt"
    torch.save(
        {k: v.detach().cpu() for k, v in token_embedding.state_dict().items()},
        emb_file,
    )
    print(f"  Saved: {emb_file}")

    # ── Prefill ──
    print("\n=== Prefill module ===")
    prefill_model = SmolVLAPrefillPart(vlm_with_expert)
    prefill_model.eval()
    prefill_model.to(device=device, dtype=export_dtype)

    prefix_embs = torch.randn(1, prefix_length, prefix_hidden_size, device=device, dtype=export_dtype)
    attn_mask = torch.ones(1, prefix_length, prefix_length, device=device, dtype=torch.long)
    pos_ids = torch.arange(prefix_length, device=device, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        flat_cache = prefill_model(prefix_embs, attn_mask, pos_ids)
    cache_shape = list(flat_cache[0].shape)
    print(f"  KV cache: {num_layers} layers × 2 = {len(flat_cache)} tensors, shape={cache_shape}")

    prefill_names_in = build_prefill_input_names()
    prefill_names_out = build_cache_output_names(num_layers)
    prefill_onnx = onnx_dir / "smolvla_prefill.onnx"
    prefill_simp = onnx_dir / "smolvla_prefill_simp.onnx"
    export_onnx(
        prefill_model,
        (prefix_embs, attn_mask, pos_ids),
        prefill_onnx,
        prefill_names_in,
        prefill_names_out,
    )
    simplify_onnx(
        prefill_onnx,
        prefill_simp,
        {
            "prefix_embs": list(prefix_embs.shape),
            "attention_mask": list(attn_mask.shape),
            "position_ids": list(pos_ids.shape),
        },
    )

    # ── Action ──
    print("\n=== Action module ===")
    action_model = SmolVLAActionPart(flow_model)
    action_model.eval()
    action_model.to(device=device, dtype=export_dtype)

    x_t = torch.randn(1, chunk_size, max_action_dim, device=device, dtype=export_dtype)
    timestep = torch.ones(1, device=device, dtype=torch.float32)
    prefix_pad_masks = torch.ones(1, prefix_length, device=device, dtype=torch.bool)
    action_inputs = (x_t, timestep, prefix_pad_masks, *flat_cache)

    action_names_in = ["x_t", "timestep", "prefix_pad_masks", *build_cache_input_names(num_layers)]
    action_onnx = onnx_dir / "smolvla_action.onnx"
    action_simp = onnx_dir / "smolvla_action_simp.onnx"
    export_onnx(
        action_model,
        action_inputs,
        action_onnx,
        action_names_in,
        ["v_t"],
    )
    action_shapes = {
        "x_t": list(x_t.shape),
        "timestep": list(timestep.shape),
        "prefix_pad_masks": list(prefix_pad_masks.shape),
    }
    for name, tensor in zip(action_names_in[3:], flat_cache, strict=True):
        action_shapes[name] = list(tensor.shape)
    simplify_onnx(action_onnx, action_simp, action_shapes)

    # Runtime artifact paths (relative to manifest dir). The .rknn files are
    # produced by the subsequent convert_to_rknn.py step; the embedding is
    # already saved above. load_compiled_manifest requires this map and
    # SmolVLARKNNRuntimeSession.load resolves each via require_artifact.
    rknn_artifacts = {
        "vision": "onnx/smolvla_vision.rknn",
        "prefill": "onnx/smolvla_prefill.rknn",
        "action": "onnx/smolvla_action.rknn",
        "embedding": "token_embedding.pt",
    }

    # ── Meta info + config.rknn.json ──
    meta = {
        "policy_type": "smolvla",
        "backend": "rknn",
        "execution": ["vision", "prefill", "action"],
        "model_path": args.model_path,
        "image_height": image_h,
        "image_width": image_w,
        "prefix_length": prefix_length,
        "n_image_tokens": n_image_tokens,
        "prefix_hidden_size": prefix_hidden_size,
        "num_layers": num_layers,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "chunk_size": chunk_size,
        "max_action_dim": max_action_dim,
        "cache_shape_per_tensor": cache_shape,
        # Runtime artifact map consumed by load_compiled_manifest. Roles must
        # match SmolVLARKNNRuntimeSession.load's require_artifact calls.
        "artifacts": rknn_artifacts,
        # Architecture params read by SmolVLARKNNRuntimeSession.load and merged
        # into the _PI05ConfigView so SmolVLARKNNModel picks up real values
        # instead of getattr fallback defaults.
        "backend_config": {
            "num_layers": num_layers,
            "prefix_length": prefix_length,
            "prefix_hidden_size": prefix_hidden_size,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
        },
        "modules": {
            "vision": {
                "onnx": "onnx/smolvla_vision_simp.onnx",
                "input_names": ["pixel_values"],
                "output_names": ["image_embeddings"],
                "input_shapes": {"pixel_values": [1, 3, image_h, image_w]},
            },
            "prefill": {
                "onnx": "onnx/smolvla_prefill_simp.onnx",
                "input_names": prefill_names_in,
                "output_names": prefill_names_out,
                "input_shapes": {
                    "prefix_embs": list(prefix_embs.shape),
                    "attention_mask": list(attn_mask.shape),
                    "position_ids": list(pos_ids.shape),
                },
            },
            "action": {
                "onnx": "onnx/smolvla_action_simp.onnx",
                "input_names": action_names_in,
                "output_names": ["v_t"],
                "input_shapes": action_shapes,
            },
        },
        "embedding": "token_embedding.pt",
        "notes": [
            "3-module RKNN split mirroring HMM architecture.",
            "vision: pixel_values → image_embeddings (NPU)",
            "prefill: prefix_embs → KV cache (NPU)",
            "action: KV cache + x_t → v_t (NPU)",
            "token_embedding runs on CPU (text token lookup).",
            "Denoise loop (num_steps=10) runs on host CPU, calling action NPU each step.",
        ],
    }
    meta_file = out_dir / "config.rknn.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n=== Meta info saved: {meta_file} ===")

    # Copy config.json for inference_service
    src_config = Path(args.model_path) / "config.json"
    if src_config.exists():
        shutil.copyfile(src_config, out_dir / "config.json")

    print(f"\nDone! Output: {out_dir}")
    print(f"  ONNX files: {onnx_dir}")
    print(f"  Token embedding: {emb_file}")
    print(f"  Config: {meta_file}")
    print("\nNext: convert each ONNX to RKNN:")
    print("  .venv-rknn/bin/python .agents/skills/rknn-convert/convert_to_rknn.py \\")
    print(f"    --onnx {onnx_dir}/smolvla_vision_simp.onnx --output {onnx_dir}/smolvla_vision.rknn --mode float16")
    print("  .venv-rknn/bin/python .agents/skills/rknn-convert/convert_to_rknn.py \\")
    print(f"    --onnx {onnx_dir}/smolvla_prefill_simp.onnx --output {onnx_dir}/smolvla_prefill.rknn --mode float16")
    print("  .venv-rknn/bin/python .agents/skills/rknn-convert/convert_to_rknn.py \\")
    print(f"    --onnx {onnx_dir}/smolvla_action_simp.onnx --output {onnx_dir}/smolvla_action.rknn --mode float16")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SmolVLA as 3 segmented ONNX modules for RKNN")
    parser.add_argument("--model_path", type=str, required=True, help="SmolVLA policy path")
    parser.add_argument("--lerobot_src", type=str, default=str(DEFAULT_LEROBOT_SRC))
    parser.add_argument("--output_dir", type=str, default="models/smolvla_rknn", help="Output directory")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--prefix_length", type=int, default=177, help="Prefix sequence length")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    export_all(parser.parse_args())
