#!/usr/bin/env python
# Copyright (c) 2025 Syslong Technology Co., Ltd. All Rights Reserved.
# Copyright (c) 2025 Shanghai Jiao Tong University
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Verify that the split PI05 (VLM + Action Expert) produces equivalent
output to the original monolithic PI05 model.

The script loads all three models from the **same** pretrained checkpoint,
constructs identical dummy (or real) inputs, and compares:

1. **KV cache equivalence**: VLM split prefix encoding vs full model prefix encoding.
2. **Action equivalence**: full denoising loop (VLM → N × AE steps) vs
   monolithic ``sample_actions``.

When ONNX paths are provided (``--vlm-onnx-path`` and ``--ae-onnx-path``),
**only** the ONNX split is compared against the full PyTorch model; the
PyTorch split models are not loaded and the PyTorch-split check is skipped
(since the ONNX check already covers the full → split equivalence).

Usage example (dummy inputs — PyTorch split)::

    python verify_pi05_split_equivalence.py \
        --pretrained-policy-path /path/to/pi05-checkpoint \
        --device cpu \
        --seed 42

Usage example (real batches — PyTorch split)::

    python verify_pi05_split_equivalence.py \
        --pretrained-policy-path /path/to/pi05-checkpoint \
        --batch-path /path/to/batches.json \
        --key-map \
            observation.images.top_view=observation.images.top \
            observation.images.hand_view=observation.images.wrist \
        --task 'pick up the cup' \
        --device cpu

Usage example (ONNX split — only compares full PyTorch vs ONNX)::

    python verify_pi05_split_equivalence.py \
        --pretrained-policy-path /path/to/pi05-checkpoint \
        --vlm-onnx-path /path/to/vlm.onnx \
        --ae-onnx-path /path/to/action_expert.onnx \
        --device cpu \
        --seed 42

cosine ≥ 0.9999 → ✅ PASS（数值等价）
0.999 ≤ cosine < 0.9999 → ⚠️ MARGINAL（dtype 差异导致的微小偏差）
cosine < 0.999 → ❌ FAIL（模型不等价）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def abs_diff_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (max_abs_diff, mean_abs_diff)."""
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    if diff.size == 0:
        return 0.0, 0.0
    return float(np.max(diff)), float(np.mean(diff))


def cosine_similarity_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Cosine similarity per leading-dim vector → (min, max, mean)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim <= 1:
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)
    else:
        a = a.reshape(-1, a.shape[-1])
        b = b.reshape(-1, b.shape[-1])
    eps = 1e-12
    dot = (a * b).sum(axis=1)
    denom = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), eps)
    cos = dot / denom
    return float(cos.min()), float(cos.max()), float(cos.mean())


def report_kv_per_layer(tag: str, a: np.ndarray, b: np.ndarray) -> None:
    """Per-layer breakdown for KV cache shaped (L, 2, B, H, T, D).

    Reports min/mean cosine and max abs diff for every (layer, k/v) pair
    so we can localise which transformer block is dragging cos_min down.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 6:
        LOGGER.warning("[%s] per-layer breakdown skipped (shape=%s)", tag, a.shape)
        return
    L = a.shape[0]
    LOGGER.info("[%s] per-layer breakdown (L=%d, k/v separate):", tag, L)
    LOGGER.info("  %-5s %-3s  %-10s %-10s %-10s %-10s", "layer", "kv", "cos_min", "cos_mean", "max_diff", "amax")
    worst_cos_min = 1.0
    worst_layer = -1
    for li in range(L):
        for kv_idx, kv_name in enumerate(("k", "v")):
            sub_a = a[li, kv_idx]  # (B, H, T, D)
            sub_b = b[li, kv_idx]
            cmin, _, cmean = cosine_similarity_stats(sub_a, sub_b)
            mx, _ = abs_diff_metrics(sub_a, sub_b)
            amax = float(np.abs(sub_a).max())
            flag = ""
            if cmin < 0.99:
                flag = "  ❌"
            elif cmin < 0.999:
                flag = "  ⚠"
            if cmin < worst_cos_min:
                worst_cos_min = cmin
                worst_layer = li
            LOGGER.info("  %-5d %-3s  %-10.6f %-10.6f %-10.4g %-10.4g%s", li, kv_name, cmin, cmean, mx, amax, flag)
    LOGGER.info("[%s] worst layer: %d (cos_min=%.6f)", tag, worst_layer, worst_cos_min)


def report(tag: str, a: np.ndarray, b: np.ndarray) -> None:
    """Log comparison metrics between two arrays."""
    max_d, mean_d = abs_diff_metrics(a, b)
    cos_min, cos_max, cos_mean = cosine_similarity_stats(a, b)
    LOGGER.info("[%s] shape: %s vs %s", tag, a.shape, b.shape)
    LOGGER.info("[%s] max abs diff : %.6g", tag, max_d)
    LOGGER.info("[%s] mean abs diff: %.6g", tag, mean_d)
    LOGGER.info("[%s] cosine sim (min/max/mean): %.6f / %.6f / %.6f", tag, cos_min, cos_max, cos_mean)


# ---------------------------------------------------------------------------
# ONNX feed helpers
# ---------------------------------------------------------------------------

# Batch key → VLM ONNX input name mapping.
# The VLM converter exports with short names ("lang_tokens", "lang_masks"),
# while standard batches use the full observation key names.
_VLM_KEY_MAP: dict[str, str] = {
    "observation.language.tokens": "lang_tokens",
    "observation.language.attention_mask": "lang_masks",
}


def _has_bf16_io(session) -> bool:
    """Return True if any input/output of the ORT session is bfloat16."""
    for io in list(session.get_inputs()) + list(session.get_outputs()):  # noqa: SIM110
        if io.type == "tensor(bfloat16)":
            return True
    return False


def _torch_to_ort_value(t: Tensor):
    """Wrap a torch tensor as an OrtValue via DLPack (zero-copy when on the same device).

    bfloat16 tensors round-trip cleanly because torch / ORT both speak DLPack;
    numpy is bypassed entirely. Tries multiple ORT API spellings for cross-
    version compatibility (``ortvalue_from_dlpack`` / ``from_dlpack``), and
    finally drops to the underlying C++ ``OrtValue.from_dlpack`` that exists
    in ``onnxruntime.capi._pybind_state``.
    """
    import onnxruntime as ort

    t = t.contiguous()

    # bool tensors: skip DLPack entirely. ORT's DLPack importer (≤ 1.20)
    # does not recognise DLPack type code 6 (kDLBool); numpy handles bool
    # losslessly and the bf16 motivation for DLPack does not apply here.
    if t.dtype == torch.bool:
        return ort.OrtValue.ortvalue_from_numpy(t.detach().cpu().numpy())

    capsule = torch.utils.dlpack.to_dlpack(t)

    # 1. Newer ORT (>=1.10): classmethod on OrtValue.
    if hasattr(ort.OrtValue, "ortvalue_from_dlpack"):
        try:
            return ort.OrtValue.ortvalue_from_dlpack(capsule)
        except Exception:  # noqa: BLE001
            capsule = torch.utils.dlpack.to_dlpack(t)  # capsule is single-use
    # 2. Some builds expose a direct ``from_dlpack`` (signature varies).
    if hasattr(ort.OrtValue, "from_dlpack"):
        try:
            return ort.OrtValue.from_dlpack(capsule, t.dtype == torch.bool)
        except TypeError:
            return ort.OrtValue.from_dlpack(capsule)
        except Exception:  # noqa: BLE001
            capsule = torch.utils.dlpack.to_dlpack(t)
    # 3. Fall back to the underlying C++ OrtValue (always present in ORT >=1.20).
    try:
        from onnxruntime.capi._pybind_state import OrtValue as C_OrtValue  # type: ignore

        try:
            inner = C_OrtValue.from_dlpack(capsule, t.dtype == torch.bool)
        except TypeError:
            inner = C_OrtValue.from_dlpack(capsule)
        # Wrap back as the Python OrtValue so the rest of the API works.
        return ort.OrtValue(inner)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Installed onnxruntime build does not expose a DLPack import API on OrtValue ({exc}); "
            "bf16 inputs cannot be passed without numpy support."
        ) from exc


def _ort_value_to_torch(ov) -> Tensor:
    """Convert an OrtValue back to a torch tensor.

    Tries (in order):
      1. ``torch.from_dlpack(ov)`` — works when OrtValue implements ``__dlpack__``
         (recent onnxruntime, follows Python array API standard).
      2. ``torch.utils.dlpack.from_dlpack(ov.to_dlpack())`` — older API.
      3. ``ov._ortvalue.to_dlpack()`` — the C++ pybind layer often exposes
         DLPack even when the Python wrapper doesn't (true for the
         standard ``onnxruntime`` / ``onnxruntime-gpu`` builds).
      4. ``torch.from_numpy(ov.numpy())`` — fallback for non-bf16 dtypes.
    """
    # 1. Array-API style: ORT exposes __dlpack__ → torch consumes directly.
    if hasattr(ov, "__dlpack__"):
        try:
            return torch.from_dlpack(ov)
        except Exception:  # noqa: BLE001
            pass
    # 2. Legacy explicit to_dlpack() on the Python wrapper.
    if hasattr(ov, "to_dlpack"):
        try:
            return torch.utils.dlpack.from_dlpack(ov.to_dlpack())
        except Exception:  # noqa: BLE001
            pass
    # 3. Drop down to the underlying C++ OrtValue, which usually has to_dlpack
    #    even when the Python wrapper omits it (common in ORT >=1.20).
    inner = getattr(ov, "_ortvalue", None)
    if inner is not None and hasattr(inner, "to_dlpack"):
        try:
            return torch.utils.dlpack.from_dlpack(inner.to_dlpack())
        except Exception:  # noqa: BLE001
            pass
    # 4. Last resort — numpy bridge (will raise on bf16).
    return torch.from_numpy(ov.numpy())


def _ort_value_to_fp32_numpy(ov) -> np.ndarray:
    """Convert an OrtValue to a fp32 numpy array (going through torch for bf16 safety)."""
    t = _ort_value_to_torch(ov)
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t.detach().cpu().numpy()


def _build_vlm_onnx_feed(
    batch: dict[str, Tensor],
    valid_names: set[str],
) -> dict[str, np.ndarray]:
    """Convert a batch dict to a numpy feed dict for the VLM ONNX session.

    Handles key remapping (``observation.language.tokens`` → ``lang_tokens``, etc.)
    and dtype conversion (bool tensors stay bool, everything else keeps its numpy
    equivalent).
    """
    feed: dict[str, np.ndarray] = {}
    for key, tensor in batch.items():
        onnx_name = _VLM_KEY_MAP.get(key, key)
        if onnx_name not in valid_names:
            continue
        arr = tensor.cpu().numpy()
        if tensor.dtype == torch.bool:
            arr = arr.astype(bool)
        feed[onnx_name] = arr
    return feed


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_full_model(policy_path: str, device: torch.device, *, local_files_only: bool):
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    LOGGER.info("Loading full PI05Policy from %s …", policy_path)
    policy = PI05Policy.from_pretrained(policy_path, local_files_only=local_files_only, strict=False)
    policy.to(device)
    policy.eval()
    return policy


def load_vlm_model(policy_path: str, device: torch.device, *, local_files_only: bool):
    from model_utils.pi05_export.modeling_pi05_vlm import PI05VLMPolicy

    LOGGER.info("Loading PI05VLMPolicy from %s …", policy_path)
    policy = PI05VLMPolicy.from_pretrained(policy_path, local_files_only=local_files_only, strict=False)
    policy.to(device)
    policy.eval()
    return policy


def load_ae_model(policy_path: str, device: torch.device, *, local_files_only: bool):
    from model_utils.pi05_export.modeling_pi05_action_expert import PI05ActionExpertPolicy

    LOGGER.info("Loading PI05ActionExpertPolicy from %s …", policy_path)
    policy = PI05ActionExpertPolicy.from_pretrained(policy_path, local_files_only=local_files_only, strict=False)
    policy.to(device)
    policy.eval()
    return policy


# ---------------------------------------------------------------------------
# Dummy input generation
# ---------------------------------------------------------------------------


def make_dummy_batch(
    config,
    device: torch.device,
    seed: int,
    batch_size: int = 1,
) -> dict[str, Tensor]:
    """Create a dummy observation batch matching the policy config."""
    torch.manual_seed(seed)

    batch: dict[str, Tensor] = {}

    # Images — one per camera defined in config.image_features
    image_features = config.image_features  # dict[str, PolicyFeature]
    for key, feature in image_features.items():
        shape = feature.shape  # (C, H, W)
        img = torch.rand(batch_size, *shape, dtype=torch.float32, device=device)
        batch[key] = img

    # Language tokens
    token_len = getattr(config, "tokenizer_max_length", 200)
    batch["observation.language.tokens"] = torch.randint(
        0, 1000, (batch_size, token_len), dtype=torch.long, device=device
    )
    batch["observation.language.attention_mask"] = torch.ones(batch_size, token_len, dtype=torch.bool, device=device)

    return batch


# ---------------------------------------------------------------------------
# Real batch loading (reuse loss_compare format)
# ---------------------------------------------------------------------------


def load_real_batches_raw(batch_path: str) -> list[dict[str, np.ndarray]]:
    """Load batches from a JSON file as numpy float32 dicts.

    This returns *raw* numpy arrays — no tensor conversion, no device
    transfer, no tokenization.  Use :func:`preprocess_real_batches` to
    run the full preprocessing pipeline afterwards.
    """
    LOGGER.info("Loading batches from %s …", batch_path)
    with open(batch_path, encoding="utf-8") as f:
        raw_batches = json.load(f)

    processed = []
    for b in raw_batches:
        batch: dict[str, np.ndarray] = {}
        for k, v in b.items():
            batch[k] = np.array(v).astype(np.float32)
        processed.append(batch)

    LOGGER.info("Loaded %d raw batch(es)", len(processed))
    return processed


def remap_batch_keys(
    batches: list[dict[str, Any]],
    key_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Rename keys in each batch dict according to *key_map*.

    *key_map* maps ``src_key → dst_key``.  Keys not in the map are kept
    unchanged.  If a *dst_key* already exists in the batch it will be
    overwritten (with a warning).
    """
    if not key_map:
        return batches

    LOGGER.info("Applying key remapping: %s", key_map)
    remapped = []
    for batch in batches:
        new_batch: dict[str, Any] = {}
        for k, v in batch.items():
            dst = key_map.get(k, k)
            if dst in new_batch:
                LOGGER.warning("Key remap collision: '%s' → '%s' overwrites existing key", k, dst)
            new_batch[dst] = v
        remapped.append(new_batch)
    return remapped


def preprocess_real_batches(
    raw_batches: list[dict[str, np.ndarray]],
    policy_path: str,
    full_policy,
    device: torch.device,
    task: str = "",
) -> list[dict[str, Tensor]]:
    """Run the full preprocessing pipeline on raw numpy batches.

    Mirrors the flow in ``loss_compare.py``'s ``predict_action``:

    1. ``prepare_observation_for_inference`` — numpy → tensor, image /255 +
       permute, add batch dim, set task string.
    2. ``preprocessor(observation)`` — rename → normalize → prompt generation
       (discretise state + build ``"Task: …, State: …;\\nAction: "`` prompt)
       → tokenise (creates ``observation.language.tokens`` /
       ``observation.language.attention_mask``) → move to device.
    """
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=full_policy,
        pretrained_path=policy_path,
    )
    LOGGER.info("Preprocessor pipeline created — running on %d batch(es) …", len(raw_batches))

    result: list[dict[str, Tensor]] = []
    for raw_batch in raw_batches:
        obs = copy(raw_batch)  # avoid mutating the original
        obs = prepare_observation_for_inference(obs, device, task)
        obs = preprocessor(obs)
        # The preprocessor's DeviceProcessorStep may move tensors to the
        # device stored in the checkpoint config (e.g. cuda:0).  Ensure
        # everything lands on the user-requested device.
        # Also strip non-tensor entries (e.g. "task", "robot_type" strings)
        # that were consumed by the preprocessor.
        clean: dict[str, Tensor] = {}
        for k, v in obs.items():
            if isinstance(v, Tensor):
                clean[k] = v.to(device) if v.device != device else v
        result.append(clean)

    LOGGER.info("Preprocessing complete.")
    return result


# ---------------------------------------------------------------------------
# Core equivalence verification
# ---------------------------------------------------------------------------


@torch.no_grad()
def verify_kv_cache(
    full_policy,
    vlm_policy,
    batch: dict[str, Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compare prefix encoding (KV cache) between full and VLM models.

    Returns (full_kv_np, vlm_kv_np, full_masks_np, vlm_masks_np).
    """
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    from model_utils.pi05_export.modeling_pi05_vlm import flatten_kv, make_att_2d_masks

    # --- Full model: run prefix encoding only ---
    images_full, img_masks_full = full_policy._preprocess_images(batch)
    tokens_full = batch[OBS_LANGUAGE_TOKENS]
    masks_full = batch[OBS_LANGUAGE_ATTENTION_MASK]

    full_model = full_policy.model
    prefix_embs, prefix_pad_masks, prefix_att_masks = full_model.embed_prefix(
        images_full, img_masks_full, tokens_full, masks_full
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = full_model._prepare_attention_masks_4d(prefix_att_2d_masks)
    full_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

    _, past_kv_full = full_model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    full_kv_tensor = flatten_kv(past_kv_full)
    full_masks = prefix_pad_masks

    # --- VLM model: run select_action ---
    vlm_kv_tensor, vlm_masks = vlm_policy.select_action(batch)

    full_kv_np = full_kv_tensor.cpu().float().numpy()
    vlm_kv_np = vlm_kv_tensor.cpu().float().numpy()
    full_masks_np = full_masks.cpu().numpy()
    vlm_masks_np = vlm_masks.cpu().numpy()

    return full_kv_np, vlm_kv_np, full_masks_np, vlm_masks_np


@torch.no_grad()
def verify_actions(
    full_policy,
    vlm_policy,
    ae_policy,
    batch: dict[str, Tensor],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare full denoising (monolithic) vs split (VLM + N × AE).

    Both use the same noise and the same number of denoising steps.

    Returns (full_actions_np, split_actions_np).
    """
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    from model_utils.pi05_export.modeling_pi05_action_expert import unflatten_kv

    config = full_policy.config
    device = next(full_policy.parameters()).device
    bsize = next(v.shape[0] for v in batch.values() if isinstance(v, Tensor))
    num_steps = config.num_inference_steps

    # --- Generate shared noise (float32, same seed) ---
    torch.manual_seed(seed)
    noise = torch.normal(
        mean=0.0,
        std=1.0,
        size=(bsize, config.chunk_size, config.max_action_dim),
        dtype=torch.float32,
        device=device,
    )

    # ========== Full model ==========
    LOGGER.info("Running full model sample_actions …")
    images_full, img_masks_full = full_policy._preprocess_images(batch)
    tokens_full = batch[OBS_LANGUAGE_TOKENS]
    masks_full = batch[OBS_LANGUAGE_ATTENTION_MASK]

    t0 = time.perf_counter()
    full_actions = full_policy.model.sample_actions(
        images_full,
        img_masks_full,
        tokens_full,
        masks_full,
        noise=noise.clone(),
        num_steps=num_steps,
    )
    t_full = time.perf_counter() - t0
    LOGGER.info("Full model inference: %.4f sec", t_full)

    # Unpad to original action dim
    original_action_dim = config.output_features[ACTION].shape[0]
    full_actions = full_actions[:, :, :original_action_dim]

    # ========== Split model: VLM + N × AE ==========
    LOGGER.info("Running split model (VLM + %d × AE) …", num_steps)

    # Step 1: VLM prefix encoding
    t0 = time.perf_counter()
    past_kv_tensor, prefix_pad_masks = vlm_policy.select_action(batch)
    t_vlm = time.perf_counter() - t0
    LOGGER.info("  VLM encoding: %.4f sec", t_vlm)

    # Step 2: Denoising loop — mirror the full model's while loop exactly
    # Full model uses float32 for dt and time.
    # AE's sample_actions does a single step with float16 dt.
    # For faithful comparison, we replicate the full model's loop logic at float32
    # and call AE's denoise_step directly.
    ae_model = ae_policy.model
    past_key_values = unflatten_kv(past_kv_tensor)

    dt = -1.0 / num_steps
    dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)
    x_t = noise.clone()
    time_val = torch.tensor(1.0, dtype=torch.float32, device=device)

    t0 = time.perf_counter()
    while time_val >= -dt_tensor / 2:
        expanded_time = time_val.expand(bsize)
        v_t = ae_model.denoise_step(
            prefix_pad_masks,
            past_key_values,
            x_t,
            expanded_time,
        )
        # Keep float32 precision for Euler update (matches full model)
        x_t = x_t + dt_tensor * v_t.float()
        time_val = time_val + dt_tensor
    t_ae = time.perf_counter() - t0
    LOGGER.info("  AE denoising (%d steps): %.4f sec", num_steps, t_ae)
    LOGGER.info("  Split total: %.4f sec", t_vlm + t_ae)

    split_actions = x_t[:, :, :original_action_dim]

    full_np = full_actions.cpu().float().numpy()
    split_np = split_actions.cpu().float().numpy()

    return full_np, split_np


# ---------------------------------------------------------------------------
# ONNX equivalence verification
# ---------------------------------------------------------------------------


