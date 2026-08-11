#!/usr/bin/env python
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
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
import logging
import os
import time
from collections.abc import Sequence
from copy import copy
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from inference_service.pi05_schedule import PI05DenoisingSchedule, load_pi05_schedule, uniform_pi05_schedule

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


def load_real_batches_raw(batch_path: str) -> list[dict[str, Any]]:
    """Load a standard observation batch or legacy JSON as raw sample dicts.

    This returns *raw* numpy arrays — no tensor conversion, no device
    transfer, no tokenization.  Use :func:`preprocess_real_batches` to
    run the full preprocessing pipeline afterwards.
    """
    from model_utils.observation_batch import load_observation_batch

    LOGGER.info("Loading batches from %s …", batch_path)
    samples = load_observation_batch(batch_path).samples
    LOGGER.info("Loaded %d raw batch(es)", len(samples))
    return samples


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
    raw_batches: list[dict[str, Any]],
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

    from inference_service.lerobot_assets import TOKENIZER_REFERENCE_KEYS, resolve_local_semantic_reference

    preprocessor_overrides = None
    tokenizer_path = resolve_local_semantic_reference(
        Path(policy_path).resolve(),
        "policy_preprocessor.json",
        TOKENIZER_REFERENCE_KEYS,
    )
    if tokenizer_path is not None:
        preprocessor_overrides = {"tokenizer_processor": {"tokenizer_name": tokenizer_path}}
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=full_policy,
        pretrained_path=policy_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    LOGGER.info("Preprocessor pipeline created — running on %d batch(es) …", len(raw_batches))

    result: list[dict[str, Tensor]] = []
    for raw_batch in raw_batches:
        obs = copy(raw_batch)  # avoid mutating the original
        batch_task = obs.pop("task", "")
        obs = prepare_observation_for_inference(obs, device, task or batch_task)
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
    full_model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"

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
def _integrate_full_velocity(full_model, prefix_pad_masks, past_key_values, noise, schedule):
    x_t = noise.clone()
    bsize = noise.shape[0]
    for time_value, next_time_value in pairwise(schedule.timesteps):
        timestep = torch.tensor(time_value, dtype=torch.float32, device=noise.device).expand(bsize)
        velocity = full_model.denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)
        x_t = x_t + (next_time_value - time_value) * velocity.float()
    return x_t


@torch.no_grad()
def verify_actions(
    full_policy,
    vlm_policy,
    ae_policy,
    batch: dict[str, Tensor],
    seed: int,
    timesteps: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare full denoising (monolithic) vs split (VLM + N × AE).

    Both use the same noise and the same number of denoising steps.

    Returns (full_actions_np, split_actions_np).
    """
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    from model_utils.pi05_export.modeling_pi05_action_expert import make_att_2d_masks, unflatten_kv

    config = full_policy.config
    device = next(full_policy.parameters()).device
    bsize = next(v.shape[0] for v in batch.values() if isinstance(v, Tensor))
    schedule = (
        uniform_pi05_schedule(config.num_inference_steps)
        if timesteps is None
        else PI05DenoisingSchedule(name="explicit", timesteps=tuple(float(value) for value in timesteps))
    )
    num_steps = schedule.step_count

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
    if timesteps is None:
        full_actions = full_policy.model.sample_actions(
            images_full,
            img_masks_full,
            tokens_full,
            masks_full,
            noise=noise.clone(),
            num_steps=num_steps,
        )
    else:
        full_model = full_policy.model
        prefix_embs, prefix_pad_masks_full, prefix_att_masks = full_model.embed_prefix(
            images_full, img_masks_full, tokens_full, masks_full
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks_full, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks_full, dim=1) - 1
        prefix_att_2d_masks_4d = full_model._prepare_attention_masks_4d(prefix_att_2d_masks)
        full_model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        _, past_kv_full = full_model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        full_actions = _integrate_full_velocity(
            full_model,
            prefix_pad_masks_full,
            past_kv_full,
            noise,
            schedule,
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

    x_t = noise.clone()

    t0 = time.perf_counter()
    for time_value, next_time_value in pairwise(schedule.timesteps):
        time_val = torch.tensor(time_value, dtype=torch.float32, device=device)
        dt_tensor = torch.tensor(next_time_value - time_value, dtype=torch.float32, device=device)
        expanded_time = time_val.expand(bsize)
        v_t = ae_model.denoise_step(
            prefix_pad_masks,
            past_key_values,
            x_t,
            expanded_time,
        )
        # Keep float32 precision for Euler update (matches full model)
        x_t = x_t + dt_tensor * v_t.float()
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


@torch.no_grad()
def verify_onnx_actions(
    full_policy,
    batch: dict[str, Tensor],
    vlm_onnx_path: str,
    ae_onnx_path: str,
    seed: int,
    timesteps: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compare full model (PyTorch) vs split ONNX (VLM ONNX + N × AE ONNX).

    The VLM ONNX model produces ``past_kv_tensor`` + ``prefix_pad_masks``.
    The AE ONNX model is called *N* times in a denoising loop; each call
    returns velocity, which this utility integrates with the declared timesteps.

    Returns a tuple of:
        ``(full_actions_np, onnx_actions_np,
           full_kv_np, onnx_kv_np,
           full_masks_np, onnx_masks_np)``.

    The ``*_kv_np`` / ``*_masks_np`` pair lets callers directly diff the
    full-PT-VLM KV cache against the VLM ONNX KV cache (i.e. measure the
    PT→ONNX export error in isolation, without it being entangled with
    the AE denoising path).
    """
    import onnxruntime as ort
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    from model_utils.pi05_export.modeling_pi05_vlm import flatten_kv, make_att_2d_masks

    config = full_policy.config
    device = next(full_policy.parameters()).device
    bsize = next(v.shape[0] for v in batch.values() if isinstance(v, Tensor))
    schedule = (
        uniform_pi05_schedule(config.num_inference_steps)
        if timesteps is None
        else PI05DenoisingSchedule(name="explicit", timesteps=tuple(float(value) for value in timesteps))
    )
    num_steps = schedule.step_count

    # --- Generate shared noise (float32, same seed) ---
    torch.manual_seed(seed)
    noise = torch.normal(
        mean=0.0,
        std=1.0,
        size=(bsize, config.chunk_size, config.max_action_dim),
        dtype=torch.float32,
        device=device,
    )

    # ========== Full model (PyTorch) ==========
    LOGGER.info("Running full model sample_actions (PyTorch) …")
    images_full, img_masks_full = full_policy._preprocess_images(batch)
    tokens_full = batch[OBS_LANGUAGE_TOKENS]
    masks_full = batch[OBS_LANGUAGE_ATTENTION_MASK]

    # --- Also extract the full model's prefix KV cache for direct
    #     comparison against the VLM ONNX output (computed below). This
    #     mirrors verify_kv_cache() but reuses the full PyTorch model as
    #     the baseline (no PT split model needed).
    full_model = full_policy.model
    prefix_embs, prefix_pad_masks_full, prefix_att_masks = full_model.embed_prefix(
        images_full, img_masks_full, tokens_full, masks_full
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks_full, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks_full, dim=1) - 1
    prefix_att_2d_masks_4d = full_model._prepare_attention_masks_4d(prefix_att_2d_masks)
    full_model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
    _, past_kv_full = full_model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    full_kv_tensor = flatten_kv(past_kv_full)
    full_kv_np = full_kv_tensor.cpu().float().numpy()
    full_masks_np = prefix_pad_masks_full.cpu().numpy()

    t0 = time.perf_counter()
    if timesteps is None:
        full_actions = full_policy.model.sample_actions(
            images_full,
            img_masks_full,
            tokens_full,
            masks_full,
            noise=noise.clone(),
            num_steps=num_steps,
        )
    else:
        full_actions = _integrate_full_velocity(
            full_model,
            prefix_pad_masks_full,
            past_kv_full,
            noise,
            schedule,
        )
    t_full = time.perf_counter() - t0
    LOGGER.info("Full model PyTorch inference: %.4f sec", t_full)

    original_action_dim = config.output_features[ACTION].shape[0]
    full_actions = full_actions[:, :, :original_action_dim]

    # ========== ONNX split: VLM ONNX + N × AE ONNX ==========
    LOGGER.info("Running ONNX split (VLM + %d × AE) …", num_steps)

    # --- Step 1: VLM ONNX inference ---
    vlm_options = ort.SessionOptions()
    vlm_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    os.environ.setdefault("ORT_SEED", "42")

    # Pick ORT providers from --device (mirrors the torch device).
    # CPU is always appended as a fallback so unsupported ops still run.
    if device.type == "cuda":
        ort_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        ort_providers = ["CPUExecutionProvider"]
    LOGGER.info("  ORT providers (from --device=%s): %s", device, ort_providers)

    vlm_session = ort.InferenceSession(
        str(vlm_onnx_path),
        sess_options=vlm_options,
        providers=ort_providers,
    )
    vlm_valid = {inp.name for inp in vlm_session.get_inputs()}
    LOGGER.info("  VLM ONNX inputs: %s", sorted(vlm_valid))
    vlm_feed = _build_vlm_onnx_feed(batch, vlm_valid)

    # Stage C — Plan A: build prefix_att_2d_masks_4d on host if the
    # ONNX model expects it (post-Plan-A exports moved this constant
    # out of the graph to avoid ATC fp16 corruption).
    if "prefix_att_2d_masks_4d" in vlm_valid and "prefix_att_2d_masks_4d" not in vlm_feed:
        from model_utils.pi05_export.prefix_mask_utils import build_prefix_att_2d_masks_4d_np

        lang_masks_np = batch[OBS_LANGUAGE_ATTENTION_MASK].cpu().numpy().astype(bool)
        num_cameras = len([k for k in batch if k.startswith("observation.images.")])

        # Determine prefix_seq_len from the ONNX input metadata
        prefix_meta = next(inp for inp in vlm_session.get_inputs() if inp.name == "prefix_att_2d_masks_4d")
        prefix_seq_len = prefix_meta.shape[-1]  # (B, 1, S, S)
        LOGGER.info("  Building prefix_att_2d_masks_4d: num_cameras=%d, prefix_seq_len=%s", num_cameras, prefix_seq_len)

        prefix_mask = build_prefix_att_2d_masks_4d_np(
            num_cameras=num_cameras,
            lang_masks=lang_masks_np,
            prefix_seq_len=int(prefix_seq_len),
        )
        vlm_feed["prefix_att_2d_masks_4d"] = prefix_mask.astype(np.float32)

    LOGGER.info("  VLM ONNX feed keys: %s", sorted(vlm_feed.keys()))

    # Default: original numpy path (unchanged for fp16/fp32 models).
    # Fallback: bf16-aware OrtValue + DLPack path — numpy can't represent
    # bfloat16, so we route through torch (which supports bf16 natively).
    vlm_use_ortvalue = _has_bf16_io(vlm_session)
    if vlm_use_ortvalue:
        LOGGER.info("  VLM ONNX has bf16 IO — using OrtValue + DLPack bridge")

    # Target device for OrtValue placement (only used in the bf16 path).
    ort_target_device = "cuda" if device.type == "cuda" else "cpu"
    ort_target_device_id = device.index or 0

    t0 = time.perf_counter()
    if vlm_use_ortvalue:
        ort_inputs: dict[str, Any] = {}
        for name, arr in vlm_feed.items():
            ort_inputs[name] = ort.OrtValue.ortvalue_from_numpy(arr, ort_target_device, ort_target_device_id)
        ort_outputs = vlm_session.run_with_ort_values(None, ort_inputs)
        # Convert outputs to fp32 numpy via torch (bf16 safe).
        vlm_outputs = [_ort_value_to_fp32_numpy(ov) for ov in ort_outputs]
    else:
        vlm_outputs = vlm_session.run(None, vlm_feed)
    t_vlm = time.perf_counter() - t0
    LOGGER.info("  VLM ONNX: %.4f sec", t_vlm)

    past_kv_np = np.asarray(vlm_outputs[0])  # past_kv_tensor
    prefix_masks_np = np.asarray(vlm_outputs[1])  # prefix_pad_masks
    LOGGER.info(
        "  VLM ONNX out — past_kv: %s %s, masks: %s %s",
        past_kv_np.shape,
        past_kv_np.dtype,
        prefix_masks_np.shape,
        prefix_masks_np.dtype,
    )

    # --- Step 2: AE ONNX denoising loop ---
    ae_session = ort.InferenceSession(
        str(ae_onnx_path),
        providers=ort_providers,
    )
    ae_valid = {inp.name for inp in ae_session.get_inputs()}
    LOGGER.info("  AE ONNX inputs: %s", sorted(ae_valid))

    # Auto-detect float dtype from the AE ONNX model's "noise" input.
    # fp16 / fp32 → original numpy loop (unchanged behavior).
    # bf16        → fallback torch + DLPack loop (numpy lacks bfloat16).
    _onnx_type_map = {"tensor(float16)": np.float16, "tensor(float)": np.float32}
    _onnx_to_torch = {
        "tensor(float16)": torch.float16,
        "tensor(float)": torch.float32,
        "tensor(bfloat16)": torch.bfloat16,
    }
    ae_input_meta = {inp.name: inp for inp in ae_session.get_inputs()}
    ae_noise_type = ae_input_meta["noise"].type if "noise" in ae_input_meta else "tensor(float16)"
    ae_use_ortvalue = _has_bf16_io(ae_session) or ae_noise_type == "tensor(bfloat16)"

    # Per-input torch dtype map (auto mode may produce mixed bf16/fp32 inputs:
    # past_kv_tensor=bf16 from gemma expert, noise/time=fp32 from projection layers).
    def _ae_dtype(name: str, default: torch.dtype) -> torch.dtype:
        meta = ae_input_meta.get(name)
        if meta is None:
            return default
        return _onnx_to_torch.get(meta.type, default)

    if ae_use_ortvalue:
        torch_dtype = _onnx_to_torch.get(ae_noise_type, torch.bfloat16)
        ae_dtypes = {
            "noise": _ae_dtype("noise", torch_dtype),
            "past_kv_tensor": _ae_dtype("past_kv_tensor", torch_dtype),
            "time": _ae_dtype("time", torch_dtype),
        }
        LOGGER.info(
            "  AE ONNX has bf16 IO — using OrtValue + DLPack bridge (per-input dtypes: %s)",
            {k: str(v) for k, v in ae_dtypes.items()},
        )
    else:
        onnx_float_dtype = _onnx_type_map.get(ae_noise_type, np.float16)
        LOGGER.info("  AE ONNX float dtype (auto-detected): %s", onnx_float_dtype)

    t0 = time.perf_counter()
    step_count = 0

    if ae_use_ortvalue:
        # bf16 fallback path: keep state as torch tensors on the AE device,
        # feed via DLPack. This avoids any numpy bf16 conversion.
        ae_torch_device = (
            torch.device(f"cuda:{ort_target_device_id}") if ort_target_device == "cuda" else torch.device("cpu")
        )

        x_t_t = noise.to(device=ae_torch_device, dtype=torch.float32).contiguous()
        past_kv_t = (
            torch.from_numpy(past_kv_np).to(device=ae_torch_device, dtype=ae_dtypes["past_kv_tensor"]).contiguous()
        )
        prefix_masks_t = torch.from_numpy(prefix_masks_np.astype(bool)).to(ae_torch_device).contiguous()

        for time_val_py, next_time_val_py in pairwise(schedule.timesteps):
            dt_py = next_time_val_py - time_val_py
            time_t = torch.full((bsize,), time_val_py, dtype=ae_dtypes["time"], device=ae_torch_device)
            ae_feed_ov = {
                "past_kv_tensor": _torch_to_ort_value(past_kv_t),
                "prefix_pad_masks": _torch_to_ort_value(prefix_masks_t),
                "time": _torch_to_ort_value(time_t),
                "noise": _torch_to_ort_value(x_t_t.to(dtype=ae_dtypes["noise"]).contiguous()),
            }
            ae_feed_ov = {k: v for k, v in ae_feed_ov.items() if k in ae_valid}

            ae_outputs = ae_session.run_with_ort_values(None, ae_feed_ov)
            velocity_t = _ort_value_to_torch(ae_outputs[0]).float()
            x_t_t = x_t_t + dt_py * velocity_t

            step_count += 1

        if x_t_t.dtype == torch.bfloat16:
            x_t_t = x_t_t.float()
        x_t_np = x_t_t.detach().cpu().numpy()
    else:
        # Original numpy path — unchanged behavior for fp16 / fp32 models.
        x_t_np = noise.cpu().numpy().astype(np.float32)
        past_kv_cast = past_kv_np.astype(onnx_float_dtype)
        prefix_masks_bool = prefix_masks_np.astype(bool)

        for time_value, next_time_value in pairwise(schedule.timesteps):
            dt = next_time_value - time_value
            time_val = onnx_float_dtype(time_value)
            time_arr = np.full((bsize,), time_val, dtype=onnx_float_dtype)
            ae_feed = {
                "past_kv_tensor": past_kv_cast,
                "prefix_pad_masks": prefix_masks_bool,
                "time": time_arr,
                "noise": x_t_np.astype(onnx_float_dtype),
            }
            # Filter by valid input names
            ae_feed = {k: v for k, v in ae_feed.items() if k in ae_valid}

            ae_outputs = ae_session.run(None, ae_feed)
            velocity = np.asarray(ae_outputs[0], dtype=np.float32)
            x_t_np = x_t_np + np.float32(dt) * velocity

            step_count += 1

    t_ae = time.perf_counter() - t0
    LOGGER.info("  AE ONNX denoising (%d steps): %.4f sec", step_count, t_ae)
    LOGGER.info("  ONNX total: %.4f sec", t_vlm + t_ae)

    onnx_actions = x_t_np[:, :, :original_action_dim]

    full_np = full_actions.cpu().float().numpy()
    onnx_np = onnx_actions.astype(np.float32)

    # Cast VLM ONNX KV / masks to float32 / bool-as-int for reporting
    onnx_kv_np = past_kv_np.astype(np.float32)
    onnx_masks_np = prefix_masks_np.astype(full_masks_np.dtype)

    return full_np, onnx_np, full_kv_np, onnx_kv_np, full_masks_np, onnx_masks_np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Verify PI05 split (VLM + AE) equivalence against the monolithic model.")
    p.add_argument(
        "--pretrained-policy-path",
        type=str,
        required=True,
        help="Path to the pretrained PI05 checkpoint (shared by all three models).",
    )
    p.add_argument("--device", type=str, default="cpu", help="Torch device (cpu, cuda:0, …)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for dummy inputs and noise.")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size for dummy inputs.")
    p.add_argument(
        "--batch-path",
        type=str,
        default=None,
        help="Optional path to real batches JSON (loss_compare format). "
        "If provided, --batch-size and --seed (for input generation) are ignored.",
    )
    p.add_argument("--max-batches", type=int, default=None, help="Max number of batches to test (for real batches).")
    p.add_argument(
        "--skip-kv-check",
        action="store_true",
        help="Skip the KV-cache equivalence check (only compare final actions).",
    )
    p.add_argument(
        "--vlm-onnx-path",
        type=str,
        default=None,
        help="Path to the exported VLM ONNX model. "
        "When both --vlm-onnx-path and --ae-onnx-path are given, "
        "only the ONNX-vs-full-PyTorch comparison is performed "
        "(the PyTorch split check is skipped).",
    )
    p.add_argument(
        "--ae-onnx-path",
        type=str,
        default=None,
        help="Path to the exported Action Expert ONNX model. Must be provided together with --vlm-onnx-path.",
    )
    p.add_argument(
        "--skip-onnx-check",
        action="store_true",
        help="Skip ONNX verification even when ONNX paths are provided.",
    )
    p.add_argument(
        "--schedule-file",
        type=str,
        default=None,
        help="Optional strict PI0.5 denoising schedule JSON used by split velocity integration.",
    )
    p.add_argument(
        "--key-map",
        type=str,
        nargs="*",
        default=None,
        metavar="SRC=DST",
        help="Remap batch keys for real batches. Each entry is SRC=DST, e.g. "
        "'observation.images.top_view=observation.images.top' "
        "'observation.images.hand_view=observation.images.wrist'. "
        "Keys not listed are kept unchanged.",
    )
    p.add_argument(
        "--task",
        type=str,
        required=True,
        help="Natural-language task prompt used by the deployed policy for "
        "prompt generation (e.g. 'pick up the cup'). Pass the same text as the "
        "robot contract.default_task so verification matches deployment "
        "conditioning instead of an empty prompt.",
    )
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--local-files-only", action="store_true", default=True)
    return p


def main() -> int:
    args = build_parser().parse_args()

    from model_utils.pi05_export._cli_ui import setup_logging

    setup_logging(args.log_level)

    timesteps = load_pi05_schedule(args.schedule_file).timesteps if args.schedule_file else None

    device = torch.device(args.device)
    policy_path = args.pretrained_policy_path

    # Determine whether ONNX verification should run
    run_onnx = not args.skip_onnx_check and args.vlm_onnx_path is not None and args.ae_onnx_path is not None
    if run_onnx:
        vlm_onnx = Path(args.vlm_onnx_path)
        ae_onnx = Path(args.ae_onnx_path)
        if not vlm_onnx.exists():
            LOGGER.error("VLM ONNX model not found: %s", vlm_onnx)
            return 1
        if not ae_onnx.exists():
            LOGGER.error("AE ONNX model not found: %s", ae_onnx)
            return 1
        LOGGER.info("ONNX verification ENABLED — VLM: %s, AE: %s", vlm_onnx, ae_onnx)
    elif args.vlm_onnx_path or args.ae_onnx_path:
        LOGGER.warning("Both --vlm-onnx-path and --ae-onnx-path must be provided for ONNX verification. Skipping.")

    # Load models — when ONNX verification is active we only need the full
    # model (PyTorch) as the baseline; the split PyTorch models are skipped.
    full_policy = load_full_model(policy_path, device, local_files_only=args.local_files_only)
    if not run_onnx:
        vlm_policy = load_vlm_model(policy_path, device, local_files_only=args.local_files_only)
        ae_policy = load_ae_model(policy_path, device, local_files_only=args.local_files_only)

    # Parse key-map
    key_map: dict[str, str] = {}
    if args.key_map:
        for entry in args.key_map:
            if "=" not in entry:
                LOGGER.error("Invalid --key-map entry (expected SRC=DST): %s", entry)
                return 1
            src, dst = entry.split("=", 1)
            key_map[src] = dst

    # Prepare batches
    if args.batch_path is not None:
        raw_batches = load_real_batches_raw(args.batch_path)
        if key_map:
            raw_batches = remap_batch_keys(raw_batches, key_map)
        if args.max_batches is not None:
            raw_batches = raw_batches[: args.max_batches]
        # Run the full preprocessing pipeline (same as loss_compare.py):
        # numpy → tensor, image normalisation, state normalisation,
        # prompt generation, tokenisation → observation.language.tokens
        batches = preprocess_real_batches(
            raw_batches,
            policy_path,
            full_policy,
            device,
            task=args.task,
        )
    else:
        batch = make_dummy_batch(full_policy.config, device, seed=args.seed, batch_size=args.batch_size)
        batches = [batch]

    all_action_max_diffs = []
    all_action_cos_means = []
    all_onnx_max_diffs: list[float] = []
    all_onnx_cos_means: list[float] = []

    for i, batch in enumerate(batches):
        LOGGER.info("=" * 60)
        LOGGER.info("Batch %d / %d", i + 1, len(batches))
        LOGGER.info("=" * 60)

        if run_onnx:
            # --- ONNX equivalence check (ONNX split vs full PyTorch) ---
            (
                full_actions_onnx,
                onnx_actions,
                full_kv_onnx,
                onnx_kv,
                full_masks_onnx,
                onnx_masks,
            ) = verify_onnx_actions(
                full_policy,
                batch,
                args.vlm_onnx_path,
                args.ae_onnx_path,
                seed=args.seed + i,
                timesteps=timesteps,
            )

            # KV cache: full PyTorch VLM vs VLM ONNX (isolates PT→ONNX
            # export error from the AE denoising path).
            if not args.skip_kv_check:
                report("KV cache (full PT vs VLM ONNX)", full_kv_onnx, onnx_kv)
                report_kv_per_layer("KV cache (full PT vs VLM ONNX)", full_kv_onnx, onnx_kv)
                mask_match = np.array_equal(full_masks_onnx, onnx_masks)
                LOGGER.info("[prefix_pad_masks] full PT vs VLM ONNX exact match: %s", mask_match)
                if not mask_match:
                    diff_count = int((full_masks_onnx != onnx_masks).sum())
                    LOGGER.warning("[prefix_pad_masks] %d element(s) differ!", diff_count)

            report("Actions (ONNX split)", full_actions_onnx, onnx_actions)

            max_d_onnx, _ = abs_diff_metrics(full_actions_onnx, onnx_actions)
            _, _, cos_mean_onnx = cosine_similarity_stats(full_actions_onnx, onnx_actions)
            all_onnx_max_diffs.append(max_d_onnx)
            all_onnx_cos_means.append(cos_mean_onnx)
        else:
            # --- KV cache check ---
            if not args.skip_kv_check:
                full_kv, vlm_kv, full_masks, vlm_masks = verify_kv_cache(full_policy, vlm_policy, batch)
                report("KV cache", full_kv, vlm_kv)

                mask_match = np.array_equal(full_masks, vlm_masks)
                LOGGER.info("[prefix_pad_masks] exact match: %s", mask_match)
                if not mask_match:
                    diff_count = int((full_masks != vlm_masks).sum())
                    LOGGER.warning("[prefix_pad_masks] %d element(s) differ!", diff_count)

            # --- Action equivalence check (PyTorch split vs full) ---
            full_actions, split_actions = verify_actions(
                full_policy,
                vlm_policy,
                ae_policy,
                batch,
                seed=args.seed + i,
                timesteps=timesteps,
            )
            report("Actions (PyTorch split)", full_actions, split_actions)

            max_d, _ = abs_diff_metrics(full_actions, split_actions)
            _, _, cos_mean = cosine_similarity_stats(full_actions, split_actions)
            all_action_max_diffs.append(max_d)
            all_action_cos_means.append(cos_mean)

    # --- Summary ---
    LOGGER.info("=" * 60)
    LOGGER.info("SUMMARY over %d batch(es)", len(batches))
    LOGGER.info("=" * 60)

    if all_action_cos_means:
        LOGGER.info("--- PyTorch split (VLM + AE) vs full model ---")
        LOGGER.info(
            "Action max abs diff  — min: %.6g, max: %.6g, mean: %.6g",
            min(all_action_max_diffs),
            max(all_action_max_diffs),
            sum(all_action_max_diffs) / len(all_action_max_diffs),
        )
        LOGGER.info(
            "Action cosine mean   — min: %.6f, max: %.6f, mean: %.6f",
            min(all_action_cos_means),
            max(all_action_cos_means),
            sum(all_action_cos_means) / len(all_action_cos_means),
        )

    if all_onnx_cos_means:
        LOGGER.info("--- ONNX split (VLM ONNX + AE ONNX) vs full model ---")
        LOGGER.info(
            "ONNX max abs diff    — min: %.6g, max: %.6g, mean: %.6g",
            min(all_onnx_max_diffs),
            max(all_onnx_max_diffs),
            sum(all_onnx_max_diffs) / len(all_onnx_max_diffs),
        )
        LOGGER.info(
            "ONNX cosine mean     — min: %.6f, max: %.6f, mean: %.6f",
            min(all_onnx_cos_means),
            max(all_onnx_cos_means),
            sum(all_onnx_cos_means) / len(all_onnx_cos_means),
        )

    # Pass/fail heuristic — use worst cosine across all checks
    all_cos = all_action_cos_means + all_onnx_cos_means
    worst_cos = min(all_cos)
    if all_onnx_cos_means and (not all_action_cos_means or min(all_onnx_cos_means) <= worst_cos):
        check_label = "ONNX split"
    else:
        check_label = "PyTorch split"

    if worst_cos >= 0.9999:
        LOGGER.info("✅ PASS — models are numerically equivalent (cosine ≥ 0.9999)")
        return 0
    elif worst_cos >= 0.999:
        LOGGER.warning("⚠️  MARGINAL — cosine ≥ 0.999 but < 0.9999 (%s), likely dtype differences", check_label)
        return 0
    else:
        LOGGER.error("❌ FAIL — cosine < 0.999 (%s), models are NOT equivalent", check_label)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
