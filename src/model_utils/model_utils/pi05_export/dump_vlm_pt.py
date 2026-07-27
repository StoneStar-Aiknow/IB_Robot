# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Licensed under the Mulan PSL v2.
"""Dump PyTorch VLM outputs for a single batch.

Pairs with the explicit manifest-driven ``pi05-om-dump`` diagnostic command.

Given the SAME ``batches.json`` file used by ``loss_compare.py`` and a
trained PI05 checkpoint, this script:

  1. Loads batch index ``--batch-index`` (default 0)
  2. Runs the full PT preprocessing + VLM forward
  3. Saves ``past_kv_tensor.npy`` and ``prefix_pad_masks.npy`` to
     ``--out-dir`` — file names match the OM dumper exactly so the two
     directories are directly comparable.

Usage (on a GPU/CPU machine, no ``acl`` required):

    python -m model_utils.pi05_export.dump_vlm_pt \\
        --policy-path /path/to/pi05_ckpt \\
        --batch-path  /path/to/batches_480640.json \\
        --batch-index 0 \\
        --out-dir     /tmp/pt_vlm_dump_0 \\
        --device      cuda

Then run ``pi05-om-dump`` on the NPU machine with the same batch and noise
seed. The command uses the selected unified Ascend deployment and never
enables dumping through production-runtime environment variables.

Compare with::

    python -m model_utils.pi05_export.dump_vlm_pt \\
        --compare-pt /tmp/pt_vlm_dump_0 \\
        --compare-om /tmp/om_vlm_dump_0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

LOGGER = logging.getLogger("dump_vlm_pt")


# ---------------------------------------------------------------------------
# Dump
# ---------------------------------------------------------------------------

# Default key remapping — matches the hard-coded renames in
# ``loss_compare.py:load_batches_as_tensors`` so the same ``batches.json``
# can be fed straight into this script.
_DEFAULT_KEY_MAP: dict[str, str] = {
    "observation.images.hand_view": "observation.images.wrist",
    "observation.images.top_view": "observation.images.top",
}
# Keys to silently drop (loss_compare also skips them).
_DEFAULT_DROP_KEYS: set[str] = {
    "observation.images.side_view",
    "observation.images.side_view_right",
}


def dump(
    *,
    policy_path: str,
    batch_path: str,
    batch_index: int,
    out_dir: str,
    device_str: str,
    key_map: dict[str, str] | None = None,
    task: str = "",
    model_dtype: str = "native",
) -> None:
    """Run PT VLM on one batch and save outputs as .npy."""
    # Lazy imports — these pull in heavy deps (transformers, etc.)
    from lerobot.configs.policies import PreTrainedConfig

    from model_utils.pi05_export.modeling_pi05_vlm import PI05VLMPolicy
    from model_utils.pi05_export.verify_pi05_split_equivalence import (
        load_real_batches_raw,
        preprocess_real_batches,
        remap_batch_keys,
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str)
    LOGGER.info("Loading PI05VLMPolicy from %s on %s …", policy_path, device)
    config = PreTrainedConfig.from_pretrained(pretrained_name_or_path=policy_path, local_files_only=False)
    if hasattr(config, "device"):
        config.device = device_str
    policy = PI05VLMPolicy.from_pretrained(policy_path, config=config, local_files_only=False, strict=False)
    policy.to(device)

    # Optional dtype cast — defaults to whatever the checkpoint provides
    # (BF16 for PI05).  Use ``--model-dtype fp16`` to match the OM/ORT
    # deployment dtype and isolate BF16↔FP16 conversion error from any
    # real ONNX-export error.
    if model_dtype == "fp16":
        policy.model = policy.model.half()
        LOGGER.info("  Cast policy.model to float16")
    elif model_dtype == "bf16":
        policy.model = policy.model.bfloat16()
        LOGGER.info("  Cast policy.model to bfloat16")
    elif model_dtype == "fp32":
        policy.model = policy.model.float()
        LOGGER.info("  Cast policy.model to float32")
    elif model_dtype != "native":
        raise ValueError(f"unknown --model-dtype: {model_dtype}")

    policy.eval()

    # Log actual running dtype for clarity.
    sample_param = next(policy.model.parameters())
    LOGGER.info("  Running PT VLM in dtype=%s", sample_param.dtype)

    LOGGER.info("Loading batches from %s …", batch_path)
    raw_batches = load_real_batches_raw(batch_path)
    if not (0 <= batch_index < len(raw_batches)):
        raise IndexError(f"batch_index {batch_index} out of range (have {len(raw_batches)} batches)")

    # Drop keys that loss_compare also drops (e.g. side_view).
    raw_batch = {k: v for k, v in raw_batches[batch_index].items() if k not in _DEFAULT_DROP_KEYS}

    # Apply default + user-supplied key remapping (rename to match the
    # checkpoint's expected ``config.image_features`` keys).
    effective_map = {**_DEFAULT_KEY_MAP, **(key_map or {})}
    if effective_map:
        LOGGER.info("Applying key remap: %s", effective_map)
        raw_batch = remap_batch_keys([raw_batch], effective_map)[0]

    LOGGER.info("Running preprocessor on batch[%d] (keys=%s) …", batch_index, sorted(raw_batch.keys()))
    # Preprocess only the requested batch to keep memory low.
    preprocessed = preprocess_real_batches(
        [raw_batch],
        policy_path=policy_path,
        full_policy=policy,
        device=device,
        task=task,
    )
    batch = preprocessed[0]

    # Mirror loss_compare's deterministic seeding so any RNG inside the
    # model (there shouldn't be any in VLM, but harmless) matches.
    torch.manual_seed(42 + batch_index)

    # ------------------------------------------------------------------
    # Dump VLM inputs using names shared by the PT, ORT, and OM diagnostics.
    # so the three sources can be diff'd directly.  Done BEFORE
    # select_action so we capture the exact tensors that get fed to the
    # OM/ONNX graph (resize+normalize happens inside the graph).
    # ------------------------------------------------------------------
    try:
        from model_utils.pi05_export.prefix_mask_utils import (
            build_prefix_att_2d_masks_4d_np,
        )

        img_keys = list(policy.config.image_features)
        for i, k in enumerate(img_keys):
            if k not in batch:
                LOGGER.warning("VLM input dump: missing image key %r in batch", k)
                continue
            img_np = batch[k].detach().cpu().numpy().astype(np.float32)
            np.save(out_path / f"vlm_in_image_{i}.npy", img_np)

        tok_np = batch["observation.language.tokens"].detach().cpu().numpy().astype(np.int64)
        msk_np = batch["observation.language.attention_mask"].detach().cpu().numpy().astype(bool)
        np.save(out_path / "vlm_in_lang_tokens.npy", tok_np)
        np.save(out_path / "vlm_in_lang_masks.npy", msk_np)

        # Derive the prefix sequence length from the checkpoint config and the
        # actual tokenized language mask instead of hardcoding constants, so a
        # checkpoint with a different image resolution / patch size or language
        # length still dumps a mask that matches what the model/OM consume.
        # image tokens per camera = (image_size / patch_size) ** 2 (SigLIP).
        vision_cfg = getattr(policy.config, "vision_config", None) or getattr(
            getattr(policy, "config", None), "vlm_config", None
        )
        image_tokens_per_camera = 256
        try:
            image_size = int(vision_cfg.image_size)
            patch_size = int(vision_cfg.patch_size)
            image_tokens_per_camera = (image_size // patch_size) ** 2
        except (AttributeError, TypeError, ValueError):
            LOGGER.warning(
                "Could not derive image tokens from vision config; falling back to %d per camera",
                image_tokens_per_camera,
            )
        lang_seq_len = int(msk_np.shape[1])
        prefix_seq_len = image_tokens_per_camera * len(img_keys) + lang_seq_len
        prefix_mask = build_prefix_att_2d_masks_4d_np(
            num_cameras=len(img_keys),
            lang_masks=msk_np,
            prefix_seq_len=prefix_seq_len,
        ).astype(np.float32)
        np.save(out_path / "vlm_in_prefix_mask_4d.npy", prefix_mask)
        LOGGER.info(
            "Dumped VLM inputs (%d image(s) + lang_tokens + lang_masks + prefix_mask_4d) under %s",
            len(img_keys),
            out_path,
        )
    except Exception as exc:
        LOGGER.warning("VLM input dump failed: %s", exc)

    LOGGER.info("Forwarding through VLM (select_action) …")
    with torch.no_grad():
        past_kv_tensor, prefix_pad_masks = policy.select_action(batch)

    # Match OM dumper dtypes:
    #   past_kv_tensor : float16  (OM stores fp16 KV cache)
    #   prefix_pad_masks: bool
    past_kv_np = past_kv_tensor.detach().cpu().to(torch.float16).numpy()
    prefix_pad_masks_np = prefix_pad_masks.detach().cpu().to(torch.bool).numpy()

    past_kv_path = out_path / "past_kv_tensor.npy"
    pad_masks_path = out_path / "prefix_pad_masks.npy"
    np.save(past_kv_path, past_kv_np)
    np.save(pad_masks_path, prefix_pad_masks_np)

    LOGGER.info("Saved %s  shape=%s dtype=%s", past_kv_path, past_kv_np.shape, past_kv_np.dtype)
    LOGGER.info("Saved %s  shape=%s dtype=%s", pad_masks_path, prefix_pad_masks_np.shape, prefix_pad_masks_np.dtype)

    # Quick stats so you can sanity-check from the log alone.
    a32 = past_kv_np.astype(np.float32)
    LOGGER.info(
        "  past_kv stats: min=%+.4g max=%+.4g mean=%+.4g std=%+.4g finite=%s",
        float(a32.min()),
        float(a32.max()),
        float(a32.mean()),
        float(a32.std()),
        bool(np.isfinite(a32).all()),
    )
    LOGGER.info(
        "  prefix_pad_masks: True=%d False=%d total=%d",
        int(prefix_pad_masks_np.sum()),
        int((~prefix_pad_masks_np).sum()),
        int(prefix_pad_masks_np.size),
    )


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _compare_pair(label_a: str, arr_a: np.ndarray, label_b: str, arr_b: np.ndarray) -> None:
    """Print element-wise stats + cosine for a single pair of arrays."""
    print(f"\n  --- {label_a}  vs  {label_b} ---")
    print(f"    {label_a}: shape={arr_a.shape} dtype={arr_a.dtype}")
    print(f"    {label_b}: shape={arr_b.shape} dtype={arr_b.dtype}")

    if arr_a.shape != arr_b.shape:
        if arr_a.size == arr_b.size:
            print(f"    ⚠️  shape differs but element count matches ({arr_a.size}) — comparing as flat vectors")
            arr_a = arr_a.reshape(-1)
            arr_b = arr_b.reshape(-1)
        else:
            print("    ❌ SHAPE MISMATCH — cannot compare element-wise")
            return

    if arr_a.dtype == np.bool_ and arr_b.dtype == np.bool_:
        n = arr_a.size
        mismatch = int(np.sum(arr_a != arr_b))
        print(f"    bool mismatch: {mismatch}/{n} ({100.0 * mismatch / n:.4f}%)")
        return

    a32 = arr_a.astype(np.float32)
    b32 = arr_b.astype(np.float32)
    diff = a32 - b32
    abs_diff = np.abs(diff)

    print(f"    L1   = {float(abs_diff.mean()):.6e}")
    print(f"    Linf = {float(abs_diff.max()):.6e}")
    print(f"    RMSE = {float(np.sqrt((diff**2).mean())):.6e}")
    print(f"    cosine_sim = {_cosine(a32, b32):.6f}")
    print(
        f"    {label_a}   stats: min={float(a32.min()):+.4g} max={float(a32.max()):+.4g} "
        f"mean={float(a32.mean()):+.4g} std={float(a32.std()):+.4g}"
    )
    print(
        f"    {label_b}   stats: min={float(b32.min()):+.4g} max={float(b32.max()):+.4g} "
        f"mean={float(b32.mean()):+.4g} std={float(b32.std()):+.4g}"
    )
    print(f"    diff stats: min={float(diff.min()):+.4g} max={float(diff.max()):+.4g} mean={float(diff.mean()):+.4g}")


def compare(
    *,
    pt_dir: str | None = None,
    ort_dir: str | None = None,
    om_dir: str | None = None,
) -> None:
    """Load up to three dumps and report all pairwise stats.

    Use any combination of ``pt_dir``, ``ort_dir``, ``om_dir``.  If two
    are given, prints one comparison; if three, prints all three pairs
    (PT-vs-ORT, PT-vs-OM, ORT-vs-OM) so you can attribute the gap.
    """
    sources: list[tuple[str, Path]] = []
    if pt_dir:
        sources.append(("PT", Path(pt_dir)))
    if ort_dir:
        sources.append(("ORT", Path(ort_dir)))
    if om_dir:
        sources.append(("OM", Path(om_dir)))

    if len(sources) < 2:
        raise ValueError("compare() needs at least two of pt_dir / ort_dir / om_dir")

    files = ["past_kv_tensor.npy", "prefix_pad_masks.npy"]
    header = "  ".join(f"{lbl}={path}" for lbl, path in sources)
    print(f"\nComparing  {header}\n" + "-" * 70)

    # Auto-discover any additional .npy files present in ALL sources
    # (e.g. vlm_in_*.npy added by the input-dump enhancement).
    common_extras = None
    for _lbl, path in sources:
        names = {p.name for p in Path(path).glob("*.npy")}
        common_extras = names if common_extras is None else (common_extras & names)
    if common_extras:
        # Keep the canonical ordering for the well-known files, then
        # append the rest sorted for stable output.
        extras = sorted(n for n in common_extras if n not in files)
        files = files + extras

    for fname in files:
        print(f"\n[{fname}]")
        # Load every available source for this file.
        loaded: dict[str, np.ndarray] = {}
        for lbl, path in sources:
            fpath = path / fname
            if not fpath.exists():
                print(f"  [SKIP {lbl}] missing {fpath}")
                continue
            loaded[lbl] = np.load(fpath)

        if len(loaded) < 2:
            print(f"  ❌ need ≥2 sources for this file, got {list(loaded.keys())}")
            continue

        # Pairwise: emit all combinations in stable order PT→ORT→OM.
        labels = [lbl for lbl, _ in sources if lbl in loaded]
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                la, lb = labels[i], labels[j]
                _compare_pair(la, loaded[la], lb, loaded[lb])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Mode 1: dump
    p.add_argument("--policy-path", type=str, default=None, help="Path to pretrained PI05 checkpoint (dump mode).")
    p.add_argument("--batch-path", type=str, default=None, help="Path to batches.json (dump mode).")
    p.add_argument("--batch-index", type=int, default=0, help="Which batch to dump (default: 0).")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory for .npy files (dump mode).")
    p.add_argument("--device", type=str, default="cuda", help="Torch device for PT inference (default: cuda).")
    p.add_argument(
        "--key-map",
        type=str,
        nargs="*",
        default=None,
        metavar="SRC=DST",
        help="Extra batch-key remappings on top of the built-in "
        "hand_view→wrist / top_view→top defaults. Format: SRC=DST.",
    )
    p.add_argument("--task", type=str, default="", help="Task description for prompt generation (default: empty).")
    p.add_argument(
        "--model-dtype",
        type=str,
        default="native",
        choices=["native", "fp16", "bf16", "fp32"],
        help="Cast PT model to this dtype before forward. "
        "'native' (default) keeps the checkpoint dtype "
        "(BF16 for PI05). Use 'fp16' for apples-to-apples "
        "comparison with OM/ORT.",
    )

    # Mode 2: compare (any 2-or-3 of pt/ort/om)
    p.add_argument("--compare-pt", type=str, default=None, help="PT dump directory (compare mode).")
    p.add_argument("--compare-ort", type=str, default=None, help="ORT-CPU dump directory (compare mode, optional).")
    p.add_argument("--compare-om", type=str, default=None, help="OM dump directory (compare mode).")

    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    # Compare mode: any 2 or 3 of pt/ort/om dirs.
    compare_dirs = [args.compare_pt, args.compare_ort, args.compare_om]
    if sum(d is not None for d in compare_dirs) >= 2:
        compare(
            pt_dir=args.compare_pt,
            ort_dir=args.compare_ort,
            om_dir=args.compare_om,
        )
        return 0

    missing = [name for name in ("policy_path", "batch_path", "out_dir") if getattr(args, name) is None]
    if missing:
        print(f"error: missing required args for dump mode: {missing}", file=sys.stderr)
        print(
            "       (or pass any two of --compare-pt / --compare-ort / --compare-om for compare mode)", file=sys.stderr
        )
        return 2

    # Parse --key-map entries (SRC=DST)
    key_map: dict[str, str] = {}
    if args.key_map:
        for entry in args.key_map:
            if "=" not in entry:
                print(f"error: invalid --key-map entry (expected SRC=DST): {entry}", file=sys.stderr)
                return 2
            src, dst = entry.split("=", 1)
            key_map[src] = dst

    dump(
        policy_path=args.policy_path,
        batch_path=args.batch_path,
        batch_index=args.batch_index,
        out_dir=args.out_dir,
        device_str=args.device,
        key_map=key_map,
        task=args.task,
        model_dtype=args.model_dtype,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
