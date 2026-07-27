# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Licensed under the Mulan PSL v2.
"""Dump ORT-CPU VLM outputs for a single batch.

Companion to :mod:`dump_vlm_pt` and the explicit manifest-driven
``pi05-om-dump`` command. This script runs the **same VLM ONNX file** that ATC
compiles into the OM, but on the CPU through onnxruntime.

By placing ORT between PT and OM in our diagnostic chain we can
attribute the ``cosine ≈ 0.9991`` gap to its actual source:

  * ``PT vs ORT-CPU``    → ONNX export error (torch → ONNX op mapping)
  * ``ORT-CPU vs OM-NPU``→ ATC compile + NPU fp16 implementation error

The PT preprocessing path (and hence the input tensors) is identical to
``dump_vlm_pt.py`` so the three dumps are directly comparable.

Usage::

    python -m model_utils.pi05_export.dump_vlm_ort \\
        --policy-path /path/to/pi05_ckpt \\
        --batch-path  /path/to/batches_480640.json \\
        --batch-index 0 \\
        --onnx-path   /path/to/pi05-vlm.onnx \\
        --out-dir     /tmp/ort_vlm_dump_0

Compare with::

    python -m model_utils.pi05_export.dump_vlm_pt \\
        --compare-pt  /tmp/pt_vlm_dump_0 \\
        --compare-ort /tmp/ort_vlm_dump_0 \\
        --compare-om  /tmp/om_vlm_dump_0
"""

from __future__ import annotations

import faulthandler

faulthandler.enable()

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

LOGGER = logging.getLogger("dump_vlm_ort")


# Mirror dump_vlm_pt's defaults so the same batches.json can be fed
# straight into either script without extra flags.
_DEFAULT_KEY_MAP: dict[str, str] = {
    "observation.images.hand_view": "observation.images.wrist",
    "observation.images.top_view": "observation.images.top",
}
_DEFAULT_DROP_KEYS: set[str] = {
    "observation.images.side_view",
    "observation.images.side_view_right",
}


def dump(
    *,
    policy_path: str,
    batch_path: str,
    batch_index: int,
    onnx_path: str,
    out_dir: str,
    device_str: str,
    key_map: dict[str, str] | None = None,
    task: str = "",
) -> None:
    """Preprocess one batch with PT, then run the VLM ONNX on CPU."""
    import onnxruntime as ort

    # Lazy imports — heavy deps.
    from model_utils.pi05_export.prefix_mask_utils import (
        build_prefix_att_2d_masks_4d_np,
    )
    from model_utils.pi05_export.verify_pi05_split_equivalence import (
        _build_vlm_onnx_feed,
        load_real_batches_raw,
        preprocess_real_batches,
        remap_batch_keys,
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load policy ONLY for preprocessing (image normalization, tokenization,
    # etc.).  We deliberately don't run select_action — the VLM forward
    # happens in onnxruntime instead.  Loading on CPU keeps GPU memory free.
    # ------------------------------------------------------------------
    from model_utils.pi05_export.modeling_pi05_vlm import PI05VLMPolicy

    device = torch.device(device_str)
    LOGGER.info("Loading PI05VLMPolicy from %s on %s (preprocess only) …", policy_path, device)
    policy = PI05VLMPolicy.from_pretrained(policy_path, local_files_only=False, strict=False)
    policy.to(device)
    policy.eval()

    # ------------------------------------------------------------------
    # Same batch loading / key remap as dump_vlm_pt.
    # ------------------------------------------------------------------
    LOGGER.info("Loading batches from %s …", batch_path)
    raw_batches = load_real_batches_raw(batch_path)
    if not (0 <= batch_index < len(raw_batches)):
        raise IndexError(f"batch_index {batch_index} out of range (have {len(raw_batches)} batches)")

    raw_batch = {k: v for k, v in raw_batches[batch_index].items() if k not in _DEFAULT_DROP_KEYS}
    effective_map = {**_DEFAULT_KEY_MAP, **(key_map or {})}
    if effective_map:
        LOGGER.info("Applying key remap: %s", effective_map)
        raw_batch = remap_batch_keys([raw_batch], effective_map)[0]

    LOGGER.info("Running preprocessor on batch[%d] (keys=%s) …", batch_index, sorted(raw_batch.keys()))
    preprocessed = preprocess_real_batches(
        [raw_batch],
        policy_path=policy_path,
        full_policy=policy,
        device=device,
        task=task,
    )
    batch = preprocessed[0]

    # ------------------------------------------------------------------
    # Open ONNX session on CPU.
    # ------------------------------------------------------------------
    LOGGER.info("Loading VLM ONNX %s on CPUExecutionProvider …", onnx_path)
    sess_opts = ort.SessionOptions()
    # Disable graph optimisations so we measure the *exported* graph as
    # faithfully as possible (the OM is also compiled from this exact graph
    # via ATC, not from an ORT-optimised version).

    # 修复 ORT 运行 segfault
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    # sess_opts.enable_cpu_mem_arena = False
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    vlm_session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    valid_names = {inp.name for inp in vlm_session.get_inputs()}
    LOGGER.info("  VLM ONNX inputs: %s", sorted([(i.name, i.shape, i.type) for i in vlm_session.get_inputs()]))

    # ------------------------------------------------------------------
    # Build feed (Plan A: prefix_att_2d_masks_4d goes in as host fp32).
    # ------------------------------------------------------------------
    vlm_feed = _build_vlm_onnx_feed(batch, valid_names)

    if "prefix_att_2d_masks_4d" in valid_names and "prefix_att_2d_masks_4d" not in vlm_feed:
        prefix_meta = next(inp for inp in vlm_session.get_inputs() if inp.name == "prefix_att_2d_masks_4d")
        # prefix_meta.shape is e.g. [1, 1, 712, 712]
        prefix_seq_len = int(prefix_meta.shape[-1])
        lang_masks_np = batch["observation.language.attention_mask"].cpu().numpy().astype(bool)
        num_cameras = sum(1 for inp in vlm_session.get_inputs() if inp.name.startswith("observation.images."))
        LOGGER.info("  Building prefix_att_2d_masks_4d: num_cameras=%d, prefix_seq_len=%d", num_cameras, prefix_seq_len)
        prefix_mask = build_prefix_att_2d_masks_4d_np(
            num_cameras=num_cameras,
            lang_masks=lang_masks_np,
            prefix_seq_len=prefix_seq_len,
        )
        vlm_feed["prefix_att_2d_masks_4d"] = prefix_mask.astype(np.float32)

    LOGGER.info("  ORT feed keys: %s", sorted(vlm_feed.keys()))

    # ------------------------------------------------------------------
    # Dump VLM inputs (file names aligned with the PT and OM diagnostics
    # so the three sources can be diff'd directly).
    # ------------------------------------------------------------------
    try:
        img_idx = 0
        for inp in vlm_session.get_inputs():
            if inp.name not in vlm_feed:
                LOGGER.warning("VLM input dump: feed missing key %r", inp.name)
                continue
            arr = np.asarray(vlm_feed[inp.name])
            if inp.name.startswith("observation.images."):
                np.save(out_path / f"vlm_in_image_{img_idx}.npy", arr)
                img_idx += 1
            elif inp.name == "lang_tokens":
                np.save(out_path / "vlm_in_lang_tokens.npy", arr)
            elif inp.name == "lang_masks":
                np.save(out_path / "vlm_in_lang_masks.npy", arr)
            elif inp.name == "prefix_att_2d_masks_4d":
                np.save(out_path / "vlm_in_prefix_mask_4d.npy", arr)
            else:
                np.save(out_path / f"vlm_in_{inp.name}.npy", arr)
        LOGGER.info("Dumped VLM inputs under %s", out_path)
    except Exception as exc:
        LOGGER.warning("VLM input dump failed: %s", exc)

    # ------------------------------------------------------------------
    # Run.
    # ------------------------------------------------------------------
    LOGGER.info("Running VLM forward on CPU …")
    vlm_outputs = vlm_session.run(None, vlm_feed)

    # The ONNX wrapper exports past_kv_tensor first, prefix_pad_masks second.
    past_kv_np = np.asarray(vlm_outputs[0]).astype(np.float16)
    prefix_pad_masks_np = np.asarray(vlm_outputs[1]).astype(bool)

    past_kv_path = out_path / "past_kv_tensor.npy"
    pad_masks_path = out_path / "prefix_pad_masks.npy"
    np.save(past_kv_path, past_kv_np)
    np.save(pad_masks_path, prefix_pad_masks_np)

    LOGGER.info("Saved %s  shape=%s dtype=%s", past_kv_path, past_kv_np.shape, past_kv_np.dtype)
    LOGGER.info("Saved %s  shape=%s dtype=%s", pad_masks_path, prefix_pad_masks_np.shape, prefix_pad_masks_np.dtype)

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
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--policy-path", type=str, required=True, help="Path to pretrained PI05 checkpoint (used for preprocessing)."
    )
    p.add_argument("--batch-path", type=str, required=True, help="Path to batches.json.")
    p.add_argument("--batch-index", type=int, default=0, help="Which batch to dump (default: 0).")
    p.add_argument("--onnx-path", type=str, required=True, help="Path to the exported VLM ONNX file.")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for .npy files.")
    p.add_argument("--device", type=str, default="cuda", help="Torch device for preprocessing only (default: cuda).")
    p.add_argument(
        "--key-map",
        type=str,
        nargs="*",
        default=None,
        metavar="SRC=DST",
        help="Extra batch-key remappings (default: hand_view→wrist, top_view→top).",
    )
    p.add_argument("--task", type=str, default="", help="Task description for prompt generation (default: empty).")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

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
        onnx_path=args.onnx_path,
        out_dir=args.out_dir,
        device_str=args.device,
        key_map=key_map,
        task=args.task,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
