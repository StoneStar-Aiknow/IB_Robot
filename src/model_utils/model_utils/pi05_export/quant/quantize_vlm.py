#!/usr/bin/env python
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Licensed under the Mulan PSL v2.
"""msModelSlim W8A8 PTQ for the PI05 **VLM** (gemma_2b) ONNX.

Thin entry point: this file only owns what is specific to the VLM —

* the real-batch calibration-data builder (image preprocessing + tokenization
  + host-built 4D prefix attention mask), and
* the default fp16-exclusion regexes (SigLIP vision tower + attention BMMs).

Everything else (msModelSlim runtime patches, the W8A8 driver, the AscendDequant
fp16 pin, and the Route-A int8 transplant) is imported unchanged from
:mod:`model_utils.pi05_export.quant.w8a8_common`.

WARNING: a real ``--batch-path`` JSON is mandatory — calibrating on random data
would make the quantized model garbage.

Examples
--------
List the quantizable nodes (decide what to keep in fp16)::

    python -m model_utils.pi05_export.quant.quantize_vlm \\
        --onnx-path /path/to/pi05-vlm.onnx --list-nodes

Quantize (ORT graph), or Route A (graft int8 onto the NPU-op graph)::

    python -m model_utils.pi05_export.quant.quantize_vlm \\
        --onnx-path   /path/to/pi05-vlm_ort.onnx \\
        --npu-onnx-path /path/to/pi05-vlm_npu.onnx \\
        --output-path /path/to/pi05-vlm_w8a8.onnx \\
        --policy-path /path/to/pi05_ckpt \\
        --batch-path  /path/to/batches_480640.json \\
        --num-calib   16 \\
        --task 'pick up the cup'
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from model_utils.pi05_export.quant import w8a8_common as common

LOGGER = logging.getLogger("quantize_vlm")

# Same defaults as dump_vlm_ort.py so a single batches.json feeds every tool.
_DEFAULT_KEY_MAP: dict[str, str] = {
    "observation.images.hand_view": "observation.images.wrist",
    "observation.images.top_view": "observation.images.top",
}
_DEFAULT_DROP_KEYS: set[str] = {
    "observation.images.side_view",
    "observation.images.side_view_right",
}


def _resize_calibration_images(feed: dict[str, np.ndarray], onnx_path: Path) -> None:
    """Match preprocessed camera tensors to static VLM ONNX image shapes."""
    import onnx

    from inference_service.pi05_image_preprocess import resize_with_pad_nchw_numpy

    model = onnx.load(str(onnx_path), load_external_data=False)
    for model_input in model.graph.input:
        name = model_input.name
        if not name.startswith("observation.images.") or name not in feed:
            continue
        dims = model_input.type.tensor_type.shape.dim
        if len(dims) != 4 or not dims[-2].HasField("dim_value") or not dims[-1].HasField("dim_value"):
            raise ValueError(f"VLM ONNX image input {name!r} must have static NCHW spatial dimensions")
        height, width = int(dims[-2].dim_value), int(dims[-1].dim_value)
        image = np.asarray(feed[name], dtype=np.float32)
        if image.shape[-2:] != (height, width):
            feed[name] = resize_with_pad_nchw_numpy(image, height, width)


# Regexes (case-insensitive, matched against ONNX node *name*) whose nodes are
# kept in fp16 by default. Tune for your exported graph via --list-nodes first.
_DEFAULT_DISABLE_REGEXES: tuple[str, ...] = (
    # SigLIP / vision tower — small FLOPs, wide dynamic range, feeds all tokens.
    r"vision",
    r"siglip",
    r"patch_embed",
    r"multi_modal_projector",
    # Non-weight matmuls in the Gemma trunk — standard W8A8 leaves these in fp16:
    #  * rotary_emb: positional encoding, constant multiply, no quant benefit.
    #  * self_attn score BMMs (Q@K^T and attn@V): activation×activation, no
    #    weight to int8, large dynamic range → keep fp16. The trailing-segment
    #    anchors avoid hitting q_proj/k_proj/v_proj/o_proj (those stay int8).
    r"rotary_emb",
    r"self_attn/MatMul(_\d+)?$",
    # NPU export fuses gate_proj + up_proj into one [up;gate] MatMul feeding
    # NPUGeglu, so Route A cannot transplant their separate donor int8 nodes.
    r"mlp/(gate_proj|up_proj)/MatMul",
)


# ---------------------------------------------------------------------------
# Calibration data (reuse the verifier's real-batch preprocessing path)
# ---------------------------------------------------------------------------


def build_calib_data(
    *,
    onnx_path: Path,
    policy_path: str,
    batch_path: str,
    num_calib: int,
    key_map: dict[str, str],
    task: str,
    device_str: str,
) -> list[list[np.ndarray]]:
    """Produce ``calib_data`` as a list of per-batch ordered input lists.

    msModelSlim's ``OnnxCalibrator`` expects each calibration sample to be a
    list of numpy arrays ordered to match the model's graph inputs.
    """
    import torch
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK

    from model_utils.pi05_export.prefix_mask_utils import build_prefix_att_2d_masks_4d_np
    from model_utils.pi05_export.verify_pi05_split_equivalence import (
        _build_vlm_onnx_feed,
        load_real_batches_raw,
        preprocess_real_batches,
        remap_batch_keys,
    )

    input_order = common.ordered_input_names(onnx_path)
    valid_names = set(input_order)
    LOGGER.info("VLM ONNX inputs (in order): %s", input_order)

    # The host-built 4D attention mask is NOT in the batch — it is derived from
    # the language mask + image-token count, exactly like dump_vlm_ort.py.
    mask_input = "prefix_att_2d_masks_4d"
    need_prefix_mask = mask_input in valid_names
    prefix_seq_len = common.onnx_input_last_dim(onnx_path, mask_input) if need_prefix_mask else 0
    num_cameras = sum(1 for n in input_order if n.startswith("observation.images."))
    if need_prefix_mask:
        LOGGER.info(
            "Will build %s on host: num_cameras=%d, prefix_seq_len=%d",
            mask_input,
            num_cameras,
            prefix_seq_len,
        )

    device = torch.device(device_str)

    # Load the VLM policy ONLY for preprocessing (image norm + tokenization).
    from model_utils.pi05_export.modeling_pi05_vlm import PI05VLMPolicy

    LOGGER.info("Loading PI05VLMPolicy from %s (preprocess only) …", policy_path)
    policy = PI05VLMPolicy.from_pretrained(policy_path, local_files_only=False, strict=False)
    policy.to(device)
    policy.eval()

    raw_batches = load_real_batches_raw(batch_path)
    raw_batches = [{k: v for k, v in b.items() if k not in _DEFAULT_DROP_KEYS} for b in raw_batches]
    raw_batches = remap_batch_keys(raw_batches, {**_DEFAULT_KEY_MAP, **key_map})

    if num_calib > 0:
        raw_batches = raw_batches[:num_calib]
    LOGGER.info("Using %d calibration batch(es)", len(raw_batches))
    if not raw_batches:
        raise ValueError("No calibration batches available — check --batch-path / --num-calib.")

    preprocessed = preprocess_real_batches(raw_batches, policy_path, policy, device, task=task)

    calib_data: list[list[np.ndarray]] = []
    for batch in preprocessed:
        feed = _build_vlm_onnx_feed(batch, valid_names)
        _resize_calibration_images(feed, onnx_path)
        if need_prefix_mask and mask_input not in feed:
            lang_masks_np = batch[OBS_LANGUAGE_ATTENTION_MASK].cpu().numpy().astype(bool)
            feed[mask_input] = build_prefix_att_2d_masks_4d_np(
                num_cameras=num_cameras,
                lang_masks=lang_masks_np,
                prefix_seq_len=prefix_seq_len,
            ).astype(np.float32)
        missing = [n for n in input_order if n not in feed]
        if missing:
            raise KeyError(
                f"Calibration feed missing required ONNX inputs {missing}. Have: {sorted(feed)}. Check --key-map."
            )
        calib_data.append([feed[name] for name in input_order])

    return calib_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="msModelSlim W8A8 PTQ for the PI05 VLM ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    common.add_common_quant_args(p)
    p.add_argument("--policy-path", type=str, default=None, help="PI05 ckpt (for calibration preprocessing).")
    p.add_argument(
        "--batch-path", type=str, default=None, help="Calibration batches JSON (same format as dump_vlm_ort.py)."
    )
    p.add_argument("--task", type=str, default="", help="Task string for prompt building during preprocessing.")
    p.add_argument(
        "--key-map",
        type=str,
        nargs="*",
        default=None,
        help="Extra src=dst batch key remaps, e.g. observation.images.top_view=observation.images.top",
    )
    return p


def _parse_key_map(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--key-map entry must be src=dst, got {item!r}")
        src, dst = item.split("=", 1)
        out[src.strip()] = dst.strip()
    return out


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    onnx_path = Path(args.onnx_path).expanduser().resolve()
    if not onnx_path.is_file():
        LOGGER.error("ONNX not found: %s", onnx_path)
        return 1

    model_proto = common.load_onnx(onnx_path)
    quantizable = common.collect_quantizable_nodes(model_proto)
    LOGGER.info("Found %d quantizable node(s) (%s).", len(quantizable), "/".join(common._QUANTIZABLE_OPS))

    disable_regexes = args.disable_regex if args.disable_regex is not None else list(_DEFAULT_DISABLE_REGEXES)
    disable_names = common.build_disable_names(
        quantizable,
        disable_regexes,
        disable_convs=not args.quantize_convs,
        disable_index_below=args.disable_index_below,
    )

    if args.list_nodes:
        common.list_nodes_and_exit(quantizable, disable_names, disable_regexes)
        return 0

    # Real quantization requires calibration inputs.
    missing_args = [
        flag
        for flag, val in (
            ("--policy-path", args.policy_path),
            ("--batch-path", args.batch_path),
        )
        if not val
    ]
    if missing_args:
        LOGGER.error("Missing required args for quantization: %s", ", ".join(missing_args))
        return 1

    output_path = common.resolve_output_path(args.output_path, onnx_path)

    npu_graph_path = None
    if args.npu_onnx_path:
        npu_graph_path = Path(args.npu_onnx_path).expanduser().resolve()
        if not npu_graph_path.is_file():
            LOGGER.error("--npu-onnx-path not found: %s", npu_graph_path)
            return 1

    calib_data = build_calib_data(
        onnx_path=onnx_path,
        policy_path=args.policy_path,
        batch_path=args.batch_path,
        num_calib=args.num_calib,
        key_map=_parse_key_map(args.key_map),
        task=args.task,
        device_str=args.device,
    )

    common.run_msmodelslim_w8a8(
        input_onnx=onnx_path,
        output_onnx=output_path,
        calib_data=calib_data,
        disable_names=disable_names,
        amp_num=args.amp_num,
        npu_graph=npu_graph_path,
    )
    if not output_path.is_file():
        raise RuntimeError(f"msModelSlim reported success but did not produce {output_path}")
    common.load_onnx(output_path)

    LOGGER.info(
        "Done. Next: ATC-compile %s on the board, then run verify_vlm_cpu_vs_om.py "
        "to measure KV cosine + action-error regression.",
        output_path.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
