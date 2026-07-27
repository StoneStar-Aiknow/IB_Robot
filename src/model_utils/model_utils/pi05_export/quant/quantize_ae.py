#!/usr/bin/env python
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Licensed under the Mulan PSL v2.
"""msModelSlim W8A8 PTQ for the PI05 **Action Expert** (gemma_300m) ONNX.

Thin entry point: this file only owns what is specific to the Action Expert —

* the calibration-data builder (loads the 4 AE inputs
  ``past_kv_tensor / prefix_pad_masks / time / noise`` produced by the VLM
  export + AE dump), and
* the default fp16-exclusion regexes (attention BMMs + the action head).

Everything else (msModelSlim runtime patches, the W8A8 driver, the AscendDequant
fp16 pin, and the Route-A int8 transplant) is imported unchanged from
:mod:`model_utils.pi05_export.quant.w8a8_common`.

Why AE W8A8 is worth it
------------------------
AE profiling: ~69% of cycles are spent in quantizable weight-GEMMs and the
kernels are MTE2 (weight-load) bound, so int8 weights cut memory traffic on the
hottest path. The AE runs ``num_inference_steps`` (=10) denoise steps per
inference, so any per-GEMM speedup is amplified ~10x — making AE quantization at
least as valuable as the VLM. The only real risk is accuracy (it emits actions
directly and errors accumulate across the 10 Euler steps), so consider
``--amp-num`` (now backed by real calib data) and an action-error regression.

Calibration inputs
-------------------
The AE consumes a fixed ``past_kv_tensor`` + ``prefix_pad_masks`` (from a VLM
forward pass) plus ``time`` and ``noise``, and runs the whole 10-step Euler
trajectory inside one ONNX forward — so one input set exercises every step's
activations. Provide calibration samples via ``--calib-dir`` (one subdirectory
per sample, each holding ``past_kv_tensor.npy`` + ``prefix_pad_masks.npy`` and
optionally ``noise.npy``), or a single set via the explicit path flags. These
are exactly the tensors ``dump_ae_pt.py`` / the VLM export already produce.

Examples
--------
::

    python -m model_utils.pi05_export.quant.quantize_ae \\
        --onnx-path   /path/to/pi05-ae_ort.onnx \\
        --npu-onnx-path /path/to/pi05-ae_npu.onnx \\
        --output-path /path/to/pi05-ae_w8a8.onnx \\
        --calib-dir   /path/to/ae_calib_dumps \\
        --num-calib   16 --amp-num 4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from model_utils.pi05_export.quant import w8a8_common as common

LOGGER = logging.getLogger("quantize_ae")

# Regexes (case-insensitive) on ONNX node *name* kept in fp16 by default.
# The AE has NO vision tower, so the VLM's siglip/patch_embed entries are gone.
_DEFAULT_DISABLE_REGEXES: tuple[str, ...] = (
    # Positional encoding — constant multiply, no quant benefit.
    r"rotary_emb",
    # Attention score BMMs (Q@K^T, attn@V): activation×activation, no weight to
    # int8, wide dynamic range → keep fp16. Anchored to avoid q/k/v/o_proj.
    r"self_attn/MatMul(_\d+)?$",
    # NPU export fuses gate_proj + up_proj into one [up;gate] MatMul feeding
    # NPUGeglu, so Route A cannot transplant their separate donor int8 nodes.
    r"mlp/(gate_proj|up_proj)/MatMul",
    # Final action head: emits the denoise velocity directly and its error
    # accumulates over all 10 Euler steps — keep it fp16 for accuracy.
    r"action_out_proj",
)


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------


def _load_array(path: Path) -> np.ndarray:
    """Load a tensor saved as ``.npy`` or a torch ``.pth`` checkpoint."""
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix in (".pth", ".pt"):
        import torch

        t = torch.load(path, map_location="cpu")
        return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    raise ValueError(f"Unsupported calibration array file: {path}")


def _find_sample(sample_dir: Path, stem: str) -> Path | None:
    """Locate ``<stem>.npy`` / ``.pth`` / ``.pt`` inside a sample directory."""
    for suffix in (".npy", ".pth", ".pt"):
        candidate = sample_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _reshape_past_kv(past_kv: np.ndarray) -> np.ndarray:
    """Coerce an OM-style ``(L*2, B, S, D)`` KV cache to PT ``(L, 2, B, 1, S, D)``.

    Mirrors ``dump_ae_pt.py``: the AE export's ``unflatten_kv`` expects the PT
    layout. A 6-D array is already PT-shaped and returned unchanged.
    """
    if past_kv.ndim == 4:
        first = past_kv.shape[0]
        if first % 2 != 0:
            raise ValueError(f"OM-style past_kv first dim ({first}) is not divisible by 2")
        n_layers = first // 2
        bsize, seq, head_dim = past_kv.shape[1:]
        past_kv = past_kv.reshape(n_layers, 2, bsize, 1, seq, head_dim)
        LOGGER.info("  reshaped past_kv to %s (assumed L=%d, H=1)", past_kv.shape, n_layers)
    return past_kv


def _build_sample(
    sample_dir: Path,
    *,
    chunk_size: int,
    max_action_dim: int,
    sample_index: int,
) -> dict[str, np.ndarray]:
    """Build one AE input set ``{past_kv_tensor, prefix_pad_masks, time, noise}``."""
    pk_path = _find_sample(sample_dir, "past_kv_tensor")
    pm_path = _find_sample(sample_dir, "prefix_pad_masks")
    if pk_path is None or pm_path is None:
        raise FileNotFoundError(
            f"Calibration sample {sample_dir} must contain past_kv_tensor.* and "
            f"prefix_pad_masks.* (got past_kv={pk_path}, pad_masks={pm_path})."
        )
    past_kv = _reshape_past_kv(_load_array(pk_path))
    pad_masks = _load_array(pm_path)
    bsize = pad_masks.shape[0]

    noise_path = _find_sample(sample_dir, "noise")
    noise_shape = (bsize, chunk_size, max_action_dim)
    if noise_path is not None:
        noise = _load_array(noise_path)
        if tuple(noise.shape) != noise_shape:
            raise ValueError(f"Noise shape {noise.shape} != expected {noise_shape} in {sample_dir}")
    else:
        # Deterministic per-sample noise so calibration is reproducible.
        rng = np.random.default_rng(42 + sample_index)
        noise = rng.standard_normal(noise_shape)

    # time starts at 1.0 for the first denoising step (one scalar per batch).
    time = np.full((bsize,), 1.0)

    return {
        "past_kv_tensor": past_kv,
        "prefix_pad_masks": pad_masks,
        "time": time,
        "noise": noise,
    }


def build_calib_data(
    *,
    onnx_path: Path,
    calib_dir: str | None,
    past_kv_path: str | None,
    prefix_pad_masks_path: str | None,
    noise_path: str | None,
    num_calib: int,
) -> list[list[np.ndarray]]:
    """Produce AE ``calib_data`` as per-sample ordered input lists.

    Each sample is ordered to match the ONNX graph inputs and cast to each
    input's declared dtype (fp16/fp32 floats, bool masks).
    """
    input_order = common.ordered_input_names(onnx_path)
    dtypes = common.onnx_input_dtypes(onnx_path)
    LOGGER.info("AE ONNX inputs (in order): %s", input_order)

    # Derive chunk_size / max_action_dim from the declared noise input shape.
    chunk_size = max_action_dim = None
    import onnx

    proto = onnx.load(str(onnx_path), load_external_data=False)
    for inp in proto.graph.input:
        if inp.name == "noise":
            dims = inp.type.tensor_type.shape.dim
            if len(dims) == 3 and dims[1].HasField("dim_value") and dims[2].HasField("dim_value"):
                chunk_size = int(dims[1].dim_value)
                max_action_dim = int(dims[2].dim_value)
            break
    if chunk_size is None or max_action_dim is None:
        raise ValueError(
            "Could not read static (chunk_size, max_action_dim) from the ONNX 'noise' "
            "input shape; re-export the AE with fixed shapes."
        )
    LOGGER.info("AE noise shape: (B, %d, %d)", chunk_size, max_action_dim)

    # Collect calibration sample directories.
    sample_dirs: list[Path] = []
    if calib_dir:
        root = Path(calib_dir).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"--calib-dir not found: {root}")
        # A subdir holding past_kv_tensor.* is one sample; if the root itself
        # holds the tensors, treat the root as a single sample.
        subdirs = sorted(d for d in root.iterdir() if d.is_dir() and _find_sample(d, "past_kv_tensor"))
        if subdirs:
            sample_dirs = subdirs
        elif _find_sample(root, "past_kv_tensor"):
            sample_dirs = [root]
        else:
            raise FileNotFoundError(
                f"--calib-dir {root} contains no past_kv_tensor.* (neither in the root "
                "nor in any immediate subdirectory)."
            )
    elif past_kv_path and prefix_pad_masks_path:
        # Single explicit sample: stage the given paths into a virtual dir by
        # building the sample directly from the file paths.
        sample_dirs = []  # handled below
    else:
        raise ValueError("Provide either --calib-dir, or both --past-kv-path and --prefix-pad-masks-path.")

    calib_data: list[list[np.ndarray]] = []

    def _emit(sample: dict[str, np.ndarray]) -> None:
        missing = [n for n in input_order if n not in sample]
        if missing:
            raise KeyError(f"AE calibration sample missing inputs {missing}; have {sorted(sample)}.")
        calib_data.append([np.ascontiguousarray(sample[name], dtype=dtypes[name]) for name in input_order])

    if sample_dirs:
        if num_calib > 0:
            sample_dirs = sample_dirs[:num_calib]
        LOGGER.info("Using %d AE calibration sample(s) from disk", len(sample_dirs))
        for i, d in enumerate(sample_dirs):
            _emit(_build_sample(d, chunk_size=chunk_size, max_action_dim=max_action_dim, sample_index=i))
    else:
        # Single explicit-path sample.
        past_kv = _reshape_past_kv(_load_array(Path(past_kv_path).expanduser()))
        pad_masks = _load_array(Path(prefix_pad_masks_path).expanduser())
        bsize = pad_masks.shape[0]
        noise_shape = (bsize, chunk_size, max_action_dim)
        if noise_path:
            noise = _load_array(Path(noise_path).expanduser())
            if tuple(noise.shape) != noise_shape:
                raise ValueError(f"Noise shape {noise.shape} != expected {noise_shape}")
        else:
            noise = np.random.default_rng(42).standard_normal(noise_shape)
        _emit(
            {
                "past_kv_tensor": past_kv,
                "prefix_pad_masks": pad_masks,
                "time": np.full((bsize,), 1.0),
                "noise": noise,
            }
        )
        LOGGER.info("Using 1 AE calibration sample from explicit paths")

    if not calib_data:
        raise ValueError("No AE calibration samples were built — check the input paths.")
    return calib_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="msModelSlim W8A8 PTQ for the PI05 Action Expert ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    common.add_common_quant_args(p)
    p.add_argument(
        "--calib-dir",
        type=str,
        default=None,
        help="Directory of AE calibration samples. Each immediate subdirectory "
        "(or the directory itself) must hold past_kv_tensor.* + "
        "prefix_pad_masks.* (+ optional noise.*), as dumped by the VLM export "
        "/ dump_ae_pt.py.",
    )
    p.add_argument("--past-kv-path", type=str, default=None, help="Single-sample past_kv_tensor (.npy/.pth).")
    p.add_argument(
        "--prefix-pad-masks-path", type=str, default=None, help="Single-sample prefix_pad_masks (.npy/.pth)."
    )
    p.add_argument(
        "--noise-path", type=str, default=None, help="Optional fixed noise (.npy/.pth) for the single-sample path."
    )
    return p


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

    if not (args.calib_dir or (args.past_kv_path and args.prefix_pad_masks_path)):
        LOGGER.error(
            "Missing calibration inputs: provide --calib-dir, or both --past-kv-path and --prefix-pad-masks-path."
        )
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
        calib_dir=args.calib_dir,
        past_kv_path=args.past_kv_path,
        prefix_pad_masks_path=args.prefix_pad_masks_path,
        noise_path=args.noise_path,
        num_calib=args.num_calib,
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
        "Done. Next: ATC-compile %s on the board, then run the AE per-step "
        "trajectory compare (dump_ae_pt.py) to measure action-error regression.",
        output_path.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
