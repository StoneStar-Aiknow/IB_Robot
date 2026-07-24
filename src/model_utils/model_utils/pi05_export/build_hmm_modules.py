#!/usr/bin/env python3
"""Compile PI0.5 HMONNX modules to Houmo HMM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import psutil

MODULES = {
    "vision": ("siglip", False, -27, 2),
    "prefill": ("gemma_2b_prefill", True, -18, 0),
    "action_in_proj": ("action_in_proj", False, None, None),
    "time_mlp": ("time_mlp", False, None, None),
    "decode": ("gemma_expert_300m_decode", True, -18, 0),
    "action_out_proj": ("action_out_proj", False, None, None),
}


def _hmonnx_path(output_dir: Path, role: str) -> Path:
    return output_dir / "hmonnx" / f"{role}.onnx"


def build_module(output_dir: Path, role: str) -> None:
    import tcim

    output_name, llm_opt, repeat_hint, flash_attention = MODULES[role]
    kwargs: dict[str, object] = {}
    if repeat_hint is not None:
        kwargs["subgraph_repeat_hint"] = repeat_hint
    if flash_attention is not None:
        kwargs["flash_attention"] = flash_attention
        kwargs["custom_msg"] = json.dumps({"flash_attention": flash_attention})
    hmm_dir = output_dir / "hmm"
    tcim.build_from_hmonnx(
        str(_hmonnx_path(output_dir, role)),
        output_name=output_name,
        ncore=2,
        target="xh2",
        output_dir=str(hmm_dir),
        work_dir=str(hmm_dir / "tcim" / output_name),
        j=psutil.cpu_count(logical=False),
        llm_opt=llm_opt,
        skip_mlir_compile=False,
        enable_common_subgraph=False,
        **kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("roles", nargs="*", choices=tuple(MODULES))
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    for role in args.roles or MODULES:
        build_module(output_dir, role)


if __name__ == "__main__":
    main()
