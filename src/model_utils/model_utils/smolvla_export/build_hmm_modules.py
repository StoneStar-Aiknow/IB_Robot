#!/usr/bin/env python3
"""Compile repository-exported SmolVLA HMONNX modules to Houmo HMM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import psutil
import tcim

MODULES = {
    "vision": {"flash_attention": 2, "llm_opt": False, "subgraph_repeat_hint": -27},
    "prefill": {"flash_attention": 0, "llm_opt": True, "subgraph_repeat_hint": -18},
    "action": {"flash_attention": 0, "llm_opt": False, "subgraph_repeat_hint": -27},
}


def build_module(output_dir: Path, role: str) -> None:
    settings = MODULES[role]
    hmonnx = output_dir / "hmonnx" / f"smolvla_{role}_xh2.onnx"
    hmm_dir = output_dir / "hmm"
    output_name = f"smolvla_{role}"
    custom_msg = json.dumps({"flash_attention": settings["flash_attention"]})
    tcim.build_from_hmonnx(
        str(hmonnx),
        output_name=output_name,
        ncore=2,
        target="xh2",
        output_dir=str(hmm_dir),
        work_dir=str(hmm_dir / "tcim" / output_name),
        j=psutil.cpu_count(logical=False),
        llm_opt=settings["llm_opt"],
        skip_mlir_compile=False,
        enable_common_subgraph=False,
        subgraph_repeat_hint=settings["subgraph_repeat_hint"],
        flash_attention=settings["flash_attention"],
        custom_msg=custom_msg,
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
