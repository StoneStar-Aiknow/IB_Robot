#!/usr/bin/env python3
"""Package compiled SmolVLA HMM modules into a strict unified deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference_manifest import load_inference_manifest
from model_utils.hmm_export import write_hmm_deployment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deployment", default="hmm")
    parser.add_argument("--target-soc", default="lq50")
    parser.add_argument("--target-runtime", default="tcim")
    args = parser.parse_args()

    bundle_root = Path(args.bundle_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    with (bundle_root / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)

    hmm_dir = output_dir / "hmm"
    tcim_dir = hmm_dir / "tcim"
    manifest_path = write_hmm_deployment(
        bundle_root,
        config,
        vision_hmm=hmm_dir / "smolvla_vision.hmm",
        vision_abi_path=tcim_dir / "smolvla_vision" / "model.json",
        embedding_path=output_dir / "token_embedding.pt",
        state_projection_path=output_dir / "state_projection.pt",
        role_artifacts={
            "prefill": (
                hmm_dir / "smolvla_prefill.hmm",
                tcim_dir / "smolvla_prefill" / "model.json",
            ),
            "action": (
                hmm_dir / "smolvla_action.hmm",
                tcim_dir / "smolvla_action" / "model.json",
            ),
        },
        deployment_name=args.deployment,
        target_soc=args.target_soc,
        target_runtime=args.target_runtime,
        vision_layout="NCHW",
    )
    manifest_path.chmod(0o644)
    load_inference_manifest(bundle_root, args.deployment)
    print(f"Strict HMM deployment verified: {manifest_path}")


if __name__ == "__main__":
    main()
