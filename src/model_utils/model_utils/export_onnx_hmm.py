import argparse
import os

import onnx
from onnxsim import simplify


def logger(msg):
    print(f"[export_onnx_hmm]: {msg}")


def strip_extra_outputs(onnx_path, output_path):
    """Keep only the ``action`` output so the compiled graph is action-only."""
    onnx_model = onnx.load(onnx_path)
    keep = ["action"]
    kept_outputs = [o for o in onnx_model.graph.output if o.name in keep]
    if len(kept_outputs) == len(onnx_model.graph.output):
        logger("No extra outputs to strip")
        os.replace(onnx_path, output_path)
        return
    logger(f"Stripping outputs: {[o.name for o in onnx_model.graph.output if o.name not in keep]}")
    while len(onnx_model.graph.output) > 0:
        onnx_model.graph.output.pop()
    for o in kept_outputs:
        onnx_model.graph.output.append(o)
    onnx.save(onnx_model, output_path)
    logger(f"Saved stripped model to {output_path}")


def process_existing_onnx(onnx_path, output_path=None):
    """Strip extra outputs and simplify an ONNX graph for the HMM toolchain."""
    if output_path is None:
        base, ext = os.path.splitext(onnx_path)
        output_path = f"{base}_hmm{ext}"

    logger(f"Step 1/2: Stripping extra outputs from {onnx_path}")
    stripped_path = output_path.replace(".onnx", "_stripped.onnx")
    strip_extra_outputs(onnx_path, stripped_path)

    logger("Step 2/2: Simplifying with onnxsim")
    onnx_model = onnx.load(stripped_path)
    model_simp, check = simplify(onnx_model)
    if not check:
        raise ValueError("Simplified ONNX model could not be validated")
    onnx.save(model_simp, output_path)

    if os.path.exists(stripped_path) and stripped_path != output_path:
        os.remove(stripped_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger(f"Done: {output_path} ({size_mb:.1f}MB)")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare an existing action-only ONNX graph; package deployable HMM modules separately"
    )
    parser.add_argument("--onnx", type=str, required=True, help="Existing ONNX model to strip and simplify")

    parser.add_argument(
        "--output", type=str, default=None, help="Output ONNX path (default: auto-named with _hmm suffix)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.onnx):
        logger(f"ONNX file not found: {args.onnx}")
        raise SystemExit(1)
    process_existing_onnx(args.onnx, args.output)
    logger(
        "Compile the policy-specific PI0.5 or SmolVLA modules with the Houmo toolchain, then run "
        "package-hmm-deployment with compiler-emitted TCIM model.json ABI inputs."
    )
