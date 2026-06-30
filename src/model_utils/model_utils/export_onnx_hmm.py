import argparse
import json
import os
import tempfile

import onnx
from onnxsim import simplify


def logger(msg):
    print(f"[export_onnx_hmm]: {msg}")


def _load_act_policy(model_path, device):
    """Load an ACT policy from a pretrained_model directory as a runtime graph.

    The exporter only needs a runtime-compatible ACT config; distill checkpoints
    may carry extra training metadata that draccus rejects, so the IB-Robot-only
    keys are stripped before parsing.
    """
    import draccus
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from safetensors.torch import load_model as load_model_as_safetensor

    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    config.pop("type", None)
    config.pop("is_ascend_om_enabled", None)
    config.pop("is_ascend_om_3403_enabled", None)

    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
        json.dump(config, f)
        sanitized_config_path = f.name

    try:
        with draccus.config_type("json"):
            act_config = draccus.parse(ACTConfig, sanitized_config_path, args=[])
    finally:
        if os.path.exists(sanitized_config_path):
            os.remove(sanitized_config_path)

    act_config.device = device
    policy = ACTPolicy(act_config)
    load_model_as_safetensor(policy, os.path.join(model_path, "model.safetensors"), strict=False, device=device)
    policy.eval()
    return policy


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


# Compiled-backend manifest written next to the policy directory so the
# ``hmm`` runtime can locate the ``.hmm`` artifact (mirrors config.om.json).
HMM_MANIFEST_BASENAME = "config.hmm.json"


def export_from_safetensors(args, config):
    import torch

    try:
        import lerobot

        logger(f"lerobot path: {lerobot.__file__}")
    except ImportError:
        logger("lerobot not available, cannot export from safetensors")
        logger("Use --onnx to process an existing ONNX model instead")
        return None

    model_path = args.policy_path
    onnx_raw = os.path.join(model_path, "act_ros2_hmm_raw.onnx")
    onnx_final = os.path.join(model_path, "act_ros2_hmm.onnx")

    input_features = config["input_features"]
    input_names = []
    image_keys = []
    dummy_tensors = []

    for key in input_features:
        if key != "observation.state" and not key.startswith("observation.images."):
            continue
        shape = [1] + list(input_features[key]["shape"])
        dummy_tensors.append(torch.randn(*shape, dtype=torch.float32, device=args.device))
        input_names.append(key)
        if key.startswith("observation.images."):
            image_keys.append(key)

    class ACTONNXWrapper(torch.nn.Module):
        def __init__(self, model, input_names, image_keys):
            super().__init__()
            self.model = model
            self.input_names = input_names
            self.image_keys = image_keys

        def forward(self, *args):
            batch = {name: tensor for name, tensor in zip(self.input_names, args, strict=False)}
            batch["observation.images"] = [batch[key] for key in self.image_keys]
            output = self.model(batch)
            if isinstance(output, dict):
                return output["action"]
            if isinstance(output, tuple):
                return output[0]
            return output

    act_policy = _load_act_policy(model_path, args.device)
    act_policy.model = act_policy.model.to(args.device)
    act_policy.model.eval()

    wrapped_model = ACTONNXWrapper(act_policy.model, input_names, image_keys)

    logger("Exporting ONNX (opset=13, action-only output)")
    torch.onnx.export(
        wrapped_model,
        tuple(dummy_tensors),
        onnx_raw,
        input_names=input_names,
        opset_version=13,
        output_names=["action"],
        do_constant_folding=True,
        verbose=False,
    )

    return process_existing_onnx(onnx_raw, onnx_final)


def write_hmm_manifest(policy_dir, config, hmm_path):
    """Write ``config.hmm.json`` so the HMM runtime resolves the ``.hmm`` artifact.

    Mirrors ``export_onnx_atc.write_om_manifest`` so the compiled-backend
    manifest contract stays uniform across OM / RKNN / HMM.
    """
    policy_type = str(config.get("type", "")).lower().strip()
    if not policy_type:
        raise ValueError("config.json is missing required policy type metadata")
    if policy_type != "act":
        raise ValueError(f"Policy {policy_type} is not supported by the HMM backend currently.")

    policy_dir = os.path.abspath(policy_dir)
    hmm_path = os.path.abspath(hmm_path)
    try:
        artifact = os.path.relpath(hmm_path, policy_dir)
    except ValueError:
        artifact = hmm_path

    manifest = {
        "schema_version": 1,
        "policy_type": policy_type,
        "backend": "hmm",
        "artifacts": {"policy": artifact},
        "execution": ["policy"],
    }
    manifest_path = os.path.join(policy_dir, HMM_MANIFEST_BASENAME)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    logger(f"Wrote HMM manifest -> {manifest_path}")
    return manifest_path


def convert_to_hmm(onnx_path, args, config):
    """Quantize (PTQ) and compile ONNX -> HMM via the Houmo toolchain.

    Two-stage, matching the verified houmo-examples workflow:

    1. ``xhquant.api.convert_onnx_to_hmonnx``: ONNX -> quantized HMONNX (w8a8).
    2. ``tcim.build_from_hmonnx``: HMONNX -> compiled ``.hmm`` for the xh2 target.

    Both steps run on the host that has ``xhquant`` and ``tcim`` installed.
    """
    try:
        from xhquant.api import (  # type: ignore[import-not-found]
            DeviceType,
            QuantScheme,
            convert_onnx_to_hmonnx,
            create_quant_config,
        )
    except ImportError:
        logger("xhquant is not installed; skipping HMM conversion")
        logger("  Install the Houmo toolchain (xhquant, tcim) on the host first.")
        return None

    try:
        import tcim  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        logger("tcim is not installed; skipping HMM conversion")
        logger("  Install the Houmo toolchain (tcim) on the host first.")
        return None

    import torch

    policy_dir = os.path.dirname(onnx_path)
    model_name = args.hmm_model_name or os.path.basename(policy_dir) or "act"
    hmonnx_path = os.path.join(policy_dir, f"{model_name}.hmonnx.onnx")
    hmm_output = args.hmm_output or os.path.join(policy_dir, "model.hmm")

    # ------------------------------------------------------------------
    # Stage 1: Post-Training Quantization (ONNX -> quantized HMONNX)
    # ------------------------------------------------------------------
    input_names = []
    input_shapes = []
    for key, feature in (config.get("input_features") or {}).items():
        if key != "observation.state" and not key.startswith("observation.images."):
            continue
        input_names.append(key)
        shape = [1] + list(feature["shape"])
        input_shapes.append(shape)

    if not input_names:
        raise ValueError("config.json does not expose any ACT state/image input features")

    output_names = ["action"]

    logger(f"Stage 1/2: PTQ ONNX -> HMONNX ({args.hmm_quant_type})")
    calib_tensors = [torch.randn(*shape, dtype=torch.float32) for shape in input_shapes]
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.hmm_quant_type)
    quant_config = create_quant_config(quant_scheme)
    convert_onnx_to_hmonnx(
        onnx_path,
        calib_tensors,
        device_type=DeviceType.XH2a,
        out_hmonnx_file=hmonnx_path,
        quant_config=quant_config,
        input_names=input_names,
        output_names=output_names,
    )
    logger(f"HMONNX saved -> {hmonnx_path}")

    # ------------------------------------------------------------------
    # Stage 2: Compile (HMONNX -> .hmm)
    # ------------------------------------------------------------------
    logger(f"Stage 2/2: Compile HMONNX -> HMM (target={args.hmm_target}, ncore={args.hmm_ncore})")
    work_dir = os.path.join(policy_dir, "tcim_work")
    os.makedirs(work_dir, exist_ok=True)
    tcim.build_from_hmonnx(
        hmonnx_path,
        output_name=model_name,
        ncore=args.hmm_ncore,
        opt_level=args.hmm_opt_level,
        target=args.hmm_target,
        batch=1,
        output_dir=policy_dir,
        work_dir=work_dir,
        enable_dynamic_image_resize=False,
    )

    # build_from_hmonnx writes ``<output_dir>/<model_name>.hmm``.
    built_hmm = os.path.join(policy_dir, f"{model_name}.hmm")
    if not os.path.isfile(built_hmm):
        # Fallback: some toolchain versions name the output ``model.hmm``.
        built_hmm = hmm_output
    if not os.path.isfile(built_hmm):
        logger(f"HMM model was not generated at {built_hmm}")
        return None

    if os.path.abspath(built_hmm) != os.path.abspath(hmm_output):
        os.replace(built_hmm, hmm_output)
    logger(f"HMM saved -> {hmm_output} ({os.path.getsize(hmm_output) / (1024 * 1024):.1f}MB)")

    write_hmm_manifest(policy_dir, config, hmm_output)
    return hmm_output


def parse_args():
    parser = argparse.ArgumentParser(description="Export ONNX for Houmo HMM (LQ50 / M50 xh2)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--policy_path", type=str, help="Path to pretrained policy model directory (export from safetensors)"
    )
    group.add_argument("--onnx", type=str, help="Path to existing ONNX model (strip + simplify only)")

    parser.add_argument(
        "--output", type=str, default=None, help="Output ONNX path (default: auto-named with _hmm suffix)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for export (cpu or cuda, only used with --policy_path)"
    )
    parser.add_argument("--convert_hmm", action="store_true", help="Also quantize + compile ONNX to HMM after export")

    # ---- HMM conversion options (host-side Houmo toolchain) ----
    parser.add_argument("--hmm_output", type=str, default=None, help="Output HMM path (default: <policy>/model.hmm)")
    parser.add_argument(
        "--hmm_model_name", type=str, default=None, help="Model name for HMM artifacts (default: dir name)"
    )
    parser.add_argument(
        "--hmm_target", type=str, default="xh2", help="Houmo target platform (default: xh2 for LQ50/M50)"
    )
    parser.add_argument("--hmm_ncore", type=int, default=2, help="Number of cores to compile for (default: 2)")
    parser.add_argument("--hmm_opt_level", type=str, default="O2", help="Compiler optimization level (default: O2)")
    parser.add_argument(
        "--hmm_quant_type",
        type=str,
        default="w8a8h1_sefp",
        help="PTQ quantization type passed to QuantScheme (default: w8a8h1_sefp)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.onnx:
        if not os.path.exists(args.onnx):
            logger(f"ONNX file not found: {args.onnx}")
            raise SystemExit(1)
        onnx_result = process_existing_onnx(args.onnx, args.output)
        config = {}
    else:
        config_path = os.path.join(args.policy_path, "config.json")
        if not os.path.exists(config_path):
            logger(f"config.json not found: {args.policy_path}")
            raise SystemExit(1)
        with open(config_path) as f:
            config = json.load(f)

        onnx_result = export_from_safetensors(args, config)
        if onnx_result is None:
            logger("Export from safetensors failed. Use --onnx with an existing ONNX model.")
            raise SystemExit(1)

    if args.convert_hmm and onnx_result:
        hmm_result = convert_to_hmm(onnx_result, args, config)
        if hmm_result is None:
            raise SystemExit(1)
