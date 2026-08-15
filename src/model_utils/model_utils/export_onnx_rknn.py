import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import onnx
from onnxsim import simplify

from model_utils.export_paths import ensure_output_parent, export_work_dir
from model_utils.inference_manifest_export import (
    artifact_bindings,
    compiled_deployment,
    package_deployment_artifact,
    read_runtime_abi,
    upsert_deployment,
)


def logger(msg):
    print(f"[export_onnx_rknn]: {msg}")


def strip_extra_outputs(onnx_path, output_path):
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
    if output_path is None:
        base, ext = os.path.splitext(onnx_path)
        output_path = f"{base}_rknn{ext}"

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


def export_from_safetensors(args, config):
    import torch

    try:
        import lerobot

        logger(f"lerobot path: {lerobot.__file__}")
    except ImportError:
        logger("lerobot not available, cannot export from safetensors")
        logger("Use --onnx to process an existing ONNX model instead")
        return None

    from lerobot.policies.act.modeling_act import ACTPolicy

    model_path = args.policy_path
    work_dir = export_work_dir(model_path, "rknn", args.work_dir)
    onnx_final = ensure_output_parent(args.output) if args.output else work_dir / "model.onnx"
    onnx_raw = onnx_final.with_name(f"{onnx_final.stem}_raw.onnx")

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

    act_policy = ACTPolicy.from_pretrained(model_path)
    act_policy.model = act_policy.model.to(args.device)
    act_policy.model.eval()

    wrapped_model = ACTONNXWrapper(act_policy.model, input_names, image_keys)

    logger("Exporting ONNX (opset=13, action-only output)")
    torch.onnx.export(
        wrapped_model,
        tuple(dummy_tensors),
        str(onnx_raw),
        input_names=input_names,
        opset_version=13,
        output_names=["action"],
        do_constant_folding=True,
        verbose=False,
    )

    return process_existing_onnx(str(onnx_raw), str(onnx_final))


def _ensure_rknn_venv(venv_python):
    venv_dir = os.path.dirname(os.path.dirname(venv_python))
    if os.path.exists(venv_python):
        return True
    logger(f"Creating .venv-rknn at {venv_dir}")
    subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
    subprocess.run([venv_python, "-m", "pip", "install", "-q", "rknn-toolkit2", "onnx", "onnxruntime"], check=True)
    logger(".venv-rknn created and dependencies installed")
    return True


def _rknn_output_paths(onnx_path, rknn_output=None, abi_output=None):
    model_output = rknn_output or str(Path(onnx_path).with_suffix(".rknn"))
    return model_output, abi_output or f"{model_output}.abi.json"


def convert_to_rknn(onnx_path, args):
    rknn_output, abi_output = _rknn_output_paths(onnx_path, args.rknn_output, args.rknn_abi_output)
    ensure_output_parent(rknn_output)
    venv_python = args.rknn_venv_python

    if not venv_python:
        logger("No rknn venv path configured, skipping RKNN conversion")
        return None

    try:
        _ensure_rknn_venv(venv_python)
    except Exception as e:
        logger(f"Failed to create .venv-rknn: {e}")
        logger(
            "  Create manually: python3 -m venv .venv-rknn && .venv-rknn/bin/pip install rknn-toolkit2 onnx onnxruntime"
        )
        return None

    convert_script = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
        ".agents",
        "skills",
        "rknn-convert",
        "convert_to_rknn.py",
    )
    if not os.path.exists(convert_script):
        logger(f"RKNN convert script not found: {convert_script}")
        return None

    cmd = [
        venv_python,
        convert_script,
        "--onnx",
        onnx_path,
        "--output",
        rknn_output,
        "--mode",
        args.rknn_mode,
        "--abi-output",
        abi_output,
    ]
    logger(f"Converting to RKNN: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger("RKNN conversion failed!")
        return None
    logger(f"RKNN model saved to {rknn_output}")
    if not os.path.isfile(abi_output):
        logger(f"RKNN conversion did not produce runtime ABI metadata: {abi_output}")
        return None
    return rknn_output, abi_output


def write_rknn_deployment(policy_path, config, rknn_output, abi_output, deployment_name="rknn"):
    policy_type = str(config.get("type", "")).lower().strip()
    if policy_type != "act":
        raise ValueError(f"ACT RKNN exporter cannot package policy type {policy_type!r}")
    runtime_inputs = [
        key
        for key in config.get("input_features", {})
        if key == "observation.state" or key.startswith("observation.images.")
    ]
    abi = read_runtime_abi(abi_output)
    input_names = [tensor.name for tensor in abi.inputs]
    if input_names != runtime_inputs:
        raise ValueError(f"RKNN runtime inputs {input_names} do not match policy runtime inputs {runtime_inputs}")
    if [tensor.name for tensor in abi.outputs] != ["action"]:
        raise ValueError("ACT RKNN runtime must expose exactly one output named 'action'")
    image_layouts = {}
    for tensor in abi.inputs:
        if not tensor.name.startswith("observation.images."):
            continue
        layout = tensor.layout
        if layout not in {"NCHW", "NHWC"}:
            raise ValueError(f"RKNN runtime ABI must declare NCHW or NHWC layout for {tensor.name!r}")
        image_layouts[tensor.name] = layout
    bindings = artifact_bindings(
        abi,
        input_semantics={name: name for name in runtime_inputs},
        output_semantics={"action": "action"},
        image_layouts=image_layouts,
    )
    packaged_model = package_deployment_artifact(
        policy_path,
        rknn_output,
        backend="rknn",
        deployment_name=deployment_name,
        role="policy",
        force_copy=True,
    )
    deployment = compiled_deployment(
        policy_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (packaged_model, "rknn")},
        execution=("policy",),
        bindings={"policy": bindings},
    )
    return upsert_deployment(policy_path, deployment_name, deployment).manifest_path


def parse_args():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    parser = argparse.ArgumentParser(description="Export ONNX for RKNN (RK3588 NPU)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--policy_path", type=str, help="Path to pretrained policy model directory (export from safetensors)"
    )
    group.add_argument("--onnx", type=str, help="Path to existing ONNX model (strip + simplify only)")
    parser.add_argument(
        "--bundle_root",
        type=str,
        default=None,
        help="Policy bundle root required to package --onnx conversion into the unified manifest",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Build work directory (default: models/_work/<bundle>/rknn when a bundle is available)",
    )

    parser.add_argument(
        "--output", type=str, default=None, help="Output ONNX path (default: auto-named with _rknn suffix)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for export (cpu or cuda, only used with --policy_path)"
    )
    parser.add_argument("--convert_rknn", action="store_true", help="Also convert ONNX to RKNN after export")
    parser.add_argument(
        "--rknn_output", type=str, default=None, help="Output RKNN path (default: same as onnx with .rknn)"
    )
    parser.add_argument(
        "--rknn_abi_output",
        type=str,
        default=None,
        help="Compiler-generated RKNN ABI JSON output path (default: <rknn_output>.abi.json)",
    )
    parser.add_argument("--deployment", type=str, default="rknn", help="Unified manifest deployment name")
    parser.add_argument(
        "--rknn_mode",
        type=str,
        default="float16",
        choices=["float16", "int8", "hybrid"],
        help="RKNN conversion mode (default: float16)",
    )
    parser.add_argument(
        "--rknn_venv_python", type=str, default=None, help="Path to .venv-rknn python (auto-detected if not set)"
    )

    args = parser.parse_args()

    if args.rknn_venv_python is None:
        candidate = os.path.join(project_root, ".venv-rknn", "bin", "python")
        if os.path.exists(candidate):
            args.rknn_venv_python = candidate

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.onnx:
        if not os.path.exists(args.onnx):
            logger(f"ONNX file not found: {args.onnx}")
            sys.exit(1)
        onnx_output = args.output
        if onnx_output is None and args.bundle_root is not None:
            onnx_output = str(export_work_dir(args.bundle_root, "rknn", args.work_dir) / "model.onnx")
        onnx_result = process_existing_onnx(args.onnx, onnx_output)
        if args.convert_rknn:
            if args.bundle_root is None:
                logger("--onnx --convert_rknn requires --bundle_root for semantic manifest validation")
                sys.exit(1)
            config_path = os.path.join(args.bundle_root, "config.json")
            if not os.path.isfile(config_path):
                logger(f"config.json not found under --bundle_root: {args.bundle_root}")
                sys.exit(1)
            with open(config_path) as f:
                config = json.load(f)
    else:
        config_path = os.path.join(args.policy_path, "config.json")
        if not os.path.exists(config_path):
            logger(f"config.json not found: {args.policy_path}")
            sys.exit(1)
        with open(config_path) as f:
            config = json.load(f)

        onnx_result = export_from_safetensors(args, config)
        if onnx_result is None:
            logger("Export from safetensors failed. Use --onnx with an existing ONNX model.")
            sys.exit(1)

    if args.convert_rknn and onnx_result:
        converted = convert_to_rknn(onnx_result, args)
        if converted is None:
            sys.exit(1)
        rknn_output, abi_output = converted
        bundle_root = args.bundle_root if args.onnx else args.policy_path
        write_rknn_deployment(bundle_root, config, rknn_output, abi_output, args.deployment)
