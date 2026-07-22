import argparse
import json
import os

import lerobot
import onnx
import torch
from onnxsim import simplify

from model_utils.export_paths import export_work_dir
from model_utils.inference_manifest_export import copy_policy_metadata_bundle


def logger(msg):
    print(f"[export_onnx_hisilicon]: {msg}")


logger(f"lerobot path: {lerobot.__file__}")


# Wrapper to rebuild observation.images list before calling ACT model,
# matching what ACTPolicy.forward does internally.
# ONNX export requires inputs to be passed as positional tensors, so this wrapper
# receives *args and reconstructs the batch dict internally.
class ACTONNXWrapper(torch.nn.Module):
    def __init__(self, model, input_names, image_keys):
        super().__init__()
        self.model = model
        self.input_names = input_names
        self.image_keys = image_keys

    def forward(self, *args):
        batch = {name: tensor for name, tensor in zip(self.input_names, args, strict=True)}
        batch["observation.images"] = [batch[key] for key in self.image_keys]
        output = self.model(batch)
        return output[0] if isinstance(output, tuple) else output


def export_act(args, config):
    from lerobot.policies.act.modeling_act import ACTPolicy

    model_path = args.policy_path
    work_dir = export_work_dir(model_path, "hisilicon", args.work_dir)
    onnx_path = str(work_dir / "model.onnx")
    simplified_path = str(work_dir / "model_simplified.onnx")
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

    policy = ACTPolicy.from_pretrained(model_path)
    policy.model = policy.model.to(args.device)
    policy.model.eval()
    wrapped_model = ACTONNXWrapper(policy.model, input_names, image_keys)

    logger("Exporting onnx")
    torch.onnx.export(
        wrapped_model,
        tuple(dummy_tensors),
        onnx_path,
        input_names=input_names,
        # mindcmd 要求 opset 版本为 13, atc 要求为 11~15
        opset_version=13,
        output_names=["action"],
        external_data=True,
        verbose=False,
        do_constant_folding=False,
    )

    logger("Simplify onnx")
    onnx_model = onnx.load(onnx_path)  # load onnx model
    model_simp, check = simplify(onnx_model)
    if not check:
        raise ValueError("Simplified ONNX model could not be validated")
    onnx.save(model_simp, simplified_path)
    if args.bundle_output is not None:
        copied = copy_policy_metadata_bundle(model_path, args.bundle_output)
        logger(f"Copied {len(copied)} required LeRobot semantic files to {args.bundle_output}")
        logger(
            "Finalize the deployment with package-compiled-deployment after the vendor toolchain produces "
            "the SD3403 OM, executable worker, and compiler runtime ABI JSON."
        )
    print("finished exporting onnx")


def parse_args():
    parser = argparse.ArgumentParser(description="Export ACT ONNX for the Hisilicon SD3403 toolchain")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference (e.g. cpu, cuda)")
    parser.add_argument("--policy_path", type=str, required=True, help="Path to pretrained policy model directory")
    parser.add_argument("--policy_type", type=str, default="act", help="Type of policy model (e.g. act)")
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="ONNX work directory (default: <bundle>/model_utils_work/hisilicon)",
    )
    parser.add_argument(
        "--bundle_output",
        type=str,
        default=None,
        help="Optional compiled bundle directory that receives required read-only LeRobot metadata",
    )
    args = parser.parse_args()

    config_path = os.path.join(args.policy_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    return args, config


if __name__ == "__main__":
    args, config = parse_args()
    if args.policy_type == "act":
        export_act(args, config)
    else:
        logger(f"Invalid option: {args.policy_type}")
