import argparse
import os

import onnx
import torch
from onnxsim import simplify

import lerobot


def logger(msg):
    print(f"[export_onnx]: {msg}")


logger(f"lerobot path: {lerobot.__file__}")


def export_act(args):
    from lerobot.policies.act.modeling_act import ACTPolicy

    model_path = args.policy_path
    onnx_path = os.path.join(model_path, "act_ros2.onnx")

    policy = ACTPolicy.from_pretrained(model_path)

    policy.model.eval()
    dummy_batch = {
        "observation.state": torch.randn(1, 6, dtype=torch.float32, device=args.device),
        "observation.images.top": torch.randn(1, 3, 240, 320, dtype=torch.float32, device=args.device),
        "observation.images.wrist": torch.randn(1, 3, 240, 320, dtype=torch.float32, device=args.device),
    }

    logger("Exporting onnx")
    torch.onnx.export(
        policy.model,
        (dummy_batch,),
        onnx_path,
        input_names=[
            "observation.state",
            "observation.images.top",
            "observation.images.wrist",
        ],
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
    model_simp, check = simplify(onnx_model)
    if not check:
        raise ValueError("Simplified ONNX model could not be validated")
    onnx.save(model_simp, os.path.join(model_path, "act_ros2_simplified.onnx"))
    print("finished exporting onnx")


def parse_args():
    parser = argparse.ArgumentParser(description="export_onnx")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference (e.g. cpu, cuda)")
    parser.add_argument(
        "--policy_path", type=str, required=True, help="Path to pretrained policy model directory"
    )
    parser.add_argument("--policy_type", type=str, default="act", help="Type of policy model (e.g. act)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.policy_type == "act":
        export_act(args)
    else:
        logger(f"Invalid option: {args.policy_type}")
