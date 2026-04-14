import argparse
import json
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, required=True, help="The path of pretrained model")
    parser.add_argument("--soc_version", type=str, required=True, help="The Ascend soc version")
    parser.add_argument("--onnx_model_path", type=str, default=None, help="The path to store onnx model")
    parser.add_argument("--om_model_path", type=str, default=None, help="The path to store om model")
    args = parser.parse_args()

    pretrained_model_path = args.pretrained_model
    onnx_model_path = args.onnx_model_path
    om_model_path = args.om_model_path
    soc_version = args.soc_version

    if onnx_model_path is None:
        onnx_model_path = pretrained_model_path + "/model.onnx"
    if om_model_path is None:
        om_model_path = pretrained_model_path + "/model.om"

    config_path = pretrained_model_path + "/config.json"
    with open(config_path) as f:
        config = json.load(f)

    return pretrained_model_path, config, onnx_model_path, om_model_path, soc_version


def export_act_model(pretrained_model_path, config, onnx_model_path, om_model_path, soc_version):
    from lerobot.policies.act.modeling_act import ACTPolicy

    input_features = config["input_features"]

    act_input_batch = {}
    input_names = []
    input_shape = []

    for key in input_features:
        # Only keep keys that ACT policy actually consumes;
        # skip unrelated entries like "observation.state.current".
        if key != "observation.state" and not key.startswith("observation.images."):
            continue

        shape = [1] + list(input_features[key]["shape"])
        shape_str = ",".join(map(str, shape))

        input_shape.append(key + ":" + shape_str)
        input_names.append(key)

        if key == "observation.state":
            act_input_batch[key] = torch.rand(shape, dtype=torch.float32, device="cpu")
        else:
            if "observation.images" not in act_input_batch:
                act_input_batch["observation.images"] = []
            act_input_batch["observation.images"].append(torch.rand(shape, dtype=torch.float32, device="cpu"))

    act_batch = {
        "batch": act_input_batch,
    }

    # Force-disable OM inference modes so that from_pretrained always
    # constructs the standard PyTorch model (self.model = ACT(config)),
    # regardless of what config.json says.  Exporting ONNX/OM requires
    # the original PyTorch graph — the OM wrapper is irrelevant here.
    act_policy = ACTPolicy.from_pretrained(
        pretrained_model_path,
        cli_overrides=[
            "--is_ascend_om_enabled=false",
            "--is_ascend_om_3403_enabled=false",
        ],
    )
    act_policy.model = act_policy.model.to("cpu")
    act_policy.model.eval()

    torch.onnx.export(
        act_policy.model,
        (act_batch,),
        onnx_model_path,
        input_names=input_names,
        opset_version=14,
        output_names=["action"],
        dynamo=False,
    )

    input_shape = '"' + ";".join(input_shape) + '"'

    transfer_om_command = (
        "atc --framework=5 --soc_version="
        + soc_version
        + " --model="
        + onnx_model_path
        + " --output="
        + om_model_path.removesuffix(".om")
        + " --input_shape="
        + input_shape
    )

    result = os.system(transfer_om_command)  # nosec B605
    return result == 0


if __name__ == "__main__":
    pretrained_model_path, config, onnx_model_path, om_model_path, soc_version = parse_args()
    policy_type = config["type"]

    if policy_type == "act":
        if not export_act_model(pretrained_model_path, config, onnx_model_path, om_model_path, soc_version):
            raise ValueError("export_act_model failed")
    else:
        raise ValueError(f"Policy {policy_type} is not supported currently.")
