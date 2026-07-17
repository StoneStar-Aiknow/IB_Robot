import argparse
import json
import subprocess
from pathlib import Path

from model_utils.export_paths import ensure_output_parent, export_work_dir
from model_utils.inference_manifest_export import (
    artifact_bindings,
    compiled_deployment,
    package_deployment_artifact,
    read_runtime_abi,
    upsert_deployment,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, required=True, help="The path of pretrained model")
    parser.add_argument("--soc_version", type=str, required=True, help="The Ascend soc version")
    parser.add_argument(
        "--work_dir", type=str, default=None, help="Build work directory (default: <bundle>/model_utils_work/ascend)"
    )
    parser.add_argument("--onnx_model_path", type=str, default=None, help="ONNX work output path")
    parser.add_argument("--om_model_path", type=str, default=None, help="ATC OM work output path")
    parser.add_argument("--deployment", type=str, default="ascend", help="Unified manifest deployment name")
    parser.add_argument(
        "--om_abi_path",
        type=str,
        default=None,
        help="Existing compiler/runtime-introspected OM ABI JSON input (default: <om_model_path>.abi.json)",
    )
    parser.add_argument(
        "--skip_onnx_export",
        action="store_true",
        help="Convert an existing ONNX file without exporting from model.safetensors first",
    )
    args = parser.parse_args()

    pretrained_model_path = args.pretrained_model
    onnx_model_path = args.onnx_model_path
    om_model_path = args.om_model_path
    soc_version = args.soc_version

    work_dir = export_work_dir(pretrained_model_path, "ascend", args.work_dir)
    if onnx_model_path is None:
        onnx_model_path = str(work_dir / "model.onnx")
    if om_model_path is None:
        om_model_path = str(work_dir / "model.om")
    ensure_output_parent(onnx_model_path)
    ensure_output_parent(om_model_path)
    if args.om_abi_path is None:
        args.om_abi_path = f"{om_model_path}.abi.json"

    config_path = pretrained_model_path + "/config.json"
    with open(config_path) as f:
        config = json.load(f)

    return (
        pretrained_model_path,
        config,
        onnx_model_path,
        om_model_path,
        soc_version,
        args.skip_onnx_export,
        args.om_abi_path,
        args.deployment,
    )


def write_ascend_deployment(
    pretrained_model_path,
    config,
    onnx_model_path,
    om_model_path,
    soc_version,
    abi_path=None,
    deployment_name="ascend",
):
    policy_type = str(config.get("type", "")).lower().strip()
    if not policy_type:
        raise ValueError("config.json is missing required policy type metadata")
    if policy_type != "act":
        raise ValueError(f"Policy {policy_type} is not supported currently.")

    policy_dir = Path(pretrained_model_path).expanduser().resolve(strict=True)
    runtime_inputs = [
        key
        for key in config.get("input_features", {})
        if key == "observation.state" or key.startswith("observation.images.")
    ]
    if abi_path is None:
        raise ValueError("Ascend deployment packaging requires compiler/runtime ABI JSON")
    abi = read_runtime_abi(abi_path)
    input_names = [tensor.name for tensor in abi.inputs]
    if input_names != runtime_inputs:
        raise ValueError(f"ACT OM runtime inputs {input_names} do not match policy runtime inputs {runtime_inputs}")
    output_names = [tensor.name for tensor in abi.outputs]
    if len(output_names) != 1:
        raise ValueError("ACT OM runtime must expose exactly one output")
    bindings = artifact_bindings(
        abi,
        input_semantics={name: name for name in runtime_inputs},
        output_semantics={output_names[0]: "action"},
        image_layouts={name: "NCHW" for name in runtime_inputs if name.startswith("observation.images.")},
    )
    packaged_om = package_deployment_artifact(
        policy_dir,
        om_model_path,
        backend="ascend",
        deployment_name=deployment_name,
        role="policy",
        force_copy=True,
    )
    deployment = compiled_deployment(
        policy_dir,
        backend="ascend",
        target_soc=soc_version,
        target_runtime="acl",
        artifacts={"policy": (packaged_om, "om")},
        execution=("policy",),
        bindings={"policy": bindings},
    )
    return upsert_deployment(policy_dir, deployment_name, deployment).manifest_path


def _act_input_shape(config):
    input_shape = []
    for key, value in config["input_features"].items():
        # Only keep keys that ACT policy actually consumes;
        # skip unrelated entries like "observation.state.current".
        if key != "observation.state" and not key.startswith("observation.images."):
            continue
        shape = [1] + list(value["shape"])
        input_shape.append(key + ":" + ",".join(map(str, shape)))
    return ";".join(input_shape)


def convert_onnx_to_om(config, onnx_model_path, om_model_path, soc_version):
    input_shape = _act_input_shape(config)
    transfer_om_command = [
        "atc",
        "--framework=5",
        f"--soc_version={soc_version}",
        f"--model={onnx_model_path}",
        f"--output={om_model_path.removesuffix('.om')}",
        f"--input_shape={input_shape}",
    ]
    return subprocess.run(transfer_om_command, check=False).returncode == 0  # nosec B603


def export_act_model(pretrained_model_path, config, onnx_model_path, om_model_path, soc_version):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    input_features = config["input_features"]

    act_input_batch = {}
    input_names = []

    for key in input_features:
        # Only keep keys that ACT policy actually consumes;
        # skip unrelated entries like "observation.state.current".
        if key != "observation.state" and not key.startswith("observation.images."):
            continue

        shape = [1] + list(input_features[key]["shape"])

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

    policy_config = PreTrainedConfig.from_pretrained(pretrained_model_path)
    act_policy = ACTPolicy.from_pretrained(pretrained_model_path, config=policy_config)
    act_policy.model = act_policy.model.to("cpu")
    act_policy.model.eval()

    class ACTONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, batch):
            output = self.model(batch)
            return output[0] if isinstance(output, tuple) else output

    torch.onnx.export(
        ACTONNXWrapper(act_policy.model),
        (act_batch,),
        onnx_model_path,
        input_names=input_names,
        opset_version=14,
        output_names=["action"],
        dynamo=False,
    )

    return convert_onnx_to_om(config, onnx_model_path, om_model_path, soc_version)


if __name__ == "__main__":
    (
        pretrained_model_path,
        config,
        onnx_model_path,
        om_model_path,
        soc_version,
        skip_onnx_export,
        om_abi_path,
        deployment_name,
    ) = parse_args()
    policy_type = config["type"]

    if policy_type == "act":
        if skip_onnx_export:
            if not convert_onnx_to_om(config, onnx_model_path, om_model_path, soc_version):
                raise ValueError("convert_onnx_to_om failed")
        elif not export_act_model(pretrained_model_path, config, onnx_model_path, om_model_path, soc_version):
            raise ValueError("export_act_model failed")
        write_ascend_deployment(
            pretrained_model_path,
            config,
            onnx_model_path,
            om_model_path,
            soc_version,
            om_abi_path,
            deployment_name,
        )
    else:
        raise ValueError(f"Policy {policy_type} is not supported currently.")
