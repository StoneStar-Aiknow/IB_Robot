#!/usr/bin/env python3
"""Export native PI0.5 modules to Houmo HMONNX."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

import onnx
import torch
from lerobot.policies.pi05 import PI05Policy
from onnxsim import simplify
from torch import nn
from torch.nn import functional as F

from model_utils.inference_manifest_export import copy_policy_metadata_bundle


class VisionWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.vision_tower = model.vision_tower.eval()
        self.multi_modal_projector = model.multi_modal_projector.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.vision_tower(pixel_values).last_hidden_state
        return self.multi_modal_projector(features)


class TimeMLPWrapper(nn.Module):
    def __init__(self, time_mlp_in: nn.Module, time_mlp_out: nn.Module):
        super().__init__()
        self.time_mlp_in = time_mlp_in
        self.time_mlp_out = time_mlp_out

    def forward(self, time_emb: torch.Tensor) -> torch.Tensor:
        return F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_emb))))


def _flatten_cache(past_key_values) -> tuple[torch.Tensor, ...]:
    """Normalize the Transformers cache result to interleaved ONNX tensors."""
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    return tuple(tensor for layer in past_key_values for tensor in layer[:2])


def _unflatten_cache(flat_cache: tuple[torch.Tensor, ...]):
    if len(flat_cache) % 2:
        raise ValueError("PI0.5 cache tensors must be interleaved key/value pairs")
    from transformers import DynamicCache

    return DynamicCache(
        ddp_cache_data=tuple((flat_cache[index], flat_cache[index + 1]) for index in range(0, len(flat_cache), 2))
    )


def _cache_names(num_layers: int) -> list[str]:
    return [name for index in range(num_layers) for name in (f"past_key_{index}", f"past_value_{index}")]


class PI05PrefillWrapper(nn.Module):
    """Run the PaliGemma prefix model and expose its KV cache as ONNX outputs."""

    def __init__(self, language_model: nn.Module, num_cache_layers: int):
        super().__init__()
        self.language_model = language_model.eval()
        self.num_cache_layers = num_cache_layers

    def forward(
        self,
        prefix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        prefix_dtype = self.language_model.layers[0].self_attn.q_proj.weight.dtype
        outputs = self.language_model(
            inputs_embeds=prefix_embs.to(dtype=prefix_dtype),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=True,
        )
        flat_cache = _flatten_cache(outputs.past_key_values)
        return flat_cache[: self.num_cache_layers * 2]


class PI05DecodeWrapper(nn.Module):
    """Run the conditioned action expert from prefill-produced KV cache tensors."""

    def __init__(self, expert_model: nn.Module):
        super().__init__()
        self.expert_model = expert_model.eval()

    def forward(
        self,
        action_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        condition: torch.Tensor,
        *flat_cache: torch.Tensor,
    ) -> torch.Tensor:
        expert_dtype = self.expert_model.layers[0].self_attn.q_proj.weight.dtype
        outputs = self.expert_model(
            inputs_embeds=action_embs.to(dtype=expert_dtype),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=_unflatten_cache(flat_cache),
            use_cache=False,
            # AdaRMSNorm's conditioning projection is retained in float32.
            adarms_cond=condition.to(dtype=torch.float32),
        )
        return outputs.last_hidden_state


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_onnx(
    module: nn.Module,
    example: torch.Tensor,
    output_path: Path,
    input_name: str,
    output_name: str,
    opset: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_name(output_path.stem.removesuffix("_simplified") + ".onnx")
    module = module.eval().float()
    torch.onnx.export(
        module,
        example,
        raw_path,
        input_names=[input_name],
        output_names=[output_name],
        opset_version=opset,
    )
    simplified, valid = simplify(onnx.load(raw_path), test_input_shapes={input_name: list(example.shape)})
    if not valid:
        raise RuntimeError(f"ONNX simplification failed: {raw_path}")
    onnx.save(simplified, output_path)
    return output_path


def _convert_to_hmonnx(onnx_path: Path, hmonnx_path: Path, inputs: tuple[torch.Tensor, ...], quant_type: str) -> None:
    from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config

    hmonnx_path.parent.mkdir(parents=True, exist_ok=True)
    scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    convert_onnx_to_hmonnx(
        str(onnx_path),
        inputs,
        out_hmonnx_file=str(hmonnx_path),
        device_type="XH2A",
        quant_config=create_quant_config(scheme),
    )


def _convert_fx_to_hmonnx(
    module: nn.Module,
    hmonnx_path: Path,
    inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    quant_type: str,
) -> None:
    from xhquant.api import DeviceType, QuantScheme, convert_fx_model_to_hmonnx, create_quant_config

    hmonnx_path.parent.mkdir(parents=True, exist_ok=True)
    scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    convert_fx_model_to_hmonnx(
        module.eval(),
        list(inputs),
        DeviceType.XH2a,
        str(hmonnx_path),
        quant_config=create_quant_config(scheme),
        input_names=input_names,
        output_names=output_names,
    )


def _convert_dynamo_to_hmonnx(
    module: nn.Module,
    hmonnx_path: Path,
    inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    quant_type: str,
) -> None:
    from xhquant.api import DeviceType, QuantScheme, convert_dynamo_model_to_hmonnx, create_quant_config

    hmonnx_path.parent.mkdir(parents=True, exist_ok=True)
    scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    convert_dynamo_model_to_hmonnx(
        module.eval(),
        list(inputs),
        DeviceType.XH2a,
        str(hmonnx_path),
        quant_config=create_quant_config(scheme),
        input_names=input_names,
        output_names=output_names,
    )


def _write_provenance(args: argparse.Namespace, output_dir: Path) -> None:
    repo_root = Path(args.repo_root).resolve()
    lerobot_root = Path(args.lerobot_src).resolve().parent
    checkpoint = Path(args.model_path).resolve() / "model.safetensors"
    provenance = {
        "schema_version": 1,
        "pipeline": "ibrobot-pi05-hmm",
        "ib_robot": {
            "commit": _git_output(repo_root, "rev-parse", "HEAD"),
            "dirty_files": _git_output(repo_root, "status", "--short").splitlines(),
        },
        "lerobot": {
            "source": str(Path(args.lerobot_src).resolve()),
            "branch": _git_output(lerobot_root, "branch", "--show-current"),
            "head": _git_output(lerobot_root, "rev-parse", "HEAD"),
            "dirty_files": _git_output(lerobot_root, "status", "--short").splitlines(),
        },
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "toolchain": {
            "image_id": os.environ.get("IBR_HOUMO_IMAGE_ID", "unknown"),
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "onnx": onnx.__version__,
            "onnxsim": _package_version("onnxsim"),
            "xhquant": _package_version("hmquant-xh2"),
            "tcim": _package_version("houmo-tcim-xh2"),
        },
        "export": {"opset": args.opset, "quant_type": args.quant_type, "seed": args.seed},
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def export_hmm(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    bundle_root = Path(args.bundle_root).resolve() if args.bundle_root else output_dir.parents[1]
    copy_policy_metadata_bundle(args.model_path, bundle_root)
    onnx_dir = output_dir / "onnx"
    hmonnx_dir = output_dir / "hmonnx"
    policy = PI05Policy.from_pretrained(args.model_path, strict=True)
    policy = policy.float().cpu().eval()
    model = policy.model
    chunk_size = int(policy.config.chunk_size)
    action_dim = int(policy.config.max_action_dim)
    expert_width = int(model.action_in_proj.out_features)
    prefix_model = model.paligemma_with_expert.paligemma.model.language_model
    expert_model = model.paligemma_with_expert.gemma_expert.model
    torch.save({"weight": prefix_model.embed_tokens.weight.detach().cpu()}, output_dir / "embedding.pt")
    if len(prefix_model.layers) != len(expert_model.layers):
        raise RuntimeError(
            f"PI0.5 prefix/expert layer mismatch: {len(prefix_model.layers)} != {len(expert_model.layers)}"
        )
    num_layers = len(expert_model.layers)
    image_height, image_width = (int(value) for value in policy.config.image_resolution)
    tokenizer_length = int(policy.config.tokenizer_max_length)
    prefix_length = len(policy.config.image_features) * 256 + tokenizer_length
    prefix_hidden_size = int(prefix_model.config.hidden_size)
    prefix_dtype = prefix_model.layers[0].self_attn.q_proj.weight.dtype
    prefix_embs = torch.randn(1, prefix_length, prefix_hidden_size, dtype=prefix_dtype)
    prefix_attention = torch.zeros(1, 1, prefix_length, prefix_length, dtype=prefix_dtype)
    prefix_positions = torch.arange(prefix_length, dtype=torch.long).unsqueeze(0)
    prefill_wrapper = PI05PrefillWrapper(prefix_model, num_layers)
    with torch.no_grad():
        flat_cache = prefill_wrapper(prefix_embs, prefix_attention, prefix_positions)
    if len(flat_cache) != num_layers * 2:
        raise RuntimeError(f"PI0.5 prefill returned {len(flat_cache)} cache tensors, expected {num_layers * 2}")

    decode_length = chunk_size
    action_embs = torch.randn(1, decode_length, expert_width, dtype=prefix_dtype)
    decode_attention = torch.zeros(1, 1, decode_length, prefix_length + decode_length, dtype=prefix_dtype)
    decode_positions = torch.arange(prefix_length, prefix_length + decode_length, dtype=torch.long).unsqueeze(0)
    condition = torch.randn(1, expert_width, dtype=prefix_dtype)

    modules = {
        "vision": (
            VisionWrapper(model.paligemma_with_expert.paligemma.model),
            torch.randn(1, 3, image_height, image_width),
            "pixel_values",
            "image_features",
        ),
        "action_in_proj": (
            model.action_in_proj,
            torch.randn(1, chunk_size, action_dim),
            "action_in",
            "action_in_proj_out",
        ),
        "time_mlp": (
            TimeMLPWrapper(model.time_mlp_in, model.time_mlp_out),
            torch.randn(1, expert_width),
            "time_emb",
            "time_mlp_out",
        ),
        "action_out_proj": (
            model.action_out_proj,
            torch.randn(1, chunk_size, expert_width),
            "action_out",
            "action_out_proj_out",
        ),
    }
    selected_roles = set(args.roles or (*modules, "prefill", "decode"))
    for role, (module, example, input_name, output_name) in modules.items():
        if role not in selected_roles:
            continue
        onnx_path = _export_onnx(
            module,
            example,
            onnx_dir / f"pi05_{role}_simplified.onnx",
            input_name,
            output_name,
            args.opset,
        )
        _convert_to_hmonnx(onnx_path, hmonnx_dir / f"{role}.onnx", (example,), args.quant_type)

    from model_utils.pi05_export.houmo_pi_gemma import HoumoPI05DecodeWrapper, HoumoPI05PrefillWrapper

    llm_modules = {
        "prefill": (
            HoumoPI05PrefillWrapper(prefix_model),
            (prefix_embs, prefix_attention, prefix_positions),
            ["prefix_embs", "attention_mask", "position_ids"],
            _cache_names(num_layers),
        ),
        "decode": (
            HoumoPI05DecodeWrapper(expert_model),
            (action_embs, decode_attention, decode_positions, condition, *flat_cache),
            ["action_embs", "attention_mask", "position_ids", "condition", *_cache_names(num_layers)],
            ["suffix_hidden"],
        ),
    }
    for role, (module, inputs, input_names, output_names) in llm_modules.items():
        if role not in selected_roles:
            continue
        export_inputs = tuple(
            tensor.to(dtype=torch.float32) if tensor.is_floating_point() else tensor for tensor in inputs
        )
        convert = _convert_dynamo_to_hmonnx if role == "decode" else _convert_fx_to_hmonnx
        convert(
            module.float().eval(),
            hmonnx_dir / f"{role}.onnx",
            export_inputs,
            input_names,
            output_names,
            args.quant_type,
        )

    _write_provenance(args, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lerobot-src", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--bundle-root",
        default=None,
        help="Bundle root receiving LeRobot metadata (default: two levels above --output-dir)",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--quant-type", default="w8a8h1_sefp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "roles",
        nargs="*",
        choices=("vision", "prefill", "action_in_proj", "time_mlp", "decode", "action_out_proj"),
    )
    export_hmm(parser.parse_args())


if __name__ == "__main__":
    main()
