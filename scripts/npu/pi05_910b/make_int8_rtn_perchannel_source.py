#!/usr/bin/env python
"""从 BF16 PI0.5 checkpoint 生成 selective-99 per-channel RTN INT8 权重源。

第一阶段输出只包含 99 个已选 Linear 的 ``qweight`` 和 ``weight_scale``，供
``make_selective_positive_quant_pi05_checkpoint.py`` 组装完整运行时 checkpoint。
权重按输出通道做对称 RTN；激活不在此处校准，运行时仍使用 dynamic per-token
absmax/127。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LLM_PATTERN = (
    r"(?:.*paligemma_with_expert\.paligemma\.model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.o_proj|mlp\.(?:gate_proj|up_proj|down_proj))$)"
)
VISION_PATTERN = (
    r"(?:.*paligemma_with_expert\.paligemma\.model\.vision_tower\.vision_model\.encoder\."
    r"layers\.\d+\.mlp\.fc2$)"
)
INCLUDE_REGEX = f"(?:{LLM_PATTERN})|(?:{VISION_PATTERN})"
EXCLUDE_REGEX = r"(?:^|\.)(embeddings|embed_tokens|norm|layernorm|lm_head)(?:\.|$)"
EXPECTED_COUNTS = {
    "llm_oproj": 18,
    "llm_mlp_gate_up_down": 54,
    "vision_mlp_fc2": 27,
}
EXPECTED_SELECTED_LINEAR_COUNT = 99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fp-model-path",
        type=Path,
        required=True,
        help="BF16 PI0.5 checkpoint 目录；必须包含 model.safetensors。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="第一阶段 RTN 权重源输出目录；必须不存在。",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module_group(module_name: str) -> str:
    if "vision_tower" in module_name and module_name.endswith(".mlp.fc2"):
        return "vision_mlp_fc2"
    if "language_model" in module_name and ".self_attn.o_proj" in module_name:
        return "llm_oproj"
    if "language_model" in module_name and ".mlp." in module_name:
        return "llm_mlp_gate_up_down"
    raise ValueError(f"selected module has no known group: {module_name}")


def main() -> None:
    args = parse_args()
    fp_model_path = args.fp_model_path.resolve()
    fp_file = fp_model_path / "model.safetensors"
    output_dir = args.output_dir.resolve()
    if not fp_file.is_file():
        raise FileNotFoundError(fp_file)
    if output_dir.exists():
        raise FileExistsError(output_dir)

    include = re.compile(INCLUDE_REGEX)
    exclude = re.compile(EXCLUDE_REGEX)
    tensors: dict[str, torch.Tensor] = {}
    error_report: dict[str, dict[str, int | float]] = {}
    with safe_open(fp_file, framework="pt", device="cpu") as fp:
        keys = list(fp.keys())
        selected_modules = sorted(
            key[: -len(".weight")]
            for key in keys
            if key.endswith(".weight")
            and include.search(key[: -len(".weight")]) is not None
            and exclude.search(key[: -len(".weight")]) is None
        )
        counts = {
            group: sum(module_group(module_name) == group for module_name in selected_modules)
            for group in EXPECTED_COUNTS
        }
        if len(selected_modules) != EXPECTED_SELECTED_LINEAR_COUNT or counts != EXPECTED_COUNTS:
            raise ValueError(
                "unexpected selected modules: "
                f"expected count={EXPECTED_SELECTED_LINEAR_COUNT}, groups={EXPECTED_COUNTS}; "
                f"got count={len(selected_modules)}, groups={counts}"
            )

        for module_name in selected_modules:
            weight = fp.get_tensor(f"{module_name}.weight").to(torch.float32)
            if not torch.isfinite(weight).all():
                raise ValueError(f"non-finite weight tensor: {module_name}.weight")

            # 每个输出行独立确定步长，避免被其他输出通道的离群值放大量化误差。
            weight_scale = (weight.abs().amax(dim=1) / 127.0).clamp_min(1e-8)
            qweight = torch.round(weight / weight_scale[:, None]).clamp(-127, 127).to(torch.int8)
            tensors[f"{module_name}.qweight"] = qweight
            tensors[f"{module_name}.weight_scale"] = weight_scale

            # round-to-nearest 的逐元素误差上界应为本输出通道量化步长的一半。
            error = (qweight.to(torch.float32) * weight_scale[:, None] - weight).abs()
            max_err_over_scale = (error.amax(dim=1) / weight_scale).amax().item()
            if max_err_over_scale > 0.5 + 1e-3:
                raise AssertionError(f"{module_name}: RTN error ratio {max_err_over_scale} exceeds scale/2 bound")
            error_report[module_name] = {
                "out_features": weight.shape[0],
                "in_features": weight.shape[1],
                "max_err_over_scale": round(max_err_over_scale, 6),
            }

    output_dir.mkdir(parents=True)
    output_file = output_dir / "model.safetensors"
    save_file(tensors, output_file, metadata={"format": "pt"})

    manifest_path = output_dir / "selection_provenance.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = ("module_name", "group", "weight_quantization", "activation_quantization")
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for module_name in selected_modules:
            writer.writerow(
                {
                    "module_name": module_name,
                    "group": module_group(module_name),
                    "weight_quantization": "per-output-channel symmetric RTN INT8",
                    "activation_quantization": "runtime dynamic per-token INT8",
                }
            )

    worst_module, worst_error = max(error_report.items(), key=lambda item: item[1]["max_err_over_scale"])
    metadata = {
        "method": "per-output-channel symmetric RTN INT8 (weight-only, no smooth, no calibration)",
        "convention": "scale[o]=max(absmax(W[o,:])/127,1e-8) fp32; q=round(W/scale).clamp(-127,127)",
        "activation_quantization": "runtime dynamic per-token symmetric absmax/127",
        "fp_model_path": str(fp_model_path),
        "fp_source_sha256": sha256(fp_file),
        "selection_rule": (
            "ViT mlp.fc2 plus Prefix LLM mlp.gate/up/down and self_attn.o_proj; action expert remains BF16"
        ),
        "include_regex": INCLUDE_REGEX,
        "exclude_regex": EXCLUDE_REGEX,
        "selected_module_count": len(selected_modules),
        "expected_active_projection_count_after_gate_up_fusion": len(selected_modules) - 18,
        "selected_counts": counts,
        "selected_modules": selected_modules,
        "selection_manifest": manifest_path.name,
        "worst_error": {"module_name": worst_module, **worst_error},
        "per_module_error": error_report,
        "output_sha256": sha256(output_file),
    }
    (output_dir / "rtn_source_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_module_count": metadata["selected_module_count"],
                "selected_counts": counts,
                "worst_error": metadata["worst_error"],
                "output_sha256": metadata["output_sha256"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
