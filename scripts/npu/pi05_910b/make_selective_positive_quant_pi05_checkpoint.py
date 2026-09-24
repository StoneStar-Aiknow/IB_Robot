#!/usr/bin/env python
"""构造 PI0.5 正收益层选择性 W8A8 no-smooth checkpoint。

生成的 runtime checkpoint 继承 BF16 源配置、处理器和 tokenizer 资源。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

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
    "dit_mlp_gate_up": 0,
    "vision_mlp_fc2": 27,
}
_SIDECAR_EXCLUDES = {".cache", ".git", "config.json", "model.safetensors"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从 BF16 与量化权重源构造已验证正收益层 checkpoint：ViT mlp.fc2、"
            "Prefix MLP gate/up/down 与 self_attn.o_proj；DiT 保持 BF16。"
        )
    )
    parser.add_argument("--fp-model-path", type=Path, required=True, help="BF16 checkpoint 目录。")
    parser.add_argument(
        "--quant-model-path",
        type=Path,
        required=True,
        help="包含选中层 qweight/weight_scale 的量化权重源目录。",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="新 checkpoint 输出目录，必须不存在。")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        required=True,
        help="正收益层判定表，用于记录选择依据与可追溯性。",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module_from_key(key: str, suffix: str) -> str:
    return key[: -len(suffix)]


def load_rtn_source_metadata(quant_model_path: Path, quant_model_sha256: str) -> tuple[Path, dict[str, object]]:
    metadata_path = quant_model_path / "rtn_source_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Selective-99 runtime checkpoint requires RTN source metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_sha256 = metadata.get("output_sha256")
    if expected_sha256 != quant_model_sha256:
        raise ValueError(f"RTN source hash mismatch: metadata={expected_sha256}, actual={quant_model_sha256}")
    expected_count = sum(EXPECTED_COUNTS.values())
    if metadata.get("selected_module_count") != expected_count:
        raise ValueError(
            "RTN source metadata has unexpected selected_module_count: "
            f"expected {expected_count}, got {metadata.get('selected_module_count')}"
        )
    method = str(metadata.get("method", ""))
    if "rtn" not in method.lower():
        raise ValueError(f"unexpected method in {metadata_path}: {method!r}")
    return metadata_path, metadata


def copy_model_sidecars(fp_model_path: Path, output_dir: Path) -> None:
    """Copy public checkpoint sidecars, including nested tokenizer assets."""

    for path in fp_model_path.iterdir():
        if path.name in _SIDECAR_EXCLUDES:
            continue
        destination = output_dir / path.name
        if path.is_dir():
            shutil.copytree(path, destination)
        elif path.is_file():
            shutil.copy2(path, destination)


def main() -> None:
    args = parse_args()
    fp_model_path = args.fp_model_path.resolve()
    quant_model_path = args.quant_model_path.resolve()
    output_dir = args.output_dir.resolve()
    source_manifest = args.source_manifest.resolve()
    required_inputs = (
        fp_model_path / "config.json",
        fp_model_path / "model.safetensors",
        quant_model_path / "model.safetensors",
        source_manifest,
    )
    for required_input in required_inputs:
        if not required_input.is_file():
            raise FileNotFoundError(required_input)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    quant_model_sha256 = sha256(quant_model_path / "model.safetensors")
    rtn_source = load_rtn_source_metadata(quant_model_path, quant_model_sha256)
    output_dir.mkdir(parents=True)

    copy_model_sidecars(fp_model_path, output_dir)

    config = json.loads((fp_model_path / "config.json").read_text(encoding="utf-8"))
    tokenizer_max_length = config.get("tokenizer_max_length")
    if isinstance(tokenizer_max_length, bool) or not isinstance(tokenizer_max_length, int) or tokenizer_max_length < 1:
        raise ValueError(
            f"BF16 config must define a positive integer tokenizer_max_length, got {tokenizer_max_length!r}"
        )
    config["quantization"] = {
        "quant_method": "int8_w8a8",
        "w_bits": 8,
        "a_bits": 8,
        "w_format": "int",
        "a_format": "int",
        "smooth": False,
        "include_regex": INCLUDE_REGEX,
        "exclude_regex": EXCLUDE_REGEX,
        "group_size": 0,
    }
    # 保留 BF16 源 checkpoint 的最大 token 能力。实际 token length 由
    # tokenizer 和请求文本共同决定，量化阶段不修改该上限。
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    include = re.compile(INCLUDE_REGEX)
    exclude = re.compile(EXCLUDE_REGEX)
    fp_file = fp_model_path / "model.safetensors"
    quant_file = quant_model_path / "model.safetensors"
    tensors = {}
    selected_modules: list[str] = []
    loaded_quant_tensor_count = 0
    with (
        safe_open(fp_file, framework="pt", device="cpu") as fp,
        safe_open(quant_file, framework="pt", device="cpu") as quant,
    ):
        fp_keys = list(fp.keys())
        quant_keys = set(quant.keys())
        selected = {
            module_from_key(key, ".weight")
            for key in fp_keys
            if key.endswith(".weight")
            and include.search(module_from_key(key, ".weight")) is not None
            and exclude.search(module_from_key(key, ".weight")) is None
        }
        selected_modules = sorted(selected)
        for module_name in selected_modules:
            for suffix in ("qweight", "weight_scale"):
                qkey = f"{module_name}.{suffix}"
                if qkey not in quant_keys:
                    raise KeyError(f"missing quant tensor: {qkey}")

        for key in fp_keys:
            if key.endswith(".weight"):
                module_name = module_from_key(key, ".weight")
                if module_name in selected:
                    for suffix in ("qweight", "weight_scale"):
                        qkey = f"{module_name}.{suffix}"
                        tensors[qkey] = quant.get_tensor(qkey)
                        loaded_quant_tensor_count += 1
                    continue
            if key.endswith(".bias"):
                module_name = module_from_key(key, ".bias")
                if module_name in selected and key in quant_keys:
                    tensors[key] = quant.get_tensor(key)
                    loaded_quant_tensor_count += 1
                    continue
            tensors[key] = fp.get_tensor(key)

    save_file(tensors, output_dir / "model.safetensors", metadata={"format": "pt"})
    counts = {
        "llm_oproj": sum("language_model" in name and ".self_attn.o_proj" in name for name in selected_modules),
        "llm_mlp_gate_up_down": sum("language_model" in name and ".mlp." in name for name in selected_modules),
        "dit_mlp_gate_up": sum("gemma_expert" in name and ".mlp." in name for name in selected_modules),
        "vision_mlp_fc2": sum("vision_tower" in name and name.endswith(".mlp.fc2") for name in selected_modules),
    }
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"unexpected selected module counts: expected {EXPECTED_COUNTS}, got {counts}")

    quant_source_metadata_path, quant_source_metadata = rtn_source
    quant_source_method = str(quant_source_metadata["method"])
    quant_source_metadata_sha256 = sha256(quant_source_metadata_path)
    accuracy_validation_warning = (
        "Uses self-described no-smooth per-output-channel RTN weights. The export is weight-only; "
        "task-level accuracy must still be validated separately."
    )

    metadata = {
        "fp_model_path": str(fp_model_path),
        "quant_model_path": str(quant_model_path),
        "source_manifest": str(source_manifest),
        "selection_rule": "ViT mlp.fc2 plus Prefix mlp.gate/up/down and self_attn.o_proj; DiT remains BF16",
        "runtime_smooth": False,
        "quant_source_method": quant_source_method,
        "quant_source_metadata": str(quant_source_metadata_path),
        "quant_source_metadata_sha256": quant_source_metadata_sha256,
        "tokenizer_max_length": tokenizer_max_length,
        "tokenizer_max_length_source": "fp_model_config",
        "latency_only_warning": accuracy_validation_warning,
        "selected_linear_count": len(selected_modules),
        "expected_active_projection_count_after_gate_up_fusion": len(selected_modules) - 18,
        "selected_counts": counts,
        "loaded_quant_tensor_count": loaded_quant_tensor_count,
        "include_regex": INCLUDE_REGEX,
        "exclude_regex": EXCLUDE_REGEX,
        "selected_modules": selected_modules,
        "source_sha256": {
            "fp_model": sha256(fp_file),
            "quant_model": quant_model_sha256,
            "source_manifest": sha256(source_manifest),
        },
        "output_sha256": sha256(output_dir / "model.safetensors"),
    }
    (output_dir / "selective_quant_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary_keys = (
        "quant_source_method",
        "selected_linear_count",
        "expected_active_projection_count_after_gate_up_fusion",
        "selected_counts",
        "loaded_quant_tensor_count",
        "tokenizer_max_length",
        "output_sha256",
    )
    print(json.dumps({key: metadata[key] for key in summary_keys}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
