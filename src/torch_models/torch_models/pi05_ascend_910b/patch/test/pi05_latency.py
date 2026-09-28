#!/usr/bin/env python
"""按需测量 PI0.5 NPU 完整 E2E 或纯模型推理时延。"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

DEFAULT_TOKEN_LENGTH = 183


class TeeStream:
    def __init__(self, primary: TextIO, log_file: TextIO) -> None:
        self.primary = primary
        self.log_file = log_file

    def write(self, value: str) -> int:
        self.primary.write(value)
        self.log_file.write(value)
        return len(value)

    def flush(self) -> None:
        self.primary.flush()
        self.log_file.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.primary, name)


@dataclass(frozen=True)
class OptimizationProfile:
    """将内部优化收敛为图编译、量化和降采样三个公开开关。"""

    graph_compile: bool = True
    quantization: bool = False
    downsample: bool = False

    def __post_init__(self) -> None:
        if not self.graph_compile and (self.quantization or self.downsample):
            raise ValueError("量化和降采样只能在图编译全优化路径中使用")

    @property
    def denoise_solver(self) -> str:
        return "ab2" if self.downsample else "euler"

    @property
    def inference_path(self) -> str:
        return "_".join(
            (
                "pi05",
                "int8" if self.quantization else "bf16",
                "torchair" if self.graph_compile else "eager",
                "ab2_6" if self.downsample else "euler_10",
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PI0.5 NPU 时延测试：选择包含 processor/tokenizer/postprocessor 的完整 E2E，"
            "或使用预构造 NPU tensor 的纯模型推理模式。"
        )
    )
    parser.add_argument("--device", required=True, help="必须显式指定 NPU，例如 npu:0。")
    parser.add_argument(
        "--bf16-model-path",
        type=Path,
        help="BF16 checkpoint；关闭量化时必填。",
    )
    parser.add_argument(
        "--int8-model-path",
        type=Path,
        help="完整 Selective-99 INT8 runtime checkpoint；开启量化时必填。",
    )
    parser.add_argument(
        "--test-mode",
        choices=("e2e", "model"),
        required=True,
        help="e2e 包含前后处理且要求 Tokenizer；model 只测模型推理且不需要 Tokenizer。",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="Tokenizer 目录；仅 --test-mode e2e 时必填。",
    )
    parser.add_argument("--token-length", type=int, default=DEFAULT_TOKEN_LENGTH)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--task-text", default="pick up the object")
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", "--iters", dest="iterations", type=int, default=100)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--graph-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="开启 TorchAir 双图和固定的全部正收益优化；默认开启。",
    )
    parser.add_argument(
        "--quantization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="使用 Selective-99 no-smooth INT8 checkpoint；依赖图编译。",
    )
    parser.add_argument(
        "--downsample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="将 Euler/10 替换为 AB2/6；依赖图编译。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="检查配置文件、Tokenizer 路径、设备和 CANN 前置条件；不检查或加载模型权重，也不构造输入。",
    )
    args = parser.parse_args()
    try:
        args.profile = OptimizationProfile(
            graph_compile=args.graph_compile,
            quantization=args.quantization,
            downsample=args.downsample,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def resolve_lerobot_root() -> Path:
    candidates: list[Path] = []
    if configured_root := os.environ.get("LEROBOT_REPO"):
        candidates.append(Path(configured_root))
    candidates.append(Path.cwd())

    script_path = Path(__file__).resolve()
    candidates.extend(script_path.parents)
    spec = importlib.util.find_spec("lerobot")
    if spec is not None and spec.origin is not None:
        candidates.append(Path(spec.origin).resolve().parents[2])

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "src" / "lerobot").is_dir():
            return resolved
    raise RuntimeError("找不到 LeRobot 根目录；请设置 LEROBOT_REPO 或从仓库根目录执行")


def configure_import_path(lerobot_root: Path) -> None:
    source_dir = lerobot_root / "src"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))


def create_output_dir(args: argparse.Namespace, lerobot_root: Path) -> Path:
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = lerobot_root / "output" / f"{timestamp}_pi05_inference"
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖已有结果: {output_dir}")
    output_dir.mkdir(parents=True)
    return output_dir


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup < 0:
        raise ValueError("--warmup 必须大于或等于 0")
    if args.graph_compile and args.warmup < 1:
        raise ValueError("图编译路径至少需要 1 次 warmup 来触发编译")
    if args.iterations < 1:
        raise ValueError("--iterations 必须大于或等于 1")
    if args.token_length < 1:
        raise ValueError("--token-length 必须大于或等于 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size 必须大于或等于 1")
    if not args.device.startswith("npu:"):
        raise ValueError("--device 必须显式指定为 npu:<index>，例如 npu:0")
    if args.quantization:
        if args.int8_model_path is None:
            raise ValueError("开启 --quantization 时必须传入 --int8-model-path")
    elif args.bf16_model_path is None:
        raise ValueError("关闭 --quantization 时必须传入 --bf16-model-path")
    if args.test_mode == "e2e" and args.tokenizer_path is None:
        raise ValueError("--test-mode e2e 必须传入 --tokenizer-path")


def resolve_model_path(args: argparse.Namespace) -> Path:
    selected = args.int8_model_path if args.quantization else args.bf16_model_path
    if selected is None:
        raise ValueError("没有可用的模型路径")
    model_path = selected.expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"模型目录缺少 config.json: {model_path}")
    if not args.dry_run and not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"模型目录缺少 model.safetensors: {model_path}")
    return model_path


def resolve_tokenizer_path(args: argparse.Namespace) -> Path | None:
    if args.test_mode != "e2e":
        return None
    if args.tokenizer_path is None:
        raise ValueError("--test-mode e2e 必须传入 --tokenizer-path")
    tokenizer_path = args.tokenizer_path.expanduser().resolve()
    if not (tokenizer_path / "tokenizer.json").is_file():
        raise FileNotFoundError(f"Tokenizer 目录缺少 tokenizer.json: {tokenizer_path}")
    return tokenizer_path


def configure_environment(profile: OptimizationProfile) -> None:
    os.environ["LEROBOT_PI05_ENABLE_VISION_NPU_PFA"] = "1" if profile.graph_compile else "0"
    os.environ["LEROBOT_PI05_NPU_ATTENTION_BACKEND"] = "hybrid" if profile.graph_compile else "default"
    os.environ["LEROBOT_PI05_QWEIGHT_LAYOUT"] = "nz" if profile.quantization else "current"
    os.environ["LEROBOT_PI05_QWEIGHT_NZ_GROUPS"] = "hot" if profile.quantization else "all"


def configure_device(device_name: str, graph_compile: bool):
    import torch
    import torch_npu  # noqa: F401

    device = torch.device(device_name)
    if device.index is None:
        raise ValueError("--device 必须包含明确的 NPU 索引")
    if not torch.npu.is_available():
        raise RuntimeError("当前环境未检测到可用 Ascend NPU")
    torch.npu.set_device(device)
    if torch.npu.current_device() != device.index:
        raise RuntimeError(f"NPU 当前设备为 {torch.npu.current_device()}，与请求设备 {device.index} 不一致")

    tbe_path = None
    if graph_compile:
        try:
            import tbe
        except ModuleNotFoundError as exc:
            raise RuntimeError("图编译需要 CANN TBE；请先 source CANN set_env.sh") from exc
        tbe_path = getattr(tbe, "__file__", None)
    return torch, device, tbe_path


def load_config(model_path: Path, args: argparse.Namespace, device):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    config = PreTrainedConfig.from_pretrained(model_path, local_files_only=True)
    if not isinstance(config, PI05Config):
        raise TypeError(f"期望 PI05Config，实际为 {type(config).__name__}")

    quantization = getattr(config, "quantization", None)
    if args.quantization:
        if quantization is None:
            raise ValueError("--quantization 要求 checkpoint 包含 quantization 配置")
        if quantization.quant_method != "int8_w8a8" or quantization.smooth:
            raise ValueError("量化路径只支持 quant_method=int8_w8a8 且 smooth=false")
    elif quantization is not None:
        raise ValueError("BF16 路径不能使用带 quantization 配置的 checkpoint")

    if args.token_length > config.tokenizer_max_length:
        raise ValueError(f"--token-length={args.token_length} 超过 checkpoint 上限 {config.tokenizer_max_length}")
    config.tokenizer_max_length = args.token_length
    config.device = str(device)
    config.dtype = "bfloat16"
    config.compile_model = False
    config.gradient_checkpointing = False
    config.compile_inference_graph = args.graph_compile
    config.compile_inference_backend = "torchair"
    config.compile_inference_fullgraph = True
    config.compile_inference_dynamic = False
    config.compile_frozen_parameter = True
    config.compile_tiling_schedule_optimize = True
    config.denoise_solver = args.profile.denoise_solver
    config.shared_prefix_fias = args.graph_compile
    config.adarms_bias_fusion = args.graph_compile
    config.adarms_bias_fusion_stage = "all"
    return config


def make_raw_batch(config, batch_size: int, task_text: str, seed: int):
    import torch
    from lerobot.utils.constants import OBS_STATE

    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw_batch: dict[str, Any] = {}
    real_image_keys = [key for key in config.image_features if ".empty_camera_" not in key]
    if not real_image_keys:
        real_image_keys = list(config.image_features)
    for key in real_image_keys:
        feature = config.image_features[key]
        raw_batch[key] = torch.rand(
            (batch_size, *feature.shape),
            generator=generator,
            dtype=torch.float32,
        )

    state_feature = config.input_features.get(OBS_STATE)
    if state_feature is None:
        raise ValueError("checkpoint 未定义 observation.state")
    raw_batch[OBS_STATE] = torch.zeros((batch_size, *state_feature.shape), dtype=torch.float32)
    raw_batch["task"] = [task_text] * batch_size
    return raw_batch


def make_model_inputs(config, batch_size: int, token_length: int, seed: int, device):
    """构造已在 NPU 上就绪的模型输入，不调用外部前后处理。"""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    image_shape = (batch_size, 3, *config.image_resolution)
    images = []
    image_masks = []
    for key in config.image_features:
        is_padding_camera = ".empty_camera_" in key
        if is_padding_camera:
            image = torch.full(image_shape, -1.0, dtype=torch.float32)
            image_mask = torch.zeros(batch_size, dtype=torch.bool)
        else:
            image = torch.rand(image_shape, generator=generator, dtype=torch.float32) * 2.0 - 1.0
            image_mask = torch.ones(batch_size, dtype=torch.bool)
        images.append(image.to(device=device))
        image_masks.append(image_mask.to(device=device))

    tokens = torch.ones((batch_size, token_length), dtype=torch.long, device=device)
    token_masks = torch.ones((batch_size, token_length), dtype=torch.bool, device=device)
    return images, image_masks, tokens, token_masks


def make_fixed_noise(config, batch_size: int, seed: int, device):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed + 2026)
    noise = torch.randn(
        (batch_size, config.chunk_size, config.max_action_dim),
        generator=generator,
        dtype=torch.float32,
    )
    return noise.to(device=device)


def replace_vit_fast_gelu(policy) -> int:
    import torch
    import torch_npu

    class NPUFastGELU(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return torch_npu.npu_fast_gelu(hidden_states)

    replaced_layers = 0
    for module in policy.model.modules():
        if module.__class__.__name__ in {"PI05SiglipMLP", "SiglipMLP"} and hasattr(module, "activation_fn"):
            module.activation_fn = NPUFastGELU()
            replaced_layers += 1
    if replaced_layers != 27:
        raise AssertionError(f"期望替换 27 层 ViT FastGELU，实际为 {replaced_layers}")
    return replaced_layers


def prepare_quantized_weights(policy) -> dict[str, Any]:
    import torch
    from lerobot.quantization.linear_int8 import Int8W8A8Linear

    def group_name(name: str) -> str | None:
        if "vision_tower" in name and name.endswith(".mlp.fc2"):
            return "vit_fc2"
        if ".language_model." not in name:
            return None
        if name.endswith(".self_attn.o_proj"):
            return "llm_o_proj"
        if name.endswith(".mlp.gate_up"):
            return "llm_gate_up"
        if name.endswith(".mlp.down_proj"):
            return "llm_down"
        return None

    group_counts: dict[str, int] = {}
    module_count = 0
    torch.npu.set_option({"ALLOW_INTERNAL_FORMAT": "enable"})
    try:
        for name, module in policy.model.named_modules():
            if not isinstance(module, Int8W8A8Linear):
                continue
            group = group_name(name)
            if group is None:
                continue
            module.prepare_npu_qweight_layout("nz")
            group_counts[group] = group_counts.get(group, 0) + 1
            module_count += 1
    finally:
        torch.npu.set_option({"ALLOW_INTERNAL_FORMAT": "disable"})
    if module_count == 0:
        raise AssertionError("量化路径没有找到可预排布的 Int8W8A8Linear")
    return {"layout": "nz", "module_count": module_count, "group_counts": group_counts}


def load_policy(model_path: Path, config, args: argparse.Namespace, device):
    import torch
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    policy = PI05Policy.from_pretrained(
        model_path,
        config=config,
        local_files_only=True,
        strict=False,
    )
    policy.to(device)
    policy.eval()
    optimization_info: dict[str, Any] = {"enabled": args.graph_compile}
    if not args.graph_compile:
        return policy, optimization_info

    optimization_info.update(
        policy.prepare_inference_optimizations(
            enable_npu_fused_ops=True,
            enable_graph_compile=True,
            enable_qkv_fusion=True,
            enable_mlp_fusion=args.quantization,
            mlp_fusion_scope="prefix",
            enable_shared_prefix_fias=True,
            enable_adarms_bias_fusion=True,
            adarms_bias_fusion_stage="all",
        )
    )
    replaced_layers = replace_vit_fast_gelu(policy)
    compile_kwargs, _ = policy.model._build_inference_compile_kwargs()
    policy.model._compiled_action_prefix_forward = torch.compile(
        policy.model._action_prefix_forward_for_compile,
        **compile_kwargs,
    )
    optimization_info["vit_fast_gelu"] = {
        "enabled": True,
        "replaced_layers": replaced_layers,
    }
    if args.quantization:
        optimization_info["qweight_layout"] = prepare_quantized_weights(policy)
    policy.eval()
    return policy, optimization_info


def synchronize(torch, device) -> None:
    torch.npu.synchronize(device)


def timed_call(torch, device, function):
    synchronize(torch, device)
    start = time.perf_counter()
    output = function()
    synchronize(torch, device)
    return output, (time.perf_counter() - start) * 1000.0


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[int((len(values) - 1) * fraction)]


def summarize(values: list[float], batch_size: int) -> dict[str, float]:
    mean_ms = statistics.mean(values)
    return {
        "latency_mean_ms": mean_ms,
        "latency_median_ms": statistics.median(values),
        "latency_p90_ms": percentile(values, 0.9),
        "latency_min_ms": min(values),
        "latency_max_ms": max(values),
        "latency_std_ms": statistics.pstdev(values),
        "amortized_latency_mean_ms_per_sample": mean_ms / batch_size,
        "throughput_samples_per_s": 1000.0 * batch_size / mean_ms,
    }


def tensor_to_numpy(output):
    import torch

    output_cpu = output.detach().cpu()
    if output_cpu.dtype is torch.bfloat16:
        output_cpu = output_cpu.float()
    return output_cpu.numpy()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, ensure_ascii=False)
    fieldnames = sorted({key for row in rows for key in row})
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace, lerobot_root: Path, output_dir: Path) -> None:
    validate_args(args)
    configure_import_path(lerobot_root)
    configure_environment(args.profile)
    torch, device, tbe_path = configure_device(args.device, args.graph_compile)

    tokenizer_path = resolve_tokenizer_path(args)
    model_path = resolve_model_path(args)
    config = load_config(model_path, args, device)
    if args.test_mode == "e2e":
        raw_batch = make_raw_batch(config, args.batch_size, args.task_text, args.seed)
        timing_target = "full_e2e"
        input_keys = sorted(raw_batch)
        processor_scope = "processor_and_postprocessor_per_iteration"
    else:
        raw_batch = None
        timing_target = "model_inference"
        input_keys = ["images", "image_masks", "tokens", "token_masks", "noise"]
        processor_scope = "prebuilt_npu_tensors_without_pre_or_postprocessing"
    metadata: dict[str, Any] = {
        "lerobot_root": str(lerobot_root),
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else None,
        "device": str(device),
        "tbe_path": tbe_path,
        "test_mode": args.test_mode,
        "optimization_switches": asdict(args.profile),
        "inference_path": args.profile.inference_path,
        "token_length": args.token_length,
        "batch_size": args.batch_size,
        "task_text": args.task_text if args.test_mode == "e2e" else None,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "input_keys": input_keys,
        "timing_targets": [timing_target],
        "processor_scope": processor_scope,
        "output_dir": str(output_dir),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    if args.dry_run:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    import numpy as np

    policy, optimization_info = load_policy(model_path, config, args, device)
    fixed_noise = make_fixed_noise(config, args.batch_size, args.seed, device)
    if args.test_mode == "e2e":
        from lerobot.policies.factory import make_pre_post_processors

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(model_path),
            preprocessor_overrides={
                "tokenizer_processor": {
                    "tokenizer_name": str(tokenizer_path),
                    "max_length": config.tokenizer_max_length,
                },
                "device_processor": {"device": str(device)},
            },
        )

        def full_e2e():
            processed = preprocessor(raw_batch)
            normalized_actions = policy.predict_action_chunk(processed, noise=fixed_noise)
            return postprocessor(normalized_actions)

        functions = {"full_e2e": full_e2e}
    else:
        images, image_masks, tokens, token_masks = make_model_inputs(
            config,
            args.batch_size,
            args.token_length,
            args.seed,
            device,
        )

        def model_inference():
            return policy.model.sample_actions(
                images,
                image_masks,
                tokens,
                token_masks,
                noise=fixed_noise,
            )

        functions = {"model_inference": model_inference}
    warmup_values: dict[str, list[float]] = {target: [] for target in functions}
    measured_values: dict[str, list[float]] = {target: [] for target in functions}
    outputs: dict[str, Any] = {}
    iteration_records: list[dict[str, Any]] = []

    for iteration in range(args.warmup):
        for target, function in functions.items():
            output, latency_ms = timed_call(torch, device, function)
            outputs[target] = output
            warmup_values[target].append(latency_ms)
            iteration_records.append(
                {
                    "phase": "warmup",
                    "target": target,
                    "iteration": iteration + 1,
                    "latency_ms": latency_ms,
                }
            )

    for iteration in range(args.iterations):
        for target, function in functions.items():
            output, latency_ms = timed_call(torch, device, function)
            outputs[target] = output
            measured_values[target].append(latency_ms)
            iteration_records.append(
                {
                    "phase": "measured",
                    "target": target,
                    "iteration": iteration + 1,
                    "latency_ms": latency_ms,
                }
            )

    summaries: list[dict[str, Any]] = []
    for target in functions:
        output = outputs[target]
        np.save(output_dir / f"actions_{target}.npy", tensor_to_numpy(output))
        row: dict[str, Any] = {
            "target": target,
            "inference_path": args.profile.inference_path,
            "processor_scope": processor_scope,
            "device": str(device),
            "batch_size": args.batch_size,
            "token_length": args.token_length,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "warmup_total_ms": sum(warmup_values[target]),
            "output_shape": list(output.shape),
            "latency_scope": "per_batch",
        }
        row.update(summarize(measured_values[target], args.batch_size))
        summaries.append(row)

    metadata["optimization_info"] = optimization_info
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    write_rows(output_dir / "summary", summaries)
    write_rows(output_dir / "iteration_records", iteration_records)

    print(f"model=pi05 device={device} path={args.profile.inference_path}")
    for row in summaries:
        print(
            f"{row['target']}: mean={row['latency_mean_ms']:.3f} ms "
            f"median={row['latency_median_ms']:.3f} ms "
            f"p90={row['latency_p90_ms']:.3f} ms"
        )
    print(f"results={output_dir}")

    gc.collect()
    torch.npu.empty_cache()


def main() -> int:
    args = parse_args()
    lerobot_root = resolve_lerobot_root()
    output_dir = create_output_dir(args, lerobot_root)
    log_path = output_dir / "pi05_latency.log"
    with (
        log_path.open("w", encoding="utf-8", buffering=1) as log_file,
        redirect_stdout(TeeStream(sys.stdout, log_file)),
        redirect_stderr(TeeStream(sys.stderr, log_file)),
    ):
        print(f"log_file={log_path}")
        try:
            run(args, lerobot_root, output_dir)
        except Exception:
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
