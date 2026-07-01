# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Export PI05 Action Expert to ONNX and emit the Ascend OM manifest entry.

The Action Expert takes KV cache (from the VLM part) + time + noise,
and performs a single Euler denoising step to produce actions.

Note: Unlike PI0, PI05's action expert does NOT take state as input.
      It uses adaRMS conditioning from time embeddings instead.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import onnx
import torch
import torch.onnx

try:
    import torch_npu
    import torch_npu.onnx

    torch.npu.set_compile_mode(jit_compile=False)
except ImportError:
    torch_npu = None


from model_utils.pi05_export._cli_ui import build_onnx_suffix, setup_logging
from model_utils.pi05_export.ascend_export_patches import (
    ascend_onnx_export_patches,
    downgrade_ir_version,
    downgrade_split_for_atc,
    sanitize_nan_initializers,
)
from model_utils.pi05_export.modeling_pi05_action_expert import PI05ActionExpertPolicy
from model_utils.pi05_export.om_manifest import upsert_pi05_om_manifest

LOGGER = logging.getLogger(__name__)


def _build_onnx_config_suffix(opset: int, dynamo: bool, dtype: str = "fp16", device: str = "cpu") -> str:
    """Deprecated thin wrapper kept for backward compatibility.

    The filename convention now lives in one place (``_cli_ui.build_onnx_suffix``)
    so the pipeline orchestrator's predicted skip/resume paths can never drift
    from what this exporter writes.
    """
    return build_onnx_suffix(opset=opset, dynamo=dynamo, dtype=dtype, device=device)


def _iter_graph_tensors(model_proto):
    """Yield every ``TensorProto`` in a model graph.

    Covers ``graph.initializer`` **and** constant tensors stored in node
    attributes (e.g. scales / sizes used by Resize nodes produced by the
    dynamo exporter).
    """
    yield from model_proto.graph.initializer
    for node in model_proto.graph.node:
        for attr in node.attribute:
            if attr.t.ByteSize() > 0:
                yield attr.t
            yield from attr.tensors


def _clean_onnx_domains(onnx_path: Path) -> None:
    """Remove extra opset_import entries, clear node domain fields,
    and consolidate all external data into a single ``.data`` file.

    This prevents ATC / other downstream tools from failing on unrecognised
    custom domains (e.g. ``pkg.onnxscript.torch_lib``).
    """
    onnx_dir = onnx_path.parent
    consolidated_data_name = onnx_path.name + ".data"

    # --- Step 1: Discover old external-data file paths ---
    # Must load WITHOUT external data: onnx.load() with the default
    # load_external_data=True reads tensor bytes into raw_data and then
    # *clears* the external_data entries, making old file locations
    # unrecoverable.
    model_meta = onnx.load(str(onnx_path), load_external_data=False)
    old_external_files: set[Path] = set()
    for tensor in _iter_graph_tensors(model_meta):
        for entry in tensor.external_data:
            if entry.key == "location":
                candidate = (onnx_dir / entry.value).resolve()
                if candidate.is_file():
                    old_external_files.add(candidate)
    del model_meta

    # --- Step 2: Load WITH external data (bytes go into raw_data) ---
    model = onnx.load(str(onnx_path))

    # --- Step 3: Clean opset domains ---
    # ATC's onnx plugin libops_all_onnx_plugin.so registers each NPUxxx op_type
    # in the default "ai.onnx" domain (per opset) AND in the "npu"::1 domain:
    #     NPURotaryMul / NPURmsNorm   : ai.onnx 11..18   (+ npu::1)
    #     NPUPromptFlashAttention     : ai.onnx 11..16,19 (+ npu::1)
    # RoPE/RMSNorm work fine via the default-domain parser, so we strip their
    # node domain to default (single ai.onnx::16 opset_import covers them).
    #
    # --- Step 3: Clean opset domains (strip ALL node domains to default) ---
    # ATC's onnx plugin libops_all_onnx_plugin.so registers each NPUxxx op_type
    # in the default "ai.onnx" domain per opset:
    #     NPURotaryMul / NPURmsNorm   : ai.onnx 11..18
    # We export at opset 16 and strip every node to the default domain so a
    # SINGLE opset_import ("", 16) covers them all (ATC allows exactly one
    # domain_version entry).  NOTE: flash attention is NOT used — it is
    # unworkable on 310P + ATC (see "NPU 算子替换总结.md"); only RoPE + RMSNorm
    # are routed to npu fused operators, and both live in the default domain.
    while len(model.opset_import) > 1:
        model.opset_import.pop()

    for node in model.graph.node:
        if node.HasField("domain"):
            node.ClearField("domain")

    # --- Step 3b: Sanitize NaN values in initializers ---
    sanitize_nan_initializers(model)

    # Opset-18 Split uses a num_outputs attribute that ATC cannot parse.
    downgrade_split_for_atc(model)

    # ORT ≤ 1.18 only supports IR version ≤ 9; dynamo export may produce 10.
    downgrade_ir_version(model)

    # --- Step 4: Delete old external data files BEFORE saving ---
    # onnx.save_model opens the data file in append mode ("ab"); if the
    # target file already exists the tensor bytes are appended instead of
    # overwritten, doubling the file size and producing wrong offsets.
    onnx_resolved = onnx_path.resolve()
    removed = 0
    for old_file in sorted(old_external_files):
        if old_file == onnx_resolved:
            continue
        try:
            old_file.unlink()
            removed += 1
        except OSError as exc:
            LOGGER.warning("Failed to remove old external data file %s: %s", old_file, exc)
    if removed:
        LOGGER.info("Removed %d stale external-data file(s)", removed)

    # --- Step 5: Save with consolidated external data ---
    # Use size_threshold=1024 so that small constant tensors (e.g. Resize
    # scales/sizes produced by the dynamo exporter) stay inline in the
    # protobuf.  ORT's shape inference engine cannot read external-data
    # tensors and will raise ShapeInferenceError if these are externalised.
    onnx.save_model(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=consolidated_data_name,
        size_threshold=1024,
    )
    LOGGER.info("Cleaned ONNX domains and re-saved to %s", onnx_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PI05 Action Expert to ONNX.")
    parser.add_argument(
        "--policy-path",
        "--pretrained-policy-path",
        dest="pretrained_policy_path",
        type=str,
        required=True,
        help="Path to the pretrained PI05 policy (contains config/model). Alias: --pretrained-policy-path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX file path (auto-generated if omitted).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/onnx", help="Directory for auto-generated output filename"
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument(
        "--past-kv-path",
        type=str,
        default="runtime_save/past_kv_tensor.pth",
        help="Path to past_kv_tensor checkpoint (produced by VLM export).",
    )
    parser.add_argument(
        "--prefix-pad-masks-path",
        type=str,
        default="runtime_save/prefix_pad_masks.pth",
        help="Path to prefix_pad_masks checkpoint (produced by VLM export).",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda:0")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for dummy inputs.")
    parser.add_argument(
        "--chunk-size", type=int, default=None, help="Action chunk size. If omitted, read from model config."
    )
    parser.add_argument(
        "--max-action-dim", type=int, default=None, help="Max action dimension. If omitted, read from model config."
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Number of denoising steps. If omitted, read from model config.",
    )
    parser.add_argument(
        "--dynamo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use torch dynamo export (default: False)",
    )
    parser.add_argument(
        "--constant-folding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable constant folding (default: True)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "fp32", "auto"],
        default="fp16",
        help="Export precision: fp16 (default, model.half()), fp32 (full precision), "
        "or auto (preserve original mixed bf16+fp32 weights). "
        "Must match the dtype used for VLM export.",
    )
    parser.add_argument(
        "--om-manifest-dir",
        type=str,
        default=None,
        help="Directory to write config.om.json (default: pretrained policy path).",
    )
    parser.add_argument(
        "--om-path",
        type=str,
        default=None,
        help="Predicted action_expert .om artifact path recorded in the manifest (default: <onnx-basename>.om).",
    )
    parser.add_argument("--skip-om-manifest", action="store_true", help="Do not write/update config.om.json.")
    parser.add_argument(
        "--fast-gelu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Ascend NPUFastGelu for gelu_pytorch_tanh during NPU export (default: False). "
        "This is faster but not numerically identical to PyTorch tanh GELU.",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--local-files-only", action="store_true", default=True, help="Load policy without network")
    return parser.parse_args()


def build_inputs(
    device: str,
    batch_size: int,
    chunk_size: int,
    max_action_dim: int,
    num_inference_steps: int,
    *,
    past_kv_path: str,
    prefix_pad_masks_path: str,
    float_dtype: torch.dtype | dict[str, torch.dtype] = torch.float16,
):
    """Build dummy inputs for PI05 Action Expert.

    Unlike PI0, PI05's action expert does NOT take state as input.
    It only needs: past_kv_tensor, prefix_pad_masks, time, noise.

    ``float_dtype`` may be a single dtype applied to all float tensors, or a
    dict mapping input name -> dtype (used by ``auto`` mode where the model
    contains mixed bf16 / fp32 weights).
    """

    def _dtype_for(key: str) -> torch.dtype:
        if isinstance(float_dtype, dict):
            return float_dtype.get(key, torch.float32)
        return float_dtype

    # Runtime tensors from VLM export
    past_kv_tensor = torch.load(past_kv_path, map_location=device)
    prefix_pad_masks = torch.load(prefix_pad_masks_path, map_location=device)

    # The saved VLM tensors carry the batch dimension produced during VLM
    # export. Reuse it as the authoritative batch size so we never mix a
    # batch-N cache with batch-M time/noise inputs (which would either fail to
    # trace or bake a wrong shape assumption into the exported ONNX graph).
    #
    # Tensor layouts (see the VLM export wrapper output):
    #   past_kv_tensor   : (num_layers, 2, batch, num_kv_heads, seq, head_dim)
    #   prefix_pad_masks : (batch, seq)
    # so the batch axis is index 2 for the KV cache and index 0 for the mask.
    PAST_KV_BATCH_AXIS = 2
    actual_batch = int(prefix_pad_masks.shape[0])
    past_kv_batch = int(past_kv_tensor.shape[PAST_KV_BATCH_AXIS])
    if past_kv_batch != actual_batch:
        raise ValueError(
            "past_kv_tensor and prefix_pad_masks have mismatched batch sizes "
            f"({past_kv_batch} vs {actual_batch}); re-run the VLM export to "
            "regenerate consistent runtime tensors"
        )
    if actual_batch != batch_size:
        raise ValueError(
            f"--batch-size={batch_size} does not match the saved VLM tensors batch={actual_batch}; "
            "pass --batch-size matching the VLM export or re-run the VLM export with the desired batch"
        )

    # Cast KV cache to the target dtype (it inherits dtype from VLM export)
    past_kv_tensor = past_kv_tensor.to(_dtype_for("past_kv_tensor"))

    # Time: start from 1.0 for the first denoising step
    time = torch.tensor(1.0, dtype=_dtype_for("time"), device=device)
    time = time.view(1).repeat(actual_batch)

    # Noise: zero noise for deterministic inference (matches pi05 sample_noise)
    noise = torch.zeros((actual_batch, chunk_size, max_action_dim), dtype=_dtype_for("noise"), device=device)

    observation = {
        "past_kv_tensor": past_kv_tensor,
        "prefix_pad_masks": prefix_pad_masks,
        "time": time,
        "noise": noise,
    }
    return observation


class ONNXWrapper(torch.nn.Module):
    def __init__(self, policy, observation, float_dtype: torch.dtype | dict[str, torch.dtype] = torch.float16):
        super().__init__()
        self.policy = policy
        self.observation = observation
        self._keys = list(observation.keys())
        self._float_dtype = float_dtype

    def _dtype_for(self, key: str) -> torch.dtype:
        if isinstance(self._float_dtype, dict):
            return self._float_dtype.get(key, torch.float32)
        return self._float_dtype

    def forward(self, *args):
        """Map positional args (same order as `self._keys`) to policy input dict with expected dtypes."""
        if len(args) != len(self._keys):
            raise ValueError(f"Expected {len(self._keys)} inputs, got {len(args)}")

        input_dict = {}
        for key, tensor in zip(self._keys, args, strict=False):
            if key == "prefix_pad_masks":
                input_dict[key] = tensor.to(torch.bool)
            else:
                input_dict[key] = tensor.to(self._dtype_for(key))

        with torch.no_grad():
            actions = self.policy.select_action(input_dict)
            return actions


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    device = args.device
    # NPU-affine fused ops (RoPE, etc.) are used automatically when exporting
    # on an NPU device; cpu/cuda exports keep the ORT-runnable fallbacks.
    use_npu_ops = str(device).startswith("npu")

    past_kv_path = Path(args.past_kv_path).expanduser()
    prefix_pad_masks_path = Path(args.prefix_pad_masks_path).expanduser()

    LOGGER.info("Loading PI05ActionExpertPolicy from %s", args.pretrained_policy_path)
    policy = PI05ActionExpertPolicy.from_pretrained(
        args.pretrained_policy_path, local_files_only=bool(args.local_files_only), strict=False
    )
    export_dtype = args.dtype  # "fp16", "fp32", or "auto"
    softmax_in_model_dtype = export_dtype != "fp32"
    if export_dtype == "auto":
        # Preserve original mixed precision (typically bf16 for the gemma
        # expert + fp32 for action/time projection layers). Build a per-input
        # dtype map by querying the layer that actually consumes each input,
        # so dummy tensors match weight dtypes during tracing.
        model = policy.model

        def _dtype_of(module) -> torch.dtype:
            return next(module.parameters()).dtype

        # past_kv_tensor flows into the gemma_expert decoder layers
        kv_dtype = _dtype_of(model.paligemma_with_expert.gemma_expert.model)
        # noise is consumed by action_in_proj; time feeds time_mlp_in (its
        # sinusoidal embedding is later cast to time tensor's dtype)
        noise_dtype = _dtype_of(model.action_in_proj)
        time_dtype = _dtype_of(model.time_mlp_in)

        float_dtype = {
            "past_kv_tensor": kv_dtype,
            "noise": noise_dtype,
            "time": time_dtype,
        }
        LOGGER.info("Export dtype: auto — keeping original model weights unchanged")
        LOGGER.info(
            "Per-input dtypes: past_kv_tensor=%s, noise=%s, time=%s",
            kv_dtype,
            noise_dtype,
            time_dtype,
        )
        LOGGER.warning(
            "auto mode produces bf16 tensors; ATC on Ascend 310/310P may fail — use 910 series for bf16 OM conversion."
        )
    else:
        float_dtype = torch.float16 if export_dtype == "fp16" else torch.float32
        # Normalize to float32 first — pretrained models may contain bfloat16
        # weights (from mixed-precision training), which are unsupported by
        # some ONNX runtimes and Ascend 310/310P.
        policy = policy.float()
        if export_dtype == "fp16":
            policy.model = policy.model.half()
        LOGGER.info("Export dtype: %s", export_dtype)
    policy.to(device)

    # --- 自动从 config 读取参数，允许 CLI 覆盖 ---
    cfg = policy.config
    chunk_size = args.chunk_size if args.chunk_size is not None else cfg.chunk_size
    max_action_dim = args.max_action_dim if args.max_action_dim is not None else cfg.max_action_dim
    num_inference_steps = args.num_inference_steps if args.num_inference_steps is not None else cfg.num_inference_steps
    LOGGER.info(
        "Action expert params (from %s): chunk_size=%d, max_action_dim=%d, num_inference_steps=%d",
        "CLI override" if args.chunk_size is not None else "model config",
        chunk_size,
        max_action_dim,
        num_inference_steps,
    )

    observation = build_inputs(
        device,
        args.batch_size,
        chunk_size,
        max_action_dim,
        num_inference_steps,
        past_kv_path=str(past_kv_path),
        prefix_pad_masks_path=str(prefix_pad_masks_path),
        float_dtype=float_dtype,
    )

    onnx_wrapper = ONNXWrapper(policy, observation, float_dtype=float_dtype)
    onnx_wrapper.policy.eval()

    # torch dynamo exporter 忽略 opset_version 参数, 固定使用 opset 18
    actual_opset = 18 if args.dynamo else args.opset
    # Normalize the device string to its bare type (cpu / cuda / npu) for the filename tag.
    device_tag = torch.device(device).type
    suffix = _build_onnx_config_suffix(actual_opset, args.dynamo, export_dtype, device_tag)
    if args.output is not None:
        base_path = Path(args.output).expanduser().resolve()
    else:
        base_path = Path(args.output_dir).expanduser().resolve() / "pi05-action_expert.onnx"
    onnx_output_path = base_path.with_name(base_path.stem + suffix + ".onnx")
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_keys = list(observation.keys())
    observation_values = []
    for k in dummy_keys:
        v = observation[k]
        if k == "prefix_pad_masks":
            observation_values.append(v.to(torch.bool))
        else:
            target_dtype = float_dtype.get(k, torch.float32) if isinstance(float_dtype, dict) else float_dtype
            observation_values.append(v.to(target_dtype))

    LOGGER.info("Loading past_kv_tensor from %s", past_kv_path)
    LOGGER.info("Loading prefix_pad_masks from %s", prefix_pad_masks_path)
    LOGGER.info("Exporting ONNX to %s", onnx_output_path)
    LOGGER.info(
        "  opset=%d (requested=%d), dynamo=%s, constant_folding=%s",
        actual_opset,
        args.opset,
        args.dynamo,
        args.constant_folding,
    )
    LOGGER.info(
        "  attention export: mqa_broadcast=True, softmax_dtype=%s",
        "model" if softmax_in_model_dtype else "fp32",
    )
    # Apply Ascend ATC compatibility patches during ONNX export
    with ascend_onnx_export_patches(
        use_npu_ops=use_npu_ops,
        fp16_softmax=softmax_in_model_dtype,
        mqa_broadcast=True,
        fast_gelu=bool(args.fast_gelu),
    ):
        torch.onnx.export(
            onnx_wrapper,
            tuple(observation_values),
            str(onnx_output_path),
            opset_version=actual_opset,
            verbose=False,
            input_names=dummy_keys,
            output_names=["action"],
            do_constant_folding=bool(args.constant_folding),
            dynamo=bool(args.dynamo),
            external_data=True,
        )

    _clean_onnx_domains(onnx_output_path)
    LOGGER.info("ONNX export finished")

    if not args.skip_om_manifest:
        if args.om_manifest_dir is not None:
            manifest_dir = Path(args.om_manifest_dir).expanduser().resolve()
        else:
            local_policy_path = Path(args.pretrained_policy_path).expanduser()
            if not local_policy_path.is_dir():
                raise ValueError(
                    "--om-manifest-dir is required when --pretrained-policy-path is not a local "
                    f"policy directory (got {args.pretrained_policy_path!r}); otherwise the manifest "
                    "would be written to a wrong location relative to the current working directory"
                )
            manifest_dir = local_policy_path.resolve()
        om_path = args.om_path if args.om_path is not None else onnx_output_path.with_suffix(".om").name
        manifest_path = upsert_pi05_om_manifest(manifest_dir, "action_expert", om_path)
        LOGGER.info("Updated OM manifest (action_expert) at %s", manifest_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
