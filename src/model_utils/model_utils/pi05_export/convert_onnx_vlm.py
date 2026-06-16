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
"""Export PI05 VLM to ONNX and emit the Ascend OM manifest entry."""

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

from model_utils.pi05_export._cli_ui import setup_logging
from model_utils.pi05_export.ascend_export_patches import (
    ascend_onnx_export_patches,
    downgrade_ir_version,
    downgrade_split_for_atc,
    normalize_slice_for_atc,
    sanitize_nan_initializers,
)
from model_utils.pi05_export.modeling_pi05_vlm import PI05VLMPolicy
from model_utils.pi05_export.om_manifest import upsert_pi05_om_manifest

LOGGER = logging.getLogger(__name__)


def _build_onnx_config_suffix(opset: int, dynamo: bool, dtype: str = "fp16", device: str = "cpu") -> str:
    """Build config suffix for ONNX filename.

    Example: _op17_nodyn_fp16_cpu
    Abbreviations: op=opset, dyn/nodyn=dynamo, fp16/fp32=dtype, trailing=export device.
    Constant folding is always on by default and therefore not encoded in the name.
    """
    parts = [
        f"op{opset}",
        "dyn" if dynamo else "nodyn",
        dtype,
        device,
    ]
    return "_" + "_".join(parts)


# -----------------
# Helpers
# -----------------


def _normalize_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def _parse_device(device: str) -> torch.device:
    try:
        parsed = torch.device(device)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Invalid --device '{device}'") from exc

    if parsed.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("No CUDA available; falling back to CPU")
        return torch.device("cpu")

    if parsed.type == "cuda" and parsed.index is not None:
        count = torch.cuda.device_count()
        if parsed.index < 0 or parsed.index >= count:
            LOGGER.warning(
                "Requested cuda:%s not available (device_count=%s); falling back to CPU", parsed.index, count
            )
            return torch.device("cpu")

    return parsed


def _move_policy_to_device(policy: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    policy = policy.to(device)
    for module in policy.modules():
        for buffer in module.buffers():
            if buffer is not None and buffer.device != device:
                buffer.data = buffer.data.to(device)
    return policy


def _detect_cameras_from_config(policy: PI05VLMPolicy) -> dict[str, tuple[int, int]]:
    """Auto-detect camera names and resolutions from the policy config.

    Reads ``config.image_features`` (all ``input_features`` with
    ``FeatureType.VISUAL``) and extracts each camera's ``(H, W)`` from the
    feature shape ``(C, H, W)``.

    Returns:
        Ordered dict mapping camera key (e.g. ``observation.images.top``)
        to ``(height, width)``.
    """
    image_features = policy.config.image_features  # dict[str, PolicyFeature]
    if not image_features:
        LOGGER.warning(
            "No VISUAL features found in config.input_features; "
            "falling back to single camera 'observation.images.top' "
            "with config.image_resolution=%s",
            policy.config.image_resolution,
        )
        h, w = policy.config.image_resolution
        return {"observation.images.top": (int(h), int(w))}

    cameras: dict[str, tuple[int, int]] = {}
    for key, feature in image_features.items():
        shape = feature.shape  # typically (C, H, W)
        if len(shape) == 3:
            _, fh, fw = shape
        elif len(shape) == 2:
            fh, fw = shape
        else:
            LOGGER.warning(
                "Unexpected shape %s for image feature '%s'; using config.image_resolution as fallback",
                shape,
                key,
            )
            fh, fw = policy.config.image_resolution
        cameras[key] = (int(fh), int(fw))

    return cameras


def _prepare_base_tensors(
    device: torch.device,
    batch_size: int,
    lang_tokens_len: int,
    seed: int,
    cameras: dict[str, tuple[int, int]],
) -> dict[str, torch.Tensor]:
    """Prepare dummy tensors for PI05 VLM export.

    Args:
        device: Target torch device.
        batch_size: Batch dimension.
        lang_tokens_len: Length of dummy language tokens.
        seed: Random seed.
        cameras: Mapping of camera key → (height, width), e.g.
            ``{"observation.images.top": (480, 640),
               "observation.images.wrist": (480, 640)}``.
    """
    torch.manual_seed(int(seed))

    tensors: dict[str, torch.Tensor] = {}
    for cam_key, (h, w) in cameras.items():
        tensors[cam_key] = torch.zeros((batch_size, 3, h, w), dtype=torch.float32, device=device) / 255.0

    tensors["lang_tokens"] = torch.randint(0, 1000, (batch_size, lang_tokens_len), dtype=torch.long, device=device)
    tensors["lang_masks"] = torch.ones((batch_size, lang_tokens_len), dtype=torch.bool, device=device)

    return tensors


class ONNXWrapper(torch.nn.Module):
    """Expose policy.select_action as a single forward op for ONNX export."""

    def __init__(self, policy: PI05VLMPolicy, example_observation: dict, device: torch.device):
        super().__init__()
        self.policy = policy.to(device)
        self.device = device
        self._keys = list(example_observation.keys())

    def forward(self, *args):
        if len(args) != len(self._keys):
            raise ValueError(f"Expected {len(self._keys)} inputs, got {len(args)}")
        input_dict = {}
        for key, tensor in zip(self._keys, args, strict=False):
            if key in ("lang_tokens", "observation.language.tokens"):
                input_dict[key] = tensor.to(torch.long)
            elif key in ("lang_masks", "observation.language.attention_mask"):
                input_dict[key] = tensor.to(torch.bool)
            else:
                input_dict[key] = tensor.to(torch.float32)

        with torch.no_grad():
            past_kv_tensor, prefix_pad_masks = self.policy.select_action(input_dict)
        return past_kv_tensor, prefix_pad_masks


def export_onnx(
    *,
    wrapper: ONNXWrapper,
    observation: dict,
    onnx_output_path: Path,
    opset: int,
    dynamo: bool = True,
    constant_folding: bool = True,
    use_npu_ops: bool = False,
) -> None:
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_keys = list(observation.keys())

    LOGGER.info("Exporting ONNX to %s", onnx_output_path)
    LOGGER.info("  opset=%d, dynamo=%s, constant_folding=%s", opset, dynamo, constant_folding)

    observation_values = [observation[k] for k in dummy_keys]

    wrapper.policy.eval()
    # 注意: dynamo=True 时 opset_version 参数会被忽略, 实际固定使用 opset 18
    # Apply Ascend ATC compatibility patches during ONNX export
    with ascend_onnx_export_patches(use_npu_ops=use_npu_ops):
        torch.onnx.export(
            wrapper,
            tuple(observation_values),
            str(onnx_output_path),
            opset_version=int(opset),
            input_names=dummy_keys,
            output_names=["past_kv_tensor", "prefix_pad_masks"],
            do_constant_folding=constant_folding,
            verbose=False,
            dynamo=dynamo,
            external_data=True,
        )

    _clean_onnx_domains(onnx_output_path)


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
    """Remove extra opset_import entries and clear node domain fields.

    This prevents ATC / other downstream tools from failing on unrecognised
    custom domains (e.g. ``pkg.onnxscript.torch_lib``).

    Additionally, when the TorchScript exporter (dynamo=False) is used with
    ``external_data=True``, each tensor is saved as a separate file (e.g.
    ``.weight``, ``.bias``, ``onnx__MatMul_12669``).  After re-saving with
    ``all_tensors_to_one_file=True`` these scattered files become stale and
    are removed here.
    """
    onnx_dir = onnx_path.parent
    consolidated_data_name = onnx_path.name + ".data"

    # --- Step 1: Discover old external-data file paths ---
    # Must load WITHOUT external data: onnx.load() with the default
    # load_external_data=True reads tensor bytes into raw_data and then
    # *clears* the external_data entries, making old file locations
    # unrecoverable.  We also iterate node-attribute tensors (not only
    # initializers) so dynamo-generated constants are covered.
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
    # The dynamo ONNX exporter occasionally corrupts a small number of
    # float16 weight values (producing NaN bit patterns like 0x7d00).
    # Replace any NaN with 0 to prevent inference-time NaN propagation.
    sanitize_nan_initializers(model)

    # Last-ditch ATC StridedSliceD workaround: collapse rotate_half-style
    # Slice pairs into a single Split node.  Must run BEFORE the Split
    # downgrade so the freshly-emitted Split (opset-18 default form) also
    # gets rewritten to the opset-13 sizes-input form ATC accepts.
    # DISABLED: the reshape-based apply_rotary_pos_emb patch eliminates
    # rotate_half Slice/Split nodes from the graph entirely, so there are
    # no pairs left to merge.  Re-enable if a future refactor reintroduces
    # half-split slicing somewhere.
    # rewrite_slice_pairs_to_split(model)

    # Opset-18 Split uses a num_outputs attribute that ATC cannot parse.
    # Downgrade to opset-13 format (2nd input = split_sizes tensor).
    downgrade_split_for_atc(model)

    # Non-dynamo Slice nodes use axes=<positive> + steps=1, which trips
    # ATC's StridedSliceD.  Rewrite to axes=<negative> with no steps so
    # the graph matches the form AE successfully compiles from.
    normalize_slice_for_atc(model)

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


def save_runtime_tensors(
    *,
    wrapper: ONNXWrapper,
    observation: dict,
    runtime_save_dir: Path,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run PyTorch inference and persist KV-cache / prefix masks.

    These ``.pth`` files are consumed by the action-expert export script,
    so they are always generated as part of the VLM export.

    Returns:
        ``(past_kv_tensor, prefix_pad_masks)`` on the original device.
    """
    dummy_keys = list(observation.keys())
    observation_values = [observation[k] for k in dummy_keys]

    wrapper.eval()
    with torch.no_grad():
        torch.manual_seed(int(seed))
        pytorch_past_kv, pytorch_prefix_mask = wrapper(*observation_values)

    runtime_save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(pytorch_past_kv, runtime_save_dir / "past_kv_tensor.pth")
    torch.save(pytorch_prefix_mask, runtime_save_dir / "prefix_pad_masks.pth")
    LOGGER.info(
        "Saved runtime tensors to %s  (past_kv: %s, prefix_mask: %s)",
        runtime_save_dir,
        tuple(pytorch_past_kv.shape),
        tuple(pytorch_prefix_mask.shape),
    )
    return pytorch_past_kv, pytorch_prefix_mask


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export PI05 VLM to ONNX and optionally validate.")
    p.add_argument("--pretrained-policy-path", type=str, required=True, help="Path or repo with config+weights")
    p.add_argument("--output", type=str, default=None, help="Output ONNX file path (auto-generated if omitted).")
    p.add_argument(
        "--output-dir", type=str, default="outputs/onnx", help="Directory for auto-generated output filename"
    )
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    p.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda:0")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size for dummy inputs.")
    p.add_argument("--seed", type=int, default=42, help="Seed for dummy inputs and ORT")
    p.add_argument(
        "--lang-tokens-len",
        type=int,
        default=200,
        help="Dummy language token length (pi05 default tokenizer_max_length=200)",
    )
    p.add_argument(
        "--image-resolution",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Override image resolution (H W) for ALL cameras. "
        "If omitted, each camera's resolution is read from the model config.",
    )
    p.add_argument("--runtime-save-dir", type=str, default="runtime_save", help="Where to dump runtime tensors")
    p.add_argument(
        "--dynamo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use torch dynamo export (default: False)",
    )
    p.add_argument(
        "--constant-folding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable constant folding (default: True)",
    )
    p.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "fp32", "auto"],
        default="fp16",
        help="Export precision: fp16 (default, model.half()), fp32 (full precision), "
        "or auto (preserve original mixed bf16+fp32 weights).",
    )
    p.add_argument(
        "--om-manifest-dir",
        type=str,
        default=None,
        help="Directory to write config.om.json (default: pretrained policy path).",
    )
    p.add_argument(
        "--om-path",
        type=str,
        default=None,
        help="Predicted VLM .om artifact path recorded in the manifest (default: <manifest-dir>/vlm.om).",
    )
    p.add_argument("--skip-om-manifest", action="store_true", help="Do not write/update config.om.json.")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    p.add_argument("--local-files-only", action="store_true", default=True, help="Load policy without network")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    setup_logging(args.log_level)

    policy_path = args.pretrained_policy_path
    export_dtype = args.dtype  # "fp16" or "fp32"
    # torch dynamo exporter 忽略 opset_version 参数, 固定使用 opset 18
    actual_opset = 18 if args.dynamo else args.opset
    runtime_save_dir = _normalize_path(args.runtime_save_dir)
    device = _parse_device(args.device)
    # NPU-affine fused ops (RoPE, etc.) are used automatically when exporting
    # on an NPU device; cpu/cuda exports keep the ORT-runnable fallbacks.
    use_npu_ops = device.type == "npu"

    suffix = _build_onnx_config_suffix(actual_opset, args.dynamo, export_dtype, device.type)
    if args.output is not None:
        base_path = _normalize_path(args.output)
    else:
        base_path = _normalize_path(str(Path(args.output_dir) / "pi05-vlm.onnx"))
    onnx_output_path = base_path.with_name(base_path.stem + suffix + ".onnx")

    LOGGER.info("Loading PI05VLMPolicy from %s", policy_path)
    policy = PI05VLMPolicy.from_pretrained(policy_path, local_files_only=bool(args.local_files_only), strict=False)
    policy = _move_policy_to_device(policy, device)
    if export_dtype == "auto":
        # Preserve original mixed precision (typically bf16 + fp32).
        LOGGER.info("Export dtype: auto — keeping original model weights unchanged")
        LOGGER.warning(
            "auto mode produces bf16 tensors; ATC on Ascend 310/310P may fail — use 910 series for bf16 OM conversion."
        )
    else:
        # Normalize to float32 first — pretrained models may contain bfloat16
        # weights (from mixed-precision training), which are unsupported by
        # some ONNX runtimes and Ascend 310/310P.
        policy = policy.float()
        if export_dtype == "fp16":
            try:
                policy.model = policy.model.half()
            except Exception:
                LOGGER.warning("Failed to convert policy.model to half; continuing in float32")
        LOGGER.info("Export dtype: %s", export_dtype)
    policy.eval()

    # --- 自动识别相机数量与分辨率 ---
    cameras = _detect_cameras_from_config(policy)
    LOGGER.info("Detected %d camera(s) from model config:", len(cameras))
    for cam_key, (ch, cw) in cameras.items():
        LOGGER.info("  %s : %dx%d", cam_key, ch, cw)

    # 允许用户通过 --image-resolution H W 统一覆盖所有相机分辨率
    if args.image_resolution is not None:
        override_h, override_w = args.image_resolution
        LOGGER.info(
            "Overriding all camera resolutions to %dx%d (--image-resolution)",
            override_h,
            override_w,
        )
        cameras = {k: (override_h, override_w) for k in cameras}

    observation = _prepare_base_tensors(
        device, int(args.batch_size), int(args.lang_tokens_len), int(args.seed), cameras
    )

    # Stage B — Plan A: precompute the 4D additive prefix attention mask
    # on the host so OPENPI_ATTENTION_MASK_VALUE (-2.38e38) never enters
    # the exported ONNX graph (ATC fp16 passes corrupt it otherwise).
    observation["prefix_att_2d_masks_4d"] = policy.compute_prefix_att_2d_masks_4d(observation).to(
        device=device, dtype=torch.float32
    )
    LOGGER.info(
        "Injected prefix_att_2d_masks_4d as model input: shape=%s dtype=%s",
        tuple(observation["prefix_att_2d_masks_4d"].shape),
        observation["prefix_att_2d_masks_4d"].dtype,
    )

    wrapper = ONNXWrapper(policy, observation, device)

    export_onnx(
        wrapper=wrapper,
        observation=observation,
        onnx_output_path=onnx_output_path,
        opset=actual_opset,
        dynamo=bool(args.dynamo),
        constant_folding=bool(args.constant_folding),
        use_npu_ops=use_npu_ops,
    )
    LOGGER.info("ONNX export finished")

    # Always save runtime tensors — action-expert export depends on them.
    save_runtime_tensors(
        wrapper=wrapper,
        observation=observation,
        runtime_save_dir=runtime_save_dir,
        seed=int(args.seed),
    )

    if not args.skip_om_manifest:
        if args.om_manifest_dir is not None:
            manifest_dir = Path(args.om_manifest_dir).expanduser().resolve()
        else:
            local_policy_path = Path(policy_path).expanduser()
            if not local_policy_path.is_dir():
                raise ValueError(
                    "--om-manifest-dir is required when --pretrained-policy-path is not a local "
                    f"policy directory (got {policy_path!r}); otherwise the manifest would be "
                    "written to a wrong location relative to the current working directory"
                )
            manifest_dir = local_policy_path.resolve()
        om_path = args.om_path if args.om_path is not None else "vlm.om"
        manifest_path = upsert_pi05_om_manifest(manifest_dir, "vlm", om_path)
        LOGGER.info("Updated OM manifest (vlm) at %s", manifest_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
