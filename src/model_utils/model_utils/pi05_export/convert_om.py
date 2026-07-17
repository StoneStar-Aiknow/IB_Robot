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
"""Convert exported PI05 ONNX (VLM and/or Action Expert) to Ascend ``.om``.

This is the missing piece that puts PI05 on par with the single-OM ACT flow
(``export_onnx_atc.py``): it wraps the ``atc`` compiler so users no longer have
to hand-write the command or construct the unified deployment manifest.

By default, ATC is invoked without ``--input_shape`` and with
``--precision_mode_v2=origin``. If a board/toolkit needs an explicit shape,
``--input-shape auto`` derives it from each ONNX graph's static inputs. The
manifest is finalized only when both roles and both ONNX ABIs are available.

Examples
--------
Convert both segments and write the manifest::

    python -m model_utils.pi05_export.convert_om \\
        --pretrained-policy-path /path/to/pi05_ckpt \\
        --soc-version Ascend310P3 \\
        --vlm-onnx   outputs/onnx/pi05-vlm_op17_nodyn_fp16_cpu.onnx \\
        --ae-onnx    outputs/onnx/pi05-action_expert_op17_nodyn_fp16_cpu.onnx

Convert only the VLM (e.g. after re-exporting just that segment)::

    python -m model_utils.pi05_export.convert_om \\
        --pretrained-policy-path /path/to/pi05_ckpt \\
        --soc-version Ascend310P3 \\
        --vlm-onnx outputs/onnx/pi05-vlm_op17_nodyn_fp16_cpu.onnx \\
        --skip-manifest
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

from inference_manifest import DeviceLink
from model_utils.inference_manifest_export import (
    artifact_bindings,
    compiled_deployment,
    package_deployment_artifact,
    read_runtime_abi,
    upsert_deployment,
)
from model_utils.pi05_export._cli_ui import Stage, print_summary, setup_logging

LOGGER = logging.getLogger("pi05_export.convert_om")


def _derive_input_shape(onnx_path: Path) -> str:
    """Build the ATC ``--input_shape`` string from an ONNX graph's static dims.

    Returns a string like ``name0:1,3,480,640;name1:1,200`` covering every
    non-initializer graph input. Raises if any input has a dynamic dim, since
    ATC needs fully static shapes for these single-shape OM artifacts.
    """
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    initializer_names = {init.name for init in model.graph.initializer}

    parts: list[str] = []
    for inp in model.graph.input:
        if inp.name in initializer_names:
            continue
        dims = inp.type.tensor_type.shape.dim
        sizes: list[int] = []
        for d in dims:
            if not d.HasField("dim_value") or d.dim_value <= 0:
                raise ValueError(
                    f"ONNX input {inp.name!r} in {onnx_path.name} has a dynamic/unknown dim; "
                    "re-export with fixed shapes before ATC conversion."
                )
            sizes.append(int(d.dim_value))
        parts.append(f"{inp.name}:{','.join(map(str, sizes))}")
    if not parts:
        raise ValueError(f"No graph inputs found in {onnx_path}")
    return ";".join(parts)


def _has_atc_arg(extra_args: list[str], name: str) -> bool:
    """Return True when raw ATC args already include ``--<name>``."""
    return any((token := arg.lstrip("-")) == name or token.startswith(f"{name}=") for arg in extra_args)


def _default_om_output(manifest_dir: Path, role: str) -> Path:
    return manifest_dir / "model_utils_work" / "ascend" / "pi05" / f"{role}.om"


def _run_atc(
    onnx_path: Path,
    om_output: Path,
    soc_version: str,
    *,
    extra_args: list[str],
    input_shape_mode: str,
) -> bool:
    """Invoke ``atc`` to compile a single ONNX to an OM. Returns success flag.

    ``atc`` appends ``.om`` itself, so ``--output`` is the path without suffix.
    The subprocess inherits stdout/stderr so the user sees ATC progress live.
    """
    output_stem = str(om_output.with_suffix(""))
    om_output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "atc",
        "--framework=5",
        f"--soc_version={soc_version}",
        f"--model={onnx_path}",
        f"--output={output_stem}",
    ]
    has_input_shape_arg = _has_atc_arg(extra_args, "input_shape")
    if input_shape_mode == "auto" and not has_input_shape_arg:
        input_shape = _derive_input_shape(onnx_path)
        LOGGER.info("  input_shape (auto): %s", input_shape)
        command.append(f"--input_shape={input_shape}")
    elif has_input_shape_arg:
        LOGGER.info("  input_shape: provided via --atc-arg")
    else:
        LOGGER.info("  input_shape: omitted")

    if not _has_atc_arg(extra_args, "precision_mode_v2"):
        command.append("--precision_mode_v2=origin")
    command.extend(extra_args)

    LOGGER.info("  $ %s", " ".join(command))
    # nosec B603 — command is built from validated paths / args, not shell.
    return subprocess.run(command, check=False).returncode == 0


def convert_role(
    *,
    role: str,
    onnx_path: Path,
    om_output: Path,
    soc_version: str,
    extra_args: list[str],
    input_shape_mode: str,
    index: int,
    total: int,
) -> Path:
    """Compile one role's ONNX to OM.

    Returns the produced ``.om`` path. Raises on ATC failure so the orchestrator
    can stop while leaving any already-produced artifacts in place (resumable).
    """
    if not onnx_path.is_file():
        raise FileNotFoundError(f"{role} ONNX not found: {onnx_path}")

    with Stage(f"ATC compile {role}", index=index, total=total):
        ok = _run_atc(onnx_path, om_output, soc_version, extra_args=extra_args, input_shape_mode=input_shape_mode)
        if not ok:
            raise RuntimeError(f"ATC failed for {role} ({onnx_path.name}); see ATC log above")
        if not om_output.is_file():
            raise RuntimeError(f"ATC reported success but {om_output} was not produced")

    return om_output


def write_pi05_ascend_deployment(
    bundle_root: Path,
    deployment_name: str,
    soc_version: str,
    vlm_abi_path: Path,
    vlm_om: Path,
    action_abi_path: Path,
    action_om: Path,
) -> Path:
    """Write one complete PI0.5 Ascend deployment from compiled runtime ABIs."""

    vlm_abi = read_runtime_abi(vlm_abi_path)
    action_abi = read_runtime_abi(action_abi_path)
    vlm_input_semantics = {
        tensor.name: (
            "observation.language.tokens"
            if tensor.name in {"lang_tokens", "observation.language.tokens"}
            else "observation.language.attention_mask"
            if tensor.name in {"lang_masks", "observation.language.attention_mask"}
            else tensor.name
        )
        for tensor in vlm_abi.inputs
    }
    vlm_bindings = artifact_bindings(
        vlm_abi,
        input_semantics=vlm_input_semantics,
        output_semantics={
            "past_kv_tensor": "internal.past_kv",
            "prefix_pad_masks": "internal.prefix_pad_masks",
        },
        image_layouts={
            semantic: "NCHW" for semantic in vlm_input_semantics.values() if semantic.startswith("observation.images.")
        },
    )
    action_bindings = artifact_bindings(
        action_abi,
        input_semantics={
            "past_kv_tensor": "internal.past_kv",
            "prefix_pad_masks": "internal.prefix_pad_masks",
            "time": "time",
            "noise": "noise",
        },
        output_semantics={"action": "action"},
    )
    links = tuple(
        DeviceLink(
            semantic=semantic,
            producer="vlm",
            consumer="action_expert",
            transport="device_pointer",
            owner="producer",
            lifetime="inference",
        )
        for semantic in ("internal.past_kv", "internal.prefix_pad_masks")
    )
    packaged_vlm = package_deployment_artifact(
        bundle_root,
        vlm_om,
        backend="ascend",
        deployment_name=deployment_name,
        role="vlm",
        force_copy=True,
    )
    packaged_action = package_deployment_artifact(
        bundle_root,
        action_om,
        backend="ascend",
        deployment_name=deployment_name,
        role="action_expert",
        force_copy=True,
    )
    deployment = compiled_deployment(
        bundle_root,
        backend="ascend",
        target_soc=soc_version,
        target_runtime="acl",
        artifacts={"vlm": (packaged_vlm, "om"), "action_expert": (packaged_action, "om")},
        execution=("vlm", "action_expert"),
        bindings={"vlm": vlm_bindings, "action_expert": action_bindings},
        device_links=links,
    )
    return upsert_deployment(bundle_root, deployment_name, deployment).manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compile exported PI05 ONNX (VLM / Action Expert) to Ascend OM via ATC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--policy-path",
        "--pretrained-policy-path",
        dest="pretrained_policy_path",
        type=str,
        required=True,
        help="Local PI05 policy bundle directory. Alias: --pretrained-policy-path.",
    )
    p.add_argument(
        "--soc-version",
        type=str,
        required=True,
        help="Target Ascend SoC, e.g. Ascend310P3 (see `npu-smi info`).",
    )
    p.add_argument("--vlm-onnx", type=str, default=None, help="VLM ONNX to compile (omit to skip the VLM).")
    p.add_argument("--ae-onnx", type=str, default=None, help="Action Expert ONNX to compile (omit to skip the AE).")
    p.add_argument(
        "--vlm-om",
        type=str,
        default=None,
        help="ATC VLM OM work output (default: <bundle>/model_utils_work/ascend/pi05/vlm.om).",
    )
    p.add_argument(
        "--ae-om",
        type=str,
        default=None,
        help="ATC Action Expert OM work output (default: <bundle>/model_utils_work/ascend/pi05/action_expert.om).",
    )
    p.add_argument(
        "--vlm-abi",
        type=str,
        default=None,
        help="Existing compiler/runtime-introspected VLM OM ABI JSON input (default: <vlm-om>.abi.json).",
    )
    p.add_argument(
        "--ae-abi",
        type=str,
        default=None,
        help="Existing compiler/runtime-introspected Action Expert OM ABI JSON input (default: <ae-om>.abi.json).",
    )
    p.add_argument(
        "--bundle-root",
        type=str,
        default=None,
        help="Policy bundle root for inference_manifest.json (default: pretrained policy path).",
    )
    p.add_argument("--skip-manifest", action="store_true", help="Do not finalize inference_manifest.json.")
    p.add_argument(
        "--manifest-only",
        action="store_true",
        help="Skip ATC and finalize the manifest from existing ONNX and OM artifacts.",
    )
    p.add_argument("--deployment", default="ascend", help="Unified manifest deployment name.")
    p.add_argument(
        "--input-shape",
        choices=("none", "auto"),
        default="none",
        help="ATC --input_shape handling: omit by default, or derive from static ONNX graph inputs.",
    )
    p.add_argument(
        "--atc-arg",
        action="append",
        default=None,
        metavar="ARG",
        help="Extra raw argument forwarded to atc (repeatable). Default adds --precision_mode_v2=origin unless overridden.",
    )
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    p.add_argument(
        "--no-summary",
        action="store_true",
        help="Suppress the final result block (used when an orchestrator prints its own summary).",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    setup_logging(args.log_level)

    if not args.vlm_onnx and not args.ae_onnx:
        LOGGER.error("Nothing to do: pass --vlm-onnx and/or --ae-onnx.")
        return 1

    if not args.manifest_only and shutil.which("atc") is None:
        LOGGER.error(
            "`atc` not found on PATH. Run this on an Ascend host with the CANN "
            "toolkit installed (and `source` the CANN environment)."
        )
        return 1

    policy_path = Path(args.pretrained_policy_path).expanduser()
    if args.bundle_root is not None:
        manifest_dir = Path(args.bundle_root).expanduser().resolve()
    else:
        if not policy_path.is_dir():
            LOGGER.error(
                "--pretrained-policy-path %s is not a local directory; pass --bundle-root explicitly.",
                policy_path,
            )
            return 1
        manifest_dir = policy_path.resolve()

    extra_args = list(args.atc_arg or [])

    jobs: list[tuple[str, Path, Path]] = []
    if args.vlm_onnx:
        vlm_onnx = Path(args.vlm_onnx).expanduser()
        vlm_om = Path(args.vlm_om).expanduser().resolve() if args.vlm_om else _default_om_output(manifest_dir, "vlm")
        jobs.append(("vlm", vlm_onnx, vlm_om))
    if args.ae_onnx:
        ae_onnx = Path(args.ae_onnx).expanduser()
        ae_om = (
            Path(args.ae_om).expanduser().resolve() if args.ae_om else _default_om_output(manifest_dir, "action_expert")
        )
        jobs.append(("action_expert", ae_onnx, ae_om))

    produced: list[tuple[str, str]] = []
    total = len(jobs)
    for i, (role, onnx_path, om_output) in enumerate(jobs, start=1):
        if args.manifest_only:
            if not onnx_path.is_file():
                raise FileNotFoundError(f"{role} ONNX not found: {onnx_path}")
            if not om_output.is_file():
                raise FileNotFoundError(f"{role} OM not found: {om_output}")
            om_path = om_output
        else:
            om_path = convert_role(
                role=role,
                onnx_path=onnx_path,
                om_output=om_output,
                soc_version=args.soc_version,
                extra_args=extra_args,
                input_shape_mode=args.input_shape,
                index=i,
                total=total,
            )
        produced.append((role, str(om_path)))

    if not args.skip_manifest:
        if not args.vlm_onnx or not args.ae_onnx:
            LOGGER.error("Unified PI0.5 manifest finalization requires both --vlm-onnx and --ae-onnx.")
            return 1
        vlm_om = next(Path(path) for role, path in produced if role == "vlm")
        action_om = next(Path(path) for role, path in produced if role == "action_expert")
        vlm_abi = Path(args.vlm_abi).expanduser() if args.vlm_abi else Path(f"{vlm_om}.abi.json")
        action_abi = Path(args.ae_abi).expanduser() if args.ae_abi else Path(f"{action_om}.abi.json")
        manifest_path = write_pi05_ascend_deployment(
            manifest_dir,
            args.deployment,
            args.soc_version,
            vlm_abi,
            vlm_om,
            action_abi,
            action_om,
        )
        LOGGER.info("  unified manifest → %s", manifest_path)

    rows = [(role, path) for role, path in produced]
    if not args.skip_manifest:
        rows.append(("manifest", str(manifest_dir / "inference_manifest.json")))
    if not args.no_summary:
        print_summary("PI05 OM conversion complete", rows, status="✅ DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
