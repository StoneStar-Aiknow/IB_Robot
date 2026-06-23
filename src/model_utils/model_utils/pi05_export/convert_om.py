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
to hand-write the command or remember to update ``config.om.json`` so the runtime
can find the artifacts.

By default, ATC is invoked without ``--input_shape`` and with
``--precision_mode_v2=origin``. If a board/toolkit needs an explicit shape,
``--input-shape auto`` derives it from each ONNX graph's static inputs. Each
successful conversion upserts its role
(``vlm`` / ``action_expert``) into ``config.om.json`` via
:func:`om_manifest.upsert_pi05_om_manifest`.

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
        --vlm-onnx outputs/onnx/pi05-vlm_op17_nodyn_fp16_cpu.onnx
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

from model_utils.pi05_export._cli_ui import Stage, print_summary, setup_logging
from model_utils.pi05_export.om_manifest import upsert_pi05_om_manifest

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


def _default_om_output(manifest_dir: Path, role: str, onnx_path: Path) -> Path:
    base = "vlm" if role == "vlm" else "action_expert"
    if onnx_path.stem.endswith("_w8a8"):
        base += "_w8a8"
    return manifest_dir / f"{base}.om"


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
    manifest_dir: Path,
    skip_manifest: bool,
    extra_args: list[str],
    input_shape_mode: str,
    index: int,
    total: int,
) -> Path:
    """Compile one role's ONNX to OM and (optionally) upsert the manifest.

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

    if not skip_manifest:
        manifest_path = upsert_pi05_om_manifest(manifest_dir, role, om_output)
        LOGGER.info("  manifest updated (%s) → %s", role, manifest_path)

    return om_output


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
        help="Local PI05 policy directory (where config.om.json is written). Alias: --pretrained-policy-path.",
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
        help="Output VLM .om path (default: <policy-path>/vlm.om, or vlm_w8a8.om for *_w8a8.onnx).",
    )
    p.add_argument(
        "--ae-om",
        type=str,
        default=None,
        help="Output Action Expert .om path (default: <policy-path>/action_expert.om, or action_expert_w8a8.om for *_w8a8.onnx).",
    )
    p.add_argument(
        "--om-manifest-dir",
        type=str,
        default=None,
        help="Directory for config.om.json (default: pretrained policy path).",
    )
    p.add_argument("--skip-om-manifest", action="store_true", help="Do not write/update config.om.json.")
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

    if shutil.which("atc") is None:
        LOGGER.error(
            "`atc` not found on PATH. Run this on an Ascend host with the CANN "
            "toolkit installed (and `source` the CANN environment)."
        )
        return 1

    policy_path = Path(args.pretrained_policy_path).expanduser()
    if args.om_manifest_dir is not None:
        manifest_dir = Path(args.om_manifest_dir).expanduser().resolve()
    else:
        if not policy_path.is_dir():
            LOGGER.error(
                "--pretrained-policy-path %s is not a local directory; pass --om-manifest-dir explicitly.",
                policy_path,
            )
            return 1
        manifest_dir = policy_path.resolve()

    extra_args = list(args.atc_arg or [])

    jobs: list[tuple[str, Path, Path]] = []
    if args.vlm_onnx:
        vlm_onnx = Path(args.vlm_onnx).expanduser()
        vlm_om = Path(args.vlm_om).expanduser() if args.vlm_om else _default_om_output(manifest_dir, "vlm", vlm_onnx)
        jobs.append(("vlm", vlm_onnx, vlm_om))
    if args.ae_onnx:
        ae_onnx = Path(args.ae_onnx).expanduser()
        ae_om = (
            Path(args.ae_om).expanduser() if args.ae_om else _default_om_output(manifest_dir, "action_expert", ae_onnx)
        )
        jobs.append(("action_expert", ae_onnx, ae_om))

    produced: list[tuple[str, str]] = []
    total = len(jobs)
    for i, (role, onnx_path, om_output) in enumerate(jobs, start=1):
        om_path = convert_role(
            role=role,
            onnx_path=onnx_path,
            om_output=om_output,
            soc_version=args.soc_version,
            manifest_dir=manifest_dir,
            skip_manifest=args.skip_om_manifest,
            extra_args=extra_args,
            input_shape_mode=args.input_shape,
            index=i,
            total=total,
        )
        produced.append((role, str(om_path)))

    rows = [(role, path) for role, path in produced]
    if not args.skip_om_manifest:
        rows.append(("manifest", str(manifest_dir / "config.om.json")))
    if not args.no_summary:
        print_summary("PI05 OM conversion complete", rows, status="✅ DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
