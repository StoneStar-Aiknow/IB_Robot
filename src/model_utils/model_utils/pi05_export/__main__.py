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
"""One-command PI05 export pipeline.

This is the single front door for the PI05 → Ascend OM toolchain. It chains the
existing, well-tested stage scripts in the right order with the right files
flowing between them, so the user does not have to remember six module paths,
keep ``--dtype`` consistent across two exports, or hand-write the ATC command.

What runs is chosen explicitly via ``--steps`` (a comma-separated list). The
available steps are:

    vlm_onnx   Export the VLM (gemma_2b) to ONNX
    ae_onnx    Export the Action Expert (gemma_300m) to ONNX
    verify     Split-vs-monolithic equivalence check   (needs --batch-path)
    vlm_quant  Quantize the VLM ONNX to W8A8           (needs --batch-path)
    vlm_om     Compile the VLM ONNX to OM via ATC      (needs --soc-version)
    ae_om      Compile the Action Expert ONNX to OM    (needs --soc-version)
    ae_quant   Capture an FP16 trajectory and quantize the AE to W8A8
    vlm_quant_om  Compile the VLM W8A8 ONNX to OM      (needs --soc-version)
    ae_quant_om   Compile the AE W8A8 ONNX to OM       (needs --soc-version)

Default ``--steps`` = ``vlm_onnx,ae_onnx,vlm_om,ae_om`` (export both segments
and compile both OMs). Quantization and verification are opt-in by listing the
corresponding step. The default produces artifacts but is not an accuracy-validated complete conversion;
the ``om-convert`` workflow explicitly includes ``verify`` with a canonical observation batch.

Step semantics
--------------
* **A step listed in --steps is always run**, even if its product already
  exists (it is rebuilt). ("You named it, so do it.") This is how you
  re-export / re-compile a single segment after a parameter change:
  ``--steps vlm_onnx,vlm_om``.
* **A step's upstream products must already exist or be requested too.** If a
  step needs an artifact that is neither in --steps nor on disk, the run stops
  with a precise "add this step" message rather than silently doing the wrong
  thing (no implicit upstream steps are added).
* ``*_om`` compiles the FP16 ONNX. Use ``*_quant_om`` to compile W8A8 ONNX.
* ``ae_quant`` uses fresh FP16 VLM/AE OMs from the same invocation to capture
  calibration trajectories automatically before quantization.
* One invocation adds or updates only ``--deployment``. Quantized and FP roles
  use the same publication path; the selected profile chooses each role's OM.

Design notes
------------
* **Intermediate products are preserved.** Nothing is deleted between stages;
  the ONNX files, compiled OM files, ``runtime_save/*.pth`` and unified manifest
  all remain on disk for inspection or a partial re-run.
* **Live feedback.** Stages run as child processes with inherited stdout/stderr,
  so the user sees real progress (export logs, ATC compile output).
* **Minimal surface.** All argument ergonomics (profile / wizard / --work-dir
  derivation / remember-last) live in ``_cli``; this entry point only plans and
  runs steps.
"""

from __future__ import annotations

import errno
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from inference_manifest import load_policy_metadata
from inference_service.pi05_schedule import uniform_pi05_schedule, write_pi05_schedule
from model_utils.export_paths import export_work_dir, resolve_outside_bundle_path
from model_utils.observation_batch import load_observation_batch
from model_utils.pi05_export import _cli
from model_utils.pi05_export._cli_ui import Stage, build_onnx_suffix, print_summary, setup_logging
from model_utils.pi05_export.convert_om import write_pi05_ascend_deployment
from model_utils.pi05_export.quant.profiles import (
    QuantizationProfile,
    metadata_path,
    validate_quantization_metadata,
)

LOGGER = logging.getLogger("pi05_export.pipeline")


def _run_module(module: str, cli_args: list[str]) -> None:
    """Run ``python -m <module> <args>`` as a child process, streaming output.

    Raises CalledProcessError on non-zero exit so the orchestrator stops and the
    user can fix the issue and re-run (already-finished steps will be skipped).
    """
    command = [sys.executable, "-m", module, *cli_args]
    LOGGER.info("  $ %s", " ".join(command))
    subprocess.run(command, check=True)  # nosec B603 — args are program-controlled.


# ---------------------------------------------------------------------------
# Resolved paths shared by every step runner.
# ---------------------------------------------------------------------------
@dataclass
class Ctx:
    args: object  # resolved argparse.Namespace
    policy_path: Path
    output_dir: Path
    runtime_save_dir: Path
    om_dir: Path
    vlm_onnx: Path
    ae_onnx: Path
    vlm_donor_onnx: Path
    ae_donor_onnx: Path
    vlm_w8a8: Path
    ae_w8a8: Path
    vlm_om: Path
    ae_om: Path
    vlm_quant_om: Path
    ae_quant_om: Path
    calibration_dir: Path
    chosen: set[str]  # steps explicitly requested in --steps
    quantization_profile: QuantizationProfile | None = None


# ---------------------------------------------------------------------------
# Per-step product paths and runners. Keyed by the same names as _cli.STEPS so
# adding a step is: register it in _cli.STEPS + add an entry here.
# ---------------------------------------------------------------------------
def _product(ctx: Ctx, step: str) -> Path | None:
    """The artifact a step produces (None for steps with no file product)."""
    return {
        "vlm_onnx": ctx.vlm_onnx,
        "ae_onnx": ctx.ae_onnx,
        "vlm_quant": ctx.vlm_w8a8,
        "ae_quant": ctx.ae_w8a8,
        "vlm_om": ctx.vlm_om,
        "ae_om": ctx.ae_om,
        "vlm_quant_om": ctx.vlm_quant_om,
        "ae_quant_om": ctx.ae_quant_om,
        "verify": None,
    }[step]


def _product_exists(ctx: Ctx, step: str) -> bool:
    prod = _product(ctx, step)
    return prod is not None and prod.is_file()


def _device_tag_from_name(path: Path) -> str | None:
    """Return the final ONNX suffix token (cpu/cuda/npu/donor) when recognizable."""
    stem = path.stem
    if stem.endswith("_w8a8"):
        stem = stem[: -len("_w8a8")]
    token = stem.rsplit("_", 1)[-1]
    return token if token in {"cpu", "cuda", "npu", "donor"} else None


def _is_npu_onnx(path: Path) -> bool:
    return _device_tag_from_name(path) == "npu"


def _is_donor_onnx(path: Path) -> bool:
    tag = _device_tag_from_name(path)
    return tag in {"cpu", "cuda", "donor"}


def _uses_fused_geglu_donor(ctx: Ctx, role: str) -> bool:
    profile = getattr(ctx, "quantization_profile", None)
    if profile is not None and profile.role(role).fused_geglu_donor is not None:
        return profile.role(role).fused_geglu_donor
    return False


def _donor_model_stem(role: str, fused_geglu: bool) -> str:
    stem = "pi05-vlm" if role == "vlm" else "pi05-action_expert"
    return stem + ("_fused-geglu" if fused_geglu else "")


def _donor_dtype(ctx: Ctx, role: str) -> str:
    profile = getattr(ctx, "quantization_profile", None)
    if profile is not None and profile.role(role).donor_dtype is not None:
        return profile.role(role).donor_dtype
    return ctx.args.dtype


def _quant_profile_args(ctx: Ctx, *, role: str, output_onnx: Path) -> list[str]:
    profile = getattr(ctx, "quantization_profile", None)
    if profile is None:
        return []
    role_profile = profile.role(role)
    args = [
        "--disable-regex",
        *role_profile.disable_regex,
        "--quantize-regex",
        *(selector.regex for selector in role_profile.selectors),
        "--quantize-regex-expected",
        *(str(selector.expected) for selector in role_profile.selectors),
        "--quant-profile-name",
        profile.name,
        "--quant-profile-hash",
        profile.digest,
        "--quant-role",
        role,
        "--quant-metadata-path",
        str(metadata_path(output_onnx)),
    ]
    if role_profile.expected_selected_nodes is not None:
        args.extend(["--expected-selected-nodes", str(role_profile.expected_selected_nodes)])
    if role_profile.expected_quantized_nodes is not None:
        args.extend(["--expected-quantized-nodes", str(role_profile.expected_quantized_nodes)])
    if role_profile.quantize_convs:
        args.append("--quantize-convs")
    if role_profile.expected_npu_geglu_nodes is not None:
        args.append("--require-npu-geglu")
        args.extend(["--expected-npu-geglu-nodes", str(role_profile.expected_npu_geglu_nodes)])
    if role == "ae" and role_profile.expected_calibration_steps is not None:
        args.extend(["--expected-calibration-steps", str(role_profile.expected_calibration_steps)])
    if role_profile.smoothquant_alpha is not None:
        args.extend(
            [
                "--smoothquant-alpha",
                str(role_profile.smoothquant_alpha),
                "--smoothquant-epsilon",
                str(role_profile.smoothquant_epsilon),
            ]
        )
    if role_profile.smoothquant_verify_rtol is not None:
        args.extend(
            [
                "--smoothquant-verify-rtol",
                str(role_profile.smoothquant_verify_rtol),
                "--smoothquant-verify-atol",
                str(role_profile.smoothquant_verify_atol),
            ]
        )
    return args


def _run_vlm_donor_onnx(ctx: Ctx) -> None:
    a = ctx.args
    fused_donor = _uses_fused_geglu_donor(ctx, "vlm")
    _run_module(
        "model_utils.pi05_export.convert_onnx_vlm",
        [
            "--pretrained-policy-path",
            str(ctx.policy_path),
            "--output-dir",
            str(ctx.output_dir),
            "--output",
            str(ctx.output_dir / f"{_donor_model_stem('vlm', fused_donor)}.onnx"),
            "--runtime-save-dir",
            str(ctx.runtime_save_dir),
            "--dtype",
            _donor_dtype(ctx, "vlm"),
            "--device",
            a.donor_device,
            *(["--fused-geglu-donor"] if fused_donor else []),
            "--skip-runtime-save",
            "--log-level",
            a.log_level,
        ],
    )


def _run_ae_donor_onnx(ctx: Ctx) -> None:
    a = ctx.args
    fused_donor = _uses_fused_geglu_donor(ctx, "ae")
    _run_module(
        "model_utils.pi05_export.convert_onnx_action_expert",
        [
            "--pretrained-policy-path",
            str(ctx.policy_path),
            "--output-dir",
            str(ctx.output_dir),
            "--output",
            str(ctx.output_dir / f"{_donor_model_stem('ae', fused_donor)}.onnx"),
            "--past-kv-path",
            str(ctx.runtime_save_dir / "past_kv_tensor.pth"),
            "--prefix-pad-masks-path",
            str(ctx.runtime_save_dir / "prefix_pad_masks.pth"),
            "--dtype",
            _donor_dtype(ctx, "ae"),
            "--device",
            a.donor_device,
            *(["--fused-geglu-donor"] if fused_donor else []),
            "--log-level",
            a.log_level,
        ],
    )


def _run_donor_onnx(ctx: Ctx, *, role: str) -> None:
    if role == "vlm":
        _run_vlm_donor_onnx(ctx)
    else:
        _run_ae_donor_onnx(ctx)


def _quant_inputs(ctx: Ctx, *, role: str) -> tuple[Path, Path | None]:
    """Return (donor_onnx, npu_onnx) for quantization, generating donor if needed."""
    deploy_onnx = ctx.vlm_onnx if role == "vlm" else ctx.ae_onnx
    donor_onnx = ctx.vlm_donor_onnx if role == "vlm" else ctx.ae_donor_onnx
    role_label = "VLM" if role == "vlm" else "Action Expert"

    if _is_npu_onnx(deploy_onnx):
        LOGGER.info("%s quant: deployment ONNX is NPU graph: %s", role_label, deploy_onnx)
        fused_donor = _uses_fused_geglu_donor(ctx, role)
        refresh_donor = f"{role}_onnx" in ctx.chosen or fused_donor
        if donor_onnx.is_file() and not refresh_donor:
            LOGGER.info("%s quant: reusing donor ONNX: %s", role_label, donor_onnx)
        else:
            if refresh_donor:
                LOGGER.info(
                    "%s quant: regenerating donor ONNX because %s_onnx is in --steps: %s",
                    role_label,
                    role,
                    donor_onnx,
                )
            else:
                LOGGER.info(
                    "%s quant: donor ONNX missing; generating with --donor-device %s: %s",
                    role_label,
                    ctx.args.donor_device,
                    donor_onnx,
                )
            _run_donor_onnx(ctx, role=role)
            if not donor_onnx.is_file():
                raise RuntimeError(f"{role_label} donor ONNX was not produced: {donor_onnx}")
        LOGGER.info("%s quant: Route A enabled (donor -> NPU graph graft).", role_label)
        return donor_onnx, deploy_onnx

    if not _is_donor_onnx(deploy_onnx):
        LOGGER.warning("%s quant: cannot infer ONNX role from filename; treating as donor: %s", role_label, deploy_onnx)
    else:
        LOGGER.info(
            "%s quant: deployment ONNX is ORT-runnable donor graph; quantizing directly: %s", role_label, deploy_onnx
        )
    return deploy_onnx, None


def _vlm_fast_gelu_scope(args) -> str:  # noqa: ANN001
    scope = getattr(args, "fast_gelu_scope", None)
    if scope is None:
        scope = "all" if getattr(args, "fast_gelu", False) else "none"
    return {"vlm-text": "text", "ae": "none"}.get(scope, scope)


def _ae_fast_gelu_scope(args) -> str:  # noqa: ANN001
    scope = getattr(args, "fast_gelu_scope", None)
    if scope is None:
        scope = "all" if getattr(args, "fast_gelu", False) else "none"
    return "all" if scope in {"all", "ae"} else "none"


def _run_vlm_onnx(ctx: Ctx) -> None:
    a = ctx.args
    _run_module(
        "model_utils.pi05_export.convert_onnx_vlm",
        [
            "--pretrained-policy-path",
            str(ctx.policy_path),
            "--output-dir",
            str(ctx.output_dir),
            "--runtime-save-dir",
            str(ctx.runtime_save_dir),
            "--dtype",
            a.dtype,
            "--device",
            a.device,
            "--fast-gelu-scope",
            _vlm_fast_gelu_scope(a),
            "--log-level",
            a.log_level,
        ],
    )


def _run_ae_onnx(ctx: Ctx) -> None:
    a = ctx.args
    _run_module(
        "model_utils.pi05_export.convert_onnx_action_expert",
        [
            "--pretrained-policy-path",
            str(ctx.policy_path),
            "--output-dir",
            str(ctx.output_dir),
            "--past-kv-path",
            str(ctx.runtime_save_dir / "past_kv_tensor.pth"),
            "--prefix-pad-masks-path",
            str(ctx.runtime_save_dir / "prefix_pad_masks.pth"),
            "--dtype",
            a.dtype,
            "--device",
            a.device,
            "--fast-gelu-scope",
            _ae_fast_gelu_scope(a),
            "--log-level",
            a.log_level,
        ],
    )


def _run_vlm_quant(ctx: Ctx) -> None:
    a = ctx.args
    donor_onnx, npu_onnx = _quant_inputs(ctx, role="vlm")
    _run_module(
        "model_utils.pi05_export.quant.quantize_vlm",
        [
            "--onnx-path",
            str(donor_onnx),
            "--output-path",
            str(ctx.vlm_w8a8),
            "--policy-path",
            str(ctx.policy_path),
            "--batch-path",
            str(Path(a.batch_path).expanduser()),
            "--num-calib",
            str(a.num_calib),
            "--amp-num",
            str(a.amp_num),
            "--amp-rank-samples",
            str(a.amp_rank_samples),
            *(["--amp-scratch-dir", str(Path(a.amp_scratch_dir).expanduser())] if a.amp_scratch_dir else []),
            "--device",
            a.device,
            *(["--task", a.task] if a.task else []),
            *(["--fused-geglu-donor"] if npu_onnx is not None and _uses_fused_geglu_donor(ctx, "vlm") else []),
            *(["--npu-onnx-path", str(npu_onnx)] if npu_onnx else []),
            *_quant_profile_args(ctx, role="vlm", output_onnx=ctx.vlm_w8a8),
            "--log-level",
            a.log_level,
        ],
    )


def _run_ae_quant(ctx: Ctx) -> None:
    a = ctx.args
    _capture_ae_calibration(ctx)
    donor_onnx, npu_onnx = _quant_inputs(ctx, role="ae")
    profile_args = _quant_profile_args(ctx, role="ae", output_onnx=ctx.ae_w8a8)
    if "--expected-calibration-steps" not in profile_args:
        profile_args.extend(["--expected-calibration-steps", str(_ae_calibration_steps(ctx))])
    _run_module(
        "model_utils.pi05_export.quant.quantize_ae",
        [
            "--onnx-path",
            str(donor_onnx),
            "--output-path",
            str(ctx.ae_w8a8),
            "--policy-path",
            str(ctx.policy_path),
            "--calib-dir",
            str(ctx.calibration_dir),
            "--num-calib",
            str(a.num_calib),
            "--amp-num",
            str(a.amp_num),
            "--amp-rank-samples",
            str(a.amp_rank_samples),
            *(["--amp-scratch-dir", str(Path(a.amp_scratch_dir).expanduser())] if a.amp_scratch_dir else []),
            *(["--fused-geglu-donor"] if npu_onnx is not None and _uses_fused_geglu_donor(ctx, "ae") else []),
            *(["--npu-onnx-path", str(npu_onnx)] if npu_onnx else []),
            *profile_args,
            "--log-level",
            a.log_level,
        ],
    )


def _capture_ae_calibration(ctx: Ctx) -> None:
    step_count = _ae_calibration_steps(ctx)
    batch_count = len(load_observation_batch(ctx.args.batch_path))
    if ctx.args.num_calib > 0:
        batch_count = min(batch_count, ctx.args.num_calib)
    if batch_count < 1:
        raise ValueError("automatic AE calibration requires at least one observation")

    calibration_parent = ctx.calibration_dir.parent
    calibration_parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ae-calibration-bundle-", dir=calibration_parent) as temporary:
        temporary_root = Path(temporary)
        bundle_root = temporary_root / "bundle"
        _link_policy_metadata(ctx.policy_path, bundle_root)
        schedule_path = write_pi05_schedule(
            uniform_pi05_schedule(step_count, name=f"calibration_uniform{step_count}"),
            temporary_root / "calibration_schedule.json",
        )
        deployment = "pi05-ae-calibration"
        write_pi05_ascend_deployment(
            bundle_root,
            deployment,
            ctx.args.soc_version,
            Path(f"{ctx.vlm_om}.abi.json"),
            ctx.vlm_om,
            Path(f"{ctx.ae_om}.abi.json"),
            ctx.ae_om,
            schedule_path,
            prefer_hardlink=True,
        )
        dump_args = [
            "--policy-path",
            str(bundle_root),
            "--deployment",
            deployment,
            "--batch-path",
            str(Path(ctx.args.batch_path).expanduser()),
            "--batch-index",
            "0",
            "--batch-count",
            str(batch_count),
            "--out-dir",
            str(ctx.calibration_dir),
            "--task",
            ctx.args.task,
            "--seed",
            "42",
        ]
        _run_module("model_utils.pi05_om_dump", dump_args)


def _ae_calibration_steps(ctx: Ctx) -> int:
    profile = ctx.quantization_profile
    role_profile = profile.action_expert if profile is not None else None
    step_count = role_profile.expected_calibration_steps if role_profile is not None else None
    if step_count is None:
        config = json.loads((ctx.policy_path / "config.json").read_text(encoding="utf-8"))
        step_count = config.get("num_inference_steps") if isinstance(config, dict) else None
    if not isinstance(step_count, int) or isinstance(step_count, bool) or step_count < 1:
        raise ValueError("automatic AE calibration requires a positive expected_calibration_steps contract")
    return step_count


def _link_policy_metadata(source_root: Path, destination_root: Path) -> None:
    source = source_root.expanduser().resolve(strict=True)
    destination_root.mkdir(parents=True, exist_ok=True)
    policy = load_policy_metadata(source, require_native_weights=False)
    for relative in policy.required_files:
        source_path = source.joinpath(*relative.split("/"))
        destination = destination_root.joinpath(*relative.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            destination.hardlink_to(source_path)
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP}:
                raise
            shutil.copy2(source_path, destination)
    _rewrite_absolute_bundle_references(source, destination_root)


def _rewrite_absolute_bundle_references(source_root: Path, destination_root: Path) -> None:
    def rewrite(value: object) -> object:
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if not isinstance(value, str) or not Path(value).is_absolute():
            return value
        try:
            relative = Path(value).resolve(strict=True).relative_to(source_root)
        except (OSError, ValueError):
            return value
        return relative.as_posix()

    for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        path = destination_root / name
        value = json.loads(path.read_text(encoding="utf-8"))
        rewritten = rewrite(value)
        if rewritten == value:
            continue
        path.unlink()
        path.write_text(json.dumps(rewritten, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_om(ctx: Ctx, *, role: str, onnx_path: Path, om_path: Path) -> None:
    a = ctx.args
    role_args = (
        ["--vlm-onnx", str(onnx_path), "--vlm-om", str(om_path)]
        if role == "vlm"
        else ["--ae-onnx", str(onnx_path), "--ae-om", str(om_path)]
    )
    _run_module(
        "model_utils.pi05_export.convert_om",
        [
            "--pretrained-policy-path",
            str(ctx.policy_path),
            "--soc-version",
            a.soc_version,
            *role_args,
            "--skip-manifest",
            "--abi-device-id",
            str(a.abi_device_id),
            *(["--acl-config-path", a.acl_config_path] if a.acl_config_path else []),
            "--no-summary",
            "--log-level",
            a.log_level,
        ],
    )


def _run_vlm_om(ctx: Ctx) -> None:
    _run_om(ctx, role="vlm", onnx_path=ctx.vlm_onnx, om_path=ctx.vlm_om)


def _run_ae_om(ctx: Ctx) -> None:
    _run_om(ctx, role="ae", onnx_path=ctx.ae_onnx, om_path=ctx.ae_om)


def _run_vlm_quant_om(ctx: Ctx) -> None:
    _validate_quantized_profile_dependencies(ctx, ["vlm_quant_om"])
    _run_om(ctx, role="vlm", onnx_path=ctx.vlm_w8a8, om_path=ctx.vlm_quant_om)


def _run_ae_quant_om(ctx: Ctx) -> None:
    _validate_quantized_profile_dependencies(ctx, ["ae_quant_om"])
    _run_om(ctx, role="ae", onnx_path=ctx.ae_w8a8, om_path=ctx.ae_quant_om)


def _run_verify(ctx: Ctx) -> None:
    a = ctx.args
    _run_module(
        "model_utils.pi05_export.verify_pi05_split_equivalence",
        [
            "--pretrained-policy-path",
            str(ctx.policy_path),
            "--vlm-onnx-path",
            str(ctx.vlm_onnx),
            "--ae-onnx-path",
            str(ctx.ae_onnx),
            "--batch-path",
            str(Path(a.batch_path).expanduser()),
            "--task",
            a.task,
            "--device",
            a.device,
            *(["--schedule-file", a.schedule_file] if a.schedule_file else []),
            "--log-level",
            a.log_level,
        ],
    )


_RUNNERS = {
    "vlm_onnx": _run_vlm_onnx,
    "ae_onnx": _run_ae_onnx,
    "vlm_quant": _run_vlm_quant,
    "ae_quant": _run_ae_quant,
    "vlm_om": _run_vlm_om,
    "ae_om": _run_ae_om,
    "vlm_quant_om": _run_vlm_quant_om,
    "ae_quant_om": _run_ae_quant_om,
    "verify": _run_verify,
}

# Pretty stage titles for the [i/total] progress banner.
_TITLES = {
    "vlm_onnx": "VLM ONNX export",
    "ae_onnx": "Action Expert ONNX export",
    "vlm_quant": "VLM W8A8 quantize",
    "ae_quant": "Action Expert W8A8 quantize",
    "vlm_om": "ATC → OM compile (VLM)",
    "ae_om": "ATC → OM compile (Action Expert)",
    "vlm_quant_om": "ATC → OM compile (VLM W8A8)",
    "ae_quant_om": "ATC → OM compile (Action Expert W8A8)",
    "verify": "Equivalence verification",
}


def _validate_product_deps(ctx: Ctx, chosen: list[str]) -> None:
    """Stop with a precise error if a step's upstream product is unavailable.

    A dependency is satisfied when it is itself in --steps (it will be produced
    this run) OR its product already exists on disk. Otherwise we refuse to run
    and tell the user to add the missing step (D1: no implicit upstream runs).
    """
    chosen_set = set(chosen)
    problems: list[str] = []
    for name in chosen:
        for dep in _cli.STEPS_BY_NAME[name].step_deps:
            if dep in chosen_set:
                continue
            if _product_exists(ctx, dep):
                continue
            dep_prod = _product(ctx, dep)
            where = f" ({dep_prod})" if dep_prod is not None else ""
            problems.append(
                f"  ✗ step '{name}' needs the product of '{dep}', but it is neither in --steps nor on disk{where}"
            )
        if name == "ae_onnx" and "vlm_onnx" not in chosen_set:
            runtime_inputs = (
                ctx.runtime_save_dir / "past_kv_tensor.pth",
                ctx.runtime_save_dir / "prefix_pad_masks.pth",
            )
            missing = [path for path in runtime_inputs if not path.is_file()]
            if missing:
                problems.append(
                    "  ✗ step 'ae_onnx' needs VLM runtime handoff tensors that are not on disk:\n"
                    + "\n".join(f"      {path}" for path in missing)
                )
    if problems:
        raise SystemExit(
            "Unsatisfied step dependencies:\n"
            + "\n".join(problems)
            + "\nTip: add the missing step(s) to --steps (e.g. --steps "
            + ",".join(sorted(chosen_set | {p for p in _cli.STEP_NAMES if any(p in pr for pr in problems)}))
            + "), or point --work-dir at the directory that already holds them."
        )


def _validate_quant_preflight(ctx: Ctx, chosen: list[str]) -> None:
    """Preflight donor/NPU-graph requirements before entering msModelSlim.

    Quantizing an NPU-op ONNX needs Route A: an ORT-runnable donor graph plus
    the NPU deployment graph. If the donor is missing, the quant step can auto-
    export it with --donor-device; this check verifies that generation is
    possible and logs whether the donor will be reused or generated.
    """
    problems: list[str] = []

    def check(role: str, deploy: Path, donor: Path) -> None:
        role_label = "VLM" if role == "vlm" else "Action Expert"
        if not _is_npu_onnx(deploy):
            LOGGER.info("%s quant preflight: deployment ONNX is donor/ORT graph: %s", role_label, deploy)
            return

        LOGGER.info("%s quant preflight: deployment ONNX is NPU graph: %s", role_label, deploy)
        fused_donor = _uses_fused_geglu_donor(ctx, role)
        refresh_donor = f"{role}_onnx" in chosen or fused_donor
        if donor.is_file() and not refresh_donor:
            LOGGER.info("%s quant preflight: donor ONNX exists and will be reused: %s", role_label, donor)
            return

        donor_device = ctx.args.donor_device.split(":", 1)[0]
        if donor_device == "npu":
            problems.append(
                f"  ✗ step '{role}_quant' needs an ORT-runnable donor ONNX, but --donor-device is npu.\n"
                "    Tip: use --donor-device cpu (default) or --donor-device cuda."
            )
            return

        if role == "ae":
            missing = [
                p
                for p in (
                    ctx.runtime_save_dir / "past_kv_tensor.pth",
                    ctx.runtime_save_dir / "prefix_pad_masks.pth",
                )
                if not p.is_file()
            ]
            if missing:
                if "vlm_onnx" in chosen:
                    LOGGER.info(
                        "%s quant preflight: AE donor needs runtime tensors; they are missing now but "
                        "vlm_onnx is in --steps and will generate them first.",
                        role_label,
                    )
                else:
                    problems.append(
                        "  ✗ step 'ae_quant' needs AE donor ONNX, but it is missing and cannot be generated "
                        "because runtime tensors are missing:\n"
                        + "\n".join(f"      {p}" for p in missing)
                        + "\n    Tip: run a VLM export first (e.g. include vlm_onnx in --steps), or point "
                        "--work-dir at existing tensors."
                    )
                    return

        if refresh_donor:
            LOGGER.info(
                "%s quant preflight: donor ONNX will be regenerated because %s_onnx is in --steps: %s",
                role_label,
                role,
                donor,
            )
        else:
            LOGGER.info(
                "%s quant preflight: donor ONNX missing; it will be generated with --donor-device %s: %s",
                role_label,
                ctx.args.donor_device,
                donor,
            )

    if "vlm_quant" in chosen:
        check("vlm", ctx.vlm_onnx, ctx.vlm_donor_onnx)
    if "ae_quant" in chosen:
        check("ae", ctx.ae_onnx, ctx.ae_donor_onnx)

    if problems:
        raise SystemExit("Quantization preflight failed:\n" + "\n".join(problems))


def _validate_quantized_profile_dependencies(ctx: Ctx, chosen: list[str]) -> None:
    """Reject stale profiled W8A8 ONNX files before compiling them."""
    profile = ctx.quantization_profile
    if profile is None:
        return
    checks = (
        ("vlm", "vlm_quant", "vlm_quant_om", ctx.vlm_w8a8),
        ("ae", "ae_quant", "ae_quant_om", ctx.ae_w8a8),
    )
    for role, quant_step, om_step, output_onnx in checks:
        if om_step not in chosen or quant_step in chosen:
            continue
        deploy_onnx = ctx.vlm_onnx if role == "vlm" else ctx.ae_onnx
        if _is_npu_onnx(deploy_onnx):
            donor_onnx = ctx.vlm_donor_onnx if role == "vlm" else ctx.ae_donor_onnx
            npu_onnx = deploy_onnx
        else:
            donor_onnx = deploy_onnx
            npu_onnx = None
        try:
            validate_quantization_metadata(
                path=metadata_path(output_onnx),
                profile=profile,
                role=role,
                policy_path=ctx.policy_path,
                donor_onnx=donor_onnx,
                npu_onnx=npu_onnx,
                output_onnx=output_onnx,
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Cannot reuse {output_onnx}: {exc}") from exc


def _remove_quantized_onnx(path: Path) -> None:
    for candidate in (path, path.with_name(path.name + ".data"), metadata_path(path)):
        candidate.unlink(missing_ok=True)


def main() -> int:
    resolved = _cli.resolve()
    args = resolved.args

    setup_logging(args.log_level)
    _cli.print_effective(resolved)

    policy_path = Path(args.policy_path).expanduser().resolve()
    if not policy_path.is_dir():
        LOGGER.error("--policy-path %s is not a local directory.", policy_path)
        return 1

    try:
        work_dir = export_work_dir(policy_path, "ascend/pi05", args.work_dir)
        output_dir = resolve_outside_bundle_path(policy_path, work_dir / "onnx")
        runtime_save_dir = resolve_outside_bundle_path(policy_path, work_dir / "runtime_save")
        om_dir = resolve_outside_bundle_path(policy_path, work_dir / "om")
        calibration_dir = resolve_outside_bundle_path(policy_path, work_dir / "calibration" / "ae")
        if args.amp_scratch_dir:
            args.amp_scratch_dir = str(resolve_outside_bundle_path(policy_path, args.amp_scratch_dir))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    device_tag = args.device.split(":", 1)[0]
    suffix = build_onnx_suffix(dtype=args.dtype, device=device_tag)
    donor_device_tag = args.donor_device.split(":", 1)[0]
    quantization_profile = getattr(resolved, "quantization_profile", None)
    vlm_donor_dtype = quantization_profile.vlm.donor_dtype if quantization_profile else None
    ae_donor_dtype = quantization_profile.action_expert.donor_dtype if quantization_profile else None
    vlm_donor_suffix = build_onnx_suffix(dtype=vlm_donor_dtype or args.dtype, device=donor_device_tag)
    ae_donor_suffix = build_onnx_suffix(dtype=ae_donor_dtype or args.dtype, device=donor_device_tag)
    fused_vlm_donor = bool(quantization_profile and quantization_profile.vlm.fused_geglu_donor)
    fused_ae_donor = bool(quantization_profile and quantization_profile.action_expert.fused_geglu_donor)
    vlm_onnx = output_dir / f"pi05-vlm{suffix}.onnx"
    ae_onnx = output_dir / f"pi05-action_expert{suffix}.onnx"
    vlm_w8a8 = vlm_onnx.with_name(vlm_onnx.stem + "_w8a8.onnx")
    ae_w8a8 = ae_onnx.with_name(ae_onnx.stem + "_w8a8.onnx")

    chosen = _cli.parse_steps(args.steps)  # registry order, validated in resolve()
    ctx = Ctx(
        args=args,
        policy_path=policy_path,
        output_dir=output_dir,
        runtime_save_dir=runtime_save_dir,
        om_dir=om_dir,
        vlm_onnx=vlm_onnx,
        ae_onnx=ae_onnx,
        vlm_donor_onnx=output_dir / f"{_donor_model_stem('vlm', fused_vlm_donor)}{vlm_donor_suffix}.onnx",
        ae_donor_onnx=output_dir / f"{_donor_model_stem('ae', fused_ae_donor)}{ae_donor_suffix}.onnx",
        vlm_w8a8=vlm_w8a8,
        ae_w8a8=ae_w8a8,
        vlm_om=om_dir / vlm_onnx.with_suffix(".om").name,
        ae_om=om_dir / ae_onnx.with_suffix(".om").name,
        vlm_quant_om=om_dir / vlm_w8a8.with_suffix(".om").name,
        ae_quant_om=om_dir / ae_w8a8.with_suffix(".om").name,
        calibration_dir=calibration_dir,
        chosen=set(chosen),
        quantization_profile=quantization_profile,
    )

    # Fail fast on missing upstream artifacts before doing any work.
    _validate_product_deps(ctx, chosen)
    _validate_quant_preflight(ctx, chosen)

    # Q2=c: every step listed in --steps is executed (a pre-existing product is
    # rebuilt, not skipped). _validate_product_deps already guaranteed each
    # step's upstream products are available, so there are no implicit steps to
    # add and nothing here is skipped — the plan is exactly the chosen steps in
    # registry (dependency) order.
    plan = list(chosen)
    total = len(plan)
    summary: list[tuple[str, str]] = []

    for step_no, name in enumerate(plan, start=1):
        prod = _product(ctx, name)
        if prod is not None and prod.is_file():
            LOGGER.info("● [%d/%d] %s — rebuilding (requested; existing %s)", step_no, total, _TITLES[name], prod)
        if prod is not None:
            if name in {"vlm_quant", "ae_quant"}:
                _remove_quantized_onnx(prod)
            else:
                prod.unlink(missing_ok=True)
            if name.endswith("_om"):
                Path(f"{prod}.abi.json").unlink(missing_ok=True)
        if name == "vlm_onnx":
            (ctx.runtime_save_dir / "past_kv_tensor.pth").unlink(missing_ok=True)
            (ctx.runtime_save_dir / "prefix_pad_masks.pth").unlink(missing_ok=True)
        try:
            with Stage(_TITLES[name], index=step_no, total=total):
                _RUNNERS[name](ctx)
        except Exception:
            if prod is not None and name in {"vlm_quant", "ae_quant"}:
                _remove_quantized_onnx(prod)
            raise
        if prod is not None and not prod.is_file():
            raise RuntimeError(f"Step {name!r} completed without producing {prod}")
        _validate_step_outputs(ctx, name)
        _append_summary(summary, ctx, name)

    vlm_step = "vlm_quant_om" if "vlm_quant_om" in ctx.chosen else "vlm_om"
    ae_step = "ae_quant_om" if "ae_quant_om" in ctx.chosen else "ae_om"
    vlm_onnx_path = ctx.vlm_w8a8 if vlm_step == "vlm_quant_om" else ctx.vlm_onnx
    ae_onnx_path = ctx.ae_w8a8 if ae_step == "ae_quant_om" else ctx.ae_onnx
    vlm_om_path = ctx.vlm_quant_om if vlm_step == "vlm_quant_om" else ctx.vlm_om
    ae_om_path = ctx.ae_quant_om if ae_step == "ae_quant_om" else ctx.ae_om
    if ctx.chosen & {"vlm_om", "ae_om", "vlm_quant_om", "ae_quant_om"} and _finalize_deployment(
        ctx,
        args.deployment,
        vlm_step,
        vlm_onnx_path,
        vlm_om_path,
        ae_step,
        ae_onnx_path,
        ae_om_path,
    ):
        summary.append(("Inference manifest", f"{ctx.policy_path / 'inference_manifest.json'} ({args.deployment})"))

    print_summary("PI05 export pipeline complete", _dedup(summary), status="✅ DONE")
    LOGGER.info("Intermediate products kept under %s, %s, and %s", output_dir, runtime_save_dir, om_dir)

    _cli.write_last(resolved)
    return 0


def _finalize_deployment(
    ctx: Ctx,
    deployment: str,
    vlm_step: str,
    vlm_onnx: Path,
    vlm_om: Path,
    action_step: str,
    ae_onnx: Path,
    ae_om: Path,
) -> bool:
    reuse_roles: set[str] = set()
    if vlm_step not in ctx.chosen:
        reuse_roles.add("vlm")
    if action_step not in ctx.chosen:
        reuse_roles.update({"action_expert", "denoising_schedule"})
    required = {role: path for role, path in (("vlm", vlm_om), ("action_expert", ae_om)) if role not in reuse_roles}
    missing = [(role, path) for role, path in required.items() if not path.is_file()]
    manifest_path = ctx.policy_path / "inference_manifest.json"
    if reuse_roles and not _manifest_has_deployment(manifest_path, deployment):
        missing.extend((role, manifest_path) for role in sorted(reuse_roles & {"vlm", "action_expert"}))
    if missing:
        details = ", ".join(f"{role}={path}" for role, path in missing)
        LOGGER.warning(
            "Deployment %s is not finalized because required new or reusable artifacts are missing (%s)",
            deployment,
            details,
        )
        return False
    _run_module(
        "model_utils.pi05_export.convert_om",
        [
            "--pretrained-policy-path",
            str(ctx.policy_path),
            "--soc-version",
            ctx.args.soc_version,
            "--vlm-onnx",
            str(vlm_onnx),
            "--vlm-om",
            str(vlm_om),
            "--ae-onnx",
            str(ae_onnx),
            "--ae-om",
            str(ae_om),
            "--deployment",
            deployment,
            "--abi-device-id",
            str(ctx.args.abi_device_id),
            *(["--acl-config-path", ctx.args.acl_config_path] if ctx.args.acl_config_path else []),
            *(["--schedule-file", ctx.args.schedule_file] if ctx.args.schedule_file else []),
            *[value for role in sorted(reuse_roles) for value in ("--reuse-artifact-role", role)],
            "--manifest-only",
            "--no-summary",
            "--log-level",
            ctx.args.log_level,
        ],
    )
    return True


def _manifest_has_deployment(manifest_path: Path, deployment: str) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    deployments = manifest.get("deployments") if isinstance(manifest, dict) else None
    return isinstance(deployments, dict) and deployment in deployments


def _validate_step_outputs(ctx: Ctx, step: str) -> None:
    required: tuple[Path, ...] = ()
    if step == "vlm_onnx":
        required = (
            ctx.runtime_save_dir / "past_kv_tensor.pth",
            ctx.runtime_save_dir / "prefix_pad_masks.pth",
        )
    elif step.endswith("_om"):
        product = _product(ctx, step)
        assert product is not None
        required = (Path(f"{product}.abi.json"),)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Step {step!r} did not produce required artifacts: {missing}")


def _dedup(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop duplicate summary rows (e.g. the OM manifest from both *_om steps)."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for row in rows:
        if row not in seen:
            seen.add(row)
            out.append(row)
    return out


def _append_summary(summary: list[tuple[str, str]], ctx: Ctx, step: str) -> None:
    """Add this step's user-facing artifact(s) to the final summary block."""
    if step == "vlm_onnx":
        summary.append(("VLM ONNX", str(ctx.vlm_onnx)))
    elif step == "ae_onnx":
        summary.append(("Action Expert ONNX", str(ctx.ae_onnx)))
    elif step == "vlm_quant":
        summary.append(("VLM ONNX (W8A8)", str(ctx.vlm_w8a8)))
    elif step == "ae_quant":
        summary.append(("Action Expert ONNX (W8A8)", str(ctx.ae_w8a8)))
    elif step == "vlm_om":
        summary.append(("VLM OM", str(ctx.vlm_om)))
    elif step == "ae_om":
        summary.append(("Action Expert OM", str(ctx.ae_om)))
    elif step == "vlm_quant_om":
        summary.append(("VLM OM (W8A8)", str(ctx.vlm_quant_om)))
    elif step == "ae_quant_om":
        summary.append(("Action Expert OM (W8A8)", str(ctx.ae_quant_om)))
    elif step == "verify":
        summary.append(("Verification", "✅ see log above"))


def console_main() -> None:
    """Entry point for the ``pi05-export`` console script.

    Wraps :func:`main` with the same friendly CalledProcessError handling the
    ``python -m`` invocation gets, then exits with the stage return code.
    """
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        LOGGER.error(
            "Pipeline stopped: a step exited with code %s. Fix the error above and re-run "
            "(only the steps listed in --steps are executed).",
            exc.returncode,
        )
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    console_main()
