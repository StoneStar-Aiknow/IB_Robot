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
    vlm_quant  Quantize the VLM ONNX to W8A8           (needs --batch-path)
    ae_quant   Quantize the Action Expert ONNX to W8A8 (calib = runtime_save)
    vlm_om     Compile the VLM ONNX to OM via ATC      (needs --soc-version)
    ae_om      Compile the Action Expert ONNX to OM    (needs --soc-version)
    verify     Split-vs-monolithic equivalence check   (needs --task)

Default ``--steps`` = ``vlm_onnx,ae_onnx,vlm_om,ae_om`` (export both segments
and compile both OMs). Quantization and verification are opt-in by listing the
corresponding step.

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
* When a segment is quantized (``*_quant`` in --steps), that segment's ``*_om``
  step automatically compiles the ``*_w8a8.onnx`` instead of the fp16 ONNX.

Design notes
------------
* **Intermediate products are preserved.** Nothing is deleted between stages;
  the ONNX files, ``runtime_save/*.pth`` and ``config.om.json`` all remain on
  disk for inspection or a partial re-run.
* **Live feedback.** Stages run as child processes with inherited stdout/stderr,
  so the user sees real progress (export logs, ATC compile output).
* **Minimal surface.** All argument ergonomics (profile / wizard / --exp-dir
  derivation / remember-last) live in ``_cli``; this entry point only plans and
  runs steps.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from model_utils.pi05_export import _cli
from model_utils.pi05_export._cli_ui import Stage, build_onnx_suffix, print_summary, setup_logging

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
    vlm_onnx: Path
    ae_onnx: Path
    vlm_donor_onnx: Path
    ae_donor_onnx: Path
    vlm_w8a8: Path
    ae_w8a8: Path
    vlm_om: Path
    ae_om: Path
    calib_dir: Path
    chosen: set[str]  # steps explicitly requested in --steps

    def vlm_atc_onnx(self) -> Path:
        """ONNX that vlm_om should compile: the W8A8 graph iff vlm was quantized."""
        return self.vlm_w8a8 if "vlm_quant" in self.chosen else self.vlm_onnx

    def ae_atc_onnx(self) -> Path:
        return self.ae_w8a8 if "ae_quant" in self.chosen else self.ae_onnx


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


def _run_vlm_donor_onnx(ctx: Ctx) -> None:
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
            a.donor_device,
            "--log-level",
            a.log_level,
        ],
    )


def _run_ae_donor_onnx(ctx: Ctx) -> None:
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
            a.donor_device,
            "--log-level",
            a.log_level,
        ],
    )


def _quant_inputs(ctx: Ctx, *, role: str) -> tuple[Path, Path | None]:
    """Return (donor_onnx, npu_onnx) for quantization, generating donor if needed."""
    deploy_onnx = ctx.vlm_onnx if role == "vlm" else ctx.ae_onnx
    donor_onnx = ctx.vlm_donor_onnx if role == "vlm" else ctx.ae_donor_onnx
    role_label = "VLM" if role == "vlm" else "Action Expert"

    if _is_npu_onnx(deploy_onnx):
        LOGGER.info("%s quant: deployment ONNX is NPU graph: %s", role_label, deploy_onnx)
        if donor_onnx.is_file():
            LOGGER.info("%s quant: reusing donor ONNX: %s", role_label, donor_onnx)
        else:
            LOGGER.info(
                "%s quant: donor ONNX missing; generating with --donor-device %s: %s",
                role_label,
                ctx.args.donor_device,
                donor_onnx,
            )
            if role == "vlm":
                _run_vlm_donor_onnx(ctx)
            else:
                _run_ae_donor_onnx(ctx)
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
            "--device",
            a.device,
            *(["--task", a.task] if a.task else []),
            *(["--npu-onnx-path", str(npu_onnx)] if npu_onnx else []),
            "--log-level",
            a.log_level,
        ],
    )


def _run_ae_quant(ctx: Ctx) -> None:
    a = ctx.args
    donor_onnx, npu_onnx = _quant_inputs(ctx, role="ae")
    _run_module(
        "model_utils.pi05_export.quant.quantize_ae",
        [
            "--onnx-path",
            str(donor_onnx),
            "--output-path",
            str(ctx.ae_w8a8),
            "--calib-dir",
            str(ctx.calib_dir),
            "--num-calib",
            str(a.num_calib),
            "--amp-num",
            str(a.amp_num),
            *(["--npu-onnx-path", str(npu_onnx)] if npu_onnx else []),
            "--log-level",
            a.log_level,
        ],
    )


def _run_vlm_om(ctx: Ctx) -> None:
    a = ctx.args
    _run_module(
        "model_utils.pi05_export.convert_om",
        [
            "--pretrained-policy-path",
            str(ctx.policy_path),
            "--soc-version",
            a.soc_version,
            "--vlm-onnx",
            str(ctx.vlm_atc_onnx()),
            "--vlm-om",
            str(ctx.vlm_om),
            "--no-summary",
            "--log-level",
            a.log_level,
        ],
    )


def _run_ae_om(ctx: Ctx) -> None:
    a = ctx.args
    _run_module(
        "model_utils.pi05_export.convert_om",
        [
            "--pretrained-policy-path",
            str(ctx.policy_path),
            "--soc-version",
            a.soc_version,
            "--ae-onnx",
            str(ctx.ae_atc_onnx()),
            "--ae-om",
            str(ctx.ae_om),
            "--no-summary",
            "--log-level",
            a.log_level,
        ],
    )


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
            "--task",
            a.task,
            "--device",
            a.device,
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
    if problems:
        raise SystemExit(
            "Unsatisfied step dependencies:\n"
            + "\n".join(problems)
            + "\nTip: add the missing step(s) to --steps (e.g. --steps "
            + ",".join(sorted(chosen_set | {p for p in _cli.STEP_NAMES if any(p in pr for pr in problems)}))
            + "), or point --exp-dir/--output-dir at the directory that already holds them."
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
        if donor.is_file():
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
                        "--runtime-save-dir/--exp-dir at existing tensors."
                    )
                    return

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


def main() -> int:
    resolved = _cli.resolve()
    args = resolved.args

    setup_logging(args.log_level)
    _cli.print_effective(resolved)

    policy_path = Path(args.policy_path).expanduser()
    if not policy_path.is_dir():
        LOGGER.error("--policy-path %s is not a local directory.", policy_path)
        return 1

    output_dir = Path(args.output_dir).expanduser()
    runtime_save_dir = Path(args.runtime_save_dir).expanduser()
    device_tag = args.device.split(":", 1)[0]
    suffix = build_onnx_suffix(dtype=args.dtype, device=device_tag)
    donor_device_tag = args.donor_device.split(":", 1)[0]
    donor_suffix = build_onnx_suffix(dtype=args.dtype, device=donor_device_tag)
    vlm_onnx = output_dir / f"pi05-vlm{suffix}.onnx"
    ae_onnx = output_dir / f"pi05-action_expert{suffix}.onnx"
    vlm_w8a8 = vlm_onnx.with_name(vlm_onnx.stem + "_w8a8.onnx")
    ae_w8a8 = ae_onnx.with_name(ae_onnx.stem + "_w8a8.onnx")

    chosen = _cli.parse_steps(args.steps)  # registry order, validated in resolve()
    vlm_atc_onnx = vlm_w8a8 if "vlm_quant" in chosen else vlm_onnx
    ae_atc_onnx = ae_w8a8 if "ae_quant" in chosen else ae_onnx
    ctx = Ctx(
        args=args,
        policy_path=policy_path,
        output_dir=output_dir,
        runtime_save_dir=runtime_save_dir,
        vlm_onnx=vlm_onnx,
        ae_onnx=ae_onnx,
        vlm_donor_onnx=output_dir / f"pi05-vlm{donor_suffix}.onnx",
        ae_donor_onnx=output_dir / f"pi05-action_expert{donor_suffix}.onnx",
        vlm_w8a8=vlm_w8a8,
        ae_w8a8=ae_w8a8,
        vlm_om=policy_path / vlm_atc_onnx.with_suffix(".om").name,
        ae_om=policy_path / ae_atc_onnx.with_suffix(".om").name,
        calib_dir=(Path(args.calib_dir).expanduser() if args.calib_dir else runtime_save_dir),
        chosen=set(chosen),
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
        with Stage(_TITLES[name], index=step_no, total=total):
            _RUNNERS[name](ctx)
        _append_summary(summary, ctx, name)

    print_summary("PI05 export pipeline complete", _dedup(summary), status="✅ DONE")
    LOGGER.info("Intermediate products kept under %s and %s", output_dir, runtime_save_dir)

    _cli.write_last(resolved)
    return 0


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
        summary.append(("OM manifest", str(ctx.policy_path / "config.om.json")))
    elif step == "ae_om":
        summary.append(("Action Expert OM", str(ctx.ae_om)))
        summary.append(("OM manifest", str(ctx.policy_path / "config.om.json")))
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
