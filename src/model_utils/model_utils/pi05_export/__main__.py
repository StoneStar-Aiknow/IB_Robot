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
keep ``--dtype`` consistent across two exports, or hand-write the ATC command:

    1. VLM ONNX export       (convert_onnx_vlm)
    2. Action Expert export  (convert_onnx_action_expert)
    3. ATC → OM compile      (convert_om)           [only with --soc-version]
    4. Equivalence verify    (verify_pi05_split_equivalence)  [only with --verify]

Design notes
------------
* **Resumable.** Each stage's output paths are predicted up-front; if the
  artifact already exists the stage is skipped (``▷ skip``). Re-running after a
  mid-pipeline failure therefore picks up where it left off. Use ``--force`` to
  rebuild everything.
* **Intermediate products are preserved.** Nothing is deleted between stages;
  the ONNX files, ``runtime_save/*.pth`` and ``config.om.json`` all remain on
  disk for inspection or a partial re-run.
* **Live feedback.** Stages run as child processes with inherited stdout/stderr,
  so the user sees real progress (export logs, ATC compile output) and never
  wonders whether the tool is stuck.
* **Minimal surface.** Only the handful of decisions a user actually makes are
  exposed (policy path, dtype, whether to compile / verify). All the internal
  export knobs keep their proven defaults.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from model_utils.pi05_export import _cli
from model_utils.pi05_export._cli_ui import Stage, build_onnx_suffix, print_summary, setup_logging

LOGGER = logging.getLogger("pi05_export.pipeline")


def _run_module(module: str, cli_args: list[str]) -> None:
    """Run ``python -m <module> <args>`` as a child process, streaming output.

    Raises CalledProcessError on non-zero exit so the orchestrator stops and the
    user can fix the issue and re-run (already-finished stages will be skipped).
    """
    command = [sys.executable, "-m", module, *cli_args]
    LOGGER.info("  $ %s", " ".join(command))
    subprocess.run(command, check=True)  # nosec B603 — args are program-controlled.


def main() -> int:
    # All argument ergonomics (profile / wizard / --exp-dir derivation /
    # remember-last) live in _cli so this entry point stays focused on stage
    # orchestration. Every historical explicit flag still works and overrides
    # whatever a profile/derivation would supply.
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
    # Bare device type (cpu / cuda / npu) is part of the exported filename.
    device_tag = args.device.split(":", 1)[0]
    suffix = build_onnx_suffix(dtype=args.dtype, device=device_tag)
    vlm_onnx = output_dir / f"pi05-vlm{suffix}.onnx"
    ae_onnx = output_dir / f"pi05-action_expert{suffix}.onnx"

    # Count stages for the [i/total] progress prefix.
    total = 2 + (1 if args.soc_version else 0) + (1 if args.verify else 0)
    step = 0

    summary: list[tuple[str, str]] = []

    # ---- Stage 1: VLM ONNX export ----
    step += 1
    if vlm_onnx.is_file() and not args.force:
        LOGGER.info("▷ [%d/%d] VLM export — skip (exists: %s)", step, total, vlm_onnx)
    else:
        with Stage("VLM ONNX export", index=step, total=total):
            _run_module(
                "model_utils.pi05_export.convert_onnx_vlm",
                [
                    "--pretrained-policy-path",
                    str(policy_path),
                    "--output-dir",
                    str(output_dir),
                    "--runtime-save-dir",
                    str(runtime_save_dir),
                    "--dtype",
                    args.dtype,
                    "--device",
                    args.device,
                    "--log-level",
                    args.log_level,
                ],
            )
    summary.append(("VLM ONNX", str(vlm_onnx)))

    # ---- Stage 2: Action Expert ONNX export ----
    step += 1
    if ae_onnx.is_file() and not args.force:
        LOGGER.info("▷ [%d/%d] Action Expert export — skip (exists: %s)", step, total, ae_onnx)
    else:
        with Stage("Action Expert ONNX export", index=step, total=total):
            _run_module(
                "model_utils.pi05_export.convert_onnx_action_expert",
                [
                    "--pretrained-policy-path",
                    str(policy_path),
                    "--output-dir",
                    str(output_dir),
                    "--past-kv-path",
                    str(runtime_save_dir / "past_kv_tensor.pth"),
                    "--prefix-pad-masks-path",
                    str(runtime_save_dir / "prefix_pad_masks.pth"),
                    "--dtype",
                    args.dtype,
                    "--device",
                    args.device,
                    "--log-level",
                    args.log_level,
                ],
            )
    summary.append(("Action Expert ONNX", str(ae_onnx)))

    # ---- Stage 3: ATC → OM (optional) ----
    if args.soc_version:
        step += 1
        vlm_om = policy_path / "vlm.om"
        ae_om = policy_path / "action_expert.om"
        if vlm_om.is_file() and ae_om.is_file() and not args.force:
            LOGGER.info("▷ [%d/%d] ATC compile — skip (both .om exist)", step, total)
        else:
            with Stage("ATC → OM compile", index=step, total=total):
                _run_module(
                    "model_utils.pi05_export.convert_om",
                    [
                        "--pretrained-policy-path",
                        str(policy_path),
                        "--soc-version",
                        args.soc_version,
                        "--vlm-onnx",
                        str(vlm_onnx),
                        "--ae-onnx",
                        str(ae_onnx),
                        "--log-level",
                        args.log_level,
                    ],
                )
        summary.append(("VLM OM", str(vlm_om)))
        summary.append(("Action Expert OM", str(ae_om)))
        summary.append(("OM manifest", str(policy_path / "config.om.json")))

    # ---- Stage 4: Verify (optional) ----
    if args.verify:
        step += 1
        with Stage("Equivalence verification", index=step, total=total):
            _run_module(
                "model_utils.pi05_export.verify_pi05_split_equivalence",
                [
                    "--pretrained-policy-path",
                    str(policy_path),
                    "--vlm-onnx-path",
                    str(vlm_onnx),
                    "--ae-onnx-path",
                    str(ae_onnx),
                    "--task",
                    args.task,
                    "--device",
                    args.device,
                    "--log-level",
                    args.log_level,
                ],
            )
        summary.append(("Verification", "✅ see log above"))

    print_summary("PI05 export pipeline complete", summary, status="✅ DONE")
    LOGGER.info("Intermediate products kept under %s and %s", output_dir, runtime_save_dir)

    # Persist this run's effective params as ``_last`` (remember-last). Only
    # after a successful pipeline so a failed run never poisons the cache.
    _cli.write_last(resolved)
    return 0


def console_main() -> None:
    """Entry point for the ``pi05-export`` console script.

    Wraps :func:`main` with the same friendly CalledProcessError handling the
    ``python -m`` invocation gets, then exits with the stage return code.
    """
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        LOGGER.error(
            "Pipeline stopped: a stage exited with code %s. Fix the error above and re-run the "
            "same command — finished stages are skipped automatically (or pass --force to rebuild).",
            exc.returncode,
        )
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    console_main()
