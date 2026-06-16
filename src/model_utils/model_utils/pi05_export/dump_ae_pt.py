# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Licensed under the Mulan PSL v2.
"""Dump the full AE (Action Expert) Euler trajectory for comparison.

Pairs with :class:`PI05OMModel`'s ``PI05_OM_DUMP_AE`` machinery.

Given a *fixed* ``past_kv_tensor`` + ``prefix_pad_masks`` + ``noise``, run
:meth:`PI05ActionExpertPolicy.model.sample_actions` for ``num_steps``
Euler steps (default: ``policy.config.num_inference_steps`` = 10 for
PI05) and save ``x_t_step{i:02d}.npy`` after every step.

This isolates the Action Expert from the VLM and gives a per-step
trajectory.  When PT and OM are fed identical inputs, ``compare()``
prints a table of cosine / L1 / Linf at each step so you can pinpoint
exactly where the trajectories start to diverge — essential for
diagnosing fp16 chaos vs single-step algebraic bugs.

Recommended workflow
--------------------

1. **VLM dump** (already done) gives us ``past_kv_tensor.npy`` and
   ``prefix_pad_masks.npy`` for batch *i*.

2. **Generate noise** once on the GPU machine and save it (loss_compare's
   ``--noise-dir`` does this). Use ``noise_0000.npy`` for batch 0.

3. **PT AE dump** (this script, GPU machine)::

       python -m model_utils.pi05_export.dump_ae_pt \\
           --policy-path  /path/to/pi05_ckpt \\
           --past-kv      /tmp/pt_vlm_dump_0/past_kv_tensor.npy \\
           --pad-masks    /tmp/pt_vlm_dump_0/prefix_pad_masks.npy \\
           --noise        /path/to/noise_dir/noise_0000.npy \\
           --out-dir      /tmp/pt_ae_dump_0 \\
           --device       cuda

4. **OM AE dump** (NPU machine): set ``PI05_OM_DUMP_AE=/tmp/om_ae_dump_0``
   when running loss_compare; PI05OMModel will save ``x_t_step{i:02d}.npy``
   for *every* Euler step of the *first* ``forward()`` call.

5. **Compare** (per-step trajectory table)::

       python -m model_utils.pi05_export.dump_ae_pt \\
           --compare-pt /tmp/pt_ae_dump_0 \\
           --compare-om /tmp/om_ae_dump_0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

LOGGER = logging.getLogger("dump_ae_pt")


# ---------------------------------------------------------------------------
# Dump (PT)
# ---------------------------------------------------------------------------


def dump(
    *,
    policy_path: str,
    past_kv_path: str,
    pad_masks_path: str,
    noise_path: str | None,
    out_dir: str,
    device_str: str,
    num_steps: int | None = None,
    model_dtype: str = "native",
) -> None:
    """Run the full AE Euler trajectory in PyTorch and dump every step.

    Saves ``x_t_step{i:02d}.npy`` for ``i = 0 … num_steps - 1``.  Each file
    contains ``x_t`` *after* Euler step ``i`` so files line up directly
    with PI05OMModel's per-step dump (set ``PI05_OM_DUMP_AE`` on the NPU
    side).
    """
    from model_utils.pi05_export.modeling_pi05_action_expert import (
        PI05ActionExpertPolicy,
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str)

    LOGGER.info("Loading PI05ActionExpertPolicy from %s …", policy_path)
    policy = PI05ActionExpertPolicy.from_pretrained(
        policy_path,
        local_files_only=False,
        strict=False,
    )
    policy.to(device)
    policy.eval()

    # Optional dtype cast — mirror loss_compare's --model_dtype so the
    # AE trajectory dump uses the SAME dtype as the e2e test that
    # produced the -0.6 cosine.  Without this we'd be comparing OM (fp16)
    # against PT (BF16 native) and trajectory drift would be polluted by
    # BF16↔FP16 conversion error rather than a real bug.
    if model_dtype == "fp16":
        policy.model = policy.model.half()
        LOGGER.info("Cast policy.model to float16")
    elif model_dtype == "bf16":
        policy.model = policy.model.bfloat16()
        LOGGER.info("Cast policy.model to bfloat16")
    elif model_dtype == "fp32":
        policy.model = policy.model.float()
        LOGGER.info("Cast policy.model to float32")
    elif model_dtype != "native":
        raise ValueError(f"unknown --model-dtype: {model_dtype}")

    cfg = policy.config
    if num_steps is None:
        num_steps = cfg.num_inference_steps
    LOGGER.info("Trajectory length: %d Euler step(s) (cfg.num_inference_steps=%d)", num_steps, cfg.num_inference_steps)

    # ---- Load inputs ----
    LOGGER.info("Loading past_kv from %s", past_kv_path)
    past_kv_np = np.load(past_kv_path)
    LOGGER.info("  past_kv shape=%s dtype=%s", past_kv_np.shape, past_kv_np.dtype)

    # If shape is OM's (L*2, B, S, D) reshape to PT's (L, 2, B, 1, S, D) by
    # inferring L from the leading dim. The unflatten_kv helper expects the
    # PT layout.
    if past_kv_np.ndim == 4:
        first = past_kv_np.shape[0]
        if first % 2 != 0:
            raise ValueError(f"OM-style past_kv first dim ({first}) is not divisible by 2")
        n_layers = first // 2
        bsize, seq, head_dim = past_kv_np.shape[1:]
        past_kv_np = past_kv_np.reshape(n_layers, 2, bsize, 1, seq, head_dim)
        LOGGER.info("  reshaped to %s (assumed L=%d, H=1)", past_kv_np.shape, n_layers)

    LOGGER.info("Loading prefix_pad_masks from %s", pad_masks_path)
    pad_masks_np = np.load(pad_masks_path)
    LOGGER.info("  pad_masks shape=%s dtype=%s", pad_masks_np.shape, pad_masks_np.dtype)

    # ---- Build noise ----
    bsize = pad_masks_np.shape[0]
    noise_shape = (bsize, cfg.chunk_size, cfg.max_action_dim)
    if noise_path is not None:
        LOGGER.info("Loading noise from %s", noise_path)
        noise_np = np.load(noise_path)
        if tuple(noise_np.shape) != noise_shape:
            raise ValueError(f"Noise shape {noise_np.shape} != expected {noise_shape}")
    else:
        LOGGER.info("Generating deterministic noise (seed=42)")
        torch.manual_seed(42)
        noise_np = torch.normal(mean=0.0, std=1.0, size=noise_shape, dtype=torch.float32).numpy()

    # ---- Convert to tensors on device ----
    # past_kv must match the action-expert parameter dtype (typically BF16
    # from the checkpoint), otherwise eager_attention_forward fails with
    # "expected scalar type BFloat16 but found Float".  Noise can stay in
    # fp32 — sample_actions casts it internally as needed.
    try:
        model_dtype_t = next(policy.model.parameters()).dtype
    except (StopIteration, AttributeError):
        model_dtype_t = torch.float32
    LOGGER.info("Action-expert weight dtype: %s — casting past_kv & noise accordingly", model_dtype_t)

    past_kv_t = torch.from_numpy(past_kv_np).to(device=device, dtype=model_dtype_t)
    pad_masks_t = torch.from_numpy(pad_masks_np).to(device).bool()
    x_t = torch.from_numpy(noise_np).to(device=device, dtype=model_dtype_t)

    # ------------------------------------------------------------------
    # Dump AE inputs (KV / pad_masks / noise) BEFORE the loop.  File
    # names match PI05OMModel so dump_ae_pt --compare can pair them up.
    # KV is saved in fp16 + the original loaded layout (same as OM dump).
    # ------------------------------------------------------------------
    try:
        np.save(out_path / "ae_in_past_kv.npy", past_kv_t.detach().cpu().to(torch.float16).numpy())
        np.save(out_path / "ae_in_prefix_pad_masks.npy", pad_masks_t.detach().cpu().bool().numpy())
        np.save(out_path / "ae_in_noise.npy", x_t.detach().cpu().to(torch.float16).numpy())
        LOGGER.info("Dumped AE inputs (past_kv, prefix_pad_masks, noise) under %s", out_path)
    except Exception as exc:
        LOGGER.warning("AE input dump failed: %s", exc)

    # Mirror PI05OMModel.forward() time/dt semantics.
    # dt and time stepping run in fp32 here for clarity; OM keeps them in
    # the model's native dtype (fp16) but the trajectory comparison is
    # against the *output* x_t at each step, not against the time scalar
    # itself, so this fp32 stepping is fine.
    dt_val = -1.0 / num_steps
    time_val = 1.0

    LOGGER.info("Running AE for %d Euler step(s) …", num_steps)
    with torch.no_grad():
        for step_idx in range(num_steps):
            time_t = torch.tensor([time_val], dtype=model_dtype_t, device=device)

            # Per-step time dump (captures fp16 accumulation drift).
            try:
                np.save(
                    out_path / f"ae_in_time_step{step_idx:02d}.npy",
                    time_t.detach().cpu().to(torch.float16).numpy(),
                )
            except Exception as exc:
                LOGGER.warning("AE time dump (step %d) failed: %s", step_idx, exc)

            x_t = policy.model.sample_actions(past_kv_t, pad_masks_t, time_t, x_t)

            # Save x_t AFTER Euler step `step_idx` — matches OM dump layout.
            x_t_np = x_t.detach().cpu().to(torch.float16).numpy()
            save_path = out_path / f"x_t_step{step_idx:02d}.npy"
            np.save(save_path, x_t_np)

            a32 = x_t_np.astype(np.float32)
            LOGGER.info(
                "  step %02d  t=%+.4f  shape=%s  min=%+.4g max=%+.4g mean=%+.4g std=%+.4g  -> %s",
                step_idx,
                time_val,
                x_t_np.shape,
                float(a32.min()),
                float(a32.max()),
                float(a32.mean()),
                float(a32.std()),
                save_path.name,
            )

            time_val += dt_val


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def compare(*, pt_dir: str, om_dir: str, action_dim: int | None = None) -> None:
    """Per-step PT-vs-OM trajectory comparison.

    Enumerates ``x_t_step*.npy`` present in *both* directories, computes
    cosine / L1 / Linf at every step, and prints a single table.  This is
    the central tool for pinpointing where a 10-step Euler trajectory
    starts to diverge between PT and the deployed OM.

    When ``action_dim`` is given, an additional table is printed that
    restricts the comparison to ``[..., :action_dim]`` of every dump.
    Rationale: PI05 pads its action chunk from the real action width
    (e.g. 6) up to ``max_action_dim`` (32) with zeros.  If the PT and
    OM models both leave the trailing 26 dims near zero, those dims
    contribute almost nothing to the inner product but a lot to both
    norms — the *full-tensor* cosine then looks artificially high
    (e.g. 0.99) even when the 6 real dims have collapsed to ~0.5
    correlation, exactly the gap observed against ``loss_compare``.
    """
    import re

    pt = Path(pt_dir)
    om = Path(om_dir)
    pat = re.compile(r"^x_t_step(\d+)\.npy$")

    def _index_files(d: Path) -> dict[int, Path]:
        out: dict[int, Path] = {}
        if not d.is_dir():
            return out
        for f in d.iterdir():
            m = pat.match(f.name)
            if m:
                out[int(m.group(1))] = f
        return out

    pt_files = _index_files(pt)
    om_files = _index_files(om)
    common = sorted(set(pt_files) & set(om_files))

    print(f"\nComparing PT={pt}  vs  OM={om}")
    print(f"  PT steps: {sorted(pt_files)}")
    print(f"  OM steps: {sorted(om_files)}")
    if not common:
        print("  ❌ No overlapping x_t_step*.npy files — nothing to compare.")
        return
    print(f"  Overlapping steps: {common}\n")

    print(
        f"{'step':>4} {'shape':>20} {'cosine':>12} {'L1':>12} "
        f"{'Linf':>12} {'PT mean':>10} {'OM mean':>10} {'PT std':>10} "
        f"{'OM std':>10}"
    )
    print("-" * 116)

    for step in common:
        pt_arr = np.load(pt_files[step])
        om_arr = np.load(om_files[step])

        if pt_arr.shape != om_arr.shape:
            if pt_arr.size == om_arr.size:
                pt_arr = pt_arr.reshape(-1)
                om_arr = om_arr.reshape(-1)
                shape_str = f"flat({pt_arr.size})"
            else:
                print(f"{step:>4}  SHAPE MISMATCH pt={pt_arr.shape} om={om_arr.shape} — skipped")
                continue
        else:
            shape_str = str(tuple(pt_arr.shape))

        pt32 = pt_arr.astype(np.float32)
        om32 = om_arr.astype(np.float32)
        diff = np.abs(pt32 - om32)
        cos = _cosine(pt32, om32)
        l1 = float(diff.mean())
        linf = float(diff.max())

        print(
            f"{step:>4} {shape_str:>20} {cos:>12.6f} {l1:>12.4e} "
            f"{linf:>12.4e} {float(pt32.mean()):>+10.4f} "
            f"{float(om32.mean()):>+10.4f} {float(pt32.std()):>10.4f} "
            f"{float(om32.std()):>10.4f}"
        )

    print("-" * 116)

    # ------------------------------------------------------------------
    # Real-action-dim only comparison.
    # ------------------------------------------------------------------
    # PI05 pads each chunk from the *real* action width (e.g. 6) up to
    # ``max_action_dim`` (32) with zeros.  If both PT and OM happen to
    # leave the trailing 26 dims near zero, those dims add roughly equal
    # mass to ||a|| and ||b|| but contribute next to nothing to <a, b>,
    # so the full-tensor cosine is dominated by "both sides are zero"
    # agreement — it can read 0.99 even when the 6 real dims have
    # already drifted to ~0.5.  ``loss_compare`` slices to the real dim
    # via ``actions[:, :, :original_action_dim]`` and is therefore
    # immune to this inflation; the table below reproduces the same
    # slice so the two tools become directly comparable.
    if action_dim is not None and action_dim > 0:
        print(f"\n--- First {action_dim} action dim(s) only ---")
        print(
            f"{'step':>4} {'shape':>20} {'cosine':>12} {'L1':>12} "
            f"{'Linf':>12} {'PT mean':>10} {'OM mean':>10} {'PT std':>10} "
            f"{'OM std':>10}"
        )
        print("-" * 116)
        for step in common:
            pt_arr = np.load(pt_files[step])
            om_arr = np.load(om_files[step])
            if pt_arr.shape != om_arr.shape:
                continue  # already warned about above
            if pt_arr.shape[-1] < action_dim:
                print(f"{step:>4}  --action-dim={action_dim} > last dim {pt_arr.shape[-1]} — skipped")
                continue
            pt_real = pt_arr[..., :action_dim].astype(np.float32)
            om_real = om_arr[..., :action_dim].astype(np.float32)
            diff_r = np.abs(pt_real - om_real)
            cos_r = _cosine(pt_real, om_real)
            shape_r = str(tuple(pt_real.shape))
            print(
                f"{step:>4} {shape_r:>20} {cos_r:>12.6f} "
                f"{float(diff_r.mean()):>12.4e} "
                f"{float(diff_r.max()):>12.4e} "
                f"{float(pt_real.mean()):>+10.4f} "
                f"{float(om_real.mean()):>+10.4f} "
                f"{float(pt_real.std()):>10.4f} "
                f"{float(om_real.std()):>10.4f}"
            )
        print("-" * 116)

    # Diagnostic: where does cosine first drop below 0.999, and where below 0.9?
    thresholds = [0.999, 0.99, 0.9, 0.5, 0.0]
    cosines: dict[int, float] = {}
    for step in common:
        pt_arr = np.load(pt_files[step]).astype(np.float32)
        om_arr = np.load(om_files[step]).astype(np.float32)
        cosines[step] = _cosine(pt_arr, om_arr)

    print("\nFirst step where cosine drops below threshold:")
    for thr in thresholds:
        first = next((s for s in common if cosines[s] < thr), None)
        if first is None:
            print(f"  cos < {thr:>5.3f} : never")
        else:
            print(f"  cos < {thr:>5.3f} : step {first}  (cos={cosines[first]:.6f})")

    # ------------------------------------------------------------------
    # AE inputs (ae_in_*.npy) — independent comparison so we can tell
    # "AE is wrong" from "AE was fed something different".
    # ------------------------------------------------------------------
    pt_extras = {p.name for p in pt.glob("ae_in_*.npy")} if pt.is_dir() else set()
    om_extras = {p.name for p in om.glob("ae_in_*.npy")} if om.is_dir() else set()
    extras = sorted(pt_extras & om_extras)
    if extras:
        print("\n=== AE INPUTS ===")
        print(f"{'file':<35} {'shape':>20} {'cosine':>12} {'L1':>12} {'Linf':>12}")
        print("-" * 95)
        for fname in extras:
            try:
                pa = np.load(pt / fname)
                oa = np.load(om / fname)
            except Exception as exc:
                print(f"{fname:<35} LOAD FAILED: {exc}")
                continue
            if pa.shape != oa.shape:
                if pa.size == oa.size:
                    pa = pa.reshape(-1)
                    oa = oa.reshape(-1)
                    shape_str = f"flat({pa.size})"
                else:
                    print(f"{fname:<35} SHAPE MISMATCH pt={pa.shape} om={oa.shape}")
                    continue
            else:
                shape_str = str(tuple(pa.shape))
            if pa.dtype == np.bool_ and oa.dtype == np.bool_:
                mismatch = int((pa != oa).sum())
                print(f"{fname:<35} {shape_str:>20} bool_mismatch={mismatch}/{pa.size}")
                continue
            p32 = pa.astype(np.float32)
            o32 = oa.astype(np.float32)
            d = np.abs(p32 - o32)
            print(
                f"{fname:<35} {shape_str:>20} {_cosine(p32, o32):>12.6f} "
                f"{float(d.mean()):>12.4e} {float(d.max()):>12.4e}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--policy-path", type=str, default=None)
    p.add_argument("--past-kv", type=str, default=None, help="Path to past_kv_tensor.npy (PT or OM dump).")
    p.add_argument("--pad-masks", type=str, default=None, help="Path to prefix_pad_masks.npy.")
    p.add_argument(
        "--noise",
        type=str,
        default=None,
        help="Optional noise .npy (e.g. from loss_compare --noise-dir). "
        "If omitted, deterministic noise (seed=42) is generated.",
    )
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of Euler steps to dump. Defaults to policy.config.num_inference_steps (10 for PI05).",
    )
    p.add_argument(
        "--model-dtype",
        type=str,
        default="native",
        choices=["native", "fp16", "bf16", "fp32"],
        help="Cast policy.model to this dtype before forward. "
        "Use 'fp16' to match loss_compare --model_dtype fp16 "
        "(the dtype where the -0.6 cosine was observed).",
    )

    p.add_argument("--compare-pt", type=str, default=None)
    p.add_argument("--compare-om", type=str, default=None)
    p.add_argument(
        "--action-dim",
        type=int,
        default=None,
        help="If given, also print a comparison table restricted to "
        "[..., :action_dim] of every dump.  Use this to reproduce "
        "loss_compare's apples-to-apples slice (e.g. --action-dim 6) "
        "and detect whether the high full-tensor cosine is being "
        "inflated by PI05's zero-padded trailing action dims.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    if args.compare_pt and args.compare_om:
        compare(
            pt_dir=args.compare_pt,
            om_dir=args.compare_om,
            action_dim=args.action_dim,
        )
        return 0

    missing = [n for n in ("policy_path", "past_kv", "pad_masks", "out_dir") if getattr(args, n) is None]
    if missing:
        print(f"error: missing required args for dump mode: {missing}", file=sys.stderr)
        return 2

    dump(
        policy_path=args.policy_path,
        past_kv_path=args.past_kv,
        pad_masks_path=args.pad_masks,
        noise_path=args.noise,
        out_dir=args.out_dir,
        device_str=args.device,
        num_steps=args.num_steps,
        model_dtype=args.model_dtype,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
