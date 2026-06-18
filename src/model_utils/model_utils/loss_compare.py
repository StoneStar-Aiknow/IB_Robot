from __future__ import annotations

import contextlib
import json
import os
import time

# Heavy imports (numpy, torch, tqdm, inference_service) are deferred to main()
# after CLI argument resolution so that --list-profiles / --help / wizard-only
# invocations remain fast (< 100ms).  LossUtils references these as module
# globals; they are assigned by _import_heavy_deps() before LossUtils is
# instantiated.
np = None  # type: ignore
torch = None  # type: ignore
tqdm = None  # type: ignore
InferenceCoordinator = None  # type: ignore


class LossUtils:
    def __init__(self, args):
        self.args = args
        self.coordinator = self.prepare_policy()
        # ``policy_type`` is reported by the engine after loading.  The CLI
        # ``--policy_type`` is kept only as a hint / fallback for backends that
        # cannot self-report (it must still match what the coordinator detects).
        detected = (self.coordinator.policy_type or "").lower()
        if detected:
            self.args.policy_type = detected

    def run(self):
        if self.args.generate_target:
            self.generate_target()
        else:
            self.compute_loss()

    def compute_loss(self):
        print("computing loss...")
        with open(self.args.target_path, encoding="utf-8") as f:
            targets = json.load(f)

        for i in range(len(targets)):
            targets[i] = torch.tensor(targets[i])

        batches = self.load_batches_as_tensors()
        start_time = time.perf_counter()
        preds = self.forward(batches)
        end_time = time.perf_counter()
        print(f"inference time: {end_time - start_time:.3f}s")

        if len(targets) != len(preds):
            raise ValueError(f"Length mismatch: targets {len(targets)} vs preds {len(preds)}")

        arr_l1 = []
        arr_cos = []
        for i in range(len(targets)):
            l1 = torch.nn.functional.l1_loss(preds[i], targets[i], reduction="mean").item()
            cos = torch.nn.functional.cosine_similarity(
                preds[i].flatten().unsqueeze(0),
                targets[i].flatten().unsqueeze(0),
            ).item()
            arr_l1.append(l1)
            arr_cos.append(cos)

        # Print summary table (unnormalized / physical action space)
        print("\n=== Unnormalized action space (post-postprocessor) ===")
        print(f"{'Batch':>6} {'L1 Loss':>12} {'Cosine Sim':>12}")
        print("-" * 32)
        for i in range(len(arr_l1)):
            print(f"{i:>6} {arr_l1[i]:>12.6f} {arr_cos[i]:>12.6f}")
        print("-" * 32)
        avg_l1 = sum(arr_l1) / len(arr_l1)
        avg_cos = sum(arr_cos) / len(arr_cos)
        print(f"{'Avg':>6} {avg_l1:>12.6f} {avg_cos:>12.6f}")

        # ------------------------------------------------------------------
        # Independent sanity check: compare in *normalized* (pre-postprocessor)
        # action space.  ``self._raw_preds`` was populated by ``forward()`` by
        # capturing the engine output *before* the postprocessor unnormalizes
        # it.  This isolates the model's true output error from any
        # unnormalization scale-up — useful for diagnosing whether a large
        # unnormalized L1 is real model drift or just dataset stats blowing the
        # number up.
        # ------------------------------------------------------------------
        raw_preds = getattr(self, "_raw_preds", None)
        raw_targets_path = getattr(self.args, "raw_target_path", None)

        # Read raw targets once and reuse in both the normalized-space comparison
        # and the PI05 distributional evaluation below.
        raw_targets_data = None
        if raw_targets_path and os.path.exists(raw_targets_path):
            with open(raw_targets_path, encoding="utf-8") as f:
                raw_targets_data = [torch.tensor(t) for t in json.load(f)]

        if raw_preds is not None and raw_targets_data is not None:
            raw_targets = raw_targets_data

            if len(raw_targets) != len(raw_preds):
                print(f"WARN: raw target/pred length mismatch: {len(raw_targets)} vs {len(raw_preds)}; skipping raw L1")
            else:
                arr_raw_l1 = []
                arr_raw_cos = []
                for i in range(len(raw_targets)):
                    rp = self._squeeze_leading(raw_preds[i].detach().cpu().float())
                    rt = self._squeeze_leading(raw_targets[i].float())
                    if rp.shape != rt.shape:
                        print(f"WARN: raw shape mismatch on batch {i}: pred={tuple(rp.shape)} target={tuple(rt.shape)}")
                        continue
                    arr_raw_l1.append(torch.nn.functional.l1_loss(rp, rt, reduction="mean").item())
                    arr_raw_cos.append(
                        torch.nn.functional.cosine_similarity(
                            rp.flatten().unsqueeze(0), rt.flatten().unsqueeze(0)
                        ).item()
                    )

                if arr_raw_l1:
                    print("\n=== Normalized action space (pre-postprocessor) ===")
                    print(f"{'Batch':>6} {'raw L1':>12} {'raw Cos':>12}")
                    print("-" * 32)
                    for i in range(len(arr_raw_l1)):
                        print(f"{i:>6} {arr_raw_l1[i]:>12.6f} {arr_raw_cos[i]:>12.6f}")
                    print("-" * 32)
                    print(
                        f"{'Avg':>6} {sum(arr_raw_l1) / len(arr_raw_l1):>12.6f} "
                        f"{sum(arr_raw_cos) / len(arr_raw_cos):>12.6f}"
                    )
        elif raw_preds is not None and raw_targets_path and not os.path.exists(raw_targets_path):
            print(
                f"\nNOTE: raw target file not found at {raw_targets_path}; "
                f"run --generate-target with --raw-target-path to create it."
            )

        # Diagnostic dump for batch 0 — physical-space numbers so you can judge
        # whether the unnormalized L1 is "small" (e.g. mm in cartesian) or
        # "large" (e.g. radians in joint space).
        if len(preds) > 0 and len(targets) > 0:
            p, t = preds[0], targets[0]
            print("\n=== batch 0 diagnostic (unnormalized) ===")
            print(f"  pred   shape : {tuple(p.shape)}  dtype={p.dtype}")
            print(f"  target shape : {tuple(t.shape)}  dtype={t.dtype}")
            print(
                f"  pred   range : [{p.min().item():+.4f}, {p.max().item():+.4f}]  "
                f"mean={p.mean().item():+.4f}  std={p.std().item():.4f}"
            )
            print(
                f"  target range : [{t.min().item():+.4f}, {t.max().item():+.4f}]  "
                f"mean={t.mean().item():+.4f}  std={t.std().item():.4f}"
            )
            diff = (p - t).abs()
            if diff.ndim >= 2:
                # Per-action-dim stats (last axis is action_dim)
                reduce_dims = tuple(range(diff.ndim - 1))
                per_dim_l1 = diff.mean(dim=reduce_dims)
                per_dim_max = diff.amax(dim=reduce_dims)
                _old_np_opts = np.get_printoptions()
                np.set_printoptions(precision=4, suppress=True, linewidth=160)
                print(f"  per-dim L1   : {per_dim_l1.cpu().numpy()}")
                print(f"  per-dim Linf : {per_dim_max.cpu().numpy()}")
                np.set_printoptions(**_old_np_opts)
            print(f"  pred   first row: {p.flatten()[: p.shape[-1]].cpu().numpy()}")
            print(f"  target first row: {t.flatten()[: t.shape[-1]].cpu().numpy()}")

        # ------------------------------------------------------------------
        # PI05 only: replace the meaningless per-sample cosine summary with
        # chaos-robust distributional metrics (see pi05_dist_metrics.py).
        # ------------------------------------------------------------------
        if self.args.policy_type == "pi05":
            from model_utils.pi05_dist_metrics import evaluate_pi05

            raw_preds_list = getattr(self, "_raw_preds", None)
            raw_targets_list = raw_targets_data
            evaluate_pi05(
                preds=preds,
                targets=targets,
                raw_preds=raw_preds_list,
                raw_targets=raw_targets_list,
            )

    def prepare_policy(self):
        # The InferenceCoordinator picks the backend from ``--device``:
        #   cuda/cpu/npu  -> native LeRobot torch policy (LeRobotPolicyWrapper)
        #   ascend_om     -> compiled OM offline model (CompiledPolicyWrapper)
        #   ascend_om_3403/rknn -> their respective compiled wrappers
        # All of them expose the same pre/infer/post pipeline, so loss_compare
        # no longer depends on the raw LeRobot policy object.
        coordinator = InferenceCoordinator(
            policy_path=self.args.policy_path,
            device=self.args.device,
        )

        # Optional dtype cast — only meaningful for the torch backend (the
        # compiled OM/RKNN models carry their own fixed dtype).  Use
        # ``--model_dtype fp16`` to match the OM/ORT deployment dtype and
        # isolate BF16<->FP16 conversion error from any real export error.
        model_dtype = getattr(self.args, "model_dtype", "native")
        raw_policy = coordinator.raw_policy
        if raw_policy is not None and hasattr(raw_policy, "model"):
            if model_dtype == "fp16":
                raw_policy.model = raw_policy.model.half()
                print("  Cast policy.model to float16")
            elif model_dtype == "bf16":
                raw_policy.model = raw_policy.model.bfloat16()
                print("  Cast policy.model to bfloat16")
            elif model_dtype == "fp32":
                raw_policy.model = raw_policy.model.float()
                print("  Cast policy.model to float32")
            elif model_dtype != "native":
                raise ValueError(f"unknown --model_dtype: {model_dtype}")

            # CRITICAL: switch to eval mode.  Without this, LoRA's default
            # dropout=0.1 stays active and randomises every forward pass
            # (independent of seed) — the resulting KV jitter feeds into
            # PI05's flow-matching ODE which is chaotic, so the 10-step
            # trajectory diverges to ~zero correlation with the OM's
            # deterministic output (observed: raw cos ~= -0.04).
            with contextlib.suppress(Exception):
                raw_policy.eval()

            try:
                sample_param = next(raw_policy.model.parameters())
                print(f"  Running PT policy in dtype={sample_param.dtype}")
            except (StopIteration, AttributeError):
                pass
        elif model_dtype != "native":
            print(
                f"  NOTE: --model_dtype={model_dtype} ignored for backend "
                f"'{coordinator.backend_type or self.args.device}' "
                f"(compiled models use their own fixed dtype)."
            )

        print(
            f"model loaded: {self.args.policy_path} "
            f"(policy_type={coordinator.policy_type}, "
            f"backend={coordinator.backend_type or 'torch'})"
        )
        return coordinator

    def load_batches_as_tensors(self):
        with open(self.args.batch_path, encoding="utf-8") as f:
            raw_batches = json.load(f)
        processed_batches = []
        for b in raw_batches:
            processed_batch = {}
            for k, v in b.items():
                if "side_view" in k:
                    continue
                elif k == "observation.images.hand_view":
                    processed_batch["observation.images.wrist"] = self._prepare_image(v)
                elif k == "observation.images.top_view":
                    processed_batch["observation.images.top"] = self._prepare_image(v)
                elif "image" in k:
                    processed_batch[k] = self._prepare_image(v)
                elif isinstance(v, str):
                    # Keep natural-language task prompts intact for VLA policies
                    # (PI0/PI05/SmolVLA): the preprocessor tokenizes them.
                    processed_batch[k] = v
                else:
                    processed_batch[k] = np.array(v).astype(np.float32)
            # VLA policies (PI0/PI05/SmolVLA) require a natural-language task
            # prompt in the observation frame; the LeRobot preprocessor routes
            # ``task`` into complementary_data and tokenizes it.  Default to an
            # empty string to match the historical loss_compare behavior
            # (``prepare_observation_for_inference`` set task="" when none was
            # given); override with ``--task``.
            if "task" not in processed_batch:
                processed_batch["task"] = self.args.task
            processed_batches.append(processed_batch)
        return processed_batches

    @staticmethod
    def _prepare_image(value) -> np.ndarray:
        """Normalize a JSON image to float32 [0, 1], HWC kept.

        The batch JSON stores images as HWC with [0, 255] pixel values (but
        encoded as float, not uint8).  The historical loss_compare path used
        ``prepare_observation_for_inference`` which *unconditionally* divided
        images by 255.  The current ``TensorPreprocessor`` only auto-divides
        when the numpy dtype is an *integer*, so a float [0, 255] image would
        slip through un-normalized and blow up the model input (observed: L1
        jumping from ~1 to ~16).  Normalize here to match the historical
        behavior, but stay idempotent if the data already happens to be [0, 1].
        """
        arr = np.array(value, dtype=np.float32)
        if arr.size and float(arr.max()) > 1.0:
            arr = arr / 255.0
        return arr

    @staticmethod
    def _squeeze_leading(t: torch.Tensor) -> torch.Tensor:
        """Drop a leading singleton batch dim so a ``(1, T, D)`` tensor and a
        ``(T, D)`` tensor compare equal.

        Targets generated by the historical pipeline stored raw (normalized)
        actions as ``(1, T, D)`` (it hooked the postprocessor input, i.e. the
        full chunk), whereas the current pipeline stores ``(T, D)``.  Squeezing
        here lets a freshly-run compute-loss compare against an older raw
        baseline without spurious shape-mismatch warnings.
        """
        if t.ndim >= 3 and t.shape[0] == 1:
            return t.squeeze(0)
        return t

    def _resolve_noise(self, batch_idx: int):
        """Generate/load deterministic noise for PI05 flow-matching.

        Returns a CPU fp32 tensor of shape (1, chunk_size, max_action_dim) or
        ``None`` when noise control is not requested/available.
        """
        if self.args.policy_type != "pi05":
            return None

        # Per-batch seed keeps diffusion/flow-matching noise deterministic
        # across runs even without --noise-dir.
        torch.manual_seed(self.args.seed + batch_idx)

        if not self.args.noise_dir:
            return None

        noise_path = os.path.join(self.args.noise_dir, f"noise_{batch_idx:04d}.npy")
        if self.args.generate_target:
            noise_shape = self._pi05_noise_shape()
            if noise_shape is None:
                return None
            noise = torch.normal(mean=0.0, std=1.0, size=noise_shape, dtype=torch.float32)
            os.makedirs(self.args.noise_dir, exist_ok=True)
            np.save(noise_path, noise.numpy())
            return noise
        return torch.from_numpy(np.load(noise_path)).float()

    def _pi05_noise_shape(self):
        """(1, chunk_size, max_action_dim) inferred from whatever config the
        backend exposes (torch policy config or compiled config view)."""
        raw_policy = self.coordinator.raw_policy
        cfg = getattr(raw_policy, "config", None)
        if cfg is not None and hasattr(cfg, "chunk_size") and hasattr(cfg, "max_action_dim"):
            return (1, int(cfg.chunk_size), int(cfg.max_action_dim))
        # Compiled backend: fall back to the engine-reported chunk size and the
        # PI05 default max_action_dim (32).  ``--noise-dir`` cross-machine flows
        # generate noise on the torch side anyway, so this branch is mostly a
        # safety net for OM-side regeneration.
        chunk = self.coordinator.chunk_size or 0
        if chunk <= 0:
            print("  WARN: cannot infer PI05 noise shape on this backend; skipping noise injection")
            return None
        return (1, int(chunk), 32)

    def _inject_noise(self, batch, noise):
        """Wire deterministic noise into whichever backend is active.

        - torch LeRobot policy: ``policy._external_noise`` is consumed by
          ``sample_noise()`` during ``predict_action_chunk``.
        - compiled OM PI05: ``batch["_noise"]`` is read by
          ``PI05CompiledAdapter.prepare_inputs``.
        """
        if noise is None:
            return batch

        raw_policy = self.coordinator.raw_policy
        if raw_policy is not None:
            # Match the action_expert weight dtype, otherwise action_in_proj
            # fails with "mat1 and mat2 must have the same dtype".
            try:
                model_dtype = next(raw_policy.model.parameters()).dtype
            except (StopIteration, AttributeError):
                model_dtype = torch.float32
            raw_policy._external_noise = noise.to(device=self.coordinator.device, dtype=model_dtype)
        else:
            # Compiled backend reads noise straight from the batch dict.
            batch = dict(batch)
            batch["_noise"] = noise
        return batch

    def forward(self, batches):
        raw_preds: list[torch.Tensor] = []
        outputs = []

        for i in tqdm(range(len(batches)), desc="forwarding"):
            # IMPORTANT: loss_compare treats each JSON batch as an independent
            # sample, but the torch PI05Policy.select_action() keeps an internal
            # ``_action_queue`` across calls. Reset it so batch i never consumes
            # leftover actions from batch i-1.  (No-op for compiled backends,
            # which are stateless.)
            raw_policy = self.coordinator.raw_policy
            if raw_policy is not None and hasattr(raw_policy, "_action_queue"):
                raw_policy._action_queue.clear()

            noise = self._resolve_noise(i)

            # Run the pipeline stage-by-stage so we can capture the raw
            # (pre-postprocessor / normalized-space) action for every backend
            # uniformly — the OM/RKNN wrappers have no Python postprocessor hook
            # we could intercept, so we split pre/infer/post explicitly here.
            batch = self.coordinator.preprocess_only(dict(batches[i]))
            batch = self._inject_noise(batch, noise)

            with torch.inference_mode():
                infer_result = self.coordinator.infer_only(batch)

            raw_action = infer_result.action
            with contextlib.suppress(Exception):
                raw_preds.append(raw_action.detach().cpu().clone())

            output = self.coordinator.postprocess_only(raw_action)
            # Normalize storage shape: drop a leading singleton batch dim so
            # ACT returns (T, D) / (D,) and PI05 returns (T, D), matching the
            # original loss_compare conventions.
            output = output.detach().cpu()
            if output.ndim >= 3 and output.shape[0] == 1:
                output = output.squeeze(0)
            outputs.append(output)

        # Stash raw preds for compute_loss / generate_target to use.
        self._raw_preds = raw_preds
        return outputs

    def generate_target(self):
        print("generating target json from batches...")
        if self.args.noise_dir:
            print(f"  noise files will be saved to: {self.args.noise_dir}")

        batches = self.load_batches_as_tensors()
        outputs = self.forward(batches)

        print(f"saving output json: length={len(outputs)}")

        for i in range(len(outputs)):
            outputs[i] = outputs[i].tolist()

        output_path = self.args.target_path
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=4)

        print(f"output saved at {output_path}")

        # Also dump raw (pre-postprocessor / normalized-space) targets so the
        # NPU side can perform an apples-to-apples comparison without the
        # unnormalization scale-up.
        raw_target_path = getattr(self.args, "raw_target_path", None)
        raw_preds = getattr(self, "_raw_preds", None)
        if raw_target_path and raw_preds:
            raw_dump = [t.tolist() for t in raw_preds]
            with open(raw_target_path, "w", encoding="utf-8") as f:
                json.dump(raw_dump, f, indent=4)
            print(f"raw (normalized) target saved at {raw_target_path}")


def _import_heavy_deps():
    """Lazy-load numpy, torch, tqdm, inference_service.

    These libraries take seconds to import (torch especially). Defer until
    after CLI argument resolution so --list-profiles / --help / wizard-only
    invocations stay fast.  Assign to module globals so LossUtils (defined at
    module level) can reference them as before.
    """
    global np, torch, tqdm, InferenceCoordinator
    import numpy as _np
    import torch as _torch
    from tqdm import tqdm as _tqdm

    from inference_service.core import InferenceCoordinator as _Coord

    np = _np
    torch = _torch
    tqdm = _tqdm
    InferenceCoordinator = _Coord


def main():
    # All argument ergonomics (profile / wizard / --exp-dir derivation /
    # remember-last) live in loss_compare_cli so this entry point stays thin
    # and LossUtils itself is unchanged.  Every historical explicit flag still
    # works and overrides whatever a profile/derivation would supply.
    try:
        from model_utils import loss_compare_cli
    except ImportError:
        # Running as a standalone script (e.g. ``python loss_compare.py`` from a
        # data directory) — import the sibling file next to this one.
        import importlib.util
        import os
        import sys

        _cli_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loss_compare_cli.py")
        _spec = importlib.util.spec_from_file_location("loss_compare_cli", _cli_path)
        loss_compare_cli = importlib.util.module_from_spec(_spec)
        # Register before exec so dataclasses (which look the module up in
        # sys.modules during class creation) resolve correctly.
        sys.modules["loss_compare_cli"] = loss_compare_cli
        _spec.loader.exec_module(loss_compare_cli)

    resolved = loss_compare_cli.resolve()
    loss_compare_cli.print_effective(resolved)

    # Lazy-load torch/numpy/tqdm/coordinator only after resolve() returns;
    # --list-profiles / --help / wizard-save exits in resolve() before this.
    _import_heavy_deps()

    loss_utils = LossUtils(resolved.args)
    loss_utils.run()

    # Persist this run's effective params as ``_last`` (replaces a separate
    # "remember last args" cache).  Only after a successful run.
    loss_compare_cli.write_last(resolved)


if __name__ == "__main__":
    main()
