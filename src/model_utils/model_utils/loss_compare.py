from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Heavy imports (numpy, torch, tqdm, inference_service) are deferred to main()
# after CLI argument resolution so that --list-profiles / --help / wizard-only
# invocations remain fast (< 100ms).  LossUtils references these as module
# globals; they are assigned by _import_heavy_deps() before LossUtils is
# instantiated.
np = None  # type: ignore
torch = None  # type: ignore
tqdm = None  # type: ignore
PureInferenceEngine = None  # type: ignore
_DIAGNOSTIC_ASCEND_OPTIONS = None


def _diagnostic_ascend_backend(context):
    from inference_service.backends.ascend.backend import AscendBackend

    if _DIAGNOSTIC_ASCEND_OPTIONS is None:
        raise RuntimeError("diagnostic Ascend schedule was not configured")
    schedule, source = _DIAGNOSTIC_ASCEND_OPTIONS
    return AscendBackend(
        int(context.runtime_options.get("device_id", 0)),
        diagnostic_schedule=schedule,
        diagnostic_schedule_source=source,
    )


def generate_pi05_noise(shape: tuple[int, ...], seed: int):
    """Generate deterministic CPU fp32 noise without changing global Torch RNG state."""
    global torch
    if torch is None:
        import torch as torch_module

        torch = torch_module
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


class LossUtils:
    def __init__(self, args):
        self.args = args
        self.engine = self.prepare_policy()
        self.args.policy_type = self.engine.policy_type.lower()

    def run(self):
        if self.args.generate_target:
            return self.generate_target()
        return self.compute_loss()

    def compute_loss(self):
        print("computing loss...")
        with open(self.args.target_path, encoding="utf-8") as f:
            targets = json.load(f)

        for i in range(len(targets)):
            targets[i] = torch.tensor(targets[i])

        batches = self.load_batches_as_tensors()
        start_time = time.perf_counter()
        preds = self.forward(batches)
        inference_elapsed = time.perf_counter() - start_time
        print(f"inference time: {inference_elapsed:.3f}s")

        if len(targets) != len(preds):
            raise ValueError(f"Length mismatch: targets {len(targets)} vs preds {len(preds)}")

        arr_l1 = []
        arr_cos = []
        for i in range(len(targets)):
            pred = preds[i].float()
            target = targets[i].float()
            l1 = torch.nn.functional.l1_loss(pred, target, reduction="mean").item()
            cos = torch.nn.functional.cosine_similarity(
                pred.flatten().unsqueeze(0),
                target.flatten().unsqueeze(0),
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
        reported_latencies = getattr(self, "_inference_latencies_ms", [])
        average_latency_ms = (
            sum(reported_latencies) / len(reported_latencies)
            if reported_latencies
            else inference_elapsed * 1000.0 / len(preds)
        )
        metrics = {
            "format": "loss-compare-metrics-v1",
            "sample_count": len(preds),
            "inference": {
                "elapsed_seconds": inference_elapsed,
                "average_latency_ms": average_latency_ms,
                "latency_source": "pipeline" if reported_latencies else "wall_clock",
            },
            "unnormalized": {"l1": avg_l1, "cosine": avg_cos},
            "normalized": {"l1": None, "cosine": None},
            "pi05_distribution": {},
        }

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
                    raw_avg_l1 = sum(arr_raw_l1) / len(arr_raw_l1)
                    raw_avg_cos = sum(arr_raw_cos) / len(arr_raw_cos)
                    metrics["normalized"] = {"l1": raw_avg_l1, "cosine": raw_avg_cos}
                    print("\n=== Normalized action space (pre-postprocessor) ===")
                    print(f"{'Batch':>6} {'raw L1':>12} {'raw Cos':>12}")
                    print("-" * 32)
                    for i in range(len(arr_raw_l1)):
                        print(f"{i:>6} {arr_raw_l1[i]:>12.6f} {arr_raw_cos[i]:>12.6f}")
                    print("-" * 32)
                    print(f"{'Avg':>6} {raw_avg_l1:>12.6f} {raw_avg_cos:>12.6f}")
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
            metrics["pi05_distribution"] = evaluate_pi05(
                preds=preds,
                targets=targets,
                raw_preds=raw_preds_list,
                raw_targets=raw_targets_list,
            )
        metrics["aggregates"] = self._aggregate_metrics(metrics)
        self._write_metrics_json(metrics)
        return metrics

    @staticmethod
    def _aggregate_metrics(metrics: dict[str, object]) -> dict[str, float | None]:
        inference = metrics["inference"]
        normalized = metrics["normalized"]
        unnormalized = metrics["unnormalized"]
        pi05 = metrics["pi05_distribution"]
        assert isinstance(inference, dict)
        assert isinstance(normalized, dict)
        assert isinstance(unnormalized, dict)
        assert isinstance(pi05, dict)
        normalized_distribution = pi05.get("normalized", {})
        wasserstein = (
            normalized_distribution.get("wasserstein", {}) if isinstance(normalized_distribution, dict) else {}
        )
        first_frame = (
            normalized_distribution.get("first_frame", {}) if isinstance(normalized_distribution, dict) else {}
        )
        return {
            "inference_time": inference["elapsed_seconds"],
            "average_latency_ms": inference["average_latency_ms"],
            "raw_l1": normalized["l1"],
            "raw_cos": normalized["cosine"],
            "unnorm_l1": unnormalized["l1"],
            "unnorm_cos": unnormalized["cosine"],
            "normalized_mean_w1_std": wasserstein.get("mean_ratio") if isinstance(wasserstein, dict) else None,
            "normalized_first_frame_cos": first_frame.get("mean_cos") if isinstance(first_frame, dict) else None,
        }

    def _write_metrics_json(self, metrics: dict[str, object]) -> None:
        destination_value = getattr(self.args, "metrics_json", None)
        if destination_value is None:
            return
        destination = Path(destination_value).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"metrics JSON saved at {destination}")

    def prepare_policy(self):
        runtime_options = {}
        if self.args.model_dtype != "native":
            runtime_options["model_dtype"] = self.args.model_dtype
        for option in ("curvature_log_path",):
            value = getattr(self.args, option, None)
            if value is not None:
                runtime_options[option] = value
        registry = None
        schedule_override_path = getattr(self.args, "schedule_override_path", None)
        if schedule_override_path is not None:
            from inference_service.backends import BACKEND_REGISTRY, BackendRegistry
            from inference_service.pi05_schedule import load_pi05_schedule

            schedule_path = Path(schedule_override_path).expanduser().resolve(strict=True)
            schedule = load_pi05_schedule(schedule_path)
            descriptors = {name: BACKEND_REGISTRY.descriptor(name) for name in BACKEND_REGISTRY.names}
            ascend = descriptors["ascend"]
            descriptors["ascend"] = ascend.__class__(
                name=ascend.name,
                factory="model_utils.loss_compare:_diagnostic_ascend_backend",
                supported_policy_families=ascend.supported_policy_families,
                target_validator=ascend.target_validator,
            )
            global _DIAGNOSTIC_ASCEND_OPTIONS
            _DIAGNOSTIC_ASCEND_OPTIONS = (schedule, str(schedule_path))
            registry = BackendRegistry(descriptors)
        engine = PureInferenceEngine(
            model_path=self.args.policy_path,
            deployment=self.args.deployment,
            pipeline_id="loss_compare",
            runtime_options=runtime_options,
            **({"registry": registry} if registry is not None else {}),
        )
        print(
            f"model loaded: {self.args.policy_path} "
            f"(policy_type={engine.policy_type}, deployment={self.args.deployment}, "
            f"backend={engine.backend_type})"
        )
        return engine

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
                    array = np.asarray(v, dtype=np.float32)
                    processed_batch[k] = array[None, ...] if array.ndim == 1 else array
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
        """Normalize a JSON image to batched float32 NCHW in [0, 1].

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
        if arr.ndim == 3 and arr.shape[-1] in {1, 3, 4}:
            arr = np.transpose(arr, (2, 0, 1))
        elif arr.ndim == 4 and arr.shape[-1] in {1, 3, 4}:
            arr = np.transpose(arr, (0, 3, 1, 2))
        if arr.ndim == 3:
            arr = arr[None, ...]
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

        noise_shape = self._pi05_noise_shape()
        noise = generate_pi05_noise(noise_shape, self.args.seed + batch_idx)
        if not self.args.noise_dir:
            if self.args.generate_target:
                raise RuntimeError("PI0.5 generate-target requires --noise-dir or --exp-dir to persist external noise")
            return noise

        noise_path = os.path.join(self.args.noise_dir, f"noise_{batch_idx:04d}.npy")
        if self.args.generate_target:
            os.makedirs(self.args.noise_dir, exist_ok=True)
            np.save(noise_path, noise.numpy())
            return noise
        loaded = torch.from_numpy(np.load(noise_path)).float()
        if tuple(loaded.shape) != noise_shape:
            raise ValueError(f"Noise shape mismatch in {noise_path}: got {tuple(loaded.shape)}, expected {noise_shape}")
        return loaded

    def _pi05_noise_shape(self):
        """Return the policy-declared PI0.5 padded noise shape."""
        chunk = self.engine.nominal_chunk_size
        max_action_dim = self.engine.max_action_dimension
        if chunk is None or max_action_dim is None:
            raise RuntimeError("PI0.5 fixed-noise comparison requires chunk_size and max_action_dim in config.json")
        return (1, int(chunk), int(max_action_dim))

    def _reset_independent_sample(self):
        capabilities = self.engine.capabilities
        if not capabilities.stateful:
            return
        if not capabilities.resettable:
            raise RuntimeError("selected stateful deployment cannot reset between independent comparison samples")
        self.engine.reset()

    def _infer_raw(self, batch, noise):
        """Run one independent sample and return its pre-postprocessor action."""
        self._reset_independent_sample()
        result = self.engine(
            dict(batch),
            control_inputs={"noise": noise},
            capture_raw_action=True,
        )
        if result.raw_action is None:
            raise RuntimeError("unified inference pipeline did not return the requested raw action")
        return torch.as_tensor(result.raw_action).detach().cpu().float()

    def _assert_noise_effective(self, batch0):
        """Prove the unified control-input noise binding is deterministic and effective.

        Noise is passed through ``PureInferenceEngine.control_inputs`` and bound
        by the selected deployment's PI0.5 codec. This check turns a missing or
        ignored manifest noise binding into a hard failure before a bad target
        is written.

        Method (3 forwards on batch 0):
          A1, A2 with noise=zeros, B with noise=3*ones.
          - determinism: A1 == A2 (else RNG leak / dropout active / eval off)
          - effectiveness: A1 != B (else noise is being ignored)
        """
        if self.args.policy_type != "pi05" or not self.args.noise_dir:
            return  # no fixed-noise requirement -> nothing to guarantee

        shape = self._pi05_noise_shape()

        nA = torch.zeros(shape, dtype=torch.float32)
        nB = torch.full(shape, 3.0, dtype=torch.float32)

        oA1 = self._infer_raw(batch0, nA)
        oB = self._infer_raw(batch0, nB)
        oA2 = self._infer_raw(batch0, nA)

        deterministic = torch.allclose(oA1, oA2, atol=1e-5)
        effective = not torch.allclose(oA1, oB, atol=1e-4)
        d_AB = (oA1 - oB).abs().mean().item()
        d_AA = (oA1 - oA2).abs().mean().item()

        if not deterministic:
            raise RuntimeError(
                "Noise self-check FAILED (non-deterministic): replaying the same "
                f"noise gave different outputs (mean|A1-A2|={d_AA:.3e}). Likely an "
                "RNG leak or active dropout — ensure policy.eval() ran. Refusing "
                "to generate non-reproducible targets."
            )
        if not effective:
            raise RuntimeError(
                "Noise self-check FAILED (no effect): two very different noises "
                f"produced ~identical outputs (mean|A-B|={d_AB:.3e}). The injected "
                "noise is being ignored by the selected deployment's unified "
                "control-input or manifest binding path. Aborting before writing "
                "a non-reproducible target."
            )
        print(f"  ✓ noise self-check passed (deterministic mean|A1-A2|={d_AA:.2e}, effective mean|A-B|={d_AB:.2e})")

    def forward(self, batches):
        raw_preds: list[torch.Tensor] = []
        outputs = []
        inference_latencies_ms = []

        for i in tqdm(range(len(batches)), desc="forwarding"):
            self._reset_independent_sample()
            noise = self._resolve_noise(i)
            result = self.engine(
                dict(batches[i]),
                request_id=f"loss-compare-{i}",
                control_inputs={"noise": noise} if noise is not None else None,
                capture_raw_action=True,
            )
            if result.raw_action is None:
                raise RuntimeError("unified inference pipeline did not return the requested raw action")
            raw_preds.append(torch.as_tensor(result.raw_action).detach().cpu().clone())
            latency_ms = getattr(result, "latency_ms", None)
            if isinstance(latency_ms, int | float) and latency_ms >= 0:
                inference_latencies_ms.append(float(latency_ms))

            output = torch.as_tensor(result.action)
            # Normalize storage shape: drop a leading singleton batch dim so
            # ACT returns (T, D) / (D,) and PI05 returns (T, D), matching the
            # original loss_compare conventions.
            output = output.detach().cpu()
            if output.ndim >= 3 and output.shape[0] == 1:
                output = output.squeeze(0)
            outputs.append(output)

        # Stash raw preds for compute_loss / generate_target to use.
        self._raw_preds = raw_preds
        self._inference_latencies_ms = inference_latencies_ms
        return outputs

    def generate_target(self):
        print("generating target json from batches...")
        if self.args.noise_dir:
            print(f"  noise files will be saved to: {self.args.noise_dir}")

        batches = self.load_batches_as_tensors()

        # Guard: before producing any target, prove the fixed noise is actually
        # consumed by this backend (PI05 + --noise-dir).  A silent noise-drop
        # here would yield targets the selected compiled deployment can never match.
        if batches:
            self._assert_noise_effective(batches[0])

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
    global np, torch, tqdm, PureInferenceEngine
    import numpy as _np
    import torch as _torch
    from tqdm import tqdm as _tqdm

    from inference_service.core import PureInferenceEngine as _Engine

    np = _np
    torch = _torch
    tqdm = _tqdm
    PureInferenceEngine = _Engine


def main():
    # All argument ergonomics (profile / wizard / --exp-dir derivation /
    # remember-last) live in loss_compare_cli so this entry point stays thin
    # and LossUtils itself stays focused on inference and metrics.
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

    # Lazy-load torch/numpy/tqdm/inference_service only after resolve() returns;
    # --list-profiles / --help / wizard-save exits in resolve() before this.
    _import_heavy_deps()

    loss_utils = LossUtils(resolved.args)
    try:
        loss_utils.run()
        # Persist this run's effective params as ``_last`` only after success.
        loss_compare_cli.write_last(resolved)
    finally:
        loss_utils.engine.close()


if __name__ == "__main__":
    main()
