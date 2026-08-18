"""Explicit manifest-driven diagnostic dump for a PI0.5 Ascend deployment."""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from model_utils.loss_compare import generate_pi05_noise
from model_utils.observation_batch import load_observation_batch

LOGGER = logging.getLogger("pi05_om_dump")


class DiagnosticCapture:
    """Write explicitly requested diagnostic values as NumPy files."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.values: dict[str, dict[str, Any]] = {}
        self._name_counts: dict[str, int] = {}

    def save(self, name: str, value: object) -> None:
        count = self._name_counts.get(name, 0)
        self._name_counts[name] = count + 1
        if name == "action_expert_in_noise":
            name = "ae_in_noise" if count == 0 else f"x_t_step{count - 1:02d}"
        elif name == "action_expert_in_time":
            name = f"ae_in_time_step{count:02d}"
        elif name == "action_expert_out_action":
            name = f"velocity_step{count:02d}"
        candidate = value
        detach = getattr(candidate, "detach", None)
        if callable(detach):
            candidate = detach()
        cpu = getattr(candidate, "cpu", None)
        if callable(cpu):
            candidate = cpu()
        array = np.asarray(candidate)
        path = self.output_dir / f"{name}.npy"
        np.save(path, array)
        self.values[name] = {
            "file": path.name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

    __call__ = save

    def reset(self, output_dir: str | Path) -> None:
        """Start a new sample directory while retaining the loaded backend."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.values = {}
        self._name_counts = {}

    def write_index(self, metadata: dict[str, Any]) -> None:
        document = {**metadata, "values": self.values}
        (self.output_dir / "diagnostic_capture.json").write_text(
            json.dumps(document, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _prepare_batch(batches, batch_index: int, task: str) -> dict[str, object]:
    if not 0 <= batch_index < len(batches):
        raise IndexError(f"batch_index {batch_index} out of range (have {len(batches)} batches)")

    prepared: dict[str, object] = {}
    for key, value in batches[batch_index].items():
        if "side_view" in key:
            continue
        if key == "observation.images.hand_view":
            key = "observation.images.wrist"
        elif key == "observation.images.top_view":
            key = "observation.images.top"
        if "image" in key:
            image = np.asarray(value, dtype=np.float32)
            if image.size and float(image.max()) > 1.0:
                image = image / 255.0
            if image.ndim == 3 and image.shape[-1] in {1, 3, 4}:
                image = np.transpose(image, (2, 0, 1))
            prepared[key] = image[None, ...]
        elif isinstance(value, str):
            prepared[key] = value
        else:
            array = np.asarray(value, dtype=np.float32)
            prepared[key] = array[None, ...]
    if task:
        prepared["task"] = task
    else:
        prepared.setdefault("task", "")
    return prepared


def _create_engine(*, policy_path: str, deployment: str, capture: DiagnosticCapture, engine_factory):
    if engine_factory is None:
        from inference_service.core import PureInferenceEngine
        from inference_service.model_sessions import AscendOmModelSession

        def create_session(context, options):
            device_id = context.runtime_options.get("device_id", 0)
            del options
            return AscendOmModelSession(int(device_id), diagnostic_capture=capture)

        engine = PureInferenceEngine(
            model_path=policy_path,
            deployment=deployment,
            pipeline_id="pi05-om-dump",
            runtime_options={},
            model_session_factory=create_session,
        )
    else:
        engine = engine_factory(
            model_path=policy_path,
            deployment=deployment,
            pipeline_id="pi05-om-dump",
            diagnostic_capture=capture,
        )
    return engine


def _dump_one(
    *,
    engine,
    capture: DiagnosticCapture,
    policy_path: str,
    deployment: str,
    batches,
    batch_index: int,
    task: str,
    seed: int,
) -> Path:
    batch = _prepare_batch(batches, batch_index, task)
    noise = generate_pi05_noise(
        (1, int(engine.nominal_chunk_size), int(engine.max_action_dimension)),
        seed + batch_index,
    )
    for key, value in batch.items():
        if not isinstance(value, str):
            capture.save(f"input_{key}", value)
    capture.save("ae_in_noise", noise)

    result = engine(
        dict(batch),
        request_id=f"pi05-om-dump-{batch_index}",
        control_inputs={"noise": noise},
        capture_raw_action=True,
    )
    if result.raw_action is None:
        raise RuntimeError("unified inference pipeline did not return the requested raw action")
    capture.save("raw_action", result.raw_action)
    capture.save("action", result.action)
    capture.write_index(
        {
            "policy_path": str(Path(policy_path)),
            "deployment": deployment,
            "backend": engine.backend_type,
            "policy_type": engine.policy_type,
            "batch_index": batch_index,
            "seed": seed + batch_index,
        }
    )
    return capture.output_dir


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _staging_path(root: Path) -> Path:
    root.parent.mkdir(parents=True, exist_ok=True)
    counter = 0
    while True:
        suffix = f"{os.getpid()}-{counter}"
        staging = root.with_name(f".{root.name}.staging-{suffix}")
        if not staging.exists():
            return staging
        counter += 1


def _rename_exchange(left: Path, right: Path) -> None:
    """Atomically exchange two paths through Linux renameat2."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError("atomic directory exchange requires renameat2")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(left), -100, os.fsencode(right), 2) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), f"{left} <-> {right}")


def _publish_staging(staging: Path, root: Path) -> None:
    if not (root.exists() or root.is_symlink()):
        os.replace(staging, root)
        return
    _rename_exchange(staging, root)
    try:
        _remove_path(staging)
    except OSError as exc:
        LOGGER.warning("Unable to remove replaced PI05 dump generation %s: %s", staging, exc)


def _dump_batches(
    *,
    policy_path: str,
    deployment: str,
    batch_path: str,
    batch_indices: list[int],
    output_dir: str,
    task: str,
    seed: int,
    engine_factory,
    sample_directories: bool,
) -> list[Path]:
    if not batch_indices:
        raise ValueError("batch_indices must not be empty")

    batches = load_observation_batch(batch_path)
    root = Path(output_dir)
    staging = _staging_path(root)
    first_output = staging / f"sample_{batch_indices[0]:04d}" if sample_directories else staging
    capture = DiagnosticCapture(first_output)
    engine = None
    try:
        engine = _create_engine(
            policy_path=policy_path,
            deployment=deployment,
            capture=capture,
            engine_factory=engine_factory,
        )
        try:
            if engine.policy_type.lower() != "pi05" or engine.backend_type != "ascend":
                raise ValueError(
                    "pi05-om-dump requires a PI0.5 deployment using the Ascend backend; "
                    f"selected policy={engine.policy_type!r}, backend={engine.backend_type!r}"
                )
            if engine.nominal_chunk_size is None or engine.max_action_dimension is None:
                raise RuntimeError("PI05 dump requires chunk_size and max_action_dim in policy metadata")

            for batch_index in batch_indices:
                sample_output = staging / f"sample_{batch_index:04d}" if sample_directories else staging
                capture.reset(sample_output)
                _dump_one(
                    engine=engine,
                    capture=capture,
                    policy_path=policy_path,
                    deployment=deployment,
                    batches=batches,
                    batch_index=batch_index,
                    task=task,
                    seed=seed,
                )
        finally:
            engine.close()

        _publish_staging(staging, root)
    finally:
        if staging.exists() or staging.is_symlink():
            try:
                _remove_path(staging)
            except OSError as exc:
                LOGGER.warning("Unable to remove PI05 dump staging path %s: %s", staging, exc)

    if sample_directories:
        return [root / f"sample_{batch_index:04d}" for batch_index in batch_indices]
    return [root]


def dump_pi05_om_batches(
    *,
    policy_path: str,
    deployment: str,
    batch_path: str,
    batch_indices: list[int],
    output_dir: str,
    task: str = "",
    seed: int = 42,
    engine_factory=None,
) -> list[Path]:
    """Dump multiple samples with one loaded Ascend engine."""
    return _dump_batches(
        policy_path=policy_path,
        deployment=deployment,
        batch_path=batch_path,
        batch_indices=batch_indices,
        output_dir=output_dir,
        task=task,
        seed=seed,
        engine_factory=engine_factory,
        sample_directories=True,
    )


def dump_pi05_om(
    *,
    policy_path: str,
    deployment: str,
    batch_path: str,
    batch_index: int,
    output_dir: str,
    task: str = "",
    seed: int = 42,
    engine_factory=None,
) -> Path:
    """Run one PI0.5 OM sample and explicitly capture reproducible diagnostics."""
    return _dump_batches(
        policy_path=policy_path,
        deployment=deployment,
        batch_path=batch_path,
        batch_indices=[batch_index],
        output_dir=output_dir,
        task=task,
        seed=seed,
        engine_factory=engine_factory,
        sample_directories=False,
    )[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True, help="Policy bundle containing inference_manifest.json.")
    parser.add_argument("--deployment", required=True, help="Exact named Ascend deployment from the manifest.")
    parser.add_argument(
        "--batch-path",
        required=True,
        help="Raw observation batch (.safetensors; legacy .json is also supported).",
    )
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-count", type=int, default=1, help="Dump consecutive batches with one loaded engine.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--task", default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_count < 1:
        raise ValueError("--batch-count must be positive")
    if args.batch_count == 1:
        dump_pi05_om(
            policy_path=args.policy_path,
            deployment=args.deployment,
            batch_path=args.batch_path,
            batch_index=args.batch_index,
            output_dir=args.out_dir,
            task=args.task,
            seed=args.seed,
        )
    else:
        dump_pi05_om_batches(
            policy_path=args.policy_path,
            deployment=args.deployment,
            batch_path=args.batch_path,
            batch_indices=list(range(args.batch_index, args.batch_index + args.batch_count)),
            output_dir=args.out_dir,
            task=args.task,
            seed=args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
