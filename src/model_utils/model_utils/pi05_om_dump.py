"""Explicit manifest-driven diagnostic dump for a PI0.5 Ascend deployment."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from model_utils.loss_compare import generate_pi05_noise


class DiagnosticCapture:
    """Write explicitly requested diagnostic values as NumPy files."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.values: dict[str, dict[str, Any]] = {}

    def save(self, name: str, value: object) -> None:
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

    def write_index(self, metadata: dict[str, Any]) -> None:
        document = {**metadata, "values": self.values}
        (self.output_dir / "diagnostic_capture.json").write_text(
            json.dumps(document, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _prepare_batch(batch_path: str | Path, batch_index: int, task: str) -> dict[str, object]:
    batches = json.loads(Path(batch_path).read_text(encoding="utf-8"))
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
    prepared.setdefault("task", task)
    return prepared


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
    capture = DiagnosticCapture(output_dir)
    registry_module = None
    if engine_factory is None:
        from inference_service.backends import BackendDescriptor, BackendRegistry
        from inference_service.backends.ascend import AscendBackend
        from inference_service.core import PureInferenceEngine

        registry_module = ModuleType(f"model_utils._pi05_dump_backend_{uuid.uuid4().hex}")

        def create_backend(context):
            device_id = context.runtime_options.get("device_id", 0)
            return AscendBackend(int(device_id), diagnostic_capture=capture)

        registry_module.create_backend = create_backend
        sys.modules[registry_module.__name__] = registry_module
        registry = BackendRegistry(
            {
                "ascend": BackendDescriptor(
                    name="ascend",
                    factory=f"{registry_module.__name__}:create_backend",
                    supported_policy_families=frozenset({"pi05"}),
                    target_validator=lambda deployment: None,
                )
            }
        )
        engine = PureInferenceEngine(
            model_path=policy_path,
            deployment=deployment,
            pipeline_id="pi05-om-dump",
            runtime_options={},
            registry=registry,
        )
    else:
        engine = engine_factory(
            model_path=policy_path,
            deployment=deployment,
            pipeline_id="pi05-om-dump",
            diagnostic_capture=capture,
        )
    try:
        if engine.policy_type.lower() != "pi05" or engine.backend_type != "ascend":
            raise ValueError(
                "pi05-om-dump requires a PI0.5 deployment using the Ascend backend; "
                f"selected policy={engine.policy_type!r}, backend={engine.backend_type!r}"
            )
        if engine.nominal_chunk_size is None or engine.max_action_dimension is None:
            raise RuntimeError("PI0.5 dump requires chunk_size and max_action_dim in policy metadata")

        batch = _prepare_batch(batch_path, batch_index, task)
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
    finally:
        engine.close()
        if registry_module is not None:
            sys.modules.pop(registry_module.__name__, None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True, help="Policy bundle containing inference_manifest.json.")
    parser.add_argument("--deployment", required=True, help="Exact named Ascend deployment from the manifest.")
    parser.add_argument("--batch-path", required=True, help="loss_compare-compatible batches.json.")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--task", default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dump_pi05_om(
        policy_path=args.policy_path,
        deployment=args.deployment,
        batch_path=args.batch_path,
        batch_index=args.batch_index,
        output_dir=args.out_dir,
        task=args.task,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
