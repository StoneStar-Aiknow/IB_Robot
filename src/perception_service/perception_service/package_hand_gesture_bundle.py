"""Package an already-built Orin engine; never compile or optimize on startup."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from inference_manifest import BundleFile, canonical_bundle_digest, load_inference_manifest, write_inference_manifest

from .hand_gesture_adapter import (
    INPUT_SEMANTIC,
    INPUT_SHAPE,
    OUTPUT_NAME,
    OUTPUT_SEMANTIC,
    OUTPUT_SHAPE,
    VERIFIED_ENGINE_SHA256,
    HandGestureAdapter,
)

BUNDLE_UUID = "54680c81-880b-5523-a4a0-4cedc2bcf655"
DEPLOYMENT_UUID = "87983598-4e6c-573d-80a0-f43775270fbe"
DEPLOYMENT = "tensorrt_orin_fp32"


def package_hand_gesture_bundle(destination, engine, *, expected_sha256=VERIFIED_ENGINE_SHA256):
    """Publish a new bundle atomically, with a digest of the copied bytes."""
    destination, engine = Path(destination), Path(engine)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle {destination}")
    if not engine.is_file() or not 0 < engine.stat().st_size <= 64 * 1024 * 1024:
        raise ValueError("engine must be a nonempty file no larger than 64 MiB")
    blob = engine.read_bytes()
    actual = hashlib.sha256(blob).hexdigest()
    if actual != expected_sha256:
        raise ValueError("engine SHA-256 differs from the explicitly selected verified engine")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hand-gesture-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "bundle"
        (staging / "assets").mkdir(parents=True)
        artifact = "artifacts/tensorrt_orin/yolov10n_gestures_320_fp32.engine"
        artifact_path = staging / artifact
        artifact_path.parent.mkdir(parents=True)
        artifact_path.write_bytes(blob)
        identity = HandGestureAdapter.identity
        (staging / "assets" / "adapter.json").write_text(
            json.dumps(
                {
                    "interface": "tensor_model",
                    "model_type": identity.model_type,
                    "operation": identity.operation,
                    "preprocessing": identity.preprocessing,
                    "postprocessing": identity.postprocessing,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        files = [BundleFile(path="assets/adapter.json")]

        def tensor(semantic, shape, **extra):
            return {"semantic": semantic, "dtype": "float32", "shape": list(shape), **extra}

        inputs = [tensor(INPUT_SEMANTIC, INPUT_SHAPE, layout="NCHW")]
        outputs = [tensor(OUTPUT_SEMANTIC, OUTPUT_SHAPE)]
        manifest = {
            "schema_version": 3,
            "bundle": {
                "uuid": BUNDLE_UUID,
                "revision": 1,
                "name": "x2_hand_gesture_yolov10n_320",
                "files": [entry.model_dump(mode="json") for entry in files],
                "digest": {
                    "algorithm": "sha256",
                    "scope": "structure",
                    "value": canonical_bundle_digest(BUNDLE_UUID, 1, "x2_hand_gesture_yolov10n_320", files),
                },
            },
            "model": {
                "interface": "tensor_model",
                "model_type": identity.model_type,
                "operation": identity.operation,
                "inputs": inputs,
                "outputs": outputs,
                "semantic_identity": {
                    "logical_model_revision": "yolov10n-hagridv2-320-cut-v1",
                    "preprocessing_contract": identity.preprocessing,
                    "output_semantics": identity.postprocessing,
                },
            },
            "deployments": {
                DEPLOYMENT: {
                    "uuid": DEPLOYMENT_UUID,
                    "revision": 1,
                    "execution_contract": {
                        "state_scope": "request",
                        "execution_structure": "direct",
                        "cancellation_granularity": "request_boundary",
                    },
                    "runtime_profile": {
                        "backend": "tensorrt",
                        "target": {"soc": "orin", "runtime": "tensorrt", "runtime_abi": "tensorrt-10.7.0-cuda-12.6"},
                        "profile": {"precision": "fp32", "engine_version": "10.7.0", "device_id": 0},
                    },
                    "execution": ["model"],
                    "artifacts": {"model": {"path": artifact, "format": "engine", "sha256": actual}},
                    "bindings": {
                        "model": {
                            "inputs": [{**inputs[0], "index": 0, "runtime_name": "images"}],
                            "outputs": [{**outputs[0], "index": 0, "runtime_name": OUTPUT_NAME}],
                        }
                    },
                }
            },
        }
        write_inference_manifest(staging / "inference_manifest.json", manifest)
        load_inference_manifest(staging, DEPLOYMENT)
        # An existing nonempty deployment can never be replaced by this rename.
        staging.rename(destination)
    return destination


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(args)
    root = package_hand_gesture_bundle(parsed.output, parsed.engine)
    print(f"packaged existing verified engine at {root}")


if __name__ == "__main__":
    main()
