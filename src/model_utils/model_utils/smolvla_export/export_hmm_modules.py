#!/usr/bin/env python3
"""Export SmolVLA modules to Houmo HMONNX with the repository LeRobot source."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

import onnx
import torch

from model_utils.smolvla_export.export_rknn_modules import export_all


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _onnx_dtype(element_type: int) -> torch.dtype:
    return {
        onnx.TensorProto.BOOL: torch.bool,
        onnx.TensorProto.FLOAT: torch.float32,
        onnx.TensorProto.FLOAT16: torch.float16,
        onnx.TensorProto.INT8: torch.int8,
        onnx.TensorProto.INT32: torch.int32,
        onnx.TensorProto.INT64: torch.int64,
    }[element_type]


def _calibration_inputs(model_path: Path) -> tuple[torch.Tensor, ...]:
    model = onnx.load(str(model_path), load_external_data=False)
    inputs: list[torch.Tensor] = []
    for value in model.graph.input:
        tensor_type = value.type.tensor_type
        shape = tuple(int(dimension.dim_value) for dimension in tensor_type.shape.dim)
        dtype = _onnx_dtype(tensor_type.elem_type)
        if dtype == torch.bool:
            tensor = torch.ones(shape, dtype=dtype)
        elif dtype.is_floating_point:
            tensor = torch.randn(shape, dtype=dtype)
        else:
            tensor = torch.ones(shape, dtype=dtype)
        inputs.append(tensor)
    return tuple(inputs)


def _convert_to_hmonnx(
    onnx_path: Path,
    hmonnx_path: Path,
    quant_type: str,
    calibration_path: Path | None = None,
) -> None:
    from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config

    hmonnx_path.parent.mkdir(parents=True, exist_ok=True)
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    calibration_inputs = (
        torch.load(calibration_path, map_location="cpu", weights_only=True)
        if calibration_path is not None
        else _calibration_inputs(onnx_path)
    )
    convert_onnx_to_hmonnx(
        str(onnx_path),
        calibration_inputs,
        out_hmonnx_file=str(hmonnx_path),
        device_type="XH2A",
        quant_config=create_quant_config(quant_scheme),
    )


def _write_provenance(args: argparse.Namespace, output_dir: Path) -> Path:
    repo_root = Path(args.repo_root).resolve()
    lerobot_root = Path(args.lerobot_src).resolve().parent
    checkpoint = Path(args.model_path).resolve() / "model.safetensors"
    provenance = {
        "schema_version": 1,
        "pipeline": "ibrobot-smolvla-hmm",
        "ib_robot": {
            "commit": _git_output(repo_root, "rev-parse", "HEAD"),
            "dirty_files": _git_output(repo_root, "status", "--short").splitlines(),
        },
        "lerobot": {
            "source": str(Path(args.lerobot_src).resolve()),
            "branch": _git_output(lerobot_root, "branch", "--show-current"),
            "head": _git_output(lerobot_root, "rev-parse", "HEAD"),
            "base": _git_output(lerobot_root, "merge-base", "HEAD", "v0.5.1"),
            "dirty_files": _git_output(lerobot_root, "status", "--short").splitlines(),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
        },
        "toolchain": {
            "image_id": os.environ.get("IBR_HOUMO_IMAGE_ID", "unknown"),
            "houmo_version": os.environ.get("HOUMO_VERSION", "unknown"),
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "transformers_selected": os.environ.get("IBR_TRANSFORMERS_SELECTED", "unknown"),
            "tokenizers": _package_version("tokenizers"),
            "diffusers": _package_version("diffusers"),
            "onnx": onnx.__version__,
            "onnxsim": _package_version("onnxsim"),
            "xhquant": _package_version("hmquant-xh2"),
            "tcim": _package_version("houmo-tcim-xh2"),
        },
        "export": {
            "device": args.device,
            "image_shape": [1, 3, args.image_height, args.image_width],
            "opset": args.opset,
            "quant_type": args.quant_type,
            "seed": args.seed,
        },
    }
    path = output_dir / "provenance.json"
    path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def export_hmm(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    rknn_args = argparse.Namespace(
        model_path=args.model_path,
        lerobot_src=args.lerobot_src,
        output_dir=str(output_dir),
        device=args.device,
        image_height=args.image_height,
        image_width=args.image_width,
        prefix_length=None,
        opset=args.opset,
        seed=args.seed,
        calibration_dir=str(output_dir / "calibration"),
    )
    export_all(rknn_args)

    onnx_dir = output_dir / "onnx"
    hmonnx_dir = output_dir / "hmonnx"
    modules = {
        "vision": onnx_dir / "smolvla_vision_simp.onnx",
        "prefill": onnx_dir / "smolvla_prefill_simp.onnx",
        "action": onnx_dir / "smolvla_action_simp.onnx",
    }
    for role, onnx_path in modules.items():
        print(f"Converting {role} ONNX to HMONNX...")
        _convert_to_hmonnx(
            onnx_path,
            hmonnx_dir / f"smolvla_{role}_xh2.onnx",
            args.quant_type,
            output_dir / "calibration" / f"{role}.pt",
        )

    provenance_path = _write_provenance(args, output_dir)
    print(f"HMM export provenance: {provenance_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lerobot-src", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--quant-type", default="w8a8h1_sefp")
    parser.add_argument("--seed", type=int, default=42)
    export_hmm(parser.parse_args())


if __name__ == "__main__":
    main()
