# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Compile the GraspGen ONNX subgraphs to Ascend OM artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

OM_MANIFEST_NAME = "graspgen.om.json"


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict) or manifest.get("model_type") != "graspgen":
        raise ValueError(f"not a GraspGen export manifest: {path}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"manifest has no artifacts: {path}")
    return manifest


def format_input_shape(inputs: dict[str, Any]) -> str:
    parts = []
    for name, shape in inputs.items():
        if not isinstance(shape, list) or not shape or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise ValueError(f"input {name!r} must have a fully static positive shape, got {shape!r}")
        parts.append(f"{name}:{','.join(str(dim) for dim in shape)}")
    return ";".join(parts)


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _compile_artifact(
    name: str,
    spec: dict[str, Any],
    manifest_dir: Path,
    output_dir: Path,
    atc_path: str,
    soc_version: str,
    precision_mode: str,
    log_level: str,
    extra_args: list[str],
    dry_run: bool,
) -> Path:
    onnx_path = Path(spec["onnx"]).expanduser()
    if not onnx_path.is_absolute():
        onnx_path = manifest_dir / onnx_path
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX artifact {name!r} not found: {onnx_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / name
    command = [
        atc_path,
        "--framework=5",
        f"--model={onnx_path}",
        f"--output={output_stem}",
        "--input_format=ND",
        f"--input_shape={format_input_shape(spec['inputs'])}",
        f"--precision_mode_v2={precision_mode}",
        f"--soc_version={soc_version}",
        f"--log={log_level}",
        *extra_args,
    ]
    print("$ " + " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)
        om_path = output_stem.with_suffix(".om")
        if not om_path.is_file():
            raise RuntimeError(f"ATC finished without producing {om_path}")
    return output_stem.with_suffix(".om")


def _resolve_atc(explicit_path: str | None, dry_run: bool) -> str:
    candidate = explicit_path or shutil.which("atc")
    if candidate:
        resolved = Path(candidate).expanduser().resolve()
        if resolved.as_posix() == "/usr/games/atc":
            candidate = None
        elif dry_run or resolved.is_file():
            return str(resolved)
    if dry_run:
        return "atc"
    raise RuntimeError(
        "CANN atc not found; source the CANN toolkit environment or pass --atc-path. "
        "/usr/games/atc is an unrelated game."
    )


def _load_existing_om_manifest(path: Path, source_manifest: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as file:
        existing = json.load(file)
    source_value = existing.get("source_manifest")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError(f"existing OM manifest has no source_manifest: {path}")
    existing_source = Path(source_value).expanduser()
    if not existing_source.is_absolute():
        existing_source = path.parent / existing_source
    if existing_source.resolve() != source_manifest.resolve():
        raise ValueError(
            f"existing OM manifest {path} belongs to a different ONNX manifest: {existing.get('source_manifest')!r}"
        )
    artifacts = existing.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError(f"existing OM manifest has invalid artifacts: {path}")
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile GraspGen ONNX subgraphs to OM")
    parser.add_argument("--manifest", required=True, help="Path to graspgen.onnx.json")
    parser.add_argument("--output-dir", default="./output/om")
    parser.add_argument("--soc-version", required=True, help="For example Ascend310P3")
    parser.add_argument("--precision-mode-v2", default="origin")
    parser.add_argument("--log", default="error", choices=["debug", "info", "warning", "error", "null"])
    parser.add_argument("--atc-path", default=None, help="CANN atc executable; overrides PATH lookup")
    parser.add_argument("--atc-arg", action="append", default=[])
    parser.add_argument("--artifacts", nargs="+", default=["all"])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    if not args.soc_version.startswith("Ascend"):
        raise ValueError("--soc-version must be an ATC SoC name such as Ascend310P3")
    atc_path = _resolve_atc(args.atc_path, args.dry_run)

    all_artifacts = manifest["artifacts"]
    requested = list(all_artifacts) if "all" in args.artifacts else args.artifacts
    unknown = [name for name in requested if name not in all_artifacts]
    if unknown:
        raise ValueError(f"unknown artifacts: {unknown}; available: {sorted(all_artifacts)}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    om_manifest_path = output_dir / OM_MANIFEST_NAME
    compiled = {} if args.dry_run else _load_existing_om_manifest(om_manifest_path, manifest_path)
    for name in requested:
        output_path = output_dir / f"{name}.om"
        if args.skip_existing and output_path.is_file():
            print(f"Skipping existing {name}: {output_path}")
            compiled[name] = _relative_or_absolute(output_path, output_dir)
            continue
        om_path = _compile_artifact(
            name=name,
            spec=all_artifacts[name],
            manifest_dir=manifest_path.parent,
            output_dir=output_dir,
            atc_path=atc_path,
            soc_version=args.soc_version,
            precision_mode=args.precision_mode_v2,
            log_level=args.log,
            extra_args=list(args.atc_arg),
            dry_run=args.dry_run,
        )
        compiled[name] = _relative_or_absolute(om_path, output_dir)

    if args.dry_run:
        print("Dry run complete; no OM manifest was written")
        return 0

    execution = [name for name in manifest.get("execution", []) if name in compiled]
    om_manifest = {
        "schema_version": 1,
        "model_type": "graspgen",
        "backend": "ascend",
        "target_runtime": "acl",
        "artifacts": compiled,
        "execution": execution,
        "backend_config": manifest.get("backend_config", {}),
        "source_manifest": str(manifest_path),
    }
    with om_manifest_path.open("w", encoding="utf-8") as file:
        json.dump(om_manifest, file, indent=2)
        file.write("\n")
    print(f"OM conversion complete. Manifest: {om_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
