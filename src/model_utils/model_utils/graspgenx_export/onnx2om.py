#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Compile the GraspGen-X diffusion loop ONNX to an Ascend offline model.

The graph is shape specific -- one grasp count and one step count -- so the
input shapes come from the export manifest rather than the command line, and
the result is named for the count it covers.

`--precision-mode force_fp32` is the mode the graph was designed for: on
Ascend310P1 it means float16 multiplies with float32 accumulation, the
arithmetic class the eager path already gets. `must_keep_origin_dtype` is
genuinely float32 but made a single 2048x2048 projection take 25 ms against
0.25 ms.

Which toolkit compiles the graph is recorded rather than assumed: a host may
carry several CANN releases, and `latest` can point at any of them. ATC embeds
its own command line in the result, so
`strings <model>.om | grep -A1 atc_cmdline` reads it back.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from model_utils.export_paths import resolve_outside_bundle_path
from model_utils.graspgenx_export.binding import validate_binding

MANIFEST_NAME = "graspgenx.onnx.json"


def _load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("model_type") != "graspgenx":
        raise ValueError(f"not a GraspGen-X export manifest: {path}")
    for key in ("onnx", "grasps", "steps", "input_shape", "sha256"):
        if key not in manifest:
            raise ValueError(f"{path} has no {key!r}")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def format_input_shape(shapes: dict) -> str:
    """ATC's --input_shape syntax: name:dim,dim;name:dim,dim.

    The manifest's sha256 covers the ONNX, not the shapes beside it, so an
    edited or stale entry would otherwise reach ATC unexamined. A negative
    dimension there is a dynamic axis ATC may well accept, producing a graph
    the loader's shape check cannot speak for.
    """
    for name, dimensions in shapes.items():
        if not dimensions:
            raise ValueError(f"input {name!r} has no shape")
        for dimension in dimensions:
            if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
                raise ValueError(f"input {name!r} has a non-positive or non-integer dimension: {dimensions}")
    return ";".join(
        f"{name}:{','.join(str(dimension) for dimension in dimensions)}" for name, dimensions in shapes.items()
    )


def resolve_output_dir(manifest: dict, manifest_path: Path, override: Path | None) -> Path:
    """Return where ATC writes, refusing the bundle the graph was exported from.

    A published bundle holds only manifest-referenced artifacts, so the .om,
    its metadata and ATC's own leavings belong in the work directory. The
    export manifest records the bundle it read; a bundle that has since moved
    away needs no protecting, so a missing one is not an error here.
    """
    output_dir = (override or manifest_path.parent).resolve()
    recorded_bundle = manifest.get("bundle_root")
    if recorded_bundle and Path(recorded_bundle).is_dir():
        resolve_outside_bundle_path(recorded_bundle, output_dir)
    return output_dir


def _resolve_atc(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("atc")
    if found:
        return found
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if home and (candidate := Path(home) / "bin/atc").is_file():
        return str(candidate)
    raise SystemExit(
        "atc not found: source the CANN toolkit's setenv.bash or pass --atc. "
        "Note which release it belongs to; a host may have several."
    )


def _release_root(atc: str) -> Path:
    """The CANN release the given atc binary belongs to."""
    resolved = Path(atc).resolve()
    # <release>/<arch>-linux/bin/atc, so the release root is three levels up.
    for candidate in (resolved.parents[2], resolved.parents[1]):
        if (candidate / "aarch64-linux").is_dir() or (candidate / "x86_64-linux").is_dir():
            return candidate
    configured = os.environ.get("ASCEND_HOME_PATH")
    if configured:
        return Path(configured)
    raise SystemExit(f"cannot tell which CANN release {atc} belongs to")


def _toolkit_environment(atc: str) -> dict[str, str]:
    """The environment ATC needs, taken from its own release's setenv.bash.

    Setting a few variables by hand is not enough and fails in a way that
    misleads: without the release's `LD_LIBRARY_PATH` the ONNX parser plugin
    never loads, and ATC then reports `E19010 No parser is registered` for
    *every* node in the graph -- including Add and Relu -- which reads like an
    unsupported-operator problem rather than a missing library. Sourcing the
    release's own script is also what keeps a host with several installs
    honest: whichever atc was asked for brings its own paths, instead of
    inheriting whatever `latest` currently points at.

    Likewise `tbe` comes from the toolkit's site-packages, not from any venv,
    so a run that keeps the venv's PYTHONPATH fails with
    `EC0010 ... No module named 'tbe'`.
    """
    release = _release_root(atc)
    architecture = "aarch64-linux" if (release / "aarch64-linux").is_dir() else "x86_64-linux"
    setenv = release / architecture / "bin/setenv.bash"
    if not setenv.is_file():
        raise SystemExit(f"{release} has no {architecture}/bin/setenv.bash; is it a full install?")

    # Sourcing has to happen in a shell, so the resulting environment is read
    # back rather than reconstructed. `env -0` keeps values containing
    # newlines intact.
    completed = subprocess.run(
        ["bash", "-c", f'source "{setenv}" >/dev/null 2>&1; env -0'],
        check=True,
        capture_output=True,
    )
    environment = {}
    for entry in completed.stdout.split(b"\0"):
        if not entry:
            continue
        name, _, value = entry.decode("utf-8", "replace").partition("=")
        environment[name] = value

    # The toolkit's Python packages, ahead of anything the caller had.
    site_packages = [
        str(path)
        for path in (
            release / "toolkit/python/site-packages",
            release / "compiler/python/site-packages",
            release / "python/site-packages",
        )
        if path.is_dir()
    ]
    if site_packages:
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = ":".join(site_packages + ([existing] if existing else []))
    environment["ASCEND_HOME_PATH"] = str(release)
    environment["ASCEND_TOOLKIT_HOME"] = str(release)

    print(f"toolkit    {release}")
    print(f"opp        {environment.get('ASCEND_OPP_PATH', '(unset)')}")
    return environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help=f"The {MANIFEST_NAME} written by export_onnx.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to the manifest's own directory.")
    parser.add_argument("--soc-version", default="Ascend310P1")
    parser.add_argument(
        "--precision-mode", default="force_fp32", choices=("force_fp32", "allow_fp32_to_fp16", "must_keep_origin_dtype")
    )
    parser.add_argument("--atc", default=None, help="Path to the atc binary.")
    parser.add_argument("--log", default="error", choices=("debug", "info", "warning", "error", "null"))
    parser.add_argument("--dry-run", action="store_true", help="Print the command and the shapes without compiling.")
    arguments = parser.parse_args()

    manifest_path = arguments.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    source = manifest_path.parent / manifest["onnx"]
    if not source.is_file():
        raise SystemExit(f"the manifest names {source}, which is not there")
    actual = _sha256(source)
    if actual != manifest["sha256"]:
        raise SystemExit(
            f"{source.name} does not match its manifest: {actual[:16]} against "
            f"{manifest['sha256'][:16]}. Re-run export_onnx rather than compiling "
            f"a graph whose provenance is unknown."
        )
    metadata_source = manifest_path.parent / manifest["metadata"]
    if _sha256(metadata_source) != manifest.get("metadata_sha256"):
        raise SystemExit("loop metadata does not match the export manifest; re-run export_onnx")
    metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
    binding = manifest.get("generator_binding")
    if not isinstance(binding, dict):
        raise SystemExit("export has no generator binding; re-run export_onnx")
    validate_binding(metadata, binding, manifest["grasps"])

    output_dir = resolve_output_dir(manifest, manifest_path, arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"diffusion_loop_g{manifest['grasps']}"
    atc = _resolve_atc(arguments.atc)
    command = [
        atc,
        f"--model={source}",
        "--framework=5",
        f"--output={stem}",
        "--input_format=ND",
        f"--input_shape={format_input_shape(manifest['input_shape'])}",
        f"--soc_version={arguments.soc_version}",
        f"--precision_mode={arguments.precision_mode}",
        f"--log={arguments.log}",
    ]
    print(" \\\n  ".join(command), flush=True)
    if arguments.dry_run:
        return 0

    compiled = stem.with_suffix(".om")
    # ATC writes fusion_result.json and kernel_meta/ into the process working
    # directory, not next to --output, so it is pinned to the work directory
    # instead of littering whatever the operator happened to stand in.
    completed = subprocess.run(command, check=False, cwd=output_dir, env=_toolkit_environment(atc))
    if completed.returncode != 0:
        # A truncated .om left behind would be picked up by the next
        # --diffusion-loop packaging run as though it had compiled.
        compiled.unlink(missing_ok=True)
        raise SystemExit(
            f"atc failed with exit code {completed.returncode}; its output is above. "
            f"Intermediates are under {output_dir}"
        )
    if not compiled.is_file():
        raise SystemExit(f"atc reported success but {compiled} is not there")

    # The metadata travels with the model: the loader checks the grasp and step
    # counts against the bundle before it will use the graph.
    metadata_destination = compiled.with_suffix(".json")
    if metadata_source.is_file() and metadata_source != metadata_destination:
        shutil.copy2(metadata_source, metadata_destination)

    print(f"\nwrote {compiled} ({compiled.stat().st_size / 1e6:.1f} MB)")
    print(f"sha256 {_sha256(compiled)}")
    print(
        f"\nPackage it into a bundle with:\n"
        f"  python -m perception_service.package_graspgenx_torch_bundle \\\n"
        f"    --bundle-root <bundle> --deployment torch_npu --device npu \\\n"
        f"    --num-grasps {manifest['grasps']} --topk {manifest['grasps']} \\\n"
        f"    --diffusion-loop {compiled}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
