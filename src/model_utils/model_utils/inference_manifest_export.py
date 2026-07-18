"""Shared helpers for producing strict unified inference manifests."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import onnx

from inference_manifest import (
    ArtifactBindings,
    BundleFile,
    CompiledDeployment,
    Deployment,
    DeploymentArtifact,
    DeploymentTarget,
    DeviceLink,
    Digest,
    InferenceManifest,
    ManifestBundle,
    TensorBinding,
    ValidatedManifest,
    canonical_bundle_digest,
    load_inference_manifest,
    load_policy_metadata,
    normalize_bundle_path,
    sha256_file,
    write_inference_manifest,
)
from inference_manifest.json_utils import load_json_strict
from inference_manifest.schema import validate_manifest_schema

_ONNX_DTYPES = {
    onnx.TensorProto.BOOL: "bool",
    onnx.TensorProto.UINT8: "uint8",
    onnx.TensorProto.INT8: "int8",
    onnx.TensorProto.INT16: "int16",
    onnx.TensorProto.INT32: "int32",
    onnx.TensorProto.INT64: "int64",
    onnx.TensorProto.FLOAT16: "float16",
    onnx.TensorProto.BFLOAT16: "bfloat16",
    onnx.TensorProto.FLOAT: "float32",
    onnx.TensorProto.DOUBLE: "float64",
}


@dataclass(frozen=True)
class RuntimeTensor:
    """One positional tensor exposed by a compiled runtime ABI."""

    name: str
    index: int
    dtype: str
    shape: tuple[int, ...]
    layout: str | None = None


@dataclass(frozen=True)
class RuntimeABI:
    """Input and output tensors in runtime order."""

    inputs: tuple[RuntimeTensor, ...]
    outputs: tuple[RuntimeTensor, ...]


def read_onnx_abi(path: str | Path) -> RuntimeABI:
    """Read the positional tensor ABI from an ONNX graph."""

    model_path = Path(path).expanduser().resolve(strict=True)
    model = onnx.load(str(model_path), load_external_data=False)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs = tuple(
        _onnx_tensor(value, index, model_path)
        for index, value in enumerate(item for item in model.graph.input if item.name not in initializer_names)
    )
    outputs = tuple(_onnx_tensor(value, index, model_path) for index, value in enumerate(model.graph.output))
    if not inputs or not outputs:
        raise ValueError(f"ONNX ABI must contain at least one input and one output: {model_path}")
    return RuntimeABI(inputs=inputs, outputs=outputs)


def read_runtime_abi(path: str | Path) -> RuntimeABI:
    """Read runtime-introspected ABI metadata emitted by a compiler toolchain."""

    metadata_path = Path(path).expanduser().resolve(strict=True)
    with metadata_path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Runtime ABI metadata must be a JSON object: {metadata_path}")
    return RuntimeABI(
        inputs=_parse_runtime_tensors(value.get("inputs"), "inputs", metadata_path),
        outputs=_parse_runtime_tensors(value.get("outputs"), "outputs", metadata_path),
    )


def read_tcim_abi(path: str | Path) -> RuntimeABI:
    """Read TCIM compiler ``model.json`` input/output descriptors."""

    metadata_path = Path(path).expanduser().resolve(strict=True)
    with metadata_path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    model = value.get("Model") if isinstance(value, dict) else None
    if not isinstance(model, dict):
        raise ValueError(f"TCIM metadata must contain a Model object: {metadata_path}")
    return RuntimeABI(
        inputs=_parse_tcim_tensors(model.get("inputs"), "inputs", metadata_path),
        outputs=_parse_tcim_tensors(model.get("outputs"), "outputs", metadata_path),
    )


def artifact_bindings(
    abi: RuntimeABI,
    *,
    input_semantics: Mapping[str, str],
    output_semantics: Mapping[str, str],
    image_layouts: Mapping[str, str] | None = None,
) -> ArtifactBindings:
    """Map every runtime ABI tensor to one explicit manifest semantic."""

    layouts = dict(image_layouts or {})
    known_semantics = set(input_semantics.values()) | set(output_semantics.values())
    unexpected_layouts = sorted(set(layouts) - known_semantics)
    if unexpected_layouts:
        raise ValueError(f"Image layouts reference unknown semantics: {unexpected_layouts}")
    return ArtifactBindings(
        inputs=tuple(_binding(tensor, input_semantics, layouts, "input") for tensor in abi.inputs),
        outputs=tuple(_binding(tensor, output_semantics, layouts, "output") for tensor in abi.outputs),
    )


def deployment_artifact(bundle_root: str | Path, path: str | Path, artifact_format: str) -> DeploymentArtifact:
    """Create a hashed artifact whose path is safely contained by the bundle."""

    root = Path(bundle_root).expanduser().resolve(strict=True)
    artifact_path = Path(path).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = root / artifact_path
    artifact_path = artifact_path.resolve(strict=True)
    if not artifact_path.is_file():
        raise ValueError(f"Deployment artifact is not a regular file: {artifact_path}")
    try:
        relative = artifact_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Deployment artifact escapes bundle root {root}: {artifact_path}") from exc
    return DeploymentArtifact(
        path=normalize_bundle_path(relative),
        format=artifact_format,
        sha256=sha256_file(artifact_path),
    )


def package_deployment_artifact(
    bundle_root: str | Path,
    source_path: str | Path,
    *,
    backend: str,
    deployment_name: str,
    role: str,
    force_copy: bool = False,
) -> Path:
    """Copy a compiler output into the canonical artifact tree when needed."""

    root = Path(bundle_root).expanduser().resolve(strict=True)
    source = Path(source_path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Deployment artifact is not a regular file: {source}")
    try:
        source.relative_to(root)
        contained = True
    except ValueError:
        contained = False
    if force_copy or not contained:
        suffix = "".join(source.suffixes)
        digest = sha256_file(source)
        relative = normalize_bundle_path(f"artifacts/{backend}/{deployment_name}/{role}-{digest[:12]}{suffix}")
        destination = root.joinpath(*relative.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source == destination:
            return source
        shutil.copy2(source, destination)
        return destination
    return source


def compiled_deployment(
    bundle_root: str | Path,
    *,
    backend: str,
    target_soc: str,
    target_runtime: str,
    artifacts: Mapping[str, tuple[str | Path, str]],
    execution: Sequence[str],
    bindings: Mapping[str, ArtifactBindings],
    device_links: Sequence[DeviceLink] = (),
) -> CompiledDeployment:
    """Build a typed compiled deployment with current artifact hashes."""

    return CompiledDeployment(
        backend=backend,
        target=DeploymentTarget(soc=target_soc, runtime=target_runtime),
        artifacts={
            role: deployment_artifact(bundle_root, path, artifact_format)
            for role, (path, artifact_format) in artifacts.items()
        },
        execution=tuple(execution),
        bindings=dict(bindings),
        device_links=tuple(device_links),
    )


def upsert_deployment(
    bundle_root: str | Path,
    deployment_name: str,
    deployment: Deployment,
    *,
    bundle_name: str | None = None,
) -> ValidatedManifest:
    """Write one deployment and prove the production strict loader accepts it."""

    root = Path(bundle_root).expanduser().resolve(strict=True)
    manifest_path = root / "inference_manifest.json"
    existing = _load_existing_manifest(manifest_path)
    deployments = dict(existing.deployments) if existing is not None else {}
    deployments[deployment_name] = deployment

    require_native_weights = any(candidate.backend == "torch" for candidate in deployments.values())
    policy = load_policy_metadata(root, require_native_weights=require_native_weights)
    if policy.external_dependencies:
        dependencies = [f"{dependency.source}={dependency.identifier!r}" for dependency in policy.external_dependencies]
        raise ValueError(
            "Unified manifest finalization requires all semantic dependencies to be local bundle assets; "
            f"vendor these references inside the bundle and update the LeRobot metadata: {dependencies}"
        )
    bundle_files = tuple(
        BundleFile(path=path, sha256=sha256_file(root.joinpath(*path.split("/")))) for path in policy.required_files
    )
    name = bundle_name or (existing.bundle.name if existing is not None else root.name)
    manifest = InferenceManifest(
        schema_version=1,
        bundle=ManifestBundle(
            name=name,
            files=bundle_files,
            digest=Digest(algorithm="sha256", value=canonical_bundle_digest(bundle_files)),
        ),
        deployments=deployments,
    )
    try:
        write_inference_manifest(manifest_path, manifest)
        validated_deployments = {name: load_inference_manifest(root, name) for name in sorted(deployments)}
    except Exception:
        if existing is None:
            manifest_path.unlink(missing_ok=True)
        else:
            write_inference_manifest(manifest_path, existing)
        raise
    return validated_deployments[deployment_name]


def copy_policy_metadata_bundle(source_root: str | Path, destination_root: str | Path) -> tuple[str, ...]:
    """Copy required read-only LeRobot semantic files into a compiled bundle."""

    source = Path(source_root).expanduser().resolve(strict=True)
    destination = Path(destination_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    policy = load_policy_metadata(source, require_native_weights=False)
    for relative in policy.required_files:
        source_path = source.joinpath(*relative.split("/"))
        destination_path = destination.joinpath(*relative.split("/"))
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
    return policy.required_files


def _onnx_tensor(value: object, index: int, path: Path) -> RuntimeTensor:
    tensor_type = value.type.tensor_type
    try:
        dtype = _ONNX_DTYPES[tensor_type.elem_type]
    except KeyError as exc:
        dtype_name = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
        raise ValueError(f"Unsupported ONNX tensor dtype {dtype_name!r} for {value.name!r} in {path}") from exc
    shape = tuple(int(dimension.dim_value) if dimension.dim_value > 0 else -1 for dimension in tensor_type.shape.dim)
    if not shape:
        raise ValueError(f"Scalar ONNX tensor {value.name!r} cannot be represented as a manifest binding: {path}")
    return RuntimeTensor(name=value.name, index=index, dtype=dtype, shape=shape)


def _parse_runtime_tensors(value: object, direction: str, path: Path) -> tuple[RuntimeTensor, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Runtime ABI {direction} must be a non-empty list: {path}")
    tensors: list[RuntimeTensor] = []
    for expected_index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"Runtime ABI {direction}[{expected_index}] must be an object: {path}")
        name = item.get("name")
        dtype = item.get("dtype")
        shape = item.get("shape")
        index = item.get("index", expected_index)
        if not isinstance(name, str) or not name:
            raise ValueError(f"Runtime ABI {direction}[{expected_index}] requires a non-empty name: {path}")
        if not isinstance(dtype, str) or dtype not in set(_ONNX_DTYPES.values()):
            raise ValueError(f"Runtime ABI tensor {name!r} has unsupported dtype {dtype!r}: {path}")
        if type(index) is not int or index < 0:
            raise ValueError(f"Runtime ABI tensor {name!r} has invalid index {index!r}: {path}")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(dimension) is not int or dimension == 0 or dimension < -1 for dimension in shape)
        ):
            raise ValueError(f"Runtime ABI tensor {name!r} has invalid shape {shape!r}: {path}")
        layout = item.get("layout")
        if isinstance(layout, str):
            layout = layout.upper()
        if layout is not None and layout not in {"NCHW", "NHWC"}:
            raise ValueError(f"Runtime ABI tensor {name!r} has invalid layout {layout!r}: {path}")
        tensors.append(RuntimeTensor(name=name, index=index, dtype=dtype, shape=tuple(shape), layout=layout))
    indices = [tensor.index for tensor in tensors]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Runtime ABI {direction} contains duplicate indices: {path}")
    if direction == "inputs" and sorted(indices) != list(range(len(indices))):
        raise ValueError(f"Runtime ABI inputs indices must be contiguous from zero: {path}")
    return tuple(tensors)


def _parse_tcim_tensors(value: object, direction: str, path: Path) -> tuple[RuntimeTensor, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"TCIM Model.{direction} must be a non-empty list: {path}")
    tensors: list[RuntimeTensor] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"TCIM Model.{direction}[{index}] must be an object: {path}")
        name = item.get("name")
        shape = item.get("shape")
        dtype = item.get("dtype")
        if not isinstance(name, str) or not name:
            raise ValueError(f"TCIM Model.{direction}[{index}] requires a non-empty name: {path}")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(dimension) is not int or dimension < 1 for dimension in shape)
        ):
            raise ValueError(f"TCIM tensor {name!r} has invalid shape {shape!r}: {path}")
        tensors.append(
            RuntimeTensor(
                name=name,
                index=index,
                dtype=_tcim_dtype(dtype, name, path),
                shape=tuple(shape),
            )
        )
    return tuple(tensors)


def _tcim_dtype(value: object, name: str, path: Path) -> str:
    if not isinstance(value, dict):
        raise ValueError(f"TCIM tensor {name!r} has invalid dtype {value!r}: {path}")
    code = str(value.get("code", "")).lower()
    bits = value.get("bits")
    if code in {"float", "fp"} and bits in {16, 32, 64}:
        return f"float{bits}"
    if code in {"int", "uint"} and bits in {8, 16, 32, 64}:
        return f"{code}{bits}"
    if code == "bool" and bits in {1, 8}:
        return "bool"
    raise ValueError(f"TCIM tensor {name!r} has unsupported dtype code={code!r}, bits={bits!r}: {path}")


def _binding(
    tensor: RuntimeTensor,
    semantics: Mapping[str, str],
    image_layouts: Mapping[str, str],
    direction: str,
) -> TensorBinding:
    try:
        semantic = semantics[tensor.name]
    except KeyError as exc:
        raise ValueError(f"No semantic mapping for runtime {direction} tensor {tensor.name!r}") from exc
    is_image = (
        semantic == "observation.image"
        or semantic.startswith("observation.image.")
        or semantic.startswith("observation.images.")
    )
    declared_layout = image_layouts.get(semantic)
    runtime_layout = tensor.layout
    if declared_layout is not None and runtime_layout is not None and declared_layout != runtime_layout:
        raise ValueError(
            f"Image semantic {semantic!r} declares layout {declared_layout}, but runtime ABI reports {runtime_layout}"
        )
    layout = runtime_layout or declared_layout
    if is_image and layout is None:
        raise ValueError(f"Image semantic {semantic!r} requires an explicit runtime layout")
    if not is_image and layout is not None:
        raise ValueError(f"Non-image semantic {semantic!r} cannot declare an image layout")
    return TensorBinding(
        semantic=semantic,
        runtime_name=tensor.name,
        index=tensor.index,
        dtype=tensor.dtype,
        shape=tensor.shape,
        layout=layout,
    )


def _load_existing_manifest(path: Path) -> InferenceManifest | None:
    if not path.is_file():
        return None
    try:
        value = load_json_strict(path)
        validate_manifest_schema(value, str(path))
        return InferenceManifest.model_validate_json(json.dumps(value, ensure_ascii=False))
    except ValueError as exc:
        raise ValueError(f"Existing unified manifest is invalid and cannot be updated: {path}: {exc}") from exc
