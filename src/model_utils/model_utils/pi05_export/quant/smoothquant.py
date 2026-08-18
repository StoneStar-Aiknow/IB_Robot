#!/usr/bin/env python
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Licensed under the Mulan PSL v2.
"""Prepare matching donor/NPU ONNX graphs for SmoothQuant W8A8 calibration."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper

from model_utils.pi05_export.quant import w8a8_common as common

_FLOAT_DTYPES = {TensorProto.FLOAT, TensorProto.FLOAT16}


@dataclass(frozen=True)
class SmoothQuantResult:
    """Paths and stable summary of a prepared SmoothQuant graph pair."""

    donor_path: Path
    npu_path: Path
    plan_path: Path
    plan_digest: str
    selected_name_hash: str
    group_count: int
    node_count: int
    scale_min: float
    scale_max: float

    @property
    def summary(self) -> dict[str, int | float | str]:
        return {
            "plan_digest": self.plan_digest,
            "selected_name_hash": self.selected_name_hash,
            "group_count": self.group_count,
            "node_count": self.node_count,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
        }


@dataclass(frozen=True)
class OutputComparison:
    """Numerical comparison for one output and one calibration sample."""

    sample_index: int
    output_name: str
    shape: tuple[int, ...]
    finite: bool
    max_abs: float
    mean_l1: float
    cosine: float
    allclose: bool


@dataclass(frozen=True)
class SmoothQuantVerification:
    """Successful donor equivalence report."""

    sample_count: int
    comparisons: tuple[OutputComparison, ...]


@dataclass(frozen=True)
class _NodeSpec:
    name: str
    op_type: str
    activation_name: str
    weight_name: str
    input_axis: int
    weight_shape: tuple[int, int]
    effective_shape: tuple[int, int]
    weight_dtype: int


@dataclass(frozen=True)
class _GroupSpec:
    donor_activation: str
    npu_activation: str
    node_names: tuple[str, ...]
    channels: int
    donor_dtype: int
    npu_dtype: int


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _validate_parameters(alpha: float, epsilon: float, selected_names: list[str]) -> None:
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"SmoothQuant alpha must be finite and in [0, 1], got {alpha!r}")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"SmoothQuant epsilon must be finite and positive, got {epsilon!r}")
    if not selected_names:
        raise ValueError("SmoothQuant requires at least one selected node")
    if any(not name for name in selected_names):
        raise ValueError("SmoothQuant selected node names must be non-empty")
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("SmoothQuant selected node names must be unique")


def _load_topology(path: Path):  # noqa: ANN202
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {path}")
    return onnx.load(str(path), load_external_data=False)


def _load_with_external_data(path: Path):  # noqa: ANN202
    model = onnx.load(str(path), load_external_data=True)
    if common._has_unloaded_external_data(model):
        raise ValueError(f"ONNX model has unresolved external data: {path}")
    return model


def _initializer_index(model) -> dict[str, object]:  # noqa: ANN001
    initializers: dict[str, object] = {}
    for initializer in model.graph.initializer:
        if initializer.name in initializers:
            raise ValueError(f"Duplicate ONNX initializer name: {initializer.name!r}")
        initializers[initializer.name] = initializer
    return initializers


def _value_dtypes(model) -> dict[str, int]:  # noqa: ANN001
    dtypes: dict[str, int] = {}
    values = (*model.graph.input, *model.graph.value_info, *model.graph.output)
    for value in values:
        if not value.type.HasField("tensor_type"):
            continue
        dtype = int(value.type.tensor_type.elem_type)
        if dtype and value.name in dtypes and dtypes[value.name] != dtype:
            raise ValueError(f"Conflicting ONNX dtypes for tensor {value.name!r}")
        if dtype:
            dtypes[value.name] = dtype
    return dtypes


def _int_attribute(node, name: str, default: int) -> int:  # noqa: ANN001
    attributes = [attribute for attribute in node.attribute if attribute.name == name]
    if not attributes:
        return default
    if len(attributes) != 1 or attributes[0].type != AttributeProto.INT:
        raise ValueError(f"Node {node.name!r} has malformed {name} attribute")
    return int(attributes[0].i)


def _inspect_graph(model, selected_names: list[str]) -> dict[str, _NodeSpec]:  # noqa: ANN001
    selected = set(selected_names)
    matches: dict[str, list[object]] = {name: [] for name in selected_names}
    for node in model.graph.node:
        if node.name in selected:
            matches[node.name].append(node)
    for name, nodes in matches.items():
        if not nodes:
            raise ValueError(f"Selected node {name!r} is missing from ONNX graph")
        if len(nodes) != 1:
            raise ValueError(f"Selected node name {name!r} is not unique in ONNX graph")

    initializers = _initializer_index(model)
    graph_inputs = {value.name for value in model.graph.input}
    value_dtypes = _value_dtypes(model)
    specs: dict[str, _NodeSpec] = {}
    for name in selected_names:
        node = matches[name][0]
        if node.op_type not in {"MatMul", "Gemm"}:
            raise ValueError(f"Selected node {name!r} must be MatMul or Gemm, got {node.op_type!r}")
        valid_input_counts = {2} if node.op_type == "MatMul" else {2, 3}
        if len(node.input) not in valid_input_counts:
            raise ValueError(f"Selected {node.op_type} node {name!r} has malformed inputs")
        if not node.input[0] or not node.input[1]:
            raise ValueError(f"Selected node {name!r} has an empty activation or weight input")
        activation_name, weight_name = node.input[:2]
        if activation_name in initializers:
            raise ValueError(f"Selected node {name!r} does not have a runtime activation as input 0")
        if weight_name in graph_inputs:
            raise ValueError(f"Selected node {name!r} uses an overridable initializer")
        weight = initializers.get(weight_name)
        if weight is None:
            raise ValueError(f"Selected node {name!r} weight {weight_name!r} is not an initializer")
        shape = tuple(int(dim) for dim in weight.dims)
        if len(shape) != 2 or any(dim <= 0 for dim in shape):
            raise ValueError(f"Selected node {name!r} weight must be a non-empty rank-2 initializer")
        if weight.data_type not in _FLOAT_DTYPES:
            raise ValueError(f"Selected node {name!r} weight must be FP16 or FP32")

        input_axis = 0
        if node.op_type == "Gemm":
            trans_a = _int_attribute(node, "transA", 0)
            trans_b = _int_attribute(node, "transB", 0)
            if trans_a != 0:
                raise ValueError(f"Selected Gemm node {name!r} has unsupported transA={trans_a}")
            if trans_b not in {0, 1}:
                raise ValueError(f"Selected Gemm node {name!r} has malformed transB={trans_b}")
            input_axis = trans_b
        effective_shape = shape if input_axis == 0 else (shape[1], shape[0])
        activation_dtype = value_dtypes.get(activation_name)
        if activation_dtype is not None and activation_dtype != weight.data_type:
            raise ValueError(
                f"Selected node {name!r} activation/weight dtype mismatch: {activation_dtype} != {weight.data_type}"
            )
        specs[name] = _NodeSpec(
            name=name,
            op_type=node.op_type,
            activation_name=activation_name,
            weight_name=weight_name,
            input_axis=input_axis,
            weight_shape=shape,
            effective_shape=effective_shape,
            weight_dtype=int(weight.data_type),
        )
    return specs


def _build_groups(
    donor_model,
    donor_specs: dict[str, _NodeSpec],
    npu_specs: dict[str, _NodeSpec],
) -> list[_GroupSpec]:  # noqa: ANN001
    for name, donor in donor_specs.items():
        npu = npu_specs[name]
        if donor.op_type != npu.op_type:
            raise ValueError(f"Selected node {name!r} op mismatch between donor/NPU: {donor.op_type} != {npu.op_type}")
        if donor.effective_shape != npu.effective_shape:
            raise ValueError(
                f"Selected node {name!r} effective weight shape mismatch between donor/NPU: "
                f"{donor.effective_shape} != {npu.effective_shape}"
            )

    grouped_names: dict[str, list[str]] = {}
    for node in donor_model.graph.node:
        if node.name in donor_specs:
            grouped_names.setdefault(donor_specs[node.name].activation_name, []).append(node.name)

    groups: list[_GroupSpec] = []
    for donor_activation, node_names in grouped_names.items():
        donor_group = [donor_specs[name] for name in node_names]
        npu_group = [npu_specs[name] for name in node_names]
        channels = {spec.effective_shape[0] for spec in donor_group}
        npu_activations = {spec.activation_name for spec in npu_group}
        donor_dtypes = {spec.weight_dtype for spec in donor_group}
        npu_dtypes = {spec.weight_dtype for spec in npu_group}
        if len(channels) != 1:
            raise ValueError(f"Selected consumers of activation {donor_activation!r} have different input widths")
        if len(npu_activations) != 1:
            raise ValueError(
                f"NPU nodes matching donor activation group {donor_activation!r} do not share one activation"
            )
        if len(donor_dtypes) != 1 or len(npu_dtypes) != 1:
            raise ValueError(f"Selected consumers of activation {donor_activation!r} have mixed weight dtypes")
        for graph_name, graph_group in (("donor", donor_group), ("NPU", npu_group)):
            axes_by_weight: dict[str, set[int]] = {}
            for spec in graph_group:
                axes_by_weight.setdefault(spec.weight_name, set()).add(spec.input_axis)
            if any(len(axes) != 1 for axes in axes_by_weight.values()):
                raise ValueError(f"A shared {graph_name} weight has incompatible orientations in one activation group")
        groups.append(
            _GroupSpec(
                donor_activation=donor_activation,
                npu_activation=npu_activations.pop(),
                node_names=tuple(node_names),
                channels=channels.pop(),
                donor_dtype=donor_dtypes.pop(),
                npu_dtype=npu_dtypes.pop(),
            )
        )
    return groups


def _validate_calib_data(calib_data: list[list[np.ndarray]], input_names: list[str]) -> None:
    if not calib_data:
        raise ValueError("SmoothQuant requires non-empty calibration data")
    for index, sample in enumerate(calib_data):
        if len(sample) != len(input_names):
            raise ValueError(
                f"Calibration sample {index} has {len(sample)} arrays, expected {len(input_names)} for {input_names}"
            )
        if any(not isinstance(array, np.ndarray) for array in sample):
            raise TypeError(f"Calibration sample {index} must contain only numpy arrays")


def _weight_absmax(model, groups: list[_GroupSpec], specs: dict[str, _NodeSpec]) -> list[np.ndarray]:  # noqa: ANN001
    initializers = _initializer_index(model)
    maxima: list[np.ndarray] = []
    for group in groups:
        group_max = np.zeros(group.channels, dtype=np.float64)
        for name in group.node_names:
            spec = specs[name]
            weight = np.asarray(numpy_helper.to_array(initializers[spec.weight_name]), dtype=np.float64)
            if weight.shape != spec.weight_shape or not np.isfinite(weight).all():
                raise ValueError(f"Selected node {name!r} has malformed or non-finite weights")
            axes = tuple(axis for axis in range(weight.ndim) if axis != spec.input_axis)
            group_max = np.maximum(group_max, np.max(np.abs(weight), axis=axes))
        maxima.append(group_max)
    return maxima


def _collect_statistics(
    donor_path: Path,
    donor_specs: dict[str, _NodeSpec],
    groups: list[_GroupSpec],
    input_names: list[str],
    calib_data: list[list[np.ndarray]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    model = _load_with_external_data(donor_path)
    if _inspect_graph(model, list(donor_specs)) != donor_specs:
        raise ValueError("Donor graph changed while preparing SmoothQuant")
    weight_maxima = _weight_absmax(model, groups, donor_specs)

    existing_outputs = {output.name for output in model.graph.output}
    for group in groups:
        if group.donor_activation not in existing_outputs:
            model.graph.output.append(helper.make_tensor_value_info(group.donor_activation, group.donor_dtype, None))

    activation_maxima = [np.zeros(group.channels, dtype=np.float64) for group in groups]
    with tempfile.TemporaryDirectory(prefix="pi05_smoothquant_") as temp_dir:
        instrumented_path = Path(temp_dir) / "instrumented.onnx"
        common.save_onnx_external(model, instrumented_path)
        del model

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("SmoothQuant calibration requires ONNX Runtime") from exc
        if "CPUExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("SmoothQuant calibration requires the ONNX Runtime CPUExecutionProvider")
        session = ort.InferenceSession(str(instrumented_path), providers=["CPUExecutionProvider"])
        session_inputs = [value.name for value in session.get_inputs()]
        if session_inputs != input_names:
            raise ValueError(f"Instrumented donor input order changed: expected {input_names}, got {session_inputs}")
        activation_names = [group.donor_activation for group in groups]
        for sample_index, sample in enumerate(calib_data):
            feed = dict(zip(input_names, sample, strict=True))
            outputs = session.run(activation_names, feed)
            for group_index, (group, output) in enumerate(zip(groups, outputs, strict=True)):
                activation = np.asarray(output)
                if activation.ndim < 1 or activation.shape[-1] != group.channels:
                    raise ValueError(
                        f"Activation {group.donor_activation!r} in calibration sample {sample_index} "
                        f"has shape {activation.shape}, expected last dimension {group.channels}"
                    )
                if not np.isfinite(activation).all():
                    raise ValueError(
                        f"Activation {group.donor_activation!r} in calibration sample {sample_index} is non-finite"
                    )
                reduce_axes = tuple(range(activation.ndim - 1))
                sample_max = np.max(np.abs(activation.astype(np.float64)), axis=reduce_axes)
                activation_maxima[group_index] = np.maximum(activation_maxima[group_index], sample_max)
    return activation_maxima, weight_maxima


def _compute_scales(
    activation_maxima: list[np.ndarray],
    weight_maxima: list[np.ndarray],
    alpha: float,
    epsilon: float,
) -> list[np.ndarray]:
    scales: list[np.ndarray] = []
    for activation, weight in zip(activation_maxima, weight_maxima, strict=True):
        scale = np.power(np.maximum(activation, epsilon), alpha) / np.power(np.maximum(weight, epsilon), 1.0 - alpha)
        scale[(activation == 0.0) & (weight == 0.0)] = 1.0
        if not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise ValueError("SmoothQuant produced a non-finite or non-positive scale")
        scales.append(scale)
    return scales


def _unique_name(base: str, used_names: set[str]) -> str:
    candidate = base
    suffix = 1
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _all_graph_names(model) -> set[str]:  # noqa: ANN001
    names = {value.name for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
    names.update(initializer.name for initializer in model.graph.initializer)
    for node in model.graph.node:
        names.update(name for name in (*node.input, *node.output) if name)
        if node.name:
            names.add(node.name)
    return names


def _scale_weight(initializer, scale: np.ndarray, input_axis: int) -> None:  # noqa: ANN001
    array = numpy_helper.to_array(initializer)
    target_dtype = np.float16 if initializer.data_type == TensorProto.FLOAT16 else np.float32
    broadcast_shape = [1] * array.ndim
    broadcast_shape[input_axis] = scale.size
    scaled = array.astype(np.float64) * scale.reshape(broadcast_shape)
    scaled = scaled.astype(target_dtype)
    if not np.isfinite(scaled).all():
        raise ValueError(f"SmoothQuant overflowed weight initializer {initializer.name!r}")
    name = initializer.name
    initializer.CopyFrom(numpy_helper.from_array(scaled, name=name))


def _rewrite_model(
    source_path: Path,
    expected_specs: dict[str, _NodeSpec],
    groups: list[_GroupSpec],
    scales: list[np.ndarray],
    *,
    npu: bool,
):  # noqa: ANN202
    model = _load_with_external_data(source_path)
    specs = _inspect_graph(model, list(expected_specs))
    if specs != expected_specs:
        raise ValueError(f"{'NPU' if npu else 'Donor'} graph changed while preparing SmoothQuant")

    node_positions = {node.name: index for index, node in enumerate(model.graph.node) if node.name in specs}
    initializers = _initializer_index(model)
    graph_inputs = {value.name for value in model.graph.input}
    graph_outputs = {value.name for value in model.graph.output}
    used_names = _all_graph_names(model)
    insertions: dict[int, list[object]] = {}

    for group_index, (group, scale) in enumerate(zip(groups, scales, strict=True)):
        activation_name = group.npu_activation if npu else group.donor_activation
        dtype = group.npu_dtype if npu else group.donor_dtype
        numpy_dtype = np.float16 if dtype == TensorProto.FLOAT16 else np.float32
        inverse = np.reciprocal(scale).astype(numpy_dtype)
        if not np.isfinite(inverse).all() or np.any(inverse <= 0.0):
            raise ValueError(f"SmoothQuant inverse scale for activation {activation_name!r} overflows {numpy_dtype}")

        base = f"__smoothquant_group_{group_index:04d}"
        inverse_name = _unique_name(f"{base}_inverse_scale", used_names)
        mul_output = _unique_name(f"{base}_activation", used_names)
        mul_name = _unique_name(f"{base}_Mul", used_names)
        model.graph.initializer.append(numpy_helper.from_array(inverse, name=inverse_name))

        group_positions = {node_positions[name] for name in group.node_names}
        weight_groups: dict[str, list[int]] = {}
        for name in group.node_names:
            position = node_positions[name]
            weight_groups.setdefault(specs[name].weight_name, []).append(position)
        for weight_name, positions in weight_groups.items():
            input_axis = specs[model.graph.node[positions[0]].name].input_axis
            own_references = {(position, 1) for position in positions}
            all_references = {
                (position, input_index)
                for position, node in enumerate(model.graph.node)
                for input_index, input_name in enumerate(node.input)
                if input_name == weight_name
            }
            initializer = initializers[weight_name]
            if all_references != own_references or weight_name in graph_inputs or weight_name in graph_outputs:
                clone_name = _unique_name(f"{base}_{weight_name}_weight", used_names)
                clone = TensorProto()
                clone.CopyFrom(initializer)
                clone.name = clone_name
                model.graph.initializer.append(clone)
                initializer = model.graph.initializer[-1]
                initializers[clone_name] = initializer
                for position in positions:
                    model.graph.node[position].input[1] = clone_name
            _scale_weight(initializer, scale, input_axis)

        for position in group_positions:
            model.graph.node[position].input[0] = mul_output
        mul = helper.make_node("Mul", [activation_name, inverse_name], [mul_output], name=mul_name)
        insertions.setdefault(min(group_positions), []).append(mul)

    original_nodes = list(model.graph.node)
    rewritten_nodes = []
    for position, node in enumerate(original_nodes):
        rewritten_nodes.extend(insertions.get(position, ()))
        rewritten_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    return model


def _external_source_paths(model_path: Path, topology) -> set[Path]:  # noqa: ANN001
    paths = {model_path.resolve()}
    for tensor in common._iter_tensors(topology):
        metadata = {entry.key: entry.value for entry in tensor.external_data}
        location = metadata.get("location")
        if location:
            paths.add((model_path.parent / location).resolve())
    return paths


def _output_paths(output_dir: Path, output_prefix: str) -> tuple[Path, Path, Path]:
    if (
        not output_prefix
        or output_prefix.startswith(".")
        or Path(output_prefix).name != output_prefix
        or output_prefix in {".", ".."}
    ):
        raise ValueError("SmoothQuant output_prefix must be a non-empty file-name prefix")
    output_dir = Path(output_dir)
    return (
        output_dir / f"{output_prefix}.donor.onnx",
        output_dir / f"{output_prefix}.npu.onnx",
        output_dir / f"{output_prefix}.plan.json",
    )


def prepare_smoothquant_pair(
    donor_path: Path,
    npu_path: Path,
    output_dir: Path,
    calib_data: list[list[np.ndarray]],
    selected_names: list[str],
    alpha: float,
    epsilon: float,
    *,
    output_prefix: str = "smoothquant",
) -> SmoothQuantResult:
    """Create matching SmoothQuant donor/NPU copies and a deterministic plan.

    Calibration samples are positional lists in donor graph-input order, matching
    the existing PI0.5 quantizer calibration builders. Only selected consumers
    are rerouted; source ONNX and external-data files are never modified.
    """
    _validate_parameters(alpha, epsilon, selected_names)
    donor_path = Path(donor_path)
    npu_path = Path(npu_path)
    donor_topology = _load_topology(donor_path)
    npu_topology = _load_topology(npu_path)
    donor_specs = _inspect_graph(donor_topology, selected_names)
    npu_specs = _inspect_graph(npu_topology, selected_names)
    groups = _build_groups(donor_topology, donor_specs, npu_specs)
    input_names = common.proto_input_names(donor_topology)
    _validate_calib_data(calib_data, input_names)

    donor_output, npu_output, plan_path = _output_paths(output_dir, output_prefix)
    protected_paths = _external_source_paths(donor_path, donor_topology)
    protected_paths.update(_external_source_paths(npu_path, npu_topology))
    generated_paths = {
        donor_output.resolve(),
        donor_output.with_name(donor_output.name + ".data").resolve(),
        npu_output.resolve(),
        npu_output.with_name(npu_output.name + ".data").resolve(),
        plan_path.resolve(),
    }
    collision = protected_paths & generated_paths
    if collision:
        raise ValueError(f"SmoothQuant output would overwrite source data: {sorted(map(str, collision))}")

    activation_maxima, weight_maxima = _collect_statistics(
        donor_path,
        donor_specs,
        groups,
        input_names,
        calib_data,
    )
    scales = _compute_scales(activation_maxima, weight_maxima, alpha, epsilon)
    donor_output.parent.mkdir(parents=True, exist_ok=True)
    donor_model = _rewrite_model(donor_path, donor_specs, groups, scales, npu=False)
    common.save_onnx_external(donor_model, donor_output)
    del donor_model
    npu_model = _rewrite_model(npu_path, npu_specs, groups, scales, npu=True)
    common.save_onnx_external(npu_model, npu_output)

    selected_name_hash = _hash_json(sorted(selected_names))
    group_plans = []
    for group, scale in zip(groups, scales, strict=True):
        group_plans.append(
            {
                "donor_activation": group.donor_activation,
                "npu_activation": group.npu_activation,
                "nodes": list(group.node_names),
                "channels": group.channels,
                "scale_min": float(np.min(scale)),
                "scale_max": float(np.max(scale)),
                "scale_sha256": hashlib.sha256(np.asarray(scale, dtype="<f8").tobytes()).hexdigest(),
            }
        )
    scale_min = min(group["scale_min"] for group in group_plans)
    scale_max = max(group["scale_max"] for group in group_plans)
    plan = {
        "version": 1,
        "alpha": float(alpha),
        "epsilon": float(epsilon),
        "selected_name_hash": selected_name_hash,
        "group_count": len(groups),
        "node_count": len(selected_names),
        "scale_min": scale_min,
        "scale_max": scale_max,
        "groups": group_plans,
    }
    plan_digest = _hash_json(plan)
    plan["plan_digest"] = plan_digest
    plan_path.write_text(json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    return SmoothQuantResult(
        donor_path=donor_output,
        npu_path=npu_output,
        plan_path=plan_path,
        plan_digest=plan_digest,
        selected_name_hash=selected_name_hash,
        group_count=len(groups),
        node_count=len(selected_names),
        scale_min=scale_min,
        scale_max=scale_max,
    )


def verify_smoothquant_outputs(
    original_donor_path: Path,
    smoothed_donor_path: Path,
    calib_data: list[list[np.ndarray]],
    *,
    rtol: float = 2e-3,
    atol: float = 2e-3,
) -> SmoothQuantVerification:
    """Verify original/smoothed donor outputs over every calibration sample."""
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0.0 or atol < 0.0:
        raise ValueError("SmoothQuant verification tolerances must be finite and non-negative")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("SmoothQuant verification requires ONNX Runtime") from exc
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("SmoothQuant verification requires the ONNX Runtime CPUExecutionProvider")

    original = ort.InferenceSession(str(original_donor_path), providers=["CPUExecutionProvider"])
    smoothed = ort.InferenceSession(str(smoothed_donor_path), providers=["CPUExecutionProvider"])
    input_names = [value.name for value in original.get_inputs()]
    smoothed_inputs = [value.name for value in smoothed.get_inputs()]
    if input_names != smoothed_inputs:
        raise ValueError(f"Original/smoothed donor inputs differ: {input_names} != {smoothed_inputs}")
    _validate_calib_data(calib_data, input_names)
    output_names = [value.name for value in original.get_outputs()]
    smoothed_outputs = [value.name for value in smoothed.get_outputs()]
    if output_names != smoothed_outputs:
        raise ValueError(f"Original/smoothed donor outputs differ: {output_names} != {smoothed_outputs}")

    comparisons: list[OutputComparison] = []
    failed = False
    for sample_index, sample in enumerate(calib_data):
        feed = dict(zip(input_names, sample, strict=True))
        expected_values = original.run(output_names, feed)
        actual_values = smoothed.run(output_names, feed)
        for output_name, expected, actual in zip(output_names, expected_values, actual_values, strict=True):
            expected_array = np.asarray(expected)
            actual_array = np.asarray(actual)
            finite = bool(np.isfinite(expected_array).all() and np.isfinite(actual_array).all())
            same_shape = expected_array.shape == actual_array.shape
            close = bool(finite and same_shape and np.allclose(expected_array, actual_array, rtol=rtol, atol=atol))
            if finite and same_shape and expected_array.size:
                expected_flat = expected_array.astype(np.float64).ravel()
                actual_flat = actual_array.astype(np.float64).ravel()
                difference = np.abs(expected_flat - actual_flat)
                max_abs = float(np.max(difference))
                mean_l1 = float(np.mean(difference))
                denominator = float(np.linalg.norm(expected_flat) * np.linalg.norm(actual_flat))
                cosine = (
                    float(np.dot(expected_flat, actual_flat) / denominator)
                    if denominator
                    else 1.0
                    if not np.any(expected_flat) and not np.any(actual_flat)
                    else 0.0
                )
            elif finite and same_shape:
                max_abs = mean_l1 = 0.0
                cosine = 1.0
            else:
                max_abs = mean_l1 = math.inf
                cosine = math.nan
            comparisons.append(
                OutputComparison(
                    sample_index=sample_index,
                    output_name=output_name,
                    shape=tuple(int(dim) for dim in actual_array.shape),
                    finite=finite,
                    max_abs=max_abs,
                    mean_l1=mean_l1,
                    cosine=cosine,
                    allclose=close,
                )
            )
            failed |= not close

    report = SmoothQuantVerification(sample_count=len(calib_data), comparisons=tuple(comparisons))
    if failed:
        raise AssertionError(f"SmoothQuant donor equivalence failed (rtol={rtol}, atol={atol}): {report}")
    return report
