#!/usr/bin/env python
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Licensed under the Mulan PSL v2.
"""Shared msModelSlim W8A8 plumbing for the PI05 VLM and Action Expert.

This module holds everything that is **identical** between the two PI05
quantization entry points (``quantize_vlm.py`` / ``quantize_ae.py``):

* ONNX graph inspection (quantizable-node inventory, fp16-exclusion selection).
* The runtime monkey-patches msModelSlim needs on this board (onnx.mapping
  shim, >2 GB external-data save, calibration DataReader fix, opset-17 fix,
  optional amp-rollback real-calib fix).
* The Resize-empty-input pre-pass.
* The W8A8 driver :func:`run_msmodelslim_w8a8`.
* The AscendDequant fp16-output pin (ATC kernel selection).
* Route A: transplanting int8 Linears onto an NPU-op graph.

Model-specific code (calibration-data construction, default fp16-exclusion
regexes) lives in the thin per-model scripts and is passed in here.

NOTE: msmodelslim / torch_npu exist ONLY on the Ascend dev board, never in the
local dev env — every patch is installed at runtime, before importing
msmodelslim, and never edits the installed package or the venv.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

LOGGER = logging.getLogger("pi05_w8a8")

# ONNX op types that msModelSlim W8A8 can quantize for these graphs.
_QUANTIZABLE_OPS: tuple[str, ...] = ("MatMul", "Gemm", "Conv")


# ---------------------------------------------------------------------------
# ONNX graph inspection
# ---------------------------------------------------------------------------


def load_onnx(onnx_path: Path):
    """Load an ONNX graph topology only (no external weight data)."""
    import onnx

    LOGGER.info("Loading ONNX graph %s …", onnx_path)
    return onnx.load(str(onnx_path), load_external_data=False)


def collect_quantizable_nodes(model_proto) -> list[tuple[str, str]]:
    """Return ``[(node_name, op_type), …]`` for every quantizable node."""
    nodes: list[tuple[str, str]] = []
    for node in model_proto.graph.node:
        if node.op_type in _QUANTIZABLE_OPS:
            name = node.name or f"<unnamed {node.op_type}>"
            nodes.append((name, node.op_type))
    return nodes


def node_index(name: str) -> int | None:
    """Trailing integer of an exporter node name, e.g. ``node_MatMul_53`` → 53."""
    import re

    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else None


def build_disable_names(
    quantizable: list[tuple[str, str]],
    disable_regexes: list[str],
    disable_convs: bool,
    disable_index_below: int | None,
) -> list[str]:
    """Select node names to keep in fp16 (excluded from quantization)."""
    import re

    compiled = [re.compile(rx, re.IGNORECASE) for rx in disable_regexes]
    disabled: list[str] = []
    regex_hits = 0
    for name, op_type in quantizable:
        if disable_convs and op_type == "Conv":
            disabled.append(name)
            continue
        if any(rx.search(name) for rx in compiled):
            disabled.append(name)
            regex_hits += 1
            continue
        if disable_index_below is not None:
            idx = node_index(name)
            if idx is not None and idx < disable_index_below:
                disabled.append(name)
    if compiled and regex_hits == 0:
        LOGGER.warning(
            "None of the --disable-regex patterns matched any node name. This export "
            "uses anonymous positional names (e.g. node_MatMul_53) — use "
            "--disable-index-below or --amp-num for accuracy protection instead."
        )
    # De-dup while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for n in disabled:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def restrict_quantizable_nodes(
    quantizable: list[tuple[str, str]],
    disable_names: list[str],
    quantize_regexes: list[str] | None,
    *,
    expected_regex_matches: list[int] | None = None,
    expected_selected_nodes: int | None = None,
) -> list[str]:
    """Keep non-matching nodes in fp16 when an explicit quantization allowlist is supplied."""
    return list(
        select_quantizable_nodes(
            quantizable,
            disable_names,
            quantize_regexes,
            expected_regex_matches=expected_regex_matches,
            expected_selected_nodes=expected_selected_nodes,
        ).disabled_names
    )


@dataclass(frozen=True)
class QuantizationSelection:
    disabled_names: tuple[str, ...]
    selected_names: tuple[str, ...]
    regex_matches: tuple[int, ...]


def select_quantizable_nodes(
    quantizable: list[tuple[str, str]],
    disable_names: list[str],
    quantize_regexes: list[str] | None,
    *,
    expected_regex_matches: list[int] | None = None,
    expected_selected_nodes: int | None = None,
) -> QuantizationSelection:
    """Apply an allowlist and optionally enforce its per-regex and total match contract."""
    disabled = set(disable_names)
    if not quantize_regexes:
        selected = tuple(name for name, _ in quantizable if name not in disabled)
        if expected_regex_matches:
            raise ValueError("Expected per-regex match counts require --quantize-regex")
        if expected_selected_nodes is not None and len(selected) != expected_selected_nodes:
            raise ValueError(
                f"Quantization profile selected {len(selected)} node(s), expected {expected_selected_nodes}"
            )
        return QuantizationSelection(tuple(disable_names), selected, ())
    import re

    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in quantize_regexes]
    if expected_regex_matches is not None and len(expected_regex_matches) != len(compiled):
        raise ValueError("--quantize-regex-expected must contain one count per --quantize-regex")
    eligible = [(name, op_type) for name, op_type in quantizable if name not in disabled]
    regex_matches = tuple(sum(bool(pattern.search(name)) for name, _ in eligible) for pattern in compiled)
    allowed = {name for name, _ in eligible if any(pattern.search(name) for pattern in compiled)}
    if not allowed:
        raise ValueError("None of the --quantize-regex patterns matched a quantizable node")
    if expected_regex_matches is not None:
        mismatches = [
            f"[{index}] matched {actual}, expected {expected}"
            for index, (actual, expected) in enumerate(zip(regex_matches, expected_regex_matches, strict=True))
            if actual != expected
        ]
        if mismatches:
            raise ValueError("Quantization profile regex match mismatch: " + "; ".join(mismatches))
    if expected_selected_nodes is not None and len(allowed) != expected_selected_nodes:
        raise ValueError(f"Quantization profile selected {len(allowed)} node(s), expected {expected_selected_nodes}")
    disabled.update(name for name, _ in quantizable if name not in allowed)
    LOGGER.info("Quantization allowlist matched %d node(s).", len(allowed - disabled))
    return QuantizationSelection(
        tuple(name for name, _ in quantizable if name in disabled),
        tuple(name for name, _ in quantizable if name in allowed and name not in disabled),
        regex_matches,
    )


def validate_fused_geglu_route(donor, npu):  # noqa: ANN001
    """Require each NPU fused GeGLU MatMul to have an identical donor target."""
    npu_producer = {output: node for node in npu.graph.node for output in node.output}
    donor_names = [node.name for node in donor.graph.node if node.name]
    if len(donor_names) != len(set(donor_names)):
        raise RuntimeError("Fused GeGLU donor contains duplicate node names")
    donor_by_name = {node.name: node for node in donor.graph.node}
    donor_init = {initializer.name: initializer for initializer in donor.graph.initializer}
    npu_init = {initializer.name: initializer for initializer in npu.graph.initializer}
    fused_targets = []
    for geglu in npu.graph.node:
        if geglu.op_type != "NPUGeglu" or not geglu.input:
            continue
        producer = npu_producer.get(geglu.input[0])
        if producer is None or producer.op_type not in {"MatMul", "Gemm"}:
            raise RuntimeError(f"NPUGeglu {geglu.name!r} is not fed by a MatMul/Gemm")
        donor_node = donor_by_name.get(producer.name)
        if donor_node is None or donor_node.op_type != producer.op_type:
            raise RuntimeError(f"Fused GeGLU donor is missing {producer.op_type} {producer.name!r}")
        if len(donor_node.input) < 2 or len(producer.input) < 2:
            raise RuntimeError(f"Fused GeGLU MatMul {producer.name!r} has no weight input")
        donor_weight = donor_init.get(donor_node.input[1])
        npu_weight = npu_init.get(producer.input[1])
        if donor_weight is None or npu_weight is None or list(donor_weight.dims) != list(npu_weight.dims):
            raise RuntimeError(f"Fused GeGLU weight shape mismatch for {producer.name!r}")
        fused_targets.append(producer.name)
    if not fused_targets:
        raise RuntimeError("NPU graph has no NPUGeglu nodes to quantize")
    LOGGER.info("Validated %d fused GeGLU MatMul donor target(s).", len(fused_targets))
    return fused_targets


def validate_npu_geglu_deployment(npu, expected: int | None = None):  # noqa: ANN001
    """Require every NPUGeglu site in the deployment graph to consume a weight MatMul/Gemm."""
    import re

    producer = {output: node for node in npu.graph.node for output in node.output}
    initializers = {initializer.name for initializer in npu.graph.initializer}
    separate_projection = re.compile(r"/mlp/(?:gate_proj|up_proj)/MatMul$")
    mixed = [node.name for node in npu.graph.node if separate_projection.search(node.name)]
    if mixed:
        raise RuntimeError(f"NPU deployment graph mixes NPUGeglu with separate gate/up projections: {mixed[:5]}")
    targets = []
    for geglu in npu.graph.node:
        if geglu.op_type != "NPUGeglu":
            continue
        parent = producer.get(geglu.input[0]) if geglu.input else None
        if (
            parent is None
            or parent.op_type not in {"MatMul", "Gemm"}
            or len(parent.input) < 2
            or parent.input[1] not in initializers
        ):
            raise RuntimeError(f"NPUGeglu {geglu.name!r} is not fed by a MatMul/Gemm")
        targets.append(parent.name)
    if not targets:
        raise RuntimeError("NPU deployment graph has no NPUGeglu nodes")
    if expected is not None and len(targets) != expected:
        raise RuntimeError(f"NPU deployment graph has {len(targets)} NPUGeglu nodes, expected {expected}")
    LOGGER.info("Validated %d NPU GeGLU deployment site(s).", len(targets))
    return targets


def ordered_input_names(onnx_path: Path) -> list[str]:
    """Graph input order (excluding initializers), read straight from the proto.

    We deliberately avoid ``onnxruntime.InferenceSession`` here: the exported
    graph may contain NPU-affine custom ops (e.g. ``NPUFastGelu``,
    ``NPURmsNorm``) that ORT cannot load, which would raise INVALID_GRAPH. The
    graph topology is all we need, and ``onnx.load`` parses it without running.
    """
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    initializer_names = {init.name for init in model.graph.initializer}
    return [inp.name for inp in model.graph.input if inp.name not in initializer_names]


def onnx_input_last_dim(onnx_path: Path, input_name: str) -> int:
    """Return the trailing static dim of a named graph input (e.g. prefix S)."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    for inp in model.graph.input:
        if inp.name == input_name:
            dims = inp.type.tensor_type.shape.dim
            last = dims[-1]
            if not last.HasField("dim_value"):
                raise ValueError(
                    f"ONNX input {input_name!r} has a dynamic trailing dim; cannot derive the trailing size."
                )
            return int(last.dim_value)
    raise KeyError(f"ONNX input {input_name!r} not found.")


def onnx_input_dtypes(onnx_path: Path) -> dict[str, np.dtype]:
    """Map each graph input name to its declared numpy dtype.

    Calibration arrays must be fed in the dtype the graph declares (fp16 vs
    fp32, bool masks), otherwise ORT's augmented-calibration run rejects or
    mis-casts them. Returns ``{name: np.dtype}`` for non-initializer inputs.
    """
    import onnx
    from onnx import helper

    model = onnx.load(str(onnx_path), load_external_data=False)
    initializer_names = {init.name for init in model.graph.initializer}
    out: dict[str, np.dtype] = {}
    for inp in model.graph.input:
        if inp.name in initializer_names:
            continue
        elem = inp.type.tensor_type.elem_type
        out[inp.name] = np.dtype(helper.tensor_dtype_to_np_dtype(elem))
    return out


# ---------------------------------------------------------------------------
# msModelSlim runtime patches (version-mismatch / multi-GB safe)
# ---------------------------------------------------------------------------


def install_onnx_mapping_shim() -> None:
    """Recreate the legacy ``onnx.mapping`` module if the installed onnx removed it.

    ``onnx.mapping`` was deprecated in onnx 1.14 and removed in 1.16+, but
    msModelSlim's ``post_training_quant`` still does
    ``from onnx.mapping import STORAGE_TENSOR_TYPE_TO_FIELD`` (and friends).
    Rather than force-downgrade onnx in the user's venv (which our own
    ``onnx.load`` relies on), we synthesise an equivalent module from the
    modern ``onnx.helper`` mapping helpers and register it in ``sys.modules``.
    """
    import importlib
    import sys
    import types

    try:
        importlib.import_module("onnx.mapping")
        return  # Native module present — nothing to do.
    except ImportError:
        pass

    import numpy as _np
    import onnx
    from onnx import TensorProto, helper

    all_dtypes = list(helper.get_all_tensor_dtypes())

    def _safe(fn, dt):
        try:
            return fn(dt)
        except Exception:  # noqa: BLE001 — some dtypes (e.g. FLOAT8) lack a np dtype.
            return None

    tensor_type_to_np = {
        dt: np_dt for dt in all_dtypes if (np_dt := _safe(helper.tensor_dtype_to_np_dtype, dt)) is not None
    }
    tensor_type_to_storage = {
        dt: st for dt in all_dtypes if (st := _safe(helper.tensor_dtype_to_storage_tensor_dtype, dt)) is not None
    }
    storage_type_to_field = {
        st: field
        for st in set(tensor_type_to_storage.values())
        if (field := _safe(helper.tensor_dtype_to_field, st)) is not None
    }

    shim = types.ModuleType("onnx.mapping")
    shim.TENSOR_TYPE_TO_NP_TYPE = tensor_type_to_np
    shim.NP_TYPE_TO_TENSOR_TYPE = {_np.dtype(v): k for k, v in tensor_type_to_np.items()}
    shim.TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE = tensor_type_to_storage
    shim.STORAGE_TENSOR_TYPE_TO_FIELD = storage_type_to_field
    shim.TensorProto = TensorProto

    sys.modules["onnx.mapping"] = shim
    onnx.mapping = shim  # type: ignore[attr-defined]
    LOGGER.info(
        "Installed onnx.mapping compatibility shim (onnx %s removed the native module).",
        getattr(onnx, "__version__", "?"),
    )


# Models > ~1.9 GB cannot be serialised inline: protobuf caps a single message
# at 2 GB. Force external-data saving above this threshold.
_PROTOBUF_INLINE_LIMIT = 1_900_000_000


def _iter_tensors(proto):  # noqa: ANN001
    from google.protobuf.descriptor import FieldDescriptor
    from onnx import TensorProto

    def tensors(message):  # noqa: ANN001
        if isinstance(message, TensorProto):
            yield message
            return
        for field, value in message.ListFields():
            if field.type != FieldDescriptor.TYPE_MESSAGE:
                continue
            is_repeated = getattr(field, "is_repeated", None)
            if is_repeated is None:
                is_repeated = field.label == FieldDescriptor.LABEL_REPEATED
            if is_repeated:
                for item in value:
                    yield from tensors(item)
            else:
                yield from tensors(value)

    yield from tensors(proto)


def _has_unloaded_external_data(proto) -> bool:  # noqa: ANN001
    """Return whether a model still references tensor bytes not loaded in memory."""
    from onnx.external_data_helper import uses_external_data

    return any(uses_external_data(tensor) and not tensor.HasField("raw_data") for tensor in _iter_tensors(proto))


def _validate_external_data_pair(model_path: Path, data_name: str) -> None:
    """Validate protobuf parsing and every external-data reference without loading tensor bytes."""
    import math

    import onnx
    from onnx import TensorProto, helper
    from onnx.external_data_helper import uses_external_data

    model = onnx.load(str(model_path), load_external_data=False)
    data_path = model_path.with_name(data_name)
    data_size = data_path.stat().st_size if data_path.is_file() else None
    for tensor in _iter_tensors(model):
        if not uses_external_data(tensor):
            continue
        metadata = {entry.key: entry.value for entry in tensor.external_data}
        if metadata.get("location") != data_name or data_size is None:
            raise RuntimeError(f"Invalid external data reference for tensor {tensor.name!r}")
        try:
            offset = int(metadata.get("offset", "0"))
            length = int(metadata["length"])
        except (KeyError, ValueError) as exc:
            raise RuntimeError(f"Invalid external data bounds for tensor {tensor.name!r}") from exc
        element_count = math.prod(tensor.dims)
        dtype_name = TensorProto.DataType.Name(tensor.data_type)
        packed_bits = (
            4 if dtype_name in {"INT4", "UINT4", "FLOAT4E2M1"} else 2 if dtype_name in {"INT2", "UINT2"} else 0
        )
        if packed_bits:
            expected_length = (element_count * packed_bits + 7) // 8
        elif dtype_name == "BFLOAT16":
            expected_length = element_count * 2
        elif dtype_name.startswith("FLOAT8"):
            expected_length = element_count
        else:
            try:
                expected_length = element_count * helper.tensor_dtype_to_np_dtype(tensor.data_type).itemsize
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Unsupported external tensor dtype for {tensor.name!r}") from exc
        if offset < 0 or length != expected_length or offset + length > data_size:
            raise RuntimeError(f"External data bounds exceed {data_name} for tensor {tensor.name!r}")


def _save_external_data_pair(
    save_model,  # noqa: ANN001
    proto,  # noqa: ANN001
    output_path: Path,
    data_name: str,
    *args,
    **kwargs,
) -> None:
    """Stage, validate, and replace an ONNX/external-data pair."""
    from onnx import TensorProto
    from onnx.external_data_helper import uses_external_data

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if Path(data_name).name != data_name or data_name.startswith(".") or data_name == output_path.name:
        raise ValueError("External ONNX data_name must be a distinct, non-hidden file name")
    if _has_unloaded_external_data(proto):
        raise ValueError("Cannot save an ONNX model with unloaded external tensors; load external data first")

    data_path = output_path.with_name(data_name)

    # Loaded external tensors may retain an old location. Normalize them to
    # inline storage so this save owns every external reference it creates.
    for tensor in _iter_tensors(proto):
        if uses_external_data(tensor):
            tensor.data_location = TensorProto.DEFAULT
            del tensor.external_data[:]

    with tempfile.TemporaryDirectory(prefix=f".{output_path.name}.", dir=output_path.parent) as tmp_dir:
        staging = Path(tmp_dir)
        staged_model = staging / output_path.name
        staged_data = staging / data_name
        unique_data_name = f".{data_name}.{uuid4().hex}"
        unique_staged_data = staging / unique_data_name
        save_kwargs = dict(kwargs)
        save_kwargs.update(
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=unique_data_name,
            convert_attribute=True,
        )
        save_model(proto, str(staged_model), *args, **save_kwargs)
        if not staged_model.is_file():
            raise RuntimeError(f"ONNX save did not produce {staged_model}")

        has_staged_data = unique_staged_data.is_file()
        external_tensors = [tensor for tensor in _iter_tensors(proto) if uses_external_data(tensor)]
        if external_tensors and not has_staged_data:
            raise RuntimeError(f"ONNX save did not produce external data {unique_staged_data}")
        for tensor in external_tensors:
            for entry in tensor.external_data:
                if entry.key == "location":
                    entry.value = data_name
        save_model(proto, str(staged_model))
        if has_staged_data:
            os.replace(unique_staged_data, staged_data)

        _validate_external_data_pair(staged_model, data_name)

        destinations = [(staged_data, data_path)] if has_staged_data else []
        destinations.append((staged_model, output_path))
        backup_destinations = [destination for _, destination in destinations]
        if not has_staged_data:
            backup_destinations.append(data_path)
        backups: dict[Path, Path] = {}
        installed: list[Path] = []
        for destination in backup_destinations:
            if destination.exists():
                backup = staging / f"backup-{uuid4().hex}"
                os.link(destination, backup)
                backups[destination] = backup
        try:
            for staged, destination in destinations:
                os.replace(staged, destination)
                installed.append(destination)
            if not has_staged_data and data_path.exists():
                data_path.unlink()
                installed.append(data_path)
        except Exception:
            for destination in reversed(installed):
                backup = backups.get(destination)
                if backup is not None:
                    os.replace(backup, destination)
                else:
                    try:  # noqa: SIM105
                        destination.unlink()
                    except FileNotFoundError:
                        pass
            raise


def install_large_model_save_patch() -> None:
    """Force ``onnx.save`` to use external data for >2 GB models.

    PI05's PaliGemma trunk is several GB; msModelSlim's internal
    ``convert_version`` / save steps call ``onnx.save(model, path)`` without
    ``save_as_external_data=True``, so protobuf's 2 GB single-message limit
    raises ``EncodeError: Failed to serialize proto``. We wrap ``onnx.save``
    and ``onnx.save_model`` to transparently spill tensors to a sidecar
    ``.data`` file when the model is large, leaving small models untouched.

    The resulting ``.onnx`` + ``.data`` pair must be kept together for ATC.
    """
    import onnx

    if getattr(onnx, "_pi05_large_save_patched", False):
        return

    _orig_save_model = onnx.save_model

    def _patched_save_model(proto, f, *args, **kwargs):  # noqa: ANN001
        if isinstance(proto, bytes):
            return _orig_save_model(proto, f, *args, **kwargs)
        if not isinstance(f, str | os.PathLike) or Path(f).suffix.lower() != ".onnx":
            return _orig_save_model(proto, f, *args, **kwargs)

        save_format = kwargs.get("format", args[0] if args else None)
        save_as_external = kwargs.get("save_as_external_data", args[1] if len(args) > 1 else False)
        other_external_options = {
            "all_tensors_to_one_file",
            "location",
            "size_threshold",
            "convert_attribute",
        }
        can_intercept = (
            save_format in (None, "protobuf")
            and not save_as_external
            and len(args) <= 2
            and not any(name in kwargs for name in other_external_options)
        )
        if can_intercept:
            try:
                too_big = proto.ByteSize() > _PROTOBUF_INLINE_LIMIT
            except Exception:  # noqa: BLE001
                # ByteSize() raises EncodeError (>2 GB) on some protobuf builds —
                # that overflow IS the signal that we must use external data.
                too_big = True
            if too_big and isinstance(f, str | os.PathLike):
                location = Path(f).name + ".data"
                if _has_unloaded_external_data(proto):
                    raise ValueError(
                        "Cannot rewrite a large ONNX model with unloaded external tensors; "
                        "load external data before saving"
                    )
                LOGGER.info(
                    "Large model (>%.1f GB inline limit): saving %s with external data → %s",
                    _PROTOBUF_INLINE_LIMIT / 1e9,
                    Path(f).name,
                    location,
                )
                clean_kwargs = dict(kwargs)
                clean_kwargs.pop("format", None)
                clean_kwargs.pop("save_as_external_data", None)
                return _save_external_data_pair(
                    _orig_save_model,
                    proto,
                    Path(f),
                    location,
                    size_threshold=1024,
                    **clean_kwargs,
                )
        return _orig_save_model(proto, f, *args, **kwargs)

    onnx.save_model = _patched_save_model
    onnx.save = _patched_save_model  # onnx.save is an alias of save_model
    onnx._pi05_large_save_patched = True  # type: ignore[attr-defined]


def save_onnx_external(model, output_path: Path, data_name: str | None = None) -> None:
    """Save ``model`` with a consolidated external-data sidecar (overwrite-safe).

    CRITICAL: ONNX's external-data writer opens the ``.data`` sidecar in append
    mode (``"ab"``). If a ``.data`` already exists at the target path, the new
    tensor bytes are appended *after* the stale ones, so the file roughly
    DOUBLES on every re-save (the model still loads — the TensorProto offsets are
    recorded against the appended position — but the leading bytes become dead
    weight). Our pipeline re-saves the same path more than once
    (``run_quantize`` → dequant-pin, and transplant → dequant-pin), so we must
    avoid writing directly over that stale pair. The input proto is consumed in
    the same way as ``onnx.save_model(..., save_as_external_data=True)`` and must
    not be reused after this call.
    """
    import onnx

    output_path = Path(output_path)
    if data_name is None:
        data_name = output_path.name + ".data"
    _save_external_data_pair(
        onnx.save_model,
        model,
        output_path,
        data_name,
        size_threshold=1024,
    )


def proto_input_names(model_proto) -> list[str]:
    """Real graph input names (graph.input minus initializers), in order."""
    initializer_names = {init.name for init in model_proto.graph.initializer}
    return [vi.name for vi in model_proto.graph.input if vi.name not in initializer_names]


def install_msmodelslim_calib_patch() -> None:
    """Make msModelSlim's calibration DataReader work for multi-GB models.

    Two problems are fixed by replacing ``DataReader._get_and_check_data``:

    1. **2 GB serialisation crash.** The stock method builds an ORT session via
       ``ort.InferenceSession(model.SerializeToString())`` *only* to read input
       metadata (``session.get_inputs()``). For PI05's multi-GB PaliGemma trunk
       that ``SerializeToString()`` overflows protobuf's 2 GB single-message
       limit (``EncodeError: Failed to serialize proto``). We derive the input
       names directly from the loaded proto's ``graph.input`` instead — no
       byte-serialisation, no session.

    2. **bool input silently dropped.** ``util.check_input_data`` validates
       dtypes against ``INPUT_DTYPE_DICT`` which only knows
       float/int64/int32 — it has no ``tensor(bool)`` entry. PI05's
       ``lang_masks`` / ``prefix_pad_masks`` is bool, so every calibration batch
       would be judged "invalid", discarded, and msModelSlim would *silently
       fall back to random calibration data* (ruining the quantization). We
       bypass that check and bind our pre-built, correctly-typed arrays straight
       to the graph inputs by position.

    The calibration data we pass is already ordered to match ``graph.input``
    (see each model's calib-data builder), so a positional zip is exact.
    """
    install_onnx_mapping_shim()  # importing msmodelslim triggers onnx.mapping

    from msmodelslim.onnx.post_training_quant.label_free import data_reader as _dr

    if getattr(_dr.DataReader, "_pi05_calib_patched", False):
        return

    def _patched_get_and_check_data(self, model, calib_data, quant_cfg=None):  # noqa: ANN001
        input_names = proto_input_names(model)
        if not calib_data:
            raise ValueError(
                "PI05 calibration requires explicit calib_data; random fallback "
                "is disabled to avoid silently mis-calibrating the W8A8 model."
            )
        valid: list[dict] = []
        for index, data in enumerate(calib_data):
            data_list = data if isinstance(data, list) else [data]
            if len(data_list) != len(input_names):
                LOGGER.warning(
                    "Calib sample %d has %d array(s) but model has %d input(s); skipping.",
                    index,
                    len(data_list),
                    len(input_names),
                )
                continue
            valid.append({name: arr for name, arr in zip(input_names, data_list, strict=False)})
        if not valid:
            raise ValueError("No usable calibration samples after input-count check.")
        LOGGER.info(
            "Calibration DataReader patched: %d sample(s) bound to inputs %s.",
            len(valid),
            input_names,
        )
        return valid

    _dr.DataReader._get_and_check_data = _patched_get_and_check_data
    _dr.DataReader._pi05_calib_patched = True  # type: ignore[attr-defined]
    LOGGER.info("Installed msModelSlim DataReader calibration patch (2 GB + bool-input safe).")


def _create_external_data_ort_session(model, scratch_dir: Path | None = None):  # noqa: ANN001
    """Create an ORT session without serializing a multi-GB model to bytes."""
    import onnxruntime

    temp_dir = tempfile.TemporaryDirectory(prefix="pi05_amp_", dir=scratch_dir)
    try:
        tmp_dir = temp_dir.name
        model_path = Path(tmp_dir) / "model.onnx"
        save_onnx_external(model, model_path)
        session = onnxruntime.InferenceSession(str(model_path))
    except Exception:
        temp_dir.cleanup()
        raise
    # ORT may reload the model when providers change, so retain the pair for
    # exactly as long as the session rather than deleting it on return.
    session._pi05_external_data_tempdir = temp_dir
    return session


def _make_fp16_qdq_ort_compatible(model) -> tuple[int, int]:  # noqa: ANN001
    """Wrap fp16 Q/DQ edges for ORT's opset-17 CPU implementation."""
    from onnx import TensorProto, helper, numpy_helper

    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    scale_names = set()
    for node in model.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear") and len(node.input) > 1:
            scale_names.add(node.input[1])
        elif node.op_type in ("QLinearMatMul", "QLinearConv"):
            scale_names.update(node.input[index] for index in (1, 4, 6) if len(node.input) > index)
        elif node.domain == "com.microsoft" and node.op_type == "QGemm":
            scale_names.update(node.input[index] for index in (1, 4, 7) if len(node.input) > index)
    fp16_scales = {
        name
        for name in scale_names
        if (scale := initializers.get(name)) is not None and scale.data_type == TensorProto.FLOAT16
    }
    for name in fp16_scales:
        scale = initializers[name]
        scale.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(scale).astype(np.float32), name=name))

    quantize_count = 0
    dequantize_count = 0
    nodes = []
    used_tensor_names = {name for node in model.graph.node for name in (*node.input, *node.output) if name}
    used_tensor_names.update(value.name for value in model.graph.input)
    used_tensor_names.update(value.name for value in model.graph.output)
    used_tensor_names.update(value.name for value in model.graph.value_info)
    used_tensor_names.update(initializers)
    used_node_names = {node.name for node in model.graph.node if node.name}
    input_casts: dict[str, str] = {}

    def unique_name(base: str, used: set[str]) -> str:
        candidate = base
        suffix = 1
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        return candidate

    for node in model.graph.node:
        if node.op_type == "QuantizeLinear" and node.input[1] in fp16_scales:
            source = node.input[0]
            cast_output = input_casts.get(source)
            if cast_output is None:
                cast_output = unique_name(source + "_amp_fp32", used_tensor_names)
                input_casts[source] = cast_output
                nodes.append(
                    helper.make_node(
                        "Cast",
                        [source],
                        [cast_output],
                        name=unique_name((node.name or node.output[0]) + "/amp_input_cast", used_node_names),
                        to=TensorProto.FLOAT,
                    )
                )
            node.input[0] = cast_output
            quantize_count += 1
        if node.op_type == "DequantizeLinear" and node.input[1] in fp16_scales:
            output = node.output[0]
            fp32_output = unique_name(output + "_amp_fp32", used_tensor_names)
            node.output[0] = fp32_output
            nodes.append(node)
            nodes.append(
                helper.make_node(
                    "Cast",
                    [fp32_output],
                    [output],
                    name=unique_name((node.name or output) + "/amp_output_cast", used_node_names),
                    to=TensorProto.FLOAT16,
                )
            )
            dequantize_count += 1
            continue
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return quantize_count, dequantize_count


def _activation_l2_error(float_array: np.ndarray, quant_array: np.ndarray) -> float:
    """Compute sensitivity without overflowing ModelSlim's fp16 norm."""
    if float_array.shape != quant_array.shape:
        raise ValueError(
            f"AMP activation shapes do not match: float={float_array.shape}, quantized={quant_array.shape}"
        )
    float_array = float_array.astype(np.float32)
    quant_array = quant_array.astype(np.float32)
    if not np.all(np.isfinite(float_array)) or not np.all(np.isfinite(quant_array)):
        raise ValueError("AMP activation arrays must contain only finite values")
    error = float(np.linalg.norm((float_array - quant_array).reshape(-1)))
    if not np.isfinite(error):
        raise ValueError("AMP activation L2 error is not finite")
    return error


def _validate_amp_rollback_count(amp_num: int, rankable_count: int) -> None:
    if not 0 < amp_num < rankable_count:
        raise ValueError(f"amp_num must be smaller than the {rankable_count} rankable quantized layers; got {amp_num}")


def _set_msmodelslim_amp_calib_samples(rollback_module, calib_samples: list[list[np.ndarray]]) -> None:  # noqa: ANN001
    rollback_module._pi05_calib_samples = calib_samples


def _set_msmodelslim_amp_scratch_dir(rollback_module, scratch_dir: Path) -> None:  # noqa: ANN001
    rollback_module._pi05_amp_scratch_dir = scratch_dir


def _update_activation_l2_sums(
    error_sums: np.ndarray,
    float_arrays: list[np.ndarray],
    quant_arrays: list[np.ndarray],
) -> None:
    if len(float_arrays) != len(error_sums) or len(quant_arrays) != len(error_sums):
        raise ValueError("AMP activation output counts do not match")
    for index, (float_array, quant_array) in enumerate(zip(float_arrays, quant_arrays, strict=True)):
        error_sums[index] += _activation_l2_error(float_array, quant_array)


def _restore_opset_imports(model, opsets: list[tuple[str, int]]) -> None:  # noqa: ANN001
    import onnx

    del model.opset_import[:]
    model.opset_import.extend(onnx.helper.make_opsetid(domain, version) for domain, version in opsets)


def install_msmodelslim_amp_patch(calib_samples: list[list[np.ndarray]], scratch_dir: Path) -> None:
    """Make ``amp_num`` rollback work with real data and multi-GB models.

    ``amp_num > 0`` asks msModelSlim to auto-roll-back the most
    quantization-sensitive layers to fp16. To rank them it runs the float and
    quantized donor models and compares per-node activation MSE
    (``rollback_quant_nodes.match_activations``). Two stock behaviours make this
    unusable for PI05 out of the box — and neither is fixed by the DataReader
    patch, because this is a *separate* code path:

    1. **Random ranking inputs.** ``match_activations`` feeds
       ``gen_model_inputs`` which returns ``np.random.random(shape)`` — the
       sensitivity ranking would be computed on meaningless noise, so the
       rolled-back layers would be essentially arbitrary.

    2. **bool dtype crash.** ``gen_model_inputs`` does
       ``np.random.random(shape).astype(INPUT_DTYPE_DICT.get(input.type))`` and
       ``INPUT_DTYPE_DICT`` has no ``tensor(bool)`` entry → ``.astype(None)``
       yields float64 for our bool masks, which ORT rejects / mis-feeds.

    We replace ``gen_model_inputs`` with one that returns real calibration data,
    bound positionally to the model inputs with their declared dtypes (bool
    included). We also replace the activation matcher:
    stock ModelSlim calls ``model.SerializeToString()`` and exposes every graph
    node as an output. The former fails for the multi-GB VLM donor, while the
    latter exhausts 32 GB of RAM. The replacement loads temporary external-data
    models by path, fetches only quantized-layer outputs, and releases the float
    session before creating the quantized session. For multi-sample ranking,
    float outputs are staged on disk so only one model and one sample's
    activations are resident at a time.
    """
    install_onnx_mapping_shim()

    import gc

    import onnx
    from msmodelslim.onnx.post_training_quant.label_free import quantize_tool as _qt
    from msmodelslim.onnx.post_training_quant.label_free import rollback_quant_nodes as _rb

    if not calib_samples:
        raise ValueError("AMP rollback requires at least one calibration sample")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    _set_msmodelslim_amp_calib_samples(_rb, calib_samples)
    _set_msmodelslim_amp_scratch_dir(_rb, scratch_dir)
    if getattr(_rb, "_pi05_amp_patched", False):
        return

    def _patched_gen_model_inputs(inputs, quant_config=None):  # noqa: ANN001
        current_sample = _rb._pi05_calib_samples[0]
        if len(current_sample) != len(inputs):
            raise ValueError(
                f"amp rollback: calib sample has {len(current_sample)} array(s) but model has {len(inputs)} input(s)."
            )
        return {inp.name: arr for inp, arr in zip(inputs, current_sample, strict=False)}

    def _bind_sample(inputs, sample):  # noqa: ANN001
        if len(sample) != len(inputs):
            raise ValueError(
                f"amp rollback: calib sample has {len(sample)} array(s) but model has {len(inputs)} input(s)."
            )
        return {inp.name: arr for inp, arr in zip(inputs, sample, strict=False)}

    def _patched_create_session(model):  # noqa: ANN001
        return _create_external_data_ort_session(model, _rb._pi05_amp_scratch_dir)

    def _patched_match_activations(float_model_path, quant_model_path, quantized_nodes=None, quant_config=None):
        quantized_node_names = set(quantized_nodes or ())
        float_topology = onnx.load(float_model_path, load_external_data=False)
        output_to_node = {
            output: node.name
            for node in float_topology.graph.node
            if node.name in quantized_node_names
            for output in node.output
        }
        float_outputs = [
            output for node in float_topology.graph.node if node.name in quantized_node_names for output in node.output
        ]
        quant_topology = onnx.load(quant_model_path, load_external_data=False)
        quant_outputs = {
            output for node in quant_topology.graph.node if node.op_type == "DequantizeLinear" for output in node.output
        }
        quant_ir_version = quant_topology.ir_version
        quant_opsets = [(entry.domain, entry.version) for entry in quant_topology.opset_import]
        output_names = list(dict.fromkeys(name for name in float_outputs if name in quant_outputs))
        if not output_names:
            raise RuntimeError("amp rollback found no shared quantized-layer outputs to compare.")
        _validate_amp_rollback_count(quant_config.amp_num, len(output_names))
        rank_samples = _rb._pi05_calib_samples
        LOGGER.info(
            "AMP rollback: comparing %d quantized-layer output(s) across %d sample(s).",
            len(output_names),
            len(rank_samples),
        )
        del float_topology, quant_topology

        def expose_outputs(model):  # noqa: ANN001
            del model.graph.output[:]
            model.graph.output.extend(onnx.ValueInfoProto(name=name) for name in output_names)

        from tempfile import TemporaryDirectory

        with TemporaryDirectory(prefix="pi05_amp_outputs_", dir=_rb._pi05_amp_scratch_dir) as output_dir:
            output_root = Path(output_dir)
            float_model = onnx.load(float_model_path)
            expose_outputs(float_model)
            float_session = _patched_create_session(float_model)
            float_inputs = float_session.get_inputs()
            for sample_index, sample in enumerate(rank_samples):
                input_dict = _bind_sample(float_inputs, sample)
                float_arrays = float_session.run(output_names, input_dict)
                for output_index, array in enumerate(float_arrays):
                    np.save(output_root / f"{sample_index}_{output_index}.npy", array, allow_pickle=False)
                del float_arrays
            del float_session, float_model
            gc.collect()

            quant_model = _rb.preprocess_quant_model(onnx.load(quant_model_path))
            quant_model.ir_version = quant_ir_version
            _restore_opset_imports(quant_model, quant_opsets)
            q_count, dq_count = _make_fp16_qdq_ort_compatible(quant_model)
            LOGGER.info(
                "AMP rollback: wrapped %d QuantizeLinear and %d DequantizeLinear fp16 edge(s).",
                q_count,
                dq_count,
            )
            expose_outputs(quant_model)
            quant_session = _patched_create_session(quant_model)
            quant_inputs = quant_session.get_inputs()
            error_sums = np.zeros(len(output_names), dtype=np.float64)
            for sample_index, sample in enumerate(rank_samples):
                input_dict = _bind_sample(quant_inputs, sample)
                quant_arrays = quant_session.run(output_names, input_dict)
                float_arrays = [
                    np.load(output_root / f"{sample_index}_{output_index}.npy", mmap_mode="r")
                    for output_index in range(len(output_names))
                ]
                _update_activation_l2_sums(error_sums, float_arrays, quant_arrays)
                del float_arrays, quant_arrays
            del quant_session, quant_model
            gc.collect()

        errors = {name: float(error_sums[index] / len(rank_samples)) for index, name in enumerate(output_names)}
        LOGGER.info(
            "AMP rollback sensitivity ranking: %s",
            [(output_to_node[name], float(error)) for name, error in sorted(errors.items(), key=lambda item: -item[1])],
        )
        return errors

    _rb.gen_model_inputs = _patched_gen_model_inputs
    _rb.get_session_for_intermediate_output = _patched_create_session
    _rb.match_activations = _patched_match_activations
    _qt.match_activations = _patched_match_activations
    _rb._pi05_amp_patched = True  # type: ignore[attr-defined]
    LOGGER.info(
        "Installed msModelSlim amp-rollback patch: layer-sensitivity ranking uses "
        "real calibration data and external-data ORT sessions."
    )


def onnx_default_opset(onnx_path: Path) -> int | None:
    """Return the ai.onnx (default-domain) opset version of an ONNX file."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            return entry.version
    return None


def install_msmodelslim_opset_patch(target_opset: int) -> None:
    """Stop msModelSlim from hard-coding the quantized model to opset 11.

    ``dag/graph.py``'s ``OnnxGraph.build_model`` builds the final quantized
    model with ``make_opsetid(domain="", version=11)`` regardless of the input
    opset. PI05's SigLIP vision tower (kept in fp16) uses ``LayerNormalization``
    which only exists from opset 17, so ATC rejects the saved graph with::

        No parser is registered for Op [.../post_layernorm/LayerNormalization,
        optype [ai.onnx::11::LayerNormalization]]

    The nodes themselves are unchanged opset-17 ops; only the declared
    ``opset_import`` is wrong. We wrap ``build_model`` to rewrite the
    default-domain opset back to the input model's version (17) and bump
    ``ir_version`` to >=8 if needed (opset 17 requires it).
    """
    install_onnx_mapping_shim()  # importing the dag package triggers onnx.mapping

    from msmodelslim.onnx.post_training_quant.dag import graph as _g

    if getattr(_g.OnnxGraph, "_pi05_opset_patched", False):
        return

    _orig_build = _g.OnnxGraph.build_model

    def _patched_build(self):  # noqa: ANN001
        model = _orig_build(self)
        for entry in model.opset_import:
            if entry.domain in ("", "ai.onnx"):
                entry.version = target_opset
        if model.ir_version < 8:  # opset >= 17 requires ir_version >= 8
            model.ir_version = 8
        return model

    _g.OnnxGraph.build_model = _patched_build
    _g.OnnxGraph._pi05_opset_patched = True  # type: ignore[attr-defined]
    LOGGER.info("Installed msModelSlim opset patch: quantized model will use opset %d.", target_opset)


# ---------------------------------------------------------------------------
# Resize empty-input pre-pass
# ---------------------------------------------------------------------------
# Shared empty fp32 initializer name for cleared Resize optional inputs.
_RESIZE_EMPTY_INIT = "pi05_resize_empty_roi_scales"


def fix_resize_empty_optional_inputs(model) -> int:
    """Replace empty-string optional ``Resize`` inputs with empty initializers.

    PI05's SigLIP image preprocessing exports a ``Resize`` that uses the
    ``sizes`` input (``F.interpolate(size=...)``), leaving the optional
    ``roi`` and ``scales`` inputs as empty strings::

        Resize(X, "", "", sizes) -> Y

    Empty strings are legal ONNX, and a plain InferenceSession loads them
    fine. But ORT's *quantization* path re-saves an augmented calibration
    model and reloads it; on that reload ORT resolves the Resize schema such
    that input 1 (``roi``) is treated as a required ("single") input and a
    bare empty string raises::

        INVALID_GRAPH: Node (/Resize)'s input 1 is marked single but has an
        empty string in the graph

    We make the node unambiguous by pointing every *interior* empty optional
    input (one that precedes a later non-empty input) at a shared zero-length
    fp32 initializer. ``roi.size()==0`` / ``scales.size()==0`` keep their
    "unspecified" semantics, so behaviour is identical — but there is no
    empty *string* left for ORT to reject.

    Returns the number of inputs rewritten.
    """
    from onnx import TensorProto, helper

    graph = model.graph
    rewritten = 0
    needs_init = False
    for node in graph.node:
        if node.op_type != "Resize":
            continue
        last_used = -1
        for idx, name in enumerate(node.input):
            if name:
                last_used = idx
        for idx in range(1, last_used):  # input 0 (X) is always present
            if not node.input[idx]:
                node.input[idx] = _RESIZE_EMPTY_INIT
                rewritten += 1
                needs_init = True

    if needs_init and all(init.name != _RESIZE_EMPTY_INIT for init in graph.initializer):
        empty = helper.make_tensor(
            name=_RESIZE_EMPTY_INIT,
            data_type=TensorProto.FLOAT,  # roi/scales are always fp32 per spec
            dims=[0],
            vals=[],
        )
        graph.initializer.append(empty)
    return rewritten


def prepare_quant_input(input_onnx: Path) -> Path:
    """Return a path to an ONNX safe to feed msModelSlim's quantizer.

    Currently this only fixes empty-string ``Resize`` optional inputs (see
    :func:`fix_resize_empty_optional_inputs`). If no fix is needed the original
    path is returned unchanged; otherwise a sibling ``*_qprep.onnx`` (+ a
    consolidated ``.data`` sidecar) is written and its path returned.
    """
    import onnx

    model = onnx.load(str(input_onnx))  # loads external data via file path
    n_fixed = fix_resize_empty_optional_inputs(model)
    if n_fixed == 0:
        return input_onnx

    fixed_path = input_onnx.with_name(input_onnx.stem + "_qprep.onnx")
    save_onnx_external(model, fixed_path)
    LOGGER.info(
        "Fixed %d empty Resize optional input(s); quant input → %s",
        n_fixed,
        fixed_path,
    )
    return fixed_path


# ---------------------------------------------------------------------------
# W8A8 driver
# ---------------------------------------------------------------------------


def run_msmodelslim_w8a8(
    *,
    input_onnx: Path,
    output_onnx: Path,
    calib_data: list[list[np.ndarray]],
    disable_names: list[str],
    amp_num: int,
    amp_rank_samples: int = 1,
    amp_scratch_dir: Path | None = None,
    npu_graph: Path | None = None,
) -> int:
    """Calibrate and export a W8A8 quantized ONNX via msModelSlim.

    Isolated so it is the only place to touch when adapting to a different
    msModelSlim release. Targets the master-branch functional API::

        run_quantize(input_model_path, output_model_path, quant_config)
        QuantConfig(quant_mode=1, is_signed_quant=True, is_per_channel=True,
                    calib_data=None, calib_method=0, quantize_nodes=None,
                    exclude_nodes=None, amp_num=0, is_optimize_graph=True, ...)

    When ``npu_graph`` is given, the quantized ORT graph is treated as a
    *calibration donor*: it is quantized here (msModelSlim must run it on CPU,
    which NPU custom ops forbid) and its int8 Linears are then grafted onto the
    NPU-op graph (Route A — see :func:`transplant_int8_into_npu_graph`).
    """
    # msModelSlim still imports the legacy onnx.mapping module — shim it in
    # before the import so newer onnx releases don't break the import chain.
    install_onnx_mapping_shim()
    # PI05 is multi-GB → force external-data saving past protobuf's 2 GB limit.
    install_large_model_save_patch()
    # Calibration DataReader: avoid the 2 GB SerializeToString crash and the
    # silent bool-input → random-data fallback (see function docstring).
    install_msmodelslim_calib_patch()
    # amp_num rollback ranks layers on its own (random) data path — make it use
    # our real calibration sample so --amp-num is meaningful and bool-safe.
    if amp_num > 0:
        if amp_rank_samples <= 0:
            raise ValueError("amp_rank_samples must be positive")
        scratch_dir = (amp_scratch_dir or output_onnx.parent).expanduser().resolve()
        install_msmodelslim_amp_patch(calib_data[:amp_rank_samples], scratch_dir)
    # msModelSlim hard-codes opset 11 when saving the quantized model; force it
    # back to the input model's opset so ATC accepts LayerNormalization (>=17).
    _target_opset = onnx_default_opset(input_onnx) or 17
    install_msmodelslim_opset_patch(_target_opset)

    try:
        from msmodelslim.onnx.post_training_quant import QuantConfig, run_quantize
    except ImportError as exc:  # pragma: no cover - depends on board install
        raise ImportError(
            "msmodelslim is required for W8A8 quantization. It is expected to be "
            "installed on the Ascend dev board. If the import path differs in your "
            "version, adapt run_msmodelslim_w8a8(). Original error: " + str(exc)
        ) from exc

    output_onnx.parent.mkdir(parents=True, exist_ok=True)

    # Fix empty-string Resize optional inputs (SigLIP image-preprocess) that
    # otherwise crash ORT's augmented-calibration-model reload.
    quant_input = prepare_quant_input(input_onnx)

    # quant_mode=1     → W8A8 (signed int8 weights + activations) on Ascend.
    # is_per_channel   → per-channel weight scales (best accuracy for Linear).
    # exclude_nodes    → node names kept in fp16 (vision-tower / non-weight
    #                    matmul protection list).
    # amp_num          → number of most-sensitive layers msModelSlim auto-rolls
    #                    back to fp16 to recover accuracy.
    # is_optimize_graph=False → CRITICAL. The default (True) runs ORT
    #   ORT_ENABLE_BASIC optimization + convert_version(opset 11) BEFORE node
    #   detection. On our fp16 semantic graph that rewrites/renames the weight
    #   initializers, so msModelSlim's get_quantized_nodes finds "0 node will
    #   be quantized" and our exclude_nodes no longer match. Disabling it runs
    #   detection on the original graph where weights are real initializers and
    #   names line up with our fp16 exclusion list.
    quant_config = QuantConfig(
        quant_mode=1,
        is_signed_quant=True,
        is_per_channel=True,
        calib_data=calib_data,
        calib_method=0,
        exclude_nodes=disable_names,
        amp_num=amp_num,
        is_optimize_graph=False,
    )

    LOGGER.info(
        "Running msModelSlim W8A8: %d calib sample(s), %d node(s) kept in fp16, amp_num=%d",
        len(calib_data),
        len(disable_names),
        amp_num,
    )

    # When an NPU-op graph is supplied (Route A) the quantized ORT graph is only
    # a *donor*: it is calibrated/quantized here (because msModelSlim must run it
    # on CPU, which the NPU custom ops forbid) and then its int8 Linears are
    # grafted onto the NPU graph. Keep the donor as a sibling file so the
    # no-NPU path (donor == final output) is unaffected when npu_graph is None.
    if npu_graph is not None:
        donor_onnx = output_onnx.with_name(output_onnx.stem + "_ortdonor.onnx")
    else:
        donor_onnx = output_onnx

    run_quantize(str(quant_input), str(donor_onnx), quant_config)
    LOGGER.info("W8A8 donor ONNX written to %s", donor_onnx)

    # Pin AscendDequant outputs to fp16 so ATC selects the (only) available
    # 310P kernel. See fix_ascend_dequant_output_dtype for the full rationale.
    fix_ascend_dequant_output_dtype(donor_onnx)

    if npu_graph is not None:
        quantized_nodes = transplant_int8_into_npu_graph(donor_onnx, npu_graph, output_onnx)
        # Re-pin: the transplant renamed every dequant output to the NPU graph's
        # downstream tensor, so the fp16 value_info must be re-declared on those.
        fix_ascend_dequant_output_dtype(output_onnx)
        LOGGER.info("Final NPU + W8A8 ONNX written to %s", output_onnx)
        return quantized_nodes
    return count_quantized_nodes(output_onnx)


def count_quantized_nodes(output_onnx: Path) -> int:
    """Count quantized compute sites by their one-to-one AscendDequant outputs."""
    model = load_onnx(output_onnx)
    return sum(node.op_type == "AscendDequant" for node in model.graph.node)


def fix_ascend_dequant_output_dtype(output_onnx: Path) -> None:
    """Declare every ``AscendDequant`` output as fp16 so ATC compiles it.

    Root cause of ATC's ``EZ3002 ... AscendDequant ... data type DT_FLOAT of
    output [y] is not supported`` (the Gemma trunk Linears all fail):

    The exported VLM/AE graph carries **no value_info** — every interior tensor
    is ``UNDEFINED``. msModelSlim's ``AscendDequant`` node likewise leaves its
    output type unset. ATC's ONNX parser then reads the output's declared
    elem_type, sees ``UNDEFINED`` (0), and maps it to GE ``DT_FLOAT`` (also 0).
    But the 310P ``AscendDequant`` AICORE kernel only implements
    ``output y = float16`` (per aic-ascend310p ops-info:
    ``Data Type:{DT_FLOAT16,...} Format:{NC1HWC0,FRACTAL_NZ,NDC1HWC0}``), so no
    kernel matches the inferred fp32 output → "No supported Ops kernel".

    Fix: add an explicit fp16 ``value_info`` for each ``AscendDequant`` output
    (and set the node's ``dtype`` attribute to the GE fp16 enum = 1). Since the
    whole graph is otherwise UNDEFINED, ATC propagates fp16 forward to the
    consumer — no dtype-mismatch edges are created. The int8/int32/uint64 quant
    tensors are untouched. This is NOT an opset/precision change; it only pins
    the dequant output that ATC was mis-inferring.
    """
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(output_onnx))  # file path → external data ok
    graph = model.graph

    existing_vi = {vi.name for vi in graph.value_info}
    pinned = 0
    for node in graph.node:
        if node.op_type != "AscendDequant":
            continue
        if not any(a.name == "dtype" for a in node.attribute):
            node.attribute.append(helper.make_attribute("dtype", 1))
        for out_name in node.output:
            if out_name and out_name not in existing_vi:
                graph.value_info.append(helper.make_tensor_value_info(out_name, TensorProto.FLOAT16, None))
                existing_vi.add(out_name)
        pinned += 1

    if pinned == 0:
        LOGGER.warning(
            "No AscendDequant nodes found in %s — dtype pin skipped (unexpected).",
            output_onnx,
        )
        return

    save_onnx_external(model, output_onnx)
    LOGGER.info("Pinned %d AscendDequant output(s) to fp16 → %s", pinned, output_onnx)


# ---------------------------------------------------------------------------
# Route A: transplant int8 Linears into an NPU-op graph
# ---------------------------------------------------------------------------
# Suffix ORT's quantizer appends to a MatMul/Gemm node name when it rewrites it
# into the QLinear* form. msModelSlim keeps that name, so the int8 MatMul in the
# donor is "<original_name>_quant" — stripping it recovers the node name as it
# appears in the (un-quantized) NPU graph.
_QUANT_NODE_SUFFIX = "_quant"


def topo_sort_graph(graph) -> None:
    """Reorder ``graph.node`` into a valid topological order (in place).

    ONNX requires every node to appear after the nodes producing its inputs.
    The transplant appends freshly-built AscendQuant/MatMul/AscendDequant nodes
    at the end and rewires downstream consumers, which breaks that invariant —
    a single Kahn pass restores it. O(V+E); the graph is already mostly sorted.
    """
    from collections import defaultdict, deque

    available: set[str] = {init.name for init in graph.initializer}
    available.update(vi.name for vi in graph.input)
    available.add("")  # optional/omitted inputs

    nodes = list(graph.node)
    waiting: dict[str, list[int]] = defaultdict(list)
    need = [0] * len(nodes)
    ready: deque[int] = deque()

    for idx, node in enumerate(nodes):
        deps = {i for i in node.input if i and i not in available}
        need[idx] = len(deps)
        if need[idx] == 0:
            ready.append(idx)
        else:
            for tensor in deps:
                waiting[tensor].append(idx)

    ordered = []
    while ready:
        idx = ready.popleft()
        node = nodes[idx]
        ordered.append(node)
        for out in node.output:
            if not out or out in available:
                continue
            available.add(out)
            for consumer in waiting.get(out, ()):  # noqa: SIM118
                need[consumer] -= 1
                if need[consumer] == 0:
                    ready.append(consumer)

    if len(ordered) != len(nodes):
        stuck = [(n.name, [i for i in n.input if i and i not in available]) for j, n in enumerate(nodes) if need[j] > 0]
        raise RuntimeError(f"Topological sort failed (cycle or dangling input). First stuck nodes: {stuck[:5]}")

    del graph.node[:]
    graph.node.extend(ordered)


def transplant_int8_into_npu_graph(donor_onnx: Path, npu_onnx: Path, output_onnx: Path) -> int:
    """Graft the donor's int8 Linears onto the NPU-op graph (Route A).

    Quantization (``AscendQuant → MatMul-int8 → AscendDequant``) only ever
    touches ``MatMul``/``Gemm``/``Conv``; the NPU fused ops we substitute
    (``NPURmsNorm``/``NPURotaryMul``/``NPUFastGelu``/...) are all *non*-quantized
    and live in the fp16 region *between* Linears. For ``NPUGeglu``, the
    ORT-runnable donor uses the same fused ``[up, gate]`` MatMul followed by an
    exact standard-ONNX GeGLU decomposition. We therefore calibrate+quantize the ORT-runnable graph
    (the *donor*, which msModelSlim can actually run on CPU) and then move each
    int8 triplet onto the NPU graph, matching Linears by their **node-name
    stem** — node names come from the module hierarchy (``/layers.0/self_attn/
    q_proj/MatMul``) and are stable across exports, unlike the global
    ``onnx::MatMul_NNNN`` weight counters which shift when surrounding ops change.

    msModelSlim's ``reduce_redundant_quant_node`` pass merges the ``AscendQuant``
    feeding q/k/v (same activation) into one shared node; we preserve that
    sharing by emitting each donor ``AscendQuant`` once and asserting every
    Linear in the group maps to the same NPU-graph activation tensor.

    Wiring per quantized Linear ``M`` (donor int8 MatMul, name ``<stem>_quant``):

        donor:  A_donor → AscendQuant Q → M(int8) → AscendDequant D → out_donor
        npu  :  A_npu   → ...(fp16 MatMul named <stem>)... → out_npu

    becomes, in the NPU graph:

        A_npu → Q'(copy, in=A_npu) → M'(copy) → D'(copy, out=out_npu)

    The fp16 MatMul/Gemm/Conv (and its now-unused fp16 params) is removed; the
    donor's int8 params + uint64 deq_scale initializers are copied over.

    Four donor structures are handled (full-quantization runs hit all of them):

    * **Bias-less MatMul/Gemm** (the Gemma trunk q/k/v/o/gate/up/down): the
      classic ``AscendQuant → MatMul → AscendDequant`` triplet above.
    * **MatMul/Gemm with bias** (the SigLIP attention/MLP projections): msModelSlim's
      ``optimize_mm_dequant_add_subgraph`` reorders the graph to
      ``AscendQuant → MatMul(int8) → Add(int32 bias) → AscendDequant``, so the
      dequant input is the bias-``Add``, not the MatMul. We trace **through** that
      Add to the int8 MatMul and **drop** the int32-bias Add: the dequant is rewired
      straight onto the MatMul output and pointed at ``out_npu`` (the *pre-bias*
      fp16 MatMul output), so the NPU graph's existing downstream fp16 bias-``Add``
      re-applies the bias. Bias therefore stays in fp16 (negligible cost) and the
      NPU graph's Add + bias initializer are left untouched.
    * **Gemm with folded bias** (AE ``time_mlp`` / AdaRMSNorm dense):
      ``AscendQuant → Gemm(int8 weight, int32 bias) → AscendDequant``. The int32
      bias lives inside the Gemm, so it must be copied with the int8 weight.
    * **Conv** (the SigLIP ``patch_embedding``): ``AscendQuant → Conv(int8 weight
      [+ bias]) → AscendDequant``. The bias (if any) lives *inside* the Conv node, so
      the dequant input is the Conv directly; we transplant the Conv with *all* its
      initializer inputs (int8 weight + bias) and remove the fp16 Conv plus its fp16
      weight/bias.
    """
    import onnx

    LOGGER.info("Loading donor (int8) graph %s …", donor_onnx)
    donor = onnx.load(str(donor_onnx))  # external data resolved via file path
    LOGGER.info("Loading NPU-op graph %s …", npu_onnx)
    npu = onnx.load(str(npu_onnx))
    dg, ng = donor.graph, npu.graph

    donor_producer = {out: node for node in dg.node for out in node.output}
    donor_init = {init.name: init for init in dg.initializer}
    npu_node_by_name = {node.name: node for node in ng.node}
    npu_init_names = {init.name for init in ng.initializer}

    new_nodes: list = []
    new_inits: list = []
    npu_nodes_to_remove: set[str] = set()
    emitted_quant: set[str] = set()
    quant_activation: dict[str, str] = {}  # donor AscendQuant name → npu activation
    added_init: set[str] = set()
    unmatched: list[tuple[str, str]] = []
    transplanted = 0

    def _npu_match(stem_quant: str):
        stripped = stem_quant[: -len(_QUANT_NODE_SUFFIX)] if stem_quant.endswith(_QUANT_NODE_SUFFIX) else stem_quant
        for candidate in (stripped, stem_quant):
            node = npu_node_by_name.get(candidate)
            if node is not None:
                return node
        return None

    for deq in dg.node:
        if deq.op_type != "AscendDequant":
            continue
        # The dequant input is either the compute node directly (bias-less
        # MatMul/Gemm, folded-bias Gemm, Conv) or an int32 bias-Add that
        # msModelSlim moved in front of the dequant for biased MatMul.
        src = donor_producer.get(deq.input[0])
        if src is None:
            unmatched.append((deq.name, f"AscendDequant input {deq.input[0]!r} has no producer"))
            continue
        if src.op_type == "Add":
            # Biased MatMul/Gemm: Add(matmul_out, int32_bias) → compute is behind it.
            compute = donor_producer.get(src.input[0])
            if compute is None or compute.op_type not in ("MatMul", "Gemm"):
                got = compute.op_type if compute is not None else "None"
                unmatched.append((deq.name, f"bias-Add {src.name!r} parent is {got}, not MatMul/Gemm"))
                continue
        else:
            compute = src
        if compute.op_type not in ("MatMul", "Gemm", "Conv"):
            unmatched.append((deq.name, f"AscendDequant traces to {compute.op_type!r}, not MatMul/Gemm/Conv"))
            continue
        quant = donor_producer.get(compute.input[0])
        if quant is None or quant.op_type != "AscendQuant":
            unmatched.append((deq.name, f"no AscendQuant before {compute.op_type} {compute.name!r}"))
            continue

        m_npu = _npu_match(compute.name)
        if m_npu is None:
            unmatched.append((deq.name, f"no NPU-graph node matching stem of {compute.name!r}"))
            continue

        a_npu = m_npu.input[0]
        out_npu = m_npu.output[0]
        # Which donor initializers to transplant alongside the int8 weight:
        #   * Conv — bias (if any) is folded inside the node → copy every param.
        #   * biased MatMul — msModelSlim emits a separate int32 bias-Add that we
        #     bypass above, so compute.input is just [quant_out, weight]; the NPU
        #     graph's own fp16 Add re-applies the bias.
        #   * biased Gemm — msModelSlim folds the int32 bias *inside* the int8 Gemm
        #     as a third input (compute.input = [quant_out, weight, bias]) with NO
        #     separate Add. That is the canonical fused-bias int8 kernel
        #     (int8_matmul + int32_bias accumulate, then AscendDequant scales), so
        #     we KEEP the bias to reproduce it exactly — and must copy the
        #     bias_quantized initializer too, else the transplanted Gemm references
        #     a tensor that is never declared and the final topo-sort fails.
        # In every case donor_param_inputs lists the initializer inputs we both keep
        # on the transplanted node and copy into the NPU graph.
        donor_param_inputs = list(compute.input[1:])
        deq_scale = deq.input[1]

        # Shared-AscendQuant consistency: q/k/v must resolve to one activation.
        prev = quant_activation.get(quant.name)
        if prev is not None and prev != a_npu:
            unmatched.append(
                (deq.name, f"shared AscendQuant {quant.name!r} maps to conflicting activations {prev!r} vs {a_npu!r}")
            )
            continue
        quant_activation[quant.name] = a_npu

        # Emit the (possibly shared) AscendQuant once, fed by the NPU activation.
        if quant.name not in emitted_quant:
            q2 = onnx.NodeProto()
            q2.CopyFrom(quant)
            del q2.input[:]
            q2.input.append(a_npu)
            new_nodes.append(q2)
            emitted_quant.add(quant.name)

        # int8 compute node: copy the donor node verbatim. Its inputs are the
        # quant output + the int8 params (weight, and for a fused-bias Gemm the
        # int32 bias) — all of which we copy into the NPU graph below, so no input
        # dangles.
        m2 = onnx.NodeProto()
        m2.CopyFrom(compute)
        new_nodes.append(m2)

        # AscendDequant: feed it straight from the int8 compute output (bypassing
        # a separate int32 bias-Add when present) and rewire its output to the NPU
        # downstream tensor.
        d2 = onnx.NodeProto()
        d2.CopyFrom(deq)
        del d2.input[:]
        d2.input.extend([compute.output[0], deq_scale])
        del d2.output[:]
        d2.output.append(out_npu)
        new_nodes.append(d2)

        for tensor_name in (*donor_param_inputs, deq_scale):
            if tensor_name in donor_init and tensor_name not in npu_init_names and tensor_name not in added_init:
                ti = onnx.TensorProto()
                ti.CopyFrom(donor_init[tensor_name])
                new_inits.append(ti)
                added_init.add(tensor_name)

        npu_nodes_to_remove.add(m_npu.name)
        transplanted += 1

    if unmatched:
        preview = "\n  ".join(f"{name}: {why}" for name, why in unmatched[:15])
        raise RuntimeError(
            f"Route A transplant could not match {len(unmatched)} quantized Linear(s) "
            f"to the NPU graph. The NPU export likely renamed/removed a Linear that "
            f"quantization touched. First mismatches:\n  {preview}"
        )
    # Drop the fp16 MatMuls that were replaced.
    kept_nodes = [n for n in ng.node if n.name not in npu_nodes_to_remove]
    del ng.node[:]
    ng.node.extend(kept_nodes)
    ng.node.extend(new_nodes)

    # Dead-node + dead-initializer elimination. Removing a fp16 compute node can
    # orphan the small nodes that fed only its weight (e.g. a Transpose/Cast in
    # front of the fp16 weight initializer). If we only checked the compute
    # node's direct inputs, those fp16 weights would stay alive through the
    # orphaned Transpose and the file would actually GROW versus a partial-quant
    # run (the int8 weight is added but the fp16 one is never freed). Iterating
    # to a fixed point drops the orphaned Transpose, then its now-unused weight.
    graph_output_names = {o.name for o in ng.output}
    while True:
        consumed: set[str] = set(graph_output_names)
        for node in ng.node:
            consumed.update(node.input)
        dead = [n for n in ng.node if n.output and all(o not in consumed for o in n.output)]
        if not dead:
            break
        dead_ids = {id(n) for n in dead}
        survivors = [n for n in ng.node if id(n) not in dead_ids]
        del ng.node[:]
        ng.node.extend(survivors)

    # Drop every initializer no surviving node references.
    still_used: set[str] = set()
    for node in ng.node:
        still_used.update(node.input)
    ng.initializer.extend(new_inits)
    freed_bytes = sum(i.ByteSize() for i in ng.initializer if i.name not in still_used)
    kept_inits = [i for i in ng.initializer if i.name in still_used]
    n_dropped = len(ng.initializer) - len(kept_inits)
    del ng.initializer[:]
    ng.initializer.extend(kept_inits)

    topo_sort_graph(ng)

    save_onnx_external(npu, output_onnx)
    LOGGER.info(
        "Route A: transplanted %d int8 Linear(s) (%d shared AscendQuant); pruned "
        "%d unused initializer(s) (~%.2f GB of fp16 weights freed) → %s",
        transplanted,
        len(emitted_quant),
        n_dropped,
        freed_bytes / 1e9,
        output_onnx,
    )
    return transplanted


# ---------------------------------------------------------------------------
# Shared CLI
# ---------------------------------------------------------------------------


def resolve_output_path(output_arg: str | None, input_onnx: Path) -> Path:
    """Resolve --output-path, forcing a ``.onnx`` suffix.

    If the user passes a bare name (e.g. ``pi05-vlm-w8a8-all``) we append
    ``.onnx`` so the artifacts land at ``<name>.onnx`` + ``<name>.onnx.data`` —
    matching the convention downstream convert/ATC scripts expect (they
    auto-complete the ``.onnx``/``.onnx.data`` pair from the stem). Without this
    the files would be the literal ``<name>`` + ``<name>.data``.
    """
    if output_arg:
        out = Path(output_arg).expanduser().resolve()
        if out.suffix.lower() != ".onnx":
            out = out.with_name(out.name + ".onnx")
        return out
    return input_onnx.with_name(input_onnx.stem + "_w8a8.onnx")


def remove_onnx_external_pair(output_path: Path) -> None:
    """Remove an ONNX protobuf and its conventional external-data sidecar."""
    for path in (output_path, output_path.with_name(output_path.name + ".data")):
        path.unlink(missing_ok=True)


def add_common_quant_args(p: argparse.ArgumentParser) -> None:
    """Add the CLI args shared by both the VLM and AE quantization scripts."""
    p.add_argument(
        "--onnx-path", type=str, required=True, help="Input fp16/fp32 ONNX (ORT-runnable calibration donor)."
    )
    p.add_argument("--output-path", type=str, default=None, help="Output W8A8 ONNX path.")
    p.add_argument(
        "--npu-onnx-path",
        type=str,
        default=None,
        help="Route A: an NPU-op ONNX (exported with --use-npu-ops; contains "
        "NPURmsNorm/NPURotaryMul/NPUFastGelu). When given, --onnx-path is used "
        "only as the ORT-runnable calibration donor, and the donor's int8 "
        "Linears are grafted onto this graph to produce the final --output-path. "
        "Both graphs MUST be exported with identical settings (opset/dtype/inputs); "
        "only --use-npu-ops should differ.",
    )
    p.add_argument("--num-calib", type=int, default=16, help="Number of calibration batches to use (<=0 = all).")
    p.add_argument("--device", type=str, default="cpu", help="Torch device for preprocessing, e.g. cpu or cuda:0.")
    p.add_argument(
        "--disable-regex",
        type=str,
        nargs="*",
        default=None,
        help="Regexes (case-insensitive) on ONNX node names to keep in fp16. "
        "Overrides the built-in defaults when provided.",
    )
    p.add_argument(
        "--quantize-regex",
        type=str,
        nargs="*",
        default=None,
        help="Optional regex allowlist for quantized node names. Non-matching nodes stay fp16; built-in exclusions still win.",
    )
    p.add_argument(
        "--quantize-regex-expected",
        type=int,
        nargs="*",
        default=None,
        help="Expected eligible-node match count for each --quantize-regex entry.",
    )
    p.add_argument(
        "--expected-selected-nodes",
        type=int,
        default=None,
        help="Expected number of unique nodes selected before quantization.",
    )
    p.add_argument(
        "--expected-quantized-nodes",
        type=int,
        default=None,
        help="Expected number of quantized compute sites in the final ONNX.",
    )
    p.add_argument("--quant-profile-name", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--quant-profile-hash", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--quant-role", choices=("vlm", "ae"), default=None, help=argparse.SUPPRESS)
    p.add_argument("--quant-metadata-path", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--quantize-convs",
        action="store_true",
        help="Also quantize Conv nodes. Off by default to protect accuracy.",
    )
    p.add_argument(
        "--disable-index-below",
        type=int,
        default=None,
        help="Keep every MatMul whose trailing node index < N in fp16 (protects "
        "early layers in anonymous-named exports). Find N via --list-nodes.",
    )
    p.add_argument(
        "--amp-num",
        type=int,
        default=0,
        help="msModelSlim auto mixed-precision fp16 fallback layer count (ranked on real calib data).",
    )
    p.add_argument(
        "--amp-rank-samples",
        type=int,
        default=1,
        help="Number of calibration samples used to rank AMP rollback layers.",
    )
    p.add_argument(
        "--amp-scratch-dir",
        type=Path,
        default=None,
        help="Disk-backed scratch directory for AMP model and activation staging (defaults to the output directory).",
    )
    p.add_argument(
        "--smoothquant-alpha",
        type=float,
        default=None,
        help="Prepare matched donor/NPU graphs with SmoothQuant before W8A8 PTQ (disabled when omitted).",
    )
    p.add_argument(
        "--smoothquant-epsilon",
        type=float,
        default=1e-5,
        help="Positive floor used while deriving SmoothQuant channel scales.",
    )
    p.add_argument(
        "--smoothquant-output-dir",
        type=Path,
        default=None,
        help="Directory for prepared SmoothQuant graph pairs and the scale plan sidecar.",
    )
    p.add_argument(
        "--smoothquant-verify-rtol",
        type=float,
        default=2e-3,
        help="Relative tolerance for original-vs-smoothed portable donor equivalence.",
    )
    p.add_argument(
        "--smoothquant-verify-atol",
        type=float,
        default=2e-3,
        help="Absolute tolerance for original-vs-smoothed portable donor equivalence.",
    )
    p.add_argument(
        "--list-nodes", action="store_true", help="Print the quantizable node inventory and exit (no quantization)."
    )
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level.")


def list_nodes_and_exit(
    quantizable: list[tuple[str, str]],
    disable_names: list[str],
    disable_regexes: list[str],
) -> None:
    """Print the quantizable-node inventory (the ``--list-nodes`` report)."""
    by_op: dict[str, int] = {}
    for _, op in quantizable:
        by_op[op] = by_op.get(op, 0) + 1
    LOGGER.info("Quantizable nodes by op-type: %s", by_op)
    matmul_idx = sorted(i for n, op in quantizable if op == "MatMul" and (i := node_index(n)) is not None)
    if matmul_idx:
        LOGGER.info(
            "MatMul node-index range: %d … %d (use --disable-index-below to keep the early matmuls in fp16)",
            matmul_idx[0],
            matmul_idx[-1],
        )
    LOGGER.info("Disable regexes: %s", disable_regexes)
    LOGGER.info("=> %d node(s) would be KEPT in fp16:", len(disable_names))
    for name in disable_names:
        LOGGER.info("    [fp16] %s", name)
    keep_quant = [n for n, _ in quantizable if n not in set(disable_names)]
    LOGGER.info("=> %d node(s) would be QUANTIZED to int8 (showing first 40):", len(keep_quant))
    for name in keep_quant[:40]:
        LOGGER.info("    [int8] %s", name)
