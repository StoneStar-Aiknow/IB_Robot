#!/usr/bin/env python
# Copyright (c) 2026 Syslong Technology Co., Ltd. All Rights Reserved.
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
from pathlib import Path

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
        already_external = kwargs.get("save_as_external_data", False)
        if not already_external:
            try:
                too_big = proto.ByteSize() > _PROTOBUF_INLINE_LIMIT
            except Exception:  # noqa: BLE001
                # ByteSize() raises EncodeError (>2 GB) on some protobuf builds —
                # that overflow IS the signal that we must use external data.
                too_big = True
            if too_big and isinstance(f, (str, Path)):  # noqa: UP038
                location = Path(f).name + ".data"
                kwargs.update(
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=location,
                    size_threshold=1024,
                )
                LOGGER.info(
                    "Large model (>%.1f GB inline limit): saving %s with external data → %s",
                    _PROTOBUF_INLINE_LIMIT / 1e9,
                    Path(f).name,
                    location,
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
    delete any stale ``.onnx`` + ``.data`` *before* writing. The in-memory model
    is unaffected: ``onnx.load`` already pulled every external tensor into
    ``raw_data`` before we get here.
    """
    import onnx

    output_path = Path(output_path)
    if data_name is None:
        data_name = output_path.name + ".data"
    for stale in (output_path, output_path.with_name(data_name)):
        try:  # noqa: SIM105
            stale.unlink()
        except FileNotFoundError:
            pass
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
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


def install_msmodelslim_amp_patch(calib_sample: list[np.ndarray]) -> None:
    """Make ``amp_num`` rollback rank layers on REAL calib data, not random noise.

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

    We replace ``gen_model_inputs`` with one that returns our first real
    calibration sample, bound positionally to the model inputs with their
    declared dtypes (bool included). This makes ``--amp-num`` meaningful and
    safe. The error session itself runs on the ORT-runnable donor (no NPU ops),
    and the AE donor is well under 2 GB so its ``SerializeToString`` is fine; if
    a future >2 GB donor needs amp, also wrap
    ``get_session_for_intermediate_output`` to spill to a temp file.
    """
    install_onnx_mapping_shim()

    from msmodelslim.onnx.post_training_quant.label_free import rollback_quant_nodes as _rb

    if getattr(_rb, "_pi05_amp_patched", False):
        return

    def _patched_gen_model_inputs(inputs, quant_config=None):  # noqa: ANN001
        if len(calib_sample) != len(inputs):
            raise ValueError(
                f"amp rollback: calib sample has {len(calib_sample)} array(s) but model has {len(inputs)} input(s)."
            )
        return {inp.name: arr for inp, arr in zip(inputs, calib_sample, strict=False)}

    _rb.gen_model_inputs = _patched_gen_model_inputs
    _rb._pi05_amp_patched = True  # type: ignore[attr-defined]
    LOGGER.info(
        "Installed msModelSlim amp-rollback patch: layer-sensitivity ranking uses "
        "REAL calibration data (bool-safe) instead of random noise."
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

