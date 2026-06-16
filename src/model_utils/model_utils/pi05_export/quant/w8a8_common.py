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
    npu_graph: Path | None = None,
) -> None:
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
        install_msmodelslim_amp_patch(calib_data[0])
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
        transplant_int8_into_npu_graph(donor_onnx, npu_graph, output_onnx)
        # Re-pin: the transplant renamed every dequant output to the NPU graph's
        # downstream tensor, so the fp16 value_info must be re-declared on those.
        fix_ascend_dequant_output_dtype(output_onnx)
        LOGGER.info("Final NPU + W8A8 ONNX written to %s", output_onnx)


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


def transplant_int8_into_npu_graph(donor_onnx: Path, npu_onnx: Path, output_onnx: Path) -> None:
    """Graft the donor's int8 Linears onto the NPU-op graph (Route A).

    Quantization (``AscendQuant → MatMul-int8 → AscendDequant``) only ever
    touches ``MatMul``/``Gemm``/``Conv``; the NPU fused ops we substitute
    (``NPURmsNorm``/``NPURotaryMul``/``NPUFastGelu``/...) are all *non*-quantized
    and live in the fp16 region *between* Linears. So the two graphs are
    identical in their quantizable subgraph and differ only in the norm/rope/
    activation islands. We therefore calibrate+quantize the ORT-runnable graph
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

    The fp16 MatMul (and its now-unused fp16 weight) are removed; the int8
    weight + uint64 deq_scale initializers are copied over.

    Three donor structures are handled (full-quantization runs hit all of them):

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
        # MatMul/Gemm, or Conv with its bias folded inside) or an int32 bias-Add
        # that msModelSlim moved in front of the dequant for biased MatMul/Gemm.
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
        # Conv keeps its bias inside the node → copy every initializer input;
        # MatMul/Gemm carry only the int8 weight (bias is the dropped int32 Add).
        if compute.op_type == "Conv":
            donor_param_inputs = [i for i in compute.input[1:]]
        else:
            donor_param_inputs = [compute.input[1]]
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

        # int8 compute node: keep donor inputs (quant output + int8 params) / output.
        m2 = onnx.NodeProto()
        m2.CopyFrom(compute)
        new_nodes.append(m2)

        # AscendDequant: feed it straight from the int8 compute output (bypassing
        # the dropped int32 bias-Add) and rewire its output to the NPU downstream
        # tensor. For biased MatMul/Gemm the NPU graph's own fp16 Add re-adds bias.
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
