"""Tool-only ACL ABI inspection helpers.

This module is used by exporters and packagers when an OM ABI sidecar is
missing.  Its optional ACL configuration path applies only to that offline
inspection process and is never part of a runtime manifest or profile.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

_ACL_DTYPES = {
    0: "float32",
    1: "float16",
    2: "int8",
    3: "int32",
    4: "uint8",
    6: "int16",
    7: "uint16",
    8: "uint32",
    9: "int64",
    10: "uint64",
    11: "float64",
    12: "bool",
    27: "bfloat16",
}


def write_acl_om_abi(
    om_path: str | Path,
    output_path: str | Path,
    *,
    device_id: int = 0,
    acl_config_path: str | None = None,
) -> Path:
    """Inspect one OM with ACL and write its runtime tensor ABI.

    ``acl_config_path`` is deliberately limited to this tool-only operation.
    Runtime initialization uses the injected ACL process provider and always
    calls the vendor default ``acl.init()``.
    """

    model_path = Path(om_path).expanduser().resolve(strict=True)
    destination = Path(output_path).expanduser().resolve()
    try:
        acl = importlib.import_module("acl")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ACL Python runtime is unavailable; source the CANN environment or provide a pre-generated OM ABI sidecar"
        ) from exc
    model_id = None
    descriptor = None
    context = None
    initialized = False
    device_set = False
    pending_error: Exception | None = None
    try:
        _acl_check(acl.init(acl_config_path) if acl_config_path else acl.init(), "acl.init")
        initialized = True
        _acl_check(acl.rt.set_device(device_id), "acl.rt.set_device")
        device_set = True
        context = _acl_result(acl.rt.create_context(device_id), "acl.rt.create_context")
        _acl_check(acl.rt.set_context(context), "acl.rt.set_context")
        model_id = _acl_result(acl.mdl.load_from_file(str(model_path)), "acl.mdl.load_from_file")
        descriptor = acl.mdl.create_desc()
        if descriptor is None:
            raise RuntimeError("acl.mdl.create_desc returned no descriptor")
        _acl_check(acl.mdl.get_desc(descriptor, model_id), "acl.mdl.get_desc")
        value = {
            "inputs": _acl_tensors(acl, descriptor, "input"),
            "outputs": _acl_tensors(acl, descriptor, "output"),
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        _validate_written_abi(destination)
        return destination
    except Exception as exc:
        pending_error = exc
        raise
    finally:
        cleanup_errors: list[str] = []
        if descriptor is not None:
            try:
                acl.mdl.destroy_desc(descriptor)
            except Exception as exc:
                cleanup_errors.append(f"acl.mdl.destroy_desc: {exc}")
        if model_id is not None:
            try:
                acl.mdl.unload(model_id)
            except Exception as exc:
                cleanup_errors.append(f"acl.mdl.unload: {exc}")
        if context is not None:
            try:
                acl.rt.destroy_context(context)
            except Exception as exc:
                cleanup_errors.append(f"acl.rt.destroy_context: {exc}")
        if device_set:
            try:
                acl.rt.reset_device(device_id)
            except Exception as exc:
                cleanup_errors.append(f"acl.rt.reset_device: {exc}")
        if initialized:
            try:
                acl.finalize()
            except Exception as exc:
                cleanup_errors.append(f"acl.finalize: {exc}")
        if cleanup_errors and pending_error is None:
            raise RuntimeError("; ".join(cleanup_errors))


def _acl_tensors(acl: Any, descriptor: object, direction: str) -> list[dict[str, object]]:
    count = getattr(acl.mdl, f"get_num_{direction}s")(descriptor)
    tensors = []
    for index in range(count):
        name = getattr(acl.mdl, f"get_{direction}_name_by_index")(descriptor, index)
        dims = _acl_result(
            getattr(acl.mdl, f"get_{direction}_dims")(descriptor, index),
            f"ACL {direction} dims",
        )
        shape = dims.get("dims") if isinstance(dims, dict) else dims
        if not isinstance(shape, list | tuple):
            raise ValueError(f"ACL {direction} {name!r} returned invalid shape {shape!r}")
        dtype_code = getattr(acl.mdl, f"get_{direction}_data_type")(descriptor, index)
        try:
            dtype = _ACL_DTYPES[dtype_code]
        except KeyError as exc:
            raise ValueError(f"Unsupported ACL dtype code {dtype_code!r} for {direction} {name!r}") from exc
        tensors.append({"name": name, "index": index, "dtype": dtype, "shape": list(shape)})
    return tensors


def _acl_result(value: object, operation: str) -> object:
    if isinstance(value, tuple) and len(value) == 2:
        result, status = value
        _acl_check(status, operation)
        return result
    return value


def _acl_check(status: object, operation: str) -> None:
    if status not in (None, 0):
        raise RuntimeError(f"{operation} failed with ACL status {status}")


def _validate_written_abi(path: Path) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Runtime ABI must be a JSON object: {path}")
    for direction in ("inputs", "outputs"):
        tensors = value.get(direction)
        if not isinstance(tensors, list) or not tensors:
            raise ValueError(f"Runtime ABI {direction} must be a non-empty list: {path}")
        indices = []
        for tensor in tensors:
            if not isinstance(tensor, dict):
                raise ValueError(f"Runtime ABI {direction} entries must be objects: {path}")
            if not isinstance(tensor.get("name"), str) or not tensor["name"]:
                raise ValueError(f"Runtime ABI {direction} entries require a name: {path}")
            if tensor.get("dtype") not in _ACL_DTYPES.values():
                raise ValueError(f"Runtime ABI {direction} contains an unsupported dtype: {path}")
            index = tensor.get("index")
            shape = tensor.get("shape")
            if type(index) is not int or index < 0 or not isinstance(shape, list) or not shape:
                raise ValueError(f"Runtime ABI {direction} contains an invalid tensor slot: {path}")
            indices.append(index)
        if len(indices) != len(set(indices)):
            raise ValueError(f"Runtime ABI {direction} contains duplicate indices: {path}")
        if direction == "inputs" and sorted(indices) != list(range(len(indices))):
            raise ValueError(f"Runtime ABI inputs indices must be contiguous from zero: {path}")


__all__ = ["write_acl_om_abi"]
