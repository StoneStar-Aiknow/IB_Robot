# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Licensed under the Mulan PSL v2.
"""Versioned, reusable quantization strategies for the PI05 export pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

QUANT_PROFILE_FORMAT = "pi05-quant-profile-v1"
QUANT_METADATA_FORMAT = "pi05-quant-artifact-v1"


@dataclass(frozen=True)
class QuantizationSelector:
    name: str
    regex: str
    expected: int


@dataclass(frozen=True)
class QuantizationRoleProfile:
    enabled: bool
    selectors: tuple[QuantizationSelector, ...] = ()
    disable_regex: tuple[str, ...] = ()
    expected_selected_nodes: int | None = None
    expected_quantized_nodes: int | None = None
    quantize_convs: bool = False
    fused_geglu_donor: bool | None = None
    expected_npu_geglu_nodes: int | None = None
    expected_calibration_steps: int | None = None
    donor_dtype: str | None = None
    smoothquant_alpha: float | None = None
    smoothquant_epsilon: float | None = None
    smoothquant_verify_rtol: float | None = None
    smoothquant_verify_atol: float | None = None


@dataclass(frozen=True)
class QuantizationProfile:
    name: str
    status: str
    target_soc: str | None
    export_device: str | None
    export_dtype: str | None
    donor_device: str | None
    fast_gelu_scope: str | None
    vlm: QuantizationRoleProfile
    action_expert: QuantizationRoleProfile

    @property
    def digest(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def role(self, name: str) -> QuantizationRoleProfile:
        if name == "vlm":
            return self.vlm
        if name == "ae":
            return self.action_expert
        raise ValueError(f"Unknown PI05 quantization role: {name!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": QUANT_PROFILE_FORMAT,
            "status": self.status,
            "target_soc": self.target_soc,
            "export_device": self.export_device,
            "export_dtype": self.export_dtype,
            "donor_device": self.donor_device,
            "fast_gelu_scope": self.fast_gelu_scope,
            "vlm": _role_as_dict(self.vlm),
            "action_expert": _role_as_dict(self.action_expert),
        }


def _role_as_dict(role: QuantizationRoleProfile) -> dict[str, Any]:
    data = {
        "enabled": role.enabled,
        "selectors": [
            {"name": selector.name, "regex": selector.regex, "expected": selector.expected}
            for selector in role.selectors
        ],
        "disable_regex": list(role.disable_regex),
        "expected_selected_nodes": role.expected_selected_nodes,
        "expected_quantized_nodes": role.expected_quantized_nodes,
        "quantize_convs": role.quantize_convs,
        "fused_geglu_donor": role.fused_geglu_donor,
        "expected_npu_geglu_nodes": role.expected_npu_geglu_nodes,
    }
    if role.expected_calibration_steps is not None:
        data["expected_calibration_steps"] = role.expected_calibration_steps
    if role.donor_dtype is not None:
        data["donor_dtype"] = role.donor_dtype
    if role.smoothquant_alpha is not None:
        data["smoothquant_alpha"] = role.smoothquant_alpha
        data["smoothquant_epsilon"] = role.smoothquant_epsilon
    if role.smoothquant_verify_rtol is not None:
        data["smoothquant_verify_rtol"] = role.smoothquant_verify_rtol
        data["smoothquant_verify_atol"] = role.smoothquant_verify_atol
    return data


def _require_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    return value


def _parse_role(value: Any, where: str) -> QuantizationRoleProfile:
    data = _require_mapping(value, where)
    allowed = {
        "enabled",
        "selectors",
        "disable_regex",
        "expected_selected_nodes",
        "expected_quantized_nodes",
        "quantize_convs",
        "fused_geglu_donor",
        "expected_npu_geglu_nodes",
        "expected_calibration_steps",
        "donor_dtype",
        "smoothquant_alpha",
        "smoothquant_epsilon",
        "smoothquant_verify_rtol",
        "smoothquant_verify_atol",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{where} contains unknown fields: {unknown}")

    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{where}.enabled must be boolean")
    selectors_raw = data.get("selectors", [])
    if not isinstance(selectors_raw, list):
        raise ValueError(f"{where}.selectors must be a list")
    selectors: list[QuantizationSelector] = []
    names: set[str] = set()
    for index, raw in enumerate(selectors_raw):
        selector = _require_mapping(raw, f"{where}.selectors[{index}]")
        if set(selector) != {"name", "regex", "expected"}:
            raise ValueError(f"{where}.selectors[{index}] must contain exactly name, regex, expected")
        name, regex, expected = selector["name"], selector["regex"], selector["expected"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"{where}.selectors[{index}].name must be unique and non-empty")
        if not isinstance(regex, str) or not regex:
            raise ValueError(f"{where}.selectors[{index}].regex must be non-empty")
        if not isinstance(expected, int) or isinstance(expected, bool) or expected < 0:
            raise ValueError(f"{where}.selectors[{index}].expected must be a non-negative integer")
        names.add(name)
        selectors.append(QuantizationSelector(name, regex, expected))

    disable_regex = data.get("disable_regex", [])
    if not isinstance(disable_regex, list) or not all(isinstance(pattern, str) for pattern in disable_regex):
        raise ValueError(f"{where}.disable_regex must be a list of strings")

    def optional_count(field: str) -> int | None:
        count = data.get(field)
        if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 0):
            raise ValueError(f"{where}.{field} must be a non-negative integer or null")
        return count

    quantize_convs = data.get("quantize_convs", False)
    fused_geglu_donor = data.get("fused_geglu_donor")
    expected_npu_geglu_nodes = optional_count("expected_npu_geglu_nodes")
    expected_calibration_steps = optional_count("expected_calibration_steps")
    donor_dtype = data.get("donor_dtype")
    if donor_dtype is not None and donor_dtype not in {"fp16", "fp32", "auto"}:
        raise ValueError(f"{where}.donor_dtype must be one of fp16, fp32, or auto")
    if expected_npu_geglu_nodes is not None and expected_npu_geglu_nodes == 0:
        raise ValueError(f"{where}.expected_npu_geglu_nodes must be positive")
    if expected_calibration_steps is not None and expected_calibration_steps == 0:
        raise ValueError(f"{where}.expected_calibration_steps must be positive")
    if not isinstance(quantize_convs, bool):
        raise ValueError(f"{where}.quantize_convs must be boolean")
    if fused_geglu_donor is not None and not isinstance(fused_geglu_donor, bool):
        raise ValueError(f"{where}.fused_geglu_donor must be boolean or null")
    if enabled and not selectors:
        raise ValueError(f"{where}.selectors must not be empty when the role is enabled")
    if enabled and (data.get("expected_selected_nodes") is None or data.get("expected_quantized_nodes") is None):
        raise ValueError(f"{where} must declare expected_selected_nodes and expected_quantized_nodes when enabled")
    smoothquant_alpha = data.get("smoothquant_alpha")
    smoothquant_epsilon = data.get("smoothquant_epsilon")
    if (smoothquant_alpha is None) != (smoothquant_epsilon is None):
        raise ValueError(f"{where} must declare smoothquant_alpha and smoothquant_epsilon together")
    if smoothquant_alpha is not None:
        if (
            isinstance(smoothquant_alpha, bool)
            or not isinstance(smoothquant_alpha, int | float)
            or not math.isfinite(smoothquant_alpha)
            or not 0.0 <= smoothquant_alpha <= 1.0
        ):
            raise ValueError(f"{where}.smoothquant_alpha must be finite and in [0, 1]")
        if (
            isinstance(smoothquant_epsilon, bool)
            or not isinstance(smoothquant_epsilon, int | float)
            or not math.isfinite(smoothquant_epsilon)
            or smoothquant_epsilon <= 0.0
        ):
            raise ValueError(f"{where}.smoothquant_epsilon must be finite and positive")
    smoothquant_verify_rtol = data.get("smoothquant_verify_rtol")
    smoothquant_verify_atol = data.get("smoothquant_verify_atol")
    if (smoothquant_verify_rtol is None) != (smoothquant_verify_atol is None):
        raise ValueError(f"{where} must declare smoothquant_verify_rtol and smoothquant_verify_atol together")
    if smoothquant_verify_rtol is not None and smoothquant_alpha is None:
        raise ValueError(f"{where} SmoothQuant verification tolerances require smoothquant_alpha and epsilon")
    for field, value in (
        ("smoothquant_verify_rtol", smoothquant_verify_rtol),
        ("smoothquant_verify_atol", smoothquant_verify_atol),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0.0
        ):
            raise ValueError(f"{where}.{field} must be finite and non-negative")

    return QuantizationRoleProfile(
        enabled=enabled,
        selectors=tuple(selectors),
        disable_regex=tuple(disable_regex),
        expected_selected_nodes=optional_count("expected_selected_nodes"),
        expected_quantized_nodes=optional_count("expected_quantized_nodes"),
        quantize_convs=quantize_convs,
        fused_geglu_donor=fused_geglu_donor,
        expected_npu_geglu_nodes=expected_npu_geglu_nodes,
        expected_calibration_steps=expected_calibration_steps,
        donor_dtype=donor_dtype,
        smoothquant_alpha=float(smoothquant_alpha) if smoothquant_alpha is not None else None,
        smoothquant_epsilon=float(smoothquant_epsilon) if smoothquant_epsilon is not None else None,
        smoothquant_verify_rtol=float(smoothquant_verify_rtol) if smoothquant_verify_rtol is not None else None,
        smoothquant_verify_atol=float(smoothquant_verify_atol) if smoothquant_verify_atol is not None else None,
    )


def parse_quantization_profile(name: str, value: Any) -> QuantizationProfile:
    data = _require_mapping(value, f"quantization profile {name!r}")
    allowed = {
        "name",
        "format",
        "status",
        "target_soc",
        "export_device",
        "export_dtype",
        "donor_device",
        "fast_gelu_scope",
        "vlm",
        "action_expert",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"quantization profile {name!r} contains unknown fields: {unknown}")
    declared_name = data.get("name")
    if declared_name is not None and declared_name != name:
        raise ValueError(f"quantization profile file/name mismatch: {declared_name!r} != {name!r}")
    if data.get("format") != QUANT_PROFILE_FORMAT:
        raise ValueError(f"quantization profile {name!r} must use format {QUANT_PROFILE_FORMAT!r}")
    status = data.get("status", "experimental")
    target_soc = data.get("target_soc")
    export_device = data.get("export_device")
    export_dtype = data.get("export_dtype")
    donor_device = data.get("donor_device")
    fast_gelu_scope = data.get("fast_gelu_scope")
    if not isinstance(status, str) or not status:
        raise ValueError(f"quantization profile {name!r}.status must be non-empty")
    for field, value in (
        ("target_soc", target_soc),
        ("export_device", export_device),
        ("export_dtype", export_dtype),
        ("donor_device", donor_device),
    ):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"quantization profile {name!r}.{field} must be a string or null")
    if export_dtype is not None and export_dtype not in {"fp16", "fp32", "auto"}:
        raise ValueError(f"quantization profile {name!r}.export_dtype must be one of fp16, fp32, or auto")
    if fast_gelu_scope is not None and fast_gelu_scope not in {"none", "all", "vision", "vlm-text", "ae"}:
        raise ValueError(
            f"quantization profile {name!r}.fast_gelu_scope must be one of none, all, vision, vlm-text, or ae"
        )
    vlm = _parse_role(data.get("vlm", {"enabled": False}), f"quantization profile {name!r}.vlm")
    action_expert = _parse_role(
        data.get("action_expert", {"enabled": False}),
        f"quantization profile {name!r}.action_expert",
    )
    if vlm.enabled and fast_gelu_scope in {"vision", "vlm-text", "all"}:
        raise ValueError(f"quantization profile {name!r} cannot combine VLM W8A8 with scope {fast_gelu_scope!r}")
    if action_expert.enabled and fast_gelu_scope in {"ae", "all"}:
        raise ValueError(
            f"quantization profile {name!r} cannot combine Action Expert W8A8 with scope {fast_gelu_scope!r}"
        )
    if vlm.fused_geglu_donor is True and vlm.expected_npu_geglu_nodes is None:
        raise ValueError(f"quantization profile {name!r} VLM fused_geglu_donor requires expected_npu_geglu_nodes")
    if action_expert.fused_geglu_donor is True and action_expert.expected_npu_geglu_nodes is None:
        raise ValueError(
            f"quantization profile {name!r} Action Expert fused_geglu_donor requires expected_npu_geglu_nodes"
        )
    if vlm.expected_calibration_steps is not None:
        raise ValueError(f"quantization profile {name!r} expected_calibration_steps applies only to action_expert")
    return QuantizationProfile(
        name=name,
        status=status,
        target_soc=target_soc,
        export_device=export_device,
        export_dtype=export_dtype,
        donor_device=donor_device,
        fast_gelu_scope=fast_gelu_scope,
        vlm=vlm,
        action_expert=action_expert,
    )


@lru_cache(maxsize=1)
def _bundled_quantization_profiles() -> tuple[QuantizationProfile, ...]:
    directory = files("model_utils.pi05_export").joinpath("quantization_profiles")
    profiles: list[QuantizationProfile] = []
    names: set[str] = set()
    for resource in sorted(directory.iterdir(), key=lambda item: item.name):
        if not resource.name.endswith((".yaml", ".yml")):
            continue
        data = yaml.safe_load(resource.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"bundled quantization profile {resource.name!r} must be a mapping")
        name = data.get("name", resource.name.rsplit(".", 1)[0])
        if not isinstance(name, str) or not name:
            raise ValueError(f"bundled quantization profile {resource.name!r} has an invalid name")
        if name in names:
            raise ValueError(f"duplicate bundled quantization profile name: {name!r}")
        names.add(name)
        profiles.append(parse_quantization_profile(name, data))
    return tuple(profiles)


def bundled_quantization_profiles() -> dict[str, QuantizationProfile]:
    return {profile.name: profile for profile in _bundled_quantization_profiles()}


def available_quantization_profiles(config: dict[str, Any]) -> dict[str, QuantizationProfile]:
    profiles = bundled_quantization_profiles()
    custom = config.get("quantization_profiles", {})
    if not isinstance(custom, dict):
        raise ValueError("quantization_profiles must be a mapping")
    collisions = sorted(set(custom) & set(profiles))
    if collisions:
        raise ValueError(f"custom quantization profiles cannot override bundled profiles: {collisions}")
    for name, value in custom.items():
        if not isinstance(name, str) or not name:
            raise ValueError("quantization profile names must be non-empty strings")
        profiles[name] = parse_quantization_profile(name, value)
    return profiles


def resolve_quantization_profile(name: str | None, config: dict[str, Any]) -> QuantizationProfile | None:
    if not name:
        return None
    profiles = available_quantization_profiles(config)
    try:
        return profiles[name]
    except KeyError as exc:
        raise ValueError(f"quantization profile {name!r} not found") from exc


def _stat_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _content_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": digest.hexdigest()}


def artifact_identity(path: Path) -> dict[str, Any]:
    files = [_content_identity(path)]
    sidecar = path.with_name(path.name + ".data")
    if sidecar.is_file():
        files.append(_content_identity(sidecar))
    return {"files": files}


def policy_identity(policy_path: Path) -> dict[str, Any]:
    files = []
    for name in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        path = policy_path / name
        if path.is_file():
            files.append(_stat_identity(path))
    return {"root": str(policy_path.resolve()), "files": files}


def metadata_path(output_onnx: Path) -> Path:
    return Path(f"{output_onnx}.quant.json")


def _names_digest(names: list[str]) -> str:
    payload = json.dumps(sorted(names), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def write_quantization_metadata(
    *,
    path: Path,
    profile_name: str,
    profile_hash: str,
    role: str,
    policy_path: Path,
    donor_onnx: Path,
    npu_onnx: Path | None,
    output_onnx: Path,
    selected_nodes: list[str],
    actual_quantized_nodes: int,
) -> None:
    data = {
        "format": QUANT_METADATA_FORMAT,
        "profile": profile_name,
        "profile_hash": profile_hash,
        "role": role,
        "policy": policy_identity(policy_path),
        "donor_onnx": artifact_identity(donor_onnx),
        "npu_onnx": artifact_identity(npu_onnx) if npu_onnx else None,
        "selected_nodes": len(selected_nodes),
        "selected_nodes_hash": _names_digest(selected_nodes),
        "actual_quantized_nodes": actual_quantized_nodes,
        "output_onnx": artifact_identity(output_onnx),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_quantization_metadata(
    *,
    path: Path,
    profile: QuantizationProfile,
    role: str,
    policy_path: Path,
    donor_onnx: Path,
    npu_onnx: Path | None,
    output_onnx: Path,
) -> None:
    if not path.is_file():
        raise ValueError(f"quantized ONNX metadata is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "format": QUANT_METADATA_FORMAT,
        "profile": profile.name,
        "profile_hash": profile.digest,
        "role": role,
        "policy": policy_identity(policy_path),
        "donor_onnx": artifact_identity(donor_onnx),
        "npu_onnx": artifact_identity(npu_onnx) if npu_onnx else None,
        "output_onnx": artifact_identity(output_onnx),
    }
    mismatches = [field for field, value in expected.items() if data.get(field) != value]
    role_profile = profile.role(role)
    if (
        role_profile.expected_quantized_nodes is not None
        and data.get("actual_quantized_nodes") != role_profile.expected_quantized_nodes
    ):
        mismatches.append("actual_quantized_nodes")
    if mismatches:
        raise ValueError(
            f"quantized ONNX metadata does not match the current profile/source ({', '.join(mismatches)}); "
            f"rerun the {role}_quant step"
        )
