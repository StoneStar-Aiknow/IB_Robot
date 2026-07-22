from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def _canonical_action_chunk(action: Any) -> np.ndarray:
    try:
        array = np.asarray(action, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("action chunks must be non-empty finite numeric 1D/2D arrays") from exc
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0 or not np.isfinite(array).all():
        raise ValueError("action chunks must be non-empty finite numeric 1D/2D arrays")
    return array


def _non_negative_integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _validate_prediction_document(document: Any, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(document, dict):
        raise ValueError(f"{label} prediction document must be an object")
    metadata = document.get("metadata")
    frames = document.get("frames")
    if not isinstance(metadata, dict) or not isinstance(frames, list):
        raise ValueError(f"{label} prediction document requires object metadata and a frame list")
    for field_name in ("planned_frame_count", "selected_frame_count", "successful_frame_count"):
        if not _non_negative_integer(metadata.get(field_name)):
            raise ValueError(f"{label} {field_name} is missing or invalid")
    if type(metadata.get("complete")) is not bool:
        raise ValueError(f"{label} complete is missing or invalid")
    if metadata["selected_frame_count"] != len(frames):
        raise ValueError(f"{label} selected_frame_count does not match recorded frames")

    successful = 0
    action_chunk_shape: tuple[int, int] | None = None
    join_values = {"frame_index": set(), "sample_timestamp_ns": set()}
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"{label} frame {index} must be an object")
        if type(frame.get("success")) is not bool:
            raise ValueError(f"{label} frame {index} success must be boolean")
        for field_name in ("frame_index", "sample_timestamp_ns"):
            value = frame.get(field_name)
            if not _non_negative_integer(value):
                raise ValueError(f"{label} frame {index} {field_name} must be a non-negative integer")
            if value in join_values[field_name]:
                raise ValueError(f"duplicate {field_name} in {label} prediction frames: {value}")
            join_values[field_name].add(value)
        if not frame["success"] or frame.get("action") is None:
            continue
        action = _canonical_action_chunk(frame["action"])
        chunk_size = frame.get("chunk_size")
        if type(chunk_size) is not int or chunk_size <= 0 or chunk_size != action.shape[0]:
            raise ValueError(f"{label} frame {index} chunk_size does not match its action payload")
        if action_chunk_shape is None:
            action_chunk_shape = action.shape
        elif action.shape != action_chunk_shape:
            raise ValueError(
                f"{label} action chunk shape varies across frames: "
                f"expected {action_chunk_shape}, frame {index} has {action.shape}"
            )
        successful += 1
    if metadata["successful_frame_count"] != successful:
        raise ValueError(f"{label} successful_frame_count does not match successful action frames")
    if metadata["planned_frame_count"] < metadata["selected_frame_count"]:
        raise ValueError(f"{label} planned_frame_count is smaller than selected_frame_count")
    expected_complete = (
        metadata["planned_frame_count"] == len(frames)
        and successful == len(frames)
        and metadata.get("deployment_identity_consistent") is True
    )
    if metadata["complete"] and not expected_complete:
        reasons = _incomplete_reasons(document, label)
        raise ValueError(
            f"{label} completeness metadata is inconsistent" + (f": {', '.join(reasons)}" if reasons else "")
        )
    return metadata, frames


def _successful_actions(document: dict[str, Any], key: str) -> dict[Any, np.ndarray]:
    actions: dict[Any, np.ndarray] = {}
    for frame in document.get("frames", []):
        if not frame.get("success") or frame.get("action") is None:
            continue
        join_value = frame["frame_index"] if key == "frame_index" else frame["sample_timestamp_ns"]
        actions[join_value] = _canonical_action_chunk(frame["action"])
    return actions


def _successful_frames(document: dict[str, Any], key: str) -> dict[Any, dict[str, Any]]:
    frames: dict[Any, dict[str, Any]] = {}
    for frame in document.get("frames", []):
        if not frame.get("success") or frame.get("action") is None:
            continue
        join_value = frame["frame_index"] if key == "frame_index" else frame["sample_timestamp_ns"]
        frames[join_value] = frame
    return frames


def _incomplete_reasons(document: dict[str, Any], label: str) -> list[str]:
    metadata = document.get("metadata", {})
    frames = document.get("frames", [])
    planned = metadata.get("planned_frame_count", len(frames))
    successful = sum(bool(frame.get("success")) and frame.get("action") is not None for frame in frames)
    reasons = []
    if len(frames) != planned:
        reasons.append(f"{label} recorded {len(frames)}/{planned} planned frames")
    if successful != planned:
        reasons.append(f"{label} has {successful}/{planned} successful action frames")
    if metadata.get("deployment_identity_consistent") is not True:
        reasons.append(f"{label} deployment identity proof is missing or inconsistent")
    frame_fingerprints = [
        frame.get("deployment_fingerprint", "")
        for frame in frames
        if frame.get("success") and frame.get("action") is not None
    ]
    if not frame_fingerprints or not all(frame_fingerprints) or len(set(frame_fingerprints)) != 1:
        reasons.append(f"{label} deployment identity is missing or varies across successful frames")
    return reasons


def _cosine_similarity(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    reference_scale = float(np.max(np.abs(reference_flat)))
    candidate_scale = float(np.max(np.abs(candidate_flat)))
    if reference_scale == 0.0 or candidate_scale == 0.0:
        return None
    reference_scaled = reference_flat / reference_scale
    candidate_scaled = candidate_flat / candidate_scale
    reference_norm = float(np.linalg.norm(reference_scaled))
    candidate_norm = float(np.linalg.norm(candidate_scaled))
    if reference_norm == 0.0 or candidate_norm == 0.0:
        return None
    value = float(np.dot(reference_scaled, candidate_scaled) / (reference_norm * candidate_norm))
    if not math.isfinite(value):
        raise ValueError("cosine similarity produced a non-finite value")
    return value


def _stable_mean(values: np.ndarray, *, axis: int | None = None) -> np.ndarray | float:
    scale = np.max(np.abs(values), axis=axis, keepdims=True)
    safe_scale = np.where(scale == 0, 1.0, scale)
    mean = np.mean(values / safe_scale, axis=axis, keepdims=True) * scale
    squeezed = np.squeeze(mean, axis=axis) if axis is not None else np.squeeze(mean)
    if not np.isfinite(squeezed).all():
        raise ValueError("metric mean produced a non-finite value")
    return float(squeezed) if np.ndim(squeezed) == 0 else squeezed


def _stable_rmse(values: np.ndarray) -> float:
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return 0.0
    value = scale * math.sqrt(float(np.mean(np.square(values / scale))))
    if not math.isfinite(value):
        raise ValueError("metric RMSE produced a non-finite value")
    return value


def _cosine_summary(values: list[float | None]) -> tuple[float | None, float | None, int]:
    valid = [value for value in values if value is not None]
    if not valid:
        return None, None, len(values)
    return float(np.mean(valid)), float(np.min(valid)), len(values) - len(valid)


def compare_prediction_documents(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    join_key: str = "frame_index",
    allow_incompatible: bool = False,
    action_mean: np.ndarray | None = None,
    action_std: np.ndarray | None = None,
) -> dict[str, Any]:
    if join_key not in {"frame_index", "sample_timestamp_ns"}:
        raise ValueError(f"unsupported prediction join key: {join_key}")
    ref_meta, _ = _validate_prediction_document(reference, "reference")
    cand_meta, _ = _validate_prediction_document(candidate, "candidate")
    compatibility_errors: list[str] = []
    if not isinstance(ref_meta.get("contract_fingerprint"), str) or not ref_meta["contract_fingerprint"]:
        compatibility_errors.append("reference contract_fingerprint is missing")
    if not isinstance(cand_meta.get("contract_fingerprint"), str) or not cand_meta["contract_fingerprint"]:
        compatibility_errors.append("candidate contract_fingerprint is missing")
    if ref_meta.get("contract_fingerprint") != cand_meta.get("contract_fingerprint"):
        compatibility_errors.append("contract_fingerprint differs")
    for key in (
        "bag_digest",
        "timestamp_policy",
        "frame_stride",
        "policy_state_mode",
        "replay_timestamp_mode",
        "policy_bundle_digest",
    ):
        if not ref_meta.get(key) or not cand_meta.get(key):
            compatibility_errors.append(f"{key} is missing")
        elif ref_meta[key] != cand_meta[key]:
            compatibility_errors.append(f"{key} differs")
    compatibility_errors.extend(_incomplete_reasons(reference, "reference"))
    compatibility_errors.extend(_incomplete_reasons(candidate, "candidate"))
    if compatibility_errors and not allow_incompatible:
        raise ValueError("incompatible prediction files: " + ", ".join(compatibility_errors))

    ref_actions = _successful_actions(reference, join_key)
    cand_actions = _successful_actions(candidate, join_key)
    ref_frames = _successful_frames(reference, join_key)
    cand_frames = _successful_frames(candidate, join_key)
    for label, metadata, actions in (
        ("reference", ref_meta, ref_actions),
        ("candidate", cand_meta, cand_actions),
    ):
        action_dim = metadata.get("action_dim")
        if type(action_dim) is not int or action_dim <= 0:
            raise ValueError(f"{label} action_dim is missing or invalid")
        if any(action.shape[-1] != action_dim for action in actions.values()):
            raise ValueError(f"{label} action payload does not match action_dim {action_dim}")
        frame_fingerprints = {
            frame["deployment_fingerprint"]
            for frame in (reference if label == "reference" else candidate)["frames"]
            if frame["success"] and frame.get("action") is not None
        }
        if metadata.get("deployment_fingerprint") not in frame_fingerprints or len(frame_fingerprints) != 1:
            raise ValueError(f"{label} metadata deployment fingerprint does not match successful frames")
        listed_fingerprints = metadata.get("deployment_fingerprints")
        if listed_fingerprints is not None and listed_fingerprints != sorted(frame_fingerprints):
            raise ValueError(f"{label} deployment_fingerprints metadata is inconsistent")
    if ref_meta["action_dim"] != cand_meta["action_dim"]:
        raise ValueError("incompatible prediction files: action_dim differs")
    if set(ref_actions) != set(cand_actions):
        compatibility_errors.append("successful frame sets differ")
        if not allow_incompatible:
            raise ValueError("incompatible prediction files: " + ", ".join(compatibility_errors))
    common = sorted(set(ref_actions) & set(cand_actions))
    if not common:
        raise ValueError("no matching successful frames to compare")
    if join_key == "frame_index":
        mismatched_timestamps = [
            key
            for key in common
            if ref_frames[key].get("sample_timestamp_ns") != cand_frames[key].get("sample_timestamp_ns")
        ]
        if mismatched_timestamps:
            compatibility_errors.append(f"sample_timestamp_ns differs for frames: {mismatched_timestamps}")
            if not allow_incompatible:
                raise ValueError("incompatible prediction files: " + ", ".join(compatibility_errors))

    comparable_actions: list[tuple[np.ndarray, np.ndarray]] = []
    mismatched_shapes: list[Any] = []
    for frame_key in common:
        ref_arr = ref_actions[frame_key]
        cand_arr = cand_actions[frame_key]
        if ref_arr.shape != cand_arr.shape:
            mismatched_shapes.append(frame_key)
            continue
        comparable_actions.append((ref_arr, cand_arr))
    if mismatched_shapes:
        compatibility_errors.append(f"action shapes differ for frames: {mismatched_shapes}")
        raise ValueError("incompatible prediction files: " + ", ".join(compatibility_errors))
    if not comparable_actions:
        raise ValueError(f"no comparable frames after shape checks; mismatched={mismatched_shapes}")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            diffs = [
                np.abs(candidate_action - reference_action) for reference_action, candidate_action in comparable_actions
            ]
    except FloatingPointError as exc:
        raise ValueError("action metric arithmetic produced non-finite values") from exc
    flat = np.concatenate([diff.reshape(-1) for diff in diffs])
    per_dim_stack = np.concatenate([diff.reshape(-1, diff.shape[-1]) for diff in diffs if diff.ndim >= 1], axis=0)
    reference_stack = np.concatenate([reference_action for reference_action, _ in comparable_actions], axis=0)
    candidate_stack = np.concatenate([candidate_action for _, candidate_action in comparable_actions], axis=0)
    frame_cosines = [
        _cosine_similarity(reference_action, candidate_action)
        for reference_action, candidate_action in comparable_actions
    ]
    first_step_cosines = [
        _cosine_similarity(reference_action[0], candidate_action[0])
        for reference_action, candidate_action in comparable_actions
    ]
    mean_frame_cosine, min_frame_cosine, undefined_frame_cosines = _cosine_summary(frame_cosines)
    mean_first_step_cosine, min_first_step_cosine, undefined_first_step_cosines = _cosine_summary(first_step_cosines)
    result = {
        "reference_backend": ref_meta.get("backend", {}).get("name"),
        "candidate_backend": cand_meta.get("backend", {}).get("name"),
        "join_key": join_key,
        "matched_frames": len(common),
        "compared_frames": len(diffs),
        "missing_in_candidate": len(set(ref_actions) - set(cand_actions)),
        "extra_in_candidate": len(set(cand_actions) - set(ref_actions)),
        "mismatched_shape_frames": mismatched_shapes,
        "mae": _stable_mean(flat),
        "max_error": float(np.max(flat)),
        "rmse": _stable_rmse(flat),
        "cosine_similarity": _cosine_similarity(reference_stack, candidate_stack),
        "mean_frame_cosine_similarity": mean_frame_cosine,
        "min_frame_cosine_similarity": min_frame_cosine,
        "undefined_frame_cosine_count": undefined_frame_cosines,
        "mean_first_step_cosine_similarity": mean_first_step_cosine,
        "min_first_step_cosine_similarity": min_first_step_cosine,
        "undefined_first_step_cosine_count": undefined_first_step_cosines,
        "per_action_dim_cosine_similarity": [
            _cosine_similarity(reference_stack[:, dimension], candidate_stack[:, dimension])
            for dimension in range(reference_stack.shape[-1])
        ],
        "per_action_dim_mae": np.asarray(_stable_mean(per_dim_stack, axis=0), dtype=float).tolist(),
        "per_action_dim_max_error": np.max(per_dim_stack, axis=0).astype(float).tolist(),
        "compatibility_warnings": compatibility_errors,
    }
    if (action_mean is None) != (action_std is None):
        raise ValueError("action normalization requires both mean and std")
    if action_mean is not None and action_std is not None:
        mean = np.asarray(action_mean, dtype=np.float64)
        std = np.asarray(action_std, dtype=np.float64)
        if (
            mean.shape != (reference_stack.shape[-1],)
            or std.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std <= 0)
        ):
            raise ValueError(
                f"action normalization stats must have shape {(reference_stack.shape[-1],)}, finite values, "
                "and positive std"
            )
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                normalized_pairs = [
                    ((reference_action - mean) / std, (candidate_action - mean) / std)
                    for reference_action, candidate_action in comparable_actions
                ]
        except FloatingPointError as exc:
            raise ValueError("action normalization produced non-finite values") from exc
        normalized_reference = np.concatenate([reference_action for reference_action, _ in normalized_pairs], axis=0)
        normalized_candidate = np.concatenate([candidate_action for _, candidate_action in normalized_pairs], axis=0)
        normalized_frame_cosines = [
            _cosine_similarity(reference_action, candidate_action)
            for reference_action, candidate_action in normalized_pairs
        ]
        normalized_first_step_cosines = [
            _cosine_similarity(reference_action[0], candidate_action[0])
            for reference_action, candidate_action in normalized_pairs
        ]
        normalized_frame_mean, normalized_frame_min, normalized_frame_undefined = _cosine_summary(
            normalized_frame_cosines
        )
        normalized_first_mean, normalized_first_min, normalized_first_undefined = _cosine_summary(
            normalized_first_step_cosines
        )
        result.update(
            {
                "normalized_cosine_similarity": _cosine_similarity(normalized_reference, normalized_candidate),
                "normalized_mean_frame_cosine_similarity": normalized_frame_mean,
                "normalized_min_frame_cosine_similarity": normalized_frame_min,
                "normalized_undefined_frame_cosine_count": normalized_frame_undefined,
                "normalized_mean_first_step_cosine_similarity": normalized_first_mean,
                "normalized_min_first_step_cosine_similarity": normalized_first_min,
                "normalized_undefined_first_step_cosine_count": normalized_first_undefined,
                "normalized_per_action_dim_cosine_similarity": [
                    _cosine_similarity(normalized_reference[:, dimension], normalized_candidate[:, dimension])
                    for dimension in range(normalized_reference.shape[-1])
                ],
            }
        )
    return result


def _backend_label(document: dict[str, Any], fallback: str) -> str:
    return str(document.get("metadata", {}).get("backend", {}).get("name") or fallback)


def _safe_filename_stem(value: str) -> str:
    stem = Path(value).stem or value
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)


def _paired_successful_actions(
    reference: dict[str, Any], candidate: dict[str, Any], join_key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref_actions = _successful_actions(reference, join_key)
    cand_actions = _successful_actions(candidate, join_key)
    common = sorted(set(ref_actions) & set(cand_actions))
    keys: list[Any] = []
    ref_values: list[np.ndarray] = []
    cand_values: list[np.ndarray] = []
    for key in common:
        ref_arr = ref_actions[key]
        cand_arr = cand_actions[key]
        if ref_arr.shape != cand_arr.shape:
            continue
        keys.append(key)
        ref_values.append(ref_arr)
        cand_values.append(cand_arr)
    if not ref_values:
        raise ValueError("no comparable successful actions available for plotting")
    shapes = {value.shape for value in ref_values}
    if len(shapes) != 1:
        raise ValueError(f"action chunk shape varies across frames and cannot be plotted: {sorted(shapes)}")
    return np.asarray(keys), np.stack(ref_values), np.stack(cand_values)


def default_plot_dir(reference_path: str | Path, out_path: str | Path | None) -> Path:
    if out_path:
        path = Path(out_path).expanduser()
        return path.with_name(f"{path.stem}_plots")
    reference = Path(reference_path).expanduser()
    return reference.with_name(f"{reference.stem}_plots")


def write_compare_plots(
    *,
    reference: dict[str, Any],
    candidate: dict[str, Any],
    candidate_path: str | Path,
    output_dir: str | Path,
    join_key: str,
    action_step: int = 0,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("comparison plotting requires matplotlib") from exc

    keys, ref_actions, cand_actions = _paired_successful_actions(reference, candidate, join_key)
    if action_step < 0 or action_step >= ref_actions.shape[1]:
        raise ValueError(f"plot action step {action_step} is outside chunk size {ref_actions.shape[1]}")

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_label = _backend_label(reference, "reference")
    cand_label = _backend_label(candidate, "candidate")
    cand_stem = _safe_filename_stem(str(candidate_path))
    if join_key == "sample_timestamp_ns":
        x_values = (keys - keys[0]).astype(np.float64) / 1_000_000_000.0
    else:
        x_values = keys.astype(np.float64) if np.issubdtype(keys.dtype, np.number) else np.arange(len(keys))
    try:
        with np.errstate(over="raise", invalid="raise"):
            diff = np.abs(cand_actions - ref_actions)
    except FloatingPointError as exc:
        raise ValueError("plot action arithmetic produced non-finite values") from exc
    frame_mae = np.asarray(_stable_mean(diff.reshape(diff.shape[0], -1), axis=1))
    frame_max = diff.reshape(diff.shape[0], -1).max(axis=1)
    per_dim_mae = np.asarray(_stable_mean(diff.reshape(diff.shape[0], -1, diff.shape[-1]), axis=1))
    plotted: list[str] = []

    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.28, "font.size": 10})
    fig = plt.figure(figsize=(16, 10), dpi=150)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.3)
    ax_error = fig.add_subplot(grid[0])
    ax_error.plot(x_values, frame_mae, label="per-frame MAE", color="#2563eb", linewidth=1.8)
    ax_error.plot(x_values, frame_max, label="per-frame max error", color="#dc2626", linewidth=1.3, alpha=0.85)
    worst_idx = int(np.argmax(frame_mae))
    ax_error.axvline(x_values[worst_idx], color="#7c2d12", linestyle="--", linewidth=1.2, label="worst frame")
    ax_error.set_title(f"{ref_label} vs {cand_label}: action error per frame", fontsize=14, weight="bold")
    ax_error.set_xlabel("seconds from first sample" if join_key == "sample_timestamp_ns" else join_key)
    ax_error.set_ylabel("absolute error")
    ax_error.legend(loc="upper left")

    ax_dim = fig.add_subplot(grid[1], sharex=ax_error)
    colors = ["#1d4ed8", "#ea580c", "#16a34a", "#9333ea", "#0891b2", "#be123c"]
    for dim in range(per_dim_mae.shape[1]):
        ax_dim.plot(
            x_values,
            per_dim_mae[:, dim],
            label=f"action dim {dim + 1}",
            linewidth=1.35,
            color=colors[dim % len(colors)],
        )
    ax_dim.set_title("Per-action-dimension mean absolute error", fontsize=14, weight="bold")
    ax_dim.set_xlabel(join_key)
    ax_dim.set_ylabel("mean absolute error")
    ax_dim.legend(loc="upper left", ncol=3)
    error_path = out_dir / f"{cand_stem}_error_lines.png"
    fig.savefig(error_path, bbox_inches="tight")
    plt.close(fig)
    plotted.append(str(error_path))

    action_dim = ref_actions.shape[-1]
    rows = int(math.ceil(action_dim / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(18, max(5, rows * 4)), dpi=150, sharex=True)
    axes_array = np.asarray(axes).reshape(-1)
    for dim in range(action_dim):
        ax = axes_array[dim]
        ax.plot(x_values, ref_actions[:, action_step, dim], label=ref_label, color="#2563eb", linewidth=1.35)
        ax.plot(
            x_values,
            cand_actions[:, action_step, dim],
            label=cand_label,
            color="#dc2626",
            linewidth=1.1,
            alpha=0.85,
        )
        ax.set_title(f"action dim {dim + 1}", fontsize=12, weight="bold")
        ax.set_ylabel("raw value")
        ax.legend(loc="best")
    for ax in axes_array[action_dim:]:
        ax.axis("off")
    axes_array[min(action_dim - 1, len(axes_array) - 1)].set_xlabel(join_key)
    fig.suptitle(f"Raw action values: {ref_label} vs {cand_label} (chunk step {action_step})", fontsize=16)
    overview_path = out_dir / f"{cand_stem}_raw_action_overview.png"
    fig.savefig(overview_path, bbox_inches="tight")
    plt.close(fig)
    plotted.append(str(overview_path))

    for dim in range(action_dim):
        fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
        ax.plot(x_values, ref_actions[:, action_step, dim], label=ref_label, color="#2563eb", linewidth=1.7)
        ax.plot(
            x_values,
            cand_actions[:, action_step, dim],
            label=cand_label,
            color="#dc2626",
            linewidth=1.35,
            alpha=0.85,
        )
        ax.set_title(f"Raw action dim {dim + 1}: {ref_label} vs {cand_label}", fontsize=14, weight="bold")
        ax.set_xlabel(join_key)
        ax.set_ylabel(f"action[{dim + 1}] value")
        ax.legend(loc="best")
        path = out_dir / f"{cand_stem}_action_dim_{dim + 1}_raw.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        plotted.append(str(path))
    return plotted
