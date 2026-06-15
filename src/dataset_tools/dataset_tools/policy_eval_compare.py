from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def _successful_actions(document: dict[str, Any], key: str) -> dict[Any, np.ndarray]:
    actions: dict[Any, np.ndarray] = {}
    for frame in document.get("frames", []):
        if not frame.get("success") or frame.get("action") is None:
            continue
        join_value = frame["frame_index"] if key == "frame_index" else frame["sample_timestamp_ns"]
        actions[join_value] = np.asarray(frame["action"], dtype=np.float64)
    return actions


def compare_prediction_documents(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    join_key: str = "frame_index",
    allow_incompatible: bool = False,
) -> dict[str, Any]:
    ref_meta = reference.get("metadata", {})
    cand_meta = candidate.get("metadata", {})
    compatibility_errors: list[str] = []
    if ref_meta.get("contract_fingerprint") != cand_meta.get("contract_fingerprint"):
        compatibility_errors.append("contract_fingerprint differs")
    if ref_meta.get("action_dim") != cand_meta.get("action_dim"):
        compatibility_errors.append("action_dim differs")
    if compatibility_errors and not allow_incompatible:
        raise ValueError("incompatible prediction files: " + ", ".join(compatibility_errors))

    ref_actions = _successful_actions(reference, join_key)
    cand_actions = _successful_actions(candidate, join_key)
    common = sorted(set(ref_actions) & set(cand_actions))
    if not common:
        raise ValueError("no matching successful frames to compare")

    diffs: list[np.ndarray] = []
    mismatched_shapes: list[Any] = []
    for frame_key in common:
        ref_arr = ref_actions[frame_key]
        cand_arr = cand_actions[frame_key]
        if ref_arr.shape != cand_arr.shape:
            mismatched_shapes.append(frame_key)
            continue
        diffs.append(np.abs(cand_arr - ref_arr))
    if not diffs:
        raise ValueError(f"no comparable frames after shape checks; mismatched={mismatched_shapes}")

    flat = np.concatenate([diff.reshape(-1) for diff in diffs])
    per_dim_stack = np.concatenate([diff.reshape(-1, diff.shape[-1]) for diff in diffs if diff.ndim >= 1], axis=0)
    return {
        "reference_backend": ref_meta.get("backend", {}).get("name"),
        "candidate_backend": cand_meta.get("backend", {}).get("name"),
        "join_key": join_key,
        "matched_frames": len(common),
        "compared_frames": len(diffs),
        "missing_in_candidate": len(set(ref_actions) - set(cand_actions)),
        "extra_in_candidate": len(set(cand_actions) - set(ref_actions)),
        "mismatched_shape_frames": mismatched_shapes,
        "mae": float(np.mean(flat)),
        "max_error": float(np.max(flat)),
        "rmse": float(math.sqrt(np.mean(np.square(flat)))),
        "per_action_dim_mae": np.mean(per_dim_stack, axis=0).astype(float).tolist(),
        "per_action_dim_max_error": np.max(per_dim_stack, axis=0).astype(float).tolist(),
        "compatibility_warnings": compatibility_errors,
    }


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
        if ref_arr.ndim == 1:
            ref_arr = ref_arr.reshape(1, -1)
            cand_arr = cand_arr.reshape(1, -1)
        keys.append(key)
        ref_values.append(ref_arr)
        cand_values.append(cand_arr)
    if not ref_values:
        raise ValueError("no comparable successful actions available for plotting")
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
    x_values = keys.astype(np.float64) if np.issubdtype(keys.dtype, np.number) else np.arange(len(keys))
    diff = np.abs(cand_actions - ref_actions)
    frame_mae = diff.reshape(diff.shape[0], -1).mean(axis=1)
    frame_max = diff.reshape(diff.shape[0], -1).max(axis=1)
    per_dim_mae = diff.reshape(diff.shape[0], -1, diff.shape[-1]).mean(axis=1)
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
    ax_error.set_xlabel(join_key)
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
